import torch
import copy
from typing import List
from agentevolver.schema.trajectory import Sample
from best_logger import print_dict, print_listofdict
from agentevolver.module.context_manager.cmt_linear import ExtendedMessage, Linear_CMT
from agentevolver.module.context_manager.cmt_linear import find_sublist_indices, replace_token_ids
from best_logger import register_logger, print_dict, print_nested, NestedJsonItem, SeqItem
from textwrap import dedent

class LinearThinkCMT(Linear_CMT):
    """
    线性思考上下文管理器模板 (Linear Think Context Manager Template)。
    它不仅处理 LLM 和环境之间的对话流，还特别支持思维链（Thinking Process）的管理。
    该类以线性方式管理上下文窗口、分词和消息历史，并针对多轮对话中的思维过程进行了优化。

    继承自 Linear_CMT，复用了基础的线性管理逻辑。

    Attributes:
        config: 包含环境和模型设置的配置对象。
        tokenizer: 用于处理文本的分词器实例。
        full_context (List[ExtendedMessage]): 对话中所有消息的列表。
        current_context_status (str): 上下文的当前状态。
        max_seq_length (int): 上下文窗口的最大序列长度。
        max_env_output_length (int): 环境输出的最大长度。
        terminal_rewards_dict (dict): 存储终止奖励的字典。
        grouped_steps (List[List[ExtendedMessage]]): 存储分组后的交互步骤，用于生成多个训练样本。
    """

    def __init__(self, config, tokenizer):
        """
        初始化 LinearThinkCMT 类。

        Args:
            config: 配置对象。
            tokenizer: 分词器实例。
        """
        super().__init__(config, tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        self.full_context: List[ExtendedMessage] = []
        self.current_context_status = ""
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        max_model_len: int = self.config.actor_rollout_ref.rollout.max_model_len

        # 对于 Think 模式，Prompt 往往很长（包含了之前的思考过程），所以需要较大的 max_prompt_length
        assert self.config.data.max_response_length < self.config.data.max_prompt_length, "think linear template requires a big max_prompt_length"

        self.max_seq_length: int = max_model_len - max_response_length
        assert self.max_seq_length <= self.config.data.max_prompt_length, "max_seq_length should be less than or equal to max_prompt_length"


        self.max_env_output_length: int = self.config.actor_rollout_ref.rollout.max_env_len
        self.blackout_token_combo = tokenizer.encode("<|im_start|>assistant\n")

        self.terminal_rewards_dict = {}
        self.latest_llm_interaction_socket: List[ExtendedMessage] = None
        # grouped_steps 用于存储每一轮交互的快照，以便在 group_tokenize 中生成多个样本
        self.grouped_steps: List[List[ExtendedMessage]] = []

        self.discarded = False
        self.is_terminated = False
        self.reward = None
        self.context_time_cost = 0
        
        # 是否强制模型输出思考过程
        self.force_think = config.actor_rollout_ref.rollout.force_think
        self.env_feedin_preference = config.env_service.env_feedin_preference
        
        # 配置思考提示词 (Think Hint)
        if not self.force_think:
            # 默认提示
            self.think_hint: str = "\n\nThink about the next step before answering. Your thought (<think>...</think>) should be as short and concise as possible."
        else:
            # 根据环境偏好（Box 或 Code）设置强制思考的 Prompt
            if self.env_feedin_preference == "box":
                force_think_prompt = dedent("""
                    Additional requirements: Think before action! You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce your answer with \\box{}.
                    For example:
                    <think>...your thinking process...</think>
                    \\box{...your final answer...}
                """)
            elif self.env_feedin_preference == "code":
                force_think_prompt = dedent("""
                    Additional requirements: Think before action! You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce the next-step action.
                    For example:
                    <think>...your thinking process...</think>
                    ```python
                    # your action here
                    ```
                """)
            else:
                raise ValueError(f"Unsupported env_feedin_preference: {self.env_feedin_preference}")
            self.think_hint: str = force_think_prompt

    def _get_seq_length(self, messages: List[dict]) -> int:
        """
        计算给定消息列表在应用聊天模板并分词后的序列长度。

        Args:
            messages (List[dict]): 消息字典列表。

        Returns:
            int: Token 序列长度。
        """
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # ⭐ 应用聊天模板
        return len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])  # ⭐ 计算 Token 数量


    def check_context_token_num_safe(self, messages: List[dict]) -> bool:
        """
        检查消息的总 Token 数是否在允许范围内。

        Args:
            messages (List[dict]): 消息字典列表。

        Returns:
            bool: 如果安全返回 True，否则返回 False。
        """
        return self._get_seq_length(messages) < self.max_seq_length  # ⭐ 比较 Token 数与最大限制


    @property
    def steps(self):
        """
        属性：获取 'future' 模式的上下文步骤。

        Returns:
            Any: 准备好的上下文。
        """
        return self.prepare_previous_context(mod='future')  # ⭐ 获取 'future' 上下文


    def prepare_next_llm_context(self):
        """
        为下一次 LLM 调用准备上下文。
        
        关键逻辑：
        1. 过滤上下文，只保留相关作者。
        2. 处理历史中的 LLM 消息：
           - 移除 `<think>...</think>` 标签及其内容（如果不需要训练历史思考）。
           - 如果配置要求，可能会保留思考过程。
        3. 处理环境和初始化消息：
           - 如果不是最后一条消息，追加 `/no_think` 标记（提示模型不要再思考，因为已经过去）。
           - 如果是最后一条消息，追加 `think_hint`（提示模型开始思考）。

        Returns:
            dict: 更新后的上下文消息字典列表。
        """
        self.latest_llm_interaction_socket = []
        # 过滤上下文
        self.latest_llm_interaction_socket = self.filter_context_via_authors(["initialization", "llm", "env"])  # ⭐ 过滤作者

        for index, ext_msg in enumerate(list(self.latest_llm_interaction_socket)):
            # 判断是否是最后一条消息
            is_last = (index == len(self.latest_llm_interaction_socket) - 1)
            
            # 根据消息类型处理
            if ext_msg.author == "llm":
                # 处理之前的 LLM 消息：移除 think 标签
                import re
                new_ext_msg_content = re.sub(r'<think>.*?</think>', '', ext_msg.content, flags=re.DOTALL).strip()  # ⭐ 移除 <think> 内容
                new_ext_msg_content = new_ext_msg_content.replace("<think>", "")
                new_ext_msg_content = new_ext_msg_content.replace("</think>", "")
                
                # 配置：是否训练历史中的推理 Token
                if self.config.actor_rollout_ref.rollout.train_history_infer_token:
                    assert ext_msg.author == "llm"
                    self.latest_llm_interaction_socket[index] = ExtendedMessage(
                        author=ext_msg.author,
                        role=ext_msg.role,
                        content=new_ext_msg_content,
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                    )
                else:
                    # 默认情况：将历史 LLM 消息标记为不训练，并且移除了思考过程
                    assert ext_msg.author == "llm"
                    author_override = "llm(do_not_train)"
                    self.latest_llm_interaction_socket[index] = ExtendedMessage(
                        author=author_override,
                        role=ext_msg.role,
                        content=new_ext_msg_content,
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                    )
            elif ext_msg.author in ["env", "initialization"]:
                if self.config.actor_rollout_ref.rollout.train_history_infer_token:
                    # 如果需要推断历史 Token，则添加标记
                    if not is_last:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + "\n/no_think",
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
                    else:
                        # 最后一条消息：添加思考提示
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + self.think_hint,
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
                else:
                    # 默认情况
                    if not is_last:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future,
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
                    else:
                        # 最后一条消息：添加思考提示
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + self.think_hint,
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
            else:
                raise RuntimeError(f"Unknown author {ext_msg.author} in latest_llm_interaction_socket")

        dict_context = self.to_role_content(self.latest_llm_interaction_socket)  # ⭐ 转换为字典列表
        return dict_context



    def generate_log(self, task_id):
        """
        生成指定任务 ID 的日志。
        遍历所有的分组步骤 (grouped_steps)，为每一步生成详细的 Token 级日志。

        Args:
            task_id (str): 任务 ID。

        Returns:
            None
        """
        nested_items_print_buffer = {}
        for index, ext_steps in enumerate(self.grouped_steps):
            cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps, debug=True)  # ⭐ 分词
            text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]
            input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]
            loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in cmt_tokenized["loss_mask"]]
            buffer = {
                "text_arr": text_arr,
                "input_id_arr": input_id_arr,
                "loss_mask_color_arr": loss_mask_color_arr,
            }
            reward = self.reward.outcome
            task_outcome = str(self.reward.success_rate)
            selectors = [task_id, task_outcome, str(index)]
            len_prompt_ids = len(cmt_tokenized["prompt_ids"])
            len_response_ids = len(cmt_tokenized["response_ids"])
            len_input_ids = len(cmt_tokenized["input_ids"])
            assert len_prompt_ids + len_response_ids == len_input_ids, "len_prompt_ids + len_response_ids should equal to len_input_ids"
            
            # 构建日志条目
            nested_items_print_buffer[f".".join(selectors)] = NestedJsonItem(
                item_id=f"item",
                outcome=task_outcome,
                len_prompt_ids=len_prompt_ids,
                len_response_ids=len_response_ids,
                len_input_ids=len_input_ids,
                reward=f"{float(reward):.3f}",
                content=SeqItem(
                    text=buffer['text_arr'],  
                    title=buffer['text_arr'],  
                    count=buffer['input_id_arr'],  
                    color=buffer['loss_mask_color_arr']  
                )
            )
        print_nested(nested_items_print_buffer,
            main_content="This is the main content of the nested JSON",
            header=f"Training, task {task_id}",
            mod="rollout",
            narrow=False,
            attach="copy this"
        )


    def save_init_input(self, init_input_arr:list, add_nothink):
        """
        保存初始输入。调用父类方法。
        """
        super().save_init_input(init_input_arr, add_nothink)
        return


    def save_llm_output(self, llm_output, input_msg_ref):
        """
        保存 LLM 输出。
        除了基本的保存操作外，它还会将当前的交互快照添加到 `grouped_steps` 中。
        这是为了支持多轮对话的分步训练（每一步都可以作为一个独立的训练样本）。

        Args:
            llm_output: LLM 输出。
            input_msg_ref: 输入消息引用。

        Returns:
            None
        """
        ext_msg = super().save_llm_output(llm_output, input_msg_ref)  # ⭐ 调用父类方法保存输出
        
        # 创建包含新消息的当前交互深拷贝
        this_interaction = copy.deepcopy(self.latest_llm_interaction_socket + [ext_msg])  
        
        # 将本次交互记录添加到 grouped_steps
        self.grouped_steps += [this_interaction]  # ⭐ 记录交互快照
        
        # 重置 socket，准备下一轮
        self.latest_llm_interaction_socket = []  
        return


    def save_env_output(self, env_output:dict, input_msg_ref:List[dict]=None, add_nothink=False):
        super().save_env_output(env_output, input_msg_ref, add_nothink)
        return


    def prepare_world_interaction(self) -> str:
        """
        获取完整上下文中的最新内容（通常是 LLM 生成的包含动作的代码或文本）。

        Returns:
            str: 最新内容。
        """
        latest_content = self.full_context[-1].content  # ⭐ 获取最新内容
        return latest_content


    def group_tokenize(self):
        """
        对 `grouped_steps` 中的每一组交互步骤进行分词，并生成对应的 Sample 对象列表。
        这允许将一个多轮对话拆分为多个独立的训练样本，每个样本对应对话中的一步。

        Returns:
            list: Sample 对象列表。
        """
        sample_arr = []
        max_num_group = self.config.actor_rollout_ref.rollout.multi_turn.max_sample_per_task
        
        for index, ext_steps in enumerate(self.grouped_steps):
            if index >= max_num_group:
                print(f"Warning: group_tokenize only process first {max_num_group} groups, but got {len(self.grouped_steps)} groups")
                break
            
            # 对当前组进行分词
            cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps, debug=True)  # ⭐ 分词
            
            # 创建 Sample 对象
            sample = Sample(
                data_id=self.data_id,
                rollout_id=self.rollout_id,
                task_id=self.task_id,
                minor_index_id=index, # 区分同一个对话中的不同步骤
                messages=self.to_role_content(ext_steps),
                input_ids=cmt_tokenized["input_ids"],
                prompt_ids=cmt_tokenized["prompt_ids"],
                response_ids=cmt_tokenized["response_ids"],
                attention_mask=cmt_tokenized["attention_mask"],
                prompt_attention_mask=cmt_tokenized["prompt_attention_mask"],
                response_attention_mask=cmt_tokenized["response_attention_mask"],
                loss_mask=cmt_tokenized["loss_mask"],
                prompt_loss_mask=cmt_tokenized["prompt_loss_mask"],
                response_loss_mask=cmt_tokenized["response_loss_mask"],
                position_ids=cmt_tokenized["position_ids"],
                prompt_position_ids=cmt_tokenized["prompt_position_ids"],
                response_position_ids=cmt_tokenized["response_position_ids"],
                reward_scores=self.reward.model_dump(), # 奖励重复
                max_prompt_len=self.config.data.max_prompt_length,
                max_response_len=self.config.data.max_response_length,
                max_model_len=self.config.data.max_response_length + self.config.data.max_prompt_length,
            )
            sample.truncate_output_ids()
            assert len(sample.response_ids) != 0, "response_ids should not be empty"
            sample_arr += [sample]
        return sample_arr