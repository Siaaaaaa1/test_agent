import copy
import uuid
import json
import re
import torch
from typing import List, Union
from agentevolver.schema.trajectory import Sample, Reward
from agentevolver.schema.trajectory import Sample, Trajectory
from agentevolver.utils.compute_madness import repetition_penalty_reward_scalar
from agentevolver.module.context_manager.cmt_base import ExtendedMessage, ContextManagerBase
from agentevolver.module.context_manager.cmt_base import find_sublist_indices, replace_token_ids
from best_logger import register_logger, print_listofdict, print_dict, print_nested, NestedJsonItem, SeqItem
from agentevolver.module.exp_manager.exp_manager import ExperienceWorker, TrajExpConfig


class Linear_CMT(Trajectory, ContextManagerBase):
    """
    线性上下文管理器模板 (Linear Context Manager Template)。
    它负责处理 LLM 与环境之间的对话流，以线性方式（按时间顺序）管理上下文窗口、分词和消息历史。
    继承自 Trajectory (用于存储轨迹数据) 和 ContextManagerBase (实现管理接口)。

    Attributes:
        config: 包含环境和模型设置的配置对象。
        tokenizer: 用于处理文本的分词器实例。
        full_context (List[ExtendedMessage]): 对话中所有消息的列表。
        current_context_status (str): 上下文的当前状态。
        max_seq_length (int): 上下文窗口的最大序列长度（通常等于 max_model_len - max_response_len）。
        max_env_output_length (int): 环境输出的最大长度。
        terminal_rewards_dict (dict): 存储终止奖励的字典。
    """

    def __init__(self, config, tokenizer):
        """
        初始化 Linear_CMT 类。

        Args:
            config: 配置对象。
            tokenizer: 分词器实例。
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.full_context: List[ExtendedMessage] = []  # ⭐ 初始化列表以存储所有对话消息
        self.current_context_status = ""
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        max_model_len: int = self.config.actor_rollout_ref.rollout.max_model_len
        # 计算 Context Window 的最大允许长度（给 Prompt 留出的空间）
        self.max_seq_length: int = max_model_len - max_response_length  
        self.max_env_output_length: int = self.config.actor_rollout_ref.rollout.max_env_len
        
        # 定义需要被抹黑（不计算 Loss 或特殊处理）的 Token 组合，这里是 Assistant 的起始符
        self.blackout_token_combo = tokenizer.encode("<|im_start|>assistant\n")
        self.generated_token_cnt = 0

        self.terminal_rewards_dict = {}
        self.discarded = False
        self.is_terminated = False
        self.reward: Union[Reward, None] = None
        self.context_time_cost = 0
        self.tag: str = ""
        self.task_id: str = ""
        # self.task_train_exp_mode: str = ""
        self.current_batch_success_rate:float = -1.0
        self.llm_output_mistakes = {}
        # self.experiences = []

        # 确保配置合理：Prompt + Response 的最大长度不应超过模型的支持长度
        assert self.config.data.max_prompt_length + self.config.data.max_response_length <= max_model_len  # ⭐ 确保长度限制配置正确


    def prepare_previous_context(self, mod='future'):
        """
        为未来的 LLM 调用准备输入上下文。

        Args:
            mod (str, optional): 格式化上下文的模式。默认为 'future'。
                                 - 'future': 使用每个消息的 `content_for_future` 属性（可能经过处理/剪裁）。
                                 - 'raw': 使用每个消息的原始 `content`。

        Returns:
            list: 包含 role 和 content 的消息字典数组，格式化为 LLM 输入。

        Raises:
            ValueError: 如果提供了未知的模式。
        """
        if mod == 'future':
            message_arr = [
                {"role": c.role, "content": c.content_for_future}  # ⭐ 使用 future 内容格式化消息
                for c in self.full_context
            ]
            return message_arr

        elif mod == 'raw':
            message_arr = [
                {"role": c.role, "content": c.content}  # ⭐ 使用原始内容格式化消息
                for c in self.full_context
            ]
            return message_arr

        else:
            raise ValueError(f"Unknown mod {mod} in prepare_previous_context, only support 'future' and 'raw'")


    def check_context_token_num_safe(self, messages: List[dict]):
        """
        检查准备好的上下文消息的总 Token 数是否在安全限制内。

        Args:
            messages (List[dict]): 要检查的消息列表。

        Returns:
            bool: 如果总 Token 数小于允许的最大序列长度，则返回 True，否则返回 False。
        """
        def get_seq_length(messages):
            # 使用聊天模板将消息转换为文本，并计算其 Token 数
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])  # ⭐ 计算总 Token 数
        messages = self.prepare_previous_context(mod="raw")
        return get_seq_length(messages) < self.max_seq_length   # 检查是否小于 max_seq_length (例如 20480)


    def get_inc(self, text_frag_from, text_frag_to):
        """
        获取从 `text_frag_from` 到 `text_frag_to` 的增量 Token 数组。
        
        Args:
            text_frag_from (str): 起始文本片段。
            text_frag_to (str): 结束文本片段。

        Returns:
            Tuple[List[int], str]: 包含增量 Token ID 列表和描述 Token 长度详情的消息元组。
        """
        tokenizer_output = self.tokenizer(text_frag_from, return_tensors="pt", padding=False)
        tokenizer_input_ids = tokenizer_output["input_ids"][0].tolist()
        token_ids_acc = tokenizer_input_ids

        tokenizer_output = self.tokenizer(text_frag_to, return_tensors="pt", padding=False)
        input_ids = tokenizer_output["input_ids"][0].tolist()
        input_id_increment = input_ids[len(token_ids_acc):]  # ⭐ 获取本步骤新增的 Token
        overlap_length = 0
        for i in range(len(token_ids_acc)):
            if i < len(token_ids_acc) and input_ids[i] == token_ids_acc[i]: overlap_length += 1
            else: break
        msg = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
        # print(msg)
        return input_id_increment, msg

    def remove_last_context(self):
        """
        如果最后一条消息不是由 LLM 生成的，则将其从完整上下文中移除。
        这通常用于清理对话历史，确保上下文以 LLM 的回复结束（或者移除未完成的 User 消息）。

        Returns:
            None
        """
        if len(self.full_context) > 0:  # ⭐ 检查上下文是否非空
            if self.full_context[-1].author != "llm":  # ⭐ 确保最后一条消息不是来自 LLM
                self.full_context.pop(-1)  # ⭐ 移除最后一条消息

    def remove_last_non_llm_msg(self, ext_msg_list:List[ExtendedMessage]):
        """
        (静态辅助方法风格) 从给定的消息列表中移除最后一条非 LLM 消息。

        Args:
            ext_msg_list (List[ExtendedMessage]): 扩展消息对象列表。

        Returns:
            List[ExtendedMessage]: 更新后的列表。
        """
        if len(ext_msg_list) > 0:
            if ext_msg_list[-1].author != "llm":
                ext_msg_list.pop(-1)  # ⭐ 如果最后一条不是 LLM 消息则移除
        return ext_msg_list



    @property
    def steps(self):
        """
        属性：获取格式化为 'future' 模式的步骤列表（即上下文）。

        Returns:
            dict: 准备好的上下文列表。
        """
        return self.prepare_previous_context(mod='future')  # ⭐ 获取 'future' 模式的上下文

    def json(self):
        """
        将 'future' 模式的上下文转换为 JSON 格式字符串。

        Returns:
            str: JSON 格式的上下文字符串。
        """
        return json.dumps(self.prepare_previous_context(mod='future'), ensure_ascii=False, indent=2)  # ⭐ 转换为 JSON 字符串

    def prepare_next_llm_context(self):
        """
        为下一次 LLM 交互准备上下文。
        调用 `prepare_previous_context` 并使用 'future' 模式。

        Returns:
            list: 准备好的消息列表。
        """
        return self.prepare_previous_context(mod='future')  # ⭐ 准备 LLM 上下文


    def save_init_input(self, init_input_arr:list, add_nothink: bool=False):
        """
        保存并处理初始输入消息到上下文中。

        Args:
            init_input_arr (list): 初始输入消息数组，每个消息应包含 'role' 和 'content'。
            add_nothink (bool, optional): 如果为 True，在最后一条消息内容后追加 "/no_think"。默认为 False。

        Note:
            - 使用提供的消息初始化上下文。
            - 计算每条消息的 Token 数组。
            - 保存前会验证上下文是否为空。
        """
        # 保存基本信息
        assert len(self.full_context) == 0, "full_context should be empty when saving init input"
        for index, llm_msg in enumerate(init_input_arr):
            if (index == len(init_input_arr) - 1):
                if add_nothink:
                    llm_msg['content'] += "\n/no_think"
            ext_msg = ExtendedMessage(
                author="initialization",
                role=llm_msg['role'],
                content=llm_msg['content'],
                token_generator="manual",
                tokenizer=self.tokenizer,
            )
            self.full_context += [ext_msg]  # ⭐ 将扩展消息添加到完整上下文

        # 计算每条消息的 Token 数组 (增量计算)
        token_ids_acc = []
        for llm_msg, ext_msg, index in zip(init_input_arr, self.full_context, range(len(init_input_arr))):
            text_with_chat_template = self.tokenizer.apply_chat_template(init_input_arr[:(index+1)], tokenize=False)
            tokenizer_output = self.tokenizer(text_with_chat_template, return_tensors="pt", padding=False)
            input_ids = tokenizer_output["input_ids"][0].tolist()
            # 获取本步骤新增的 Token
            input_id_increment = input_ids[len(token_ids_acc):]  
            
            # 计算重叠长度用于调试
            overlap_length = 0
            for i in range(len(token_ids_acc)):
                if (i < len(token_ids_acc)) and (input_ids[i] == token_ids_acc[i]): overlap_length += 1
                else: break
            ext_msg._info = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
            ext_msg.token_arr = input_id_increment  # ⭐ 设置扩展消息的 Token 数组
            token_ids_acc += input_ids
        return

    def influence_extra_reward(self, llm_output):
        """
        评估 LLM 输出的重复性并应用惩罚奖励。
        如果存在惩罚（非零），则记录日志，并将最小惩罚值存储在 mistakes 字典中。

        Args:
            llm_output (dict): LLM 的输出，预期包含 'content' 键。

        Returns:
            None
        """
        # 计算重复性惩罚奖励
        this_msg_repetition_penalty_reward = repetition_penalty_reward_scalar(completion=llm_output['content'])  
        if this_msg_repetition_penalty_reward != 0:
            print_dict({
                "reason": "repetition_penalty_reward",
                "content": llm_output['content'],
                "score": this_msg_repetition_penalty_reward,
            })
        if 'repetition_penalty_reward' not in self.llm_output_mistakes:
            self.llm_output_mistakes['repetition_penalty_reward'] = 0
        # 更新错误字典，保留最小（最差）的惩罚值
        self.llm_output_mistakes['repetition_penalty_reward'] = min(this_msg_repetition_penalty_reward, self.llm_output_mistakes['repetition_penalty_reward'])  

    def save_llm_output(self, llm_output, input_msg_ref, auto_register_full_context=True):
        """
        保存 LLM 的输出到完整上下文中。

        Args:
            llm_output (dict): LLM 的输出，包含 'role', 'content' 和 'tokens'。
            input_msg_ref: 用于计算 Token 增量的输入消息引用。
            auto_register_full_context (bool): 是否将输出注册到 full_context 中。

        Returns:
            ExtendedMessage: 处理后的扩展消息对象。

        Note:
            - 处理 LLM 输出并将其添加到对话历史。
            - 自动处理 Token 生成和 Prompt 管理。
        """
        # 保存基本信息
        assert isinstance(llm_output, dict)
        token_generator = "manual" if 'tokens' in llm_output else "auto"
        ext_msg = ExtendedMessage(
            author="llm",
            role=llm_output['role'],
            content=llm_output['content'],
            token_generator=token_generator,
            tokenizer=self.tokenizer,
        )  # ⭐ 创建 LLM 输出的 ExtendedMessage 对象
        if auto_register_full_context:
            self.full_context += [ext_msg]  # ⭐ 如果开启自动注册，则添加到完整上下文

        # 检查错误 (如重复生成)
        if auto_register_full_context:
            self.influence_extra_reward(llm_output)  # ⭐ 计算额外奖励影响

        # 生成 Token 逻辑
        def get_token_inc_from_vllm_response(input_msg_ref) -> List[int]:
            # 计算 generation prompt 的 token
            generation_prompt_token, msg = self.get_inc(
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=False),
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=True),
            )
            # completion_token_arr 将包含 generation_prompt 头部
            completion_token_arr, msg2 = self.get_inc(
                # ...  <|im_end|>
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False),
                # ...  <|im_end|><|im_start|>...<|im_end|>
                self.tokenizer.apply_chat_template(input_msg_ref + [ {"role": llm_output['role'],  "content": llm_output['content']} ], tokenize=False),
            )
            vllm_output_raw_token = [t.token_id for t in llm_output['tokens']]
            self.generated_token_cnt += len(vllm_output_raw_token)  # ⭐ 增加生成的 Token 计数
            # 替换占位符 Token 为实际生成的 Token
            final_token_arr = replace_token_ids(place_holder=completion_token_arr, replace_with=vllm_output_raw_token, begin=generation_prompt_token, end=[self.tokenizer.eos_token_id])
            return final_token_arr

        if token_generator == "manual":
            token_arr_method2 = get_token_inc_from_vllm_response(input_msg_ref)  # ⭐ 使用 vLLM 响应生成 Token 增量
            ext_msg.token_arr = token_arr_method2  # ⭐ 设置 Token 数组
        return ext_msg


    def save_llm_output_do_not_register_full_context(self, llm_output, input_msg_ref):
        """
        保存 LLM 输出但不注册到完整上下文中。
        (通常用于中间推理步骤或临时生成)

        Args:
            llm_output: LLM 输出。
            input_msg_ref: 输入消息引用。

        Returns:
            保存操作的结果。
        """
        return Linear_CMT.save_llm_output(self, llm_output, input_msg_ref, auto_register_full_context=False)  # ⭐ 调用保存方法但不注册


    def save_env_output(self, env_output:dict, input_msg_ref:List[dict]=None, add_nothink=False):
        """
        保存并处理环境输出到上下文中。

        Args:
            env_output (dict): 环境输出，包含 'content'。
            input_msg_ref (List[dict], optional): 用于 Token 计算的参考消息。
            add_nothink (bool, optional): 是否在内容后追加 '/no_think'。

        Note:
            - 如果超出 max_env_output_length，会对环境输出进行剪裁。
            - 将输出处理为对话中的 User 消息。
            - 计算并存储环境响应的 Token 数组。
        """
        assert isinstance(env_output, dict)
        if ('content' not in env_output) and ('error' in env_output):
            env_output['content'] = f"[Error from environment: {env_output['error']}]"
        elif ('content' not in env_output) or (not env_output['content']):
            env_output['content'] = '[No content provided by the environment]'
        if add_nothink:
            env_output['content'] += " /no_think"
        ext_msg = ExtendedMessage(
            author="env",
            role="user",
            content=env_output['content'],
            clip=True,
            clip_token_limit=self.max_env_output_length,
            token_generator="auto",
            tokenizer=self.tokenizer,
        )  # ⭐ 创建环境输出的 ExtendedMessage
        self.full_context += [ext_msg]  # ⭐ 添加到完整上下文
        return

    def to_role_content(self, ext_msg_array: List[ExtendedMessage]) -> List[dict]:
        """
        将 ExtendedMessage 对象列表转换为包含 'role' 和 'content' 的字典列表。

        Args:
            ext_msg_array (List[ExtendedMessage]): ExtendedMessage 对象列表。

        Returns:
            List[dict]: 字典列表。
        """
        return [{"role": ext_msg.role, "content": ext_msg.content_for_future} for ext_msg in ext_msg_array]  # ⭐ 转换格式

    def prepare_world_interaction(self) -> str:
        """
        在与环境交互之前处理最新的模型内容。
        通常用于从 LLM 输出中提取代码块或指令。

        Returns:
            str: 处理后的内容（例如提取出的 Python 代码），如果没有代码块则返回原始内容。
        """
        latest_content = self.full_context[-1].content
        return latest_content

    def filter_context_via_author(self, author: str) -> List[ExtendedMessage]:
        """
        过滤完整上下文，只保留特定作者的消息，并返回深拷贝。

        Args:
            author (str): 作者名称。

        Returns:
            List[ExtendedMessage]: 过滤后的消息列表深拷贝。
        """
        return copy.deepcopy([ c for c in self.full_context if c.author == author ])  # ⭐ 过滤并深拷贝

    def filter_context_via_authors(self, authors: str) -> List[ExtendedMessage]:
        """
        过滤完整上下文，只保留指定作者集合中的消息。

        Args:
            authors (str): 作者名称字符串（或列表）。

        Returns:
            List[ExtendedMessage]: 过滤后的消息列表。
        """
        return copy.deepcopy([ c for c in self.full_context if c.author in authors ])  # ⭐ 过滤相关作者的消息

    def group_tokenize(self):
        """
        对完整上下文进行分词，格式化为适合模型训练的 Sample 对象。
        包含 input IDs, attention masks, position IDs 等必要属性。

        Returns:
            List[Sample]: 包含单个 Sample 对象的列表，代表分词后的上下文。
        """
        # assert self.latest_llm_interaction_socket is None, "unprocessed message buffer! forget to call `save_llm_output` after `prepare_next_llm_context`?"
        sample_arr = []
        ext_steps=self.full_context
        # 调用分词逻辑
        cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps)
        
        # 创建 Sample 对象
        sample = Sample(
            data_id=self.data_id,
            rollout_id=self.rollout_id,
            task_id=self.task_id,
            minor_index_id=0,
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
            reward_scores=self.reward.model_dump(), # 奖励在每个样本中重复
            max_prompt_len=self.config.data.max_prompt_length,
            max_response_len=self.config.data.max_response_length,
            max_model_len=self.config.data.max_response_length + self.config.data.max_prompt_length,
        )
        sample.truncate_output_ids()  # ⭐ 确保输出 ID 不超过允许长度
        sample_arr += [sample]
        return sample_arr


    def group_render_token_log(self):
        """
        生成用于可视化的 Token 日志数据。
        """
        ext_steps=self.full_context
        cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps)
        text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]
        input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]
        loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in cmt_tokenized["loss_mask"]]
        return {
            "text_arr": text_arr,
            "input_id_arr": input_id_arr,
            "loss_mask_color_arr": loss_mask_color_arr,
        }


    def generate_log(self, task_id):
        """
        生成并打印指定任务的详细日志。
        包括分词步骤、Input ID、Loss Mask 颜色可视化以及奖励信息。

        Args:
            task_id (str): 任务 ID。

        Returns:
            None
        """
        nested_items_print_buffer = {}
        ext_steps=self.full_context  # ⭐ 获取任务的完整上下文
        cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps)  # ⭐ 分词
        text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]  # ⭐ 解码为文本
        input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]  # ⭐ 转为字符串 ID
        loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in cmt_tokenized["loss_mask"]]  # ⭐ 根据 mask 生成颜色
        buffer = {
            "text_arr": text_arr,
            "input_id_arr": input_id_arr,
            "loss_mask_color_arr": loss_mask_color_arr,
        }
        len_prompt_ids = len(cmt_tokenized["prompt_ids"])  
        len_response_ids = len(cmt_tokenized["response_ids"])  
        len_input_ids = len(cmt_tokenized["input_ids"])  
        reward = self.reward.outcome  # ⭐ 获取奖励结果
        task_outcome = str(self.reward.success_rate)  # ⭐ 获取任务成功率
        final_reward = self.reward_patch(self.reward).outcome  # ⭐ 获取修补后的最终奖励
        selectors = [task_id, task_outcome]
        
        # 构建嵌套 JSON 用于格式化打印
        nested_items_print_buffer[f".".join(selectors)] = NestedJsonItem(
            item_id=f"item",
            outcome=task_outcome,
            len_prompt_ids=len_prompt_ids,
            len_response_ids=len_response_ids,
            len_input_ids=len_input_ids,
            reward=f"{float(reward):.3f}",
            final_reward=final_reward,
            content=SeqItem(
                text = buffer['text_arr'],  # 文本内容
                title = buffer['text_arr'], # 鼠标悬停提示
                count = buffer['input_id_arr'], # 高亮信息
                color = buffer['loss_mask_color_arr']   # 颜色编码
            )
        )
        print_nested(nested_items_print_buffer,  # ⭐ 打印嵌套 JSON 日志
            main_content="This is the main content of the nested JSON",
            header=f"Training task {task_id} (Final Reward {final_reward})",
            mod="rollout",
            narrow=False,
            attach="Copy Sample Message"
        )
        print_listofdict(  # ⭐ 打印对话列表
            self.steps,
            header=f"Training task {task_id} (Final Reward {final_reward})",
            mod="conversation",
            narrow=False,
        )

    def reward_patch(self, reward):
        """
        创建奖励的深拷贝，并根据内部状态（如 Madness）进行修改。

        Args:
            reward (object): 原始奖励对象。

        Returns:
            object: 修改后的奖励对象深拷贝。
        """
        _reward = copy.deepcopy(reward)  # ⭐ 深拷贝避免修改原始对象
        # 如果模型输出过于疯狂（重复、乱码等），可以在这里扣分
        # if self.compute_madness() < 0: _reward.outcome = -1.0
        return _reward


    def compute_madness(self) -> float:
        """
        评估模型输出的 '疯狂度' (Madness)。
        基于错误字典中的记录（如重复惩罚），判断是否超过阈值。

        Returns:
            float: 如果任何错误比例低于阈值（负值表示惩罚），返回 -1.0，否则返回 0.0。
        """
        threshold = -0.01
        for k, v in self.llm_output_mistakes.items():
            if v < threshold: return -1.0  # ⭐ 检查错误是否超过阈值
        return 0.0


    def tokenize_steps(self, ext_steps: List[ExtendedMessage], debug=False) -> dict:
        """
        对给定的扩展消息列表进行分词，处理 Prompt 和 Response 的分割，并为模型训练准备数据。
        如果需要，还会处理经验信息（Experience）的提取和管理。

        Args:
            ext_steps (List[ExtendedMessage]): 扩展消息列表。
            debug (bool, optional): 是否开启调试。默认为 False。

        Returns:
            dict: 包含分词后数据的字典 (input_ids, masks, position_ids 等)。
        """
        from verl.utils.model import compute_position_id_with_mask
        # 移除最后一条非 LLM 消息，确保训练数据以 LLM 回复结尾（或符合训练范式）
        ext_steps = self.remove_last_non_llm_msg(copy.deepcopy(ext_steps))  # ⭐ 移除末尾非 LLM 消息

        # 处理经验回放逻辑：根据 content_for_future 可能替换内容
        exp_worker = ExperienceWorker(self.config)
        for i, ext_msg in enumerate(ext_steps):
            experience, new_content = exp_worker.manage_training_context(ext_msg.content_for_future, self.metadata)
            if experience:
                ext_steps[i] = ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=new_content,
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    uuid=ext_msg.uuid,
                )

        # 映射并构建 Token 列表
        input_ids = []
        attention_mask = []
        loss_mask = []
        split_prompt_reponse_index = -1 # 用于分割 Prompt 和 Response 的索引
        
        for ext_msg in ext_steps:
            # 找到分割点：第一个需要训练的消息（通常是第一个 LLM 回复）之前的都是 Prompt
            if (split_prompt_reponse_index == -1) and (ext_msg.need_training):
                split_prompt_reponse_index = len(input_ids)
                assert ext_msg.author == 'llm', "The first message after initialization should be from LLM, not from env or user"
            input_ids += ext_msg.token_arr
            attention_mask += [1] * len(ext_msg.token_arr)
            loss_mask += ext_msg.get_loss_mask(blackout_token_combo=self.blackout_token_combo)

        assert split_prompt_reponse_index != -1, "split_prompt_reponse_index should not be -1, at least one message should be in the context"
        position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()  # ⭐ 计算 Position IDs

        # 分离 Prompt 和 Response 部分
        prompt_ids =            input_ids[:split_prompt_reponse_index]
        prompt_attention_mask = attention_mask[:split_prompt_reponse_index]
        prompt_position_ids =   position_ids[:split_prompt_reponse_index]
        prompt_loss_mask =      loss_mask[:split_prompt_reponse_index]

        response_ids =              input_ids[split_prompt_reponse_index:]
        response_attention_mask =   attention_mask[split_prompt_reponse_index:]
        response_position_ids =     position_ids[split_prompt_reponse_index:]
        response_loss_mask =        loss_mask[split_prompt_reponse_index:]

        # 构建返回字典
        cmt_tokenized = {}
        cmt_tokenized["input_ids"] = input_ids
        cmt_tokenized["prompt_ids"] = prompt_ids
        cmt_tokenized["response_ids"] = response_ids
        cmt_tokenized["attention_mask"] = attention_mask
        cmt_tokenized["prompt_attention_mask"] = prompt_attention_mask
        cmt_tokenized["response_attention_mask"] = response_attention_mask
        cmt_tokenized["loss_mask"] = loss_mask
        cmt_tokenized["prompt_loss_mask"] = prompt_loss_mask
        cmt_tokenized["response_loss_mask"] = response_loss_mask
        cmt_tokenized["position_ids"] = position_ids
        cmt_tokenized["prompt_position_ids"] = prompt_position_ids
        cmt_tokenized["response_position_ids"] = response_position_ids

        return cmt_tokenized