# https://arxiv.org/pdf/2505.10978

import torch
import copy
from typing import List
from best_logger import print_dict, print_listofdict
from agentevolver.module.context_manager.cmt_linear import ExtendedMessage, Linear_CMT
from agentevolver.module.context_manager.cmt_linear import find_sublist_indices, replace_token_ids
from agentevolver.schema.trajectory import Sample
from agentevolver.utils.markdown_parser import read_markdown_and_extract_sections

# 定义初始指令 Prompt
first_instruction = """Use strict markdown format
# next-step instruction code
CODE_PLACEHOLDER (use ``` ... ``` to wrap the code)"""

# 定义环境反馈后的额外指令 Prompt，强制 LLM 进行状态总结和记忆提取
env_extra_instruction = """
---
1. Extract all useful information from the environment feedback above.
2. Generate code as next-step instruction.
3. Any information that is not extracted will be lost, so please extract all useful information relevant to current task such as accounts, keys, access tokens, etc.
4. Use strict markdown format, and strictly answer with the following 4 sections:
    - `# current step`
    - `# previous instruction code`
    - `# relevant environment feedback`
    - `# next-step instruction code`

---

# current step
STEP_PLACEHOLDER (step index and step summary)

# previous instruction code
PREVIOUS_CODE_PLACEHOLDER (use ``` ... ``` to wrap the code)

# relevant environment feedback
FEEDBACK_PLACEHOLDER (list of information, or traceback)

# next-step instruction code
CODE_PLACEHOLDER (use ``` ... ``` to wrap the code)
"""

from pydantic import BaseModel, Field


class GroupedSteps(BaseModel):
    num_groups: int = Field(default=0, description="Number of groups in the grouped steps")
    grouped_step_list: List[List[dict]] = Field(default_factory=list, description="List of grouped steps, each containing a list of ExtendedMessage objects")


class MemoryCoreCMT(Linear_CMT):
    """
    MemoryCoreCMT 类：实现了基于记忆的核心上下文构建逻辑。
    它重写了 `prepare_next_llm_context` 和 `save_llm_output`，
    将对话历史结构化为：[Instruction] -> [Memory History] -> [Previous Code] -> [Env Feedback]。
    """

    def prepare_next_llm_context(self) -> List[dict]:
        """
        为下一次 LLM 调用准备上下文。
        结构：
        [Instruction] -> ( [Memory History] -> [Previous Instruction] -> [Env Feedback + Extra Instruction] )
        
        关键逻辑：
        1. 如果是第一次交互，仅发送初始指令。
        2. 如果是后续交互，构造包含以下部分的 Prompt：
           - Part 1: 初始化指令 (System Prompt)。
           - Part 2: 记忆历史 (Memory History) - 之前所有步骤的摘要累积。
           - Part 3: 上一步指令代码 (Previous Instruction Code) - 从上一次 LLM 输出中提取。
           - Part 4: 环境反馈 + 额外指令 (Env Feedback + Extra Instruction)。

        Returns:
            List[dict]: 格式化后的上下文消息列表。
        """
        assert self.latest_llm_interaction_socket is None, "`prepare_next_llm_context` must be called at proper time!"
        self.current_step += 1
        self.latest_llm_interaction_socket = []
        is_first_interaction = (len(self.filter_context_via_author("llm")) == 0)


        if is_first_interaction:
            # 处理第一次交互
            part_1_instruction_array = self.filter_context_via_author("initialization")
            part_1_instruction_array += [
                ExtendedMessage(
                    author="initialization",
                    role="user",
                    content=first_instruction,
                    token_generator="auto",
                    tokenizer=self.tokenizer,
                )
            ]
            self.latest_llm_interaction_socket += part_1_instruction_array
            dict_context = self.to_role_content(self.latest_llm_interaction_socket)
            return dict_context

        # ---------- Part 1: 初始化指令 ----------
        part_1_instruction_array = self.filter_context_via_author("initialization")

        # ---------- Part 2: 记忆历史 ----------
        memory_history_ext_msg = self.filter_context_via_author("memory")
        if len(memory_history_ext_msg) != 0:
            str_concat_buffer = "Previous steps:\n---\n"
            for ext_msg in memory_history_ext_msg:
                str_concat_buffer += ext_msg.content_for_future + "\n"
            part_2_memory_history_array = [ExtendedMessage(
                author="memory",
                role='assistant',
                content=str_concat_buffer,
                token_generator="auto",
                tokenizer=self.tokenizer,
            )]
        else:
            part_2_memory_history_array = []
            
        # ---------- Part 3: 上一步指令代码 ----------
        # 从 LLM 上一次的输出中提取出 pure code
        last_llm_result = self.filter_context_via_author("llm")[-1]
        last_llm_result_decompose, _, find_nothing = read_markdown_and_extract_sections(
            markdown_text=last_llm_result.content_for_future,
            expected_sections=["current step", "previous instruction code", "relevant environment feedback", "next-step instruction code"],
            default_placeholder="❌ not available."
        )
        if not find_nothing:
            last_llm_result_next_step_code = last_llm_result_decompose['next-step instruction code']
            part_3_previous_instruction = ExtendedMessage(
                author="llm",
                role='assistant',
                content=last_llm_result_next_step_code,
                token_generator="auto",
                tokenizer=self.tokenizer,
            )
        else:
            # 如果解析失败，则使用原始内容
            part_3_previous_instruction = ExtendedMessage(
                author="llm",
                role='assistant',
                content=last_llm_result.content_for_future,
                token_generator="auto",
                tokenizer=self.tokenizer,
            )
            
        # ---------- Part 4: 环境反馈 + 额外指令 ----------
        message_arr_env_last = self.filter_context_via_author("env")[-1]
        part_4_env_and_env_extra = ExtendedMessage(
            author="env",
            role='user',
            content=message_arr_env_last.content_for_future + env_extra_instruction,
            token_generator="auto",
            tokenizer=self.tokenizer,
        )
        
        # 组装上下文
        self.latest_llm_interaction_socket = part_1_instruction_array + \
                                             part_2_memory_history_array + \
                                             [part_3_previous_instruction] + \
                                             [part_4_env_and_env_extra]

        dict_context = self.to_role_content(self.latest_llm_interaction_socket)
        return dict_context


    def save_llm_output(self, llm_output, input_msg_ref):
        """
        保存 LLM 的输出到完整上下文，并从中提取结构化记忆。

        Args:
            llm_output (dict): LLM 的输出，包含 'role', 'content' 和 'tokens'。
            input_msg_ref: 用于计算 Token 增量的输入消息引用。

        Note:
            - 解析 LLM 输出的 Markdown，提取 "current step", "feedback", "next-step code" 等部分。
            - 将提取的信息构造成一个新的 "memory" 类型的消息并保存。
        """
        def get_token_inc_from_vllm_response(input_msg_ref) -> List[int]:
            """
            计算 VLLM 响应的 Token 增量。
            """
            generation_prompt_token, msg = self.get_inc(
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=False),
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=True),
            )  # ⭐ 计算 generation prompt 的增量
            completion_token_arr, msg2 = self.get_inc(
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False),
                self.tokenizer.apply_chat_template(input_msg_ref + [ {"role": llm_output['role'],  "content": llm_output['content']} ], tokenize=False),
            )  # ⭐ 计算 completion 的增量
            vllm_output_raw_token = [t.token_id for t in llm_output['tokens']]
            final_token_arr = replace_token_ids(place_holder=completion_token_arr, replace_with=vllm_output_raw_token, begin=generation_prompt_token, end=[self.tokenizer.eos_token_id])
            return final_token_arr

        # 保存基本信息
        assert isinstance(llm_output, dict)
        assert self.latest_llm_interaction_socket is not None, "`save_llm_output` must be called at proper time!"
        ext_msg = ExtendedMessage(
            author="llm",
            role=llm_output['role'],
            content=llm_output['content'],
            token_generator="manual",
            tokenizer=self.tokenizer,
        )  # ⭐ 创建 LLM 输出的 ExtendedMessage 对象
        self.full_context += [ext_msg]
        ext_msg.token_arr = get_token_inc_from_vllm_response(input_msg_ref)
        
        # 记录交互快照
        this_interaction = copy.deepcopy(self.latest_llm_interaction_socket + [ext_msg])
        self.grouped_steps += [this_interaction]
        self.latest_llm_interaction_socket = None

        # 提取记忆 (Extract Memory)
        # 使用正则表达式解析 Markdown 格式的输出
        lm_result_decompose, find_everything, find_nothing = read_markdown_and_extract_sections(
            markdown_text=llm_output['content'],
            expected_sections=["current step", "previous instruction code", "relevant environment feedback", "next-step instruction code"],
            default_placeholder="❌ not available."
        )  # ⭐ 解析并提取部分
        from textwrap import dedent
        # 构造 Memory Block
        memory_construct = dedent("""
        >> step: {current_step}
        >> instruction:
        {next_step_instruction_code}
        >> feedback from environment:
        {relevant_environment_feedback}
        ---
        """).format(
            current_step=lm_result_decompose['current step'],
            next_step_instruction_code=lm_result_decompose['next-step instruction code'],
            relevant_environment_feedback=lm_result_decompose['relevant environment feedback']
        )  # ⭐ 格式化 Memory
        ext_msg_memory = ExtendedMessage(
            author="memory",
            role="assistant",
            content=memory_construct,
            token_generator="auto",
            tokenizer=self.tokenizer,
        )
        if find_everything:
            self.full_context += [ext_msg_memory]
        return



class MemoryCMT(MemoryCoreCMT):
    """
    MemoryCMT 类：MemoryCoreCMT 的具体实现，用于管理基于记忆的完整对话流程。
    它继承自 MemoryCoreCMT，提供了完整的属性和方法实现。

    Attributes:
        config: 配置对象。
        tokenizer: 分词器。
        full_context: 消息历史。
        current_context_status: 上下文状态。
        max_seq_length: 最大序列长度。
        terminal_rewards_dict: 奖励字典。
        current_step: 当前步数计数器。
    """

    def __init__(self, config, tokenizer):
        """
        初始化 MemoryCMT。

        Args:
            config: 配置对象。
            tokenizer: 分词器。
        """
        self.config = config
        self.tokenizer = tokenizer
        self.full_context: List[ExtendedMessage] = []
        self.current_context_status = ""
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        max_model_len: int = self.config.actor_rollout_ref.rollout.max_model_len
        self.max_seq_length: int = max_model_len - max_response_length  # ⭐ 计算最大上下文长度
        self.max_env_output_length: int = self.config.actor_rollout_ref.rollout.max_env_len
        self.blackout_token_combo = tokenizer.encode("<|im_start|>assistant\n")

        self.terminal_rewards_dict = {}
        self.reward = None
        self.latest_llm_interaction_socket: List[ExtendedMessage] = None
        self.grouped_steps: List[List[ExtendedMessage]] = []
        self.data_id = None
        self.rollout_id = None
        self.task_id = None

        self.current_step = 0

    @property
    def steps(self):
        # MemoryCMT 的上下文结构是非线性的，不支持直接作为 steps 输出
        raise NotImplementedError("MemoryCMT does not support steps.")

    def generate_log(self, task_id):
        """
        生成任务日志。

        Args:
            task_id (int): 任务 ID。

        Returns:
            GroupedSteps: 包含分组步骤的对象。
        """
        result = GroupedSteps()
        result.num_groups = len(self.grouped_steps)
        for steps in self.grouped_steps:
            result.grouped_step_list += [self.to_role_content(steps)]  # ⭐ 转换格式并添加到结果列表
        grouped_steps: GroupedSteps = result
        # for index, steps in enumerate(grouped_steps.grouped_step_list):
        #     print_listofdict(steps, mod='appworld_io', header=f'Task-{task_id} {index}/{grouped_steps.num_groups}')
        return



    def check_context_token_num_safe(self, messages: List[dict]):
        """
        检查上下文 Token 数量是否安全。调用父类实现。

        Args:
            messages (List[dict]): 消息列表。

        Returns:
            bool: 是否安全。
        """
        return super().check_context_token_num_safe(messages)  # ⭐ 调用父类方法


    def prepare_previous_context(self, mod='future'):
        """
        为未来 LLM 调用准备输入上下文。

        Args:
            mod (str): 模式 ('future' 或 'raw')。

        Returns:
            list: 格式化后的消息字典列表。
        """
        if mod=='future':
            message_arr = [
                {"role": c.role, "content": c.content_for_future}
                for c in self.full_context
            ]
            return message_arr

        elif mod=='raw':
            message_arr = [
                {"role": c.role, "content": c.content}
                for c in self.full_context
            ]
            return message_arr

        else:
            raise ValueError(f"Unknown mod {mod} in prepare_previous_context, only support 'future' and 'raw'")


    def save_init_input(self, init_input_arr:list, add_nothink):
        """
        保存并处理初始输入消息。

        Args:
            init_input_arr (list): 初始消息列表。
            add_nothink: 是否添加 no_think 标记。
        """
        # 保存基本信息，确保调用时机正确
        assert self.latest_llm_interaction_socket is None, "`save_init_input` must be called at proper time!"  # ⭐ 确保调用时机正确
        super().save_init_input(init_input_arr, add_nothink)




    def save_env_output(self, env_output:dict, input_msg_ref:List[dict]=None, add_nothink=False):
        """
        保存并处理环境输出。

        Args:
            env_output (dict): 环境输出。
            input_msg_ref (List[dict], optional): 参考消息。
            add_nothink (bool, optional): 是否添加 no_think 标记。
        """
        assert self.latest_llm_interaction_socket is None, "`save_env_output` must be called at proper time!"  # ⭐ 确保调用时机正确
        super().save_env_output(env_output, input_msg_ref=input_msg_ref, add_nothink=add_nothink)


    def prepare_world_interaction(self):
        """
        在与环境交互前处理最新的模型内容。
        从 LLM 输出的 Markdown 中提取 `next-step instruction code` 部分作为实际动作。

        Returns:
            str: 提取出的指令代码。
        """
        ext_message_arr_memory = self.filter_context_via_author("llm")

        # 提取记忆部分
        lm_result_decompose, find_everything, find_nothing = read_markdown_and_extract_sections(
            markdown_text=ext_message_arr_memory[-1].content,
            expected_sections=["current step", "previous instruction code", "relevant environment feedback", "next-step instruction code"],
            default_placeholder="❌ not available."
        )

        return lm_result_decompose['next-step instruction code']


    def group_tokenize(self):
        """
        对分组后的对话步骤进行分词，并生成 Sample 对象列表。
        这会将长对话拆分为多个训练样本，每个样本包含特定的上下文和目标输出。

        Returns:
            list[Sample]: Sample 对象列表。
        """
        # assert self.latest_llm_interaction_socket is None, "unprocessed message buffer! forget to call `save_llm_output` after `prepare_next_llm_context`?"
        sample_arr = []
        max_num_group = 30 # self.config.actor_rollout_ref.rollout.multi_turn.max_steps
        for index, ext_steps in enumerate(self.grouped_steps):
            if index >= max_num_group:
                print(f"Warning: group_tokenize only process first {max_num_group} groups, but got {len(self.grouped_steps)} groups")
                break
            cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps)  # ⭐ 对当前组进行分词
            sample = Sample(
                data_id=self.data_id,
                rollout_id=self.rollout_id,
                task_id=self.task_id,
                minor_index_id=index,
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
            sample.truncate_output_ids()  # ⭐ 截断输出 ID
            sample_arr += [sample]
        return sample_arr