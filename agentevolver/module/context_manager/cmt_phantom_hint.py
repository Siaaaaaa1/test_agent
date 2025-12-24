import torch
import copy
from typing import List
from agentevolver.schema.trajectory import Sample
from best_logger import print_dict, print_listofdict
from agentevolver.module.context_manager.cmt_linear import ExtendedMessage, Linear_CMT
from agentevolver.module.context_manager.cmt_linear import find_sublist_indices, replace_token_ids

class LinearThinkCMT(Linear_CMT):
    """
    线性思考上下文管理器模板 (Linear Think Context Manager Template)。
    它是 Linear_CMT 的扩展，旨在管理 LLM 与环境之间的对话流程，特别关注带有思维过程的上下文。
    该类以线性方式管理上下文窗口、分词和消息历史，并增加了对 Phantom Hint（幻影提示/外部提示）的支持。

    Attributes:
        config: 包含环境和模型设置的配置对象。
        tokenizer: 用于处理文本的分词器实例。
        full_context (List[ExtendedMessage]): 对话中所有消息的列表。
        current_context_status (str): 上下文的当前状态。
        max_seq_length (int): 上下文窗口的最大序列长度。
        max_env_output_length (int): 环境输出的最大长度。
        terminal_rewards_dict (dict): 存储终止奖励的字典。
        contain_phantom_hint (bool): 标志位，指示是否包含 Phantom Hint。
        past_trajectory: 过去的轨迹数据，可能用作提示。
    """

    def __init__(self, config, tokenizer, contain_phantom_hint=False, past_trajectory=None):
        """
        初始化 LinearThinkCMT 类。

        Args:
            config: 包含环境和模型设置的配置对象。
            tokenizer: 用于处理文本的分词器实例。
            contain_phantom_hint (bool, optional): 标志位，指示是否包含幻影提示（Phantom Hint）。默认为 False。
                                                   这通常用于在上下文中注入额外的提示信息（例如来自检索增强生成 RAG 的信息或过去的成功经验），以引导模型思考。
            past_trajectory (optional): 过去的轨迹数据。默认为 None。
                                        如果提供了过去轨迹，它可能会被用作 Few-shot 示例或上下文参考。
        """
        super().__init__(config, tokenizer)  # ⭐ 调用父类 Linear_CMT 的初始化方法
        self.contain_phantom_hint = contain_phantom_hint  # ⭐ 设置是否包含 Phantom Hint
        self.past_trajectory = past_trajectory  # ⭐ 存储过去的轨迹数据
        self.helper_llm_handle = None  # 辅助 LLM 句柄（可能用于生成提示或处理上下文）

    def save_init_input(self, init_input_arr:list, add_nothink: bool=False):
        """
        保存初始输入消息。
        如果包含 Phantom Hint，可能会在这里进行特殊处理（省略号部分暗示了这一点）。
        
        Args:
            init_input_arr (list): 初始输入消息列表。
            add_nothink (bool): 是否添加 no_think 标记。
        """
        if self.contain_phantom_hint:
            # 此处省略了处理 Phantom Hint 的具体逻辑
            # 可能涉及将 past_trajectory 格式化并插入到 init_input_arr 中
            ...

        # 调用父类的 save_init_input 方法完成标准的保存流程
        return super().save_init_input(init_input_arr, add_nothink)