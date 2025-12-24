from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Reward(BaseModel):
    """
    奖励数据模型 (Reward)。
    
    用于封装 Agent 执行任务后的评估结果。
    """
    # 任务结果评分，通常 1.0 表示成功，0.0 表示失败
    outcome: float = Field(default=0.0)
    
    # 成功率 (可能用于多次尝试的统计)
    success_rate: float = Field(default=0.0)
    
    # "疯狂度" (Madness) 指标，可能用于衡量 Agent 行为的异常程度或探索性
    madness: float = Field(default=0.0)
    
    # 结果描述，默认为 "Outcome 1 denotes success, and 0 denotes failure."
    description: str = Field(default="Outcome 1 denotes success, and 0 denotes failure.")

    # 额外的元数据字典
    metadata: dict = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        """
        判断任务是否成功。
        默认逻辑是 outcome 大于 0 即视为成功。
        """
        return self.outcome > 0


class Trajectory(object):
    """
    轨迹对象 (Trajectory)。
    
    记录了 Agent 与环境交互的完整过程，包括每一步的动作、观察、最终的奖励以及任务指令。
    这不是一个 Pydantic 模型，而是一个标准的 Python 类。
    """
    data_id: str = ""      # 数据唯一标识
    rollout_id: str = ""   # 采样/推理过程的标识

    steps: List[dict] | None = None # 轨迹步骤列表，包含动作和观察
    query: str = ""                 # 任务指令 (Prompt)

    is_terminated: bool = False     # 任务是否已终止
    reward: Reward | None = None    # 关联的奖励对象

    metadata: dict | None = None    # 元数据
    
    def __init__(self, 
                 data_id: str="",
                 rollout_id: str="",
                 steps: List[dict] | None = None,
                 query: str="",
                 is_terminated: bool = False,
                 reward: Reward | None = None,
                 metadata: dict | None = None):
        """
        初始化轨迹对象。
        """
        self.data_id = data_id
        self.rollout_id = rollout_id
        if steps is not None:
            self.steps = steps  # 注意：在某些上下文管理器 (cmt) 初始化中可能会跳过此步骤，因为 steps 可能是一个函数
        self.query = query
        self.is_terminated = is_terminated
        self.reward = reward
        self.metadata = metadata if metadata is not None else {}
    

    @property
    def success(self) -> bool:
        """
        判断整条轨迹是否成功。
        需要同时满足：存在奖励对象 且 奖励结果大于 0。
        """
        return self.reward is not None and self.reward.outcome > 0


class Sample(BaseModel):
    """
    单个训练样本的数据模型 (Sample)。
    
    这是用于训练（如 PPO/RLHF）的底层数据结构，包含了经过 Tokenizer 处理后的 ID、Mask 等张量数据。
    """

    # 基础标识信息
    data_id: str = 0
    task_id: str = 0
    rollout_id: str = 0
    minor_index_id: int = 0
    
    # 原始消息列表 (Prompt + Response)
    messages: List[dict] = []
    # 额外信息
    extras: Dict[str, Any] = {}
    
    # ------ 模型输入的 Token 数据 ------
    # 完整的输入 ID (Prompt + Response)
    input_ids: List[int] = None
    # 仅 Prompt 部分的 ID
    prompt_ids: List[int] = None
    # 仅 Response 部分的 ID
    response_ids: List[int] = None
    
    # ------ Attention Masks ------
    attention_mask: List[int] = None
    prompt_attention_mask: List[int] = None
    response_attention_mask: List[int] = None
    
    # ------ Position IDs ------
    position_ids: List[int] = None
    prompt_position_ids: List[int] = None
    response_position_ids: List[int] = None
    
    # ------ Loss Masks (用于指示哪些 token 需要计算 Loss) ------
    loss_mask: List[int] = None
    prompt_loss_mask: List[int] = None
    response_loss_mask: List[int] = None
    
    # 奖励分数
    reward_scores: Dict[str, Any] = None
    
    # 长度限制配置
    max_prompt_len: int
    max_response_len: int
    max_model_len: int

    def truncate_output_ids(self) -> None:
        """
        截断输出 ID 以符合模型长度限制。
        
        该方法会检查 Prompt 和 Response 的长度，并根据 max_prompt_len 和 max_response_len 进行截断。
        如果发生了截断，会重新拼接完整的 input_ids, attention_mask 等字段。
        """
        # 1. 完整性检查：确保各部分数据长度一致
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask)
        assert len(self.prompt_ids) == len(self.prompt_attention_mask) == len(self.prompt_position_ids) == len(self.prompt_loss_mask)
        assert len(self.response_ids) == len(self.response_attention_mask) == len(self.response_position_ids) == len(self.response_loss_mask)
        assert isinstance(self.input_ids, list) and isinstance(self.prompt_ids, list) and isinstance(self.response_ids, list)

        truncate_any = False

        # 2. 检查 Prompt 长度
        if len(self.prompt_ids) > self.max_prompt_len:
            truncate_any = True
            print(f"-------------------------------------------------------------------------------------------------------")
            print(f"Warning: prompt_ids length {len(self.prompt_ids)} exceeds max_prompt_len {self.max_prompt_len}, truncating.")
            print(f"-------------------------------------------------------------------------------------------------------")
            # 目前策略：如果 Prompt 过长，直接抛出异常，要求用户调整输入数据（而不是默默截断导致上下文丢失）
            raise RuntimeError("Prompt length exceeds maximum allowed length. Please adjust the input data.")
            # 下面的代码是截断逻辑（保留后半部分），但在 raise 之后不会执行
            self.prompt_ids = self.prompt_ids[-self.max_prompt_len:]
            self.prompt_attention_mask = self.prompt_attention_mask[-self.max_prompt_len:]
            self.prompt_position_ids = self.prompt_position_ids[-self.max_prompt_len:]
            self.prompt_loss_mask = self.prompt_loss_mask[-self.max_prompt_len:]

        # 3. 检查 Response 长度
        if len(self.response_ids) > self.max_response_len:
            truncate_any = True
            print(f"-------------------------------------------------------------------------------------------------------")
            print(f"Warning: response_ids length {len(self.response_ids)} exceeds max_response_len {self.max_response_len}, truncating.")
            print(f"-------------------------------------------------------------------------------------------------------")
            # 截断 Response：保留前半部分
            self.response_ids = self.response_ids[: self.max_response_len]
            self.response_attention_mask = self.response_attention_mask[: self.max_response_len]
            self.response_position_ids = self.response_position_ids[: self.max_response_len]
            self.response_loss_mask = self.response_loss_mask[: self.max_response_len]

        # 4. 如果发生了截断，重新拼接完整的序列数据
        if truncate_any:
            self.input_ids = self.prompt_ids + self.response_ids
            self.attention_mask = self.prompt_attention_mask + self.response_attention_mask
            self.position_ids = self.prompt_position_ids + self.response_position_ids
            self.loss_mask = self.prompt_loss_mask + self.response_loss_mask

    def discard(self) -> None:
        """
        丢弃该经验样本。
        目前禁止直接调用此方法，抛出运行时错误。
        """
        raise RuntimeError('Never use this method.')