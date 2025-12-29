from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Task(BaseModel):
    """
    训练中使用的任务定义 (Task)。
    
    这个类封装了单个任务的所有必要信息，包括其在环境中的 ID、类型、具体指令（Query）以及用于评估的标准答案。
    它是 Dataset 和 TaskManager 之间传递的基本单元。
    """
    # 任务的唯一标识符，必须提供
    task_id: str = Field(default=...)

    # 任务所属的环境类型 (例如 "appworld", "webshop" 等)，默认为 "appworld"
    env_type: str = Field(default="appworld")
    
    # 是否为开放式查询 (Open Query) 的标志位。
    # 开放式查询通常意味着任务没有唯一解或明确的程序化停止条件，往往依赖 LLM Judge 进行评估。
    # FIXME: 原代码注释提示这是一个调试项，需要检查系统各处是否正确处理了这个属性。
    open_query: bool = Field() # FIXME debug, check if every instance handles this new attr. default False.

    # 任务的元数据字典，用于存储环境特定的配置或上下文信息
    metadata: dict = Field(default_factory=dict)
    
    # 具体的查询指令字符串。
    # 如果设置了此字段，在训练开始前，它将作为 Agent 接收到的输入指令（Prompt）。
    # 如果为 None，可能需要从 metadata 或环境中获取原始指令。
    query: Optional[str] = Field(default=None) 
    
    # 标准答案 (Ground Truth)。
    # 这是 TaskManager 在生成/演化任务（阶段 2）时生成的合成真值。
    # 它通常用于基于模型的评估器 (LLM Judge) 中，作为参考轨迹 (Reference Trajectory) 来评判 Agent 的表现。
    ground_truth: Optional[str] = Field(default=None, description="ground truth")
    
    metrics: Dict = Field(default_factory=dict, description="Task execution metrics")
    # 评估器类型标识。
    # "env": 表示使用环境自带的评估函数（如代码执行结果、网页状态匹配）。
    # 其他值: 可能指向特定的 LLM 评分器或规则评分器。
    evaluator: str = Field(default="env")


class TaskObjective(BaseModel):
    """
    仅在任务探索 (Exploration) 和提取 (Extraction) 阶段使用的任务目标封装类。
    
    它包裹了 `Task` 对象，并附加了在任务生成过程中计算出的元数据（如置信度和奖励）。
    这主要用于 TaskManager 内部的逻辑，用于筛选高质量的合成任务。
    """
    
    # 内部封装的核心 Task 对象
    task: Task = Field(..., description="task")
    
    # 生成该任务目标的置信度分数 (Confidence)。
    # 用于过滤掉生成的低质量或不确定的任务。
    confidence: Optional[float] = Field(None, description="confidence")
    
    # 该任务关联的奖励值 (Reward)。
    # 可能代表该任务的难度价值，或者完成该任务预期的回报。
    reward: Optional[float] = Field(None, description="reward")
    
    @property
    def objective(self):
        """
        快捷属性：获取任务的目标指令 (Query)。
        """
        return self.task.query
    
    @property
    def ground_truth(self):
        """
        快捷属性：获取任务的标准答案 (Ground Truth)。
        """
        return self.task.ground_truth
    
    @ground_truth.setter
    def ground_truth(self, v):
        """
        快捷设置器：设置任务的标准答案。
        这会直接修改内部 `task` 对象的 `ground_truth` 属性。
        """
        self.task.ground_truth = v