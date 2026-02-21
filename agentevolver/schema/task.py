from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class Task(BaseModel):
    task_id: str = Field(default=...)
    env_type: str = Field(default="appworld")
    open_query: bool = Field(default=False) 
    metadata: dict = Field(default_factory=dict)
    
    query: Optional[str] = Field(default=None, description="The active query for this task (New/Current)") 
    
    # [关键字段] 当前任务的标准答案 (经过清洗的 Code 或 总结出的 Code)
    ground_truth: Optional[Any] = Field(default=None, description="Current refined Ground Truth (Code)")
    
    # [新增] 原始轨迹步骤 (Raw Steps)，用于保留完整的执行记录
    raw_trajectory: Optional[List[Any]] = Field(default=None, description="Raw execution trajectory steps")
    
    # [重命名] 种子任务的 Ground Truth
    origin_ground_truth: Optional[Any] = Field(default=None, description="The Ground Truth of the seed task")
    
    # [重命名] 种子任务的 Query
    origin_query: Optional[str] = Field(default=None, description="The query of the seed task")

    metrics: Dict = Field(default_factory=dict, description="Task execution metrics")
    evaluator: str = Field(default="env")


class TaskObjective(BaseModel):
    """
    仅在任务探索 (Exploration) 和提取 (Extraction) 阶段使用的任务目标封装类。
    """
    
    # 内部封装的核心 Task 对象
    task: Task = Field(..., description="task")
    
    # 生成该任务目标的置信度分数 (Confidence)。
    confidence: Optional[float] = Field(None, description="confidence")
    
    # 该任务关联的奖励值 (Reward)。
    reward: Optional[float] = Field(None, description="reward")
    
    @property
    def objective(self):
        return self.task.query
    
    @property
    def ground_truth(self):
        return self.task.ground_truth
    
    @ground_truth.setter
    def ground_truth(self, v):
        self.task.ground_truth = v