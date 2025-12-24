import abc
from typing import Any, Protocol

from agentevolver.schema.task import Task, TaskObjective


class LlmClient(Protocol):
    """
    LLM 客户端协议接口。
    
    定义了与大语言模型交互的标准接口，期望返回生成的文本内容字符串。
    使用 Protocol 使得任何实现了 chat 方法的类都可以被视为 LlmClient，无需显式继承。
    """
    def chat(
        self, messages: list[dict[str, str]], sampling_params: dict[str, Any]
    ) -> str: 
        """
        发送聊天请求并获取回复内容。

        Args:
            messages (list[dict]): 消息列表，格式通常为 [{"role": "user", "content": "..."}, ...]
            sampling_params (dict): 采样参数，如 temperature, max_tokens 等。

        Returns:
            str: 模型生成的文本回复。
        """
        ...

class LlmRawClient(Protocol):
    """
    原始 LLM 客户端协议接口。
    
    与 LlmClient 类似，但期望返回包含完整元数据的原始响应字典（如 token 使用统计、logprobs 等）。
    """
    def chat(
        self, messages: list[dict[str, str]], sampling_params: dict[str, Any]
    ) -> dict: 
        """
        发送聊天请求并获取完整响应对象。

        Returns:
            dict: 包含生成内容及元数据的字典。
        """
        ...

class TaskObjectiveRetrieval(abc.ABC):
    """
    任务目标检索器 (TaskObjectiveRetrieval) 的抽象基类。
    
    该组件用于存储和检索已经生成的任务目标 (TaskObjective)。
    在任务演化过程中，它主要用于：
    1. 历史记录：记录基于某个种子任务已经生成了哪些新任务。
    2. 去重：防止生成重复的任务目标。
    3. 上下文辅助：在生成新任务时，检索相似或相关的旧任务作为参考。
    """

    @abc.abstractmethod
    def retrieve_objectives(self, task: Task) -> list[TaskObjective]: 
        """
        根据给定的种子任务检索相关的历史任务目标。

        Args:
            task (Task): 用于查询的种子任务。

        Returns:
            list[TaskObjective]: 相关的任务目标列表。
        """
        ...

    @abc.abstractmethod
    def add_objective(self, objective: TaskObjective): 
        """
        将一个新的任务目标添加到存储中。

        Args:
            objective (TaskObjective): 需要存储的新生成的任务目标。
        """
        ...
    
    @abc.abstractmethod
    def reset(self):
        """
        重置存储状态，清空所有记录。通常在每一轮演化开始时调用。
        """
        ...



class NaiveTaskObjectiveRetrieval(TaskObjectiveRetrieval):
    """
    朴素的任务目标检索器实现。
    
    使用内存字典进行简单的基于 Task ID 的精确匹配存储。
    适用于单次运行且不需要跨会话持久化的场景。
    """

    def __init__(self):
        # 初始化内存字典
        # 键: 任务 ID (str)
        # 值: 该任务 ID 关联的任务目标列表 (list[TaskObjective])
        # 注意：假设单次训练会话只包含相同 env_type 的任务，因此直接使用 task_id 作为键是安全的。
        self._mp: dict[str, list[TaskObjective]] = {}

    def retrieve_objectives(self, task: Task) -> list[TaskObjective]:
        """
        检索指定任务 ID 已有的所有目标。
        """
        if task.task_id not in self._mp:
            return []
        return self._mp[task.task_id]

    def add_objective(self, objective: TaskObjective):
        """
        将新的任务目标添加到内部映射中。
        如果映射中不存在该任务 ID，则初始化一个空列表。

        Args:
            objective (TaskObjective): 要添加的任务目标。
        """
        # 如果当前任务 ID 不在字典中，初始化列表
        if objective.task.task_id not in self._mp:
            self._mp[objective.task.task_id] = []  # ⭐ Initialize an empty list for the task ID if it doesn't exist

        # 将目标追加到列表中
        self._mp[objective.task.task_id].append(objective)  # ⭐ Add the objective to the list for the task ID

    def reset(self):
        """
        清空内部映射，移除所有存储的任务目标。
        """
        self._mp = {}  # ⭐ Clear the internal mapping