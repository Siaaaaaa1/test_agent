from typing import Any, Callable

from omegaconf import DictConfig

from agentevolver.client.env_client import EnvClient
from agentevolver.schema.trajectory import Trajectory


class BaseAgentFlow(object):
    """
    Agent 工作流基类 (BaseAgentFlow)。
    这个类定义了 Agent 执行任务时的基本接口和配置参数。
    它将“如何调用 LLM”和“交互的限制条件”封装起来，供子类使用。
    """

    def __init__(self,
                 llm_chat_fn: Callable,
                 tokenizer: Any,
                 config: DictConfig = None,
                 **kwargs):
        """
        初始化 BaseAgentFlow，加载必要的组件。

        Args:
            llm_chat_fn (Callable): 一个可调用函数，用于发送 Prompt 给 LLM 并获取回复。
            tokenizer (Any): 分词器，用于计算 Token 数量，处理截断等。
            config (DictConfig, optional): 全局配置对象 (通常来自 Hydra)。默认为 None。
            **kwargs: 其他关键字参数。
        """
        # super.__init__(**kwargs)
        self.llm_chat_fn: Callable = llm_chat_fn  # ⭐ 存储 LLM 对话函数 (这是 Agent 的"大脑"接口)
        self.tokenizer = tokenizer  # ⭐ 存储分词器
        self.config: DictConfig = config  # ⭐ 存储配置信息
        
        # 从配置中读取关键的限制参数，用于控制 Rollout 过程
        # max_steps: 允许 Agent 与环境交互的最大轮数 (防止无限循环)
        self.max_steps: int = self.config.actor_rollout_ref.rollout.multi_turn.max_steps  
        
        # max_model_len: 模型上下文窗口的最大长度限制 (超过此长度通常需要截断)
        self.max_model_len: int = self.config.actor_rollout_ref.rollout.max_model_len  
        
        # max_env_len: 环境返回的观测结果 (Observation) 的最大长度限制
        # (有些环境可能会返回巨大的报错信息或网页内容，需要限制以节省 Token)
        self.max_env_len: int = self.config.actor_rollout_ref.rollout.max_env_len  

    def execute(self, trajectory: Trajectory, env: EnvClient, instance_id: str, **kwargs) -> Trajectory:
        """
        抽象方法：在给定环境中执行任务，生成轨迹。
        
        注意：这个方法在基类中未实现，必须由子类（如 AgentFlow）重写。
        子类将在这里实现具体的 "Think -> Act -> Observe" 循环逻辑。

        Args:
            trajectory (Trajectory): 初始的轨迹对象 (或上下文管理器)，通常包含初始的任务指令。
            env (EnvClient): 环境客户端，用于向远程环境发送动作 (Step) 和获取反馈。
            instance_id (str): 当前正在操作的环境实例 ID (由 EnvWorker 创建)。
            **kwargs: 其他关键字参数。

        Returns:
            Trajectory: 执行完毕后的轨迹对象，包含了完整的交互历史。
        """
        raise NotImplementedError # ⭐ 这是一个接口定义，强制子类必须实现具体的执行逻辑