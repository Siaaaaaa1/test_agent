import json
import time
from typing import Optional, cast

from loguru import logger

from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.agent_flow.reward_calculator import RewardCalculator


class ModifiedAgentFlow(AgentFlow):
    """
    修改后的 Agent 工作流 (ModifiedAgentFlow)。

    继承自标准的 `AgentFlow`，主要用于在任务生成或探索阶段定制 Agent 的行为。
    它允许通过 `enable_context_generator` 参数来覆盖默认的上下文处理逻辑。
    
    例如，在随机探索生成新任务时，我们可能希望 Agent 更加“短视”或者不依赖复杂的历史上下文管理，
    这时就可以将其设置为 False。
    """

    def __init__(
        self,
        reward_calculator: Optional[RewardCalculator] = None,
        enable_context_generator: Optional[bool] = None, 
        **kwargs
    ):
        """
        初始化 ModifiedAgentFlow。

        Args:
            reward_calculator (Optional[RewardCalculator]): 
                奖励计算器实例。用于在 Agent 交互过程中即时计算奖励。
            
            enable_context_generator (Optional[bool]): 
                是否启用上下文生成器 (Context Generator) 的标志位。
                - 上下文生成器通常负责管理对话历史、裁剪上下文或添加系统提示。
                - 在 `LlmRandomSamplingExploreStrategy` 中，此参数被设置为 `False`，
                  意味着在探索阶段简化了 Agent 的上下文构建过程。
            
            **kwargs: 
                传递给父类 `AgentFlow` 的其他初始化参数，例如 `llm_chat_fn` (LLM 调用函数)、
                `tokenizer`、`config` 等。
        """
        # 调用父类 AgentFlow 的初始化方法，传递通用的参数
        super().__init__(reward_calculator, **kwargs)
        
        # 存储自定义的配置
        # 这个属性可能会在 step() 或 build_prompt() 等方法中被检查，以决定是否执行特定逻辑
        self._enable_context_generator = enable_context_generator