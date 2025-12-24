import copy
import os
import time
from typing import Callable, NotRequired, Optional, Sequence, TypedDict, Unpack
import uuid

from loguru import logger

from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig
from agentevolver.module.task_manager.agent_flow import ModifiedAgentFlow
from agentevolver.module.task_manager.base import LlmClient
from agentevolver.module.env_manager.env_worker import EnvWorker
from agentevolver.module.task_manager.strategies.common.prompts.prompt_explore import get_agent_interaction_system_prompt
from agentevolver.module.task_manager.strategies.common.prompts.prompt_summarize import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)
from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.module.task_manager.prelude_profiles import bfcl, appworld
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory


# 定义初始化参数的类型结构，用于类型检查和文档化
class LlmRandomSamplingExploreStrategyProps(TypedDict):
    max_explore_step: int          # 探索过程中的最大步数
    max_llm_retries: int           # LLM 调用失败时的最大重试次数
    env_url: str                   # 环境服务的 URL 地址
    exploration_llm_temperature: NotRequired[float] # 探索时 LLM 的温度（控制随机性），可选
    exploration_llm_top_p: NotRequired[float]       # Nucleus sampling 参数，可选
    exploration_llm_top_k: NotRequired[int]         # Top-k sampling 参数，可选
    
    

class LlmRandomSamplingExploreStrategy(TaskExploreStrategy):
    """
    基于 LLM 随机采样的任务探索策略。
    
    工作流程：
    1. Explore: 使用较高温度（高随机性）的 LLM Agent 在环境中自由探索，生成操作轨迹。
    2. Summarize: 将生成的轨迹交给另一个 LLM，反向推导出这就轨迹对应的用户指令（Query）和标准答案（Ground Truth）。
    """
    
    def __init__(self, * , tokenizer, config,**kwargs: Unpack[LlmRandomSamplingExploreStrategyProps]):
        """
        初始化探索策略。

        Args:
            tokenizer: 分词器，用于计算 token 数量等。
            config: 全局配置对象。
            **kwargs: 包含 LlmRandomSamplingExploreStrategyProps 定义的参数。
        """
        self._tokenizer = tokenizer
        # 配置对象，将在 EnvWorker 和 AgentFlow 中使用
        self._config = config
        
        # 提取参数，设置默认值
        self._max_llm_retries = kwargs.get("max_llm_retries", 3)
        self._max_explore_step = kwargs.get("max_explore_step", 10)
        self._env_service_url = kwargs.get("env_url")
        
        # 设置探索时的采样参数，默认比较高以鼓励多样性
        self._exploration_llm_temperature=kwargs.get("exploration_llm_temperature", 1.0)
        self._exploration_llm_top_p=kwargs.get("exploration_llm_top_p", 1.0)
        self._exploration_llm_top_k=kwargs.get("exploration_llm_top_k", 1)
        
    
    def explore(self, task: Task, data_id: str, rollout_id: str) -> list[Trajectory]:
        """
        步骤 1: 探索 (Explore)
        
        在指定任务环境中运行 Agent，生成交互轨迹。

        Args:
            task (Task): 种子任务（用于初始化环境）。
            data_id (str): 数据 ID。
            rollout_id (str): Rollout ID。

        Returns:
            list[Trajectory]: 生成的轨迹列表（通常包含一条轨迹）。
        """
        # 1. 初始化环境工作者 (EnvWorker)，负责与环境交互
        env_worker = EnvWorker(
            task=task,
            config=self._config, # FIXME: 代码注释指出必须使用这些参数，且默认值可能不正确，需注意
            thread_index=0,
            tokenizer=self._tokenizer
        )
        
        # 2. 构造 LLM 聊天函数，注入探索专用的采样参数（高 Temperature）
        llm_chat_fn = self._get_llm_chat_fn(self.llm_client_explore,
            sampling_params={
                "temperature": self._exploration_llm_temperature,
                "top_p": self._exploration_llm_top_p,
                "top_k": self._exploration_llm_top_k,
            }
        )
        
        # 3. 初始化 Agent 工作流 (AgentFlow)
        # 使用 ModifiedAgentFlow，这是专门为任务生成修改过的 Agent 逻辑
        agent_flow: BaseAgentFlow = ModifiedAgentFlow(
            enable_context_generator=False, # 禁用上下文生成器，简化流程
            llm_chat_fn=llm_chat_fn,
            tokenizer=self._tokenizer,
            config=self._config,
        )
        agent_flow.max_steps = self._max_explore_step  # 设置最大交互步数
        agent_flow.max_model_len = 102400 # TODO: 硬编码的最大长度，后续需优化
        
        # 4. 执行 Agent，获取轨迹
        # execute 方法会循环调用 agent_flow.step() 直到任务结束或达到最大步数
        traj = env_worker.execute(
            data_id=data_id,
            rollout_id=rollout_id,
            traj_exp_config=TrajExpConfig(add_exp=False), # 不添加到经验回放池
            agent_flow=agent_flow,
            tmux={
                'step':[0],  # 用于多线程控制的共享变量
                'token':[0],
            },
            stop=[False], # 停止标志位
            # 注入探索专用的系统提示词 (System Prompt)
            system_prompt=get_agent_interaction_system_prompt(self.env_profile),
        )

        return [traj]
    
    def summarize(self, task: Task, trajectory: Trajectory) -> list[TaskObjective]:
        """
        步骤 2: 总结 (Summarize)
        
        分析探索生成的轨迹，反向推断出这条轨迹完成的任务指令 (Query) 和结果 (Ground Truth)。

        Args:
            task (Task): 原始种子任务。
            trajectory (Trajectory): 探索步骤生成的轨迹。

        Returns:
            list[TaskObjective]: 包含新生成的 Query 和 Ground Truth 的任务目标列表。
        """
        # 1. 构造总结用的 LLM 函数，使用与探索相同的采样参数
        llm_fn = self._get_llm_chat_fn(
            self.llm_client_summarize,
            sampling_params={
                "temperature": self._exploration_llm_temperature,
                "top_p": self._exploration_llm_top_p,
                "top_k": self._exploration_llm_top_k,
            }
        )
        
        # 2. 检索旧的任务目标，用于去重或作为负样本提示 LLM 避免生成重复内容
        old_objectives = self._old_retrival.retrieve_objectives(task)
        
        # 3. 数据脱敏/掩码处理
        # 种子任务中可能包含原始的 Query，为了避免 Summarize 模型直接抄袭原始 Query，
        # 我们将轨迹中涉及原始 Query 的部分（通常在开头）替换为 [MASKED]。
        # [0]: system prompt, [1]: 可能是 user query 或 user role 下的 system prompt, [2]: user query
        trajectory.steps[1]['content'] = '[MASKED]'
        trajectory.steps[2]['content'] = "[MASKED]"
        
        # 4. 获取总结用的 Prompt
        # 包含轨迹内容、旧任务列表以及环境特定的格式要求
        system_prompt, user_prompt = get_task_summarize_prompt(
            [trajectory], old_objectives, self.env_profile
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # 5. 调用 LLM 生成总结
        llm_output = llm_fn(messages=messages)["content"]
        
        # 6. 解析结果并封装
        task = task.copy()
        task.evaluator = 'synthetic' # 标记评估器为合成类型
        # 从 LLM 的文本输出中解析出结构化的任务 (Query, GT 等)
        tasks = parse_tasks_from_response(task, llm_output)
        return tasks
    
    def _get_llm_chat_fn(self, llm_client:LlmClient, sampling_params: Optional[dict] = None) -> Callable:
        """
        辅助函数：封装 LLM 客户端调用，增加重试机制和参数合并。

        Args:
            llm_client (LlmClient): 基础 LLM 客户端。
            sampling_params (Optional[dict]): 默认采样参数。

        Returns:
            Callable: 封装后的聊天函数。
        """
        def llm_chat(
            messages: list[dict[str, str]],
            custom_sampling_params: Optional[dict] = None,
            request_id: Optional[str] = None,
        ) -> dict:
            """
            实际执行聊天的闭包函数。
            输入格式: [{"role": "system", "value": "..."}, {"role": "user", "value": "..."}]
            输出格式: {"role": "assistant", "content": "..."}
            """
            # 合并采样参数：自定义参数 > 默认参数
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            input_messages = copy.deepcopy(messages)
            res = None
            
            # 带指数退避 (Exponential Backoff) 的重试循环
            for i in range(self._max_llm_retries):
                try:
                    res = llm_client.chat(
                        messages=input_messages, sampling_params=updated_sampling_params
                    )
                    # 如果结果不为空，跳出重试
                    if res is not None and res != "":
                        break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    # 指数退避：第一次等 1s，第二次等 2s，第三次等 4s...
                    time.sleep(2**i)

            assert res is not None and res != "", f"LLM client failed to chat"
            
            # 返回标准化的响应字典
            return {
                "role": "assistant",
                "content": res,
            }

        return llm_chat