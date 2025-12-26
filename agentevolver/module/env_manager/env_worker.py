from typing import Optional
import uuid

from omegaconf import DictConfig
from loguru import logger
from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
# 导入不同的上下文管理器，用于处理对话历史的格式（如线性记录、带思考过程的记录等）
from agentevolver.module.context_manager.cmt_linear import Linear_CMT, ExtendedMessage
from agentevolver.module.context_manager.cmt_linear_think import LinearThinkCMT
from agentevolver.module.context_manager.cmt_context_clip import SelfContextClipCMT
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig
from typing import List, Dict, Any, Optional


class EnvWorker(object):
    """
    环境工作者类 (EnvWorker)
    职责：负责管理单个任务在环境中的完整生命周期，包括环境初始化、上下文管理、Agent 交互执行以及资源释放。
    """

    def __init__(self, task: Task, instance_id: str = None, thread_index: int = None, tokenizer=None,
                 config: DictConfig = None):
        """
        初始化 EnvWorker。

        参数 (Args):
            task (Task): 包含任务详情的任务对象（如任务ID、环境类型、指令等）。
            instance_id (str, optional): 实例的唯一标识符。如果不提供，将生成一个新的 UUID。
                                         用于在服务端区分不同的并发环境实例。
            thread_index (int, optional): 如果是多线程运行，这是线程的索引（用于日志或调试）。
            tokenizer (optional): 用于处理文本的分词器（计算 Token 数等）。
            config (DictConfig, optional): 全局配置对象，包含环境服务 URL、Agent 设置等。
        """
        self.config = config  # 保存配置
        # 初始化环境客户端，用于与远程环境服务通信 (通过 HTTP 请求)
        self.env = EnvClient(base_url=config.env_service.env_url)  
        
        # open_query 标志表示这是一个开放式问题，没有明确的程序化停止条件，
        # 需要 Agent 自己决定何时停止（通常通过调用 complete_task 工具）。
        self.is_open_query = task.open_query 
        
        self.task = task  # 保存任务对象
        self.env_type: str = task.env_type  # 环境类型 (例如 "appworld", "webshop")
        self.task_id: str = task.task_id  # 任务 ID (例如具体的某个用户场景 ID)

        # =========== [新增代码] ===========
        # 解析实际用于环境加载的物理 ID (Physical ID)
        # 1. 优先尝试从 metadata 获取 'env_sandbox_id' (由 ApiDriven 策略注入)
        # 2. 如果没有，则回退使用 task.task_id (适用于 Random 策略或标准数据集)
        self.env_physical_id = task.task_id
        if task.metadata and "env_sandbox_id" in task.metadata:
            self.env_physical_id = task.metadata["env_sandbox_id"]
            logger.debug(f"[EnvWorker] Using Physical ID {self.env_physical_id} instead of Logical ID {self.task_id}")
        # =================================
        
        # 如果未提供 instance_id，则生成一个随机 UUID。这确保了每次执行都是独立的。
        self.instance_id: str = instance_id if instance_id is not None else uuid.uuid4().hex  
        self.thread_index: int = thread_index  # 线程索引
        self.tokenizer = tokenizer  # 分词器

    def execute(self, data_id: str, rollout_id: str, traj_exp_config: TrajExpConfig, agent_flow: BaseAgentFlow, tmux:dict, stop:list[bool], system_prompt: Optional[str] = None, **kwargs) -> Trajectory:
        """
        核心执行方法：在环境中执行任务，生成并返回轨迹。

        参数 (Args):
            data_id (str): 数据的唯一标识符 (通常对应数据集中的索引)。
            rollout_id (str): 此次采样的标识符 (如果对同一个任务进行多次采样)。
            traj_exp_config (TrajExpConfig): 轨迹经验配置 (控制是否添加历史经验、训练模式等)。
            agent_flow (BaseAgentFlow): Agent 的工作流逻辑 (负责调用 LLM、解析动作、执行工具)。
            tmux (dict): 用于多线程状态同步或监控的字典 (如记录 step 数)。
            stop (list[bool]): 停止标志列表，用于外部强制停止执行。
            system_prompt (Optional[str]): 自定义的系统提示词 (System Prompt)，如果有则会覆盖或插入。
            **kwargs: 其他关键字参数。

        返回 (Returns):
            Trajectory: 任务执行生成的轨迹对象，包含所有的对话历史、动作和奖励。
        """

        try:
            # 1. 创建环境实例
            # 调用远程 API 初始化环境，返回初始状态 (通常包含 System Prompt 和 User Query)
            # params={'is_open_query': ...} 告诉环境这是一个开放式任务
            init_response = self.env.create_instance(env_type=self.env_type,
                                                    task_id=self.env_physical_id,
                                                    instance_id=self.instance_id,
                                                    params={'is_open_query': self.is_open_query})

            # init_response["state"] 通常是一个消息列表: [{"role": "system", ...}, {"role": "user", ...}]
            init_messages: list[dict] = init_response["state"]
            
            # 校验初始消息格式，确保包含 System 和 User 两条消息
            assert isinstance(init_messages, list) and len(init_messages)==2, "init_messages must be list and its length must be 2"
            
            # 2. 替换查询 (Query)
            # 如果 Task 对象中指定了新的 query (通常是好奇心模块生成的合成指令)，则替换环境默认的 query
            if self.task.query is not None:
                assert init_messages[-1]["role"] == "user", "the latest message from environment must be user query"
                init_messages[-1]["content"] = self.task.query
            else:
                # 否则使用环境返回的默认 query
                self.task.query = init_messages[-1]["content"]

            # 3. 插入自定义 System Prompt (如果存在)
            if system_prompt is not None:
                # FIXME: 这是一个临时的修复逻辑
                assert self.task.query is not None
                # 将占位符替换为实际的 query
                system_prompt = system_prompt.replace('[USER_QUESTION]', self.task.query)
                # 插入新的 prompt 到消息列表中，通常作为 User 消息之前的提示
                # 注意：这里逻辑删除了原来的 query 并插入了新的，可能是为了重构 Prompt 结构
                init_messages.insert(1, {"role": "user", "content": system_prompt})
                init_messages.pop() # 移除原来的最后一条消息 (original query)

            # 4. 初始化上下文管理器 (Context Manager / Trajectory)
            # 根据配置选择不同的上下文模板，用于管理对话历史的格式
            if self.config.actor_rollout_ref.rollout.context_template == "linear":
                # 线性上下文：标准的对话历史记录
                traj_cmt: Linear_CMT = Linear_CMT(self.config, self.tokenizer)
            elif self.config.actor_rollout_ref.rollout.context_template == "linear_think":
                # 思考上下文：支持 CoT (Chain of Thought) 的特殊格式
                traj_cmt: LinearThinkCMT = LinearThinkCMT(self.config, self.tokenizer)
            elif self.config.actor_rollout_ref.rollout.context_template == "context_selfclip":
                # 剪裁上下文：支持自动压缩/剪裁上下文
                traj_cmt: SelfContextClipCMT = SelfContextClipCMT(self.config, self.tokenizer, self.llm_chat_fn)
            else:
                raise ValueError(f"Unsupported context template: {self.config.actor_rollout_ref.rollout.context_template}")

            # 设置轨迹对象的元数据
            traj_cmt.data_id = data_id
            traj_cmt.rollout_id = rollout_id
            traj_cmt.task_id = self.task_id
            traj_cmt.instance_id = self.instance_id
            # traj_cmt.task_train_exp_mode = self.task.metadata.get("task_train_exp_mode")
            assert self.task.query is not None
            traj_cmt.query = self.task.query

            # (注释掉的代码可能是关于经验回放的逻辑，暂时未使用)
            # traj_exp_config.query=self.task.query
            # init_messages, traj_exp_config = self.exp_worker.manage_rollout_context(...)

            # 5. 执行 Agent Flow
            # 将一切准备就绪后，移交给 AgentFlow 进行循环交互
            # AgentFlow 会负责：LLM 推理 -> 解析动作 -> Env.step -> 更新 Context -> 循环
            traj_cmt: Trajectory = agent_flow.execute(
                context_manager=traj_cmt,  # 传入初始化好的上下文管理器
                init_messages=init_messages, # 传入初始消息
                env=self.env,              # 传入环境客户端
                instance_id=self.instance_id,
                tmux=tmux,                 # 传入状态监控
                stop=stop,                 # 传入停止信号
                thread_index=self.thread_index,
                task_id=self.task_id,
                traj_exp_config=traj_exp_config,
                data_id=data_id,
                rollout_id=rollout_id,
                query=self.task.query,
                **kwargs
            )  # ⭐ 执行任务并生成完整轨迹
            
            # 6. 释放环境资源
            # 任务完成后，必须释放远程环境实例，避免资源泄漏
            self.env.release_instance(self.instance_id)

        except Exception as e:
            # 异常处理：如果在执行过程中发生任何错误，也要确保释放环境实例
            self.env.release_instance(self.instance_id)
            raise RuntimeError(f"env.create_instance failed! error={e.args}") from e

        # 返回生成的轨迹
        return traj_cmt