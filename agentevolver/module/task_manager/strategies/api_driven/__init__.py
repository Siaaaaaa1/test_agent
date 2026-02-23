import json
import os
import random
import time
import itertools
import threading
import copy
from typing import List, Dict, Any, Optional, Set, Callable, Union, Tuple
from types import SimpleNamespace
from collections import defaultdict

from loguru import logger
from omegaconf import DictConfig
import traceback

# 基础模块
from agentevolver.module.task_manager.base import LlmClient
from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from agentevolver.utils.utils import extract_json_from_str
from agentevolver.utils.debug_utils import debug_log
from agentevolver.module.task_manager.strategies.api_driven.prompts.intra_domain import parse_intra_purpose_from_response
from agentevolver.module.task_manager.strategies.api_driven.prompts.cross_domain import parse_cross_purpose_from_response
from agentevolver.module.task_manager.rewards.integrated_reward import IntegratedRewardCalculator
# 环境与执行模块
from agentevolver.module.env_manager.env_worker import EnvWorker, TrajExpConfig
from agentevolver.module.task_manager.agent_flow import ModifiedAgentFlow

# Prompt 与 Profile 相关
from agentevolver.module.task_manager.prelude_profiles import appworld, bfcl, webshop
from agentevolver.module.task_manager.strategies.api_driven.prompts import (
    INTRA_DOMAIN_SELECTOR_PROMPT,
    parse_intra_selector,
    INTRA_DOMAIN_GENERATOR_PROMPT,
    parse_intra_generator,
    CROSS_DOMAIN_SELECTOR_PROMPT,
    parse_cross_selector,
    CROSS_DOMAIN_GENERATOR_PROMPT,
    parse_cross_generator,
    get_agent_interaction_system_prompt,
    get_intra_schema,
    get_cross_schema,
)
from agentevolver.module.task_manager.strategies.api_driven.prompts.prompt_summarize import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)
from agentevolver.client.llm_client_mix import Mix_DashScopeClient
from agentevolver.module.task_manager.strategies.api_driven.prompts.prompt_summarize_first import (
    get_direct_verify_prompt,
    parse_direct_verification
)

# AppWorld 依赖注入
from appworld.environment import AppWorld

UNIVERSAL_INFO_PROVIDERS = {"notes", "gmail", "simple_messages", "calendar", "contacts"}


class ApiDrivenExploreStrategy(TaskExploreStrategy):
    """
    API 驱动的探索策略类
    核心功能：
    1. 任务生成 (Generation)：基于预设的 App 和 API 知识库，通过 LLM 生成 Intra-domain (同领域) 和 Cross-domain (跨领域) 的指令任务。
    2. 任务执行 (Explore)：将生成的任务下发到沙盒环境中，使用多模型轮询的方式驱动 Agent 执行，收集环境反馈轨迹。
    3. 任务总结 (Summarize)：对执行成功的轨迹进行归纳总结，提取并提炼为高质量的强化学习/微调数据集。
    """

    # ================= [新增] 预定义的合法 APP 组合白名单 =================
    VALID_CROSS_COMBINATIONS = [
        ['venmo', 'splitwise'], ['venmo', 'gmail'], ['venmo', 'phone'], ['venmo', 'simple_note'],
        ['amazon', 'venmo'], ['amazon', 'splitwise'], ['amazon', 'todoist'], ['amazon', 'simple_note'], ['amazon', 'gmail'], ['amazon', 'phone'],
        ['spotify', 'phone'], ['spotify', 'gmail'], ['spotify', 'simple_note'],
        ['gmail', 'todoist'], ['gmail', 'file_system'], ['gmail', 'simple_note'], ['gmail', 'phone'],
        ['simple_note', 'spotify'], ['simple_note', 'gmail'], ['simple_note', 'phone'], ['simple_note', 'todoist'],
        ['phone', 'gmail'], ['phone', 'simple_note'], ['phone', 'todoist'],
        ['todoist', 'simple_note'], ['todoist', 'gmail'], ['todoist', 'phone'],
        ['splitwise', 'venmo'], ['splitwise', 'gmail'], ['splitwise', 'phone'], ['splitwise', 'simple_note'],
        ['file_system', 'gmail'], ['file_system', 'simple_note'], ['file_system', 'todoist']
    ]

    def _get_valid_app_chain(self, target_apps: List[str]) -> Optional[List[str]]:
        """
        [新增] 基于 VALID_CROSS_COMBINATIONS 过滤并重排 APP，寻找合理的逻辑执行链 (A -> B -> C)。
        如果无法形成合法链条，则返回 None。
        
        Args:
            target_apps (List[str]): 待验证的 APP 列表 (支持 2 个、3 个或更多)
            
        Returns:
            Optional[List[str]]: 经过逻辑排序的 APP 列表，或者 None
        """
        # 将合法的 2-APP 组合转换为 set，实现 O(1) 的快速查找
        valid_edges = set(tuple(comb) for comb in self.VALID_CROSS_COMBINATIONS)
        
        # 遍历目标 APP 的所有可能的排列顺序
        for perm in itertools.permutations(target_apps):
            is_valid_chain = True
            
            # 检查当前排列下，所有相邻的 APP 是否都在白名单中
            for i in range(len(perm) - 1):
                if (perm[i], perm[i+1]) not in valid_edges:
                    is_valid_chain = False
                    break
                    
            if is_valid_chain:
                # 找到了一条完全合法的调用顺序 (例如: amazon -> venmo -> splitwise)
                return list(perm) 
                
        # 所有排列都走不通，说明这几个 APP 拼在一起逻辑不通，直接抛弃
        return None

    def __init__(self, tokenizer, config: DictConfig, llm_client: Optional[LlmClient] = None, **kwargs):
        """
        初始化策略类，挂载环境配置、加载知识库并初始化所需的大模型客户端。
        
        Args:
            tokenizer: 用于计算 token 的分词器实例
            config: 系统的全局配置字典 (DictConfig)
            llm_client: 可选传入的外部通用大模型客户端实例
            **kwargs: 其他动态注入的配置参数 (例如 env_profile, active_apps 等)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        
        self._env_profile = kwargs.get("env_profile")

        # ================= [Init 修改] =================
        # 1. 常规客户端 (self.llm_client): 用于 generate 和 summarize 阶段，通常是廉价或默认模型
        if llm_client:
            self.llm_client = llm_client
        else:
            logger.warning("[ApiDriven] No LlmClient passed, creating a standard LlmClient.")
            self.llm_client = LlmClient(config)
            
        self._summarize_client = kwargs.get("llm_client_summarize", self.llm_client)

        # 2. 探索专用客户端 (self.explore_client): 仅用于 explore (Agent 交互) 阶段，通常是能力最强的混合路由模型
        logger.info("[ApiDriven] Initializing Mix_DashScopeClient specifically for 'explore' phase.")
        try:
            self.explore_client = Mix_DashScopeClient(
                model_name="HY-Qwen3-235B-A22B-Instruct-2507", 
                temperature=config.get("exploration_llm_temperature", 0.7)
            )
        except Exception as e:
            logger.error(f"[ApiDriven] Failed to init Mix_DashScopeClient: {e}. Fallback to standard client.")
            self.explore_client = self.llm_client
        
        # 3. 统计相关初始化：用于记录各个模型在 explore 阶段的成功率，便于后期分析模型表现
        self._stats_lock = threading.Lock()
        self._total_finished_tasks = 0
        self._model_performance = defaultdict(lambda: {"attempts": 0, "success": 0})
        # ================= [Init 结束] =================

        self._max_llm_retries = kwargs.get("max_llm_retries", 5)
        self._lock = threading.Lock() 
        
        # --- 路径与文件配置：指定知识库和记忆存储的硬盘位置 ---
        self.api_knowledge_path = kwargs.get(
            "api_knowledge_path", 
            "./agentevolver/preprocess/output/appworld_tool_manual.json"
        )
        self.task_labels_path = kwargs.get(
            "task_labels_path", 
            "./agentevolver/preprocess/output/task_app_labels_train.json"
        )
        
        base_memory_dir = "data/memory/api_driven"
        self.intra_memory_path = kwargs.get("intra_memory_path", os.path.join(base_memory_dir, "intra_domain_success.json"))
        self.cross_memory_path = kwargs.get("cross_memory_path", os.path.join(base_memory_dir, "cross_domain_success.json"))
        
        # 划定允许探索的 App 范围
        self.active_apps = set(kwargs.get("active_apps", ['amazon','gmail','spotify','venmo','simple_note','todoist','splitwise','phone','file_system']))
        
        # 将静态 JSON 文件加载到内存
        self.api_knowledge = self._load_json(self.api_knowledge_path)
        if not self.api_knowledge:
            logger.warning(f"API Knowledge not found at {self.api_knowledge_path}.")

        # 初始化用于探索环境的沙盒 ID 池，并创建一个循环迭代器以便持续分配
        self.sandbox_ids_pool = self._load_sandbox_task_ids(self.task_labels_path)
        self.sandbox_id_iterator = itertools.cycle(self.sandbox_ids_pool)
        
        # 加载历史探索记忆，用于去重和策略调优
        self.intra_memory_data = self._load_json(self.intra_memory_path)
        self.explored_intra_apps = set(self.intra_memory_data.get("explored_apps", []))
        self.cross_memory_data = self._load_json(self.cross_memory_path)
        
        self.env_profile_name = self.config.get("env_service", {}).get("env_type", "appworld")

        logger.info(f"[ApiDriven] Initialized. Strategy ready.")

    # ================= [辅助逻辑] 统计与空闲检查 (新增) =================
    
    def _record_model_result(self, model_name: str, is_success: bool):
        """
        记录单个大模型在环境中执行任务的成功或失败次数。
        
        Args:
            model_name (str): 调用的模型名称
            is_success (bool): 任务是否执行成功（通常指 Reward >= 0.8）
        """
        # 使用互斥锁保证多线程下统计数据的绝对安全
        with self._stats_lock:
            self._model_performance[model_name]["attempts"] += 1
            if is_success:
                self._model_performance[model_name]["success"] += 1

    def _report_progress_if_needed(self):
        """
        定期（每完成50个任务）在控制台打印当前各个大模型的执行表现统计报表。
        """
        with self._stats_lock:
            self._total_finished_tasks += 1
            current_count = self._total_finished_tasks
            
            # 每当完成数量是 50 的倍数时，打印统计日志
            if current_count % 50 == 0:
                logger.info(f"\n====== Model Performance Report (Processed {current_count} Tasks) ======")
                logger.info(f"{'Model Name':<40} | {'Calls':<6} | {'Succ':<6} | {'Fail':<6} | {'Rate':<8}")
                logger.info("-" * 80)
                
                # 按照调用次数从高到低排序模型
                sorted_stats = sorted(self._model_performance.items(), key=lambda x: x[1]['attempts'], reverse=True)
                
                # 遍历计算并打印成功率
                for model, stats in sorted_stats:
                    attempts = stats["attempts"]
                    success = stats["success"]
                    fail = attempts - success
                    rate = (success / attempts * 100) if attempts > 0 else 0.0
                    logger.info(f"{model:<40} | {attempts:<6} | {success:<6} | {fail:<6} | {rate:.2f}%")
                logger.info("========================================================================\n")

    def _get_model_idle_score(self, model_name: str) -> Tuple[int, int]:
        """
        获取指定模型的空闲分数，用于在多个主模型之间做负载均衡路由。
        
        Args:
            model_name (str): 待评估的模型名称
            
        Returns:
            Tuple[int, int]: 返回 (并发空闲数, RPM空闲数) 的元组，值越大代表模型越空闲可以立即接客。
        """
        # 如果当前的客户端不支持状态查询（说明不是 MixClient），直接返回兜底值
        if not hasattr(self.explore_client, "_get_model_state"):
            return (0, 0)

        try:
            # 获取底层的模型并发控制锁和时间戳记录
            state = self.explore_client._get_model_state(model_name)
            semaphore = state.get("semaphore")
            # semaphore._value 表示当前还剩下多少个并发额度可以使用
            concurrency_free = semaphore._value if semaphore else 0
            
            rate_lock = state.get("rate_lock")
            timestamps = state.get("timestamps")
            rpm_free = 0
            
            # 计算 RPM (Requests Per Minute) 剩余额度
            if rate_lock and isinstance(timestamps, list):
                with rate_lock:
                    now = time.time()
                    # 筛出最近一分钟内的调用记录
                    valid_timestamps = [t for t in timestamps if now - t < 60.0]
                    max_rpm = getattr(self.explore_client, "MAX_RPM", 30)
                    rpm_free = max_rpm - len(valid_timestamps)
            
            return (concurrency_free, rpm_free)
            
        except Exception as e:
            logger.warning(f"Failed to check idle score for {model_name}: {e}")
            return (0, 0)

    # ================= [核心] 执行循环逻辑 =================

    def explore(self, task: Task, data_id: str, rollout_id: str) -> List[Trajectory]:
        """
        执行探索任务的核心入口：将生成的自然语言 Query 下发至真实环境交由 Agent 解决。
        
        工作流：
        1. 锁定环境沙盒 ID (优先使用生成阶段的 ID 以保证数据一致性)。
        2. 智能路由大模型 (比较多个模型的空闲度，优先调用不排队的模型)。
        3. 实例化 EnvWorker 和 AgentFlow，并在沙盒中执行任务直至成功或触发最大步数。
        4. 评估执行轨迹（Reward），对失败情况进行多模型 fallback 重试。
        
        Args:
            task (Task): 待执行的任务对象，内含 user_query 和 metadata。
            data_id (str): 数据流的唯一追踪 ID。
            rollout_id (str): 该轮 rollout 的批次 ID。
            
        Returns:
            List[Trajectory]: 返回包含执行步骤和最终 Reward 的轨迹列表。如果彻底失败，返回空列表或包含失败原因的最后一个轨迹。
        """
        # [修改] 优先复用在 Generator 阶段绑定好的沙盒 ID，确保前后环境数据绝对一致
        if task.metadata is None:
            task.metadata = {}
            
        real_sandbox_id = task.metadata.get("env_sandbox_id")
        if not real_sandbox_id:
            # 如果是遗留任务没有 sandbox_id，则动态分配一个
            real_sandbox_id = self.get_next_sandbox_id()
            task.metadata["env_sandbox_id"] = real_sandbox_id
            
        debug_log(self.config, "api_explore_start", {
            "task_id": task.task_id,
            "data_id": data_id,
            "phase": task.metadata.get('phase'),
            "real_sandbox_id": real_sandbox_id
        })

        # --- 动态构建模型尝试列表 ---
        tier1_model_a = "HY-Qwen3-235B-A22B-Instruct-2507"
        tier1_model_b = "DeepSeek-V3-Online"
        
        # 探测当前哪个一线大模型比较闲
        score_a = self._get_model_idle_score(tier1_model_a)
        score_b = self._get_model_idle_score(tier1_model_b)
        
        logger.debug(f"[Explore] Model Capacity - {tier1_model_a}: {score_a}, {tier1_model_b}: {score_b}")
        
        # 优先级判断：空闲分数高的排前面，一样空闲则随机打乱
        if score_a > score_b:
            tier1_models = [tier1_model_a, tier1_model_b]
        elif score_b > score_a:
            tier1_models = [tier1_model_b, tier1_model_a]
        else:
            tier1_models = [tier1_model_a, tier1_model_b]
            random.shuffle(tier1_models)
            
        # 准备兜底的二线模型
        tier2_models = ["azure-gpt-5-mini", "azure-gpt-5"]
        candidate_models = tier1_models + tier2_models
        
        last_trajectory = None
        max_steps = self.config.get("max_explore_step", 50) 

        # 遍历候选模型列表（多重 fallback 机制）
        for model_name in candidate_models:
            logger.info(f"[Explore] Task {data_id}: Trying model '{model_name}'...")

            thread_idx = 0
            if task.metadata and 'thread_index' in task.metadata:
                thread_idx = task.metadata['thread_index']
            
            # 初始化与沙盒交互的 Worker
            env_worker = EnvWorker(
                task=task,
                config=self.config, 
                thread_index=thread_idx,
                tokenizer=self.tokenizer
            )

            # 动态构造当前遍历到的大模型的聊天闭包函数 (Chat Closure)
            if model_name in ["DeepSeek-V3-Online","HY-Qwen3-235B-A22B-Instruct-2507"]:
                sampling_params = {
                    "temperature": self.config.get("exploration_llm_temperature", 0.5),
                    "top_p": self.config.get("exploration_llm_top_p", 0.9),
                    "top_k": self.config.get("exploration_llm_top_k", 50),
                    "model": model_name,
                }
                llm_chat_fn = self._get_llm_chat_fn(self.explore_client, sampling_params=sampling_params)
            else:
                sampling_params = {"model": model_name}
                llm_chat_fn = self._get_llm_chat_fn(self.explore_client, sampling_params=sampling_params)

            # 初始化环境奖励计算器（Judge）
            reward_calculator = IntegratedRewardCalculator(task=task)

            # 组装完整的智能体工作流架构
            agent_flow = ModifiedAgentFlow(
                llm_chat_fn=llm_chat_fn,
                tokenizer=self.tokenizer,
                config=self.config,
                enable_context_generator=False,
                reward_calculator=reward_calculator 
            )
            agent_flow._reward_calculator = reward_calculator
            agent_flow.max_steps = max_steps
            agent_flow.max_model_len = self.config.get("max_model_len", 102400)

            try:
                # 获取该特定环境的角色系统 Prompt
                system_prompt = get_agent_interaction_system_prompt(self._env_profile)

                # 开始正式在环境中驱动智能体进行探索
                trajectory = env_worker.execute(
                    data_id=data_id, 
                    rollout_id=rollout_id,
                    traj_exp_config=TrajExpConfig(add_exp=False),
                    agent_flow=agent_flow,
                    tmux={'step': [0], 'token': [0]},
                    stop=[False],
                    system_prompt=system_prompt,
                )
                
                last_trajectory = trajectory
                
                is_success = False
                current_score = 0.0
                
                # [关键判定] 通过 Judge 计算最终反馈分数，评估任务是否圆满完成
                if trajectory and trajectory.reward:
                    current_score = trajectory.reward.outcome
                    judge_reason = getattr(trajectory.reward, "reason", "No detailed reasoning provided")
                    logger.info(f"📝 [Judge Result] Task: {data_id} | Model: {model_name} | Score: {current_score}\nReasoning: {judge_reason}")
                    
                    # 只有分数 >= 0.8 才被视为真正有效的可供微调的成功轨迹
                    if current_score >= 0.8:
                        is_success = True
                    else:
                        logger.warning(f"[Explore] Model {model_name} score {current_score} < 0.8. Marked as Fail.")
                else:
                    logger.warning(f"[Explore] Model {model_name} produced no reward object.")

                # 记录该模型的表现
                self._record_model_result(model_name, is_success)
                
                # 如果成功，立即返回轨迹结束探索，节省算力
                if is_success:
                    logger.info(f"[Explore] Model {model_name} SUCCEEDED (Score: {current_score}, Steps: {len(trajectory.steps)}). Returning result.")
                    self._report_progress_if_needed()
                    return [trajectory]
                else:
                    # 如果失败，清理资源并进入 next loop 尝试下一个更强的兜底模型
                    logger.warning(f"[Explore] Model {model_name} FAILED or Score too low. Retrying with next model...")
                    try:
                        if hasattr(env_worker, 'env') and env_worker.env:
                             pass 
                    except: pass
                    continue

            except Exception as e:
                # 捕获并记录环境崩溃或大模型接口熔断的极端异常
                logger.error(f"[Explore] Critical Error with model {model_name}: {e}")
                traceback.print_exc()
                self._record_model_result(model_name, False)
                continue

        # 走到这里意味着所有配置的模型都尝试过了，并且全部失败
        logger.warning(f"[Explore] All models failed.")
        self._report_progress_if_needed()

        # 返回最后一个模型跑出的失败轨迹（供后续分析或打负面标签）
        if last_trajectory:
            logger.warning(f"[Explore] Returning result from last model ({candidate_models[-1]}).")
            return [last_trajectory]
        
        return []

    # ================= 总结逻辑 =================

    def summarize(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        统一的轨迹总结提炼入口。
        基于任务的元数据 phase (阶段) 来路由到对应的具体总结方法。
        
        Args:
            task (Task): 原生成阶段封装的任务描述
            trajectory (Trajectory): Explore 阶段产生的交互流水账
            
        Returns:
            List[TaskObjective]: 提炼打包好的、包含强化学习所需的 Reward 和 GroundTruth 的标准目标集合。
        """
        # 如果轨迹完全为空，没有价值，直接抛弃
        if not trajectory or not trajectory.steps:
            return []

        phase = task.metadata.get("phase", "unknown")
        results = []
        # 路由
        if phase == "intra":
            results = self.summarize_intra(task, trajectory)
        elif phase == "extra":
            results = self.summarize_cross(task, trajectory)
        
        return results if results else []

    def get_next_sandbox_id(self) -> str:
        """
        从预设的沙盒池中循环安全地获取下一个可用的沙盒 ID。
        
        Returns:
            str: 如 'train_001' 等环境实例标识
        """
        try:
            return next(self.sandbox_id_iterator)
        except StopIteration:
            # 兜底：如果迭代器出问题，返回一个默认的安全 ID
            return "train_001"

    # ================= 上下文构建辅助方法 (Text-Based Format) =================
    
    def _get_enhanced_context(self, app_name: str, anchor_api_names: List[str]) -> Tuple[str, str, str]:
        """
        将 JSON 格式的 API 知识库转化为大模型更容易阅读和理解的格式化纯文本。
        
        Args:
            app_name (str): 目标 App 的名称
            anchor_api_names (List[str]): 在此轮任务中被强行选定的核心 API (锚点 API)
            
        Returns:
            Tuple[str, str, str]: (所有 App 的概况, 目标 App 的全量 API 简述, 锚定 API 的详细参数文档)
        """
        # 1. 构建所有可用 App 的全局描述
        global_lines = []
        for app, details in self.api_knowledge.items():
            desc = details.get("description", "No description available.")
            global_lines.append(f'APP: "{app}"\ndescription: "{desc}"')
        all_apps_info = "\n\n".join(global_lines)

        # 2. 提取指定 App 内部所有的 API 的基础描述
        target_app_data = self.api_knowledge.get(app_name, {})
        target_app_apis = target_app_data.get("apis", {})
        
        api_lines = []
        for api_key, details in target_app_apis.items():
            desc = details.get("description", "No description available.")
            full_call_name = details.get("call_name", api_key)
            api_lines.append(f"{full_call_name}: {desc}")
        target_app_apis_info = "\n".join(api_lines)

        # 3. 为被选中的 Anchor API 提取详细的参数 (parameters) 和返回值 (returns) 结构
        anchor_details_list = []
        for input_api_name in anchor_api_names:
            # 尝试剥离多级路径前缀获取短名称
            if "." in input_api_name:
                short_name = input_api_name.split('.')[-1]
                full_path = input_api_name
            else:
                short_name = input_api_name
                full_path = f"apis.{app_name}.{short_name}"

            api_data = None
            if short_name in target_app_apis:
                api_data = target_app_apis[short_name]
            
            # 如果知识库里有这个 API，组装成带有 Markdown 风格的参数说明块
            if api_data:
                correct_call_name = api_data.get("call_name", full_path)
                params_str = json.dumps(api_data.get("parameters", []), indent=2)
                returns_str = json.dumps(api_data.get("returns", {}), indent=2)
                
                block = (
                    f'api_name: "{correct_call_name}"\n'
                    f'description: "{api_data.get("description", "")}"\n'
                    f'parameters: {params_str}\n'
                    f'returns: {returns_str}'
                )
                anchor_details_list.append(block)
        
        anchor_apis_detailed_info = "\n\n---\n\n".join(anchor_details_list) if anchor_details_list else "None provided."

        return all_apps_info, target_app_apis_info, anchor_apis_detailed_info
    
    # ================= 任务生成 (Generation) =================
    def _fetch_real_table_data(self, sandbox_id: str, app_tables: Dict[str, List[str]]) -> str:
        """
        核心数据桥梁探针：根据大模型选择的表名，动态探测并提取底层 SQLite 的真实状态快照。
        
        Args:
            sandbox_id (str): 目标沙盒环境对应的 ID (如 'train_001')
            app_tables (Dict[str, List[str]]): LLM 选拔器输出的要求查询的表结构 (如 {"amazon": ["orders"]})
            
        Returns:
            str: 包含真实数据且经过字段清洗和截断处理后的 JSON 字符串
        """
        fetched_data = {}
        try:
            # 唤起对应的真实环境上下文以挂载底层数据库
            with AppWorld(task_id=sandbox_id) as world:
                for app_name, table_names in app_tables.items():
                    # 检查环境的 ORM models 是否存在该应用
                    table_names = table_names[:2]

                    if not hasattr(world.models, app_name):
                        logger.warning(f"[Data Fetch] 环境中不存在 App: {app_name}")
                        continue
                    
                    app_models = getattr(world.models, app_name)
                    app_data = {}
                    
                    for table_name in table_names:
                        # --- 模糊匹配逻辑：大模型容易造出单复数、大小写不一的表名，在此处做容错映射 ---
                        table_lower = table_name.lower()
                        # 粗暴去 's' 匹配单数形式的 ORM Class Name
                        singular_name = table_lower[:-1] if table_lower.endswith('s') else table_lower
                        
                        matched_cls = None
                        # 遍历该 App 命名空间下的所有类进行碰撞
                        for attr_name in dir(app_models):
                            attr_lower = attr_name.lower()
                            if attr_lower == table_lower or attr_lower == singular_name:
                                matched_cls = getattr(app_models, attr_name)
                                break
                        
                        # 如果确实是继承自底层 SQLModel 且有 .all() 接口的表模型
                        if matched_cls and hasattr(matched_cls, "all") and callable(matched_cls.all):
                            try:
                                records = matched_cls.all()
                                if not records:
                                    continue
                                
                                parsed_records = []
                                # 【限制策略】每张表最多取前 2 条记录，防止塞入 Prompt 时导致 Context Window 爆炸
                                for r in records[:2]:  
                                    record_dict = {}
                                    # 将 ORM 对象解包为字典
                                    for k, v in r.__dict__.items():
                                        # 数据脱敏：过滤掉系统内部前缀变量、密码和认证 Token 等大模型不需要的纯噪音
                                        if k.startswith("_") or "password" in k.lower() or "token" in k.lower():
                                            continue
                                        # 文本截断：保留核心语义即可，拒绝几千字的邮件正文喧宾夺主
                                        if isinstance(v, str) and len(v) > 60:
                                            v = v[:57] + "..."
                                        record_dict[k] = v
                                    parsed_records.append(record_dict)
                                    
                                app_data[table_name] = parsed_records
                            except Exception as e:
                                logger.warning(f"[Data Fetch] 提取表数据失败 {app_name}.{table_name}: {e}")
                                
                    if app_data:
                        fetched_data[app_name] = app_data
                        
        except Exception as e:
            logger.error(f"[Data Fetch] 无法加载 AppWorld 沙盒 {sandbox_id}: {e}")
            
        # 返回 JSON 供 Prompt 直接内嵌使用
        return json.dumps(fetched_data, indent=2, ensure_ascii=False) if fetched_data else "{}"

    def generate_intra_task(self, app_name: str, task: Task = None) -> List[Task]:
        """
        两阶段数据生成：同领域 (Intra-domain)
        第一阶段：(Selector) 大模型从 App 全局视角挑选出一组最符合人类逻辑的 API 搭配，并指出需要查看哪些表。
        第二阶段：(Generator) Python 使用探针抓取真实数据，大模型根据真实的底层表单内容创造出带有模糊性、探索性的 User Query。
        
        Args:
            app_name (str): 目标进行任务生成的单体 App
            task (Task, optional): 待拷贝继承元信息的种子任务
            
        Returns:
            List[Task]: 生成出的最终任务列表
        """
        generated_tasks = []
        app_data = self.api_knowledge.get(app_name, {})
        all_apis = list(app_data.get("apis", {}).keys())
        
        # [修复] 防御机制：由于生成规则是要将 2 到 3 个 API 拼成一组，如果该 App 的 API 数量不到 2 个，直接拦截放弃避免越界
        if len(all_apis) < 2: 
            return []

        # [新增] 在所有操作开始前，提前分配并锁定一个实体沙盒 ID，保证生成和执行都在同一个次元
        real_sandbox_id = self.get_next_sandbox_id()

        # 第一阶段前置：随机产生 3 组各不相同的候选 API 组合
        candidate_groups = []
        for _ in range(3):
            api_count = random.choice([2, 3])
            candidate_groups.append(random.sample(all_apis, min(api_count, len(all_apis))))

        groups_str = "\n".join([f"Group {i+1}: {g}" for i, g in enumerate(candidate_groups)])
        # 获取该 App 的描述和概况提供给大模型
        all_apps_info, target_app_apis_info, _ = self._get_enhanced_context(app_name, [])

        # --- Stage 1: 呼叫大模型评委 (Selector) ---
        db_schema_overview = get_intra_schema([app_name]) # [新增] 获取动态 Schema

        selector_prompt = INTRA_DOMAIN_SELECTOR_PROMPT.format(
            CANDIDATE_COUNT=3,
            APP_NAME=app_name,
            ALL_APPS_DESC=all_apps_info,
            TARGET_APP_API_DESCS=target_app_apis_info,
            CANDIDATE_GROUPS_STR=groups_str,
            DB_SCHEMA_OVERVIEW=db_schema_overview # [新增] 填补坑位
        )
        
        sel_response = self._chat_with_retry(messages=[{"role": "user", "content": selector_prompt}])
        if not sel_response: return []
        
        # 解析评委选定的 API 组合和明确点名需要的数据库表名
        selection_data = parse_intra_selector(sel_response.content)
        if not selection_data or "selected_apis" not in selection_data: return []

        selected_apis = selection_data["selected_apis"]
        required_tables = selection_data.get("required_tables", [])

        # --- 中转层：深入底层沙盒打捞该任务对应的真实数据 ---
        # [修改] 传入前面分配好的沙盒 ID 获取该沙盒在此刻的真实物化视图
        db_content = self._fetch_real_table_data(real_sandbox_id, {app_name: required_tables})
        
        # 获取该组被选定 API 的极其详细的参数规格说明
        _, _, anchor_apis_detailed_info = self._get_enhanced_context(app_name, selected_apis)

        # --- Stage 2: 呼叫大模型生成器 (Generator) ---
        # 利用真数据和大纲，要求大模型不把话说明白，创造出一个 "Exploratory" (探索性) 的指令
        generator_prompt = INTRA_DOMAIN_GENERATOR_PROMPT.format(
            APP_NAME=app_name,
            SELECTED_APIS=json.dumps(selected_apis),
            ALL_APIS_BRIEF=target_app_apis_info,
            ANCHOR_API_DETAILS=anchor_apis_detailed_info,
            DB_CONTENT=db_content
        )

        gen_response = self._chat_with_retry(messages=[{"role": "user", "content": generator_prompt}])
        if not gen_response: return []

        parsed_scenarios = parse_intra_generator(gen_response.content)
        
        # 收尾：组装符合系统流转格式的 Task 数据集对象
        for scenario in parsed_scenarios:
            new_task = copy.deepcopy(task) if task else Task()
            new_task.query = scenario.get("user_query", "")
            # 将核心上下文塞入 metadata，尤其是锁定的沙盒 ID (env_sandbox_id)
            new_task.metadata = {
                "phase": "intra",
                "env_sandbox_id": real_sandbox_id, # [新增] 向 explore() 方法透传唯一的环境令牌
                "target_app": app_name,
                "target_api": scenario.get("target_api", ""),
                "selected_apis_context": selected_apis,
                "required_tables": required_tables,
                "origin_query": new_task.query
            }
            generated_tasks.append(new_task)

        return generated_tasks

    def generate_cross_task(self, target_apps: List[str], task: Task = None) -> List[Task]:
        """
        两阶段数据生成：跨领域 (Cross-domain)
        逻辑核心与 Intra 基本一致，难点在于如何按照特定的统计概率模型来分配来自 2-3 个 App 里的共计 2-5 个 API，从而产生高价值的“跨域信息串联”任务（例如从 Venmo 扣款后在 Todoist 里划掉任务）。
        
        Args:
            target_apps (List[str]): 参与串联的 App 名称列表
            task (Task, optional): 待拷贝继承元信息的种子任务
            
        Returns:
            List[Task]: 生成出的最终任务列表
        """
        ordered_apps = self._get_valid_app_chain(target_apps)
        if not ordered_apps:
            logger.debug(f"[Cross-Gen] 拦截并丢弃不合理的跨域组合/顺序: {target_apps}")
            return []
            
        # 覆写为具有合理逻辑顺序的 APP 列表 (传给后续 Prompt 组装时，大模型会按照这个顺序列举)
        target_apps = ordered_apps 

        generated_tasks = []
        
        # 按照需求设定的特定权重 [40%, 30%, 20%, 10%] 随机抽选本次任务将要使用的 API 总数
        total_apis = random.choices([2, 3, 4, 5], weights=[0.4, 0.3, 0.2, 0.1])[0]
        # 安全保障：强制 API 总数不能少于牵扯到的 App 总数，否则会导致分配黑洞
        total_apis = max(total_apis, len(target_apps))
        # 需求定义：2-3个API提供3组候选，更多则提供5组候选
        num_candidates = 3 if total_apis in [2, 3] else 5

        # 提前分配沙盒 ID 建立环境绑定
        real_sandbox_id = self.get_next_sandbox_id()

        candidate_groups = []
        apps_info_lines = []
        
        # 把每个牵涉到的 App 的介绍信息全部揉合在一起
        for app in target_apps:
            all_apps_info, app_apis_info, _ = self._get_enhanced_context(app, [])
            apps_info_lines.append(f"--- APP: {app} ---\n{app_apis_info}")

        # 循环产生多组候选组合给后续的大模型评委过目
        for _ in range(num_candidates):
            current_group = {}
            # 基础分配：每个受害（划掉）涉及的 App 必须至少有一个 API 参与
            remaining = total_apis - len(target_apps)
            allocations = {app: 1 for app in target_apps}
            
            # 边界保护：理论上前面被 max 限制后不会走到这个 if
            if remaining < 0: 
                allocations = {app: (1 if i < total_apis else 0) for i, app in enumerate(target_apps)}
            else:
                # 剩余名额随机分赃
                for _ in range(remaining):
                    allocations[random.choice(target_apps)] += 1

            # 真正去知识库里采摘对应数量的 API 标识符
            for app, count in allocations.items():
                if count == 0: continue
                all_apis = list(self.api_knowledge.get(app, {}).get("apis", {}).keys())
                current_group[app] = random.sample(all_apis, min(count, len(all_apis)))
            candidate_groups.append(current_group)

        groups_str = "\n".join([f"Group {i+1}: {json.dumps(g)}" for i, g in enumerate(candidate_groups)])
        apps_info_str = "\n\n".join(apps_info_lines)

        # --- Stage 1: 呼叫大模型评委 ---
        db_schema_overview = get_cross_schema(target_apps) # [新增]

        selector_prompt = CROSS_DOMAIN_SELECTOR_PROMPT.format(
            CANDIDATE_COUNT=num_candidates,
            APP_COUNT=len(target_apps),
            APPS_INFO_STR=apps_info_str,
            CANDIDATE_GROUPS_STR=groups_str,
            DB_SCHEMA_OVERVIEW=db_schema_overview # [新增]
        )

        sel_response = self._chat_with_retry(messages=[{"role": "user", "content": selector_prompt}])
        if not sel_response: return []

        # 获取评委选中的唯一跨域组合，并获悉两个 App 各自需要翻阅什么表
        selection_data = parse_cross_selector(sel_response.content)
        if not selection_data or "selected_apis" not in selection_data: return []

        selected_apis = selection_data["selected_apis"]
        required_tables = selection_data.get("required_tables", {})

        # --- 中转层：深入底层沙盒打捞跨域的真实数据 ---
        # [修改] 传递沙盒令牌，探针会自动去不同 App 的 SQLModel 里提取真数据并拼接
        db_content = self._fetch_real_table_data(real_sandbox_id, required_tables)
        
        # 将多个 App 被选中的核心 API 文档揉在一起
        anchor_details_combined = []
        for app, apis in selected_apis.items():
            _, _, anchor_detail = self._get_enhanced_context(app, apis)
            anchor_details_combined.append(f"[{app} Details]\n{anchor_detail}")
        
        all_apis_brief_combined = []
        for app in target_apps:
            _, app_apis_info, _ = self._get_enhanced_context(app, [])
            all_apis_brief_combined.append(f"[{app} APIs]\n{app_apis_info}")
        
        # --- Stage 2: 呼叫大模型生成器 ---
        # 强制要求大模型写出隐晦的“跨域搜索联动行动”指令
        generator_prompt = CROSS_DOMAIN_GENERATOR_PROMPT.format(
            APPS_NAMES=", ".join(target_apps),
            SELECTED_APIS_JSON=json.dumps(selected_apis),
            ALL_APIS_BRIEF="\n".join(all_apis_brief_combined),
            ANCHOR_API_DETAILS="\n".join(anchor_details_combined),
            DB_CONTENT=db_content
        )

        gen_response = self._chat_with_retry(messages=[{"role": "user", "content": generator_prompt}])
        if not gen_response: return []

        parsed_scenarios = parse_cross_generator(gen_response.content)

        # 收尾：组装 Task 并植入环境追踪芯片 (env_sandbox_id)
        for scenario in parsed_scenarios:
            new_task = copy.deepcopy(task) if task else Task()
            new_task.query = scenario.get("user_query", "")
            new_task.metadata = {
                "phase": "extra",
                "env_sandbox_id": real_sandbox_id, # [新增] 锁定执行环境
                "target_apps": target_apps,
                "source_api": scenario.get("source_info_api", ""),
                "target_api": scenario.get("target_action_api", ""),
                "logic_pattern": scenario.get("logic_pattern", ""),
                "selected_apis_context": selected_apis,
                "required_tables": required_tables,
                "origin_query": new_task.query
            }
            generated_tasks.append(new_task)

        return generated_tasks

    # ================= 阶段总结逻辑 (Summarize) =================

    def summarize_intra(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        单域探索反思与提取：让大模型审核刚刚跑完的轨迹流水账，归纳意图并将之包装为 RL 训练标准结构。
        
        Args:
            task (Task): 任务描述体
            trajectory (Trajectory): Agent 执行返回的过程步骤和反馈
            
        Returns:
            List[TaskObjective]: 标准化提取出的带置信度和原始信息的强化学习训练目标列表。
        """
        client = self._summarize_client
        llm_fn = self._get_llm_chat_fn(client)
        
        # [安全脱敏] 将前几个步骤里面可能包含人类直接 Prompt 指令的地方进行 MASK，防止模型作弊
        masked_trajectory = copy.deepcopy(trajectory)
        if len(masked_trajectory.steps) > 2:
            if masked_trajectory.steps[1].get('role') == 'user':
                masked_trajectory.steps[1]['content'] = '[MASKED]'
            if masked_trajectory.steps[2].get('role') == 'user':
                masked_trajectory.steps[2]['content'] = '[MASKED]'

        # 调取并组装总结反思专用的系统和用户 Prompt
        system_prompt, user_prompt = get_task_summarize_prompt(
            [masked_trajectory], old_objectives=task.query, profile=self._env_profile
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 呼叫大模型分析这段流水账，吐出精简干净的提炼数据
        try:
            llm_response = llm_fn(messages=messages)
            llm_output = llm_response["content"]
        except Exception as e:
            logger.error(f"[Summarize Intra] LLM call failed: {e}")
            return []
        
        # 拷贝母板并打上合成的标签
        task_copy = task.copy()
        task_copy.evaluator = 'synthetic'
        
        # 通过解析器将 LLM 输出的 JSON 剥离重组为完整的任务对象
        tasks = parse_tasks_from_response(task_copy, llm_output)

        # 提取环境裁定官给出的分数和评语
        reward_info = None
        if trajectory.reward:
            reward_info = {
                "outcome": trajectory.reward.outcome,
                "reason": getattr(trajectory.reward, "reason", "No reason provided")
            }

        # 挂载极其重要的追溯元数据
        for task_obj in tasks:
            if task_obj.task.metadata is None:
                task_obj.task.metadata = {}
            
            task_obj.task.origin_query = task.query
            task_obj.task.metadata["summary_analysis_process"] = llm_output
            task_obj.task.metadata["execution_reward"] = reward_info
            task_obj.task.metadata["source_data_id"] = task.metadata.get("data_id")
            task_obj.task.metadata["generation_type"] = "evolved"

        return tasks

    def summarize_cross(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        跨域探索反思与提取：机制与 summarize_intra 基本一致，专门处理多 App 串联行动。
        
        Args:
            task (Task): 任务描述体
            trajectory (Trajectory): 跨越不同 App 生成的轨迹步骤
            
        Returns:
            List[TaskObjective]: 打包的 RL 训练目标
        """
        client = self._summarize_client
        llm_fn = self._get_llm_chat_fn(client)
        
        # 数据脱敏防作弊
        masked_trajectory = copy.deepcopy(trajectory)
        if len(masked_trajectory.steps) > 2:
            if masked_trajectory.steps[1].get('role') == 'user':
                masked_trajectory.steps[1]['content'] = '[MASKED]'
            if masked_trajectory.steps[2].get('role') == 'user':
                masked_trajectory.steps[2]['content'] = '[MASKED]'
        
        system_prompt, user_prompt = get_task_summarize_prompt(
            [masked_trajectory], old_objectives=task.query, profile=self._env_profile
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            llm_response = llm_fn(messages=messages)
            llm_output = llm_response["content"]
        except Exception as e:
            logger.error(f"[Summarize Cross] LLM call failed: {e}")
            return []
            
        task_copy = task.copy()
        task_copy.evaluator = 'synthetic'
        
        tasks = parse_tasks_from_response(task_copy, llm_output)
        
        reward_info = None
        if trajectory.reward:
            reward_info = {
                "outcome": trajectory.reward.outcome,
                "reason": getattr(trajectory.reward, "reason", "No reason provided")
            }

        for task_obj in tasks:
            if task_obj.task.metadata is None:
                task_obj.task.metadata = {}
            
            task_obj.task.origin_query = task.query
            task_obj.task.metadata["summary_analysis_process"] = llm_output
            task_obj.task.metadata["execution_reward"] = reward_info
            task_obj.task.metadata["source_data_id"] = task.metadata.get("data_id")
            task_obj.task.metadata["generation_type"] = "evolved"
            
        return tasks

    def verify_direct_gt(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        """
        针对原始指令进行直接验证模式。
        即判断 Agent 盲跑产生的轨迹是否直接构成了解决该 Query 的标准 Ground Truth (真值参考)。
        如果大模型认定符合，则提炼轨迹将其净化成 Ground Truth 保存。
        
        Args:
            task (Task): 原始查询任务
            trajectory (Trajectory): 环境执行反馈的整条操作链
            
        Returns:
            Optional[TaskObjective]: 成功验证则返回包含标准代码 (refined_code) 的目标对象，不符合逻辑则返回 None 遗弃。
        """
        # 第一层铁门限：如果底层环境已经给该轨迹打了极低的分（说明代码语法错或全乱来），就无需再花钱请 LLM 看了
        if trajectory.reward and trajectory.reward.outcome < 0.3:
            logger.info(f"[Verify] Skipping task {task.task_id} due to very low reward {trajectory.reward.outcome}")
            return None

        client = self._summarize_client
        llm_fn = self._get_llm_chat_fn(client)

        system_prompt, user_prompt = get_direct_verify_prompt(
            task, trajectory, self._env_profile
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            llm_response = llm_fn(messages=messages)
            llm_output = llm_response["content"]
        except Exception as e:
            logger.error(f"[Verify] LLM call failed: {e}")
            return None

        # 解析大模型是否给该过程盖章“合格”
        result = parse_direct_verification(llm_output)

        if result.get("is_valid"):
            new_task = task.copy()
            new_task.evaluator = 'synthetic' 
            
            # 植入净化后的标准代码，直接化身为该任务的 Ground Truth
            new_task.ground_truth = result.get("refined_code", "")
            new_task.origin_query = task.query 
            
            # 打标签保存
            if new_task.metadata is None: new_task.metadata = {}
            new_task.metadata["verification_reason"] = result.get("reason")
            new_task.metadata["verification_confidence"] = result.get("confidence")
            new_task.metadata["data_pair_type"] = "direct_verified"
            
            return TaskObjective(
                task=new_task,
                confidence=result.get("confidence", 0.9),
                reward=trajectory.reward.outcome if trajectory.reward else 0.0
            )
        else:
            # 验证失败，说明大模型认为这个轨迹属于瞎碰死耗子没逻辑，不要
            logger.info(f"[Verify] Task {task.task_id} rejected: {result.get('reason')}")
            return None

    # ================= 辅助私有方法 =================

    def _chat_with_retry(self, messages: List[Dict], **kwargs) -> Optional[Any]:
        """
        带退避指数休眠的重试封装的极简请求函数。
        用以抵抗临时网络中断或并发限流导致的请求中断。
        """
        for i in range(self._max_llm_retries):
            try:
                response = self.llm_client.chat(messages=messages, **kwargs)
                
                # 做一层对 DashScope (产出文本) 与 OpenAI (产出对象) 的抽象适配
                if isinstance(response, str):
                    if response.strip():
                        return SimpleNamespace(content=response)
                elif response and hasattr(response, 'content') and response.content:
                    return response
                    
            except Exception as e:
                logger.warning(f"LLM call failed: {e}. Retry {i+1}...")
            
            if i < self._max_llm_retries - 1:
                # 每次重试等待倍增 (2, 4, 8, 16秒)
                time.sleep(2 ** i)
                
        return None

    def _load_sandbox_task_ids(self, path: str) -> List[str]:
        """从预处理的文件中导入系统中所有合格的沙盒环境标识 ID"""
        if not os.path.exists(path):
            return ["train_001"]
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [item["TaskID"] for item in data if "TaskID" in item]
        except:
            return ["train_001"]

    def _check_api_called(self, trajectory: Trajectory, api_name: str) -> bool:
        """纯工具人函数：暴力遍历轨迹日志，检查特定 API 是否被 Agent 召唤过"""
        if not trajectory or not trajectory.steps: return False
        for step in trajectory.steps:
            if step.get('role') == "tool" and not step.error:
                if api_name and api_name in step.tool_name: return True
        return False

    def _check_app_usage(self, trajectory: Trajectory, app_name: str) -> bool:
        """纯工具人函数：检查整条轨迹日志内是否有涉及到指定的 App"""
        if not trajectory or not trajectory.steps: return False
        app_apis = self.api_knowledge.get(app_name, {}).get("apis", {}).keys()
        for step in trajectory.steps:
            if step.get('role') == "tool":
                if app_name and app_name.lower() in step.tool_name.lower(): return True
                for api in app_apis:
                    if api in step.tool_name: return True
        return False

    def _load_json(self, path: str) -> Dict:
        """极简 JSON 文件安全读取器"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {}

    def _save_intra_memory(self, app_name: str):
        """将某个已被有效探索过的单体 App 持久化记录，避免重复无意义劳作"""
        os.makedirs(os.path.dirname(self.intra_memory_path), exist_ok=True)
        current_data = self._load_json(self.intra_memory_path)
        current_apps = set(current_data.get("explored_apps", []))
        current_apps.add(app_name)
        with open(self.intra_memory_path, 'w', encoding='utf-8') as f:
            json.dump({"explored_apps": list(current_apps)}, f, indent=2)

    def _save_cross_memory(self, metadata: Dict):
        """记录跨域探索成功的组合日志"""
        os.makedirs(os.path.dirname(self.cross_memory_path), exist_ok=True)
        current_data = self._load_json(self.cross_memory_path)
        if "logs" not in current_data: current_data["logs"] = []
        
        log_entry = {
            "synthesized_user_query": metadata.get("synthesized_user_query"),
            "info_app": metadata.get("info_app"),
            "exec_app": metadata.get("exec_app"),
            "timestamp": time.time()
        }
        current_data["logs"].append(log_entry)
        
        with open(self.cross_memory_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2)

    def _get_llm_chat_fn(self, llm_client: LlmClient, sampling_params: Optional[dict] = None) -> Callable:
        """
        核心适配器：将通用的 LLM Client 调用包装为符合 AgentFlow 期待的高阶函数形式。
        闭包机制在此锁住了客户端实例和独属配置。
        """
        def llm_chat(
            messages: list[dict[str, str]],
            custom_sampling_params: Optional[dict] = None,
            request_id: Optional[str] = None,
        ) -> dict:
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            input_messages = copy.deepcopy(messages)
            res = None
            
            # 再套一层强行护驾：拦截 Agent 循环时可能突发的断网抽风
            for i in range(self._max_llm_retries):
                try:
                    res = llm_client.chat(
                        messages=input_messages, sampling_params=updated_sampling_params
                    )
                    if res is not None and res != "":
                        break
                except Exception as e:
                    logger.warning(f"llm_chat retry {i} error: {e}")
                    time.sleep(2**i)

            # 当一切手段耗尽，使用温和的安全词退场以免主进程连环报错
            if res is None or res == "":
                res = "I apologize, but I encountered an error generating a response."

            return {
                "role": "assistant",
                "content": res,
            }

        return llm_chat