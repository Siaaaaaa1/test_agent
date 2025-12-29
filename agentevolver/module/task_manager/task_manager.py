from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import functools
import hashlib
import json
import os
import random
import time
from typing import (
    Optional, Sequence, TypedDict, Unpack, List, Any, Iterable
)

from loguru import logger
from omegaconf import DictConfig
import requests
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

# 内部模块引入
from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.task_manager import adapter
from agentevolver.module.task_manager.adapter import OnflyRlDataset, to_rl_dataset
from agentevolver.module.task_manager.data_mixture import MixtureStrategy
from agentevolver.module.task_manager.filters.llm_filter import LlmFilter
from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.module.task_manager.filters.filters import NaiveTaskPostFilter, TaskPostFilter

from agentevolver.module.task_manager.base import LlmClient, TaskObjectiveRetrieval
from agentevolver.module.task_manager.strategies.random import LlmRandomSamplingExploreStrategy
from agentevolver.module.task_manager.strategies.api_driven import ApiDrivenExploreStrategy

from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from verl.utils.dataset.rl_dataset import RLHFDataset
from agentevolver.utils.debug_utils import debug_log

# --- 类型定义 ---

class TaskManagerProps(TypedDict):
    """TaskManager 的可选配置参数"""
    num_explore_threads: int  # 探索任务时的线程数
    n: int # 膨胀系数：每个种子任务期望演化出的新任务数量

class RewardProps(TypedDict):
    """奖励与评分器相关的配置"""
    original_grader: str  # 原始任务（种子）使用的评分器
    synthetic_grader: str # 合成任务（演化出的）使用的评分器

# --- 工具函数 ---

def get_exploration_strategy(name: str, strategy_args, *, tokenizer, config, llm_client) -> TaskExploreStrategy:
    logger.info(f"loading exploration strategy {name}")
    print(f"[get_exploration_strategy] Loading strategy: {name}, args: {strategy_args}") # ADDED
    debug_log(config, "init_exploration_strategy", {"name": name, "args": strategy_args})
    if name == "random":
        print(f"[get_exploration_strategy] Initialized LlmRandomSamplingExploreStrategy") # ADDED
        return LlmRandomSamplingExploreStrategy(
            tokenizer=tokenizer, 
            config=config, 
            **strategy_args
            )
    elif name == "api_driven":
        print(f"[get_exploration_strategy] Initialized ApiDrivenExploreStrategy") # ADDED
        return ApiDrivenExploreStrategy(
            tokenizer=tokenizer, 
            config=config, 
            llm_client=llm_client,
            **strategy_args
        )
    else:
        print(f"[get_exploration_strategy] Error: Unknown strategy {name}") # ADDED
        raise NotImplementedError(f"exploration strategy {name} not implemented")

# ================= TaskManager 类 =================

class TaskManager(object):
    """
    任务管理器：负责任务的生命周期管理
    包括：加载种子、触发探索生成、过滤低质量任务、维护生成断点（Checkpoint）。
    """

    def __init__(
        self,
        config: DictConfig,
        exploration_strategy: str,
        env_profile: EnvProfile,
        exploration_strategy_args,
        llm_client: LlmClient,
        old_retrival: TaskObjectiveRetrieval,
        mixture_strategy: MixtureStrategy,
        reward_config: RewardProps,
        tokenizer,
        env_service_url: str,
        agent_flow: Optional[AgentFlow] = None,
        env_worker: Optional[Any] = None, 
        **kwargs: Unpack[TaskManagerProps],
    ):
        """
        初始化任务管理器，注入所有必要的依赖项。
        """
        print(f"[TaskManager.__init__] Start initializing. Strategy: {exploration_strategy}, Env URL: {env_service_url}") # ADDED
        self._config = config
        self._tokenizer = tokenizer
        
        debug_log(self._config, "task_manager_init_start", {
            "exploration_strategy": exploration_strategy,
            "env_service_url": env_service_url
        })

        # 1. 实例化探索策略（Random 或 API-Driven）
        self._exploration_strategy = get_exploration_strategy(
            exploration_strategy, 
            exploration_strategy_args, 
            tokenizer=tokenizer, 
            config=config,
            llm_client=llm_client 
        )
        self._llm_client = llm_client
        self._old_retrival = old_retrival       # 用于任务检索和去重的存储器
        self._mixture_strategy = mixture_strategy # 数据混合策略（原始 vs 合成）
        self._reward_config = reward_config
        self._env_service_url = env_service_url
        self._num_exploration_threads = kwargs.get("num_explore_threads", 10)
        self._n = kwargs.get("n", 1)

        # 保存 Agent 执行相关的组件
        self.agent_flow = agent_flow  # 定义了 Agent 如何思考和行动的流程
        self.env_worker = env_worker  # 与环境（沙箱）交互的 Worker

        # 2. 初始化过滤器链
        # 实时过滤器：生成过程中立即执行（如基础格式检查）
        self._realtime_filters: list[TaskPostFilter] = [NaiveTaskPostFilter()]
        # 后置过滤器：生成完成后执行（如昂贵的 LLM 质量打分）
        self._post_filter: list[TaskPostFilter] = [
            LlmFilter(env_service_url, llm_client, self._num_exploration_threads, tokenizer=tokenizer, config=config)
        ]

        self._tasks: list[Task] = [] # 存储加载的种子任务
        
        debug_log(self._config, "task_manager_init_end", {
            "num_explore_threads": self._num_exploration_threads,
            "expansion_n": self._n
        })
        print(f"[TaskManager.__init__] Initialization complete. Threads: {self._num_exploration_threads}, Expansion N: {self._n}") # ADDED

    @property
    def seed_tasks(self):
        """获取当前加载的所有种子任务列表"""
        return self._tasks
    
    @property
    def seed_task_objectives(self):
        """将种子任务包装为 TaskObjective 对象，初始置信度为 1.0"""
        return [TaskObjective(task=task, confidence=1.0, reward=None) for task in self.seed_tasks]

    # --- 任务加载逻辑 ---

    def load_tasks(self, tasks: Sequence[Task]):
        """直接加载 Task 对象列表"""
        print(f"[load_tasks] Loading {len(tasks)} tasks directly.") # ADDED
        self._tasks.extend(tasks)
        assert all([x.query is None for x in self._tasks]), "种子任务的 query 必须为空（待演化）"
        logger.info(f"loaded tasks, #tasks={len(self._tasks)}")
        debug_log(self._config, "load_tasks_direct", {"count": len(tasks), "total": len(self._tasks)})
        print(f"[load_tasks] Tasks loaded. Total tasks: {len(self._tasks)}") # ADDED

    def load_tasks_from_dataset(self, dataset: RLHFDataset, *, env_type: str):
        """从 verl 的 RLHFDataset 中加载并转换为 Task"""
        print(f"[load_tasks_from_dataset] Loading from dataset for env_type: {env_type}") # ADDED
        new_tasks = adapter.convert_to_tasks(dataset, env_type=env_type, grader=self._reward_config["original_grader"])
        self._tasks.extend(new_tasks)
        assert all([x.query is None for x in self._tasks]), "种子任务的 query 必须为空"
        logger.info(f"loaded tasks from dataset, #tasks={len(self._tasks)}")
        debug_log(self._config, "load_tasks_from_dataset", {"env_type": env_type, "count": len(new_tasks)})
        print(f"[load_tasks_from_dataset] Success. Added {len(new_tasks)} tasks. Total: {len(self._tasks)}") # ADDED

    def load_tasks_from_environment(self, env: EnvClient, *, env_type: str, split: str, params: Optional[dict] = None):
        """从环境服务端拉取可用的任务 ID，并构造种子任务"""
        try:
            print(f"[load_tasks_from_environment] Requesting env profile. Type: {env_type}, Split: {split}, URL: {self._env_service_url}") # ADDED
            debug_log(self._config, "load_tasks_from_env_start", {"env_type": env_type, "split": split})
            response = env.get_env_profile(env_type, split, params)
            new_tasks = [Task(task_id=str(x), env_type=env_type, open_query=False, evaluator=self._reward_config["original_grader"]) for x in response]
            self._tasks.extend(new_tasks)
            assert all([x.query is None for x in self._tasks]), "种子任务的 query 必须为空"
            logger.info(f"loaded tasks from environment, #tasks={len(self._tasks)}")
            debug_log(self._config, "load_tasks_from_env_end", {"count": len(new_tasks)})
            print(f"[load_tasks_from_environment] Loaded {len(new_tasks)} tasks. Total tasks: {len(self._tasks)}") # ADDED
        except requests.exceptions.RequestException as e:
            logger.error(f"failed to load tasks from environment: {e}")
            print(f"[load_tasks_from_environment] FAILED. Error: {str(e)}") # ADDED
            debug_log(self._config, "load_tasks_from_env_error", {"error": str(e)})
            raise
        return len(response)

    def register_filter(self, filter: TaskPostFilter):
        """允许外部注册额外的实时过滤器"""
        print(f"[register_filter] Registering new filter: {type(filter).__name__}") # ADDED
        self._realtime_filters.append(filter)

    def _compute_tasks_hash(self, tasks: Sequence[Task]) -> str:
        """根据当前任务列表计算 MD5 哈希，用于验证断点文件是否过期"""
        task_strs = [f"{task.task_id}:{task.env_type}" for task in tasks]
        combined_str = "|".join(task_strs)
        val = hashlib.md5(combined_str.encode()).hexdigest()
        # print(f"[_compute_tasks_hash] Computed hash: {val} for {len(tasks)} tasks") # ADDED (Optional, can be verbose)
        return val

    # --- 核心任务生成流程 ---

    def generate_task(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        生成任务的总入口：根据当前策略类型（API驱动 vs 随机采样）选择不同的执行流。
        """
        strategy_type = "api_driven" if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy) else "random"
        print(f"[generate_task] Start generation. Strategy: {strategy_type}, Input Tasks: {len(tasks)}") # ADDED
        debug_log(self._config, "generate_task_entry", {"strategy": strategy_type, "num_input_tasks": len(tasks)})
        
        if strategy_type == "api_driven":
            return self._generate_task_api_driven(tasks, show_progress=show_progress, resume_file=resume_file)
        else:
            return self._generate_task_random(tasks, show_progress=show_progress, resume_file=resume_file)

    def _generate_task_random(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        Random 策略下的任务生成：
        特点：任务之间无耦合，支持高度并行的 ThreadPool 探索。
        """
        print(f"[_generate_task_random] Starting random generation. Resume file: {resume_file}") # ADDED
        if resume_file is None:
            resume_file = '.generate_task.checkpoint.json'

        current_tasks_hash = self._compute_tasks_hash(tasks)
        res = []
        processed_indices = set()
        
        debug_log(self._config, "gen_random_start", {"resume_file": resume_file})

        # 1. 尝试从断点文件恢复
        if resume_file and os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    checkpoint = json.load(f)
                    if checkpoint.get('tasks_hash') != current_tasks_hash:
                        logger.warning(f"任务哈希不匹配，正在删除过期的断点文件。")
                        print(f"[_generate_task_random] Checkpoint hash mismatch. Deleting {resume_file}") # ADDED
                        debug_log(self._config, "gen_random_checkpoint_mismatch", {})
                        os.remove(resume_file)
                    else:
                        res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in checkpoint.get('results', [])]
                        processed_indices = {int(i) for i in checkpoint.get('processed_indices', [])}
                        logger.info(f"从断点恢复: 已加载 {len(res)} 条结果")
                        print(f"[_generate_task_random] Resumed from checkpoint. Loaded {len(res)} results.") # ADDED
                        debug_log(self._config, "gen_random_resumed", {"loaded_count": len(res)})
            except Exception as e:
                logger.warning(f"断点加载失败: {e}, 将重新开始生成")
                print(f"[_generate_task_random] Failed to load checkpoint: {e}") # ADDED
                debug_log(self._config, "gen_random_checkpoint_error", {"error": str(e)})

        # 将任务池扩大 n 倍
        task_q = list(copy.copy(tasks)) * self._n
        parallel_num = max(1, min(self._num_exploration_threads, len(tasks)))
        
        print(f"[_generate_task_random] Execution Plan: Total Tasks={len(task_q)}, Parallel Workers={parallel_num}") # ADDED
        debug_log(self._config, "gen_random_execution_start", {
            "total_tasks_to_process": len(task_q),
            "parallel_num": parallel_num
        })

        # 2. 并行执行探索与总结
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as pool:
            batch_indices = list(range(0, len(task_q), parallel_num))
            for idx, i in enumerate(tqdm(batch_indices, desc="generating tasks (random)", disable=not show_progress)):
                if idx in processed_indices: continue

                # 提交线程池处理：探索 + 总结
                futures = [
                    pool.submit(self._exlore_and_summarize, task, "unknown", "unknown")
                    for task in task_q[i : i + parallel_num]
                ]
                task_objectives = sum([future.result() for future in futures], [])
                res.extend(task_objectives)
                
                # 3. 每批次后进行实时过滤并更新检索库，防止后续生成重复任务
                pre_filter_len = len(res)
                res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
                
                if len(res) < pre_filter_len:
                    print(f"[_generate_task_random] Batch {idx}: Filtered out {pre_filter_len - len(res)} tasks.") # ADDED
                    debug_log(self._config, "gen_random_realtime_filtered", {
                        "batch_index": idx,
                        "dropped": pre_filter_len - len(res)
                    })

                self._old_retrival.reset()
                for j in res:
                    self._old_retrival.add_objective(j)

                processed_indices.add(idx)
                # 4. 保存断点
                if resume_file:
                    self._save_checkpoint(resume_file, res, processed_indices, len(batch_indices), current_tasks_hash)

        print(f"[_generate_task_random] Generation finished. Total candidates: {len(res)}. Proceeding to post-filter.") # ADDED
        return self._apply_post_filter(res)

    def _generate_task_api_driven(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        print(f"[_generate_task_api_driven] Starting API-Driven generation.") # ADDED
        strategy_args = self._config.task_manager.get('exploration_strategy_args', {})
        a = strategy_args.get('a', 1)
        b = strategy_args.get('b', 1)
        
        debug_mode = self._config.get("debug_log", False)
        if debug_mode:
            logger.warning("Debug mode enabled: forcing single thread and single task execution for API Driven generation.")
            print("[_generate_task_api_driven] DEBUG MODE ENABLED.") # ADDED
            random.seed(42)

        if resume_file is None:
            resume_file = '.generate_task_api'
        
        intra_ckpt_path = f"{resume_file}.intra.json"
        cross_ckpt_path = f"{resume_file}.extra.json"
        current_tasks_hash = self._compute_tasks_hash(tasks)
        
        debug_log(self._config, "gen_api_driven_start", {
            "a": a, "b": b,
            "intra_ckpt": intra_ckpt_path,
            "cross_ckpt": cross_ckpt_path
        })

        # 获取基础数据
        api_knowledge = getattr(self._exploration_strategy, 'api_knowledge', {})
        active_apps_set = getattr(self._exploration_strategy, 'active_apps', set(api_knowledge.keys()))
        
        # --- (ADDED) Debug Logs ---
        print("\n" + "="*20 + " [DEBUG LOGS] TaskManager " + "="*20)
        print(f"api_knowledge (keys): {list(api_knowledge.keys())}")
        print("api_knowledge content:", api_knowledge)
        print("active_apps_set:", active_apps_set)
        print("="*60 + "\n", flush=True)
        # --------------------------

        # =================================================================
        # 定义并行执行的原子函数 (Wrapper Functions)
        # =================================================================
        
        def process_intra_task(idx: int, api_info: dict, seed_task: Task) -> List[TaskObjective]:
            """单线程处理：单域任务生成 -> 探索 -> 总结"""

            print(f"[process_intra_task] Start idx={idx}, App={api_info.get('app_name')}, API={api_info.get('api_name')}") # ADDED
            debug_log(self._config, "tm_process_intra_start", {
                "idx": idx,
                "app_name": api_info.get("app_name"),
                "api_name": api_info.get("api_name"),
                "seed_task_id": seed_task.task_id
            })

            try:
                # 1. 生成任务描述
                # 注意：深拷贝种子任务以避免副作用
                current_task = copy.deepcopy(seed_task)

                # --- 新增代码开始 ---
                # 确保 metadata 存在
                if current_task.metadata is None:
                    current_task.metadata = {}
                
                # 计算并注入线程索引 (使用取模确保索引在 0 到 num_threads-1 之间)
                # 这样不同的并发任务会分配到不同的模拟环境/端口
                thread_idx = idx % self._num_exploration_threads
                current_task.metadata['thread_index'] = thread_idx
                # --- 新增代码结束 ---

                current_task = self._exploration_strategy.generate_intra_task(
                    app_name=api_info["app_name"], 
                    target_api_name=api_info["api_name"],
                    task=current_task
                )
                if not current_task:
                    print(f"[process_intra_task] Skipped: Generation returned None for idx={idx}") # ADDED
                    debug_log(self._config, "tm_process_intra_skipped", {"idx": idx, "reason": "generation_returned_none"})
                    return []
                
                # [FIX]: 生成唯一 data_id 并传递给 explore，避免日志覆盖
                data_id = f"gen_intra_{idx}"
                current_task.metadata["data_id"] = data_id
                
                debug_log(self._config, "tm_process_intra_exploring", {"idx": idx, "data_id": data_id})

                # 2. 执行探索 (耗时操作) - 传入正确的 ID
                trajectories = self._exploration_strategy.explore(current_task, data_id, data_id)
                
                debug_log(self._config, "tm_process_intra_explored", {
                    "idx": idx, 
                    "trajectory_count": len(trajectories),
                    "steps": len(trajectories[0].steps) if trajectories else 0
                })

                # 3. 总结结果 (Strategy 内部已加锁处理内存保存)
                results = []
                if trajectories and trajectories[0].steps:
                    results = self._exploration_strategy.summarize(current_task, trajectories[0])
                
                print(f"[process_intra_task] Completed idx={idx}. Generated {len(results)} objectives.") # ADDED
                debug_log(self._config, "tm_process_intra_completed", {
                    "idx": idx,
                    "data_id": data_id,
                    "generated_objectives_count": len(results)
                })

                return results if results else []
            except Exception as e:
                logger.error(f"[Intra-Task Error] Index {idx}: {e}")
                print(f"[process_intra_task] Error in idx={idx}: {e}") # ADDED
                debug_log(self._config, "tm_process_intra_error", {
                    "idx": idx,
                    "error": str(e)
                })
                return []

        def process_cross_task(idx: int, app_list: List[str], seed_task: Task) -> List[TaskObjective]:
            """单线程处理：跨域任务生成 -> 探索 -> 总结"""

            print(f"[process_cross_task] Start idx={idx}, Candidate Apps Count={len(app_list)}") # ADDED
            debug_log(self._config, "tm_process_cross_start", {
                "idx": idx,
                "candidate_apps_count": len(app_list),
                "seed_task_id": seed_task.task_id
            })

            try:
                current_task = copy.deepcopy(seed_task)

                # --- 新增代码开始 ---
                # 确保 metadata 存在
                if current_task.metadata is None:
                    current_task.metadata = {}
                
                # 计算并注入线程索引 (使用取模确保索引在 0 到 num_threads-1 之间)
                # 这样不同的并发任务会分配到不同的模拟环境/端口
                thread_idx = idx % self._num_exploration_threads
                current_task.metadata['thread_index'] = thread_idx
                # --- 新增代码结束 ---

                current_task = self._exploration_strategy.generate_cross_task(app_list=app_list, task=current_task)
                if not current_task:
                    print(f"[process_cross_task] Skipped: Generation returned None for idx={idx}") # ADDED
                    debug_log(self._config, "tm_process_cross_skipped", {"idx": idx, "reason": "generation_returned_none"})
                    return []
                
                # [FIX]: 生成唯一 data_id 并传递给 explore，避免日志覆盖
                data_id = f"gen_cross_{idx}"
                current_task.metadata["data_id"] = data_id
                
                debug_log(self._config, "tm_process_cross_exploring", {"idx": idx, "data_id": data_id})

                # 2. 执行探索 - 传入正确的 ID
                trajectories = self._exploration_strategy.explore(current_task, data_id, data_id)
                
                debug_log(self._config, "tm_process_cross_explored", {
                    "idx": idx, 
                    "trajectory_count": len(trajectories),
                    "steps": len(trajectories[0].steps) if trajectories else 0
                })

                results = []
                if trajectories and trajectories[0].steps:
                    results = self._exploration_strategy.summarize(current_task, trajectories[0])

                print(f"[process_cross_task] Completed idx={idx}. Generated {len(results)} objectives.") # ADDED
                debug_log(self._config, "tm_process_cross_completed", {
                    "idx": idx,
                    "data_id": data_id,
                    "generated_objectives_count": len(results)
                })

                return results if results else []
            except Exception as e:
                logger.error(f"[Cross-Task Error] Index {idx}: {e}")
                print(f"[process_cross_task] Error in idx={idx}: {e}") # ADDED
                debug_log(self._config, "tm_process_cross_error", {"idx": idx, "error": str(e)})
                return []

        # =================================================================
        # 阶段 1: 单域探索 (Intra-Domain) - 并行化
        # =================================================================
        active_apis = []
        for app_name in sorted(list(active_apps_set)):
            if app_name in api_knowledge:
                apis = api_knowledge[app_name].get("apis", {})
                for api_name in sorted(apis.keys()):
                    active_apis.append({"app_name": app_name, "api_name": api_name})

        if debug_mode:
            logger.info(f"[Debug] Truncating Intra-Domain API list (Original: {len(active_apis)}) to 1.")
            print(f"[_generate_task_api_driven] Intra-Domain truncated to 1. Original count: {len(active_apis)}") # ADDED
            active_apis = active_apis[:1]

        intra_task_pool = list(copy.copy(tasks)) * a
        # 重新构造任务池：每个待测 API 分配一个 base task
        if len(intra_task_pool) < len(active_apis):
             intra_task_pool = (intra_task_pool * (len(active_apis) // len(intra_task_pool) + 1))[:len(active_apis)]
        
        intra_res = []
        intra_processed_idx = set()
        
        # 尝试恢复断点
        if os.path.exists(intra_ckpt_path):
            try:
                with open(intra_ckpt_path, 'r') as f:
                    cp = json.load(f)
                    if cp.get('tasks_hash') == current_tasks_hash:
                        intra_res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in cp.get('results', [])]
                        intra_processed_idx = {int(i) for i in cp.get('processed_indices', [])}
                        logger.info(f"Intra-Domain resumed: {len(intra_res)} tasks loaded.")
                        print(f"[_generate_task_api_driven] Intra resumed. Loaded {len(intra_res)} tasks.") # ADDED
                        debug_log(self._config, "gen_api_intra_resumed", {"loaded_count": len(intra_res)})
            except Exception as e:
                logger.warning(f"Failed to load intra checkpoint: {e}")
                print(f"[_generate_task_api_driven] Failed to load intra checkpoint: {e}") # ADDED

        # 使用线程池执行
        total_intra = min(len(active_apis), len(intra_task_pool))
        print(f"[_generate_task_api_driven] Stage 1 Plan: Intra-Domain Tasks: {total_intra}") # ADDED
        if len(intra_processed_idx) < total_intra:
            parallel_num = max(1, min(self._num_exploration_threads, total_intra))

            if debug_mode: parallel_num = 1

            remaining_indices = [i for i in range(total_intra) if i not in intra_processed_idx]
            batch_size = parallel_num * 2 
            batches = [remaining_indices[i:i + batch_size] for i in range(0, len(remaining_indices), batch_size)]

            debug_log(self._config, "gen_api_intra_exec_start", {
                "total_intra": total_intra, 
                "remaining": len(remaining_indices),
                "parallel_num": parallel_num
            })

            pbar = tqdm(total=total_intra, desc="Stage 1: Intra-Domain (Parallel)", disable=not show_progress)
            pbar.update(len(intra_processed_idx))

            with ThreadPoolExecutor(max_workers=parallel_num) as pool:
                for batch_idx, batch_idxs in enumerate(batches):
                    future_to_idx = {
                        pool.submit(process_intra_task, idx, active_apis[idx], intra_task_pool[idx]): idx 
                        for idx in batch_idxs
                    }

                    batch_results_count = 0
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            objs = future.result()
                            intra_res.extend(objs)
                            batch_results_count += len(objs)
                        except Exception as e:
                            logger.error(f"Unhandled exception in future for task {idx}: {e}")
                            print(f"[_generate_task_api_driven] Future exception for idx={idx}: {e}") # ADDED
                        finally:
                            intra_processed_idx.add(idx)
                            pbar.update(1)
                    
                    intra_res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, intra_res)
                    self._save_checkpoint(intra_ckpt_path, intra_res, intra_processed_idx, total_intra, current_tasks_hash)
                    
                    print(f"[_generate_task_api_driven] Intra Batch {batch_idx} done. Saved checkpoint.") # ADDED
                    debug_log(self._config, "gen_api_intra_batch_done", {
                        "batch_idx": batch_idx, 
                        "results_in_batch": batch_results_count,
                        "total_results": len(intra_res)
                    })
            
            pbar.close()

        # =================================================================
        # 阶段 2: 跨域合成 (Cross-Domain) - 并行化
        # =================================================================
        cross_res = []
        active_apps_list = list(active_apps_set)
        
        cross_task_pool = list(copy.copy(tasks)) * b

        if debug_mode:
            logger.info(f"[Debug] Truncating Cross-Domain task pool (Original: {len(cross_task_pool)}) to 1.")
            print(f"[_generate_task_api_driven] Cross-Domain truncated to 1. Original count: {len(cross_task_pool)}") # ADDED
            cross_task_pool = cross_task_pool[:1]

        cross_processed_idx = set()
        
        if os.path.exists(cross_ckpt_path):
            try:
                with open(cross_ckpt_path, 'r') as f:
                    cp = json.load(f)
                    cross_res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in cp.get('results', [])]
                    cross_processed_idx = {int(i) for i in cp.get('processed_indices', [])}
                    logger.info(f"Cross-Domain resumed: {len(cross_res)} tasks loaded.")
                    print(f"[_generate_task_api_driven] Cross resumed. Loaded {len(cross_res)} tasks.") # ADDED
                    debug_log(self._config, "gen_api_cross_resumed", {"loaded_count": len(cross_res)})
            except Exception: pass

        print(f"[_generate_task_api_driven] Stage 2 Plan: Cross-Domain Tasks: {len(cross_task_pool)}") # ADDED
        if len(cross_processed_idx) < len(cross_task_pool):
            parallel_num = max(1, min(self._num_exploration_threads, len(cross_task_pool)))
            if debug_mode: parallel_num = 1
            remaining_indices = [i for i in range(len(cross_task_pool)) if i not in cross_processed_idx]
            batches = [remaining_indices[i:i + parallel_num * 2] for i in range(0, len(remaining_indices), parallel_num * 2)]

            debug_log(self._config, "gen_api_cross_exec_start", {
                "total_cross": len(cross_task_pool), 
                "remaining": len(remaining_indices),
                "parallel_num": parallel_num
            })

            pbar = tqdm(total=len(cross_task_pool), desc="Stage 2: Cross-Domain (Parallel)", disable=not show_progress)
            pbar.update(len(cross_processed_idx))

            with ThreadPoolExecutor(max_workers=parallel_num) as pool:
                for batch_idx, batch_idxs in enumerate(batches):
                    future_to_idx = {
                        pool.submit(process_cross_task, idx, active_apps_list, cross_task_pool[idx]): idx 
                        for idx in batch_idxs
                    }

                    batch_results_count = 0
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            objs = future.result()
                            cross_res.extend(objs)
                            batch_results_count += len(objs)
                        except Exception as e:
                            logger.error(f"Unhandled exception in cross task {idx}: {e}")
                            print(f"[_generate_task_api_driven] Future exception for idx={idx}: {e}") # ADDED
                        finally:
                            cross_processed_idx.add(idx)
                            pbar.update(1)
                    
                    current_batch_all = intra_res + cross_res
                    filtered_all = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, current_batch_all)
                    # 只保存 extra 的部分
                    cross_res_to_save = filtered_all[len(intra_res):] 
                    self._save_checkpoint(cross_ckpt_path, cross_res_to_save, cross_processed_idx, len(cross_task_pool), current_tasks_hash)
                    
                    print(f"[_generate_task_api_driven] Cross Batch {batch_idx} done. Saved checkpoint.") # ADDED
                    debug_log(self._config, "gen_api_cross_batch_done", {
                        "batch_idx": batch_idx,
                        "results_in_batch": batch_results_count,
                        "total_cross_results": len(cross_res_to_save)
                    })
            
            pbar.close()
        
        total_results = intra_res + cross_res
        print(f"[_generate_task_api_driven] Generation Finished. Intra: {len(intra_res)}, Cross: {len(cross_res)}. Applying Post-filter.") # ADDED
        debug_log(self._config, "gen_api_driven_finish", {
            "total_intra": len(intra_res),
            "total_cross": len(cross_res),
            "total": len(total_results)
        })
        return self._apply_post_filter(total_results)

    def _save_checkpoint(self, path, results, processed_indices, total, hash_val):
        """保存任务生成的断点信息到 JSON 文件"""
        try:
            print(f"[_save_checkpoint] Saving checkpoint to {path}. Total processed: {len(processed_indices)}") # ADDED
            checkpoint_data = {
                'results': [obj.dict() for obj in results],
                'processed_indices': list(processed_indices),
                'total_batches': total,
                'tasks_hash': hash_val,
                'timestamp': time.time()
            }
            with open(path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            logger.warning(f"保存断点失败: {e}")
            print(f"[_save_checkpoint] ERROR: Failed to save checkpoint: {e}") # ADDED

    def _apply_post_filter(self, res: List[TaskObjective]) -> List[TaskObjective]:
        """应用耗时较长的后置过滤器（如 LLM 质量核验），并打乱数据顺序"""
        print(f"[_apply_post_filter] Start filtering {len(res)} results.") # ADDED
        debug_log(self._config, "post_filter_start", {"input_count": len(res)})
        
        # 先应用实时过滤器（确保最终一致性）
        res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
        
        logger.info("正在对生成的任务进行后置过滤（Post-Filter）...")
        cnt_before = len(res)
        res = functools.reduce(lambda x, f: f.filter(x), self._post_filter, res)
        logger.info(f"后置过滤完成: 过滤前={cnt_before}, 过滤后={len(res)}")
        
        debug_log(self._config, "post_filter_end", {
            "input_count": cnt_before,
            "output_count": len(res),
            "filtered_out": cnt_before - len(res)
        })
        
        random.shuffle(res)
        print(f"[_apply_post_filter] Done. Filtered from {cnt_before} to {len(res)}. Shuffled.") # ADDED
        return res

    # --- 执行与总结的助手方法 ---

    def _exlore_and_summarize(self, task: Task, data_id: str, rollout_id: str) -> list[TaskObjective]:
        """随机策略下：执行探索并对轨迹进行总结"""
        try:
            print(f"[_exlore_and_summarize] Processing task: {task.task_id}") # ADDED
            trajectories = self._step_explore(task, data_id, rollout_id)
            task_objectives = sum([self._step_summarize(task, trajectory) for trajectory in trajectories], [])
            
            # 安全检查
            valid_objs = []
            for x in task_objectives:
                if x.task.open_query:
                    valid_objs.append(x)
                else:
                    print(f"[_exlore_and_summarize] Invalid task objective (open_query=False): {x.task.task_id}") # ADDED
                    debug_log(self._config, "random_explore_invalid_task", {"reason": "open_query_false", "task_id": task.task_id})
            
            return valid_objs
        except Exception as e:
            logger.error(f"Error in random explore: {e}")
            print(f"[_exlore_and_summarize] Error: {e}") # ADDED
            debug_log(self._config, "random_explore_error", {"error": str(e), "task_id": task.task_id})
            return []

    def _step_explore(self, task: Task, data_id: str, rollout_id: str) -> list[Trajectory]:
        """调用策略的 explore 方法（Random 专用）"""
        print(f"[_step_explore] Calling strategy.explore for task {task.task_id}") # ADDED
        return self._exploration_strategy.explore(task, data_id, rollout_id)

    def _step_summarize(self, task: Task, trajectory: Trajectory) -> list[TaskObjective]:
        """调用策略的 summarize 方法（Random 专用）"""
        print(f"[_step_summarize] Calling strategy.summarize for task {task.task_id}") # ADDED
        return self._exploration_strategy.summarize(task, trajectory)

# ================= 数据集类 =================

class FullDataset(Dataset):
    """
    静态数据集：一次性生成/加载所有合成任务，并与原始种子任务混合。
    支持缓存到本地文件。
    """
    def __init__(self, manager: TaskManager, mixture_strategy: MixtureStrategy, reward_config: RewardProps, cache_path: Optional[str] = None, *, tokenizer, config, processor):
        debug_log(config, "FullDataset_init", {"cache_path": cache_path})
        print(f"[FullDataset.__init__] Initializing. Cache path: {cache_path}") # ADDED
        self._manager = manager
        self._tasks = self._manager.seed_task_objectives
        self._mixture_strategy = mixture_strategy
        self._reward_config = reward_config
        self._cache_path = cache_path
        
        self._tokenizer = tokenizer
        self._config = config
        self._processor = processor
        
        self._objectives = []
        self._synthetic_objectives = []

        # 如果策略需要合成数据，则加载缓存或生成新任务
        if self._mixture_strategy.need_synthetic:
            logger.info("正在准备合成任务数据...")
            print("[FullDataset.__init__] Need synthetic data.") # ADDED
            if self._cache_path is not None and os.path.exists(self._cache_path):
                debug_log(config, "FullDataset_load_cache", {})
                self.load_from_file()
            else:
                debug_log(config, "FullDataset_generate_new", {})
                print("[FullDataset.__init__] Cache not found or not set. Generating new tasks.") # ADDED
                self.reload_new_task()
                if self._cache_path is not None: self.save_to_file()
        
        self._rebuild_dataset()

    def _rebuild_dataset(self):
        """混合原始数据和合成数据，并转换为训练格式"""
        print("[_rebuild_dataset] Mixing synthetic and original tasks.") # ADDED
        self._objectives = self._mixture_strategy.mix_data(self._synthetic_objectives, self._tasks)
        
        # --- 添加保护逻辑 ---
        if len(self._objectives) == 0:
            logger.error("【严重错误】没有可用的训练数据！可能是环境服务挂了，或者 Debug 模式下生成的任务全部被过滤了。")
            print("[_rebuild_dataset] FATAL: Dataset is empty!") # ADDED
            raise ValueError("Dataset is empty. Please check env_service status or disable debug_log.")
        # -------------------

        debug_log(self._config, "FullDataset_rebuild", {
            "synthetic_count": len(self._synthetic_objectives),
            "seed_count": len(self._tasks),
            "mixed_total": len(self._objectives)
        })
        print(f"[_rebuild_dataset] Total objectives: {len(self._objectives)}. Converting to RL dataset.") # ADDED
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)

    def save_to_file(self):
        """将生成的合成任务目标保存到 JSONL 文件"""
        print(f"[save_to_file] Saving to {self._cache_path}") # ADDED
        assert self._cache_path is not None
        with open(self._cache_path, "w") as f:
            f.writelines([ob.json() + "\n" for ob in self._synthetic_objectives])
        debug_log(self._config, "FullDataset_saved", {"path": self._cache_path, "count": len(self._synthetic_objectives)})

    def load_from_file(self):
        """从缓存文件加载合成任务，并修正评分器配置"""
        print(f"[load_from_file] Loading from {self._cache_path}") # ADDED
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "r") as f:
                self._synthetic_objectives = []
                for line in f:
                    if not line.strip(): continue
                    t = json.loads(line)
                    tmp = TaskObjective.parse_obj(t)
                    self._synthetic_objectives.append(tmp)
            # 为合成数据打上对应的评分器标签
            for item in self._synthetic_objectives:
                item.task.evaluator = self._reward_config["synthetic_grader"]
            debug_log(self._config, "FullDataset_loaded_count", {"count": len(self._synthetic_objectives)})
            print(f"[load_from_file] Loaded {len(self._synthetic_objectives)} synthetic objectives.") # ADDED
        else:
            print(f"[load_from_file] Error: File not found {self._cache_path}") # ADDED
            raise FileNotFoundError(f"找不到缓存文件 {self._cache_path}")

    def reload_new_task(self):
        """调用 TaskManager 重新触发演化生成流程"""
        print("[reload_new_task] Triggering task generation.") # ADDED
        self._synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        for item in self._synthetic_objectives:
            item.task.evaluator = self._reward_config["synthetic_grader"]
        print(f"[reload_new_task] Generated {len(self._synthetic_objectives)} new tasks.") # ADDED

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset) if self._dataset else 0


class AutoReloadDataset(IterableDataset):
    """
    可迭代数据集：在训练过程中，当数据耗尽时，动态触发 TaskManager 生成新任务（On-the-fly）。
    """
    def __init__(self, manager: TaskManager, tasks: Iterable[Task], bs: int, mix_origins: bool = False, *, tokenizer, config, processor):
        print(f"[AutoReloadDataset.__init__] Initializing. BS={bs}") # ADDED
        self._manager = manager
        self._tasks = tasks
        self._bs = bs
        self._tokenizer = tokenizer
        self._config = config
        self._processor = processor
        self._dataset = OnflyRlDataset(release_used_dataset=True)
        debug_log(config, "AutoReloadDataset_init", {"batch_size": bs})

    def reload(self):
        """动态拉取一批种子任务并演化出新任务，加入数据集队列"""
        print("[AutoReloadDataset.reload] Reloading data.") # ADDED
        debug_log(self._config, "AutoReloadDataset_reload_start", {})
        delta = []
        for task in self._tasks:
            delta.append(task)
            if len(delta) == self._bs: break

        if not delta: 
            print("[AutoReloadDataset.reload] No seed tasks available.") # ADDED
            debug_log(self._config, "AutoReloadDataset_reload_empty_source", {})
            return 0

        # 调用演化逻辑
        ls = self._manager.generate_task(delta)
        # 确保生成了足够的数据
        while len(ls) < self._bs * self._manager._n:
            logger.debug("合成数据量不足，正在重试生成...")
            print(f"[AutoReloadDataset.reload] Not enough data ({len(ls)}). Retrying.") # ADDED
            debug_log(self._config, "AutoReloadDataset_retry", {"current": len(ls), "target": self._bs * self._manager._n})
            ls = self._manager.generate_task(delta)

        self._dataset.append_dataset(to_rl_dataset(ls, self._tokenizer, self._config, self._processor))
        print(f"[AutoReloadDataset.reload] Appended {len(ls)} new examples.") # ADDED
        debug_log(self._config, "AutoReloadDataset_reload_finish", {"new_data_count": len(ls)})
        return self._dataset.num_rest_data

    def __iter__(self):
        return self

    def __next__(self):
        """获取下一条数据，如果为空则尝试 reload"""
        if self._dataset.num_rest_data == 0:
            if self.reload() == 0: raise StopIteration
        return next(self._dataset)