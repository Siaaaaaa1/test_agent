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
from agentevolver.module.task_manager.filters.api_llm_pre_filter import LlmQualityPreFilter

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

def get_exploration_strategy(name: str, strategy_args, *, tokenizer, config, llm_client, env_profile) -> TaskExploreStrategy:
    logger.info(f"loading exploration strategy {name}")
    if name == "random":
        return LlmRandomSamplingExploreStrategy(
            tokenizer=tokenizer, 
            config=config,
            env_profile=env_profile,
            **strategy_args
            )
    elif name == "api_driven":
        return ApiDrivenExploreStrategy(
            tokenizer=tokenizer, 
            config=config, 
            llm_client=llm_client,
            env_profile=env_profile,
            **strategy_args
        )
    else:
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
        self._config = config
        self._tokenizer = tokenizer
        
        # 1. 实例化探索策略（Random 或 API-Driven）
        self._exploration_strategy = get_exploration_strategy(
            exploration_strategy, 
            exploration_strategy_args, 
            tokenizer=tokenizer, 
            config=config,
            llm_client=llm_client,
            env_profile=env_profile
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
        self.api_llm_pre_filter = [
            LlmQualityPreFilter(llm_client, num_threads=self._num_exploration_threads)
        ]

        self._tasks: list[Task] = [] # 存储加载的种子任务
        

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
        self._tasks.extend(tasks)
        assert all([x.query is None for x in self._tasks]), "种子任务的 query 必须为空（待演化）"
        logger.info(f"loaded tasks, #tasks={len(self._tasks)}")

    def load_tasks_from_dataset(self, dataset: RLHFDataset, *, env_type: str):
        """从 verl 的 RLHFDataset 中加载并转换为 Task"""
        new_tasks = adapter.convert_to_tasks(dataset, env_type=env_type, grader=self._reward_config["original_grader"])
        self._tasks.extend(new_tasks)
        assert all([x.query is None for x in self._tasks]), "种子任务的 query 必须为空"
        logger.info(f"loaded tasks from dataset, #tasks={len(self._tasks)}")

    def load_tasks_from_environment(self, env: EnvClient, *, env_type: str, split: str, params: Optional[dict] = None):
        """从环境服务端拉取可用的任务 ID，并构造种子任务"""
        try:
            response = env.get_env_profile(env_type, split, params)
            new_tasks = [Task(task_id=str(x), env_type=env_type, open_query=False, evaluator=self._reward_config["original_grader"]) for x in response]
            self._tasks.extend(new_tasks)
            assert all([x.query is None for x in self._tasks]), "种子任务的 query 必须为空"
            logger.info(f"loaded tasks from environment, #tasks={len(self._tasks)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"failed to load tasks from environment: {e}")
            raise
        return len(response)

    def register_filter(self, filter: TaskPostFilter):
        """允许外部注册额外的实时过滤器"""
        self._realtime_filters.append(filter)

    def _compute_tasks_hash(self, tasks: Sequence[Task]) -> str:
        """根据当前任务列表计算 MD5 哈希，用于验证断点文件是否过期"""
        task_strs = [f"{task.task_id}:{task.env_type}" for task in tasks]
        combined_str = "|".join(task_strs)
        val = hashlib.md5(combined_str.encode()).hexdigest()
        return val

    # --- 核心任务生成流程 ---

    def generate_task(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        生成任务的总入口：根据当前策略类型（API驱动 vs 随机采样）选择不同的执行流。
        """
        strategy_type = "api_driven" if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy) else "random"
        
        if strategy_type == "api_driven":
            return self._generate_task_api_driven(tasks, show_progress=show_progress, resume_file=resume_file)
        else:
            return self._generate_task_random(tasks, show_progress=show_progress, resume_file=resume_file)

    def _generate_task_random(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        Random 策略下的任务生成：
        特点：任务之间无耦合，支持高度并行的 ThreadPool 探索。
        """
        if resume_file is None:
            resume_file = '.generate_task.checkpoint.json'

        current_tasks_hash = self._compute_tasks_hash(tasks)
        res = []
        processed_indices = set()
        
        # 1. 尝试从断点文件恢复
        if resume_file and os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    checkpoint = json.load(f)
                    if checkpoint.get('tasks_hash') != current_tasks_hash:
                        logger.warning(f"任务哈希不匹配，正在删除过期的断点文件。")
                        os.remove(resume_file)
                    else:
                        res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in checkpoint.get('results', [])]
                        processed_indices = {int(i) for i in checkpoint.get('processed_indices', [])}
                        logger.info(f"从断点恢复: 已加载 {len(res)} 条结果")
            except Exception as e:
                logger.warning(f"断点加载失败: {e}, 将重新开始生成")

        # 将任务池扩大 n 倍
        task_q = list(copy.copy(tasks)) * self._n
        parallel_num = max(1, min(self._num_exploration_threads, len(tasks)))
        
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
                res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
                
                self._old_retrival.reset()
                for j in res:
                    self._old_retrival.add_objective(j)

                processed_indices.add(idx)
                # 4. 保存断点
                if resume_file:
                    self._save_checkpoint(resume_file, res, processed_indices, len(batch_indices), current_tasks_hash)

        return self._apply_post_filter(res)

    def _generate_task_api_driven(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        重构后的 API-Driven 生成流程：
        1. 阶段一：批量生成任务描述 (Gen) -> 保存
        2. 阶段二：批量预过滤 (Filter) -> 保存
        3. 阶段三：批量探索与总结 (Explore) -> 保存
        """
        strategy_args = self._config.task_manager.get('exploration_strategy_args', {})
        a = strategy_args.get('a', 1)
        b = strategy_args.get('b', 1)
        # debug_mode = self._config.get("debug_log", False)
        debug_mode = False
        
        logger.info(f"[API-Driven] Strategy Args: a={a}, b={b}, debug_log={debug_mode}")
        if debug_mode:
            logger.warning("Debug mode enabled: forcing single thread.")

        if resume_file is None:
            resume_file = '.generate_task_api'
        
        current_tasks_hash = self._compute_tasks_hash(tasks)
        
        # 内部 helper：加载中间状态的 Task 列表
        def load_intermediate_tasks(path: str) -> Optional[List[Task]]:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        if data.get('tasks_hash') == current_tasks_hash:
                            logger.info(f"Loaded intermediate tasks from {path}: {len(data['tasks'])} items.")
                            return [Task.parse_obj(t) for t in data['tasks']]
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {path}: {e}")
            return None

        # 内部 helper：保存中间状态的 Task 列表
        def save_intermediate_tasks(path: str, task_list: List[Task]):
            try:
                with open(path, 'w') as f:
                    json.dump({
                        'tasks': [t.dict() for t in task_list],
                        'tasks_hash': current_tasks_hash,
                        'timestamp': time.time()
                    }, f, indent=2)
                logger.info(f"Saved intermediate tasks to {path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint {path}: {e}")

        # =================================================================
        # WORKER FUNCTIONS (只负责具体的单步逻辑)
        # =================================================================

        def worker_generate_intra(idx: int, api_dict: dict, seed_task: Task) -> Optional[Task]:
            """Worker: 仅生成 Intra 任务描述"""
            try:
                current_task = copy.deepcopy(seed_task)
                if current_task.metadata is None: current_task.metadata = {}
                current_task.metadata['thread_index'] = idx % self._num_exploration_threads
                
                # 生成 Query
                current_task = self._exploration_strategy.generate_intra_task(api_dict, task=current_task)
                if not current_task: return None
                
                # 设置 Data ID
                data_id = f"gen_intra_{idx}"
                current_task.metadata["data_id"] = data_id
                
                # Log Input Trace
                debug_log(self._config, "evolution_trace", {
                    "type": "intra_input",
                    "data_id": data_id,
                    "app": current_task.metadata.get("target_app"),
                    "api": current_task.metadata.get("target_api"),
                    "generated_task_query": current_task.query,
                    "task_metadata": current_task.metadata
                })
                return current_task
            except Exception as e:
                logger.error(f"[Intra-Gen] Error idx {idx}: {e}")
                return None

        def worker_explore_intra(task: Task) -> List[TaskObjective]:
            """Worker: 执行 Intra 探索和总结"""
            try:
                data_id = task.metadata.get("data_id", f"unknown_{random.randint(0,1000)}")
                logger.info(f"[Intra-Explore] Exploring {data_id}...")
                
                trajectories = self._exploration_strategy.explore(task, data_id, data_id)
                
                # Log Output Trace
                simple_trajs = []
                for t in trajectories:
                    steps_data = [s if isinstance(s, dict) else (s.dict() if hasattr(s, 'dict') else str(s)) for s in t.steps]
                    simple_trajs.append({"steps_count": len(t.steps), "steps": steps_data})
                
                debug_log(self._config, "evolution_trace", {
                    "type": "intra_output",
                    "data_id": data_id,
                    "trajectories": simple_trajs
                })

                results = []
                if trajectories and trajectories[0].steps:
                    results = self._exploration_strategy.summarize(task, trajectories[0])
                return results if results else []
            except Exception as e:
                logger.error(f"[Intra-Explore] Error: {e}", exc_info=True)
                return []

        def worker_generate_cross(idx: int, api_dict1: dict, api_dict2: dict, seed_task: Task) -> Optional[Task]:
            """Worker: 仅生成 Cross 任务描述"""
            try:
                current_task = copy.deepcopy(seed_task)
                if current_task.metadata is None: current_task.metadata = {}
                current_task.metadata['thread_index'] = idx % self._num_exploration_threads
                
                current_task = self._exploration_strategy.generate_cross_task(api_dict1=api_dict1, api_dict2=api_dict2, task=current_task)
                if not current_task: return None

                data_id = f"gen_cross_{idx}"
                current_task.metadata["data_id"] = data_id

                debug_log(self._config, "evolution_trace", {
                    "type": "cross_input",
                    "data_id": data_id,
                    "target_apps": f"{api_dict1.get('app_name')} & {api_dict2.get('app_name')}",
                    "generated_task_query": current_task.query,
                    "task_metadata": current_task.metadata
                })
                return current_task
            except Exception as e:
                logger.error(f"[Cross-Gen] Error idx {idx}: {e}")
                return None

        def worker_explore_cross(task: Task) -> List[TaskObjective]:
            """Worker: 执行 Cross 探索和总结"""
            try:
                data_id = task.metadata.get("data_id", f"unknown_{random.randint(0,1000)}")
                logger.info(f"[Cross-Explore] Exploring {data_id}...")
                
                trajectories = self._exploration_strategy.explore(task, data_id, data_id)
                
                simple_trajs = []
                for t in trajectories:
                    steps_data = [s if isinstance(s, dict) else (s.dict() if hasattr(s, 'dict') else str(s)) for s in t.steps]
                    simple_trajs.append({"steps_count": len(t.steps), "steps": steps_data})
                
                debug_log(self._config, "evolution_trace", {
                    "type": "cross_output",
                    "data_id": data_id,
                    "trajectories": simple_trajs
                })

                results = []
                if trajectories and trajectories[0].steps:
                    results = self._exploration_strategy.summarize(task, trajectories[0])
                return results if results else []
            except Exception as e:
                logger.error(f"[Cross-Explore] Error: {e}", exc_info=True)
                return []

        # 获取基础数据
        api_knowledge = getattr(self._exploration_strategy, 'api_knowledge', {})
        active_apps_set = getattr(self._exploration_strategy, 'active_apps', set(api_knowledge.keys()))

        # =================================================================
        # PART 1: INTRA-DOMAIN (生成 -> 过滤 -> 探索)
        # =================================================================
        
        # 1.1 准备 API 组合
        api_list = []
        for app_name in sorted(active_apps_set):
            if app_name not in api_knowledge: continue
            apis = api_knowledge[app_name].get("apis", [])
            if not apis: continue
            sample_count = min(len(apis), 5)
            for _ in range(len(apis)):
                selected_apis = random.sample(list(apis.values()), sample_count)
                this_turn_apis = [api["call_name"] for api in selected_apis]
                api_list.append({"app_name":app_name, "apis_name_list":this_turn_apis})
        
        if debug_mode: api_list = api_list[:1]
        random.shuffle(api_list)
        intra_task_pool = list(copy.copy(tasks)) * a
        total_intra = min(len(api_list), len(intra_task_pool))
        
        # 定义文件路径
        intra_gen_path = f"{resume_file}.intra.generated.json"
        intra_filtered_path = f"{resume_file}.intra.filtered.json"
        intra_final_path = f"{resume_file}.intra.json" # Final objectives

        # --- Step 1: Intra Generation ---
        generated_intra_tasks = load_intermediate_tasks(intra_gen_path)
        if generated_intra_tasks is None:
            generated_intra_tasks = []
            logger.info(f"[Intra-Gen] Start generating {total_intra} task descriptions...")
            
            with ThreadPoolExecutor(max_workers=1 if debug_mode else self._num_exploration_threads) as pool:
                futures = []
                for idx in range(total_intra):
                    futures.append(pool.submit(worker_generate_intra, idx, api_list[idx], intra_task_pool[idx]))
                
                for f in tqdm(as_completed(futures), total=len(futures), desc="Intra Generation", disable=not show_progress):
                    res = f.result()
                    if res: generated_intra_tasks.append(res)
            
            save_intermediate_tasks(intra_gen_path, generated_intra_tasks)
        else:
            logger.info("[Intra-Gen] Skipped (Loaded from Checkpoint)")

        # --- Step 2: Intra Filtering ---
        filtered_intra_tasks = load_intermediate_tasks(intra_filtered_path)
        if filtered_intra_tasks is None:
            logger.info(f"[Intra-Filter] Filtering {len(generated_intra_tasks)} tasks...")
            # 批量调用 Pre-Filter
            # 注意：api_llm_pre_filter 通常接受 list 返回 list，这里我们一次性传入所有生成的任务
            # 如果列表过大，可以考虑分批，这里假设内存足够
            filtered_intra_tasks = generated_intra_tasks
            for f_filter in self.api_llm_pre_filter:
                filtered_intra_tasks = f_filter.filter(filtered_intra_tasks)
            
            logger.info(f"[Intra-Filter] {len(generated_intra_tasks)} -> {len(filtered_intra_tasks)} tasks remaining.")
            save_intermediate_tasks(intra_filtered_path, filtered_intra_tasks)
        else:
            logger.info("[Intra-Filter] Skipped (Loaded from Checkpoint)")

        # --- Step 3: Intra Exploration ---
        intra_res = []
        # 尝试加载最终结果
        if os.path.exists(intra_final_path):
            try:
                with open(intra_final_path, 'r') as f:
                    cp = json.load(f)
                    if cp.get('tasks_hash') == current_tasks_hash:
                        intra_res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in cp.get('results', [])]
                        logger.info(f"[Intra-Explore] Loaded {len(intra_res)} objectives from checkpoint.")
            except Exception: pass

        if not intra_res and filtered_intra_tasks:
            logger.info(f"[Intra-Explore] Exploring {len(filtered_intra_tasks)} tasks...")
            
            processed_indices = set() # 这里仅做计数用，因为已经有了 filtered_tasks 列表
            batch_res = []
            
            with ThreadPoolExecutor(max_workers=1 if debug_mode else self._num_exploration_threads) as pool:
                futures = {pool.submit(worker_explore_intra, t): i for i, t in enumerate(filtered_intra_tasks)}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Intra Exploration", disable=not show_progress):
                    try:
                        objs = future.result()
                        # 实时应用 post filter
                        filtered_objs = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, objs)
                        batch_res.extend(filtered_objs)
                    except Exception as e:
                        logger.error(f"Error in exploration future: {e}")
            
            intra_res = batch_res
            self._save_checkpoint(intra_final_path, intra_res, set(range(len(filtered_intra_tasks))), len(filtered_intra_tasks), current_tasks_hash)

        logger.info(f"[Intra-Domain] Completed. Collected {len(intra_res)} objectives.")


        # =================================================================
        # PART 2: CROSS-DOMAIN (生成 -> 过滤 -> 探索)
        # =================================================================
        
        # 2.1 准备 API Pairs
        valid_apps_list = [app for app in sorted(active_apps_set) if app in api_knowledge and api_knowledge[app].get("apis")]
        final_pair_data = []
        for app_name_a in valid_apps_list:
            apis_a_all = api_knowledge[app_name_a].get("apis", [])
            other_apps = [x for x in valid_apps_list if x != app_name_a]
            if not other_apps: continue
            random.shuffle(other_apps)
            
            loop_count = max(len(apis_a_all), len(other_apps))
            for i in range(loop_count):
                app_name_b = other_apps[i % len(other_apps)]
                apis_b_all = api_knowledge[app_name_b].get("apis", [])
                
                # Sample APIs
                s_a = random.sample(list(apis_a_all.values()), min(len(apis_a_all), 5))
                s_b = random.sample(list(apis_b_all.values()), min(len(apis_b_all), 5))
                
                final_pair_data.append([
                    {"app_name": app_name_a, "apis_name_list": [x["call_name"] for x in s_a]},
                    {"app_name": app_name_b, "apis_name_list": [x["call_name"] for x in s_b]}
                ])

        random.shuffle(final_pair_data)
        cross_task_pool = list(copy.copy(tasks)) * b
        if debug_mode: cross_task_pool = cross_task_pool[:1]
        
        total_cross = min(len(final_pair_data), len(cross_task_pool))

        cross_gen_path = f"{resume_file}.cross.generated.json"
        cross_filtered_path = f"{resume_file}.cross.filtered.json"
        cross_final_path = f"{resume_file}.extra.json" # Match original naming convention

        # --- Step 1: Cross Generation ---
        generated_cross_tasks = load_intermediate_tasks(cross_gen_path)
        if generated_cross_tasks is None:
            generated_cross_tasks = []
            logger.info(f"[Cross-Gen] Start generating {total_cross} task descriptions...")
            
            with ThreadPoolExecutor(max_workers=1 if debug_mode else self._num_exploration_threads) as pool:
                futures = []
                for idx in range(total_cross):
                    futures.append(pool.submit(worker_generate_cross, idx, final_pair_data[idx][0], final_pair_data[idx][1], cross_task_pool[idx]))
                
                for f in tqdm(as_completed(futures), total=len(futures), desc="Cross Generation", disable=not show_progress):
                    res = f.result()
                    if res: generated_cross_tasks.append(res)
            
            save_intermediate_tasks(cross_gen_path, generated_cross_tasks)
        else:
            logger.info("[Cross-Gen] Skipped (Loaded from Checkpoint)")

        # --- Step 2: Cross Filtering ---
        filtered_cross_tasks = load_intermediate_tasks(cross_filtered_path)
        if filtered_cross_tasks is None:
            logger.info(f"[Cross-Filter] Filtering {len(generated_cross_tasks)} tasks...")
            filtered_cross_tasks = generated_cross_tasks
            for f_filter in self.api_llm_pre_filter:
                filtered_cross_tasks = f_filter.filter(filtered_cross_tasks)
            
            logger.info(f"[Cross-Filter] {len(generated_cross_tasks)} -> {len(filtered_cross_tasks)} tasks remaining.")
            save_intermediate_tasks(cross_filtered_path, filtered_cross_tasks)
        else:
            logger.info("[Cross-Filter] Skipped (Loaded from Checkpoint)")

        # --- Step 3: Cross Exploration ---
        cross_res = []
        if os.path.exists(cross_final_path):
            try:
                with open(cross_final_path, 'r') as f:
                    cp = json.load(f)
                    if cp.get('tasks_hash') == current_tasks_hash:
                        cross_res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in cp.get('results', [])]
                        logger.info(f"[Cross-Explore] Loaded {len(cross_res)} objectives from checkpoint.")
            except Exception: pass

        if not cross_res and filtered_cross_tasks:
            logger.info(f"[Cross-Explore] Exploring {len(filtered_cross_tasks)} tasks...")
            batch_res = []
            with ThreadPoolExecutor(max_workers=1 if debug_mode else self._num_exploration_threads) as pool:
                futures = {pool.submit(worker_explore_cross, t): i for i, t in enumerate(filtered_cross_tasks)}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Cross Exploration", disable=not show_progress):
                    try:
                        objs = future.result()
                        filtered_objs = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, objs)
                        batch_res.extend(filtered_objs)
                    except Exception as e:
                        logger.error(f"Error in cross exploration future: {e}")
            
            cross_res = batch_res
            self._save_checkpoint(cross_final_path, cross_res, set(range(len(filtered_cross_tasks))), len(filtered_cross_tasks), current_tasks_hash)

        logger.info(f"[Cross-Domain] Completed. Collected {len(cross_res)} objectives.")

        # =================================================================
        # Final Merge
        # =================================================================
        total_results = intra_res + cross_res
        logger.info(f"[API-Driven] All stages finished. Total raw results: {len(total_results)}")
        return self._apply_post_filter(total_results)

    def _save_checkpoint(self, path, results, processed_indices, total, hash_val):
        """保存任务生成的断点信息到 JSON 文件"""
        try:
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

    def _apply_post_filter(self, res: List[TaskObjective]) -> List[TaskObjective]:
        """应用耗时较长的后置过滤器（如 LLM 质量核验），并打乱数据顺序"""
        
        # 先应用实时过滤器（确保最终一致性）
        res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
        
        logger.info("正在对生成的任务进行后置过滤（Post-Filter）...")
        cnt_before = len(res)
        res = functools.reduce(lambda x, f: f.filter(x), self._post_filter, res)
        logger.info(f"后置过滤完成: 过滤前={cnt_before}, 过滤后={len(res)}")
        
        random.shuffle(res)
        return res

    # --- 执行与总结的助手方法 ---

    def _exlore_and_summarize(self, task: Task, data_id: str, rollout_id: str) -> list[TaskObjective]:
        """随机策略下：执行探索并对轨迹进行总结"""
        try:
            trajectories = self._step_explore(task, data_id, rollout_id)
            task_objectives = sum([self._step_summarize(task, trajectory) for trajectory in trajectories], [])
            
            # 安全检查
            valid_objs = []
            for x in task_objectives:
                if x.task.open_query:
                    valid_objs.append(x)
            
            return valid_objs
        except Exception as e:
            logger.error(f"Error in random explore: {e}")
            return []

    def _step_explore(self, task: Task, data_id: str, rollout_id: str) -> list[Trajectory]:
        """调用策略的 explore 方法（Random 专用）"""
        return self._exploration_strategy.explore(task, data_id, rollout_id)

    def _step_summarize(self, task: Task, trajectory: Trajectory) -> list[TaskObjective]:
        """调用策略的 summarize 方法（Random 专用）"""
        return self._exploration_strategy.summarize(task, trajectory)

# ================= 数据集类 =================

class FullDataset(Dataset):
    """
    静态数据集：一次性生成/加载所有合成任务，并与原始种子任务混合。
    支持缓存到本地文件。
    """
    def __init__(self, manager: TaskManager, mixture_strategy: MixtureStrategy, reward_config: RewardProps, cache_path: Optional[str] = None, *, tokenizer, config, processor):
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
            if self._cache_path is not None and os.path.exists(self._cache_path):
                self.load_from_file()
            else:
                self.reload_new_task()
                if self._cache_path is not None: self.save_to_file()
        
        self._rebuild_dataset()

    def _rebuild_dataset(self):
        """混合原始数据和合成数据，并转换为训练格式"""
        self._objectives = self._mixture_strategy.mix_data(self._synthetic_objectives, self._tasks)
        
        # --- 添加保护逻辑 ---
        if len(self._objectives) == 0:
            logger.error("【严重错误】没有可用的训练数据！可能是环境服务挂了，或者 Debug 模式下生成的任务全部被过滤了。")
            raise ValueError("Dataset is empty. Please check env_service status or disable debug_log.")
        # -------------------

        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)

    def set_mixture_strategy(self, strategy: MixtureStrategy):
        """
        Sets the mixture strategy for the TaskManager and logs the update.

        Args:
            strategy (MixtureStrategy): The new mixture strategy to be set.
        """
        self._mixture_strategy = strategy  # ⭐ Update the mixture strategy
        logger.info(f"mixture strategy updated to: {type(strategy).__name__}")

    def save_to_file(self):
        """
        Saves the JSON representation of each synthetic objective to a specified file.

        Args:
            filepath (str): The path to the file where the objectives will be saved.

        Returns:
            None
        """
        assert self._cache_path is not None
        with open(self._cache_path, "w") as f:
            f.writelines([ob.json() + "\n" for ob in self._synthetic_objectives])  # ⭐ Writes each objective's JSON to the file
        logger.info(f"Saved {len(self._objectives)} objectives to {self._cache_path}")  # ⭐ Logs the number of objectives saved

    def load_from_file(self):
        """
        Loads objectives from a specified file. This function is currently incomplete.

        Args:
            filepath (str): The path to the file from which the objectives will be loaded.

        Returns:
            None
        """
        if self._cache_path is None:
            logger.error("trying to load synthetic objectives from file, but cache_path is not set")
            return
        
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "r") as f:
                self._synthetic_objectives = []
                for line in filter(lambda x: x.strip() != "", f.readlines()):
                    # patch old data: open query
                    t=json.loads(line)
                    assert 'task' in t
                    if 'open_query' not in t['task']:
                        t['task']['open_query'] = True # all synthetic data is open query
                    
                    # patch old data: ground_truth
                    tmp=TaskObjective.parse_obj(t)
                    if tmp.ground_truth is None:
                        tmp.ground_truth = json.loads(line)['ground_truth']
                    self._synthetic_objectives.append(tmp)
        else:
            raise FileNotFoundError(f"failed to load synthetic objectives from file {self._cache_path}, file not found")
        
        # check if all synthetic objectives have ground_truth
        for item in self._synthetic_objectives:
            assert item.ground_truth is not None

        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]  # ⭐ Update the evaluator for each task


    def reload_new_task(self):
        """
        Regenerates the synthetic objectives, updates their evaluators, and rebuilds the dataset.

        This method is used to refresh the task objectives and ensure they are up-to-date with the current configuration.
        """
        self._synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]  # ⭐ Update the evaluator for each task
        

    def get_statistics(self) -> dict:
        """
        Computes and returns a dictionary containing statistics about the tasks, such as the total number of tasks,
        the number of synthetic and original tasks, the ratio of synthetic tasks, and the strategy information.

        Returns:
            dict: A dictionary with keys 'total', 'synthetic', 'original', 'synthetic_ratio', and 'strategy_info'.
        """
        if not self._objectives:
            return {
                "total": 0,
                "synthetic": 0,
                "original": 0,
                "synthetic_ratio": 0.0,
                "strategy_info": str(self._mixture_strategy)
            }

        synthetic_count = sum(1 for obj in self._objectives if obj.task.evaluator != "env")  # ⭐ Count the number of synthetic tasks
        original_count = len(self._objectives) - synthetic_count  # ⭐ Calculate the number of original tasks

        return {
            "total": len(self._objectives),
            "synthetic": synthetic_count,
            "original": original_count,
            "synthetic_ratio": synthetic_count / len(self._objectives) if len(self._objectives) > 0 else 0,
            "strategy_info": str(self._mixture_strategy)
        }

    def __getitem__(self, index):
        """
        Allows indexing of the TaskManager instance to access items in the underlying dataset.

        Args:
            index (int): The index of the item to retrieve from the dataset.

        Returns:
            The item at the specified index in the dataset.

        Raises:
            RuntimeError: If the dataset has not been loaded.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call reload() or load_from_file() first.")  # ⭐ Ensures the dataset is loaded before accessing
        return self._dataset[index]

    def __len__(self):
        if self._dataset is None:
            return 0
        return len(self._dataset)

class AutoReloadDataset(IterableDataset):
    """
    可迭代数据集：在训练过程中，当数据耗尽时，动态触发 TaskManager 生成新任务（On-the-fly）。
    """
    def __init__(self, manager: TaskManager, tasks: Iterable[Task], bs: int, mix_origins: bool = False, *, tokenizer, config, processor):
        self._manager = manager
        self._tasks = tasks
        self._bs = bs
        self._tokenizer = tokenizer
        self._config = config
        self._processor = processor
        self._dataset = OnflyRlDataset(release_used_dataset=True)

    def reload(self):
        delta = []
        for task in self._tasks:
            delta.append(task)
            if len(delta) == self._bs:
                break

        ls = self._manager.generate_task(delta)
        while len(ls) < self._bs * self._manager._n:
            logger.debug("failed to generate enough tasks, retrying")
            ls = self._manager.generate_task(delta)

        self._dataset.append_dataset(to_rl_dataset(ls, self._tokenizer, self._config,self._processor))
        return self._dataset.num_rest_data

    def __iter__(self):
        return self

    def __next__(self):
        """
        Fetches the next task from the dataset. If no tasks are left, it tries to reload the dataset.
        If reloading does not provide any new tasks, it raises a StopIteration exception.

        Returns:
            Any: The next task from the dataset.

        Raises:
            StopIteration: If there are no more tasks left after attempting to reload the dataset.
        """
        if self._dataset.num_rest_data == 0:  # ⭐ Check if there are any remaining tasks
            logger.debug("no data left")
            if self.reload() == 0:  # ⭐ Attempt to reload the dataset
                logger.debug("no task left, stop reloading and iteration")
                raise StopIteration
        return next(self._dataset)  # ⭐ Get the next task from the dataset