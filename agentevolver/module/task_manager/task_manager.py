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
    if name == "random":
        # Random 策略似乎通过基类或后续注入处理，也可以根据需要传递
        return LlmRandomSamplingExploreStrategy(
            tokenizer=tokenizer, 
            config=config, 
            **strategy_args
            )
    elif name == "api_driven":
        # [修改点 2]: 将 llm_client 传递给 ApiDriven 策略
        return ApiDrivenExploreStrategy(
            tokenizer=tokenizer, 
            config=config, 
            llm_client=llm_client,  # <--- 传递实例
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
            llm_client=llm_client  # <--- 传入
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
        self._tasks.extend(adapter.convert_to_tasks(dataset, env_type=env_type, grader=self._reward_config["original_grader"]))
        assert all([x.query is None for x in self._tasks]), "种子任务的 query 必须为空"
        logger.info(f"loaded tasks from dataset, #tasks={len(self._tasks)}")

    def load_tasks_from_environment(self, env: EnvClient, *, env_type: str, split: str, params: Optional[dict] = None):
        """从环境服务端拉取可用的任务 ID，并构造种子任务"""
        try:
            response = env.get_env_profile(env_type, split, params)
            self._tasks.extend([Task(task_id=str(x), env_type=env_type, open_query=False, evaluator=self._reward_config["original_grader"]) for x in response])
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
        return hashlib.md5(combined_str.encode()).hexdigest()

    # --- 核心任务生成流程 ---

    def generate_task(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        生成任务的总入口：根据当前策略类型（API驱动 vs 随机采样）选择不同的执行流。
        """
        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
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
        strategy_args = self._config.task_manager.get('exploration_strategy_args', {})
        a = strategy_args.get('a', 1)
        b = strategy_args.get('b', 1)
        
        if resume_file is None:
            resume_file = '.generate_task_api'
        
        intra_ckpt_path = f"{resume_file}.intra.json"
        cross_ckpt_path = f"{resume_file}.extra.json"
        current_tasks_hash = self._compute_tasks_hash(tasks)
        
        # 获取基础数据
        api_knowledge = getattr(self._exploration_strategy, 'api_knowledge', {})
        active_apps_set = getattr(self._exploration_strategy, 'active_apps', set(api_knowledge.keys()))
        
        # =================================================================
        # 定义并行执行的原子函数 (Wrapper Functions)
        # =================================================================
        
        def process_intra_task(idx: int, api_info: dict, seed_task: Task) -> List[TaskObjective]:
            """单线程处理：单域任务生成 -> 探索 -> 总结"""
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
                if not current_task: return []
                
                # [FIX]: 生成唯一 data_id 并传递给 explore，避免日志覆盖
                data_id = f"gen_intra_{idx}"
                current_task.metrics["data_id"] = data_id
                
                # 2. 执行探索 (耗时操作) - 传入正确的 ID
                trajectories = self._exploration_strategy.explore(current_task, data_id, data_id)
                
                # 3. 总结结果 (Strategy 内部已加锁处理内存保存)
                results = []
                if trajectories and trajectories[0].steps:
                    results = self._exploration_strategy.summarize(current_task, trajectories[0])
                return results if results else []
            except Exception as e:
                logger.error(f"[Intra-Task Error] Index {idx}: {e}")
                return []

        def process_cross_task(idx: int, app_list: List[str], seed_task: Task) -> List[TaskObjective]:
            """单线程处理：跨域任务生成 -> 探索 -> 总结"""
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
                if not current_task: return []
                
                # [FIX]: 生成唯一 data_id 并传递给 explore，避免日志覆盖
                data_id = f"gen_cross_{idx}"
                current_task.metrics["data_id"] = data_id
                
                # 2. 执行探索 - 传入正确的 ID
                trajectories = self._exploration_strategy.explore(current_task, data_id, data_id)
                
                results = []
                if trajectories and trajectories[0].steps:
                    results = self._exploration_strategy.summarize(current_task, trajectories[0])
                return results if results else []
            except Exception as e:
                logger.error(f"[Cross-Task Error] Index {idx}: {e}")
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
            except Exception as e:
                logger.warning(f"Failed to load intra checkpoint: {e}")

        # 使用线程池执行
        total_intra = min(len(active_apis), len(intra_task_pool))
        if len(intra_processed_idx) < total_intra:
            parallel_num = max(1, min(self._num_exploration_threads, total_intra))
            
            remaining_indices = [i for i in range(total_intra) if i not in intra_processed_idx]
            batch_size = parallel_num * 2 
            batches = [remaining_indices[i:i + batch_size] for i in range(0, len(remaining_indices), batch_size)]

            pbar = tqdm(total=total_intra, desc="Stage 1: Intra-Domain (Parallel)", disable=not show_progress)
            pbar.update(len(intra_processed_idx))

            with ThreadPoolExecutor(max_workers=parallel_num) as pool:
                for batch_idxs in batches:
                    future_to_idx = {
                        pool.submit(process_intra_task, idx, active_apis[idx], intra_task_pool[idx]): idx 
                        for idx in batch_idxs
                    }

                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            objs = future.result()
                            intra_res.extend(objs)
                        except Exception as e:
                            logger.error(f"Unhandled exception in future for task {idx}: {e}")
                        finally:
                            intra_processed_idx.add(idx)
                            pbar.update(1)
                    
                    intra_res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, intra_res)
                    self._save_checkpoint(intra_ckpt_path, intra_res, intra_processed_idx, total_intra, current_tasks_hash)
            
            pbar.close()

        # =================================================================
        # 阶段 2: 跨域合成 (Cross-Domain) - 并行化
        # =================================================================
        cross_res = []
        active_apps_list = list(active_apps_set)
        
        cross_task_pool = list(copy.copy(tasks)) * b
        cross_processed_idx = set()
        
        if os.path.exists(cross_ckpt_path):
            try:
                with open(cross_ckpt_path, 'r') as f:
                    cp = json.load(f)
                    cross_res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in cp.get('results', [])]
                    cross_processed_idx = {int(i) for i in cp.get('processed_indices', [])}
                    logger.info(f"Cross-Domain resumed: {len(cross_res)} tasks loaded.")
            except Exception: pass

        if len(cross_processed_idx) < len(cross_task_pool):
            parallel_num = max(1, min(self._num_exploration_threads, len(cross_task_pool)))
            remaining_indices = [i for i in range(len(cross_task_pool)) if i not in cross_processed_idx]
            batches = [remaining_indices[i:i + parallel_num * 2] for i in range(0, len(remaining_indices), parallel_num * 2)]

            pbar = tqdm(total=len(cross_task_pool), desc="Stage 2: Cross-Domain (Parallel)", disable=not show_progress)
            pbar.update(len(cross_processed_idx))

            with ThreadPoolExecutor(max_workers=parallel_num) as pool:
                for batch_idxs in batches:
                    future_to_idx = {
                        pool.submit(process_cross_task, idx, active_apps_list, cross_task_pool[idx]): idx 
                        for idx in batch_idxs
                    }

                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            objs = future.result()
                            cross_res.extend(objs)
                        except Exception as e:
                            logger.error(f"Unhandled exception in cross task {idx}: {e}")
                        finally:
                            cross_processed_idx.add(idx)
                            pbar.update(1)
                    
                    current_batch_all = intra_res + cross_res
                    filtered_all = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, current_batch_all)
                    # 只保存 extra 的部分
                    cross_res_to_save = filtered_all[len(intra_res):] 
                    self._save_checkpoint(cross_ckpt_path, cross_res_to_save, cross_processed_idx, len(cross_task_pool), current_tasks_hash)
            
            pbar.close()
        
        return self._apply_post_filter(intra_res + cross_res)

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
        trajectories = self._step_explore(task, data_id, rollout_id)
        task_objectives = sum([self._step_summarize(task, trajectory) for trajectory in trajectories], [])
        assert all([x.task.open_query == True for x in task_objectives]), "所有合成任务必须包含 Query"
        return task_objectives

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
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)

    def save_to_file(self):
        """将生成的合成任务目标保存到 JSONL 文件"""
        assert self._cache_path is not None
        with open(self._cache_path, "w") as f:
            f.writelines([ob.json() + "\n" for ob in self._synthetic_objectives])

    def load_from_file(self):
        """从缓存文件加载合成任务，并修正评分器配置"""
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
        else:
            raise FileNotFoundError(f"找不到缓存文件 {self._cache_path}")

    def reload_new_task(self):
        """调用 TaskManager 重新触发演化生成流程"""
        self._synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        for item in self._synthetic_objectives:
            item.task.evaluator = self._reward_config["synthetic_grader"]

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset) if self._dataset else 0


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
        """动态拉取一批种子任务并演化出新任务，加入数据集队列"""
        delta = []
        for task in self._tasks:
            delta.append(task)
            if len(delta) == self._bs: break

        if not delta: return 0

        # 调用演化逻辑
        ls = self._manager.generate_task(delta)
        # 确保生成了足够的数据
        while len(ls) < self._bs * self._manager._n:
            logger.debug("合成数据量不足，正在重试生成...")
            ls = self._manager.generate_task(delta)

        self._dataset.append_dataset(to_rl_dataset(ls, self._tokenizer, self._config, self._processor))
        return self._dataset.num_rest_data

    def __iter__(self):
        return self

    def __next__(self):
        """获取下一条数据，如果为空则尝试 reload"""
        if self._dataset.num_rest_data == 0:
            if self.reload() == 0: raise StopIteration
        return next(self._dataset)