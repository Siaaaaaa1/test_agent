from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import functools
import hashlib
import json
import os
import pickle
import random
import threading
import time
from typing import (
    Callable,
    Iterable,
    NotRequired,
    Optional,
    Sequence,
    TypedDict,
    Unpack,
    List,
    Dict,
    Any
)

import hydra
from loguru import logger
from omegaconf import DictConfig
import requests
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.module.task_manager import adapter
from agentevolver.module.task_manager.adapter import OnflyRlDataset, to_rl_dataset
from agentevolver.module.task_manager.data_mixture import MixtureStrategy, OriginalOnlyStrategy
from agentevolver.module.task_manager.filters.llm_filter import LlmFilter
from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.module.task_manager.filters.filters import NaiveTaskPostFilter, TaskPostFilter

from agentevolver.module.task_manager.base import LlmClient, TaskObjectiveRetrieval
from agentevolver.module.task_manager.strategies.random import LlmRandomSamplingExploreStrategy
# 引入新的策略
from agentevolver.module.task_manager.strategies.api_driven import ApiDrivenExploreStrategy

from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from verl.utils.dataset.rl_dataset import RLHFDataset

# 定义 TaskManager 的可选参数类型
class TaskManagerProps(TypedDict):
    num_explore_threads: int  # 探索任务时的线程数
    n: int # n 代表每个种子任务被探索/扩展的次数。

# 定义奖励相关的配置参数类型
class RewardProps(TypedDict):
    original_grader: str  # 原始任务的评分器名称
    synthetic_grader: str # 合成任务的评分器名称

def get_exploration_strategy(name: str, strategy_args, *, tokenizer, config) -> TaskExploreStrategy:
    """
    工厂函数：根据名称获取探索策略实例。
    """
    logger.info(f"loading exploration strategy {name}")
    if name == "random":
        return LlmRandomSamplingExploreStrategy(tokenizer=tokenizer, config=config, **strategy_args)
    elif name == "api_driven":
        return ApiDrivenExploreStrategy(tokenizer=tokenizer, config=config, **strategy_args)
    else:
        raise NotImplementedError(f"exploration strategy {name} not implemented")

class TaskManager(object):
    """
    任务管理器 (TaskManager)
    负责管理种子任务、生成新任务（探索与演化）、过滤任务以及与环境和 LLM 的交互。
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
        self._config = config
        self._tokenizer = tokenizer
        # 初始化探索策略
        self._exploration_strategy = get_exploration_strategy(
            exploration_strategy, 
            exploration_strategy_args, 
            tokenizer=tokenizer, 
            config=config
        )
        self._llm_client = llm_client
        self._old_retrival = old_retrival
        self._mixture_strategy = mixture_strategy
        self._reward_config = reward_config
        self._env_service_url = env_service_url
        self._num_exploration_threads = kwargs.get("num_explore_threads", 10)
        self._n = kwargs.get("n", 1)

        # 保存执行组件
        self.agent_flow = agent_flow
        self.env_worker = env_worker

        # 初始化过滤器链
        self._realtime_filters: list[TaskPostFilter] = [NaiveTaskPostFilter()]
        self._post_filter: list[TaskPostFilter] = [
            LlmFilter(env_service_url, llm_client, self._num_exploration_threads, tokenizer=tokenizer, config=config)
        ]

        self._tasks: list[Task] = [] 
        
        # 注入依赖
        self._exploration_strategy._inject_deps(
            self._old_retrival,
            self._llm_client,
            DashScopeClient(model_name='qwen3-235b-a22b-instruct-2507', max_tokens=8192),
            env_profile=env_profile
        )

    @property
    def seed_tasks(self):
        return self._tasks
    
    @property
    def seed_task_objectives(self):
        return [TaskObjective(task=task, confidence=1.0, reward=None) for task in self.seed_tasks]

    def load_tasks(self, tasks: Sequence[Task]):
        self._tasks.extend(tasks)
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks, #tasks={len(self._tasks)}")

    def load_tasks_from_dataset(self, dataset: RLHFDataset, *, env_type: str):
        self._tasks.extend(adapter.convert_to_tasks(dataset, env_type=env_type, grader=self._reward_config["original_grader"]))
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks from dataset, #tasks={len(self._tasks)}")

    def load_tasks_from_environment(self, env: EnvClient, *, env_type: str, split: str, params: Optional[dict] = None):
        try:
            response = env.get_env_profile(env_type, split, params)
            self._tasks.extend([Task(task_id=str(x), env_type=env_type, open_query=False, evaluator=self._reward_config["original_grader"]) for x in response])
            assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
            logger.info(f"loaded tasks from environment, #tasks={len(self._tasks)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"failed to load tasks from environment: {e}")
            raise
        return len(response)

    def register_filter(self, filter: TaskPostFilter):
        self._realtime_filters.append(filter)

    def _get_onthefly_dataset(self, bs: int, tokenizer, config, processor):
        raise NotImplementedError("get_onthefly_dataset is not implemented")

    def _compute_tasks_hash(self, tasks: Sequence[Task]) -> str:
        task_strs = [f"{task.task_id}:{task.env_type}" for task in tasks]
        combined_str = "|".join(task_strs)
        return hashlib.md5(combined_str.encode()).hexdigest()

    def generate_task(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        核心生成入口：根据策略类型分发。
        """
        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            return self._generate_task_api_driven(tasks, show_progress=show_progress, resume_file=resume_file)
        else:
            return self._generate_task_random(tasks, show_progress=show_progress, resume_file=resume_file)

    def _generate_task_random(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """Random 策略：使用多线程并行探索"""
        if resume_file is None:
            resume_file = '.generate_task.checkpoint.json'

        current_tasks_hash = self._compute_tasks_hash(tasks)
        res = []
        processed_indices = set()
        
        # 尝试加载 checkpoint
        if resume_file and os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    checkpoint = json.load(f)
                    if checkpoint.get('tasks_hash') != current_tasks_hash:
                        logger.warning(f"Tasks hash mismatch. Removing checkpoint.")
                        os.remove(resume_file)
                    else:
                        res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in checkpoint.get('results', [])]
                        processed_indices = {int(i) for i in checkpoint.get('processed_indices', [])}
                        logger.info(f"Resumed from checkpoint: {len(res)} results loaded")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting from scratch")

        task_q = list(copy.copy(tasks)) * self._n
        parallel_num = max(1, min(self._num_exploration_threads, len(tasks)))
        
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as pool:
            batch_indices = list(range(0, len(task_q), parallel_num))
            for idx, i in enumerate(tqdm(batch_indices, desc="generating tasks (random)", disable=not show_progress)):
                if idx in processed_indices:
                    continue

                futures = [
                    pool.submit(self._exlore_and_summarize, task, "unknown", "unknown")
                    for task in task_q[i : i + parallel_num]
                ]
                task_objectives = sum([future.result() for future in futures], [])
                res.extend(task_objectives)
                
                # 实时过滤与检索更新
                res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
                self._old_retrival.reset()
                for j in res:
                    self._old_retrival.add_objective(j)

                processed_indices.add(idx)

                if resume_file:
                    self._save_checkpoint(resume_file, res, processed_indices, len(batch_indices), current_tasks_hash)

        return self._apply_post_filter(res)

    def _generate_task_api_driven(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        API Driven 策略：串行执行，维护环境状态。
        不同于 Random 策略的并行，API 策略通常依赖于环境的连续状态变化，因此采用 While 循环顺序生成。
        """
        # 1. 初始化 checkpoint 文件路径
        if resume_file is None:
            resume_file = '.generate_task_api.checkpoint.json'

        # 2. 计算任务哈希与目标数量
        # 计算当前种子任务的哈希值，用于验证断点文件是否匹配（防止种子任务变了但还在用旧断点）
        current_tasks_hash = self._compute_tasks_hash(tasks)
        res = []
        # 目标生成数量 = 种子任务数 * 膨胀系数 n
        target_count = len(tasks) * self._n
        
        # 3. 尝试加载断点 (Checkpoint)
        if resume_file and os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    checkpoint = json.load(f)
                    # 验证哈希一致性
                    if checkpoint.get('tasks_hash') != current_tasks_hash:
                        logger.warning("Tasks hash mismatch for API strategy.")
                    else:
                        # 恢复已生成的任务目标
                        res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in checkpoint.get('results', [])]
                        logger.info(f"Resumed API generation: {len(res)} results loaded.")
            except Exception as e:
                logger.warning(f"Failed to load API checkpoint: {e}")

        # 4. 初始化进度条
        pbar = tqdm(total=target_count, desc="generating tasks (api)", disable=not show_progress)
        pbar.update(len(res))

        # 5. 主生成循环
        # 持续生成直到达到目标数量，或者策略判定结束
        while len(res) < target_count:
            # 5.1 策略决定当前阶段 (State Machine Decision)
            # 策略对象内部维护状态，决定是进行单域探索 (Intra) 还是跨域探索 (Cross/Extra)
            phase = self._exploration_strategy.decide_phase()
            new_objectives = []

            # 5.2 根据阶段生成并执行任务
            if phase == "intra":
                # 生成单域任务 -> 执行 Agent -> 总结反思
                task = self._exploration_strategy.generate_intra_task()
                if task:
                    new_objectives = self._explore_and_summarize_intra(task)
            
            elif phase == "extra":
                # 生成跨域任务 -> 执行 Agent (含数据注入) -> 总结验证
                task = self._exploration_strategy.generate_cross_task()
                if task:
                    new_objectives = self._explore_and_summarize_extra(task)
            else:
                # 策略返回未知阶段或指示结束 (None)，跳出循环
                logger.info(f"API Strategy finished or unknown phase: {phase}")
                break
            
            # 5.3 异常处理：如果没有生成有效目标
            if not new_objectives:
                logger.debug("No valid objectives generated in this step.")
                # 这里可以添加重试逻辑或死循环保护 (例如连续 N 次失败则退出)
                pass
            
            # 5.4 更新结果集
            res.extend(new_objectives)
            pbar.update(len(new_objectives))

            # 5.5 实时过滤与检索库更新
            # 应用实时过滤器 (如去重、格式检查)
            res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
            # 更新旧任务检索器 (Retrieval Memory)，防止后续生成重复的任务
            self._old_retrival.reset()
            for j in res:
                self._old_retrival.add_objective(j)
            
            # 5.6 保存断点
            # 每次循环后保存，确保意外中断时损失最小
            if resume_file:
                self._save_checkpoint(resume_file, res, [], target_count, current_tasks_hash)

        pbar.close()
        
        # 6. 后置过滤 (Post-Filter)
        # 应用更耗时或全局性的过滤器 (如 LLM 质量评分)，并打乱顺序以供训练使用
        return self._apply_post_filter(res)

    def _save_checkpoint(self, path, results, processed_indices, total, hash_val):
        """提取的公共 Checkpoint 保存逻辑"""
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
            logger.warning(f"Failed to save checkpoint: {e}")

    def _apply_post_filter(self, res: List[TaskObjective]) -> List[TaskObjective]:
        # 确保实时过滤器最终执行
        res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
        
        logger.info("running post filter on generated tasks")
        cnt_before_filter = len(res)
        res = functools.reduce(lambda x, f: f.filter(x), self._post_filter, res)
        logger.info(f"finish post filter: #before={cnt_before_filter}, #after={len(res)}")
        
        random.shuffle(res)
        return res

    # -------------------------------------------------------------------------
    #  Execution & Summary Helpers
    # -------------------------------------------------------------------------

    def _exlore_and_summarize(self, task: Task, data_id: str, rollout_id: str) -> list[TaskObjective]:
        """Random 策略使用的单步逻辑"""
        trajectories = self._step_explore(task, data_id, rollout_id)
        task_objectives = sum([self._step_summarize(task, trajectory) for trajectory in trajectories], [])
        assert all([x.task.open_query == True for x in task_objectives]), "all synthetic tasks must have open query"
        return task_objectives

    def _step_explore(self, task: Task, data_id: str, rollout_id: str) -> list[Trajectory]:
        return self._exploration_strategy.explore(task, data_id, rollout_id)

    def _step_summarize(self, task: Task, trajectory: Trajectory) -> list[TaskObjective]:
        return self._exploration_strategy.summarize(task, trajectory)

    def _explore_and_summarize_intra(self, task: Task) -> List[TaskObjective]:
        trajectory = self._step_explore_intra(task)
        if not trajectory or not trajectory.steps:
            return []
        objective = self._step_summarize_intra(task, trajectory)
        return [objective] if objective else []

    def _explore_and_summarize_extra(self, task: Task) -> List[TaskObjective]:
        trajectory = self._step_explore_extra(task) # 内部包含 inject_context
        if not trajectory or not trajectory.steps:
            return []
        objective = self._step_summarize_extra(task, trajectory)
        return [objective] if objective else []

    def _step_explore_intra(self, task: Task) -> Trajectory:
        logger.info(f"[TaskManager] Executing Intra-Domain Task: {task.instruction[:50]}...")
        return self._execute_agent_loop(task)

    def _step_explore_extra(self, task: Task) -> Trajectory:
        logger.info(f"[TaskManager] Executing Cross-Domain Task: {task.instruction[:50]}...")
        return self._execute_agent_loop(task)

    def _step_summarize_intra(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            return self._exploration_strategy.summarize_intra(task, trajectory)
        return None

    def _step_summarize_extra(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            return self._exploration_strategy.summarize_cross(task, trajectory)
        return None

    def _execute_agent_loop(self, task: Task) -> Trajectory:
        """通用的 Agent 执行循环，包含 Reset 和 Injection"""
        if not self.env_worker or not self.agent_flow:
            logger.error("EnvWorker or AgentFlow not initialized. Cannot execute agent loop.")
            return Trajectory(steps=[])

        # 1. 确保使用真实的 Sandbox ID
        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            real_sandbox_id = self._exploration_strategy.get_next_sandbox_id()
            if real_sandbox_id:
                task.task_id = real_sandbox_id
        
        # 2. Reset 环境
        try:
            self.env_worker.reset(task)
        except Exception as e:
            logger.error(f"Env reset failed: {e}")
            return Trajectory(steps=[])

        # 3. 数据注入 (Cross-Domain)
        if task.metrics and task.metrics.get("setup_action") == "inject_data":
            self._inject_context(task.metrics)

        # 4. 执行 Agent Flow
        try:
            trajectory = self.agent_flow.run(task, self.env_worker)
            return trajectory
        except Exception as e:
            logger.error(f"Agent flow execution failed: {e}")
            return Trajectory(steps=[])

    def _inject_context(self, metrics: Dict):
        try:
            app = metrics.get("app")
            content = metrics.get("content")
            # 兼容不同的 EnvWorker 接口
            if hasattr(self.env_worker, "execute_god_command"):
                logger.info(f"[Data Injection] App: {app}, Content: {content}")
                # self.env_worker.execute_god_command(...)
            elif hasattr(self.env_worker, "execute"):
                # self.env_worker.execute(...)
                pass
            else:
                logger.warning("EnvWorker does not support direct execution for data injection.")
        except Exception as e:
            logger.warning(f"Data injection failed: {e}")


class FullDataset(Dataset):
    def __init__(self,
                 manager: TaskManager,
                 mixture_strategy: MixtureStrategy,
                 reward_config: RewardProps,
                 cache_path: Optional[str] = None,
                 *,
                 tokenizer,
                 config,
                 processor):
        self._manager = manager
        self._tasks = self._manager.seed_task_objectives
        assert all([x.task.evaluator == reward_config["original_grader"] for x in self._tasks]), "task evaluator must be set as the config"
        self._mixture_strategy = mixture_strategy
        self._reward_config = reward_config
        self._cache_path = cache_path
        
        self._tokenizer = tokenizer
        self._config = config
        self._processor = processor
        
        self._objectives = []
        self._dataset = None
        self._synthetic_objectives = []

        if self._mixture_strategy.need_synthetic:
            logger.info("preparing synthetic tasks (准备合成任务)")
            if self._cache_path is not None and os.path.exists(self._cache_path):
                logger.info(f"loading synthetic tasks from file {self._cache_path}")
                self.load_from_file()
            else:
                self.reload_new_task()
                if self._cache_path is not None:
                    logger.debug("saving synthetic tasks to cache file")
                    self.save_to_file()
        else:
            logger.info(f"the mixture strategy need no synthetic data ({self._mixture_strategy}), skipping synthetic data...")
        
        self._rebuild_dataset()

    def _rebuild_dataset(self):
        self._objectives = self._mixture_strategy.mix_data(self._synthetic_objectives, self._tasks)
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)
        logger.info(f"Auto-refreshed dataset: #objectives={len(self._objectives)}, #rlhf={len(self._dataset)}")

    def update(self):
        if not self._synthetic_objectives:
            logger.warning("No synthetic objectives available, did you call load_from_file() or reload() first?")
        self._rebuild_dataset()
        logger.info("Dataset updated manually via update().")

    def set_mixture_strategy(self, strategy: MixtureStrategy):
        self._mixture_strategy = strategy
        logger.info(f"mixture strategy updated to: {type(strategy).__name__}")

    def save_to_file(self):
        assert self._cache_path is not None
        with open(self._cache_path, "w") as f:
            f.writelines([ob.json() + "\n" for ob in self._synthetic_objectives])
        logger.info(f"Saved {len(self._objectives)} objectives to {self._cache_path}")

    def load_from_file(self):
        if self._cache_path is None:
            logger.error("trying to load synthetic objectives from file, but cache_path is not set")
            return
        
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "r") as f:
                self._synthetic_objectives = []
                for line in filter(lambda x: x.strip() != "", f.readlines()):
                    t = json.loads(line)
                    assert 'task' in t
                    if 'open_query' not in t['task']:
                        t['task']['open_query'] = True
                    
                    tmp = TaskObjective.parse_obj(t)
                    if tmp.ground_truth is None:
                        tmp.ground_truth = json.loads(line)['ground_truth']
                    self._synthetic_objectives.append(tmp)
        else:
            raise FileNotFoundError(f"failed to load synthetic objectives from file {self._cache_path}, file not found")
        
        for item in self._synthetic_objectives:
            assert item.ground_truth is not None

        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator = self._reward_config["synthetic_grader"]

    def reload_new_task(self):
        self._synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator = self._reward_config["synthetic_grader"]

    def get_statistics(self) -> dict:
        if not self._objectives:
            return {
                "total": 0,
                "synthetic": 0,
                "original": 0,
                "synthetic_ratio": 0.0,
                "strategy_info": str(self._mixture_strategy)
            }

        synthetic_count = sum(1 for obj in self._objectives if obj.task.evaluator != "env")
        original_count = len(self._objectives) - synthetic_count

        return {
            "total": len(self._objectives),
            "synthetic": synthetic_count,
            "original": original_count,
            "synthetic_ratio": synthetic_count / len(self._objectives) if len(self._objectives) > 0 else 0,
            "strategy_info": str(self._mixture_strategy)
        }

    def __getitem__(self, index):
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call reload() or load_from_file() first.")
        return self._dataset[index]

    def __len__(self):
        if self._dataset is None:
            return 0
        return len(self._dataset)

class AutoReloadDataset(IterableDataset):
    def __init__(self, manager: TaskManager, tasks: Iterable[Task], bs: int, mix_origins: bool = False, *, tokenizer, config, processor):
        self._manager = manager
        self._tasks = tasks
        self._bs = bs
        self._mix_origins = mix_origins
        assert self._mix_origins == False, "mix_origins is not supported yet"
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

        self._dataset.append_dataset(to_rl_dataset(ls, self._tokenizer, self._config, self._processor))
        return self._dataset.num_rest_data

    def __iter__(self):
        return self

    def __next__(self):
        if self._dataset.num_rest_data == 0:
            logger.debug("no data left")
            if self.reload() == 0:
                logger.debug("no task left, stop reloading and iteration")
                raise StopIteration
        return next(self._dataset)