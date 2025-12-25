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
from torch.utils.data import IterableDataset,Dataset
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
    n: int # n must be placed here. The task manager needs to plan the task execution order to avoid potential duplicate queries resulting from simultaneously exploring the same task.
           # n 代表每个种子任务被探索/扩展的次数。TaskManager 需要规划执行顺序以避免重复查询。

# 定义奖励相关的配置参数类型
class RewardProps(TypedDict):
    original_grader:str  # 原始任务的评分器名称 (e.g., "env" 或具体的 rule-based)
    synthetic_grader:str # 合成任务的评分器名称 (通常是 "model-based" 或其他)

class TaskManager(object):
    """
    任务管理器 (TaskManager)
    负责管理种子任务、生成新任务（探索与演化）、过滤任务以及与环境和 LLM 的交互。
    它是 AgentEvolver 中数据流转的核心组件。
    """

    def __init__(
        self,
        config: DictConfig,
        exploration_strategy: str,
        env_profile:EnvProfile,
        exploration_strategy_args,
        llm_client: LlmClient,
        old_retrival: TaskObjectiveRetrieval,
        mixture_strategy: MixtureStrategy,
        reward_config: RewardProps,
        tokenizer,
        env_service_url: str,
        # 新增：为了支持细粒度控制的 Agent 执行，传入执行组件
        agent_flow: Optional[AgentFlow] = None,
        env_worker: Optional[Any] = None, 
        **kwargs: Unpack[TaskManagerProps],
    ):
        """
        初始化 TaskManager。

        Args:
            config (DictConfig): 全局配置对象。
            exploration_strategy (str): 使用的探索策略名称 (如 'random', 'api_driven')。
            env_profile (EnvProfile): 环境配置文件，描述环境特征。
            exploration_strategy_args: 传给探索策略的具体参数。
            llm_client (LlmClient): 用于生成的 LLM 客户端。
            old_retrival (TaskObjectiveRetrieval): 用于检索旧任务目标的组件，常用于去重或避免循环。
            mixture_strategy (MixtureStrategy): 数据混合策略 (如何混合原始数据和合成数据)。
            reward_config (RewardProps): 奖励函数/评分器的配置。
            tokenizer: 分词器。
            env_service_url (str): 环境服务的 URL 地址。
            agent_flow (Optional[AgentFlow]): 智能体工作流 (用于 API Driven 策略)。
            env_worker (Optional[EnvWorker]): 环境执行器 (用于 API Driven 策略)。
            **kwargs: 包含 num_explore_threads 和 n 等其他参数。
        """
        self._config = config
        self._tokenizer = tokenizer
        # 初始化探索策略 (Factory pattern)
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

        # 保存执行组件 (新策略需要)
        self.agent_flow = agent_flow
        self.env_worker = env_worker

        # 初始化过滤器链
        self._realtime_filters: list[TaskPostFilter] = [NaiveTaskPostFilter()]
        self._post_filter: list[TaskPostFilter] = [
            LlmFilter(env_service_url, llm_client, self._num_exploration_threads, tokenizer=tokenizer, config=config)
        ]

        self._tasks: list[Task]=[] # 存储种子任务列表
        
        # 注入依赖到探索策略中
        # 注意：ApiDrivenExploreStrategy 可能不需要所有的依赖，或者需要额外的依赖
        self._exploration_strategy._inject_deps(
            self._old_retrival,
            self._llm_client,
            DashScopeClient(model_name='qwen3-235b-a22b-instruct-2507', max_tokens=8192),
            env_profile=env_profile
        )

    @property
    def seed_tasks(self):
        """
        返回当前的种子任务列表。
        """
        return self._tasks
    
    @property
    def seed_task_objectives(self):
        """
        将种子任务包装为 TaskObjective 对象，默认置信度为 1.0。
        """
        return [TaskObjective(task=task,confidence=1.0,reward=None) for task in self.seed_tasks]

    def load_tasks(self,tasks:Sequence[Task]):
        """
        直接加载任务列表。
        """
        self._tasks.extend(tasks)
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks, #tasks={len(self._tasks)}")

    def load_tasks_from_dataset(self, dataset: RLHFDataset,*, env_type:str):
        """
        从 RLHF 数据集加载任务。
        """
        self._tasks.extend(adapter.convert_to_tasks(dataset,env_type=env_type,grader=self._reward_config["original_grader"]))
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks from dataset, #tasks={len(self._tasks)}")

    def load_tasks_from_environment(self, env: EnvClient, *, env_type: str, split: str, params: Optional[dict] = None):
        """
        从远程环境服务加载任务配置。
        """
        try:
            response = env.get_env_profile(env_type, split, params)
            self._tasks.extend([Task(task_id=str(x),env_type=env_type,open_query=False,evaluator=self._reward_config["original_grader"]) for x in response])
            assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
            logger.info(f"loaded tasks from environment, #tasks={len(self._tasks)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"failed to load tasks from environment: {e}")
            raise
        return len(response)

    def register_filter(self, filter: TaskPostFilter):
        """
        注册额外的实时过滤器。
        """
        self._realtime_filters.append(filter)

    def _get_onthefly_dataset(self, bs: int, tokenizer, config,processor):
        raise NotImplementedError("get_onthefly_dataset is not implemented")

    def _compute_tasks_hash(self, tasks: Sequence[Task]) -> str:
        task_strs = [f"{task.task_id}:{task.env_type}" for task in tasks]
        combined_str = "|".join(task_strs)
        return hashlib.md5(combined_str.encode()).hexdigest()

    def generate_task(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        核心方法：生成新任务（兼容旧有的并行探索逻辑）。
        """
        if resume_file is None:
            resume_file = '.generate_task.checkpoint.json'

        current_tasks_hash = self._compute_tasks_hash(tasks)
        res = []
        processed_indices = set()
        if resume_file and os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    checkpoint = json.load(f)
                    if checkpoint['tasks_hash'] != current_tasks_hash:
                        logger.warning(f"Tasks hash mismatch. Expected: {current_tasks_hash}, got: {checkpoint['tasks_hash']}. Removing checkpoint.")
                        os.remove(resume_file)
                    else:
                        res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in checkpoint.get('results', [])]
                        processed_indices = {int(i) for i in checkpoint.get('processed_indices', [])}
                        logger.info(f"Resumed from checkpoint: {len(res)} results loaded, {len(processed_indices)} batches processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting from scratch")

        task_q = list(copy.copy(tasks)) * self._n

        parallel_num = min(self._num_exploration_threads, len(tasks))
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as pool:
            batch_indices = list(range(0, len(task_q), parallel_num))
            for idx, i in enumerate(tqdm(batch_indices, desc="generating tasks", disable=not show_progress)):
                if idx in processed_indices:
                    continue

                futures = [
                    pool.submit(self._exlore_and_summarize, task, data_id, rollout_id)
                    for task, data_id, rollout_id in zip(
                        task_q[i : i + parallel_num],
                        ["unknown"] * parallel_num,
                        ["unknown"] * parallel_num,
                    )
                ]
                task_objectives = sum([future.result() for future in futures], [])
                res.extend(task_objectives)
                
                res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
                
                self._old_retrival.reset()
                for j in res:
                    self._old_retrival.add_objective(j)

                processed_indices.add(idx)

                if resume_file:
                    try:
                        checkpoint_data = {
                            'results': [obj.dict() for obj in res],
                            'processed_indices': list(processed_indices),
                            'total_batches': len(batch_indices),
                            'tasks_hash': current_tasks_hash,
                            'timestamp': time.time()
                        }
                        with open(resume_file, 'w') as f:
                            json.dump(checkpoint_data, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to save checkpoint: {e}")

        res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
        
        logger.info("running post filter on generated tasks")
        cnt_before_filter=len(res)
        res = functools.reduce(lambda x, f: f.filter(x), self._post_filter, res)
        cnt_after_filter=len(res)
        logger.info(f"finish post filter: #before={cnt_before_filter}, #after={cnt_after_filter}")
        
        random.shuffle(res)

        return res

    def _exlore_and_summarize(self,task:Task,data_id:str,rollout_id:str)->list[TaskObjective]:
        """
        单次生成任务的完整流程：探索 -> 总结。
        """
        trajectories=self._step_explore(task,data_id,rollout_id)
        task_objectives=sum([self._step_summarize(task,trajectory) for trajectory in trajectories],[])
        assert all([x.task.open_query==True for x in task_objectives]), "all synthetic tasks must have open query"
        return task_objectives

    def _step_explore(self, task: Task, data_id: str, rollout_id: str)->list[Trajectory]:
        """
        步骤 1: 探索环境。
        注意：在 Random 策略中，explore 通常包含了自己的执行逻辑。
        """
        return self._exploration_strategy.explore(task,data_id,rollout_id)

    def _step_summarize(self, task: Task, trajectory: Trajectory) -> list[TaskObjective]:
        """
        步骤 2: 总结探索结果。
        """
        return self._exploration_strategy.summarize(task, trajectory)

    # =========================================================================
    #  New Methods for API-Driven Strategy (Extensions)
    #  新增方法：用于支持基于 API 驱动的增量学习框架，不影响原有 generate_task 逻辑
    # =========================================================================

    def generate_task_api(self, **kwargs) -> Optional[Task]:
        """
        专门用于 API Driven 策略的任务生成入口。
        根据 Strategy 的状态决定生成 Intra 还是 Cross 任务。
        """
        if not isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            logger.warning("generate_task_api called but strategy is not ApiDriven.")
            return None

        # 由 Strategy 决定当前处于哪个阶段 (Intra 还是 Extra/Cross)
        phase = self._exploration_strategy.decide_phase()
        
        if phase == "intra":
            return self._exploration_strategy.generate_intra_task()
        elif phase == "extra":
            return self._exploration_strategy.generate_cross_task()
        else:
            logger.info("All exploration phases completed or no tasks available.")
            return None

    def _explore_and_summarize_intra(self, task: Task) -> List[TaskObjective]:
        """
        单域探索完整流程：执行 -> 验证 -> 反向归纳
        """
        # 1. 执行
        trajectory = self._step_explore_intra(task)
        if not trajectory or not trajectory.steps:
            return []
            
        # 2. 总结
        objective = self._step_summarize_intra(task, trajectory)
        if objective:
            return [objective]
        return []

    def _explore_and_summarize_extra(self, task: Task) -> List[TaskObjective]:
        """
        跨域探索完整流程：环境注入 -> 执行 -> 验证
        """
        # 1. 执行 (包含数据注入)
        trajectory = self._step_explore_extra(task)
        if not trajectory or not trajectory.steps:
            return []
            
        # 2. 总结
        objective = self._step_summarize_extra(task, trajectory)
        if objective:
            return [objective]
        return []

    def _step_explore_intra(self, task: Task) -> Trajectory:
        """
        执行单域任务 (Intra-Domain Execution)
        """
        logger.info(f"[TaskManager] Executing Intra-Domain Task: {task.instruction[:50]}...")
        return self._execute_agent_loop(task)

    def _step_explore_extra(self, task: Task) -> Trajectory:
        """
        执行跨域任务 (Cross-Domain Execution)
        此处会包含环境数据注入 (Context Injection) 的逻辑。
        """
        logger.info(f"[TaskManager] Executing Cross-Domain Task: {task.instruction[:50]}...")
        return self._execute_agent_loop(task)

    def _step_summarize_intra(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        """
        调用策略进行单域总结 (反向归纳)。
        """
        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            return self._exploration_strategy.summarize_intra(task, trajectory)
        return None

    def _step_summarize_extra(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        """
        调用策略进行跨域总结 (验证)。
        """
        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            return self._exploration_strategy.summarize_cross(task, trajectory)
        return None

    def _execute_agent_loop(self, task: Task) -> Trajectory:
        """
        通用的 Agent 执行循环。
        用于 API Driven 策略，需要 self.env_worker 和 self.agent_flow 已初始化。
        """
        if not self.env_worker or not self.agent_flow:
            logger.error("EnvWorker or AgentFlow not initialized. Cannot execute agent loop.")
            return Trajectory(steps=[])

        # 1. 获取真实的 Sandbox ID (如果是 API Driven 策略)
        # 这样确保我们是在一个真实可用的环境快照上运行
        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            real_sandbox_id = self._exploration_strategy.get_next_sandbox_id()
            task.task_id = real_sandbox_id
        
        # 2. Reset 环境
        try:
            obs = self.env_worker.reset(task)
        except Exception as e:
            logger.error(f"Env reset failed: {e}")
            return Trajectory(steps=[])

        # 3. 数据注入 Hook (针对 Cross-Domain)
        # 必须在 reset 之后，run 之前执行
        if task.metrics and task.metrics.get("setup_action") == "inject_data":
            self._inject_context(task.metrics)

        # 4. 执行 Agent Flow
        try:
            # 运行 Agent，生成轨迹
            trajectory = self.agent_flow.run(task, self.env_worker)
            return trajectory
        except Exception as e:
            logger.error(f"Agent flow execution failed: {e}")
            import traceback
            traceback.print_exc()
            return Trajectory(steps=[])

    def _inject_context(self, metrics: Dict):
        """
        处理跨域任务的环境预设 (Setup Context)。
        例如：在笔记应用中插入一条记录，供后续任务查询。
        """
        try:
            app = metrics.get("app")
            content = metrics.get("content")
            # 这是一个示例，具体实现依赖于 EnvWorker 暴露的能力 (例如 God Mode API)
            if hasattr(self.env_worker, "execute_god_command") or hasattr(self.env_worker, "execute"):
                logger.info(f"[Data Injection] App: {app}, Content: {content}")
                # 实际逻辑需要根据 EnvWorker 的具体接口来实现
                # e.g., self.env_worker.execute(f"apis.{app}.create_note(content='{content}')")
            else:
                logger.warning("EnvWorker does not support direct execution for data injection.")
        except Exception as e:
            logger.warning(f"Data injection failed: {e}")


def get_exploration_strategy(name:str, strategy_args, *, tokenizer, config)->TaskExploreStrategy:
    """
    工厂函数：根据名称获取探索策略实例。
    """
    logger.info(f"loading exploration strategy {name}")
    if name=="random":
        return LlmRandomSamplingExploreStrategy(tokenizer=tokenizer,config=config,**strategy_args)
    elif name == "api_driven":
        # 支持新的 API 驱动策略
        return ApiDrivenExploreStrategy(tokenizer=tokenizer, config=config, **strategy_args)
    else:
        raise NotImplementedError(f"exploration strategy {name} not implemented")


class FullDataset(Dataset):
    """
    全量数据集类 (FullDataset)
    支持混合策略 (MixtureStrategy) 以及数据自动刷新。
    它负责将原始任务和合成任务结合，提供给训练器使用。
    """

    def __init__(self,
                 manager: TaskManager,
                 mixture_strategy: MixtureStrategy,
                 reward_config:RewardProps,
                 cache_path: Optional[str] = None,
                 *,
                 tokenizer,
                 config,
                 processor):
        """
        初始化 FullDataset。

        Args:
            manager (TaskManager): 任务管理器实例。
            mixture_strategy (MixtureStrategy): 数据混合策略。
            reward_config (RewardProps): 奖励配置。
            cache_path (Optional[str]): 合成数据的缓存路径。
        """
        self._manager = manager
        # 获取种子任务目标
        self._tasks = self._manager.seed_task_objectives
        # 确保任务评估器配置正确
        assert all([x.task.evaluator==reward_config["original_grader"] for x in self._tasks]), "task evaluator must be set as the config"
        self._mixture_strategy = mixture_strategy
        self._reward_config=reward_config
        self._cache_path = cache_path
        
        self._tokenizer = tokenizer
        self._config = config
        self._processor = processor
        
        self._objectives = []
        self._dataset = None
        self._synthetic_objectives = []

        # 标记：用于指示数据集是否需要在 epoch 后刷新
        self._refresh_after_epoch = False
        
        # 准备合成数据集
        if self._mixture_strategy.need_synthetic:
            logger.info("preparing synthetic tasks (准备合成任务)")
            # 优先从缓存文件加载
            if self._cache_path is not None and os.path.exists(self._cache_path):
                logger.info(f"loading synthetic tasks from file {self._cache_path}")
                self.load_from_file() # 加载合成数据
            else:
                # 否则重新生成
                self.reload_new_task() # 生成合成数据
                if self._cache_path is not None:
                    logger.debug("saving synthetic tasks to cache file")
                    self.save_to_file()
        else:
            logger.info(f"the mixture strategy need no synthetic data ({self._mixture_strategy}), skipping synthetic data...")
        
        # 构建混合后的数据集
        self._rebuild_dataset()
        

    def _rebuild_dataset(self):
        """
        使用当前的混合策略重新生成数据集。
        
        过程：
        1. 使用 mixture_strategy 混合 _synthetic_objectives 和 _tasks。
        2. 将混合后的 objectives 转换为 RLDataset 格式 (verl 库格式)。
        """
        self._objectives = self._mixture_strategy.mix_data(self._synthetic_objectives, self._tasks)
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)
        logger.info(f"Auto-refreshed dataset: #objectives={len(self._objectives)}, #rlhf={len(self._dataset)}")

    def update(self):
        """
        手动触发数据集重建。
        通常在合成数据更新后调用。
        """
        if not self._synthetic_objectives:
            logger.warning("No synthetic objectives available, did you call load_from_file() or reload() first?")
        self._rebuild_dataset()
        logger.info("Dataset updated manually via update().")

    def set_mixture_strategy(self, strategy: MixtureStrategy):
        """
        动态更新混合策略。
        """
        self._mixture_strategy = strategy
        logger.info(f"mixture strategy updated to: {type(strategy).__name__}")

    def save_to_file(self):
        """
        将合成任务目标保存到缓存文件。
        格式为 JSONL。
        """
        assert self._cache_path is not None
        with open(self._cache_path, "w") as f:
            f.writelines([ob.json() + "\n" for ob in self._synthetic_objectives])
        logger.info(f"Saved {len(self._objectives)} objectives to {self._cache_path}")

    def load_from_file(self):
        """
        从缓存文件加载合成任务目标。
        包含对旧数据格式的兼容性补丁 (Patching)。
        """
        if self._cache_path is None:
            logger.error("trying to load synthetic objectives from file, but cache_path is not set")
            return
        
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "r") as f:
                self._synthetic_objectives = []
                for line in filter(lambda x: x.strip() != "", f.readlines()):
                    # Patch: 确保存在 open_query 字段
                    t=json.loads(line)
                    assert 'task' in t
                    if 'open_query' not in t['task']:
                        t['task']['open_query'] = True # all synthetic data is open query
                    
                    # Patch: 修复 ground_truth 结构
                    tmp=TaskObjective.parse_obj(t)
                    if tmp.ground_truth is None:
                        tmp.ground_truth = json.loads(line)['ground_truth']
                    self._synthetic_objectives.append(tmp)
        else:
            raise FileNotFoundError(f"failed to load synthetic objectives from file {self._cache_path}, file not found")
        
        # 检查所有合成数据是否都有 GT
        for item in self._synthetic_objectives:
            assert item.ground_truth is not None

        logger.info("patching grader config to all synthetic data")
        # 将合成数据的评估器设置为配置文件中指定的 synthetic_grader
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]


    def reload_new_task(self):
        """
        调用 TaskManager 生成全新的合成任务，并更新 evaluator 配置。
        """
        self._synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]
        

    def get_statistics(self) -> dict:
        """
        获取当前数据集的统计信息 (总数、合成/原始比例、策略信息)。
        """
        if not self._objectives:
            return {
                "total": 0,
                "synthetic": 0,
                "original": 0,
                "synthetic_ratio": 0.0,
                "strategy_info": str(self._mixture_strategy)
            }

        # 通过 evaluator 类型区分合成任务和原始任务 (原始任务通常是 'env' 或特定 grader)
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
        """
        索引访问，代理到底层的 RLDataset。
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call reload() or load_from_file() first.")
        return self._dataset[index]

    def __len__(self):
        if self._dataset is None:
            return 0
        return len(self._dataset)


# 数据自动重载的包装器 (目前尚未完全支持)
class AutoReloadDataset(IterableDataset):
    """
    AutoReloadDataset
    注意：DataLoader 的 worker 数量必须为 1。
    """
    def __init__(self,manager:TaskManager, tasks:Iterable[Task], bs: int, mix_origins:bool=False, *, tokenizer, config, processor):
        self._manager=manager
        self._tasks=tasks
        self._bs = bs
        self._mix_origins=mix_origins
        assert self._mix_origins==False, "mix_origins is not supported yet"
        self._tokenizer = tokenizer
        self._config=config
        self._processor = processor

        self._dataset = OnflyRlDataset(release_used_dataset=True)

    def reload(self):
        """
        按需生成一批新的任务数据。
        """
        delta = []
        for task in self._tasks:
            delta.append(task)
            if len(delta) == self._bs:
                break

        # 调用 manager 生成任务
        ls = self._manager.generate_task(delta)
        # 确保生成的数量满足 batch size * 膨胀系数
        while len(ls) < self._bs * self._manager._n:
            logger.debug("failed to generate enough tasks, retrying")
            ls = self._manager.generate_task(delta)

        self._dataset.append_dataset(to_rl_dataset(ls, self._tokenizer, self._config,self._processor))
        return self._dataset.num_rest_data

    def __iter__(self):
        """
        返回迭代器自身。
        """
        return self

    def __next__(self):
        """
        获取下一个数据。
        如果当前 dataset 用尽，尝试 reload() 生成新数据。
        """
        if self._dataset.num_rest_data == 0:
            logger.debug("no data left")
            if self.reload() == 0:
                logger.debug("no task left, stop reloading and iteration")
                raise StopIteration
        return next(self._dataset)