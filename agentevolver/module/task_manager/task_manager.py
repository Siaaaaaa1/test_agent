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
        **kwargs: Unpack[TaskManagerProps],
    ):
        """
        初始化 TaskManager。

        Args:
            config (DictConfig): 全局配置对象。
            exploration_strategy (str): 使用的探索策略名称 (如 'random')。
            env_profile (EnvProfile): 环境配置文件，描述环境特征。
            exploration_strategy_args: 传给探索策略的具体参数。
            llm_client (LlmClient): 用于生成的 LLM 客户端。
            old_retrival (TaskObjectiveRetrieval): 用于检索旧任务目标的组件，常用于去重或避免循环。
            mixture_strategy (MixtureStrategy): 数据混合策略 (如何混合原始数据和合成数据)。
            reward_config (RewardProps): 奖励函数/评分器的配置。
            tokenizer: 分词器。
            env_service_url (str): 环境服务的 URL 地址。
            **kwargs: 包含 num_explore_threads 和 n 等其他参数。
        """
        self._config = config
        # 初始化探索策略 (Factory pattern)
        self._exploration_strategy=get_exploration_strategy(exploration_strategy,exploration_strategy_args,tokenizer=tokenizer,config=config)  # ⭐ Initialize the exploration strategy
        self._llm_client = llm_client
        self._old_retrival = old_retrival
        self._mixture_strategy = mixture_strategy
        self._reward_config=reward_config
        self._env_service_url = env_service_url
        self._tokenizer = tokenizer  # tokenizer 在此处主要用于长度计算或传递给下游组件
        self._num_exploration_threads = kwargs["num_explore_threads"] or 10 # 默认 10 个线程进行并行探索
        self._n = kwargs["n"] # 膨胀系数：每个种子任务生成多少个新任务

        # 初始化过滤器链
        # 实时过滤器：生成过程中立即应用 (如简单的规则过滤)
        self._realtime_filters: list[TaskPostFilter] = [NaiveTaskPostFilter()]
        # 后置过滤器：生成完成后应用 (如基于 LLM 的质量筛选)
        self._post_filter: list[TaskPostFilter] = [LlmFilter(env_service_url,llm_client,self._num_exploration_threads,tokenizer=tokenizer,config=config)]  # ⭐ Initialize the post filter

        self._tasks: list[Task]=[] # 存储种子任务列表
        # 注入依赖到探索策略中，使其能够访问检索器、LLM 和环境信息
        self._exploration_strategy._inject_deps(self._old_retrival,self._llm_client,DashScopeClient(model_name='qwen3-235b-a22b-instruct-2507',max_tokens=8192),env_profile=env_profile)  # ⭐ Inject dependencies into the exploration strategy

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
        # 确保种子任务没有预设的 query (query 应该是在探索过程中生成的)
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks, #tasks={len(self._tasks)}")

    def load_tasks_from_dataset(self, dataset: RLHFDataset,*, env_type:str):
        """
        从 RLHF 数据集加载任务。

        Args:
            dataset (RLHFDataset): 源数据集。
            env_type (str): 任务所属的环境类型。
        """
        # 使用 adapter 将 dataset 转换为内部 Task 对象，并设置 evaluator
        self._tasks.extend(adapter.convert_to_tasks(dataset,env_type=env_type,grader=self._reward_config["original_grader"]))  # ⭐ Convert dataset to tasks and add to the task list
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks from dataset, #tasks={len(self._tasks)}")

    def load_tasks_from_environment(self, env: EnvClient, *, env_type: str, split: str, params: Optional[dict] = None):
        """
        从远程环境服务加载任务配置。

        Args:
            env (EnvClient): 环境客户端。
            env_type (str): 环境类型 (e.g., 'appworld', 'webshop')。
            split (str): 数据集划分 ('train', 'test' 等)。
            params (Optional[dict]): 额外请求参数。
        """
        try:
            response = env.get_env_profile(env_type, split, params)
            # 将环境返回的 profile 转换为 Task 对象
            self._tasks.extend([Task(task_id=str(x),env_type=env_type,open_query=False,evaluator=self._reward_config["original_grader"]) for x in response])  # ⭐ Create Task objects from the response and add to the task list
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
        """
        获取即时生成的数据集 (On-the-fly)。
        目前尚未实现，因为自动重载不支持混合策略。
        """
        # autoreloaddataset does not support mixture
        raise NotImplementedError("get_onthefly_dataset is not implemented")
        # return AutoReloadDataset(self,iter(self._tasks),bs,self._mix_original_tasks,tokenizer=tokenizer,config=config,processor=processor)


    def _compute_tasks_hash(self, tasks: Sequence[Task]) -> str:
        """
        计算任务列表的 MD5 哈希值。
        
        用于在断点续传 (resume) 时验证当前的种子任务是否与检查点文件中的任务一致。
        """
        task_strs = [f"{task.task_id}:{task.env_type}" for task in tasks]
        combined_str = "|".join(task_strs)
        return hashlib.md5(combined_str.encode()).hexdigest()  # ⭐ Compute the MD5 hash of the combined string

    def generate_task(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        核心方法：生成新任务。
        
        流程：
        1. 检查断点续传文件，如果存在且哈希匹配，则加载进度。
        2. 将任务队列扩展 N 倍 (self._n)。
        3. 使用线程池 (ThreadPoolExecutor) 并行执行 `_exlore_and_summarize`。
        4. 收集生成的结果 (TaskObjective)。
        5. 应用实时过滤器 (Realtime Filter)。
        6. 将结果加入 `_old_retrival` 以供后续去重/检索参考。
        7. 定期保存检查点。
        8. 应用后置过滤器 (Post Filter，通常更耗时)。
        9. 打乱结果并返回。

        Args:
            tasks (Sequence[Task]): 种子任务列表。
            show_progress (bool): 是否显示进度条。
            resume_file (Optional[str]): 检查点文件路径。

        Returns:
            list[TaskObjective]: 生成的任务目标列表。
        """
        if resume_file is None:
            resume_file = '.generate_task.checkpoint.json'

        # 计算当前任务哈希
        current_tasks_hash = self._compute_tasks_hash(tasks)
        # 尝试加载 Checkpoint
        res = []
        processed_indices = set()
        if resume_file and os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    checkpoint = json.load(f)
                    # 校验哈希，如果不匹配说明种子任务变了，删除旧的 checkpoint
                    if checkpoint['tasks_hash'] != current_tasks_hash:
                        logger.warning(f"Tasks hash mismatch. Expected: {current_tasks_hash}, got: {checkpoint['tasks_hash']}. Removing checkpoint.")
                        os.remove(resume_file)
                    else:
                        res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in checkpoint.get('results', [])]
                        processed_indices = {int(i) for i in checkpoint.get('processed_indices', [])}
                        logger.info(f"Resumed from checkpoint: {len(res)} results loaded, {len(processed_indices)} batches processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting from scratch")

        # 任务队列扩充：每个任务重复 N 次以生成不同的变体
        task_q = list(copy.copy(tasks)) * self._n

        # 并行执行配置
        parallel_num = min(self._num_exploration_threads, len(tasks))
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as pool:
            # 按 parallel_num 分批处理
            batch_indices = list(range(0, len(task_q), parallel_num))
            for idx, i in enumerate(tqdm(batch_indices, desc="generating tasks", disable=not show_progress)):
                # 如果该批次在 checkpoint 中已存在，则跳过
                if idx in processed_indices:
                    continue

                # 提交任务到线程池
                futures = [
                    pool.submit(self._exlore_and_summarize, task, data_id, rollout_id)
                    for task, data_id, rollout_id in zip(
                        task_q[i : i + parallel_num],
                        ["unknown"] * parallel_num, # 占位符 ID
                        ["unknown"] * parallel_num, # 占位符 ID
                    )
                ]
                # 获取结果
                task_objectives = sum([future.result() for future in futures], [])  # ⭐ Collect results from all futures
                res.extend(task_objectives)
                
                # 应用实时过滤器 (reduce 模式)
                res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
                
                # 更新检索器状态 (用于去重)
                self._old_retrival.reset()
                for j in res:
                    self._old_retrival.add_objective(j)

                # 标记该批次完成
                processed_indices.add(idx)

                # 保存 Checkpoint
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


        # 再次应用实时过滤器 (确保最后的结果也被过滤)
        res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
        
        # 应用后置过滤器 (LLM Filter 等)
        logger.info("running post filter on generated tasks")
        cnt_before_filter=len(res)
        res = functools.reduce(lambda x, f: f.filter(x), self._post_filter, res)  # ⭐ Apply post filters to the results
        cnt_after_filter=len(res)
        logger.info(f"finish post filter: #before={cnt_before_filter}, #after={cnt_after_filter}")
        
        random.shuffle(res)  # ⭐ Shuffle the final list of task objectives

        return res


    def _exlore_and_summarize(self,task:Task,data_id:str,rollout_id:str)->list[TaskObjective]:
        """
        单次生成任务的完整流程：探索 -> 总结。

        Args:
            task (Task): 种子任务。
            data_id (str): 数据 ID。
            rollout_id (str): Rollout ID。

        Returns:
            list[TaskObjective]: 生成的任务目标列表。
        """
        # 步骤 1: 探索 (Explore)，获取轨迹
        trajectories=self._step_explore(task,data_id,rollout_id)  # ⭐ Explore the environment
        # 步骤 2: 总结 (Summarize)，将轨迹转换为任务目标 (Instruction + Ground Truth)
        task_objectives=sum([self._step_summarize(task,trajectory) for trajectory in trajectories],[])  # ⭐ Summarize the exploration results
        # 校验：合成任务必须标记为 open_query
        assert all([x.task.open_query==True for x in task_objectives]), "all synthetic tasks must have open query"
        return task_objectives


    def _step_explore(self, task: Task, data_id: str, rollout_id: str)->list[Trajectory]:
        """
        步骤 1: 探索环境，找出可能的动作及其结果。
        这通常涉及让 LLM Agent 在环境中交互，生成 Trajectory。
        """
        return self._exploration_strategy.explore(task,data_id,rollout_id)  # ⭐ Execute the exploration strategy


    def _step_summarize(
        self, task: Task, trajectory: Trajectory
    ) -> list[TaskObjective]:
        """
        步骤 2: 总结探索结果，生成 TASK (Query 和 Ground Truth)。
        这通常涉及分析 Trajectory，提取出有价值的 instruction-action 对。
        """
        return self._exploration_strategy.summarize(task, trajectory)  # ⭐ Execute the summarization strategy


def get_exploration_strategy(name:str, strategy_args, *, tokenizer, config)->TaskExploreStrategy:
    """
    工厂函数：根据名称获取探索策略实例。
    """
    logger.info(f"loading exploration strategy {name}")
    if name=="random":
        # 目前仅实现了随机采样策略
        return LlmRandomSamplingExploreStrategy(tokenizer=tokenizer,config=config,**strategy_args)
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
        self._objectives = self._mixture_strategy.mix_data(self._synthetic_objectives, self._tasks)  # ⭐ Mixes synthetic objectives with current tasks
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)  # ⭐ Converts the mixed data into an RL dataset
        logger.info(f"Auto-refreshed dataset: #objectives={len(self._objectives)}, #rlhf={len(self._dataset)}")  # ⭐ Logs the number of objectives and RLHF items

    def update(self):
        """
        手动触发数据集重建。
        通常在合成数据更新后调用。
        """
        if not self._synthetic_objectives:
            logger.warning("No synthetic objectives available, did you call load_from_file() or reload() first?")
        self._rebuild_dataset()  # ⭐ Rebuilds the dataset
        logger.info("Dataset updated manually via update().")

    def set_mixture_strategy(self, strategy: MixtureStrategy):
        """
        动态更新混合策略。
        """
        self._mixture_strategy = strategy  # ⭐ Update the mixture strategy
        logger.info(f"mixture strategy updated to: {type(strategy).__name__}")

    def save_to_file(self):
        """
        将合成任务目标保存到缓存文件。
        格式为 JSONL。
        """
        assert self._cache_path is not None
        with open(self._cache_path, "w") as f:
            f.writelines([ob.json() + "\n" for ob in self._synthetic_objectives])  # ⭐ Writes each objective's JSON to the file
        logger.info(f"Saved {len(self._objectives)} objectives to {self._cache_path}")  # ⭐ Logs the number of objectives saved

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
            item.task.evaluator=self._reward_config["synthetic_grader"]  # ⭐ Update the evaluator for each task


    def reload_new_task(self):
        """
        调用 TaskManager 生成全新的合成任务，并更新 evaluator 配置。
        """
        self._synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]  # ⭐ Update the evaluator for each task
        

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
        索引访问，代理到底层的 RLDataset。
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call reload() or load_from_file() first.")  # ⭐ Ensures the dataset is loaded before accessing
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
        if self._dataset.num_rest_data == 0:  # ⭐ Check if there are any remaining tasks
            logger.debug("no data left")
            if self.reload() == 0:  # ⭐ Attempt to reload the dataset
                logger.debug("no task left, stop reloading and iteration")
                raise StopIteration
        return next(self._dataset)  # ⭐ Get the next task from the dataset