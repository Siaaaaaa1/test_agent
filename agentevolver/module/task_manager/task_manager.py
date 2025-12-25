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
    Callable, Iterable, NotRequired, Optional, Sequence, 
    TypedDict, Unpack, List, Dict, Any
)

import hydra
from loguru import logger
from omegaconf import DictConfig
import requests
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

# 内部模块引入
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

def get_exploration_strategy(name: str, strategy_args, *, tokenizer, config) -> TaskExploreStrategy:
    """
    探索策略工厂函数：根据配置名称返回具体的策略实例。
    - random: 随机采样策略
    - api_driven: 基于 API 知识库的链式探索策略
    """
    logger.info(f"loading exploration strategy {name}")
    if name == "random":
        return LlmRandomSamplingExploreStrategy(tokenizer=tokenizer, config=config, **strategy_args)
    elif name == "api_driven":
        return ApiDrivenExploreStrategy(tokenizer=tokenizer, config=config, **strategy_args)
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
            config=config
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
        
        # 3. 为策略注入依赖：将检索器、LLM 和环境配置文件传给策略内部使用
        self._exploration_strategy._inject_deps(
            self._old_retrival,
            self._llm_client,
            DashScopeClient(model_name='qwen3-235b-a22b-instruct-2507', max_tokens=8192), # 用于辅助任务的额外 Client
            env_profile=env_profile
        )

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
        """
        API Driven 策略：两阶段池化生成模式。
        a, b 来源于 config.task_manager.exploration_strategy_args
        第一阶段：单域池 (active_apps) 扩大 n * a 倍。
        第二阶段：跨域池 (seed_tasks) 扩大 n * b 倍。
        """
        # 1. 从配置中读取超参 a 和 b
        # 假设配置路径为 task_manager.exploration_strategy_args
        strategy_args = self._config.task_manager.get('exploration_strategy_args', {})
        a = strategy_args.get('a', 1)  # 单域倍数，默认1
        b = strategy_args.get('b', 1)  # 跨域倍数，默认1
        n = self._n # 基础膨胀系数
        
        if resume_file is None:
            resume_file = '.generate_task_api'
        
        intra_ckpt_path = f"{resume_file}.intra.json"
        cross_ckpt_path = f"{resume_file}.extra.json"
        current_tasks_hash = self._compute_tasks_hash(tasks)
        
        # --- 阶段 1: 单域探索 (Intra-Domain) ---
        active_apps = list(self._exploration_strategy.active_apps)
        # 定义单域任务池：扩大 n * a 倍
        intra_pool = active_apps * (n * a)
        intra_res = []
        intra_processed_idx = set()
        
        # 恢复单域断点
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

        # 单域生成循环
        if len(intra_processed_idx) < len(intra_pool):
            pbar = tqdm(total=len(intra_pool), desc="Stage 1: Intra-Domain (a)", disable=not show_progress)
            pbar.update(len(intra_processed_idx))
            
            for idx, app_name in enumerate(intra_pool):
                if idx in intra_processed_idx:
                    continue
                
                # 传入指定的 app_name
                task = self._exploration_strategy.generate_intra_task(app_name=app_name)
                if task:
                    new_obj = self._explore_and_summarize_intra(task)
                    if new_obj:
                        intra_res.extend(new_obj)
                
                intra_processed_idx.add(idx)
                pbar.update(1)
                
                # 实时过滤并保存断点
                intra_res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, intra_res)
                self._save_checkpoint(intra_ckpt_path, intra_res, intra_processed_idx, len(intra_pool), current_tasks_hash)
            pbar.close()

        # --- 阶段 2: 跨域合成 (Cross-Domain) ---
        # 只有单域任务全部处理完（idx 达到 n*a 种组合）才进行
        cross_res = []
        if len(intra_processed_idx) >= len(intra_pool):
            # 定义跨域任务池：种子任务扩大 n * b 倍
            cross_pool = list(tasks) * (n * b)
            cross_processed_idx = set()
            
            if os.path.exists(cross_ckpt_path):
                try:
                    with open(cross_ckpt_path, 'r') as f:
                        cp = json.load(f)
                        cross_res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in cp.get('results', [])]
                        cross_processed_idx = {int(i) for i in cp.get('processed_indices', [])}
                        logger.info(f"Cross-Domain resumed: {len(cross_res)} tasks loaded.")
                except Exception: pass

            pbar = tqdm(total=len(cross_pool), desc="Stage 2: Cross-Domain (b)", disable=not show_progress)
            pbar.update(len(cross_processed_idx))
            
            for idx, seed_task in enumerate(cross_pool):
                if idx in cross_processed_idx:
                    continue
                
                # 生成跨域任务，并绑定到种子环境 ID 上
                task = self._exploration_strategy.generate_cross_task()
                if task:
                    task.task_id = seed_task.task_id
                    new_obj = self._explore_and_summarize_extra(task)
                    if new_obj:
                        cross_res.extend(new_obj)
                
                cross_processed_idx.add(idx)
                pbar.update(1)
                
                # 实时过滤与检索库更新（混合两部分结果去重）
                current_batch_all = intra_res + cross_res
                current_batch_all = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, current_batch_all)
                
                # 这里的 cross_res 应该是过滤后剔除 intra 部分的增量
                # 简单处理：更新检索库
                self._old_retrival.reset()
                for j in current_batch_all: self._old_retrival.add_objective(j)
                
                self._save_checkpoint(cross_ckpt_path, cross_res, cross_processed_idx, len(cross_pool), current_tasks_hash)
            pbar.close()
        
        # 合并结果并应用最终的后置质量过滤
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

    def _explore_and_summarize_intra(self, task: Task) -> List[TaskObjective]:
        """单域任务的具体执行链路：执行 Agent -> 调用策略总结"""
        trajectory = self._step_explore_intra(task)
        if not trajectory or not trajectory.steps: return []
        objective = self._step_summarize_intra(task, trajectory)
        return [objective] if objective else []

    def _explore_and_summarize_extra(self, task: Task) -> List[TaskObjective]:
        """跨域任务的具体执行链路：执行 Agent（含预埋数据）-> 验证总结"""
        trajectory = self._step_explore_extra(task) 
        if not trajectory or not trajectory.steps: return []
        objective = self._step_summarize_extra(task, trajectory)
        return [objective] if objective else []

    def _step_explore_intra(self, task: Task) -> Trajectory:
        logger.info(f"[TaskManager] 正在执行单域探索: {task.instruction[:50]}...")
        return self._execute_agent_loop(task)

    def _step_explore_extra(self, task: Task) -> Trajectory:
        logger.info(f"[TaskManager] 正在执行跨域探索: {task.instruction[:50]}...")
        return self._execute_agent_loop(task)

    def _step_summarize_intra(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        """调用 API 策略的单域总结逻辑（反向翻译）"""
        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            return self._exploration_strategy.summarize_intra(task, trajectory)
        return None

    def _step_summarize_extra(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        """调用 API 策略的跨域总结逻辑（流验证）"""
        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            return self._exploration_strategy.summarize_cross(task, trajectory)
        return None

    def _execute_agent_loop(self, task: Task) -> Trajectory:
        """
        通用的 Agent 执行循环。
        1. 重置环境；2. 注入数据（如果是跨域）；3. 运行 AgentFlow；4. 返回执行轨迹。
        """
        if not self.env_worker or not self.agent_flow:
            logger.error("EnvWorker 或 AgentFlow 未初始化，无法执行 Agent。")
            return Trajectory(steps=[])

        # 获取真实的沙箱 ID 作为环境锚点
        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            real_sandbox_id = self._exploration_strategy.get_next_sandbox_id()
            if real_sandbox_id: task.task_id = real_sandbox_id
        
        # 1. 环境重置
        try:
            self.env_worker.reset(task)
        except Exception as e:
            logger.error(f"环境重置失败: {e}")
            return Trajectory(steps=[])

        # 2. 跨域数据预埋 (Injection)
        if task.metrics and task.metrics.get("setup_action") == "inject_data":
            self._inject_context(task.metrics)

        # 3. 驱动 Agent 按照设定的 Instruction 行动
        try:
            trajectory = self.agent_flow.run(task, self.env_worker)
            return trajectory
        except Exception as e:
            logger.error(f"AgentFlow 执行报错: {e}")
            return Trajectory(steps=[])

    def _inject_context(self, metrics: Dict):
        """向环境 Workers 注入初始数据（例如在邮件里预埋一条订单信息）"""
        try:
            app = metrics.get("app")
            content = metrics.get("content")
            # 根据不同的环境 Worker 接口执行“上帝命令”注入数据
            if hasattr(self.env_worker, "execute_god_command"):
                logger.info(f"[数据注入] App: {app}, 内容: {content}")
                # self.env_worker.execute_god_command(...)
            elif hasattr(self.env_worker, "execute"):
                pass
            else:
                logger.warning("EnvWorker 不支持数据注入接口。")
        except Exception as e:
            logger.warning(f"数据注入失败: {e}")


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