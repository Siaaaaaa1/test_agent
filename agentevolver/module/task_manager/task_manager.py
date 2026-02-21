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
import threading

io_lock = threading.Lock()

# --- 类型定义 ---
LEVEL_WEIGHTS = {
    "Very High": 10.0,
    "High": 5.0,
    "Medium": 2.0,
    "Low": 1.0,
    "Very Low": 0.5,
}

def get_weighted_api_sample(api_dict, k=5):
    """
    基于 Generality 等级进行加权无放回采样
    """
    apis = list(api_dict.values())
    if len(apis) <= k:
        return apis

    # 计算每个 API 的权重
    weights = []
    for api in apis:
        assessment = api.get("generality_assessment", {})
        level = assessment.get("generality_level", "Unknown")
        # 获取权重，如果 LLM 返回了不在表里的字符串，默认给个中等权重或保底
        w = LEVEL_WEIGHTS.get(level, 1.0)
        weights.append(w)

    # 执行加权无放回采样
    # 原理：利用 random.choices 依次选出不重复的元素
    sampled_apis = []
    available_apis = apis[:]
    available_weights = weights[:]

    for _ in range(k):
        if not available_apis:
            break
        # 选出一个索引
        choice_idx = random.choices(range(len(available_apis)), weights=available_weights, k=1)[0]
        sampled_apis.append(available_apis.pop(choice_idx))
        available_weights.pop(choice_idx)
    
    return sampled_apis

class TaskManagerProps(TypedDict):
    """TaskManager 的可选配置参数"""
    num_explore_threads: int  # 探索任务时的线程数
    n: int # 膨胀系数：每个种子任务期望演化出的新任务数量

class RewardProps(TypedDict):
    """奖励与评分器相关的配置"""
    original_grader: str  # 原始任务（种子）使用的评分器
    synthetic_grader: str # 合成任务（演化出的）使用的评分器

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
        self._num_exploration_threads = kwargs.get("num_explore_threads", 5)
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
        self._hindsight_file_offset = 0  # 记录读取文件的位置
        self._hindsight_file_path = self._config.task_manager.get('exploration_strategy_args', {}).get('hindsight_data_path', './tasks_explored/hindsight_supplement.jsonl')
        

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

    def load_new_hindsight_tasks(self, file_path: str = None) -> int:
        """
        尝试从 Hindsight 文件中增量加载新任务。
        
        Args:
            file_path: 文件路径，如果为 None 则使用默认路径。
            
        Returns:
            int: 新加载的任务数量。
        """
        target_path = file_path or self._hindsight_file_path
        if not os.path.exists(target_path):
            return 0

        new_tasks = []
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                # 移动到上次读取的位置
                f.seek(self._hindsight_file_offset)
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # 构造新的 Task 对象
                        # 注意：确保这里使用了正确的 grader/evaluator 配置
                        task = Task(
                            task_id=data['task_id'],
                            query=data['query'], # Hindsight 生成的 Query 已经是具体的了
                            env_type="hindsight", # 或者沿用原始 env_type，如 'appworld'
                            open_query=True, # 标记为 True 以便 Adapter 处理
                            evaluator=self._reward_config.get("synthetic_grader", "default"),
                            extra_info={"ground_truth": data.get('ground_truth')}
                        )
                        new_tasks.append(task)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid json line in hindsight file.")
                        continue
                
                # 更新文件指针位置
                self._hindsight_file_offset = f.tell()
                
        except Exception as e:
            logger.error(f"Failed to load hindsight tasks: {e}")
            return 0

        if new_tasks:
            # 将新任务加入到内部任务列表中
            self._tasks.extend(new_tasks)
            logger.info(f"🔥 [Dynamic Loading] Successfully added {len(new_tasks)} new hindsight tasks to dataset!")
            return len(new_tasks)
        
        return 0

    def _compute_tasks_hash(self, tasks: Sequence[Task]) -> str:
        """根据当前任务列表计算 MD5 哈希，用于验证断点文件是否过期"""
        task_strs = [f"{task.task_id}:{task.env_type}" for task in tasks]
        combined_str = "|".join(task_strs)
        val = hashlib.md5(combined_str.encode()).hexdigest()
        return val

    # ================= 过滤器统计与调试工具方法 (NEW) =================
    
    def _get_item_identifier(self, item: Any) -> str:
        """尝试获取任务的唯一标识符用于对比"""
        if isinstance(item, TaskObjective):
            return item.task.task_id
        elif isinstance(item, Task):
            return item.task_id
        elif isinstance(item, dict):
            return item.get("task_id", str(id(item)))
        else:
            return str(id(item))

    def _get_item_desc(self, item: Any) -> str:
        """尝试获取任务的描述（Query），用于展示被过滤的原因"""
        task = None
        if isinstance(item, TaskObjective):
            task = item.task
        elif isinstance(item, Task):
            task = item
        
        if task:
            # 优先展示 query，如果没有则展示 metadata 中的 data_id
            return f"[Query]: {task.query}" if task.query else f"[ID]: {task.task_id} (No Query)"
        return str(item)[:100]

    def _apply_filters_with_report(self, items: List[Any], filters: List[Any], stage_name: str) -> List[Any]:
        """
        替代 functools.reduce 的过滤器执行链。
        功能：执行过滤并打印统计报告，展示被过滤掉的样本详情。
        """
        if not items:
            return []
        
        current_items = items
        # 仅在非空时记录开始，避免刷屏
        if len(current_items) > 0:
            logger.info(f"🛡️ [过滤器报告 - {stage_name}] 初始数量: {len(current_items)}")

        for f in filters:
            filter_name = f.__class__.__name__
            before_count = len(current_items)
            
            # 建立索引以便查找被丢弃的项
            before_map = {self._get_item_identifier(item): item for item in current_items}
            
            # 执行过滤
            current_items = f.filter(current_items)
            
            after_count = len(current_items)
            dropped_count = before_count - after_count
            
            if dropped_count > 0:
                logger.warning(f"❌ [Filter: {filter_name}] 过滤掉了 {dropped_count} 个样本 (剩余: {after_count})")
                
                # 找出被丢弃的样本
                after_ids = set(self._get_item_identifier(item) for item in current_items)
                dropped_items = [item for uid, item in before_map.items() if uid not in after_ids]
                
                # 打印前 3 个被丢弃样本的原因（Query）
                for i, dropped in enumerate(dropped_items[:3]):
                    logger.warning(f"   -> 丢弃样本示例 #{i+1}: {self._get_item_desc(dropped)}")
                if dropped_count > 3:
                    logger.warning(f"   -> ... 以及其他 {dropped_count - 3} 个")
            else:
                logger.info(f"✅ [Filter: {filter_name}] 无损通过 (剩余: {after_count})")

        return current_items

    # =================================================================

    # --- 核心任务生成流程 ---

    def generate_task(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        生成任务的总入口：根据当前策略类型（API驱动 vs 随机采样）选择不同的执行流。
        [修改说明] 
        1. 适配 Generation-Only Mode：如果检测到 GEN_OUTPUT_DIR 环境变量，强制禁用断点恢复，并重定向输出。
        """
        # [修改] 检测纯生成模式的环境变量
        gen_output_dir = os.environ.get("GEN_OUTPUT_DIR")
        if gen_output_dir:
            logger.info(f"⚡ [Force Execution] Detected GEN_OUTPUT_DIR. Reseting resume_file to force fresh generation.")
            # 强制清空 resume_file 以忽略断点
            resume_file = None 
            # 如果是 random 策略，这里可以设置新的 checkpoint 路径（API-Driven 在内部函数处理了）
            # random 策略暂时没有很好的全路径支持，但 API-Driven 是核心
        
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
                
                # [MODIFIED] 使用带报告的过滤器替代 functools.reduce
                batch_filtered = self._apply_filters_with_report(
                    task_objectives, 
                    self._realtime_filters, 
                    f"Random-Batch-{idx}-Realtime"
                )
                
                res.extend(batch_filtered)
                
                self._old_retrival.reset()
                for j in batch_filtered:
                    self._old_retrival.add_objective(j)

                processed_indices.add(idx)
                # 4. 保存断点
                if resume_file:
                    self._save_checkpoint(resume_file, res, processed_indices, len(batch_indices), current_tasks_hash)

        return self._apply_post_filter(res)

    def _generate_task_api_driven(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        重构后的 API-Driven 生成流程：支持全链路流式写入 (Streaming) 和强制执行 (Force Execution)
        [修改说明] 增加了 verify_direct_gt 的调用
        """
        generate_task_only = self._config.task_manager.get('generate_task_only', False)
        strategy_args = self._config.task_manager.get('exploration_strategy_args', {})
        a = strategy_args.get('a', 1)
        b = strategy_args.get('b', 1)
        debug_mode = False 
        
        logger.info(f"[API-Driven] Strategy Args: a={a}, b={b}, debug_log={debug_mode}")
        if debug_mode:
            logger.warning("Debug mode enabled: forcing single thread.")

        gen_output_dir = os.environ.get("GEN_OUTPUT_DIR")
        if gen_output_dir:
            base_name = "generated_tasks"
            resume_file = os.path.join(gen_output_dir, base_name)
            logger.info(f"📂 [Isolation] Redirecting all generation outputs to: {resume_file}")
        elif resume_file is None:
            resume_file = '.generate_task_api'
        
        current_tasks_hash = self._compute_tasks_hash(tasks)
        
        def load_intermediate_tasks(path: str) -> Optional[List[Task]]:
            if os.path.exists(path):
                try:
                    tasks_list = []
                    with open(path, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    if "task" in data and "processed_indices" in data:
                                        return [Task.parse_obj(t) for t in data['tasks']]
                                    else:
                                        tasks_list.append(Task.parse_obj(data))
                                except: pass
                    logger.info(f"Loaded {len(tasks_list)} tasks from stream file {path}")
                    return tasks_list
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {path}: {e}")
            return None

        def thread_safe_append(path: str, items: List[Any]):
            if not items: return
            with io_lock:
                try:
                    with open(path, 'a', encoding='utf-8') as f:
                        for item in items:
                            obj = item.dict() if hasattr(item, 'dict') else item
                            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.error(f"Failed to append to {path}: {e}")

        # =================================================================
        # WORKER FUNCTIONS 
        # =================================================================

        def worker_generate_intra(idx: int, api_dict: dict, seed_task: Task) -> List[Task]:
            generated_tasks_list = []
            try:
                base_task = copy.deepcopy(seed_task)
                if base_task.metadata is None: base_task.metadata = {}
                base_task.metadata['thread_index'] = idx % self._num_exploration_threads
                
                tasks = self._exploration_strategy.generate_intra_task(api_dict, task=base_task)
                
                if not tasks: return []
                
                for sub_idx, current_task in enumerate(tasks):
                    data_id = f"gen_intra_{idx}_{sub_idx}"
                    current_task.metadata["data_id"] = data_id
                    generated_tasks_list.append(current_task)
                return generated_tasks_list
            except Exception as e:
                logger.error(f"[Intra-Gen] Error idx {idx}: {e}", exc_info=True)
                return []

        def worker_generate_cross(idx: int, api_dict1: dict, api_dict2: dict, seed_task: Task) -> List[Task]:
            generated_tasks_list = []
            try:
                base_task = copy.deepcopy(seed_task)
                if base_task.metadata is None: base_task.metadata = {}
                base_task.metadata['thread_index'] = idx % self._num_exploration_threads
                
                tasks = self._exploration_strategy.generate_cross_task(api_dict1=api_dict1, api_dict2=api_dict2, task=base_task)
                
                if not tasks: return []

                for sub_idx, current_task in enumerate(tasks):
                    data_id = f"gen_cross_{idx}_{sub_idx}"
                    current_task.metadata["data_id"] = data_id
                    generated_tasks_list.append(current_task)
                return generated_tasks_list
            except Exception as e:
                logger.error(f"[Cross-Gen] Error idx {idx}: {e}", exc_info=True)
                return []

        def worker_explore_intra(task: Task) -> List[TaskObjective]:
            try:
                data_id = task.metadata.get("data_id", f"unknown_{random.randint(0,1000)}")
                logger.info(f"[Intra-Explore] Exploring {data_id}...")
                
                # 1. 执行探索
                trajectories = self._exploration_strategy.explore(task, data_id, data_id)
                
                # 筛选成功轨迹
                success_traj = None
                if trajectories and trajectories[0].reward and trajectories[0].reward.outcome >= 0.7:
                    success_traj = trajectories[0]

                if not success_traj:
                    return []

                reward_val = success_traj.reward.outcome
                # [关键] 获取原始轨迹步骤
                # 使用 dict() 序列化防止引用问题，保存完整的原始执行流
                raw_gt_steps = [s.dict() if hasattr(s, 'dict') else s for s in success_traj.steps]

                # =========================================================
                # Step 1: 尝试生成 Direct Verified Pair (Refined Code)
                # =========================================================
                direct_verified_obj = self._exploration_strategy.verify_direct_gt(task, success_traj)
                
                # [核心修改] 定义 Evolved 阶段需要的“起源”信息
                # 1. origin_query 始终是当前的 task.query
                origin_query_for_evolved = task.query
                # 2. origin_gt 默认为 None (因为你说“最最初始的task.ground_truth需要丢弃”，它是空的)
                origin_gt_for_evolved = None 

                if direct_verified_obj:
                    # --- A. 保存 Direct 结果 ---
                    direct_verified_obj.task.raw_trajectory = raw_gt_steps
                    
                    # Direct 任务本身的 Origin GT 也是 None (因为它来自种子)
                    direct_verified_obj.task.origin_ground_truth = None
                    direct_verified_obj.task.origin_query = task.query
                    
                    direct_verified_obj.task.metadata["source_data_id"] = data_id
                    direct_verified_obj.task.metadata["execution_reward"] = {"outcome": reward_val}
                    
                    logger.info(f"✅ [Intra] Direct GT Verified for {data_id}")
                    thread_safe_append(intra_direct_path, [direct_verified_obj])

                    # --- B. 更新 Evolved 的 Origin GT ---
                    # 只有当验证成功，我们才拥有一个“起源 GT”。
                    # 将这个 Direct 验证出的 Refined Code 传给 Evolved 任务作为 Origin。
                    origin_gt_for_evolved = direct_verified_obj.task.ground_truth 
                else:
                    logger.info(f"⚠️ [Intra] Direct GT Verification Failed for {data_id}. Origin GT will be None.")

                # =========================================================
                # Step 2: 生成 Evolved Pair (New Query + New GT)
                # =========================================================
                evolved_results = self._exploration_strategy.summarize(task, success_traj)

                if evolved_results:
                    for res in evolved_results:
                        # 1. 填充 Raw Trajectory (始终携带原始轨迹)
                        res.task.raw_trajectory = raw_gt_steps
                        
                        # 2. 填充 Origin 信息
                        # 这里使用了上面计算好的变量：
                        # - 如果 Direct Verify 成功：Origin GT = Refined Code
                        # - 如果 Direct Verify 失败：Origin GT = None (初始 GT 被丢弃)
                        res.task.origin_ground_truth = origin_gt_for_evolved
                        res.task.origin_query = origin_query_for_evolved
                        
                        # 3. 填充元数据
                        res.confidence = 0  # 初始置信度
                        res.reward = reward_val
                        res.task.metadata.update({
                            "data_pair_type": "evolved",
                            "source_data_id": data_id,
                            # 标记该演化任务是否基于一个已验证的代码
                            "has_verified_origin": (origin_gt_for_evolved is not None)
                        })
                    
                    thread_safe_append(intra_evolved_path, evolved_results)

                return evolved_results if evolved_results else []

            except Exception as e:
                logger.error(f"[Intra-Explore] Error: {e}", exc_info=True)
                return []

        def worker_explore_cross(task: Task) -> List[TaskObjective]:
            try:
                data_id = task.metadata.get("data_id", f"unknown_{random.randint(0,1000)}")
                logger.info(f"[Cross-Explore] Exploring {data_id}...")
                
                # 1. 执行探索
                trajectories = self._exploration_strategy.explore(task, data_id, data_id)
                
                # 筛选成功轨迹 (Reward >= 0.7)
                success_traj = None
                if trajectories and trajectories[0].reward and trajectories[0].reward.outcome >= 0.7:
                    success_traj = trajectories[0]

                if not success_traj:
                    return []

                reward_val = success_traj.reward.outcome
                # [关键] 获取原始轨迹步骤 (Raw Steps)
                # 使用 dict() 序列化防止引用问题
                raw_gt_steps = [s.dict() if hasattr(s, 'dict') else s for s in success_traj.steps]

                # =========================================================
                # Step 1: 尝试生成 Direct Verified Pair (Refined Code)
                # =========================================================
                direct_verified_obj = self._exploration_strategy.verify_direct_gt(task, success_traj)
                
                # [核心修改] 定义 Evolved 阶段需要的“起源”信息
                # 1. origin_query 始终是当前的 task.query
                origin_query_for_evolved = task.query
                # 2. origin_gt 默认为 None (丢弃最原始的空 GT)
                origin_gt_for_evolved = None 

                if direct_verified_obj:
                    # --- A. 保存 Direct 结果 ---
                    # 1. 填充 Raw Trajectory
                    direct_verified_obj.task.raw_trajectory = raw_gt_steps
                    
                    # Direct 任务本身的 Origin GT 也是 None (因为它来自种子)
                    direct_verified_obj.task.origin_ground_truth = None
                    direct_verified_obj.task.origin_query = task.query
                    
                    direct_verified_obj.task.metadata["source_data_id"] = data_id
                    direct_verified_obj.task.metadata["execution_reward"] = {"outcome": reward_val}
                    
                    logger.info(f"✅ [Cross] Direct GT Verified for {data_id}")
                    thread_safe_append(cross_direct_path, [direct_verified_obj])

                    # --- B. 更新 Evolved 的 Origin GT ---
                    # 只有当验证成功，将 Refined Code 传给 Evolved 任务作为 Origin
                    origin_gt_for_evolved = direct_verified_obj.task.ground_truth
                else:
                    logger.info(f"⚠️ [Cross] Direct GT Verification Failed for {data_id}. Origin GT will be None.")

                # =========================================================
                # Step 2: 生成 Evolved Pair (New Query + New GT)
                # =========================================================
                evolved_results = self._exploration_strategy.summarize(task, success_traj)

                if evolved_results:
                    for res in evolved_results:
                        # 1. 填充 Raw Trajectory (始终携带原始轨迹)
                        res.task.raw_trajectory = raw_gt_steps
                        
                        # 2. 填充 Origin 信息
                        # 逻辑：如果 Direct 成功，则继承 Refined Code；否则为 None
                        res.task.origin_ground_truth = origin_gt_for_evolved
                        res.task.origin_query = origin_query_for_evolved
                        
                        # 3. 填充元数据
                        res.confidence = 0 # 初始置信度
                        res.reward = reward_val
                        res.task.metadata.update({
                            "data_pair_type": "evolved",
                            "source_data_id": data_id,
                            # 标记该演化任务是否基于一个已验证的代码
                            "has_verified_origin": (origin_gt_for_evolved is not None)
                        })
                    
                    thread_safe_append(cross_evolved_path, evolved_results)

                return evolved_results if evolved_results else []

            except Exception as e:
                logger.error(f"[Cross-Explore] Error: {e}", exc_info=True)
                return []
            
        # 获取基础数据
        api_knowledge = getattr(self._exploration_strategy, 'api_knowledge', {})
        active_apps_set = getattr(self._exploration_strategy, 'active_apps', set(api_knowledge.keys()))

        # 定义文件路径模板 (改为 .jsonl)
        intra_gen_path = f"{resume_file}.intra.generated.jsonl"
        intra_filtered_path = f"{resume_file}.intra.filtered.jsonl"
        intra_final_path = f"{resume_file}.intra.jsonl"
        
        cross_gen_path = f"{resume_file}.cross.generated.jsonl"
        cross_filtered_path = f"{resume_file}.cross.filtered.jsonl"
        cross_final_path = f"{resume_file}.extra.jsonl"

        # [新增] 定义两对数据的保存路径
        # 1. Direct Pair (Original Query + Trajectory)
        intra_direct_path = f"{resume_file}.intra.direct.jsonl"
        cross_direct_path = f"{resume_file}.cross.direct.jsonl"
        
        # 2. Evolved Pair (New Query + Trajectory) - 这里复用 final_path 或者定义新的
        intra_evolved_path = f"{resume_file}.intra.evolved.jsonl" 
        cross_evolved_path = f"{resume_file}.cross.evolved.jsonl"
        

        # =================================================================
        # PART 1: INTRA-DOMAIN (生成 -> 过滤)
        # =================================================================
        logger.info("=== Starting PART 1: Intra-Domain Generation & Filtering ===")
        
        # 1.1 准备 Intra API 组合 (保持原样)
        api_list = []
        for app_name in sorted(active_apps_set):
            if app_name not in api_knowledge: continue
            apis = api_knowledge[app_name].get("apis", [])
            if not apis: continue
            sample_count = min(len(apis), 3)
            for _ in range(len(apis)):
                selected_apis = random.sample(list(apis.values()), sample_count)
                this_turn_apis = [api["call_name"] for api in selected_apis]
                api_list.append({"app_name":app_name, "apis_name_list":this_turn_apis})

        random.shuffle(api_list)
        intra_task_pool = (list(copy.copy(tasks)) * int(a + 1))[:int(len(tasks) * a)]
        if debug_mode: intra_task_pool = intra_task_pool[:1]

        target_len_intra = len(intra_task_pool)
        
        if len(api_list) > 0 and target_len_intra > 0:
            repeat_factor = (target_len_intra // len(api_list)) + 1
            api_list = (api_list * repeat_factor)[:target_len_intra]
        else:
            target_len_intra = 0
        total_intra = target_len_intra
        
        # --- Step 1.1: Intra Generation (流式增量生成) ---
        generated_intra_tasks = load_intermediate_tasks(intra_gen_path)
        if generated_intra_tasks is None:
            generated_intra_tasks = []

        current_count = len(generated_intra_tasks)
        if current_count < total_intra:
            needed = total_intra - current_count
            logger.info(f"[Intra-Gen] Generating {needed} tasks (Streaming Mode)...")
            
            with ThreadPoolExecutor(max_workers=1 if debug_mode else self._num_exploration_threads) as pool:
                futures = []
                for idx in range(current_count, total_intra):
                    if idx >= len(api_list): break
                    futures.append(pool.submit(worker_generate_intra, idx, api_list[idx], intra_task_pool[idx]))
                
                for f in tqdm(as_completed(futures), total=len(futures), desc="Intra Generation", disable=not show_progress):
                    try:
                        res = f.result()
                        if res: 
                            res_list = res if isinstance(res, list) else [res]
                            # [修改] 立即流式保存到磁盘
                            thread_safe_append(intra_gen_path, res_list)
                            generated_intra_tasks.extend(res_list)
                    except Exception as e:
                        logger.error(f"[Intra-Gen] Future result error: {e}")
            # 不再需要 save_intermediate_tasks (整体 dump)，因为已经流式写入了
        else:
            logger.info(f"[Intra-Gen] Skipped (Loaded {current_count} tasks)")

        # --- Step 1.2: Intra Filtering (流式增量过滤) ---
        filtered_intra_tasks = load_intermediate_tasks(intra_filtered_path)
        if filtered_intra_tasks is None:
            filtered_intra_tasks = []

        # [新增] 检查过滤比例，超过 50% 则跳过
        if len(generated_intra_tasks) > 0 and (len(filtered_intra_tasks) / len(generated_intra_tasks) > 0.5):
            logger.info(f"[Intra-Filter] Skipped: Filtered tasks ({len(filtered_intra_tasks)}) > 50% of generated ({len(generated_intra_tasks)}).")
        else:
            # 找出哪些 generated 任务还没被 filter 处理
            filtered_ids = set()
            for t in filtered_intra_tasks:
                if t.metadata and "data_id" in t.metadata:
                    filtered_ids.add(t.metadata["data_id"])

            pending_filter_tasks = []
            for t in generated_intra_tasks:
                t_id = t.metadata.get("data_id")
                if not t_id or t_id not in filtered_ids:
                    pending_filter_tasks.append(t)

            if pending_filter_tasks:
                logger.info(f"[Intra-Filter] Filtering {len(pending_filter_tasks)} tasks...")
                
                # 执行过滤
                newly_filtered = self._apply_filters_with_report(
                    pending_filter_tasks, 
                    self.api_llm_pre_filter, 
                    "Intra-Pre-Filter-Incremental"
                )
                
                # [修改] 立即追加保存过滤后的结果
                thread_safe_append(intra_filtered_path, newly_filtered)
                filtered_intra_tasks.extend(newly_filtered)
            else:
                logger.info("[Intra-Filter] All tasks filtered.")

        # =================================================================
        # PART 2: CROSS-DOMAIN (迭代式生成 -> 过滤)
        # =================================================================
        logger.info("=== Starting PART 2: Cross-Domain Generation & Filtering (Target-Driven) ===")

        # ------------------------------------------------------------------
        # 2.1 准备基础数据池 (Source Pools)
        # ------------------------------------------------------------------
        
        # A. 筛选种子任务 (只保留符合 valid_apps_list 的任务)
        valid_apps_list = [app for app in sorted(active_apps_set) if app in api_knowledge and api_knowledge[app].get("apis")]
        valid_apps_set = set(valid_apps_list)
        
        valid_candidate_tasks = [
            t for t in tasks 
            if getattr(t, 'app', None) in valid_apps_set or getattr(t, 'app_name', None) in valid_apps_set
        ]
        
        # 兜底：如果没有符合条件的种子任务，使用全部任务
        if not valid_candidate_tasks:
            logger.warning(f"[Cross-Gen] No tasks matched valid apps. Using all {len(tasks)} tasks as candidates.")
            valid_candidate_tasks = tasks
        
        # B. 准备足够的 API 组合 (生成一个较大的池子，后续循环采样)
        base_pair_data = []
        for app_name_a in valid_apps_list:
            apis_a_all = api_knowledge[app_name_a].get("apis", {})
            other_apps = [x for x in valid_apps_list if x != app_name_a]
            if not other_apps: continue
            random.shuffle(other_apps)
            
            # 这里可以适当增加组合密度
            loop_count = max(len(apis_a_all), len(other_apps)) 
            for i in range(loop_count):
                app_name_b = other_apps[i % len(other_apps)]
                apis_b_all = api_knowledge[app_name_b].get("apis", {})
                s_a = get_weighted_api_sample(apis_a_all, k=5)
                s_b = get_weighted_api_sample(apis_b_all, k=5)
                base_pair_data.append([
                    {"app_name": app_name_a, "apis_name_list": [x["call_name"] for x in s_a]},
                    {"app_name": app_name_b, "apis_name_list": [x["call_name"] for x in s_b]}
                ])
        
        if not base_pair_data:
            logger.error("[Cross-Gen] No valid API pairs generated. Aborting Part 2.")
            target_valid_count = 0
        else:
            random.shuffle(base_pair_data)
            logger.info(f"[Cross-Gen] Prepared {len(base_pair_data)} base API pairs.")

        # ------------------------------------------------------------------
        # 2.2 设定目标与加载状态
        # ------------------------------------------------------------------
        target_valid_count = int(len(tasks) * b)
        if debug_mode: target_valid_count = 1
        
        # 加载已生成的 raw tasks (用于避免 ID 冲突或统计)
        generated_cross_tasks = load_intermediate_tasks(cross_gen_path)
        if generated_cross_tasks is None: generated_cross_tasks = []
        
        # 加载已过滤的 valid tasks (这是我们要达标的计数器)
        filtered_cross_tasks = load_intermediate_tasks(cross_filtered_path)
        if filtered_cross_tasks is None: filtered_cross_tasks = []

        current_valid_count = len(filtered_cross_tasks)
        logger.info(f"[Cross-Loop] Target Valid: {target_valid_count} | Current Valid: {current_valid_count}")

        # ------------------------------------------------------------------
        # 2.3 闭环迭代 (Generate -> Filter -> Check)
        # ------------------------------------------------------------------
        max_loop_attempts = 10 # 防止无限循环（如果过滤通过率极低）
        loop_idx = 0
        
        # 定义一个全局的生成索引，延续之前的计数
        global_gen_idx = len(generated_cross_tasks)

        while current_valid_count < target_valid_count and loop_idx < max_loop_attempts:
            loop_idx += 1
            needed = target_valid_count - current_valid_count
            
            # [策略] 设定本轮生成的 Raw 任务数量
            # 为了减少轮次，假设通过率是 50%，所以我们生成 needed * 2 的数量
            # 或者是 1.5 倍，最小生成一批 (例如 num_threads * 2) 以利用并发
            oversample_factor = 2.0 
            batch_size = int(needed * oversample_factor)
            batch_size = max(batch_size, self._num_exploration_threads * 2) # 保证并发度
            
            if debug_mode: batch_size = 1
            
            logger.info(f"--- [Cross-Loop {loop_idx}] Needed: {needed}, Planning to generate batch: {batch_size} ---")

            # --- A. 准备本轮 Batch 数据 ---
            batch_pairs = []
            batch_seeds = []
            
            # 循环采样 API Pairs 和 Seed Tasks
            for i in range(batch_size):
                batch_pairs.append(base_pair_data[i % len(base_pair_data)])
                batch_seeds.append(valid_candidate_tasks[i % len(valid_candidate_tasks)])
            
            # --- B. 执行生成 (Batch Generation) ---
            newly_generated_batch = []
            with ThreadPoolExecutor(max_workers=1 if debug_mode else self._num_exploration_threads) as pool:
                futures = []
                for i in range(batch_size):
                    # 使用 global_gen_idx 保证 ID 唯一且递增
                    current_idx = global_gen_idx + i 
                    futures.append(pool.submit(
                        worker_generate_cross, 
                        current_idx, 
                        batch_pairs[i][0], 
                        batch_pairs[i][1], 
                        batch_seeds[i]
                    ))
                
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Cross Gen (Round {loop_idx})", disable=not show_progress):
                    try:
                        res = f.result()
                        if res: 
                            res_list = res if isinstance(res, list) else [res]
                            thread_safe_append(cross_gen_path, res_list)
                            newly_generated_batch.extend(res_list)
                    except Exception as e:
                        logger.error(f"[Cross-Gen] Batch error: {e}")
            
            global_gen_idx += batch_size # 更新全局索引
            generated_cross_tasks.extend(newly_generated_batch) # 更新内存记录
            
            if not newly_generated_batch:
                logger.warning(f"[Cross-Loop {loop_idx}] No tasks generated in this batch. Trying next round...")
                continue

            # --- C. 执行过滤 (Batch Filtering) ---
            logger.info(f"[Cross-Filter] Filtering batch of {len(newly_generated_batch)} tasks...")
            
            newly_filtered_batch = self._apply_filters_with_report(
                newly_generated_batch, 
                self.api_llm_pre_filter, 
                f"Cross-Filter-Round-{loop_idx}"
            )
            
            # 保存过滤结果
            if newly_filtered_batch:
                thread_safe_append(cross_filtered_path, newly_filtered_batch)
                filtered_cross_tasks.extend(newly_filtered_batch)
            
            # --- D. 更新状态 ---
            current_valid_count = len(filtered_cross_tasks)
            pass_rate = len(newly_filtered_batch) / len(newly_generated_batch) if newly_generated_batch else 0
            logger.info(f"[Cross-Loop {loop_idx}] Batch Pass Rate: {pass_rate:.1%}. Total Valid: {current_valid_count}/{target_valid_count}")

            if current_valid_count >= target_valid_count:
                logger.info("[Cross-Loop] Target reached!")
                break
        
        if current_valid_count < target_valid_count:
            logger.warning(f"[Cross-Loop] Max attempts ({max_loop_attempts}) reached. Final count: {current_valid_count}/{target_valid_count}")
        else:
            logger.info(f"[Cross-Filter] Successfully collected {len(filtered_cross_tasks)} valid tasks.")


        # =================================================================
        # PART 3: INTRA-DOMAIN (探索)
        # =================================================================
        logger.info("=== Starting PART 3: Intra-Domain Exploration ===")
        
        # 3.1 尝试加载结果 (流式读取) 并构建已探索 ID 集合
        intra_res = []
        explored_ids_intra = set()
        
        if os.path.exists(intra_final_path):
             try:
                 with open(intra_final_path, 'r', encoding='utf-8') as f:
                     for line in f:
                         if line.strip(): 
                             data = json.loads(line)
                             intra_res.append(data)
                             # [新增] 尝试从结果中提取原始任务的 data_id 以便去重
                             # 假设结果结构包含 task 字段，或者根据实际 TaskObjective 结构调整
                             try:
                                 # 优先检查 obj['task']['metadata']['data_id']
                                 t_data = data.get("task", {})
                                 if "metadata" in t_data:
                                     mid = t_data["metadata"].get("data_id")
                                     if mid: explored_ids_intra.add(mid)
                             except: pass
                 logger.info(f"[Intra-Explore] Loaded {len(intra_res)} objectives.")
             except Exception as e:
                 logger.warning(f"[Intra-Explore] Failed to load existing results: {e}")

        # 3.2 检查是否跳过 (Skip Logic)
        # 如果已有结果数量占总任务数的比例超过 50%，则认为已跑完，跳过
        if len(filtered_intra_tasks) > 0 and (len(intra_res) / len(filtered_intra_tasks) > 0.5):
            logger.info(f"[Intra-Explore] Skipped: Explored tasks ({len(intra_res)}) > 50% of candidates ({len(filtered_intra_tasks)}).")
        else:
            # 找出哪些任务还没被探索 (计算差集)
            pending_explore_intra = []
            for t in filtered_intra_tasks:
                t_id = t.metadata.get("data_id")
                # 如果没有 ID (异常) 或者 ID 不在已探索集合中，则需要执行
                if not t_id or t_id not in explored_ids_intra:
                    pending_explore_intra.append(t)

            if pending_explore_intra:
                logger.info(f"[Intra-Explore] Exploring {len(pending_explore_intra)} tasks (Incremental)...")
                
                with ThreadPoolExecutor(max_workers=1 if debug_mode else self._num_exploration_threads) as pool:
                    futures = {pool.submit(worker_explore_intra, t): i for i, t in enumerate(pending_explore_intra)}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Intra Exploration", disable=not show_progress):
                        try:
                            objs = future.result()
                            filtered_objs = self._apply_filters_with_report(objs, self._realtime_filters, "Intra-Worker-Realtime")
                            if filtered_objs:
                                # [修改] 立即流式保存探索结果
                                thread_safe_append(intra_final_path, filtered_objs)
                                intra_res.extend(filtered_objs)
                        except Exception as e:
                            logger.error(f"Error in exploration future: {e}")
            else:
                logger.info("[Intra-Explore] All tasks already covered by existing results.")

        logger.info(f"[Intra-Domain] Completed. Collected {len(intra_res)} objectives.")


        # =================================================================
        # PART 4: CROSS-DOMAIN (探索)
        # =================================================================
        logger.info("=== Starting PART 4: Cross-Domain Exploration ===")

        cross_res = []
        explored_ids_cross = set()

        if os.path.exists(cross_final_path):
             try:
                 with open(cross_final_path, 'r', encoding='utf-8') as f:
                     for line in f:
                         if line.strip(): 
                             data = json.loads(line)
                             cross_res.append(data)
                             try:
                                 t_data = data.get("task", {})
                                 if "metadata" in t_data:
                                     mid = t_data["metadata"].get("data_id")
                                     if mid: explored_ids_cross.add(mid)
                             except: pass
                 logger.info(f"[Cross-Explore] Loaded {len(cross_res)} objectives.")
             except Exception as e:
                 logger.warning(f"[Cross-Explore] Failed to load existing results: {e}")

        # 4.2 检查是否跳过 (Skip Logic)
        if len(filtered_cross_tasks) > 0 and (len(cross_res) / len(filtered_cross_tasks) > 0.5):
            logger.info(f"[Cross-Explore] Skipped: Explored tasks ({len(cross_res)}) > 50% of candidates ({len(filtered_cross_tasks)}).")
        else:
            pending_explore_cross = []
            for t in filtered_cross_tasks:
                t_id = t.metadata.get("data_id")
                if not t_id or t_id not in explored_ids_cross:
                    pending_explore_cross.append(t)
            
            if pending_explore_cross:
                logger.info(f"[Cross-Explore] Exploring {len(pending_explore_cross)} tasks (Incremental)...")
                with ThreadPoolExecutor(max_workers=1 if debug_mode else self._num_exploration_threads) as pool:
                    futures = {pool.submit(worker_explore_cross, t): i for i, t in enumerate(pending_explore_cross)}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Cross Exploration", disable=not show_progress):
                        try:
                            objs = future.result()
                            filtered_objs = self._apply_filters_with_report(objs, self._realtime_filters, "Cross-Worker-Realtime")
                            if filtered_objs:
                                # [修改] 立即流式保存探索结果
                                thread_safe_append(cross_final_path, filtered_objs)
                                cross_res.extend(filtered_objs)
                        except Exception as e:
                            logger.error(f"Error in cross exploration future: {e}")
            else:
                logger.info("[Cross-Explore] All tasks already covered by existing results.")

        logger.info(f"[Cross-Domain] Completed. Collected {len(cross_res)} objectives.")

        # =================================================================
        # Final Merge
        # =================================================================
        # 由于上面逻辑为了流式写入修改了变量，这里需要重新构造对象列表以便后续 Post-Filter
        # 注意：这里的 intra_res 可能包含 dict (从文件读的) 或对象 (新生成的)
        # 为了兼容 _apply_post_filter，需要统一为 TaskObjective 对象
        total_results = []
        for item in intra_res + cross_res:
            if isinstance(item, TaskObjective):
                total_results.append(item)
            else:
                try:
                    total_results.append(TaskObjective.parse_obj(item))
                except: pass

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
        
        # [MODIFIED] 先应用实时过滤器（确保最终一致性）- 带报告
        res = self._apply_filters_with_report(res, self._realtime_filters, "PostProcess-Realtime")
        
        logger.info("正在对生成的任务进行后置过滤（Post-Filter）...")
        cnt_before = len(res)
        
        # [MODIFIED] 应用后置过滤器 - 带报告
        res = self._apply_filters_with_report(res, self._post_filter, "PostProcess-LLM")
        
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

    def update(self):
        """
        Manually triggers the rebuilding of the dataset.

        This method first checks if there are any synthetic objectives available. If not, it logs a warning suggesting
        that `load_from_file()` or `reload()` should be called first. It then rebuilds the dataset and logs an
        informational message upon completion.

        Returns:
            None
        """
        if not self._synthetic_objectives:
            logger.warning("No synthetic objectives available, did you call load_from_file() or reload() first?")
        self._rebuild_dataset()  # ⭐ Rebuilds the dataset
        logger.info("Dataset updated manually via update().")


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

        # ls = self._manager.generate_task(delta)
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