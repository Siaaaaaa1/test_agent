from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import functools
import hashlib
import json
import os
import random
import time
import threading
from typing import (
    Optional, Sequence, TypedDict, Unpack, List, Any, Iterable
)

from loguru import logger
from omegaconf import DictConfig
import requests
import numpy as np  # 引入 numpy 以优化采样性能
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
import uuid

# 全局 IO 锁，用于在多线程环境下安全地追加写入文件
io_lock = threading.Lock()

# --- 类型定义 ---
# 定义不同泛化等级（Generality）对应的权重字典，用于控制采样概率
LEVEL_WEIGHTS = {
    "Very High": 10.0,
    "High": 5.0,
    "Medium": 2.0,
    "Low": 1.0,
    "Very Low": 0.5,
}

def get_weighted_api_sample(api_dict, k=5):
    """
    基于 Generality 等级进行加权无放回采样，获取 API 子集。
    通过将大模型评估的 API 泛化等级映射为数值权重，利用 numpy 进行概率分布采样。

    Args:
        api_dict (dict): 包含 API 信息的字典。
        k (int, optional): 需要采样的 API 数量，默认为 5。

    Returns:
        list: 采样出的 API 详情列表。
    """
    apis = list(api_dict.values())
    if len(apis) <= k:
        return apis

    weights = []
    for api in apis:
        assessment = api.get("generality_assessment", {})
        level = assessment.get("generality_level", "Unknown")
        w = LEVEL_WEIGHTS.get(level, 1.0)
        weights.append(w)

    weights_arr = np.array(weights)
    probs = weights_arr / weights_arr.sum() 
    chosen_indices = np.random.choice(len(apis), size=k, replace=False, p=probs)
    
    return [apis[i] for i in chosen_indices]

class TaskManagerProps(TypedDict):
    """TaskManager 的可选配置参数"""
    num_explore_threads: int  # 探索任务时的并发线程数
    n: int # 膨胀系数：每个种子任务期望演化出的新任务数量

class RewardProps(TypedDict):
    """奖励与评分器相关的配置"""
    original_grader: str  # 原始任务（种子）使用的评分器标识
    synthetic_grader: str # 合成任务（演化出的）使用的评分器标识

def get_exploration_strategy(name: str, strategy_args, *, tokenizer, config, llm_client, env_profile) -> TaskExploreStrategy:
    """
    根据策略名称，实例化并返回对应的任务探索策略（工厂函数）。

    Args:
        name (str): 策略名称（如 "random", "api_driven"）。
        strategy_args (dict): 传递给具体策略类的额外参数字典。
        tokenizer: 模型对应的 tokenizer 实例。
        config (DictConfig): 全局配置对象。
        llm_client (LlmClient): 用于大模型交互的客户端。
        env_profile (EnvProfile): 目标环境的画像配置。

    Returns:
        TaskExploreStrategy: 实例化后的具体探索策略对象。
    """
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

# ================= ApiDrivenPipeline =================

class ApiDrivenPipeline:
    """
    专门负责处理 API-Driven 策略的复杂生成、过滤、探索生命周期的 Pipeline。
    """
    def __init__(self, manager: "TaskManager", tasks: Sequence[Task], show_progress: bool = False, resume_file: Optional[str] = None):
        """
        初始化 API 驱动的数据生成流水线，准备上下文及文件读写锁。
        """
        self.manager = manager
        self.tasks = tasks
        self.show_progress = show_progress
        self.mem_lock = threading.Lock() 
        
        self.strategy_args = manager._config.task_manager.get('exploration_strategy_args', {})
        self.a = self.strategy_args.get('a', 1) 
        self.b = self.strategy_args.get('b', 1) 
        self.debug_mode = False 
        
        logger.info(f"[API-Driven] Strategy Args: a={self.a}, b={self.b}, debug_log={self.debug_mode}")
        if self.debug_mode:
            logger.warning("Debug mode enabled: forcing single thread.")

        gen_output_dir = os.environ.get("GEN_OUTPUT_DIR")
        if gen_output_dir:
            base_name = "generated_tasks"
            self.resume_file = os.path.join(gen_output_dir, base_name)
            logger.info(f"📂 [Isolation] Redirecting all generation outputs to: {self.resume_file}")
        else:
            self.resume_file = resume_file or '.generate_task_api'
            
        self._init_paths()
        
        self.api_knowledge = getattr(self.manager._exploration_strategy, 'api_knowledge', {})
        self.active_apps_set = getattr(self.manager._exploration_strategy, 'active_apps', set(self.api_knowledge.keys()))

    def _init_paths(self):
        """
        挂载所有流式文件的落地路径，细分各阶段（Intra/Cross，生成/过滤/演化）的产物存储。
        """
        self.intra_gen_path = f"{self.resume_file}.intra.generated.jsonl"
        self.intra_filtered_path = f"{self.resume_file}.intra.filtered.jsonl"
        self.intra_final_path = f"{self.resume_file}.intra.jsonl"
        self.intra_direct_path = f"{self.resume_file}.intra.direct.jsonl"
        self.intra_evolved_path = f"{self.resume_file}.intra.evolved.jsonl" 
        
        self.cross_gen_path = f"{self.resume_file}.cross.generated.jsonl"
        self.cross_filtered_path = f"{self.resume_file}.cross.filtered.jsonl"
        self.cross_final_path = f"{self.resume_file}.extra.jsonl"
        self.cross_direct_path = f"{self.resume_file}.cross.direct.jsonl"
        self.cross_evolved_path = f"{self.resume_file}.cross.evolved.jsonl"

    def _load_intermediate_tasks(self, path: str) -> Optional[List[Task]]:
        """
        按行读取流式 jsonl 文件，还原为内存中的 Task 列表。
        [修复] 修正了读取旧格式时提早 return 导致后续行被忽略的 Bug。
        """
        if os.path.exists(path):
            try:
                tasks_list = []
                with open(path, 'r') as f:
                    for line in f:
                        if line.strip(): 
                            try:
                                data = json.loads(line)
                                if "task" in data and "processed_indices" in data:
                                    # 修复：改为 extend 而非 return，允许继续读取下一行
                                    tasks_list.extend([Task.parse_obj(t) for t in data['tasks']])
                                else:
                                    tasks_list.append(Task.parse_obj(data))
                            except: pass 
                logger.info(f"Loaded {len(tasks_list)} tasks from stream file {path}")
                return tasks_list
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {path}: {e}")
        return None

    def _thread_safe_append(self, path: str, items: List[Any]):
        """
        通过获取全局 IO 锁，安全地将对象序列化为 JSON 行并追加入文件。
        """
        if not items: return
        with io_lock:
            try:
                with open(path, 'a', encoding='utf-8') as f:
                    for item in items:
                        obj = item.dict() if hasattr(item, 'dict') else item
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Failed to append to {path}: {e}")

    def _worker_generate_intra(self, idx: int, app_name: str, seed_task: Task) -> List[Task]:
        """
        同领域任务的生成 Worker，调用大模型生成面向单 App 的查询。
        """
        try:
            base_task = copy.deepcopy(seed_task)
            if base_task.metadata is None: base_task.metadata = {}
            base_task.metadata['thread_index'] = idx % self.manager._num_exploration_threads
            
            tasks = self.manager._exploration_strategy.generate_intra_task(app_name, task=base_task)
            if not tasks: return []
            
            for sub_idx, current_task in enumerate(tasks):
                current_task.metadata["data_id"] = f"gen_intra_{idx}_{sub_idx}"
            return tasks
        except Exception as e:
            logger.error(f"[Intra-Gen] Error idx {idx}: {e}")
            return []

    def _worker_generate_cross(self, idx: int, target_apps: List[str], seed_task: Task) -> List[Task]:
        """
        跨领域任务的生成 Worker，调用大模型生成多 App 协同作业的查询。
        """
        try:
            base_task = copy.deepcopy(seed_task)
            if base_task.metadata is None: base_task.metadata = {}
            base_task.metadata['thread_index'] = idx % self.manager._num_exploration_threads
            
            tasks = self.manager._exploration_strategy.generate_cross_task(target_apps, task=base_task)
            if not tasks: return []
            
            for sub_idx, current_task in enumerate(tasks):
                current_task.metadata["data_id"] = f"gen_cross_{idx}_{sub_idx}"
            return tasks
        except Exception as e:
            logger.error(f"[Cross-Gen] Error idx {idx}: {e}")
            return []

    def _worker_explore_intra(self, task: Task) -> List[TaskObjective]:
        """
        同领域任务探索 Worker：在环境中执行生成的任务，收集有效轨迹和直接验证标签。
        """
        try:
            data_id = task.metadata.get("data_id", f"unknown_{random.randint(0,1000)}")
            trajectories = self.manager._exploration_strategy.explore(task, data_id, data_id)
            success_traj = trajectories[0] if (trajectories and trajectories[0].reward and trajectories[0].reward.outcome >= 0.7) else None
            if not success_traj: return []

            reward_val = success_traj.reward.outcome
            raw_gt_steps = [s.dict() if hasattr(s, 'dict') else s for s in success_traj.steps]
            direct_verified_obj = self.manager._exploration_strategy.verify_direct_gt(task, success_traj)
            
            origin_query_for_evolved = task.query
            origin_gt_for_evolved = None 

            if direct_verified_obj:
                direct_verified_obj.task.raw_trajectory = raw_gt_steps
                direct_verified_obj.task.origin_ground_truth = None
                direct_verified_obj.task.origin_query = task.query
                direct_verified_obj.task.metadata["source_data_id"] = data_id
                direct_verified_obj.task.metadata["execution_reward"] = {"outcome": reward_val}
                
                self._thread_safe_append(self.intra_direct_path, [direct_verified_obj])
                origin_gt_for_evolved = direct_verified_obj.task.ground_truth 

            evolved_results = self.manager._exploration_strategy.summarize(task, success_traj)
            if evolved_results:
                for res in evolved_results:
                    res.task.raw_trajectory = raw_gt_steps
                    res.task.origin_ground_truth = origin_gt_for_evolved
                    res.task.origin_query = origin_query_for_evolved
                    res.confidence = 0 
                    res.reward = reward_val
                    res.task.metadata.update({"data_pair_type": "evolved", "source_data_id": data_id, "has_verified_origin": (origin_gt_for_evolved is not None)})
                self._thread_safe_append(self.intra_evolved_path, evolved_results)
            return evolved_results if evolved_results else []
        except Exception as e:
            logger.error(f"[Intra-Explore] Error: {e}")
            return []

    def _worker_explore_cross(self, task: Task) -> List[TaskObjective]:
        """
        跨领域任务探索 Worker：同上，主要负责多 App 联动环境执行及提取。
        """
        try:
            data_id = task.metadata.get("data_id", f"unknown_{random.randint(0,1000)}")
            trajectories = self.manager._exploration_strategy.explore(task, data_id, data_id)
            success_traj = trajectories[0] if (trajectories and trajectories[0].reward and trajectories[0].reward.outcome >= 0.7) else None
            if not success_traj: return []

            reward_val = success_traj.reward.outcome
            raw_gt_steps = [s.dict() if hasattr(s, 'dict') else s for s in success_traj.steps]
            direct_verified_obj = self.manager._exploration_strategy.verify_direct_gt(task, success_traj)
            
            origin_query_for_evolved = task.query
            origin_gt_for_evolved = None 

            if direct_verified_obj:
                direct_verified_obj.task.raw_trajectory = raw_gt_steps
                direct_verified_obj.task.origin_ground_truth = None
                direct_verified_obj.task.origin_query = task.query
                direct_verified_obj.task.metadata["source_data_id"] = data_id
                direct_verified_obj.task.metadata["execution_reward"] = {"outcome": reward_val}
                self._thread_safe_append(self.cross_direct_path, [direct_verified_obj])
                origin_gt_for_evolved = direct_verified_obj.task.ground_truth

            evolved_results = self.manager._exploration_strategy.summarize(task, success_traj)
            if evolved_results:
                for res in evolved_results:
                    res.task.raw_trajectory = raw_gt_steps
                    res.task.origin_ground_truth = origin_gt_for_evolved
                    res.task.origin_query = origin_query_for_evolved
                    res.confidence = 0 
                    res.reward = reward_val
                    res.task.metadata.update({"data_pair_type": "evolved", "source_data_id": data_id, "has_verified_origin": (origin_gt_for_evolved is not None)})
                self._thread_safe_append(self.cross_evolved_path, evolved_results)
            return evolved_results if evolved_results else []
        except Exception as e:
            logger.error(f"[Cross-Explore] Error: {e}")
            return []

    def run(self) -> List[TaskObjective]:
        """
        执行 API-Driven 流水线。涵盖多阶段：生成 -> 初筛 -> 环境推演 -> 提取归档 -> 后置大模型打分。
        """
        target_files = [
            self.intra_direct_path, 
            self.cross_direct_path,
            self.intra_evolved_path, 
            self.cross_evolved_path
        ]
        
        all_files_exist = all(os.path.exists(p) for p in target_files)

        if all_files_exist:
            logger.info("⚡ [中间级拦截] 检测到四个底层探索产物文件已全部齐备，跳过耗时的生成与环境探索阶段！")
        else:
            logger.info("⚠️ [正常流程] 底层探索产物不全或不存在，开始执行完整的生成与探索...")
            
            # === PART 1: INTRA-DOMAIN ===
            logger.info("=== Starting PART 1: Intra-Domain Generation & Filtering ===")
            valid_apps_intra = [app for app in sorted(self.active_apps_set) if self.api_knowledge.get(app, {}).get("apis")]
            random.shuffle(valid_apps_intra)
            
            intra_task_pool = (list(copy.copy(self.tasks)) * int(self.a + 1))[:int(len(self.tasks) * self.a)]
            if self.debug_mode: intra_task_pool = intra_task_pool[:1]
            target_len_intra = len(intra_task_pool)

            generated_intra_tasks = self._load_intermediate_tasks(self.intra_gen_path) or []
            current_count = len(generated_intra_tasks)
            
            if current_count < target_len_intra and valid_apps_intra:
                with ThreadPoolExecutor(max_workers=1 if self.debug_mode else self.manager._num_exploration_threads) as pool:
                    futures = []
                    for idx in range(current_count, target_len_intra):
                        target_app = valid_apps_intra[idx % len(valid_apps_intra)]
                        futures.append(pool.submit(self._worker_generate_intra, idx, target_app, intra_task_pool[idx]))
                    
                    for f in tqdm(as_completed(futures), total=len(futures), desc="Intra Gen", disable=not self.show_progress):
                        res = f.result()
                        if res: 
                            res_list = res if isinstance(res, list) else [res]
                            self._thread_safe_append(self.intra_gen_path, res_list)
                            with self.mem_lock: generated_intra_tasks.extend(res_list)
            
            filtered_intra_tasks = self._load_intermediate_tasks(self.intra_filtered_path) or []
            if len(generated_intra_tasks) > 0 and (len(filtered_intra_tasks) / len(generated_intra_tasks) > 0.5):
                logger.info(f"Reusing cached filtered intra tasks: {len(filtered_intra_tasks)}/{len(generated_intra_tasks)}")
            else:
                filtered_ids = {t.metadata["data_id"] for t in filtered_intra_tasks if t.metadata and "data_id" in t.metadata}
                pending = [t for t in generated_intra_tasks if t.metadata.get("data_id") not in filtered_ids]
                if pending:
                    newly_filtered = self.manager._apply_filters_with_report(pending, self.manager.api_llm_pre_filter, "Intra-Pre-Filter")
                    self._thread_safe_append(self.intra_filtered_path, newly_filtered)
                    with self.mem_lock: filtered_intra_tasks.extend(newly_filtered)

            # === PART 2: CROSS-DOMAIN ===
            logger.info("=== Starting PART 2: Cross-Domain Generation & Filtering ===")
            valid_apps_cross = [app for app in sorted(self.active_apps_set) if self.api_knowledge.get(app, {}).get("apis")]
            valid_apps_set = set(valid_apps_cross)
            valid_candidate_tasks = [t for t in self.tasks if getattr(t, 'app', None) in valid_apps_set or getattr(t, 'app_name', None) in valid_apps_set] or self.tasks
            
            target_valid_count = 1 if self.debug_mode else int(len(self.tasks) * self.b)
            generated_cross_tasks = self._load_intermediate_tasks(self.cross_gen_path) or []
            filtered_cross_tasks = self._load_intermediate_tasks(self.cross_filtered_path) or []
            current_valid_count = len(filtered_cross_tasks)
            global_gen_idx = len(generated_cross_tasks)
            loop_idx = 0

            while current_valid_count < target_valid_count and loop_idx < 10:
                loop_idx += 1
                needed = target_valid_count - current_valid_count
                batch_size = max(int(needed * 2.0), self.manager._num_exploration_threads * 2) if not self.debug_mode else 1
                
                newly_generated = []
                with ThreadPoolExecutor(max_workers=1 if self.debug_mode else self.manager._num_exploration_threads) as pool:
                    futures = []
                    for i in range(batch_size):
                        app_count = random.choices([2, 3], weights=[0.8, 0.2])[0]
                        app_count = min(app_count, len(valid_apps_cross))
                        target_apps = random.sample(valid_apps_cross, app_count)
                        seed_task = valid_candidate_tasks[i % len(valid_candidate_tasks)]
                        futures.append(pool.submit(self._worker_generate_cross, global_gen_idx + i, target_apps, seed_task))
                    
                    for f in tqdm(as_completed(futures), total=len(futures), desc=f"Cross Gen R{loop_idx}", disable=not self.show_progress):
                        res = f.result()
                        if res: 
                            res_list = res if isinstance(res, list) else [res]
                            self._thread_safe_append(self.cross_gen_path, res_list)
                            with self.mem_lock: newly_generated.extend(res_list)
                
                global_gen_idx += batch_size
                with self.mem_lock: generated_cross_tasks.extend(newly_generated)
                
                if not newly_generated: continue
                
                newly_filtered = self.manager._apply_filters_with_report(newly_generated, self.manager.api_llm_pre_filter, f"Cross-Filter-R{loop_idx}")
                if newly_filtered:
                    self._thread_safe_append(self.cross_filtered_path, newly_filtered)
                    with self.mem_lock: filtered_cross_tasks.extend(newly_filtered)
                current_valid_count = len(filtered_cross_tasks)

            # === PART 3: INTRA-DOMAIN EXPLORE ===
            logger.info("=== Starting PART 3: Intra-Domain Exploration ===")
            intra_res = []
            explored_ids_intra = set()
            if os.path.exists(self.intra_final_path):
                with open(self.intra_final_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        intra_res.append(data)
                        try: explored_ids_intra.add(data.get("task", {}).get("metadata", {}).get("data_id"))
                        except: pass
            
            if len(filtered_intra_tasks) > 0 and (len(intra_res) / len(filtered_intra_tasks) > 0.5):
                pass
            else:
                pending_explore_intra = [t for t in filtered_intra_tasks if t.metadata.get("data_id") not in explored_ids_intra]
                if pending_explore_intra:
                    with ThreadPoolExecutor(max_workers=1 if self.debug_mode else self.manager._num_exploration_threads) as pool:
                        futures = {pool.submit(self._worker_explore_intra, t): i for i, t in enumerate(pending_explore_intra)}
                        for future in tqdm(as_completed(futures), total=len(futures), desc="Intra Explore", disable=not self.show_progress):
                            filtered_objs = self.manager._apply_filters_with_report(future.result(), self.manager._realtime_filters, "Intra-Worker")
                            if filtered_objs:
                                self._thread_safe_append(self.intra_final_path, filtered_objs)
                                with self.mem_lock: intra_res.extend(filtered_objs)

            # === PART 4: CROSS-DOMAIN EXPLORE ===
            logger.info("=== Starting PART 4: Cross-Domain Exploration ===")
            cross_res = []
            explored_ids_cross = set()
            if os.path.exists(self.cross_final_path):
                with open(self.cross_final_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        cross_res.append(data)
                        try: explored_ids_cross.add(data.get("task", {}).get("metadata", {}).get("data_id"))
                        except: pass

            if len(filtered_cross_tasks) > 0 and (len(cross_res) / len(filtered_cross_tasks) > 0.5):
                pass
            else:
                pending_explore_cross = [t for t in filtered_cross_tasks if t.metadata.get("data_id") not in explored_ids_cross]
                if pending_explore_cross:
                    with ThreadPoolExecutor(max_workers=1 if self.debug_mode else self.manager._num_exploration_threads) as pool:
                        futures = {pool.submit(self._worker_explore_cross, t): i for i, t in enumerate(pending_explore_cross)}
                        for future in tqdm(as_completed(futures), total=len(futures), desc="Cross Explore", disable=not self.show_progress):
                            filtered_objs = self.manager._apply_filters_with_report(future.result(), self.manager._realtime_filters, "Cross-Worker")
                            if filtered_objs:
                                self._thread_safe_append(self.cross_final_path, filtered_objs)
                                with self.mem_lock: cross_res.extend(filtered_objs)

        # === Final Merge & Post Filter (支持断点续传) ===
        logger.info("=== Starting Final Merge & Filtering (Direct & Evolved, 支持断点续传) ===")
        total_results = []
        
        for path in target_files:
            if os.path.exists(path):
                count = 0
                logger.info(f"📦 正在提取底层产物: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            data = json.loads(line)
                            total_results.append(TaskObjective.parse_obj(data))
                            count += 1
                        except Exception as e:
                            pass
                logger.info(f" -> 提取了 {count} 条数据。")

        logger.info(f"🔍 数据聚合完成，共 {len(total_results)} 条。送入 Post Filter (LLM 裁判)...")
        
        passed_cache_path = self.resume_file + ".post_filter_passed.jsonl"
        processed_ids_path = self.resume_file + ".post_filter_processed.json"
        
        processed_ids = set()
        if os.path.exists(processed_ids_path):
            try:
                with open(processed_ids_path, 'r', encoding='utf-8') as f:
                    processed_ids = set(json.load(f))
            except Exception: pass
            
        final_survivors = []
        if os.path.exists(passed_cache_path):
            with open(passed_cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        final_survivors.append(TaskObjective.parse_obj(json.loads(line)))
                        
        pending_results = []
        for obj in total_results:
            uid = obj.task.metadata.get("data_id") if (obj.task.metadata and "data_id" in obj.task.metadata) else obj.task.task_id
            if uid not in processed_ids:
                pending_results.append((uid, obj))

        if pending_results:
            logger.info(f"🚀 发现 {len(pending_results)} 条待过滤数据 (已跳过 {len(processed_ids)} 条历史记录)，开始分批送入 LLM 裁判...")
            
            batch_size = 10 
            for i in tqdm(range(0, len(pending_results), batch_size), desc="Post Filtering Batches"):
                batch_tuples = pending_results[i : i + batch_size]
                batch_objs = [item[1] for item in batch_tuples]
                batch_uids = [item[0] for item in batch_tuples]
                
                batch_survivors = self.manager._apply_post_filter(batch_objs)
                
                if batch_survivors:
                    final_survivors.extend(batch_survivors)
                    self._thread_safe_append(passed_cache_path, batch_survivors)
                
                processed_ids.update(batch_uids)
                with open(processed_ids_path, 'w', encoding='utf-8') as f:
                    json.dump(list(processed_ids), f)
        else:
            logger.info("✅ 所有聚合数据均已在历史中过滤完毕，直接使用缓存的过滤结果。")

        gen_output_dir = os.environ.get("GEN_OUTPUT_DIR", "")
        if gen_output_dir:
            output_file = os.path.join(gen_output_dir, "tasks_explored.train.json")
        else:
            output_file = "tasks_explored.train.json"
            
        logger.info(f"💾 Post Filter 全部完成！最终剩余 {len(final_survivors)} 条优质数据。正在写入 {output_file} ...")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([obj.dict() for obj in final_survivors], f, ensure_ascii=False, indent=2)
            logger.info(f"🎉 黄金训练集已成功保存至：{output_file}！下次启动将直接短路读取。")
        except Exception as e:
            logger.error(f"❌ 写入 {output_file} 失败: {e}")

        return final_survivors


# ================= TaskManager 类 =================

class TaskManager(object):
    """
    任务生命周期核心管理器：负责统一调度探索策略、过滤器、任务组装及数据融合逻辑。
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
        初始化管理器上下文、挂载依赖服务。
        [修复] 添加 _already_loaded_target_file 的显式初始化。
        """
        self._config = config
        self._tokenizer = tokenizer
        self._exploration_strategy = get_exploration_strategy(
            exploration_strategy, 
            exploration_strategy_args, 
            tokenizer=tokenizer, 
            config=config,
            llm_client=llm_client,
            env_profile=env_profile
        )
        self._llm_client = llm_client
        self._old_retrival = old_retrival       
        self._mixture_strategy = mixture_strategy 
        self._reward_config = reward_config
        self._env_service_url = env_service_url
        self._num_exploration_threads = kwargs.get("num_explore_threads", 5)
        self._n = kwargs.get("n", 1)
        self._already_loaded_target_file = False  # [修复] 补齐属性初始化

        self.agent_flow = agent_flow  
        self.env_worker = env_worker  

        self._realtime_filters: list[TaskPostFilter] = [NaiveTaskPostFilter()]
        self._post_filter: list[TaskPostFilter] = [
            LlmFilter(env_service_url, llm_client, self._num_exploration_threads, tokenizer=tokenizer, config=config)
        ]
        self.api_llm_pre_filter = [
            LlmQualityPreFilter(llm_client, num_threads=self._num_exploration_threads)
        ]
        
        self._tasks: list[Task] = [] 
        self._hindsight_file_offset = 0  
        self._hindsight_file_path = self._config.task_manager.get('exploration_strategy_args', {}).get('hindsight_data_path', './tasks_explored/hindsight_supplement.jsonl')
        
    @property
    def seed_tasks(self):
        """获取当前持有的原始（种子）任务列表。"""
        return self._tasks
    
    @property
    def seed_task_objectives(self):
        """将原始种子任务打包为强化学习的目标格式（满置信度、无Reward）。"""
        return [TaskObjective(task=task, confidence=1.0, reward=None) for task in self.seed_tasks]

    def load_tasks(self, tasks: Sequence[Task]):
        """加载已初始化的种子任务并做格式断言拦截。"""
        self._tasks.extend(tasks)
        assert all([x.query is None for x in self._tasks]), "种子任务的 query 必须为空（待演化）"
        logger.info(f"loaded tasks, #tasks={len(self._tasks)}")

    def load_tasks_from_dataset(self, dataset: RLHFDataset, *, env_type: str):
        """从标准 RLHF 数据集结构中提取并转换种子任务。"""
        new_tasks = adapter.convert_to_tasks(dataset, env_type=env_type, grader=self._reward_config["original_grader"])
        self._tasks.extend(new_tasks)
        assert all([x.query is None for x in self._tasks]), "种子任务的 query 必须为空"
        logger.info(f"loaded tasks from dataset, #tasks={len(self._tasks)}")

    def load_tasks_from_environment(self, env: EnvClient, *, env_type: str, split: str, params: Optional[dict] = None):
        """通过远程 EnvClient 服务直接拉取特定环境切片（Split）的任务。"""
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
        """动态增加一个基于实时行为的局部过滤器。"""
        self._realtime_filters.append(filter)

    def load_new_hindsight_tasks(self, file_path: str = None) -> int:
        """
        加载补充流中产生的 hindsight（后见之明/重标签）任务以扩充训练集。
        通过维护 _hindsight_file_offset 实现只读取新增内容。
        """
        target_path = file_path or self._hindsight_file_path
        if not os.path.exists(target_path): return 0
        new_tasks = []
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                f.seek(self._hindsight_file_offset)
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        task = Task(
                            task_id=data['task_id'], query=data['query'], env_type="hindsight",
                            open_query=True, evaluator=self._reward_config.get("synthetic_grader", "default"),
                            extra_info={"ground_truth": data.get('ground_truth')}
                        )
                        new_tasks.append(task)
                    except json.JSONDecodeError: continue
                self._hindsight_file_offset = f.tell()
        except Exception as e:
            logger.error(f"Failed to load hindsight tasks: {e}")
            return 0
        if new_tasks:
            self._tasks.extend(new_tasks)
            return len(new_tasks)
        return 0

    def _compute_tasks_hash(self, tasks: Sequence[Task]) -> str:
        """计算任务集的 MD5 哈希，用于校验缓存匹配度。"""
        combined_str = "|".join([f"{task.task_id}:{task.env_type}" for task in tasks])
        return hashlib.md5(combined_str.encode()).hexdigest()

    def _get_item_identifier(self, item: Any) -> str:
        """多态地提取数据对象的唯一 ID 标识。"""
        if isinstance(item, TaskObjective): return item.task.task_id
        elif isinstance(item, Task): return item.task_id
        elif isinstance(item, dict): return item.get("task_id", str(id(item)))
        else: return str(id(item))

    def _get_item_desc(self, item: Any) -> str:
        """提取对象的短描述用于 Log 报告。"""
        task = item.task if isinstance(item, TaskObjective) else (item if isinstance(item, Task) else None)
        if task: return f"[Query]: {task.query}" if task.query else f"[ID]: {task.task_id} (No Query)"
        return str(item)[:100]

    def _apply_filters_with_report(self, items: List[Any], filters: List[Any], stage_name: str) -> List[Any]:
        """
        执行一条由多个 Filter 对象组成的流水线，并详细报告丢弃统计和示例。
        """
        if not items: return []
        current_items = items
        if len(current_items) > 0: logger.info(f"🛡️ [过滤器报告 - {stage_name}] 初始数量: {len(current_items)}")
        for f in filters:
            filter_name = f.__class__.__name__
            before_count = len(current_items)
            before_map = {self._get_item_identifier(item): item for item in current_items}
            current_items = f.filter(current_items)
            after_count = len(current_items)
            dropped_count = before_count - after_count
            if dropped_count > 0:
                logger.warning(f"❌ [Filter: {filter_name}] 过滤掉了 {dropped_count} 个样本 (剩余: {after_count})")
                after_ids = set(self._get_item_identifier(item) for item in current_items)
                dropped_items = [item for uid, item in before_map.items() if uid not in after_ids]
                for i, dropped in enumerate(dropped_items[:3]):
                    logger.warning(f"   -> 丢弃样本示例 #{i+1}: {self._get_item_desc(dropped)}")
                if dropped_count > 3: logger.warning(f"   -> ... 以及其他 {dropped_count - 3} 个")
            else:
                logger.info(f"✅ [Filter: {filter_name}] 无损通过 (剩余: {after_count})")
        return current_items

    def generate_task(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        总入口：最高级拦截。如果有训练终态文件，直接全量读取并跳过所有生成。
        """
        gen_output_dir = os.environ.get("GEN_OUTPUT_DIR", "")
        if gen_output_dir:
            target_file = os.path.join(gen_output_dir, "tasks_explored.train.json")
        else:
            target_file = "tasks_explored.train.json"
        
        if os.path.exists(target_file):
            if getattr(self, "_already_loaded_target_file", False):
                return []
                
            logger.info(f"⚡ [最高级拦截] 检测到最终数据集 {target_file}，无视策略，直接全量加载！")
            self._already_loaded_target_file = True
            
            total_results = []
            try:
                with open(target_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content.startswith('['): 
                        data_list = json.loads(content)
                        for item in data_list:
                            total_results.append(TaskObjective.parse_obj(item))
                    else:
                        for line in content.split('\n'):
                            if line.strip():
                                total_results.append(TaskObjective.parse_obj(json.loads(line)))
                
                logger.info(f"✅ 成功从 {target_file} 一次性加载了 {len(total_results)} 条数据！")
                return total_results 
            except Exception as e:
                logger.error(f"❌ 读取或解析 {target_file} 失败: {e}")
                raise

        strategy_type = "api_driven" if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy) else "random"
        if strategy_type == "api_driven":
            pipeline = ApiDrivenPipeline(self, tasks, show_progress, resume_file)
            return pipeline.run()
        else:
            return self._generate_task_random(tasks, show_progress=show_progress, resume_file=resume_file)
        
    def _generate_task_random(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        早期版本的纯随机策略生成管线，保留用于降级与兼容测试。
        """
        if resume_file is None: resume_file = '.generate_task.checkpoint.json'
        current_tasks_hash = self._compute_tasks_hash(tasks)
        res = []
        processed_indices = set()
        
        if resume_file and os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    checkpoint = json.load(f)
                    if checkpoint.get('tasks_hash') != current_tasks_hash:
                        os.remove(resume_file) 
                    else:
                        res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in checkpoint.get('results', [])]
                        processed_indices = {int(i) for i in checkpoint.get('processed_indices', [])}
            except Exception as e:
                logger.warning(f"断点加载失败: {e}, 将重新开始生成")

        task_q = list(copy.copy(tasks)) * self._n
        parallel_num = max(1, min(self._num_exploration_threads, len(tasks)))
        
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as pool:
            batch_indices = list(range(0, len(task_q), parallel_num))
            for idx, i in enumerate(tqdm(batch_indices, desc="generating tasks (random)", disable=not show_progress)):
                if idx in processed_indices: continue
                futures = [pool.submit(self._exlore_and_summarize, task, "unknown", "unknown") for task in task_q[i : i + parallel_num]]
                task_objectives = sum([future.result() for future in futures], [])
                batch_filtered = self._apply_filters_with_report(task_objectives, self._realtime_filters, f"Random-Batch-{idx}-Realtime")
                res.extend(batch_filtered)
                
                self._old_retrival.reset()
                for j in batch_filtered: self._old_retrival.add_objective(j)
                processed_indices.add(idx)
                if resume_file: self._save_checkpoint(resume_file, res, processed_indices, len(batch_indices), current_tasks_hash)

        return self._apply_post_filter(res)

    def _save_checkpoint(self, path, results, processed_indices, total, hash_val):
        """将当前 Batch 生成的成就序列化到 JSON 中作为恢复检查点。"""
        try:
            checkpoint_data = {
                'results': [obj.dict() for obj in results],
                'processed_indices': list(processed_indices),
                'total_batches': total,
                'tasks_hash': hash_val,
                'timestamp': time.time()
            }
            with open(path, 'w') as f: json.dump(checkpoint_data, f, indent=2)
        except Exception as e: logger.warning(f"保存断点失败: {e}")

    def _apply_post_filter(self, res: List[TaskObjective]) -> List[TaskObjective]:
        """统筹所有全局过滤逻辑并打乱数据返回最终打标形态。"""
        res = self._apply_filters_with_report(res, self._realtime_filters, "PostProcess-Realtime")
        cnt_before = len(res)
        res = self._apply_filters_with_report(res, self._post_filter, "PostProcess-LLM")
        logger.info(f"后置过滤完成: 过滤前={cnt_before}, 过滤后={len(res)}")
        random.shuffle(res)
        return res

    def _exlore_and_summarize(self, task: Task, data_id: str, rollout_id: str) -> list[TaskObjective]:
        """单步执行环境互动（Explore）并总结轨迹（Summarize）的聚合短路方法。"""
        try:
            trajectories = self._step_explore(task, data_id, rollout_id)
            task_objectives = sum([self._step_summarize(task, trajectory) for trajectory in trajectories], [])
            valid_objs = [x for x in task_objectives if x.task.open_query]
            return valid_objs
        except Exception as e:
            logger.error(f"Error in random explore: {e}")
            return []

    def _step_explore(self, task: Task, data_id: str, rollout_id: str) -> list[Trajectory]:
        """将底层策略的环境推演暴露。"""
        return self._exploration_strategy.explore(task, data_id, rollout_id)

    def _step_summarize(self, task: Task, trajectory: Trajectory) -> list[TaskObjective]:
        """将底层策略的任务提取方法暴露。"""
        return self._exploration_strategy.summarize(task, trajectory)

# ================= 数据集类 =================

class FullDataset(Dataset):
    """
    静态 Pytorch 数据集封装：
    用于一次性完整准备整个阶段的强化学习交互数据，通过 MixtureStrategy 调和原始分布与合成数据的比例。
    """
    def __init__(self, manager: TaskManager, mixture_strategy: MixtureStrategy, reward_config: RewardProps, cache_path: Optional[str] = None, *, tokenizer, config, processor):
        """初始化全量数据集实例。"""
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

        if self._mixture_strategy.need_synthetic:
            logger.info("正在准备合成任务数据...")
            if self._cache_path is not None and os.path.exists(self._cache_path):
                self.load_from_file()
            else:
                self.reload_new_task()
                if self._cache_path is not None: self.save_to_file()
        
        self._rebuild_dataset()

    def _rebuild_dataset(self):
        """核心数据组装方法：按既定配方混合两者后转换为 RL 引擎特有格式。"""
        self._objectives = self._mixture_strategy.mix_data(self._synthetic_objectives, self._tasks)
        if len(self._objectives) == 0:
            logger.error("【严重错误】没有可用的训练数据！可能是环境服务挂了，或者 Debug 模式下生成的任务全部被过滤了。")
            raise ValueError("Dataset is empty. Please check env_service status or disable debug_log.")
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)

    def update(self):
        """提供给外部控制器的手动热更新接口。"""
        if not self._synthetic_objectives:
            logger.warning("No synthetic objectives available, did you call load_from_file() or reload() first?")
        self._rebuild_dataset() 
        logger.info("Dataset updated manually via update().")

    def set_mixture_strategy(self, strategy: MixtureStrategy):
        """动态替换数据混合策略（用于课程学习或分布偏移衰减）。"""
        self._mixture_strategy = strategy  
        logger.info(f"mixture strategy updated to: {type(strategy).__name__}")

    def save_to_file(self):
        """将生成的合成任务物化到本地 JSONL 缓存。"""
        assert self._cache_path is not None
        with open(self._cache_path, "w") as f:
            f.writelines([ob.json() + "\n" for ob in self._synthetic_objectives])  
        logger.info(f"Saved {len(self._objectives)} objectives to {self._cache_path}")  

    def load_from_file(self):
        """从本地缓存文件反推解析为内存里的 TaskObjective 列表。"""
        if self._cache_path is None:
            logger.error("trying to load synthetic objectives from file, but cache_path is not set")
            return
        
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "r") as f:
                self._synthetic_objectives = []
                for line in f:
                    if not line.strip(): continue
                    t = json.loads(line)
                    assert 'task' in t
                    if 'open_query' not in t['task']:
                        t['task']['open_query'] = True 
                    
                    tmp = TaskObjective.parse_obj(t)
                    if tmp.ground_truth is None:
                        tmp.ground_truth = t.get('ground_truth')
                    self._synthetic_objectives.append(tmp)
        else:
            raise FileNotFoundError(f"failed to load synthetic objectives from file {self._cache_path}, file not found")
        
        for item in self._synthetic_objectives:
            assert item.ground_truth is not None

        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]  

    def reload_new_task(self):
        """强制触发 TaskManager 走一轮完整的任务合成流以覆盖旧缓存。"""
        self._synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]  

    def get_statistics(self) -> dict:
        """获取当前组合的统计分布画像。"""
        if not self._objectives:
            return {"total": 0, "synthetic": 0, "original": 0, "synthetic_ratio": 0.0, "strategy_info": str(self._mixture_strategy)}
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
        """支持索引获取内部包装的 RlDataset 元素。"""
        if self._dataset is None: raise RuntimeError("Dataset not loaded.")  
        return self._dataset[index]

    def __len__(self):
        """返回数据集实际可用的总样本数。"""
        return 0 if self._dataset is None else len(self._dataset)

class AutoReloadDataset(IterableDataset):
    """
    支持边训边生成的流式 IterableDataset 封装：
    特别适用于难以将大规模数据全部持留于内存的大模型在线训练 (On-fly RL) 场景。
    """
    def __init__(self, manager: TaskManager, tasks: Iterable[Task], bs: int, mix_origins: bool = False, *, tokenizer, config, processor):
        """
        初始化流式数据集上下文。
        [修复] 添加独立的迭代器维护游标，防止多轮 Reload 采样卡死在列表头部。
        """
        self._manager = manager
        self._tasks = tasks
        self._task_iter = iter(tasks)  # [修复] 维护真正的迭代器状态
        self._bs = bs
        self._tokenizer = tokenizer
        self._config = config
        self._processor = processor
        self._dataset = OnflyRlDataset(release_used_dataset=True)

    def reload(self):
        """
        触发拉取/生成下一批次所需训练数据的逻辑挂载点。
        [修复] 修正了获取 delta 时对列表对象的错误遍历方式。
        """
        delta = []
        try:
            for _ in range(self._bs):
                delta.append(next(self._task_iter))
        except StopIteration:
            pass # 种子任务池已耗尽，此时可能获取到的 len(delta) < bs

        # 如果没有任务可领，通知迭代停止
        if not delta:
            return 0

        ls = self._manager.generate_task(delta)
        
        retry_count = 0
        max_retries = 3 
        
        while len(ls) < self._bs * self._manager._n and retry_count < max_retries:
            logger.debug(f"数据不足期望量，正在尝试重新获取... ({retry_count}/{max_retries})")
            new_ls = self._manager.generate_task(delta)
            
            if not new_ls:
                break
                
            ls.extend(new_ls)
            retry_count += 1

        if not ls:
            return 0 

        self._dataset.append_dataset(to_rl_dataset(ls, self._tokenizer, self._config, self._processor))
        return self._dataset.num_rest_data

    def __iter__(self):
        """使得类实例成为一个标准的 Python 迭代器。"""
        return self

    def __next__(self):
        """
        动态按需吐出数据，底层数据空载时安全挂起并触发生成引擎补充弹药。
        """
        if self._dataset.num_rest_data == 0:  
            logger.debug("no data left")
            if self.reload() == 0:  
                logger.debug("no task left, stop reloading and iteration")
                raise StopIteration
        return next(self._dataset)