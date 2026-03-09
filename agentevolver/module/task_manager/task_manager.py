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
    Optional, Sequence, TypedDict, Unpack, List, Any, Iterable, Dict, Tuple
)

from loguru import logger
from omegaconf import DictConfig
import requests
import numpy as np
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm
import uuid

# 内部模块引入
from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.task_manager import adapter
from agentevolver.module.task_manager.adapter import OnflyRlDataset, to_rl_dataset
from agentevolver.module.task_manager.data_mixture import MixtureStrategy
from agentevolver.module.task_manager.filters.llm_filter import LlmFilter
from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.module.task_manager.filters.filters import NaiveTaskPostFilter, TaskPostFilter

# 统一使用 DashScopeClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.task_manager.base import TaskObjectiveRetrieval
from agentevolver.module.task_manager.strategies.random import LlmRandomSamplingExploreStrategy
from agentevolver.module.task_manager.strategies.api_driven import ApiDrivenExploreStrategy

from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from verl.utils.dataset.rl_dataset import RLHFDataset
from agentevolver.utils.debug_utils import debug_log
from agentevolver.module.task_manager.filters.api_llm_pre_filter import LlmQualityPreFilter

# 全局 IO 锁
io_lock = threading.Lock()

# --- 采样权重配置 ---
LEVEL_WEIGHTS = {
    "Very High": 10.0,
    "High": 5.0,
    "Medium": 2.0,
    "Low": 1.0,
    "Very Low": 0.5,
}

def get_weighted_api_sample(api_dict, k=5):
    """基于 Generality 等级进行加权采样"""
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
    num_explore_threads: int
    n: int

class RewardProps(TypedDict):
    original_grader: str
    synthetic_grader: str

def get_exploration_strategy(name: str, strategy_args, *, tokenizer, config, llm_client, env_profile) -> TaskExploreStrategy:
    """策略工厂"""
    logger.info(f"loading exploration strategy {name}")
    if name == "random":
        return LlmRandomSamplingExploreStrategy(
            tokenizer=tokenizer, config=config, env_profile=env_profile, **strategy_args
        )
    elif name == "api_driven":
        return ApiDrivenExploreStrategy(
            tokenizer=tokenizer, config=config, llm_client=llm_client, env_profile=env_profile, **strategy_args
        )
    else:
        raise NotImplementedError(f"exploration strategy {name} not implemented")

# ================= ApiDrivenPipeline (完整逻辑实现) =================

class ApiDrivenPipeline:
    def __init__(self, manager: "TaskManager", tasks: Sequence[Task], show_progress: bool = False, resume_file: Optional[str] = None):
        self.manager = manager
        self.tasks = tasks
        self.show_progress = show_progress
        self.mem_lock = threading.Lock()
        
        self.strategy_args = manager._config.task_manager.get('exploration_strategy_args', {})
        self.a = self.strategy_args.get('a', 1)
        self.b = self.strategy_args.get('b', 1)
        self.debug_mode = False 
        
        gen_output_dir = os.environ.get("GEN_OUTPUT_DIR")
        if gen_output_dir:
            base_name = "generated_tasks"
            self.resume_file = os.path.join(gen_output_dir, base_name)
        else:
            self.resume_file = resume_file or '.generate_task_api'
            
        self._init_paths()
        self.api_knowledge = getattr(self.manager._exploration_strategy, 'api_knowledge', {})
        self.active_apps_set = getattr(self.manager._exploration_strategy, 'active_apps', set(self.api_knowledge.keys()))

    def _init_paths(self):
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
        if os.path.exists(path):
            tasks_list = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            tasks_list.append(Task.parse_obj(data))
                        except: pass
            return tasks_list
        return None

    def _thread_safe_append(self, path: str, items: List[Any]):
        if not items: return
        with io_lock:
            try:
                with open(path, 'a', encoding='utf-8') as f:
                    for item in items:
                        obj = item.dict() if hasattr(item, 'dict') else item
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Failed to append to {path}: {e}")

    # --- Worker Methods ---
    def _worker_generate_intra(self, idx: int, app_name: str, seed_task: Task) -> List[Task]:
        try:
            base_task = copy.deepcopy(seed_task)
            if base_task.metadata is None: base_task.metadata = {}
            base_task.metadata['thread_index'] = idx % self.manager._num_exploration_threads
            tasks = self.manager._exploration_strategy.generate_intra_task(app_name, task=base_task)
            for sub_idx, t in enumerate(tasks or []):
                t.metadata["data_id"] = f"gen_intra_{idx}_{sub_idx}"
            return tasks or []
        except Exception as e:
            logger.error(f"[Intra-Gen] Error idx {idx}: {e}"); return []

    def _worker_generate_cross(self, idx: int, target_apps: List[str], seed_task: Task) -> List[Task]:
        try:
            base_task = copy.deepcopy(seed_task)
            if base_task.metadata is None: base_task.metadata = {}
            base_task.metadata['thread_index'] = idx % self.manager._num_exploration_threads
            tasks = self.manager._exploration_strategy.generate_cross_task(target_apps, task=base_task)
            for sub_idx, t in enumerate(tasks or []):
                t.metadata["data_id"] = f"gen_cross_{idx}_{sub_idx}"
            return tasks or []
        except Exception as e:
            logger.error(f"[Cross-Gen] Error idx {idx}: {e}"); return []

    def _worker_explore_intra(self, task: Task) -> List[TaskObjective]:
        try:
            data_id = task.metadata.get("data_id", f"gen_{uuid.uuid4().hex[:6]}")
            trajectories = self.manager._exploration_strategy.explore(task, data_id, data_id)
            success_traj = trajectories[0] if (trajectories and trajectories[0].reward and trajectories[0].reward.outcome >= 0.7) else None
            if not success_traj: return []

            reward_val = success_traj.reward.outcome
            reward_info = {"outcome": reward_val, "reason": getattr(success_traj.reward, "reason", "")}
            raw_gt_steps = [s.dict() if hasattr(s, 'dict') else s for s in success_traj.steps]
            
            # 1. 直接验证并保存元数据
            direct_obj = self.manager._exploration_strategy.verify_direct_gt(task, success_traj)
            origin_gt = None
            if direct_obj:
                direct_obj.task.raw_trajectory = raw_gt_steps
                direct_obj.task.origin_query = task.query
                direct_obj.task.metadata.update({"source_data_id": data_id, "execution_reward": reward_info, "generation_type": "direct_verify"})
                self._thread_safe_append(self.intra_direct_path, [direct_obj])
                origin_gt = direct_obj.task.ground_truth 

            # 2. 演化总结 (Strategy 内部已负责 summary_analysis_process 赋值)
            evolved_results = self.manager._exploration_strategy.summarize(task, success_traj)
            if evolved_results:
                for res in evolved_results:
                    res.task.raw_trajectory = raw_gt_steps
                    res.task.origin_ground_truth = origin_gt
                    res.task.origin_query = task.query
                    res.reward = reward_val
                    res.task.metadata.update({
                        "execution_reward": reward_info, "source_data_id": data_id,
                        "generation_type": "evolved", "has_verified_origin": (origin_gt is not None)
                    })
                self._thread_safe_append(self.intra_evolved_path, evolved_results)
            return evolved_results or []
        except Exception as e:
            logger.error(f"Intra Explore failed: {e}"); return []

    def _worker_explore_cross(self, task: Task) -> List[TaskObjective]:
        try:
            data_id = task.metadata.get("data_id", f"gen_cross_{uuid.uuid4().hex[:6]}")
            trajectories = self.manager._exploration_strategy.explore(task, data_id, data_id)
            success_traj = trajectories[0] if (trajectories and trajectories[0].reward and trajectories[0].reward.outcome >= 0.7) else None
            if not success_traj: return []

            reward_val = success_traj.reward.outcome
            reward_info = {"outcome": reward_val, "reason": getattr(success_traj.reward, "reason", "")}
            raw_gt_steps = [s.dict() if hasattr(s, 'dict') else s for s in success_traj.steps]
            
            direct_obj = self.manager._exploration_strategy.verify_direct_gt(task, success_traj)
            origin_gt = None
            if direct_obj:
                direct_obj.task.raw_trajectory = raw_gt_steps
                direct_obj.task.origin_query = task.query
                direct_obj.task.metadata.update({"source_data_id": data_id, "execution_reward": reward_info, "generation_type": "direct_verify_cross"})
                self._thread_safe_append(self.cross_direct_path, [direct_obj])
                origin_gt = direct_obj.task.ground_truth

            evolved_results = self.manager._exploration_strategy.summarize(task, success_traj)
            if evolved_results:
                for res in evolved_results:
                    res.task.raw_trajectory = raw_gt_steps
                    res.task.origin_ground_truth = origin_gt
                    res.task.origin_query = task.query
                    res.reward = reward_val
                    res.task.metadata.update({
                        "execution_reward": reward_info, "source_data_id": data_id,
                        "generation_type": "evolved_cross", "involved_apps": task.metadata.get("target_apps", [])
                    })
                self._thread_safe_append(self.cross_evolved_path, evolved_results)
            return evolved_results or []
        except Exception as e:
            logger.error(f"Cross Explore failed: {e}"); return []

    def run(self) -> List[TaskObjective]:
        """主运行管线：Intra -> Cross -> 聚合"""
        target_files = [self.intra_direct_path, self.cross_direct_path, self.intra_evolved_path, self.cross_evolved_path]
        if all(os.path.exists(p) for p in target_files):
            logger.info("⚡ [中间级拦截] 检测到底层产物齐备，直接聚合。")
        else:
            # === PART 1: INTRA-DOMAIN ===
            logger.info("=== PART 1: Intra-Domain Phase ===")
            valid_apps_intra = [app for app in sorted(self.active_apps_set) if self.api_knowledge.get(app, {}).get("apis")]
            intra_pool = (list(copy.copy(self.tasks)) * int(self.a + 1))[:int(len(self.tasks) * self.a)]
            
            gen_intra = self._load_intermediate_tasks(self.intra_gen_path) or []
            if len(gen_intra) < len(intra_pool):
                with ThreadPoolExecutor(max_workers=self.manager._num_exploration_threads) as pool:
                    futures = [pool.submit(self._worker_generate_intra, i, valid_apps_intra[i % len(valid_apps_intra)], intra_pool[i]) 
                               for i in range(len(gen_intra), len(intra_pool))]
                    for f in tqdm(as_completed(futures), total=len(futures), desc="Intra Gen"):
                        res = f.result()
                        if res: self._thread_safe_append(self.intra_gen_path, res); gen_intra.extend(res)
            
            filt_intra = self.manager._apply_filters_with_report(gen_intra, self.manager.api_llm_pre_filter, "Intra-Pre")
            with ThreadPoolExecutor(max_workers=self.manager._num_exploration_threads) as pool:
                list(tqdm(pool.map(self._worker_explore_intra, filt_intra), total=len(filt_intra), desc="Intra Explore"))

            # === PART 2: CROSS-DOMAIN ===
            logger.info("=== PART 2: Cross-Domain Phase ===")
            target_cross_count = int(len(self.tasks) * self.b)
            gen_cross = self._load_intermediate_tasks(self.cross_gen_path) or []
            if len(gen_cross) < target_cross_count:
                with ThreadPoolExecutor(max_workers=self.manager._num_exploration_threads) as pool:
                    futures = []
                    valid_apps_list = list(self.active_apps_set)
                    # [修改] 移除了 valid_candidate_tasks 过滤逻辑，直接使用 self.tasks
                    for i in range(len(gen_cross), target_cross_count):
                        target_apps = random.sample(valid_apps_list, min(random.choice([2, 3]), len(valid_apps_list)))
                        # 核心修改：直接使用原始种子任务 self.tasks[i % len(self.tasks)]
                        futures.append(pool.submit(self._worker_generate_cross, i, target_apps, self.tasks[i % len(self.tasks)]))
                    for f in tqdm(as_completed(futures), total=len(futures), desc="Cross Gen"):
                        res = f.result()
                        if res: self._thread_safe_append(self.cross_gen_path, res); gen_cross.extend(res)
            
            filt_cross = self.manager._apply_filters_with_report(gen_cross, self.manager.api_llm_pre_filter, "Cross-Pre")
            with ThreadPoolExecutor(max_workers=self.manager._num_exploration_threads) as pool:
                list(tqdm(pool.map(self._worker_explore_cross, filt_cross), total=len(filt_cross), desc="Cross Explore"))

        # === Final Aggregation & Post-Filter ===
        total_results = []
        for path in target_files:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip(): total_results.append(TaskObjective.parse_obj(json.loads(line)))

        return self.manager._apply_post_filter(total_results)

# ================= TaskManager (调度中心) =================

class TaskManager(object):
    def __init__(self, config: DictConfig, exploration_strategy: str, env_profile: EnvProfile, 
                 exploration_strategy_args, llm_client: DashScopeClient, old_retrival: TaskObjectiveRetrieval,
                 mixture_strategy: MixtureStrategy, reward_config: RewardProps, tokenizer,
                 env_service_url: str, **kwargs):
        
        self._config, self._tokenizer, self._llm_client = config, tokenizer, llm_client
        
        # [修改] 硬约束：强制注入 100 RPM 和 20 并发限制
        self._llm_client._max_rpm = 100
        self._llm_client._semaphore = threading.BoundedSemaphore(20)

        self._exploration_strategy = get_exploration_strategy(
            exploration_strategy, exploration_strategy_args, tokenizer=tokenizer, 
            config=config, llm_client=llm_client, env_profile=env_profile
        )
        
        self._num_exploration_threads = kwargs.get("num_explore_threads", 5)
        self._n = kwargs.get("n", 1)
        self._tasks: list[Task] = [] 

        self.api_llm_pre_filter = [LlmQualityPreFilter(llm_client, num_threads=self._num_exploration_threads)]
        self._post_filter = [LlmFilter(env_service_url, llm_client, self._num_exploration_threads, tokenizer=tokenizer, config=config)]
        self._realtime_filters = [NaiveTaskPostFilter()]
        
        self._already_loaded_target_file = False

    def generate_task(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """最高级拦截逻辑：存在最终文件则直接加载"""
        out_dir = os.environ.get("GEN_OUTPUT_DIR", "")
        target_file = os.path.join(out_dir, "tasks_explored.train.json") if out_dir else "tasks_explored.train.json"
        
        if os.path.exists(target_file):
            if self._already_loaded_target_file: return []
            logger.info(f"⚡ [Supreme Interceptor] Loading existing dataset: {target_file}")
            self._already_loaded_target_file = True
            with open(target_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [TaskObjective.parse_obj(item) for item in data]

        if isinstance(self._exploration_strategy, ApiDrivenExploreStrategy):
            pipeline = ApiDrivenPipeline(self, tasks, show_progress, resume_file)
            return pipeline.run()
        else:
            return self._generate_task_random(tasks, show_progress, resume_file)

    def _apply_post_filter(self, items: List[TaskObjective]) -> List[TaskObjective]:
        return self._apply_filters_with_report(items, self._post_filter, "Final-Judge")

    def _apply_filters_with_report(self, items: List[Any], filters: List[Any], stage_name: str) -> List[Any]:
        if not items: return []
        curr = items
        for f in filters: curr = f.filter(curr)
        return curr

    def load_tasks(self, tasks: Sequence[Task]):
        self._tasks.extend(tasks); logger.info(f"Loaded {len(self._tasks)} tasks.")

    def _compute_tasks_hash(self, tasks: Sequence[Task]) -> str:
        return hashlib.md5("|".join([f"{t.task_id}" for t in tasks]).encode()).hexdigest()

# ================= 数据集适配器 =================

class FullDataset(Dataset):
    def __init__(self, manager, mixture_strategy, reward_config, cache_path=None, *, tokenizer, config, processor):
        self._manager, self._tokenizer, self._config, self._processor = manager, tokenizer, config, processor
        self._tasks, self._mixture_strategy, self._reward_config, self._cache_path = manager.seed_task_objectives, mixture_strategy, reward_config, cache_path
        self._synthetic_objectives = []
        
        if self._mixture_strategy.need_synthetic:
            if self._cache_path and os.path.exists(self._cache_path): self.load_from_file()
            else:
                self.reload_new_task()
                if self._cache_path: self.save_to_file()
        self._rebuild_dataset()

    def load_from_file(self):
        with open(self._cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = TaskObjective.parse_obj(json.loads(line))
                    obj.task.evaluator = self._reward_config["synthetic_grader"]
                    self._synthetic_objectives.append(obj)

    def reload_new_task(self):
        self._synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        for item in self._synthetic_objectives: item.task.evaluator = self._reward_config["synthetic_grader"]

    def _rebuild_dataset(self):
        self._objectives = self._mixture_strategy.mix_data(self._synthetic_objectives, self._tasks)
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)

    def __getitem__(self, idx): return self._dataset[idx]
    def __len__(self): return len(self._dataset)

class AutoReloadDataset(IterableDataset):
    def __init__(self, manager, tasks, bs, *, tokenizer, config, processor):
        self._manager, self._bs = manager, bs
        self._task_iter = iter(tasks) 
        self._tokenizer, self._config, self._processor = tokenizer, config, processor
        self._dataset = OnflyRlDataset(release_used_dataset=True)

    def reload(self):
        delta = []
        for _ in range(self._bs):
            try: delta.append(next(self._task_iter))
            except StopIteration: break
        if not delta: return 0
        ls = self._manager.generate_task(delta)
        if not ls: return 0
        self._dataset.append_dataset(to_rl_dataset(ls, self._tokenizer, self._config, self._processor))
        return self._dataset.num_rest_data

    def __iter__(self): return self
    def __next__(self):
        if self._dataset.num_rest_data == 0 and self.reload() == 0: raise StopIteration
        return next(self._dataset)