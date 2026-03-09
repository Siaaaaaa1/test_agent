import json
import os
import random
import time
import itertools
import threading
import copy
from typing import List, Dict, Any, Optional, Set, Callable, Union, Tuple
from types import SimpleNamespace
from collections import defaultdict

from loguru import logger
from omegaconf import DictConfig
import traceback

# 基础模块
from agentevolver.module.task_manager.base import LlmClient
from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from agentevolver.utils.utils import extract_json_from_str
from agentevolver.utils.debug_utils import debug_log
from agentevolver.module.task_manager.rewards.integrated_reward import IntegratedRewardCalculator
# 环境与执行模块
from agentevolver.module.env_manager.env_worker import EnvWorker, TrajExpConfig
from agentevolver.module.task_manager.agent_flow import ModifiedAgentFlow

# Prompt 与 Profile 相关
from agentevolver.module.task_manager.prelude_profiles import appworld, bfcl, webshop
from agentevolver.module.task_manager.strategies.api_driven.prompts import (
    INTRA_DOMAIN_SELECTOR_PROMPT,
    parse_intra_selector,
    INTRA_DOMAIN_GENERATOR_PROMPT,
    parse_intra_generator,
    CROSS_DOMAIN_SELECTOR_PROMPT,
    parse_cross_selector,
    CROSS_DOMAIN_GENERATOR_PROMPT,
    parse_cross_generator,
    get_agent_interaction_system_prompt,
    get_intra_schema,
    get_cross_schema,
)
from agentevolver.module.task_manager.strategies.api_driven.prompts.prompt_summarize import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)

# [修改] 使用我们新实现的 DashScopeClient
from agentevolver.client.llm_client import DashScopeClient

from agentevolver.module.task_manager.strategies.api_driven.prompts.prompt_summarize_first import (
    get_direct_verify_prompt,
    parse_direct_verification
)

UNIVERSAL_INFO_PROVIDERS = {"notes", "gmail", "simple_messages", "calendar", "contacts"}

class ApiDrivenExploreStrategy(TaskExploreStrategy):
    """
    API 驱动的探索策略类
    """

    VALID_CROSS_COMBINATIONS = [
        ['venmo', 'splitwise'], ['venmo', 'gmail'], ['venmo', 'phone'], ['venmo', 'simple_note'],
        ['amazon', 'venmo'], ['amazon', 'splitwise'], ['amazon', 'todoist'], ['amazon', 'simple_note'], ['amazon', 'gmail'], ['amazon', 'phone'],
        ['spotify', 'phone'], ['spotify', 'gmail'], ['spotify', 'simple_note'],
        ['gmail', 'todoist'], ['gmail', 'file_system'], ['gmail', 'simple_note'], ['gmail', 'phone'],
        ['simple_note', 'spotify'], ['simple_note', 'gmail'], ['simple_note', 'phone'], ['simple_note', 'todoist'],
        ['phone', 'gmail'], ['phone', 'simple_note'], ['phone', 'todoist'],
        ['todoist', 'simple_note'], ['todoist', 'gmail'], ['todoist', 'phone'],
        ['splitwise', 'venmo'], ['splitwise', 'gmail'], ['splitwise', 'phone'], ['splitwise', 'simple_note'],
        ['file_system', 'gmail'], ['file_system', 'simple_note'], ['file_system', 'todoist']
    ]

    def _get_valid_app_chain(self, target_apps: List[str]) -> Optional[List[str]]:
        valid_edges = set(tuple(comb) for comb in self.VALID_CROSS_COMBINATIONS)
        for perm in itertools.permutations(target_apps):
            is_valid_chain = True
            for i in range(len(perm) - 1):
                if (perm[i], perm[i+1]) not in valid_edges:
                    is_valid_chain = False
                    break
            if is_valid_chain:
                return list(perm) 
        return None

    def __init__(self, tokenizer, config: DictConfig, llm_client: Optional[DashScopeClient] = None, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self._env_profile = kwargs.get("env_profile")

        # [修改] 初始化 DashScopeClient。如果外部没传，内部新建。
        # generate/summarize 默认使用 qwen3.5-plus
        if llm_client:
            self.llm_client = llm_client
        else:
            logger.info("[ApiDriven] Initializing default DashScopeClient (qwen3.5-plus) for Gen/Sum.")
            self.llm_client = DashScopeClient(model_name="qwen3.5-plus")
            
        self._summarize_client = kwargs.get("llm_client_summarize", self.llm_client)

        # explore 阶段同样使用该 client，但在调用时会根据策略切换 model_name
        self.explore_client = self.llm_client 
        
        self._stats_lock = threading.Lock()
        self._total_finished_tasks = 0
        self._model_performance = defaultdict(lambda: {"attempts": 0, "success": 0})

        self._max_llm_retries = kwargs.get("max_llm_retries", 5)
        self._lock = threading.Lock() 
        
        self.api_knowledge_path = kwargs.get(
            "api_knowledge_path", 
            "./agentevolver/preprocess/output/appworld_tool_manual.json"
        )
        self.task_labels_path = kwargs.get(
            "task_labels_path", 
            "./agentevolver/preprocess/output/task_app_labels_train.json"
        )
        
        base_memory_dir = "data/memory/api_driven"
        self.intra_memory_path = kwargs.get("intra_memory_path", os.path.join(base_memory_dir, "intra_domain_success.json"))
        self.cross_memory_path = kwargs.get("cross_memory_path", os.path.join(base_memory_dir, "cross_domain_success.json"))
        
        self.active_apps = set(kwargs.get("active_apps", ['amazon','gmail','spotify','venmo','simple_note','todoist','splitwise','phone','file_system']))
        self.api_knowledge = self._load_json(self.api_knowledge_path)

        self.sandbox_ids_pool = self._load_sandbox_task_ids(self.task_labels_path)
        self.sandbox_id_iterator = itertools.cycle(self.sandbox_ids_pool)
        
        self.intra_memory_data = self._load_json(self.intra_memory_path)
        self.explored_intra_apps = set(self.intra_memory_data.get("explored_apps", []))
        self.cross_memory_data = self._load_json(self.cross_memory_path)
        
        self.env_profile_name = self.config.get("env_service", {}).get("env_type", "appworld")

        logger.info(f"[ApiDriven] Strategy initialized with DashScopeClient.")

    def _record_model_result(self, model_name: str, is_success: bool):
        with self._stats_lock:
            self._model_performance[model_name]["attempts"] += 1
            if is_success:
                self._model_performance[model_name]["success"] += 1

    def _report_progress_if_needed(self):
        with self._stats_lock:
            self._total_finished_tasks += 1
            if self._total_finished_tasks % 50 == 0:
                logger.info(f"\n====== Model Performance Report (Processed {self._total_finished_tasks} Tasks) ======")
                logger.info(f"{'Model Name':<30} | {'Calls':<6} | {'Succ':<6} | {'Rate':<8}")
                logger.info("-" * 60)
                sorted_stats = sorted(self._model_performance.items(), key=lambda x: x[1]['attempts'], reverse=True)
                for model, stats in sorted_stats:
                    rate = (stats["success"] / stats["attempts"] * 100) if stats["attempts"] > 0 else 0.0
                    logger.info(f"{model:<30} | {stats['attempts']:<6} | {stats['success']:<6} | {rate:.2f}%")
                logger.info("=" * 60 + "\n")

    def _get_model_idle_score(self) -> Tuple[int, int]:
        """
        [修改] 基于新 DashScopeClient 的信号量获取当前空闲度。
        由于所有请求通过同一个 client 实例，直接返回该实例的状态。
        """
        try:
            concurrency_free = self.explore_client._semaphore._value
            with self.explore_client._rate_limit_lock:
                now = time.time()
                valid_ts = [t for t in self.explore_client._request_timestamps if now - t < 60.0]
                rpm_free = self.explore_client._max_rpm - len(valid_ts)
            return (concurrency_free, rpm_free)
        except:
            return (0, 0)

    def explore(self, task: Task, data_id: str, rollout_id: str) -> List[Trajectory]:
        """
        探索核心逻辑：实现 qwen3.5-plus -> qwen3-max 的阶梯回退。
        """
        if task.metadata is None:
            task.metadata = {}
            
        real_sandbox_id = task.metadata.get("env_sandbox_id") or self.get_next_sandbox_id()
        task.metadata["env_sandbox_id"] = real_sandbox_id
            
        debug_log(self.config, "api_explore_start", {
            "task_id": task.task_id,
            "real_sandbox_id": real_sandbox_id
        })

        # [修改] 只有两个候选模型：默认 qwen3.5-plus，失败则换 qwen3-max
        candidate_models = ["qwen3.5-plus", "qwen3-max"]
        
        last_trajectory = None
        max_steps = self.config.get("max_explore_step", 50) 

        for model_name in candidate_models:
            # 记录尝试
            logger.info(f"[Explore] Task {data_id}: Trying model '{model_name}'...")
            
            # 检查当前空闲情况（虽然 DashScopeClient 内部会阻塞等待，但在外部记录一下）
            c_free, r_free = self._get_model_idle_score()
            logger.debug(f"[Explore] Client Status - Concurrency Free: {c_free}, RPM Free: {r_free}")

            thread_idx = task.metadata.get('thread_index', 0)
            env_worker = EnvWorker(task=task, config=self.config, thread_index=thread_idx, tokenizer=self.tokenizer)

            # 配置采样参数
            sampling_params = {
                "model": model_name,
                "temperature": self.config.get("exploration_llm_temperature", 0.7),
            }
            if model_name == "qwen3-max":
                 sampling_params.update({
                     "top_p": self.config.get("exploration_llm_top_p", 0.9),
                     "top_k": self.config.get("exploration_llm_top_k", 50),
                 })

            llm_chat_fn = self._get_llm_chat_fn(self.explore_client, sampling_params=sampling_params)
            reward_calculator = IntegratedRewardCalculator(task=task)

            agent_flow = ModifiedAgentFlow(
                llm_chat_fn=llm_chat_fn,
                tokenizer=self.tokenizer,
                config=self.config,
                enable_context_generator=False,
                reward_calculator=reward_calculator 
            )
            agent_flow._reward_calculator = reward_calculator
            agent_flow.max_steps = max_steps

            try:
                system_prompt = get_agent_interaction_system_prompt(self._env_profile)
                trajectory = env_worker.execute(
                    data_id=data_id, rollout_id=rollout_id,
                    traj_exp_config=TrajExpConfig(add_exp=False),
                    agent_flow=agent_flow,
                    tmux={'step': [0], 'token': [0]},
                    stop=[False],
                    system_prompt=system_prompt,
                )
                
                last_trajectory = trajectory
                is_success = False
                
                if trajectory and trajectory.reward:
                    score = trajectory.reward.outcome
                    logger.info(f"📝 [Judge] Task: {data_id} | Model: {model_name} | Score: {score}")
                    if score >= 0.8:
                        is_success = True
                
                self._record_model_result(model_name, is_success)
                
                if is_success:
                    logger.info(f"[Explore] SUCCEEDED with {model_name}.")
                    self._report_progress_if_needed()
                    return [trajectory]
                else:
                    logger.warning(f"[Explore] FAILED with {model_name}. Falling back...")
                    continue

            except Exception as e:
                logger.error(f"[Explore] Critical Error with {model_name}: {e}")
                self._record_model_result(model_name, False)
                continue

        self._report_progress_if_needed()
        return [last_trajectory] if last_trajectory else []

    def summarize(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        总结逻辑统一使用 qwen3.5-plus。
        """
        if not trajectory or not trajectory.steps: return []
        phase = task.metadata.get("phase", "unknown")
        
        # 确保使用总结客户端（默认 qwen3.5-plus）
        results = []
        if phase == "intra":
            results = self.summarize_intra(task, trajectory)
        elif phase == "extra":
            results = self.summarize_cross(task, trajectory)
        
        return results

    def get_next_sandbox_id(self) -> str:
        try:
            return next(self.sandbox_id_iterator)
        except:
            return "train_001"

    def _get_enhanced_context(self, app_name: str, anchor_api_names: List[str]) -> Tuple[str, str, str]:
        global_lines = []
        for app, details in self.api_knowledge.items():
            desc = details.get("description", "No description available.")
            global_lines.append(f'APP: "{app}"\ndescription: "{desc}"')
        all_apps_info = "\n\n".join(global_lines)

        target_app_data = self.api_knowledge.get(app_name, {})
        target_app_apis = target_app_data.get("apis", {})
        
        api_lines = []
        for api_key, details in target_app_apis.items():
            desc = details.get("description", "No description available.")
            full_call_name = details.get("call_name", api_key)
            api_lines.append(f"{full_call_name}: {desc}")
        target_app_apis_info = "\n".join(api_lines)

        anchor_details_list = []
        for input_api_name in anchor_api_names:
            short_name = input_api_name.split('.')[-1] if "." in input_api_name else input_api_name
            api_data = target_app_apis.get(short_name)
            if api_data:
                params_str = json.dumps(api_data.get("parameters", []), indent=2)
                returns_str = json.dumps(api_data.get("returns", {}), indent=2)
                block = f'api_name: "{api_data.get("call_name", short_name)}"\ndescription: "{api_data.get("description", "")}"\nparameters: {params_str}\nreturns: {returns_str}'
                anchor_details_list.append(block)
        
        anchor_apis_detailed_info = "\n\n---\n\n".join(anchor_details_list) if anchor_details_list else "None provided."
        return all_apps_info, target_app_apis_info, anchor_apis_detailed_info
    
    def _fetch_real_table_data(self, sandbox_id: str, app_tables: Dict[str, List[str]]) -> str:
        import requests
        fetched_data = {}
        try:
            env_url = self.config.get("env_service", {}).get("env_url", "http://localhost:8080")
            response = requests.post(f"{env_url}/fetch_db_data", json={"sandbox_id": sandbox_id, "app_tables": app_tables}, timeout=15.0)
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    fetched_data = result.get("data", {})
        except Exception as e:
            logger.error(f"[Data Fetch] Error: {e}")
        return json.dumps(fetched_data, indent=2, ensure_ascii=False) if fetched_data else "{}"

    def generate_intra_task(self, app_name: str, task: Task = None) -> List[Task]:
        """
        [修改] 路径生成逻辑仅运行一次默认的 qwen3.5-plus。
        """
        all_apis = list(self.api_knowledge.get(app_name, {}).get("apis", {}).keys())
        if len(all_apis) < 2: return []

        real_sandbox_id = self.get_next_sandbox_id()
        candidate_groups = [random.sample(all_apis, min(random.choice([2, 3]), len(all_apis))) for _ in range(3)]
        groups_str = "\n".join([f"Group {i+1}: {g}" for i, g in enumerate(candidate_groups)])
        all_apps_info, target_app_apis_info, _ = self._get_enhanced_context(app_name, [])

        # Stage 1: Selector
        db_schema_overview = get_intra_schema([app_name])
        selector_prompt = INTRA_DOMAIN_SELECTOR_PROMPT.format(
            CANDIDATE_COUNT=3, APP_NAME=app_name, ALL_APPS_DESC=all_apps_info,
            TARGET_APP_API_DESCS=target_app_apis_info, CANDIDATE_GROUPS_STR=groups_str,
            DB_SCHEMA_OVERVIEW=db_schema_overview
        )
        
        # 强制使用 qwen3.5-plus
        sel_response = self._chat_with_retry(messages=[{"role": "user", "content": selector_prompt}], model="qwen3.5-plus")
        if not sel_response: return []
        
        selection_data = parse_intra_selector(sel_response.content)
        if not selection_data or "selected_apis" not in selection_data: return []

        selected_apis = selection_data["selected_apis"]
        required_tables = selection_data.get("required_tables", [])
        db_content = self._fetch_real_table_data(real_sandbox_id, {app_name: required_tables})
        _, _, anchor_apis_detailed_info = self._get_enhanced_context(app_name, selected_apis)

        # Stage 2: Generator
        generator_prompt = INTRA_DOMAIN_GENERATOR_PROMPT.format(
            APP_NAME=app_name, SELECTED_APIS=json.dumps(selected_apis),
            ALL_APIS_BRIEF=target_app_apis_info, ANCHOR_API_DETAILS=anchor_apis_detailed_info,
            DB_CONTENT=db_content
        )

        gen_response = self._chat_with_retry(messages=[{"role": "user", "content": generator_prompt}], model="qwen3.5-plus")
        if not gen_response: return []

        parsed_scenarios = parse_intra_generator(gen_response.content)
        generated_tasks = []
        for scenario in parsed_scenarios:
            new_task = copy.deepcopy(task) if task else Task()
            new_task.query = scenario.get("user_query", "")
            new_task.metadata = {
                "phase": "intra", "env_sandbox_id": real_sandbox_id, "target_app": app_name,
                "selected_apis_context": selected_apis, "required_tables": required_tables,
                "origin_query": new_task.query
            }
            generated_tasks.append(new_task)
        return generated_tasks

    def generate_cross_task(self, target_apps: List[str], task: Task = None) -> List[Task]:
        ordered_apps = self._get_valid_app_chain(target_apps)
        if not ordered_apps: return []
        target_apps = ordered_apps 

        total_apis = max(random.choices([2, 3, 4, 5], weights=[0.4, 0.3, 0.2, 0.1])[0], len(target_apps))
        num_candidates = 3 if total_apis in [2, 3] else 5
        real_sandbox_id = self.get_next_sandbox_id()

        apps_info_lines = []
        for app in target_apps:
            _, app_apis_info, _ = self._get_enhanced_context(app, [])
            apps_info_lines.append(f"--- APP: {app} ---\n{app_apis_info}")

        candidate_groups = []
        for _ in range(num_candidates):
            current_group = {}
            allocations = {app: 1 for app in target_apps}
            remaining = total_apis - len(target_apps)
            for _ in range(max(0, remaining)): allocations[random.choice(target_apps)] += 1
            for app, count in allocations.items():
                all_apis = list(self.api_knowledge.get(app, {}).get("apis", {}).keys())
                current_group[app] = random.sample(all_apis, min(count, len(all_apis)))
            candidate_groups.append(current_group)

        # Stage 1: Selector
        db_schema_overview = get_cross_schema(target_apps)
        selector_prompt = CROSS_DOMAIN_SELECTOR_PROMPT.format(
            CANDIDATE_COUNT=num_candidates, APP_COUNT=len(target_apps),
            APPS_INFO_STR="\n\n".join(apps_info_lines), CANDIDATE_GROUPS_STR="\n".join([f"Group {i+1}: {json.dumps(g)}" for i, g in enumerate(candidate_groups)]),
            DB_SCHEMA_OVERVIEW=db_schema_overview
        )

        sel_response = self._chat_with_retry(messages=[{"role": "user", "content": selector_prompt}], model="qwen3.5-plus")
        if not sel_response: return []

        selection_data = parse_cross_selector(sel_response.content)
        if not selection_data or "selected_apis" not in selection_data: return []

        selected_apis = selection_data["selected_apis"]
        required_tables = selection_data.get("required_tables", {})
        db_content = self._fetch_real_table_data(real_sandbox_id, required_tables)
        
        anchor_details_combined = []
        all_apis_brief_combined = []
        for app in target_apps:
            _, app_apis_info, anchor_detail = self._get_enhanced_context(app, selected_apis.get(app, []))
            anchor_details_combined.append(f"[{app} Details]\n{anchor_detail}")
            all_apis_brief_combined.append(f"[{app} APIs]\n{app_apis_info}")
        
        # Stage 2: Generator
        generator_prompt = CROSS_DOMAIN_GENERATOR_PROMPT.format(
            APPS_NAMES=", ".join(target_apps), SELECTED_APIS_JSON=json.dumps(selected_apis),
            ALL_APIS_BRIEF="\n".join(all_apis_brief_combined), ANCHOR_API_DETAILS="\n".join(anchor_details_combined),
            DB_CONTENT=db_content
        )

        gen_response = self._chat_with_retry(messages=[{"role": "user", "content": generator_prompt}], model="qwen3.5-plus")
        if not gen_response: return []

        parsed_scenarios = parse_cross_generator(gen_response.content)
        generated_tasks = []
        for scenario in parsed_scenarios:
            new_task = copy.deepcopy(task) if task else Task()
            new_task.query = scenario.get("user_query", "")
            new_task.metadata = {
                "phase": "extra", "env_sandbox_id": real_sandbox_id, "target_apps": target_apps,
                "selected_apis_context": selected_apis, "required_tables": required_tables,
                "origin_query": new_task.query
            }
            generated_tasks.append(new_task)
        return generated_tasks

    def summarize_intra(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        [重点恢复] 完整保留所有 Metadata 赋值，确保科研数据不丢失。
        """
        if not trajectory or not trajectory.steps: return []
        
        # 强制使用 qwen3.5-plus 进行总结
        client = self._summarize_client
        llm_fn = self._get_llm_chat_fn(client, sampling_params={"model": "qwen3.5-plus"})
        
        # 数据脱敏
        masked_trajectory = copy.deepcopy(trajectory)
        if len(masked_trajectory.steps) > 2:
            for i in [1, 2]:
                if i < len(masked_trajectory.steps) and masked_trajectory.steps[i].get('role') == 'user':
                    masked_trajectory.steps[i]['content'] = '[MASKED]'

        system_prompt, user_prompt = get_task_summarize_prompt(
            [masked_trajectory], old_objectives=task.query, profile=self._env_profile
        )
        
        try:
            llm_response = llm_fn(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
            llm_output = llm_response["content"]
        except Exception as e:
            logger.error(f"[Summarize Intra] LLM call failed: {e}")
            return []
        
        task_copy = task.copy()
        task_copy.evaluator = 'synthetic'
        tasks = parse_tasks_from_response(task_copy, llm_output)

        # 提取奖励信息
        reward_info = None
        if trajectory.reward:
            reward_info = {
                "outcome": trajectory.reward.outcome,
                "reason": getattr(trajectory.reward, "reason", "No reason provided")
            }

        # ✅ [完整恢复埋点] 
        for task_obj in tasks:
            if task_obj.task.metadata is None:
                task_obj.task.metadata = {}
            
            task_obj.task.origin_query = task.query
            task_obj.task.metadata["summary_analysis_process"] = llm_output
            task_obj.task.metadata["execution_reward"] = reward_info
            task_obj.task.metadata["source_data_id"] = task.metadata.get("data_id")
            task_obj.task.metadata["generation_type"] = "evolved"

        return tasks

    def summarize_cross(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        跨域探索反思与提取：确保记录跨 App 联动的元数据。
        """
        if not trajectory or not trajectory.steps:
            return []

        # 1. 强制使用总结专用客户端及模型
        client = self._summarize_client
        llm_fn = self._get_llm_chat_fn(client, sampling_params={"model": "qwen3.5-plus"})
        
        # 2. 数据脱敏（防止模型通过 Prompt 泄露直接获知答案）
        masked_trajectory = copy.deepcopy(trajectory)
        if len(masked_trajectory.steps) > 2:
            for i in [1, 2]:
                if i < len(masked_trajectory.steps) and masked_trajectory.steps[i].get('role') == 'user':
                    masked_trajectory.steps[i]['content'] = '[MASKED]'

        # 3. 构造 Prompt（get_task_summarize_prompt 会根据 profile 自动适配多 App 场景）
        system_prompt, user_prompt = get_task_summarize_prompt(
            [masked_trajectory], old_objectives=task.query, profile=self._env_profile
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            llm_response = llm_fn(messages=messages)
            llm_output = llm_response["content"]
        except Exception as e:
            logger.error(f"[Summarize Cross] LLM call failed: {e}")
            return []
        
        # 4. 解析结果
        task_copy = task.copy()
        task_copy.evaluator = 'synthetic'
        tasks = parse_tasks_from_response(task_copy, llm_output)

        # 5. 提取奖励反馈
        reward_info = None
        if trajectory.reward:
            reward_info = {
                "outcome": trajectory.reward.outcome,
                "reason": getattr(trajectory.reward, "reason", "No reason provided")
            }

        # 6. ✅ [元数据注入] 增加 cross 专有的 target_apps 记录
        for task_obj in tasks:
            if task_obj.task.metadata is None:
                task_obj.task.metadata = {}
            
            task_obj.task.origin_query = task.query
            task_obj.task.metadata["summary_analysis_process"] = llm_output
            task_obj.task.metadata["execution_reward"] = reward_info
            task_obj.task.metadata["source_data_id"] = task.metadata.get("data_id")
            task_obj.task.metadata["generation_type"] = "evolved"
            
            # 记录参与联动的 APPs，方便后续按 App 组合统计成功率
            task_obj.task.metadata["involved_apps"] = task.metadata.get("target_apps", [])
            task_obj.task.metadata["logic_pattern"] = task.metadata.get("logic_pattern", "unknown")

        return tasks

    def verify_direct_gt(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        if trajectory.reward and trajectory.reward.outcome < 0.3: return None
        llm_fn = self._get_llm_chat_fn(self._summarize_client, sampling_params={"model": "qwen3.5-plus"})
        system_prompt, user_prompt = get_direct_verify_prompt(task, trajectory, self._env_profile)
        try:
            llm_output = llm_fn(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])["content"]
            result = parse_direct_verification(llm_output)
            if result.get("is_valid"):
                new_task = task.copy()
                new_task.evaluator = 'synthetic'
                new_task.ground_truth = result.get("refined_code", "")
                return TaskObjective(task=new_task, confidence=result.get("confidence", 0.9), reward=trajectory.reward.outcome)
        except: pass
        return None

    def _chat_with_retry(self, messages: List[Dict], **kwargs) -> Optional[Any]:
        for i in range(self._max_llm_retries):
            try:
                response = self.llm_client.chat(messages=messages, **kwargs)
                if isinstance(response, str) and response.strip():
                    return SimpleNamespace(content=response)
            except: time.sleep(2 ** i)
        return None

    def _load_sandbox_task_ids(self, path: str) -> List[str]:
        if not os.path.exists(path): return ["train_001"]
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return [item["TaskID"] for item in json.load(f) if "TaskID" in item]
        except: return ["train_001"]

    def _load_json(self, path: str) -> Dict:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f: return json.load(f)
            except: pass
        return {}

    def _get_llm_chat_fn(self, llm_client: DashScopeClient, sampling_params: Optional[dict] = None) -> Callable:
        def llm_chat(messages: list[dict[str, str]], custom_sampling_params: Optional[dict] = None, **kwargs) -> dict:
            params = {**(sampling_params or {}), **(custom_sampling_params or {}), **kwargs}
            res = llm_client.chat_with_retry(messages=messages, max_retries=self._max_llm_retries, **params)
            return {"role": "assistant", "content": res or "Error"}
        return llm_chat