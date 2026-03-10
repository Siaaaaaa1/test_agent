import json
import os
import random
import time
import itertools
import threading
import copy
import requests  # [修改] 移至全局导入
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
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.task_manager.strategies.api_driven.prompts.prompt_summarize_first import (
    get_direct_verify_prompt,
    parse_direct_verification
)

UNIVERSAL_INFO_PROVIDERS = {"notes", "gmail", "simple_messages", "calendar", "contacts"}


class ApiDrivenExploreStrategy(TaskExploreStrategy):
    """
    API 驱动的探索策略类
    核心功能：
    1. 任务生成 (Generation)：基于预设的 App 和 API 知识库，通过 LLM 生成 Intra-domain (同领域) 和 Cross-domain (跨领域) 的指令任务。
    2. 任务执行 (Explore)：将生成的任务下发到沙盒环境中，使用多模型轮询的方式驱动 Agent 执行，收集环境反馈轨迹。
    3. 任务总结 (Summarize)：对执行成功的轨迹进行归纳总结，提取并提炼为高质量的强化学习/微调数据集。
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
        """
        基于 VALID_CROSS_COMBINATIONS 过滤并重排 APP，寻找合理的逻辑执行链 (A -> B -> C)。
        如果无法形成合法链条，则返回 None。
        """
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

    def __init__(self, tokenizer, config: DictConfig, llm_client: Optional[LlmClient] = None, **kwargs):
        """
        初始化策略类，挂载环境配置、加载知识库并初始化所需的大模型客户端以及各类路径配置。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self._env_profile = kwargs.get("env_profile")

        if llm_client:
            self.llm_client = llm_client
        else:
            logger.warning("[ApiDriven] No LlmClient passed, creating a standard LlmClient.")
            self.llm_client = LlmClient(config)
            
        self._summarize_client = kwargs.get("llm_client_summarize", self.llm_client)

        logger.info("[ApiDriven] Initializing DashScopeClient for 'explore' phase.")
        try:
            self.explore_client = DashScopeClient(
                model_name="qwen3.5-plus",
                temperature=config.get("exploration_llm_temperature", 0.7)
            )
        except Exception as e:
            logger.error(f"[ApiDriven] Failed to init DashScopeClient: {e}. Fallback to standard client.")
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
        if not self.api_knowledge:
            logger.warning(f"API Knowledge not found at {self.api_knowledge_path}.")

        self.sandbox_ids_pool = self._load_sandbox_task_ids(self.task_labels_path)
        self.sandbox_id_iterator = itertools.cycle(self.sandbox_ids_pool)
        
        self.intra_memory_data = self._load_json(self.intra_memory_path)
        self.explored_intra_apps = set(self.intra_memory_data.get("explored_apps", []))
        self.cross_memory_data = self._load_json(self.cross_memory_path)
        
        self.env_profile_name = self.config.get("env_service", {}).get("env_type", "appworld")

        logger.info(f"[ApiDriven] Initialized. Strategy ready.")

    def _record_model_result(self, model_name: str, is_success: bool):
        """记录单个大模型在环境中执行任务的成功或失败次数，保证多线程安全。"""
        with self._stats_lock:
            self._model_performance[model_name]["attempts"] += 1
            if is_success:
                self._model_performance[model_name]["success"] += 1

    def _report_progress_if_needed(self):
        """定期在控制台打印当前各个大模型的执行表现统计报表。"""
        with self._stats_lock:
            self._total_finished_tasks += 1
            current_count = self._total_finished_tasks
            
            if current_count % 50 == 0:
                logger.info(f"\n====== Model Performance Report (Processed {current_count} Tasks) ======")
                logger.info(f"{'Model Name':<40} | {'Calls':<6} | {'Succ':<6} | {'Fail':<6} | {'Rate':<8}")
                logger.info("-" * 80)
                
                sorted_stats = sorted(self._model_performance.items(), key=lambda x: x[1]['attempts'], reverse=True)
                
                for model, stats in sorted_stats:
                    attempts = stats["attempts"]
                    success = stats["success"]
                    fail = attempts - success
                    rate = (success / attempts * 100) if attempts > 0 else 0.0
                    logger.info(f"{model:<40} | {attempts:<6} | {success:<6} | {fail:<6} | {rate:.2f}%")
                logger.info("========================================================================\n")

    def _get_model_idle_score(self, model_name: str) -> Tuple[int, int]:
        """获取指定模型的空闲分数，用于在多个主模型之间做负载均衡路由。"""
        if not hasattr(self.explore_client, "_get_model_state"):
            return (0, 0)

        try:
            state = self.explore_client._get_model_state(model_name)
            semaphore = state.get("semaphore")
            concurrency_free = semaphore._value if semaphore else 0
            
            rate_lock = state.get("rate_lock")
            timestamps = state.get("timestamps")
            rpm_free = 0
            
            if rate_lock and isinstance(timestamps, list):
                with rate_lock:
                    now = time.time()
                    valid_timestamps = [t for t in timestamps if now - t < 60.0]
                    max_rpm = getattr(self.explore_client, "MAX_RPM", 30)
                    rpm_free = max_rpm - len(valid_timestamps)
            
            return (concurrency_free, rpm_free)
            
        except Exception as e:
            logger.warning(f"Failed to check idle score for {model_name}: {e}")
            return (0, 0)

    def explore(self, task: Task, data_id: str, rollout_id: str) -> List[Trajectory]:
        """
        执行探索任务的核心入口：将生成的自然语言 Query 下发至真实环境交由 Agent 解决。
        利用环境沙盒并在多模型之间实施 Fallback 机制。
        """
        if task.metadata is None:
            task.metadata = {}
            
        real_sandbox_id = task.metadata.get("env_sandbox_id")
        if not real_sandbox_id:
            real_sandbox_id = self.get_next_sandbox_id()
            task.metadata["env_sandbox_id"] = real_sandbox_id
            
        debug_log(self.config, "api_explore_start", {
            "task_id": task.task_id,
            "data_id": data_id,
            "phase": task.metadata.get('phase'),
            "real_sandbox_id": real_sandbox_id
        })

        # [修改] 改为从配置中动态获取候选模型，避免硬编码
        candidate_models = self.config.get("candidate_models", ["qwen3.5-plus"])
        
        last_trajectory = None
        max_steps = self.config.get("max_explore_step", 50) 

        for model_name in candidate_models:
            logger.info(f"[Explore] Task {data_id}: Trying model '{model_name}'...")

            thread_idx = 0
            if task.metadata and 'thread_index' in task.metadata:
                thread_idx = task.metadata['thread_index']
            
            env_worker = EnvWorker(
                task=task,
                config=self.config, 
                thread_index=thread_idx,
                tokenizer=self.tokenizer
            )

            sampling_params = {
                "temperature": self.config.get("exploration_llm_temperature", 0.5),
                "model": model_name,
            }
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
            agent_flow.max_model_len = self.config.get("max_model_len", 102400)

            try:
                system_prompt = get_agent_interaction_system_prompt(self._env_profile)

                trajectory = env_worker.execute(
                    data_id=data_id, 
                    rollout_id=rollout_id,
                    traj_exp_config=TrajExpConfig(add_exp=False),
                    agent_flow=agent_flow,
                    tmux={'step': [0], 'token': [0]},
                    stop=[False],
                    system_prompt=system_prompt,
                )
                
                last_trajectory = trajectory
                
                is_success = False
                current_score = 0.0
                
                if trajectory and trajectory.reward:
                    current_score = trajectory.reward.outcome
                    judge_reason = getattr(trajectory.reward, "reason", "No detailed reasoning provided")
                    logger.info(f"📝 [Judge Result] Task: {data_id} | Model: {model_name} | Score: {current_score}\nReasoning: {judge_reason}")
                    
                    if current_score >= 0.8:
                        is_success = True
                    else:
                        logger.warning(f"[Explore] Model {model_name} score {current_score} < 0.8. Marked as Fail.")
                else:
                    logger.warning(f"[Explore] Model {model_name} produced no reward object.")

                self._record_model_result(model_name, is_success)
                
                if is_success:
                    logger.info(f"[Explore] Model {model_name} SUCCEEDED (Score: {current_score}, Steps: {len(trajectory.steps)}). Returning result.")
                    self._report_progress_if_needed()
                    # [修改] 成功退出前清理环境资源
                    if hasattr(env_worker, 'env') and hasattr(env_worker.env, 'close'):
                        env_worker.env.close()
                    return [trajectory]
                else:
                    logger.warning(f"[Explore] Model {model_name} FAILED or Score too low. Retrying with next model...")
                    # [修改] 失败重试前清理已占用的环境资源
                    try:
                        if hasattr(env_worker, 'env') and env_worker.env:
                             if hasattr(env_worker.env, 'close'):
                                 env_worker.env.close()
                    except Exception as e_close:
                        logger.warning(f"Failed to close environment during retry: {e_close}")
                    continue

            except Exception as e:
                logger.error(f"[Explore] Critical Error with model {model_name}: {e}")
                traceback.print_exc()
                self._record_model_result(model_name, False)
                # 异常发生也尝试清理
                try:
                    if hasattr(env_worker, 'env') and env_worker.env and hasattr(env_worker.env, 'close'):
                        env_worker.env.close()
                except:
                    pass
                continue

        logger.warning(f"[Explore] All models failed.")
        self._report_progress_if_needed()

        if last_trajectory:
            logger.warning(f"[Explore] Returning result from last model ({candidate_models[-1]}).")
            return [last_trajectory]
        
        return []

    def summarize(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """统一的轨迹总结提炼入口，基于阶段路由到同域或跨域提取逻辑。"""
        if not trajectory or not trajectory.steps:
            return []

        phase = task.metadata.get("phase", "unknown")
        results = []
        if phase == "intra":
            results = self.summarize_intra(task, trajectory)
        elif phase == "extra":
            results = self.summarize_cross(task, trajectory)
        
        return results if results else []

    def get_next_sandbox_id(self) -> str:
        """从预设的沙盒池中循环安全地获取下一个可用的沙盒 ID，使用锁保护并发迭代。"""
        with self._lock:  # [修改] 保护迭代器
            try:
                return next(self.sandbox_id_iterator)
            except StopIteration:
                return "train_001"
    
    def _get_enhanced_context(self, app_name: str, anchor_api_names: List[str]) -> Tuple[str, str, str]:
        """将 JSON 格式的 API 知识库转化为大模型更容易阅读和理解的格式化纯文本。"""
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
            if "." in input_api_name:
                short_name = input_api_name.split('.')[-1]
                full_path = input_api_name
            else:
                short_name = input_api_name
                full_path = f"apis.{app_name}.{short_name}"

            api_data = None
            if short_name in target_app_apis:
                api_data = target_app_apis[short_name]
            
            if api_data:
                correct_call_name = api_data.get("call_name", full_path)
                params_str = json.dumps(api_data.get("parameters", []), indent=2)
                returns_str = json.dumps(api_data.get("returns", {}), indent=2)
                
                block = (
                    f'api_name: "{correct_call_name}"\n'
                    f'description: "{api_data.get("description", "")}"\n'
                    f'parameters: {params_str}\n'
                    f'returns: {returns_str}'
                )
                anchor_details_list.append(block)
        
        anchor_apis_detailed_info = "\n\n---\n\n".join(anchor_details_list) if anchor_details_list else "None provided."

        return all_apps_info, target_app_apis_info, anchor_apis_detailed_info
    
    def _fetch_real_table_data(self, sandbox_id: str, app_tables: Dict[str, List[str]]) -> str:
        """核心数据桥梁探针：通过 API 请求远端 AppWorld 服务，动态探测并提取底层真实状态。"""
        fetched_data = {}
        try:
            env_url = self.config.get("env_service", {}).get("env_url", "http://localhost:8080")
            target_endpoint = f"{env_url}/fetch_db_data"
            
            payload = {
                "sandbox_id": sandbox_id,
                "app_tables": app_tables
            }
            
            response = requests.post(target_endpoint, json=payload, timeout=15.0)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    fetched_data = result.get("data", {})
                else:
                    logger.warning(f"[Data Fetch] 远端服务返回错误: {result.get('error')}")
            else:
                logger.warning(f"[Data Fetch] API 请求失败，状态码: {response.status_code}")
                
        except Exception as e:
            logger.error(f"[Data Fetch] 无法连接到 AppWorld 远端服务打捞数据: {e}")
            
        return json.dumps(fetched_data, indent=2, ensure_ascii=False) if fetched_data else "{}"

    def generate_intra_task(self, app_name: str, task: Task = None) -> List[Task]:
        """同领域 (Intra-domain) 任务生成，经历评委筛选与生成器构建双阶段。"""
        generated_tasks = []
        app_data = self.api_knowledge.get(app_name, {})
        all_apis = list(app_data.get("apis", {}).keys())
        
        if len(all_apis) < 2: 
            return []

        real_sandbox_id = self.get_next_sandbox_id()

        candidate_groups = []
        for _ in range(3):
            api_count = random.choice([2, 3])
            candidate_groups.append(random.sample(all_apis, min(api_count, len(all_apis))))

        groups_str = "\n".join([f"Group {i+1}: {g}" for i, g in enumerate(candidate_groups)])
        all_apps_info, target_app_apis_info, _ = self._get_enhanced_context(app_name, [])

        db_schema_overview = get_intra_schema([app_name]) 

        selector_prompt = INTRA_DOMAIN_SELECTOR_PROMPT.format(
            CANDIDATE_COUNT=3,
            APP_NAME=app_name,
            ALL_APPS_DESC=all_apps_info,
            TARGET_APP_API_DESCS=target_app_apis_info,
            CANDIDATE_GROUPS_STR=groups_str,
            DB_SCHEMA_OVERVIEW=db_schema_overview 
        )
        
        sel_response = self._chat_with_retry(messages=[{"role": "user", "content": selector_prompt}])
        if not sel_response: return []
        
        selection_data = parse_intra_selector(sel_response.content)
        if not selection_data or "selected_apis" not in selection_data: return []

        selected_apis = selection_data["selected_apis"]
        required_tables = selection_data.get("required_tables", [])

        db_content = self._fetch_real_table_data(real_sandbox_id, {app_name: required_tables})
        _, _, anchor_apis_detailed_info = self._get_enhanced_context(app_name, selected_apis)

        generator_prompt = INTRA_DOMAIN_GENERATOR_PROMPT.format(
            APP_NAME=app_name,
            SELECTED_APIS=json.dumps(selected_apis),
            ALL_APIS_BRIEF=target_app_apis_info,
            ANCHOR_API_DETAILS=anchor_apis_detailed_info,
            DB_CONTENT=db_content
        )

        gen_response = self._chat_with_retry(messages=[{"role": "user", "content": generator_prompt}])
        if not gen_response: return []

        parsed_scenarios = parse_intra_generator(gen_response.content)
        
        for scenario in parsed_scenarios:
            new_task = copy.deepcopy(task) if task else Task()
            new_task.query = scenario.get("user_query", "")
            new_task.metadata = {
                "phase": "intra",
                "env_sandbox_id": real_sandbox_id, 
                "target_app": app_name,
                "target_api": scenario.get("target_api", ""),
                "selected_apis_context": selected_apis,
                "required_tables": required_tables,
                "origin_query": new_task.query
            }
            generated_tasks.append(new_task)

        return generated_tasks

    def generate_cross_task(self, target_apps: List[str], task: Task = None) -> List[Task]:
        """跨领域 (Cross-domain) 任务生成，自动计算不同 App 的 API 比例分配。"""
        ordered_apps = self._get_valid_app_chain(target_apps)
        if not ordered_apps:
            logger.debug(f"[Cross-Gen] 拦截并丢弃不合理的跨域组合/顺序: {target_apps}")
            return []
            
        target_apps = ordered_apps 
        generated_tasks = []
        
        total_apis = random.choices([2, 3, 4, 5], weights=[0.4, 0.3, 0.2, 0.1])[0]
        total_apis = max(total_apis, len(target_apps))
        num_candidates = 3 if total_apis in [2, 3] else 5

        real_sandbox_id = self.get_next_sandbox_id()

        candidate_groups = []
        apps_info_lines = []
        
        for app in target_apps:
            all_apps_info, app_apis_info, _ = self._get_enhanced_context(app, [])
            apps_info_lines.append(f"--- APP: {app} ---\n{app_apis_info}")

        for _ in range(num_candidates):
            current_group = {}
            remaining = total_apis - len(target_apps)
            allocations = {app: 1 for app in target_apps}
            
            # [修改] 移除了不必要的 if remaining < 0 废代码分支，直接分配剩余额度
            for _ in range(remaining):
                allocations[random.choice(target_apps)] += 1

            for app, count in allocations.items():
                if count == 0: continue
                all_apis = list(self.api_knowledge.get(app, {}).get("apis", {}).keys())
                current_group[app] = random.sample(all_apis, min(count, len(all_apis)))
            candidate_groups.append(current_group)

        groups_str = "\n".join([f"Group {i+1}: {json.dumps(g)}" for i, g in enumerate(candidate_groups)])
        apps_info_str = "\n\n".join(apps_info_lines)

        db_schema_overview = get_cross_schema(target_apps)

        selector_prompt = CROSS_DOMAIN_SELECTOR_PROMPT.format(
            CANDIDATE_COUNT=num_candidates,
            APP_COUNT=len(target_apps),
            APPS_INFO_STR=apps_info_str,
            CANDIDATE_GROUPS_STR=groups_str,
            DB_SCHEMA_OVERVIEW=db_schema_overview 
        )

        sel_response = self._chat_with_retry(messages=[{"role": "user", "content": selector_prompt}])
        if not sel_response: return []

        selection_data = parse_cross_selector(sel_response.content)
        if not selection_data or "selected_apis" not in selection_data: return []

        selected_apis = selection_data["selected_apis"]
        required_tables = selection_data.get("required_tables", {})

        db_content = self._fetch_real_table_data(real_sandbox_id, required_tables)
        
        anchor_details_combined = []
        for app, apis in selected_apis.items():
            _, _, anchor_detail = self._get_enhanced_context(app, apis)
            anchor_details_combined.append(f"[{app} Details]\n{anchor_detail}")
        
        all_apis_brief_combined = []
        for app in target_apps:
            _, app_apis_info, _ = self._get_enhanced_context(app, [])
            all_apis_brief_combined.append(f"[{app} APIs]\n{app_apis_info}")
        
        generator_prompt = CROSS_DOMAIN_GENERATOR_PROMPT.format(
            APPS_NAMES=", ".join(target_apps),
            SELECTED_APIS_JSON=json.dumps(selected_apis),
            ALL_APIS_BRIEF="\n".join(all_apis_brief_combined),
            ANCHOR_API_DETAILS="\n".join(anchor_details_combined),
            DB_CONTENT=db_content
        )

        gen_response = self._chat_with_retry(messages=[{"role": "user", "content": generator_prompt}])
        if not gen_response: return []

        parsed_scenarios = parse_cross_generator(gen_response.content)

        for scenario in parsed_scenarios:
            new_task = copy.deepcopy(task) if task else Task()
            new_task.query = scenario.get("user_query", "")
            new_task.metadata = {
                "phase": "extra",
                "env_sandbox_id": real_sandbox_id, 
                "target_apps": target_apps,
                "source_api": scenario.get("source_info_api", ""),
                "target_api": scenario.get("target_action_api", ""),
                "logic_pattern": scenario.get("logic_pattern", ""),
                "selected_apis_context": selected_apis,
                "required_tables": required_tables,
                "origin_query": new_task.query
            }
            generated_tasks.append(new_task)

        return generated_tasks

    def summarize_intra(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """单域探索反思与提取，通过脱敏轨迹生成规范的 RL 目标。"""
        client = self._summarize_client
        llm_fn = self._get_llm_chat_fn(client)
        
        masked_trajectory = copy.deepcopy(trajectory)
        # [修改] 使用自适应方法安全地处理对象和字典的属性读写，防止类型冲突报错
        if len(masked_trajectory.steps) > 2:
            for i in [1, 2]:
                step = masked_trajectory.steps[i]
                if isinstance(step, dict):
                    if step.get('role') == 'user':
                        step['content'] = '[MASKED]'
                else:
                    if getattr(step, 'role', None) == 'user':
                        if hasattr(step, 'content'):
                            setattr(step, 'content', '[MASKED]')
                        elif isinstance(getattr(step, '__dict__', None), dict):
                            step.__dict__['content'] = '[MASKED]'

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
            logger.error(f"[Summarize Intra] LLM call failed: {e}")
            return []
        
        task_copy = task.copy()
        task_copy.evaluator = 'synthetic'
        
        tasks = parse_tasks_from_response(task_copy, llm_output)

        reward_info = None
        if trajectory.reward:
            reward_info = {
                "outcome": trajectory.reward.outcome,
                "reason": getattr(trajectory.reward, "reason", "No reason provided")
            }

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
        """跨域探索反思与提取，提炼跨 App 任务轨迹并包装为 RL 目标。"""
        client = self._summarize_client
        llm_fn = self._get_llm_chat_fn(client)
        
        masked_trajectory = copy.deepcopy(trajectory)
        # [修改] 使用自适应方法脱敏
        if len(masked_trajectory.steps) > 2:
            for i in [1, 2]:
                step = masked_trajectory.steps[i]
                if isinstance(step, dict):
                    if step.get('role') == 'user':
                        step['content'] = '[MASKED]'
                else:
                    if getattr(step, 'role', None) == 'user':
                        if hasattr(step, 'content'):
                            setattr(step, 'content', '[MASKED]')
                        elif isinstance(getattr(step, '__dict__', None), dict):
                            step.__dict__['content'] = '[MASKED]'
        
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
            
        task_copy = task.copy()
        task_copy.evaluator = 'synthetic'
        
        tasks = parse_tasks_from_response(task_copy, llm_output)
        
        reward_info = None
        if trajectory.reward:
            reward_info = {
                "outcome": trajectory.reward.outcome,
                "reason": getattr(trajectory.reward, "reason", "No reason provided")
            }

        for task_obj in tasks:
            if task_obj.task.metadata is None:
                task_obj.task.metadata = {}
            
            task_obj.task.origin_query = task.query
            task_obj.task.metadata["summary_analysis_process"] = llm_output
            task_obj.task.metadata["execution_reward"] = reward_info
            task_obj.task.metadata["source_data_id"] = task.metadata.get("data_id")
            task_obj.task.metadata["generation_type"] = "evolved"
            
        return tasks

    def verify_direct_gt(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        """针对原始指令进行直接验证模式，判断Agent行为是否可以直接作为 Ground Truth 提炼。"""
        if trajectory.reward and trajectory.reward.outcome < 0.3:
            logger.info(f"[Verify] Skipping task {task.task_id} due to very low reward {trajectory.reward.outcome}")
            return None

        client = self._summarize_client
        llm_fn = self._get_llm_chat_fn(client)

        system_prompt, user_prompt = get_direct_verify_prompt(
            task, trajectory, self._env_profile
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            llm_response = llm_fn(messages=messages)
            llm_output = llm_response["content"]
        except Exception as e:
            logger.error(f"[Verify] LLM call failed: {e}")
            return None

        result = parse_direct_verification(llm_output)

        if result.get("is_valid"):
            new_task = task.copy()
            new_task.evaluator = 'synthetic' 
            
            new_task.ground_truth = result.get("refined_code", "")
            new_task.origin_query = task.query 
            
            if new_task.metadata is None: new_task.metadata = {}
            new_task.metadata["verification_reason"] = result.get("reason")
            new_task.metadata["verification_confidence"] = result.get("confidence")
            new_task.metadata["data_pair_type"] = "direct_verified"
            
            return TaskObjective(
                task=new_task,
                confidence=result.get("confidence", 0.9),
                reward=trajectory.reward.outcome if trajectory.reward else 0.0
            )
        else:
            logger.info(f"[Verify] Task {task.task_id} rejected: {result.get('reason')}")
            return None

    def _chat_with_retry(self, messages: List[Dict], **kwargs) -> Optional[Any]:
        """带退避指数休眠的重试请求函数，用于提升外部大模型接口稳定性。"""
        for i in range(self._max_llm_retries):
            try:
                response = self.llm_client.chat(messages=messages, **kwargs)
                
                if isinstance(response, str):
                    if response.strip():
                        return SimpleNamespace(content=response)
                elif response and hasattr(response, 'content') and response.content:
                    return response
                    
            except Exception as e:
                logger.warning(f"LLM call failed: {e}. Retry {i+1}...")
            
            if i < self._max_llm_retries - 1:
                time.sleep(2 ** i)
                
        return None

    def _load_sandbox_task_ids(self, path: str) -> List[str]:
        """从预处理的文件中导入系统中所有合格的沙盒环境标识 ID 列表。"""
        if not os.path.exists(path):
            return ["train_001"]
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [item["TaskID"] for item in data if "TaskID" in item]
        except:
            return ["train_001"]

    def _check_api_called(self, trajectory: Trajectory, api_name: str) -> bool:
        """检查特定 API 是否被 Agent 召唤过，自适应对象和字典属性。"""
        if not trajectory or not trajectory.steps: return False
        for step in trajectory.steps:
            # [修改] 兼容字典或类对象的取值逻辑
            role = step.get('role') if isinstance(step, dict) else getattr(step, 'role', None)
            has_error = step.get('error', False) if isinstance(step, dict) else getattr(step, 'error', False)
            tool_name = step.get('tool_name', '') if isinstance(step, dict) else getattr(step, 'tool_name', '')
            
            if role == "tool" and not has_error:
                if api_name and api_name in tool_name: return True
        return False

    def _check_app_usage(self, trajectory: Trajectory, app_name: str) -> bool:
        """检查整条轨迹日志内是否有涉及到指定的 App，自适应对象和字典属性。"""
        if not trajectory or not trajectory.steps: return False
        app_apis = self.api_knowledge.get(app_name, {}).get("apis", {}).keys()
        for step in trajectory.steps:
            role = step.get('role') if isinstance(step, dict) else getattr(step, 'role', None)
            tool_name = step.get('tool_name', '') if isinstance(step, dict) else getattr(step, 'tool_name', '')
            
            if role == "tool":
                if app_name and app_name.lower() in tool_name.lower(): return True
                for api in app_apis:
                    if api in tool_name: return True
        return False

    def _load_json(self, path: str) -> Dict:
        """极简的 JSON 文件安全读取器，遇错返回空字典。"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {}

    def _save_intra_memory(self, app_name: str):
        """将已被有效探索过的单体 App 持久化记录，使用锁保证多线程文件 IO 安全。"""
        with self._lock: # [修改] 添加线程锁
            os.makedirs(os.path.dirname(self.intra_memory_path), exist_ok=True)
            current_data = self._load_json(self.intra_memory_path)
            current_apps = set(current_data.get("explored_apps", []))
            current_apps.add(app_name)
            with open(self.intra_memory_path, 'w', encoding='utf-8') as f:
                json.dump({"explored_apps": list(current_apps)}, f, indent=2)

    def _save_cross_memory(self, metadata: Dict):
        """记录跨域探索成功的组合日志，使用锁保证多线程文件 IO 安全。"""
        with self._lock: # [修改] 添加线程锁
            os.makedirs(os.path.dirname(self.cross_memory_path), exist_ok=True)
            current_data = self._load_json(self.cross_memory_path)
            if "logs" not in current_data: current_data["logs"] = []
            
            log_entry = {
                "synthesized_user_query": metadata.get("synthesized_user_query"),
                "info_app": metadata.get("info_app"),
                "exec_app": metadata.get("exec_app"),
                "timestamp": time.time()
            }
            current_data["logs"].append(log_entry)
            
            with open(self.cross_memory_path, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, indent=2)

    def _get_llm_chat_fn(self, llm_client: LlmClient, sampling_params: Optional[dict] = None) -> Callable:
        """核心适配器闭包，封装通用的 LLM 客户端为 AgentFlow 接受的高阶函数形态。"""
        def llm_chat(
            messages: list[dict[str, str]],
            custom_sampling_params: Optional[dict] = None,
            request_id: Optional[str] = None,
        ) -> dict:
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            input_messages = copy.deepcopy(messages)
            res = None
            res_content = "I apologize, but I encountered an error generating a response." # [修改] 默认兜底字符串
            
            for i in range(self._max_llm_retries):
                try:
                    res = llm_client.chat(
                        messages=input_messages, sampling_params=updated_sampling_params
                    )
                    if res is not None and res != "":
                        # [修改] 解析可能的对象并提取纯文本，避免 AgentFlow 字典拼接报错
                        if hasattr(res, 'content'):
                            res_content = res.content
                        elif isinstance(res, dict) and 'content' in res:
                            res_content = res['content']
                        else:
                            res_content = str(res)
                        break
                except Exception as e:
                    logger.warning(f"llm_chat retry {i} error: {e}")
                    time.sleep(2**i)

            return {
                "role": "assistant",
                "content": res_content, # [修改] 返回安全的纯字符串
            }

        return llm_chat