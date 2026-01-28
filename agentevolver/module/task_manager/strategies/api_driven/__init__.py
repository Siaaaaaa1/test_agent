import json
import os
import random
import time
import itertools
import threading
import copy
from typing import List, Dict, Any, Optional, Set, Callable, Union
from types import SimpleNamespace

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
from agentevolver.module.task_manager.strategies.api_driven.prompts.intra_domain import parse_intra_purpose_from_response
from agentevolver.module.task_manager.strategies.api_driven.prompts.cross_domain import parse_cross_purpose_from_response

# 环境与执行模块
from agentevolver.module.env_manager.env_worker import EnvWorker, TrajExpConfig
from agentevolver.module.task_manager.agent_flow import ModifiedAgentFlow

# Prompt 与 Profile 相关
from agentevolver.module.task_manager.prelude_profiles import appworld, bfcl, webshop
from agentevolver.module.task_manager.strategies.api_driven.prompts import (
    INTRA_DOMAIN_PURPOSE_PROMPT,
    CROSS_DOMAIN_PURPOSE_PROMPT,
    get_agent_interaction_system_prompt
)
from agentevolver.module.task_manager.strategies.api_driven.prompts.prompt_summarize import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)
from agentevolver.client.llm_client_mix import Mix_DashScopeClient
UNIVERSAL_INFO_PROVIDERS = {"notes", "gmail", "simple_messages", "calendar", "contacts"}

class ApiDrivenExploreStrategy(TaskExploreStrategy):
    """
    API 驱动的探索策略类
    包含：任务生成(Intra/Cross) -> 任务执行(Explore) -> 任务总结(Summarize)
    """

    def __init__(self, tokenizer, config: DictConfig, llm_client: Optional[LlmClient] = None, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        
        self._env_profile = kwargs.get("env_profile")

        # ================= [Init 修改] =================
        # 1. 常规客户端 (self.llm_client): 用于 generate 和 summarize
        # 保持兼容性：如果外部传入了 client，直接使用
        if llm_client:
            self.llm_client = llm_client
        else:
            logger.warning("[ApiDriven] No LlmClient passed, creating a standard LlmClient.")
            self.llm_client = LlmClient(config)
            
        self._summarize_client = kwargs.get("llm_client_summarize", self.llm_client)

        # 2. 探索专用客户端 (self.explore_client): 仅用于 explore 阶段
        # 强制尝试初始化 Mix_DashScopeClient 以支持动态路由和隔离限流
        logger.info("[ApiDriven] Initializing Mix_DashScopeClient specifically for 'explore' phase.")
        try:
            # 假设 Mix_DashScopeClient 已正确 import
            self.explore_client = Mix_DashScopeClient(
                model_name="qwen235", # 默认占位，实际会被 sampling_params 覆盖
                temperature=config.get("exploration_llm_temperature", 0.7)
            )
        except Exception as e:
            logger.error(f"[ApiDriven] Failed to init Mix_DashScopeClient: {e}. Fallback to standard client.")
            self.explore_client = self.llm_client
        # ================= [Init 结束] =================

        self._max_llm_retries = kwargs.get("max_llm_retries", 3)
        self._lock = threading.Lock() 
        
        # --- 路径与文件配置 ---
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

    # ================= [核心] 执行循环逻辑 =================

    def explore(self, task: Task, data_id: str, rollout_id: str) -> List[Trajectory]:
        """
        [ApiDriven] 执行探索任务的核心入口。
        
        修改点：
        1. 使用 self.explore_client (Mix_DashScopeClient)。
        2. 轮询策略：qwen235 -> dsv3 -> azure-gpt-5-mini -> azure-gpt-5。
        3. 成功判定：使用 trajectory.success (即 reward > 0) 作为成功标准。
           - 如果 Success: 立即返回。
           - 如果 Fail (Logic Error 或 Timeout): 尝试下一个模型。
        """
        # 1. 动态获取沙箱 ID
        real_sandbox_id = self.get_next_sandbox_id()
        if real_sandbox_id:
            if task.metadata is None:
                task.metadata = {}
            task.metadata["env_sandbox_id"] = real_sandbox_id
            
        debug_log(self.config, "api_explore_start", {
            "task_id": task.task_id,
            "data_id": data_id,
            "phase": task.metadata.get('phase'),
            "real_sandbox_id": real_sandbox_id
        })

        # 定义模型尝试队列 (根据 Mix_DashScopeClient 的路由逻辑)
        candidate_models = ["HY-Qwen3-235B-A22B-Instruct-2507", "DeepSeek-V3-Online", "azure-gpt-5-mini", "azure-gpt-5"]
        
        last_trajectory = None
        max_steps = self.config.get("max_explore_step", 25) 

        # 开始模型轮询
        for model_name in candidate_models:
            logger.info(f"[Explore] Task {data_id}: Trying model '{model_name}'...")

            # 2. 初始化环境工作者 (每次重置以防污染)
            thread_idx = 0
            if task.metadata and 'thread_index' in task.metadata:
                thread_idx = task.metadata['thread_index']
            
            env_worker = EnvWorker(
                task=task,
                config=self.config, 
                thread_index=thread_idx,
                tokenizer=self.tokenizer
            )

            # TODO: 将对话的参数需要加入LLM 
            # # 3. 构造 LLM 聊天函数 (注入当前模型参数)
            if model_name in ["HY-Qwen3-235B-A22B-Instruct-2507", "DeepSeek-V3-Online"]:
                sampling_params = {
                    "temperature": self.config.get("exploration_llm_temperature", 0.5),
                    "top_p": self.config.get("exploration_llm_top_p", 0.9),
                    "top_k": self.config.get("exploration_llm_top_k", 50),
                    "model": model_name, # MixClient 根据此字段路由
                }
                # 使用 explore_client
                llm_chat_fn = self._get_llm_chat_fn(
                    self.explore_client,
                    sampling_params=sampling_params
                )
            else:
                # 使用 explore_client
                llm_chat_fn = self._get_llm_chat_fn(
                    self.explore_client
                )

            # 4. 初始化 Agent 工作流
            agent_flow = ModifiedAgentFlow(
                llm_chat_fn=llm_chat_fn,
                tokenizer=self.tokenizer,
                config=self.config,
                enable_context_generator=False
            )
            
            agent_flow.max_steps = max_steps
            agent_flow.max_model_len = self.config.get("max_model_len", 102400)

            # 5. 执行 Agent
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
                
                # [核心判定修改] 使用 Trajectory.success 属性
                # 如果成功，立即返回；如果失败（无论是否超时），继续尝试下一个模型
                if trajectory and trajectory.success:
                    logger.info(f"[Explore] Model {model_name} SUCCEEDED (Steps: {len(trajectory.steps)}). Returning result.")
                    return [trajectory]
                else:
                    fail_reason = "Max Steps Reached" if len(trajectory.steps) >= max_steps else "Task Logic Failed"
                    logger.warning(f"[Explore] Model {model_name} FAILED ({fail_reason}). Success check returned False. Retrying with next model...")
                    
                    # 资源清理
                    try:
                        if hasattr(env_worker, 'env') and env_worker.env:
                             pass 
                    except: pass
                    continue

            except Exception as e:
                logger.error(f"[Explore] Critical Error with model {model_name}: {e}")
                traceback.print_exc()
                # 异常情况也继续尝试下一个模型
                continue

        # 如果所有模型都失败，返回最后一个模型的结果（即使是 Failed 状态）
        if last_trajectory:
            logger.warning(f"[Explore] All models failed. Returning result from last model ({candidate_models[-1]}).")
            return [last_trajectory]
        
        # 极端情况：没有任何轨迹生成
        return []
    # ================= 总结逻辑 =================

    def summarize(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        统一的总结入口，根据任务阶段路由到具体逻辑。
        """
        if not trajectory or not trajectory.steps:
            return []

        phase = task.metadata.get("phase", "unknown")
        
        results = []
        if phase == "intra":
            results = self.summarize_intra(task, trajectory)
        elif phase == "extra":
            results = self.summarize_cross(task, trajectory)
        
        # 如果 summarize 子方法返回 None 或空列表，则返回空
        return results if results else []

    def get_next_sandbox_id(self) -> str:
        try:
            return next(self.sandbox_id_iterator)
        except StopIteration:
            return "train_001"

    # ================= 任务生成 (Generation) =================
    
    def generate_intra_task(self, api_data: Union[dict, List[dict]], task: Task = None) -> List[Task]:
        """
        生成单域探索任务：针对特定 App 的 API 生成 Prompt。
        支持传入单个 api_dict 或 api_dict 列表。
        返回生成的 Task 列表。
        """
        generated_tasks = []

        # [Fix] 统一输入为列表处理
        api_dict_list = api_data if isinstance(api_data, list) else [api_data]

        for api_dict in api_dict_list:
            # 1. 空值校验
            if not api_dict:
                logger.warning("[Intra-Gen] Encountered empty api_dict, skipping.")
                continue

            target_app = api_dict.get("app_name", "UnknownApp")
            
            # 2. 安全处理 API 列表转字符串
            raw_api_list = api_dict.get("apis_name_list", [])
            if isinstance(raw_api_list, list):
                api_list_str = ",".join([str(x) for x in raw_api_list])
            else:
                api_list_str = str(raw_api_list)

            logger.debug(f"[Intra-Gen] Preparing prompt for App: {target_app}")

            # 3. Prompt 格式化 (带容错)
            try:
                prompt = INTRA_DOMAIN_PURPOSE_PROMPT.format(
                    APP_NAME=target_app,
                    API_LIST=api_list_str
                )
            except KeyError as e:
                logger.warning(f"[Intra-Gen] .format() failed ({e}). Switching to .replace().")
                prompt = INTRA_DOMAIN_PURPOSE_PROMPT.replace("{APP_NAME}", target_app).replace("{API_LIST}", api_list_str)
            except Exception as e:
                logger.error(f"[Intra-Gen] Prompt formatting critical error: {e}")
                continue

            # 4. 调用 LLM
            try:
                response = self._chat_with_retry(messages=[{"role": "user", "content": prompt}])
            except Exception as e:
                logger.error(f"[Intra-Gen] Chat API call failed for {target_app}: {e}")
                continue

            if not response: 
                logger.warning(f"[Intra-Gen] No response from LLM for {target_app}")
                continue

            # 5. 解析结果 (现在返回的是 List[dict])
            parsed_scenarios = parse_intra_purpose_from_response(response.content)

            if not parsed_scenarios:
                logger.warning(f"[Intra-Gen] Failed to parse JSON for app {target_app}")
                continue
            
            # 6. 为每个场景生成独立的 Task 对象
            for scenario in parsed_scenarios:
                # 使用 deepcopy 避免修改原始模板，如果没有模板则新建
                new_task = copy.deepcopy(task) if task else Task()
                
                new_task.query = scenario["user_query"]
                new_task.metadata = {
                    "phase": "intra", 
                    "target_app": target_app,
                    "app1_apis": list(api_dict.get("apis_name_list", [])),
                    "origin_query": scenario["user_query"],
                    "target_api": scenario["target_api"],
                    "prompt": prompt,
                }
                generated_tasks.append(new_task)
                
            logger.info(f"[Intra-Gen] Generated {len(parsed_scenarios)} tasks for App: {target_app}")

        return generated_tasks


    def generate_cross_task(self, api_dict1: dict, api_dict2: dict, task: Task = None) -> List[Task]:
        """
        生成跨域探索任务：选择两个 App，合成跨应用场景。
        返回 Task 列表 (因为 Prompt 现在一次生成多个场景)。
        """
        generated_tasks = []

        # 基础校验
        if not api_dict1 or not api_dict2:
            logger.error("[Cross-Gen] One of the api_dicts is None.")
            return []

        app1_name = api_dict1.get("app_name", "App1")
        app2_name = api_dict2.get("app_name", "App2")
        
        # 安全转换 API List
        apis1_str = ",".join(api_dict1.get("apis_name_list", []))
        apis2_str = ",".join(api_dict2.get("apis_name_list", []))

        # 1. Prompt 格式化 (带容错)
        try:
            prompt = CROSS_DOMAIN_PURPOSE_PROMPT.format(
                APP_NAME1=app1_name,
                API_LIST1=apis1_str,
                APP_NAME2=app2_name,
                API_LIST2=apis2_str
            )
        except KeyError as e:
            logger.warning(f"[Cross-Gen] .format() failed ({e}). Switching to .replace().")
            prompt = CROSS_DOMAIN_PURPOSE_PROMPT \
                .replace("{APP_NAME1}", app1_name) \
                .replace("{API_LIST1}", apis1_str) \
                .replace("{APP_NAME2}", app2_name) \
                .replace("{API_LIST2}", apis2_str)

        # 2. 调用 LLM
        try:
            response = self._chat_with_retry(messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            logger.error(f"[Cross-Gen] Chat API call failed: {e}")
            return []

        if not response: 
            return []
        
        # 3. 解析结果 (返回 List[dict])
        try:
            parsed_scenarios = parse_cross_purpose_from_response(response.content)
        except Exception as e:
            logger.error(f"[Cross-Gen] JSON parse logic error: {e}")
            return []

        if not parsed_scenarios:
            logger.warning(f"[Cross-Gen] No valid scenarios parsed for {app1_name} <-> {app2_name}")
            return []

        # 4. 构建 Task 列表
        for scenario in parsed_scenarios:
            new_task = copy.deepcopy(task) if task else Task()
            
            new_task.query = scenario["user_query"]
            new_task.metadata = {
                "phase": "extra", # 注意：你之前的代码是 extra，如果是跨域通常叫 cross 或 inter
                "app1": app1_name,
                "app2": app2_name,
                "app1_apis": list(api_dict1.get("apis_name_list", [])),
                "app2_apis": list(api_dict2.get("apis_name_list", [])),
                "origin_query": scenario["user_query"],
                "source_api" : scenario["source_info_api"],
                "target_api": scenario["target_action_api"],
                "logic_pattern": scenario.get("logic_pattern", "Unknown"),
                "prompt": prompt,
            }
            generated_tasks.append(new_task)

        logger.info(f"[Cross-Gen] Generated {len(generated_tasks)} tasks for {app1_name} <-> {app2_name}")
        return generated_tasks

    # ================= 阶段总结逻辑 (Summarize) =================

    def summarize_intra(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        单域探索总结：检查是否调用目标 API，如果调用则使用 LLM 归纳任务意图。
        """ 
        # 2. 构造 LLM 函数 (使用修正后的变量名)
        client = self._summarize_client
        llm_fn = self._get_llm_chat_fn(client)
        
        # 3. 数据脱敏 (Masking)
        masked_trajectory = copy.deepcopy(trajectory)
        if len(masked_trajectory.steps) > 2:
            # Mask user instructions to prevent leaking into summary prompt context incorrectly
            if masked_trajectory.steps[1].get('role') == 'user':
                masked_trajectory.steps[1]['content'] = '[MASKED]'
            if masked_trajectory.steps[2].get('role') == 'user':
                masked_trajectory.steps[2]['content'] = '[MASKED]'

        # 4. 生成 Prompt
        # env_profile_obj = self._get_env_profile_obj()
        system_prompt, user_prompt = get_task_summarize_prompt(
            [masked_trajectory], old_objectives=task.query, profile=self._env_profile
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 5. 调用 LLM
        try:
            llm_response = llm_fn(messages=messages)
            llm_output = llm_response["content"]
        except Exception as e:
            logger.error(f"[Summarize Intra] LLM call failed: {e}")
            return []
        
        # 6. 解析结果
        task_copy = task.copy()
        task_copy.evaluator = 'synthetic'
        tasks = parse_tasks_from_response(task_copy, llm_output)
        
        return tasks

    def summarize_cross(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        跨域探索总结：验证是否跨两个 App 进行了交互，如果是，则归纳任务。
        """
        # 2. 构造 LLM 函数 (使用修正后的变量名)
        client = self._summarize_client
        llm_fn = self._get_llm_chat_fn(client)
        
        # 3. 数据脱敏
        masked_trajectory = copy.deepcopy(trajectory)
        if len(masked_trajectory.steps) > 2:
            if masked_trajectory.steps[1].get('role') == 'user':
                masked_trajectory.steps[1]['content'] = '[MASKED]'
            if masked_trajectory.steps[2].get('role') == 'user':
                masked_trajectory.steps[2]['content'] = '[MASKED]'
        
        # 4. 生成 Prompt
        # env_profile_obj = self._get_env_profile_obj()
        system_prompt, user_prompt = get_task_summarize_prompt(
            [masked_trajectory], old_objectives=task.query, profile=self._env_profile
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 5. 调用 LLM
        try:
            llm_response = llm_fn(messages=messages)
            llm_output = llm_response["content"]
        except Exception as e:
            logger.error(f"[Summarize Cross] LLM call failed: {e}")
            return []
            
        # 6. 解析结果
        task_copy = task.copy()
        task_copy.evaluator = 'synthetic'
        tasks = parse_tasks_from_response(task_copy, llm_output)
        
        return tasks

    # ================= 辅助私有方法 =================

    # def _get_env_profile_obj(self):
    #     """辅助方法：根据配置字符串获取 Profile 对象，用于总结 Prompt 的生成"""
    #     env_type = self.env_profile_name
    #     if env_type == "appworld":
    #         return appworld.AppWorldProfile()
    #     elif env_type == "bfcl":
    #         return bfcl.BfclProfile()
    #     elif env_type == "webshop":
    #         return webshop.WebShopProfile()
    #     return None

    def _chat_with_retry(self, messages: List[Dict], **kwargs) -> Optional[Any]:
        """
        调用 LLM，处理重试并标准化返回格式
        """
        for i in range(self._max_llm_retries):
            try:
                response = self.llm_client.chat(messages=messages, **kwargs)
                
                # 兼容性处理：DashScopeClient 可能返回字符串，OpenAIClient 返回对象
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
        if not os.path.exists(path):
            return ["train_001"]
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [item["TaskID"] for item in data if "TaskID" in item]
        except:
            return ["train_001"]

    def _check_api_called(self, trajectory: Trajectory, api_name: str) -> bool:
        if not trajectory or not trajectory.steps: return False
        for step in trajectory.steps:
            if step.get('role') == "tool" and not step.error:
                # 模糊匹配，因为 tool_name 可能包含路径或类名
                if api_name and api_name in step.tool_name: return True
        return False

    def _check_app_usage(self, trajectory: Trajectory, app_name: str) -> bool:
        if not trajectory or not trajectory.steps: return False
        app_apis = self.api_knowledge.get(app_name, {}).get("apis", {}).keys()
        for step in trajectory.steps:
            if step.get('role') == "tool":
                if app_name and app_name.lower() in step.tool_name.lower(): return True
                for api in app_apis:
                    if api in step.tool_name: return True
        return False

    def _load_json(self, path: str) -> Dict:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {}

    def _save_intra_memory(self, app_name: str):
        # 注意：外部已加锁
        os.makedirs(os.path.dirname(self.intra_memory_path), exist_ok=True)
        current_data = self._load_json(self.intra_memory_path)
        current_apps = set(current_data.get("explored_apps", []))
        current_apps.add(app_name)
        with open(self.intra_memory_path, 'w', encoding='utf-8') as f:
            json.dump({"explored_apps": list(current_apps)}, f, indent=2)

    def _save_cross_memory(self, metadata: Dict):
        # 注意：外部已加锁
        os.makedirs(os.path.dirname(self.cross_memory_path), exist_ok=True)
        current_data = self._load_json(self.cross_memory_path)
        if "logs" not in current_data: current_data["logs"] = []
        
        # 简化保存的信息，避免文件过大
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
        """
        辅助函数：封装 LLM 客户端调用，生成符合 AgentFlow 要求的 callable
        """
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
            
            for i in range(self._max_llm_retries):
                try:
                    res = llm_client.chat(
                        messages=input_messages, sampling_params=updated_sampling_params
                    )
                    if res is not None and res != "":
                        break
                except Exception as e:
                    logger.warning(f"llm_chat retry {i} error: {e}")
                    time.sleep(2**i)

            if res is None or res == "":
                # Fallback empty response to prevent crash
                res = "I apologize, but I encountered an error generating a response."

            return {
                "role": "assistant",
                "content": res,
            }

        return llm_chat