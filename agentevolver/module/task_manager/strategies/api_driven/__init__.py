import json
import os
import random
import time
import itertools
import uuid
import threading
import copy
from typing import List, Dict, Any, Optional, Set, Callable

from loguru import logger
from agentevolver.module.env_manager.env_worker import EnvWorker, TrajExpConfig
# [FIX]: 使用 ModifiedAgentFlow 代替标准 AgentFlow
from agentevolver.module.task_manager.agent_flow import ModifiedAgentFlow
from agentevolver.module.task_manager.strategies.api_driven.prompts import get_agent_interaction_system_prompt
# 导入基础策略类和数据模型
from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from agentevolver.module.task_manager.base import LlmClient
from agentevolver.utils.utils import extract_json_from_str
from agentevolver.module.task_manager.strategies.api_driven.prompts.prompt_explore import (
    get_agent_interaction_system_prompt
)
# [新增]: 导入通用的总结 Prompt 和解析工具，与 Random 策略保持一致
from agentevolver.module.task_manager.strategies.api_driven.prompts.prompt_summarize import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)
# [新增]: 导入环境 Profile 以支持 get_task_summarize_prompt
from agentevolver.module.task_manager.prelude_profiles import appworld, bfcl, webshop

# 引入预定义的 Prompt 模板
from agentevolver.module.task_manager.strategies.api_driven.prompts import (
    PLAN_GENERATION_PROMPT,
    BACK_TRANSLATION_PROMPT,
    PURPOSE_SYNTHESIS_PROMPT
)
from agentevolver.utils.debug_utils import debug_log

UNIVERSAL_INFO_PROVIDERS = {"notes", "gmail", "simple_messages", "calendar", "contacts"}

class ApiDrivenExploreStrategy(TaskExploreStrategy):
    """
    API 驱动的探索策略类
    重构版：包含完整的执行循环（explore）和总结逻辑（summarize），
    修复了并发写入冲突、拼写错误及执行时 query 缺失的问题。
    """

    def __init__(self, tokenizer, config, llm_client=None, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        
        # [修改点 5]: 复用传入的客户端，仅当未传入时才新建 (兼容性处理)
        if llm_client:
            self.llm_client = llm_client
        else:
            logger.warning("[ApiDriven] No LlmClient passed, creating a new one (Redundant).")
            self.llm_client = LlmClient(config)
            
        self._max_llm_retries = kwargs.get("max_llm_retries", 3)
        self._lock = threading.Lock() # 用于保护记忆文件的并发写入
        
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
        
        # 加载数据
        self.api_knowledge = self._load_json(self.api_knowledge_path)
        if not self.api_knowledge:
            logger.warning(f"API Knowledge not found at {self.api_knowledge_path}. Exploration might fail.")

        self.sandbox_ids_pool = self._load_sandbox_task_ids(self.task_labels_path)
        self.sandbox_id_iterator = itertools.cycle(self.sandbox_ids_pool)
        logger.info(f"[ApiDriven] Loaded {len(self.sandbox_ids_pool)} Sandbox Task IDs.")

        self.intra_memory_data = self._load_json(self.intra_memory_path)
        self.explored_intra_apps = set(self.intra_memory_data.get("explored_apps", []))
        self.cross_memory_data = self._load_json(self.cross_memory_path)
        
        # 获取环境配置名称，用于 System Prompt
        self.env_profile_name = self.config.get("env_service", {}).get("env_type", "appworld")

        debug_log(self.config, "api_strategy_init", {
            "env_profile": self.env_profile_name,
            "api_knowledge_size": len(self.api_knowledge),
            "sandbox_pool_size": len(self.sandbox_ids_pool),
            "mastered_intra_apps": len(self.explored_intra_apps),
            "active_apps": list(self.active_apps)
        })
        logger.info(f"[ApiDriven] Initialized. Mastered Apps (Intra): {list(self.explored_intra_apps)}")

    # ================= [核心] 执行循环逻辑 =================

    def explore(self, task: Task, data_id: str, rollout_id: str) -> List[Trajectory]:
        """
        [ApiDriven] 执行探索任务的核心入口。
        模仿 Random 策略，使用 EnvWorker.execute 接管执行流。
        """
        # 1. 动态获取沙箱 ID (保留 API 策略特有的逻辑)
        real_sandbox_id = self.get_next_sandbox_id()
        if real_sandbox_id:
            if task.metadata is None:
                task.metadata = {}
            # 将物理环境ID存入元数据，保留 task.task_id 为生成的唯一ID (如 gen_intra_0)
            task.metadata["env_sandbox_id"] = real_sandbox_id
            
        logger.info(f"[ApiDriven] Exploring task (Phase: {task.metadata.get('phase')}) on Sandbox: {real_sandbox_id}")
        debug_log(self.config, "api_explore_start", {
            "task_id": task.task_id,
            "data_id": data_id,
            "phase": task.metadata.get('phase'),
            "real_sandbox_id": real_sandbox_id,
            "instruction": task.query + "..." if task.query else None
        })

        # 2. 初始化环境工作者 (EnvWorker)
        thread_idx = 0
        if task.metadata and 'thread_index' in task.metadata:
            thread_idx = task.metadata['thread_index']
        
        env_worker = EnvWorker(
            task=task,
            config=self.config, 
            thread_index=thread_idx,
            tokenizer=self.tokenizer
        )

        # 3. 构造 LLM 聊天函数
        sampling_params = {
            "temperature": self.config.get("exploration_llm_temperature", 0.5),
            "top_p": self.config.get("exploration_llm_top_p", 0.9),
            "top_k": self.config.get("exploration_llm_top_k", 0),
        }
        
        llm_chat_fn = self._get_llm_chat_fn(
            self.llm_client, 
            sampling_params=sampling_params
        )

        # 4. 初始化 Agent 工作流 (ModifiedAgentFlow)
        agent_flow = ModifiedAgentFlow(
            llm_chat_fn=llm_chat_fn,
            tokenizer=self.tokenizer,
            config=self.config,
            enable_context_generator=False
        )
        
        # 动态设置最大步数和模型长度
        agent_flow.max_steps = self.config.get("max_explore_step", 10) 
        agent_flow.max_model_len = self.config.get("max_model_len", 102400)

        # 5. 执行 Agent，获取轨迹
        try:
            # 获取对应的 System Prompt
            system_prompt = get_agent_interaction_system_prompt()

            trajectory = env_worker.execute(
                data_id=data_id, # [FIX]: 接收并使用 TaskManager 传递过来的唯一 ID
                rollout_id=rollout_id,
                traj_exp_config=TrajExpConfig(add_exp=False), # API 探索通常不直接添加到经验池
                agent_flow=agent_flow,
                tmux={
                    'step': [0],  # 共享计数器
                    'token': [0],
                },
                stop=[False], # 停止信号
                system_prompt=system_prompt,
            )
            
            steps_count = len(trajectory.steps) if trajectory else 0
            debug_log(self.config, "api_explore_done", {
                "data_id": data_id,
                "sandbox_id": real_sandbox_id,
                "steps_count": steps_count,
                "status": "success"
            })
            return [trajectory]

        except Exception as e:
            logger.error(f"[ApiDriven] Explore failed on Sandbox {real_sandbox_id}: {e}")
            debug_log(self.config, "api_explore_error", {
                "data_id": data_id,
                "sandbox_id": real_sandbox_id,
                "error": str(e)
            })
            return [Trajectory(steps=[])]

    # ================= 总结逻辑 =================

    def summarize(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        统一的总结入口，根据任务阶段路由到具体逻辑。
        返回 List[TaskObjective] 以支持批量生成。
        """
        if not trajectory or not trajectory.steps:
            debug_log(self.config, "api_summarize_skipped", {"reason": "empty_trajectory"})
            return []

        phase = task.metadata.get("phase", "unknown")
        debug_log(self.config, "api_summarize_start", {"phase": phase, "task_id": task.task_id})
        
        results = []
        if phase == "intra":
            results = self.summarize_intra(task, trajectory)
        elif phase == "extra":
            results = self.summarize_cross(task, trajectory)
        
        debug_log(self.config, "api_summarize_result", {
            "phase": phase, 
            "generated_objectives": len(results)
        })
        
        # 如果 summarize 子方法返回 None 或空列表，则返回空
        return results if results else []

    def get_next_sandbox_id(self) -> str:
        try:
            return next(self.sandbox_id_iterator)
        except StopIteration:
            return "train_001"

    # ================= 任务生成 (Generation) =================

    def generate_intra_task(self, app_name: str = None, target_api_name: str = None, task: Task = None) -> Optional[Task]:
        """
        生成单域探索任务。
        """
        if not app_name:
            unmastered_apps = list(self.active_apps - self.explored_intra_apps)
            app_name = random.choice(unmastered_apps) if unmastered_apps else random.choice(list(self.active_apps))
        
        app_knowledge = self.api_knowledge.get(app_name, {})
        apis = app_knowledge.get("apis", {})

        if not target_api_name:
            action_apis = [k for k, v in apis.items() if v.get("action_type") == "Executive Action"]
            info_apis_list = [k for k, v in apis.items() if v.get("action_type") == "Informational Action"]
            
            roll = random.random()
            if roll < 0.7 and action_apis:
                target_api_name = random.choice(action_apis)
            elif info_apis_list:
                target_api_name = random.choice(info_apis_list)
            else:
                target_api_name = random.choice(list(apis.keys())) if apis else None

        if not target_api_name:
            debug_log(self.config, "api_gen_intra_skip", {"reason": "no_target_api", "app_name": app_name})
            return None

        target_api_def = apis.get(target_api_name)
        is_executive = target_api_def.get("action_type") == "Executive Action"
        ref_type = "Informational Action" if is_executive else "Executive Action"
        reference_apis = {k: v for k, v in apis.items() if k != target_api_name and v.get("action_type") == ref_type}

        prompt = PLAN_GENERATION_PROMPT.format(
            target_api_name=target_api_name,
            app_name=app_name,
            target_api_details=json.dumps(target_api_def, indent=2, ensure_ascii=False),
            available_info_apis=json.dumps(reference_apis, indent=2, ensure_ascii=False)
        )

        debug_log(self.config, "api_gen_intra", {
            "type": "LLM_INPUT",
            "app_name": app_name,
            "target_api": target_api_name,
            "prompt": prompt
        })
        
        response = self._chat_with_retry(messages=[{"role": "user", "content": prompt}])

        debug_log(self.config, "api_gen_intra", {
            "type": "LLM_OUTPUT",
            "response_content": response.content if response else None,
            "success": bool(response)
        })

        if not response: return None

        # [Fix Typo & Logic] 修正属性名并赋值给 query，确保执行时生效
        task.query = response.content.strip()
        task.query = task.query 
        task.metadata = {
                "phase": "intra", 
                "target_app": app_name, 
                "target_api": target_api_name
            }
        return task

    def generate_cross_task(self, app_list: List[str] = None, task: Task = None) -> Optional[Task]:
        """
        生成跨域探索任务。
        """
        if not app_list:
            app_list = list(self.active_apps)
        
        available_apps = [app for app in app_list if app in self.api_knowledge]
        if len(available_apps) < 2:
            debug_log(self.config, "api_gen_cross_skip", {"reason": "insufficient_apps", "count": len(available_apps)})
            return None
        
        # 去重逻辑
        explored_pairs = set()
        if "logs" in self.cross_memory_data:
            for log in self.cross_memory_data["logs"]:
                pair = f"{log.get('info_app')}->{log.get('exec_app')}"
                explored_pairs.add(pair)

        info_app_name, exec_app_name = None, None
        
        for _ in range(5):
            selected_apps = random.sample(available_apps, 2)
            app_a, app_b = selected_apps[0], selected_apps[1]
            
            if app_a not in UNIVERSAL_INFO_PROVIDERS and app_b in UNIVERSAL_INFO_PROVIDERS:
                temp_info, temp_exec = app_b, app_a
            else:
                temp_info, temp_exec = app_a, app_b
            
            if f"{temp_info}->{temp_exec}" not in explored_pairs:
                info_app_name, exec_app_name = temp_info, temp_exec
                break
        
        if not info_app_name: 
            info_app_name, exec_app_name = temp_info, temp_exec

        def get_sampled_apis(app_name, intent_type, count=5):
            app_info = self.api_knowledge.get(app_name, {})
            all_apis = app_info.get("apis", {})
            candidates = []
            if intent_type == "info":
                candidates = [k for k, v in all_apis.items() if v.get("action_type") == "Informational Action"]
                if not candidates:
                    candidates = [k for k, v in all_apis.items() if v.get("method", "").upper() == "GET"]
            else:
                candidates = [k for k, v in all_apis.items() if v.get("action_type") == "Executive Action"]
                if not candidates:
                    candidates = [k for k, v in all_apis.items() if v.get("method", "").upper() != "GET"]
            
            if not candidates: candidates = list(all_apis.keys())
            sampled_keys = random.sample(candidates, min(len(candidates), count))
            return {k: all_apis[k] for k in sampled_keys}

        info_apis = get_sampled_apis(info_app_name, "info", 5)
        exec_apis = get_sampled_apis(exec_app_name, "exec", 5)

        system_tools_hint = (
            "Available System Tools:\n"
            "- supervisor: Use to coordinate steps.\n"
            f"- Context: You are exploring functionalities between {info_app_name} and {exec_app_name}. "
            "Assume the user has necessary data in the source app."
        )

        prompt = PURPOSE_SYNTHESIS_PROMPT.format(
            info_app_name=info_app_name,
            exec_app_name=exec_app_name,
            info_apis_json=json.dumps(info_apis, indent=2, ensure_ascii=False),
            exec_apis_json=json.dumps(exec_apis, indent=2, ensure_ascii=False),
            system_tools_hint=system_tools_hint
        )

        debug_log(self.config, "api_gen_cross", {
            "type": "LLM_INPUT",
            "info_app": info_app_name,
            "exec_app": exec_app_name,
            "prompt": prompt
        })

        response = self._chat_with_retry(
            messages=[{"role": "user", "content": prompt}], 
            response_format={"type": "json_object"}
        )

        debug_log(self.config, "api_gen_cross", {
            "type": "LLM_OUTPUT",
            "response_content": response.content if response else None,
            "success": bool(response)
        })

        if not response: return None
        
        try:
            res_json = extract_json_from_str(response.content)
            user_query = res_json.get("user_query", "")
            target_action = res_json.get("target_action_api", "")
        except Exception as e:
            debug_log(self.config, "api_gen_cross_error", {"error": f"json_extract_failed: {e}", "content": response.content})
            return None
        
        # [Fix Logic] 赋值给 query
        task.query = user_query
        task.query = user_query
        task.metadata = {
                "phase": "extra",
                "info_app": info_app_name,
                "exec_app": exec_app_name,
                "target_api": target_action,
                "sampled_info_apis": list(info_apis.keys()),
                "sampled_exec_apis": list(exec_apis.keys())
            }
        return task

    # ================= 阶段总结逻辑 (Summarize) =================

    def summarize_intra(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        单域探索总结：检查是否调用目标 API，如果调用则使用 LLM 归纳任务意图。
        [FIX] 使用 parse_tasks_from_response 解析出列表，逻辑模仿 Random 策略。
        """
        target_app = task.metadata.get("target_app")
        target_api = task.metadata.get("target_api")
        
        # 1. 前置检查：如果目标 API 未被调用，视为探索失败，不生成总结
        if not self._check_api_called(trajectory, target_api):
            debug_log(self.config, "api_sum_intra_fail", {"reason": "target_api_not_called", "target_api": target_api})
            return []
            
        # 2. 构造 LLM 函数
        # 优先使用专门的 summarize client，如果未定义则回退到默认 client
        client = getattr(self, "llm_client_summarize", self.llm_client)
        llm_fn = self._get_llm_chat_fn(
            client,
            sampling_params={
                "temperature": self.config.get("exploration_llm_temperature", 1.0),
                "top_p": self.config.get("exploration_llm_top_p", 1.0),
                "top_k": self.config.get("exploration_llm_top_k", -1),
            }
        )
        
        # 3. 检索旧任务目标 (如果支持)
        # old_objectives = []
        # if getattr(self, "_old_retrival", None):
        #      old_objectives = self._old_retrival.retrieve_objectives(task)

        # 4. 数据脱敏
        masked_trajectory = copy.deepcopy(trajectory)
        if len(masked_trajectory.steps) > 2:
            if masked_trajectory.steps[1].get('role') == 'user':
                masked_trajectory.steps[1]['content'] = '[MASKED]'
            if masked_trajectory.steps[2].get('role') == 'user':
                masked_trajectory.steps[2]['content'] = '[MASKED]'

        # 5. 生成 Prompt
        env_profile_obj = self._get_env_profile_obj()
        system_prompt, user_prompt = get_task_summarize_prompt(
            [masked_trajectory], env_profile_obj
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_log(self.config, "api_sum_intra", {
            "type": "LLM_INPUT",
            "target_app": target_app,
            "target_api": target_api,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        })

        # 6. 调用 LLM
        try:
            llm_response = llm_fn(messages=messages)
            llm_output = llm_response["content"]

            debug_log(self.config, "api_sum_intra", {
                "type": "LLM_OUTPUT",
                "response": llm_output
            })

        except Exception as e:
            logger.error(f"[Summarize Intra] LLM call failed: {e}")
            return []
        
        # 7. 解析结果
        task_copy = task.copy()
        task_copy.evaluator = 'synthetic'
        tasks = parse_tasks_from_response(task_copy, llm_output)
        
        # 8. 记录记忆（如果成功生成了总结，说明该 App 已被探索）
        if tasks:
            # 取第一个生成的 Query 作为记录
            user_query = tasks[0].input
            
            # [Fix Race Condition] 加锁
            with self._lock:
                if target_app not in self.explored_intra_apps:
                    self.explored_intra_apps.add(target_app)
                    self._save_intra_memory(target_app)
                
            trajectory.info["synthesized_user_query"] = user_query
            trajectory.info["exploration_type"] = "intra_domain"
            
        return tasks

    def summarize_cross(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        跨域探索总结：同样进行后验归纳，解析出实际完成的任务。
        [FIX] 调整返回值为列表，并增加 LLM 归纳步骤。
        """
        info_app = task.metadata.get("info_app")
        target_api = task.metadata.get("target_api")
        
        # 1. 前置检查
        called_info = self._check_app_usage(trajectory, info_app)
        called_exec = self._check_api_called(trajectory, target_api)
        
        if not (called_info and called_exec):
            debug_log(self.config, "api_sum_cross_fail", {
                "reason": "missing_app_usage",
                "called_info_app": called_info, 
                "called_target_api": called_exec
            })
            return []
            
        # 2. 构造 LLM 函数
        client = getattr(self, "llm_client_summarize", self.llm_client)
        llm_fn = self._get_llm_chat_fn(
            client,
            sampling_params={
                "temperature": self.config.get("exploration_llm_temperature", 1.0),
                "top_p": self.config.get("exploration_llm_top_p", 1.0),
                "top_k": self.config.get("exploration_llm_top_k", -1),
            }
        )
        
        # 3. 检索旧任务目标
        # old_objectives = []
        # if getattr(self, "_old_retrival", None):
        #      old_objectives = self._old_retrival.retrieve_objectives(task)
        
        # 4. 数据脱敏
        masked_trajectory = copy.deepcopy(trajectory)
        if len(masked_trajectory.steps) > 2:
            if masked_trajectory.steps[1].get('role') == 'user':
                masked_trajectory.steps[1]['content'] = '[MASKED]'
            if masked_trajectory.steps[2].get('role') == 'user':
                masked_trajectory.steps[2]['content'] = '[MASKED]'
        
        # 5. 生成 Prompt
        env_profile_obj = self._get_env_profile_obj()
        system_prompt, user_prompt = get_task_summarize_prompt(
            [masked_trajectory], old_objectives=[], env_profile=env_profile_obj
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_log(self.config, "api_sum_cross", {
            "type": "LLM_INPUT",
            "info_app": info_app,
            "target_api": target_api,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        })

        # 6. 调用 LLM
        try:
            llm_response = llm_fn(messages=messages)
            llm_output = llm_response["content"]

            debug_log(self.config, "api_sum_cross", {
                "type": "LLM_OUTPUT",
                "response": llm_output
            })
            
        except Exception as e:
            logger.error(f"[Summarize Cross] LLM call failed: {e}")
            return []
            
        # 7. 解析结果
        task_copy = task.copy()
        task_copy.evaluator = 'synthetic'
        tasks = parse_tasks_from_response(task_copy, llm_output)
        
        # 8. 记录记忆
        if tasks:
            user_query = tasks[0].input
            trajectory.info["synthesized_user_query"] = user_query
            trajectory.info["exploration_type"] = "cross_domain"
            
            # [Fix Race Condition] 加锁
            with self._lock:
                self._save_cross_memory(trajectory.info)
            
        return tasks

    # ================= 辅助私有方法 =================

    def _get_env_profile_obj(self):
        """辅助方法：根据配置字符串获取 Profile 对象"""
        env_type = self.env_profile_name
        if env_type == "appworld":
            return appworld.AppWorldProfile()
        elif env_type == "bfcl":
            return bfcl.BfclProfile()
        elif env_type == "webshop":
            return webshop.WebShopProfile()
        return None

    def _chat_with_retry(self, messages: List[Dict], **kwargs) -> Optional[Any]:
        from types import SimpleNamespace # 用于快速创建带属性的对象

        for i in range(self._max_llm_retries):
            try:
                # 调用 LLM 客户端
                response = self.llm_client.chat(messages=messages, **kwargs)
                
                #情况 A: Client 返回字符串 (如 DashScopeClient)
                if isinstance(response, str):
                    if response.strip(): # 确保内容不为空
                        # 包装成对象，使其具有 .content 属性，兼容调用方的 response.content
                        return SimpleNamespace(content=response)
                
                # 情况 B: Client 返回对象 (如标准 OpenAI Client)
                elif response and hasattr(response, 'content') and response.content:
                    return response
                    
            except Exception as e:
                logger.warning(f"LLM call failed: {e}. Retry {i+1}...")
            
            # 指数退避重试
            if i < self._max_llm_retries - 1:
                time.sleep(2 ** i)
                
        return None

    def _load_sandbox_task_ids(self, path: str) -> List[str]:
        if not os.path.exists(path):
            logger.error("Label file missing")
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
            if step.role == "tool" and not step.error:
                if api_name in step.tool_name: return True
        return False

    def _check_app_usage(self, trajectory: Trajectory, app_name: str) -> bool:
        if not trajectory or not trajectory.steps: return False
        app_apis = self.api_knowledge.get(app_name, {}).get("apis", {}).keys()
        for step in trajectory.steps:
            if step.role == "tool":
                if app_name.lower() in step.tool_name.lower(): return True
                for api in app_apis:
                    if api in step.tool_name: return True
        return False

    def _extract_tool_trace(self, trajectory: Trajectory) -> str:
        trace = []
        for step in trajectory.steps:
            if step.role == "assistant" and step.tool_calls:
                for tc in step.tool_calls:
                    trace.append(f"Action: {tc['name']} args={tc['arguments']}")
            elif step.role == "tool":
                content = str(step.content)
                if len(content) > 200: content = content[:200] + "..."
                trace.append(f"Observation: {content}")
        return "\n".join(trace)

    def _load_json(self, path: str) -> Dict:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {}

    def _save_intra_memory(self, app_name: str):
        # 注意：此方法应在锁保护下调用，或内部加锁。目前上层 summarise 方法已加锁。
        # 为安全起见，这里不需要额外加锁，只要保证调用方正确。
        debug_log(self.config, "api_memory_save_intra", {"app_name": app_name})
        os.makedirs(os.path.dirname(self.intra_memory_path), exist_ok=True)
        current_data = self._load_json(self.intra_memory_path)
        current_apps = set(current_data.get("explored_apps", []))
        current_apps.add(app_name)
        with open(self.intra_memory_path, 'w', encoding='utf-8') as f:
            json.dump({"explored_apps": list(current_apps)}, f, indent=2)

    def _save_cross_memory(self, metadata: Dict):
        debug_log(self.config, "api_memory_save_cross", {"synthesized_query": metadata.get("synthesized_user_query")})
        os.makedirs(os.path.dirname(self.cross_memory_path), exist_ok=True)
        current_data = self._load_json(self.cross_memory_path)
        if "logs" not in current_data: current_data["logs"] = []
        current_data["logs"].append(metadata)
        with open(self.cross_memory_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2)

    def _get_llm_chat_fn(self, llm_client:LlmClient, sampling_params: Optional[dict] = None) -> Callable:
        """
        辅助函数：封装 LLM 客户端调用，增加重试机制和参数合并。
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
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(2**i)

            assert res is not None and res != "", f"LLM client failed to chat"
            
            return {
                "role": "assistant",
                "content": res,
            }

        return llm_chat