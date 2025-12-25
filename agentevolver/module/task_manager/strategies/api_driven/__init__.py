# agentevolver/module/task_manager/strategies/api_driven/__init__.py

import json
import os
import random
import copy
import time
import itertools
from typing import List, Dict, Any, Optional, Set

from loguru import logger

from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from agentevolver.client.llm_client import LlmClient
from agentevolver.module.env_manager.env_worker import EnvWorker
from agentevolver.module.env_manager.env_manager import EnvManager
from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.utils.utils import extract_json_from_str

# 引入 Prompt
from agentevolver.module.task_manager.strategies.api_driven.prompts import (
    PLAN_GENERATION_PROMPT,
    BACK_TRANSLATION_PROMPT,
    PURPOSE_SYNTHESIS_PROMPT
)

# 定义通用信息提供者集合
UNIVERSAL_INFO_PROVIDERS = {"notes", "gmail", "simple_messages", "calendar", "contacts"}

class ApiDrivenExploreStrategy(TaskExploreStrategy):
    """
    API Driven Exploration Strategy
    Phase 1: Intra-Domain Reverse Semantic Exploration (Single App Mastery)
    Phase 2: Purpose-Driven Cross-Domain Synthesis (Compositional Generalization)
    """

    def __init__(self, tokenizer, config, **kwargs):
        super().__init__(tokenizer, config)
        self.tokenizer = tokenizer
        self.config = config
        
        # 1. 初始化 LLM Client (关键修正)
        self.llm_client = LlmClient(config)
        self._max_llm_retries = kwargs.get("max_llm_retries", 3)
        
        # --- 路径配置 ---
        self.api_knowledge_path = kwargs.get(
            "api_knowledge_path", 
            "agentevolver/preprocess/output/appworld_tool_manual.json"
        )
        self.task_labels_path = kwargs.get(
            "task_labels_path", 
            "agentevolver/preprocess/output/task_app_labels_train.json"
        )
        
        # 拆分记忆文件路径
        base_memory_dir = "data/memory/api_driven"
        self.intra_memory_path = kwargs.get("intra_memory_path", os.path.join(base_memory_dir, "intra_domain_success.json"))
        self.cross_memory_path = kwargs.get("cross_memory_path", os.path.join(base_memory_dir, "cross_domain_success.json"))
        
        self.active_apps = set(kwargs.get("active_apps", []))
        
        # 初始化组件占位符
        self.agent_flow: Optional[AgentFlow] = None
        self.env_worker: Optional[EnvWorker] = None
        
        # 2. 加载 API 知识库
        self.api_knowledge = self._load_json(self.api_knowledge_path)
        if not self.api_knowledge:
            logger.warning(f"API Knowledge not found at {self.api_knowledge_path}. Exploration might fail.")

        # 3. 获取 Sandbox Task IDs 并构建循环迭代器
        self.sandbox_ids_pool = self._load_sandbox_task_ids(self.task_labels_path)
        self.sandbox_id_iterator = itertools.cycle(self.sandbox_ids_pool)
        logger.info(f"[ApiDriven] Loaded {len(self.sandbox_ids_pool)} Sandbox Task IDs.")

        # 4. 加载已完成单域探索的 APP 记录
        self.intra_memory_data = self._load_json(self.intra_memory_path)
        # explored_intra_apps: 记录哪些 APP 已经跑通了单域探索
        self.explored_intra_apps = set(self.intra_memory_data.get("explored_apps", []))
        
        # 加载跨域日志（用于统计或避免重复，目前主要用于追加写入）
        self.cross_memory_data = self._load_json(self.cross_memory_path)

        logger.info(f"[ApiDriven] Initialized. Mastered Apps (Intra): {list(self.explored_intra_apps)}")

    def _chat_with_retry(self, messages: List[Dict], **kwargs) -> Optional[Any]:
        """
        封装 LLM 调用，增加手动实现的指数退避重试机制。
        """
        for i in range(self._max_llm_retries):
            try:
                # 调用基础 client
                response = self.llm_client.chat(messages=messages, **kwargs)
                
                # 简单验证结果有效性
                if response and response.content:
                    return response
                
                logger.warning(f"LLM returned empty response. Retry {i+1}/{self._max_llm_retries}...")
            except Exception as e:
                logger.warning(f"LLM call failed: {e}. Retry {i+1}/{self._max_llm_retries}...")
                
            # 指数退避: 1s, 2s, 4s
            time.sleep(2 ** i)
            
        logger.error(f"LLM failed after {self._max_llm_retries} retries.")
        return None

    def _load_sandbox_task_ids(self, path: str) -> List[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ApiDriven] Critical Error: Task labels file not found at '{path}'. "
                "Please run 'python -m agentevolver.preprocess.main' first."
            )
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"[ApiDriven] Failed to parse JSON from '{path}': {e}")

        task_ids = [item["TaskID"] for item in data if "TaskID" in item]
        if not task_ids:
            raise ValueError(f"[ApiDriven] No valid TaskID found in '{path}'.")
        return task_ids

    def explore(self, task: Task, data_id: str, rollout_id: str) -> List[Trajectory]:
        """
        核心入口：根据 APP 的掌握情况，决定进入 Phase 1 (单域) 还是 Phase 2 (跨域)。
        """
        self._ensure_execution_capability()

        if not self.active_apps:
            logger.warning("No active apps configured for exploration.")
            return []

        # --- 策略调度逻辑 ---
        # 1. 优先检查是否有未完成单域探索的 APP (Phase 1)
        unmastered_apps = list(self.active_apps - self.explored_intra_apps)
        
        trajectories = []
        
        if unmastered_apps:
            # Phase 1: 单域探索
            target_app = random.choice(unmastered_apps)
            logger.info(f"[Strategy] Phase 1: Intra-Domain Exploration for '{target_app}'")
            traj = self._run_intra_domain_exploration(target_app)
            
            if traj:
                trajectories.append(traj)
                # 标记该 APP 为已掌握
                self.explored_intra_apps.add(target_app)
                self._save_intra_memory(target_app)
        else:
            # Phase 2: 跨域合成 (所有 APP 已掌握单域，开始排列组合)
            if len(self.active_apps) < 2:
                logger.info("[Strategy] Not enough apps for cross-domain exploration.")
                return []
                
            target_app = random.choice(list(self.active_apps)) # 作为 Primary App
            logger.info(f"[Strategy] Phase 2: Cross-Domain Synthesis anchored on '{target_app}'")
            traj = self._run_cross_domain_synthesis(target_app)
            
            if traj:
                trajectories.append(traj)
                self._save_cross_memory(traj.info)

        return trajectories

    def summarize(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        将探索成功的 Trajectory 转化为微调数据。
        """
        if not trajectory or not trajectory.steps:
            return []
        
        user_query = trajectory.info.get("synthesized_user_query")
        if user_query:
            return [TaskObjective(input=user_query, output=trajectory)]
        return []

    # ================= 阶段二：单域逆向语义探索 =================

    def _run_intra_domain_exploration(self, app_name: str) -> Optional[Trajectory]:
        app_knowledge = self.api_knowledge.get(app_name, {})
        apis = app_knowledge.get("apis", {})
        if not apis:
            return None

        # 1. 锚定终态
        action_apis = [
            k for k, v in apis.items() 
            if v.get("action_type") == "Executive Action"
        ]
        # 兜底
        if not action_apis:
            action_apis = [
                k for k, v in apis.items() 
                if v.get("method", "").upper() != "GET" and len(v.get("parameters", [])) > 0
            ]
        
        if not action_apis:
            logger.warning(f"No executable actions found for {app_name}")
            return None
        
        target_api_name = random.choice(action_apis)
        target_api_def = apis[target_api_name]
        
        info_apis = {
            k: v for k, v in apis.items() 
            if k != target_api_name and v.get("action_type") == "Informational Action"
        }

        # 2. LLM 规划 (使用重试机制)
        prompt = PLAN_GENERATION_PROMPT.format(
            target_api_name=target_api_name,
            app_name=app_name,
            target_api_details=json.dumps(target_api_def, indent=2, ensure_ascii=False),
            available_info_apis=json.dumps(info_apis, indent=2, ensure_ascii=False)
        )
        
        response = self._chat_with_retry(messages=[{"role": "user", "content": prompt}], temperature=0.7)
        if not response:
            return None
            
        instruction = response.content.strip()

        # 3. 执行
        exec_task = Task(
            task_id="intra_placeholder", # 将被 sandbox id 覆盖
            instruction=instruction,
            metrics={"target_app": app_name}
        )
        
        trajectory = self._execute_agent_loop(exec_task)

        # 4. 验证与反向归纳
        if self._check_api_called(trajectory, target_api_name):
            logger.info(f"[Intra-Domain] Success: {target_api_name}")
            tool_trace = self._extract_tool_trace(trajectory)
            
            bt_prompt = BACK_TRANSLATION_PROMPT.format(
                tool_calls_trace=tool_trace,
                target_api_name=target_api_name
            )
            # 使用重试机制
            bt_response = self._chat_with_retry(messages=[{"role": "user", "content": bt_prompt}])
            if not bt_response:
                logger.warning("Back translation failed after retries.")
                return None
                
            user_query = bt_response.content.strip()
            
            trajectory.info["synthesized_user_query"] = user_query
            trajectory.info["exploration_type"] = "intra_domain"
            trajectory.info["target_app"] = app_name
            return trajectory
        else:
            return None

    # ================= 阶段三：跨域合成 =================

    def _run_cross_domain_synthesis(self, primary_app: str) -> Optional[Trajectory]:
        # 1. 角色定义与配对
        is_info_provider = primary_app in UNIVERSAL_INFO_PROVIDERS
        
        info_app = ""
        exec_app = ""
        
        if is_info_provider:
            info_app = primary_app
            # 必须从已掌握的 APP 中选（保证单域基础能力）
            candidates = [a for a in self.explored_intra_apps if a not in UNIVERSAL_INFO_PROVIDERS]
            if not candidates: return None
            exec_app = random.choice(candidates)
        else:
            exec_app = primary_app
            candidates = [a for a in self.explored_intra_apps if a in UNIVERSAL_INFO_PROVIDERS]
            if not candidates: return None
            info_app = random.choice(candidates)
            
        # 2. 目的合成
        info_desc = self.api_knowledge.get(info_app, {}).get("description", "")
        exec_desc = self.api_knowledge.get(exec_app, {}).get("description", "")
        exec_apis = list(self.api_knowledge.get(exec_app, {}).get("apis", {}).keys())[:15]

        prompt = PURPOSE_SYNTHESIS_PROMPT.format(
            info_app_name=info_app,
            info_app_desc=info_desc,
            exec_app_name=exec_app,
            exec_app_desc=exec_desc,
            exec_api_list=json.dumps(exec_apis)
        )
        
        # 使用重试机制
        response = self._chat_with_retry(
            messages=[{"role": "user", "content": prompt}], 
            response_format={"type": "json_object"}
        )
        if not response:
            return None
        
        try:
            scenario_data = extract_json_from_str(response.content)
            setup_context = scenario_data.get("setup_context", "")
            user_query = scenario_data.get("user_query", "")
            target_action = scenario_data.get("target_action_api", "")
        except Exception:
            return None

        # 3. 环境预设
        exec_task = Task(
            task_id="cross_placeholder",
            instruction=user_query,
            metrics={
                "setup_action": "inject_data",
                "app": info_app,
                "content": setup_context
            }
        )

        # 4. 执行与验证
        trajectory = self._execute_agent_loop(exec_task)
        
        called_info = self._check_app_usage(trajectory, info_app)
        called_exec = self._check_api_called(trajectory, target_action)
        
        if called_info and called_exec:
            logger.info(f"[Cross-Domain] Success: {info_app} -> {exec_app}")
            trajectory.info["synthesized_user_query"] = user_query
            trajectory.info["exploration_type"] = "cross_domain"
            trajectory.info["app_pair"] = f"{info_app}-{exec_app}"
            return trajectory
        
        return None

    # ================= 辅助方法 =================

    def _ensure_execution_capability(self):
        if self.agent_flow is None:
            self.agent_flow = AgentFlow(self.config, self.tokenizer)
        if self.env_worker is None:
            from agentevolver.client.env_client import EnvClient
            env_config = self.config.get("env_config", {})
            env_client = EnvClient(**env_config)
            self.env_worker = EnvWorker(env_client=env_client)

    def _execute_agent_loop(self, task: Task) -> Trajectory:
        generated_instruction = task.instruction
        try:
            real_sandbox_id = next(self.sandbox_id_iterator)
            task.task_id = real_sandbox_id
        except StopIteration:
            logger.error("Sandbox ID iterator exhausted.")
            return Trajectory(steps=[])
        
        obs = self.env_worker.reset(task)
        task.instruction = generated_instruction
        
        if task.metrics and task.metrics.get("setup_action") == "inject_data":
            self._inject_context(task.metrics)

        try:
            return self.agent_flow.run(task, self.env_worker)
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return Trajectory(steps=[])

    def _inject_context(self, metrics: Dict):
        try:
            app = metrics.get("app")
            content = metrics.get("content")
            if hasattr(self.env_worker, "execute"):
                # 简单日志记录，实际应调用 self.env_worker.execute(...)
                logger.info(f"[Inject] {app}: {content}")
        except Exception as e:
            logger.warning(f"Injection failed: {e}")

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
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_intra_memory(self, app_name: str):
        """保存单域探索进度"""
        os.makedirs(os.path.dirname(self.intra_memory_path), exist_ok=True)
        # 加载最新状态防止覆盖
        current_data = self._load_json(self.intra_memory_path)
        current_apps = set(current_data.get("explored_apps", []))
        current_apps.add(app_name)
        
        with open(self.intra_memory_path, 'w', encoding='utf-8') as f:
            json.dump({"explored_apps": list(current_apps)}, f, indent=2)

    def _save_cross_memory(self, metadata: Dict):
        """保存跨域探索的元数据日志"""
        os.makedirs(os.path.dirname(self.cross_memory_path), exist_ok=True)
        current_data = self._load_json(self.cross_memory_path)
        if "logs" not in current_data:
            current_data["logs"] = []
        
        current_data["logs"].append(metadata)
        
        with open(self.cross_memory_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2)