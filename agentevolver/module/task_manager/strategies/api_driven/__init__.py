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
from agentevolver.module.task_manager.base import LlmClient
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
    Decomposed into:
    1. Task Generation (Intra/Cross)
    2. Summarization/Back-Translation
    """

    def __init__(self, tokenizer, config, **kwargs):
        super().__init__(tokenizer, config)
        self.tokenizer = tokenizer
        self.config = config
        
        # 1. 初始化 LLM Client
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
        self.explored_intra_apps = set(self.intra_memory_data.get("explored_apps", []))
        self.cross_memory_data = self._load_json(self.cross_memory_path)

        logger.info(f"[ApiDriven] Initialized. Mastered Apps (Intra): {list(self.explored_intra_apps)}")

    # ================= 核心接口：兼容 TaskManager 调用 =================

    def get_next_sandbox_id(self) -> str:
        """提供给 TaskManager 获取环境锚点"""
        try:
            return next(self.sandbox_id_iterator)
        except StopIteration:
            return "train_001" # Should not happen with cycle

    def decide_phase(self) -> str:
        """决定当前是单域还是跨域阶段"""
        unmastered_apps = list(self.active_apps - self.explored_intra_apps)
        if unmastered_apps:
            return "intra"
        if len(self.active_apps) >= 2:
            return "extra" # cross-domain
        return "done"

    # ================= 阶段生成逻辑 (Generation) =================

    def generate_intra_task(self) -> Optional[Task]:
        """生成单域探索任务 (Plan Generation)"""
        unmastered_apps = list(self.active_apps - self.explored_intra_apps)
        if not unmastered_apps:
            return None
            
        app_name = random.choice(unmastered_apps)
        logger.info(f"[ApiDriven] Generating Intra-Domain Task for: {app_name}")
        
        app_knowledge = self.api_knowledge.get(app_name, {})
        apis = app_knowledge.get("apis", {})
        
        # 筛选 API
        action_apis = [k for k, v in apis.items() if v.get("action_type") == "Executive Action"]
        if not action_apis:
            action_apis = [k for k, v in apis.items() if v.get("method", "").upper() != "GET" and len(v.get("parameters", [])) > 0]
        
        if not action_apis:
            logger.warning(f"No executable actions for {app_name}")
            return None

        target_api_name = random.choice(action_apis)
        target_api_def = apis[target_api_name]
        info_apis = {k: v for k, v in apis.items() if k != target_api_name and v.get("action_type") == "Informational Action"}

        # LLM 规划
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
        
        # 构造 Task 对象
        task = Task(
            task_id="intra_placeholder", # 将在 Manager 中被覆盖
            instruction=instruction,
            metrics={
                "phase": "intra",
                "target_app": app_name,
                "target_api": target_api_name
            }
        )
        return task

    def generate_cross_task(self) -> Optional[Task]:
        """生成跨域合成任务 (Purpose Synthesis)"""
        is_info_provider = lambda a: a in UNIVERSAL_INFO_PROVIDERS
        
        # 选择 App Pair
        primary_app = random.choice(list(self.explored_intra_apps)) # 必须是已掌握的
        info_app, exec_app = "", ""
        
        if is_info_provider(primary_app):
            info_app = primary_app
            candidates = [a for a in self.explored_intra_apps if not is_info_provider(a)]
            if not candidates: return None
            exec_app = random.choice(candidates)
        else:
            exec_app = primary_app
            candidates = [a for a in self.explored_intra_apps if is_info_provider(a)]
            if not candidates: return None
            info_app = random.choice(candidates)

        logger.info(f"[ApiDriven] Generating Cross-Domain Task: {info_app} -> {exec_app}")

        # 目的合成
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

        task = Task(
            task_id="cross_placeholder",
            instruction=user_query,
            metrics={
                "phase": "extra",
                "setup_action": "inject_data",
                "app": info_app,
                "content": setup_context,
                "info_app": info_app,
                "exec_app": exec_app,
                "target_api": target_action
            }
        )
        return task

    # ================= 阶段总结逻辑 (Summarize) =================

    def summarize_intra(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        """单域：验证 API 调用并进行反向归纳"""
        target_app = task.metrics.get("target_app")
        target_api = task.metrics.get("target_api")
        
        if not self._check_api_called(trajectory, target_api):
            logger.info(f"[Intra-Domain] Failed: Target API {target_api} not called.")
            return None
            
        logger.info(f"[Intra-Domain] Success: Back-translating...")
        tool_trace = self._extract_tool_trace(trajectory)
        
        bt_prompt = BACK_TRANSLATION_PROMPT.format(
            tool_calls_trace=tool_trace,
            target_api_name=target_api
        )
        
        bt_response = self._chat_with_retry(messages=[{"role": "user", "content": bt_prompt}])
        if not bt_response:
            return None
            
        user_query = bt_response.content.strip()
        
        # 更新状态：标记 APP 为已掌握
        if target_app not in self.explored_intra_apps:
            self.explored_intra_apps.add(target_app)
            self._save_intra_memory(target_app)
            
        # 构造训练样本
        trajectory.info["synthesized_user_query"] = user_query
        trajectory.info["exploration_type"] = "intra_domain"
        
        return TaskObjective(input=user_query, output=trajectory)

    def summarize_cross(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        """跨域：验证信息流和执行动作"""
        info_app = task.metrics.get("info_app")
        exec_app = task.metrics.get("exec_app")
        target_api = task.metrics.get("target_api")
        user_query = task.instruction # 跨域的任务已经是合成好的 query，不需要反向归纳，只需验证
        
        called_info = self._check_app_usage(trajectory, info_app)
        called_exec = self._check_api_called(trajectory, target_api)
        
        if called_info and called_exec:
            logger.info(f"[Cross-Domain] Success: {info_app} -> {exec_app}")
            trajectory.info["synthesized_user_query"] = user_query
            trajectory.info["exploration_type"] = "cross_domain"
            
            # 保存日志
            self._save_cross_memory(trajectory.info)
            return TaskObjective(input=user_query, output=trajectory)
        
        return None

    # ================= 辅助方法 =================

    def _chat_with_retry(self, messages: List[Dict], **kwargs) -> Optional[Any]:
        for i in range(self._max_llm_retries):
            try:
                response = self.llm_client.chat(messages=messages, **kwargs)
                if response and response.content: return response
            except Exception as e:
                logger.warning(f"LLM call failed: {e}. Retry {i+1}...")
            time.sleep(2 ** i)
        return None

    def _load_sandbox_task_ids(self, path: str) -> List[str]:
        if not os.path.exists(path):
            # 简化处理，实际应抛出异常
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
        os.makedirs(os.path.dirname(self.intra_memory_path), exist_ok=True)
        current_data = self._load_json(self.intra_memory_path)
        current_apps = set(current_data.get("explored_apps", []))
        current_apps.add(app_name)
        with open(self.intra_memory_path, 'w', encoding='utf-8') as f:
            json.dump({"explored_apps": list(current_apps)}, f, indent=2)

    def _save_cross_memory(self, metadata: Dict):
        os.makedirs(os.path.dirname(self.cross_memory_path), exist_ok=True)
        current_data = self._load_json(self.cross_memory_path)
        if "logs" not in current_data: current_data["logs"] = []
        current_data["logs"].append(metadata)
        with open(self.cross_memory_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2)