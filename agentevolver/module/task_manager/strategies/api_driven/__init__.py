import json
import os
import random
import copy
import time
import itertools
from typing import List, Dict, Any, Optional, Set

from loguru import logger

# 导入基础策略类和数据模型
from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from agentevolver.module.task_manager.base import LlmClient
from agentevolver.utils.utils import extract_json_from_str

# 引入预定义的 Prompt 模板
from agentevolver.module.task_manager.strategies.api_driven.prompts import (
    PLAN_GENERATION_PROMPT,     # 用于生成单域探索计划
    BACK_TRANSLATION_PROMPT,    # 用于将执行轨迹反向翻译为用户查询
    PURPOSE_SYNTHESIS_PROMPT    # 用于合成跨域任务目的
)

# 定义通用信息提供者 App 集合（通常作为跨域任务的数据源，如从邮件或笔记中获取信息）
UNIVERSAL_INFO_PROVIDERS = {"notes", "gmail", "simple_messages", "calendar", "contacts"}

class ApiDrivenExploreStrategy(TaskExploreStrategy):
    """
    API 驱动的探索策略类
    该策略将探索分解为：
    1. 任务生成阶段（单域/跨域）
    2. 总结与反向翻译阶段（验证并提炼训练数据）
    """

    def __init__(self, tokenizer, config, **kwargs):
        """
        初始化探索策略
        :param tokenizer: 分词器对象
        :param config: 配置字典
        :param kwargs: 其他可选参数，如路径配置、活跃 App 列表等
        """
        super().__init__(tokenizer, config)
        self.tokenizer = tokenizer
        self.config = config
        
        # 1. 初始化 LLM 客户端，用于生成和总结任务
        self.llm_client = LlmClient(config)
        self._max_llm_retries = kwargs.get("max_llm_retries", 3) # 最大重试次数
        
        # --- 路径与文件配置 ---
        # API 知识库路径（包含所有 App 的 API 定义）
        self.api_knowledge_path = kwargs.get(
            "api_knowledge_path", 
            "agentevolver/preprocess/output/appworld_tool_manual.json"
        )
        # 任务标签路径（用于获取 Sandbox Task ID）
        self.task_labels_path = kwargs.get(
            "task_labels_path", 
            "agentevolver/preprocess/output/task_app_labels_train.json"
        )
        
        # 记忆文件路径：记录哪些 App 已经成功探索过
        base_memory_dir = "data/memory/api_driven"
        self.intra_memory_path = kwargs.get("intra_memory_path", os.path.join(base_memory_dir, "intra_domain_success.json"))
        self.cross_memory_path = kwargs.get("cross_memory_path", os.path.join(base_memory_dir, "cross_domain_success.json"))
        
        # 当前环境下激活的 App 列表
        self.active_apps = set(kwargs.get("active_apps", []))
        
        # 2. 加载 API 知识库数据
        self.api_knowledge = self._load_json(self.api_knowledge_path)
        if not self.api_knowledge:
            logger.warning(f"API Knowledge not found at {self.api_knowledge_path}. Exploration might fail.")

        # 3. 加载沙箱任务 ID 池，并构建无限循环迭代器，作为环境重置的锚点
        self.sandbox_ids_pool = self._load_sandbox_task_ids(self.task_labels_path)
        self.sandbox_id_iterator = itertools.cycle(self.sandbox_ids_pool)
        logger.info(f"[ApiDriven] Loaded {len(self.sandbox_ids_pool)} Sandbox Task IDs.")

        # 4. 加载已完成单域探索的 APP 记录（长期记忆）
        self.intra_memory_data = self._load_json(self.intra_memory_path)
        self.explored_intra_apps = set(self.intra_memory_data.get("explored_apps", []))
        self.cross_memory_data = self._load_json(self.cross_memory_path)

        logger.info(f"[ApiDriven] Initialized. Mastered Apps (Intra): {list(self.explored_intra_apps)}")

    # ================= 核心接口：兼容 TaskManager 调用 =================

    def get_next_sandbox_id(self) -> str:
        """
        获取下一个沙箱 ID，为 Agent 提供初始环境状态。
        TaskManager 会调用此函数来决定在哪个环境下开始探索。
        """
        try:
            return next(self.sandbox_id_iterator)
        except StopIteration:
            return "train_001" # 理论上不会发生，因为使用了 cycle

    def decide_phase(self) -> str:
        """
        决定当前的探索阶段。
        1. 如果还有未探索的活跃 App，优先进行 'intra'（单域）。
        2. 如果全部掌握且有多个 App，进行 'extra'（跨域）。
        3. 否则结束探索。
        """
        unmastered_apps = list(self.active_apps - self.explored_intra_apps)
        if unmastered_apps:
            return "intra"
        if len(self.active_apps) >= 2:
            return "extra" # 跨域探索
        return "done"

    # ================= 阶段生成逻辑 (Generation) =================

    def generate_intra_task(self, app_name: str = None, target_api_name: str = None) -> Optional[Task]:
        """
        生成单域探索任务：支持指定 API，并按 7:3 比例探索执行/信息类功能。
        """
        # 1. 确定目标 App
        if not app_name:
            unmastered_apps = list(self.active_apps - self.explored_intra_apps)
            app_name = random.choice(unmastered_apps) if unmastered_apps else random.choice(list(self.active_apps))
        
        logger.debug(f"[ApiDriven] Generating Task for: {app_name}")
        app_knowledge = self.api_knowledge.get(app_name, {})
        apis = app_knowledge.get("apis", {})

        # 2. 确定目标 API
        if not target_api_name:
            # 分类 API
            action_apis = [k for k, v in apis.items() if v.get("action_type") == "Executive Action"]
            info_apis_list = [k for k, v in apis.items() if v.get("action_type") == "Informational Action"]
            
            roll = random.random()
            if roll < 0.7 and action_apis:
                # 70% 概率探索执行类
                target_api_name = random.choice(action_apis)
            elif info_apis_list:
                # 30% 概率探索信息类（或执行类为空时兜底）
                target_api_name = random.choice(info_apis_list)
            else:
                # 极端兜底：随机选一个
                target_api_name = random.choice(list(apis.keys())) if apis else None

        if not target_api_name:
            logger.warning(f"No suitable APIs found for {app_name}")
            return None

        target_api_def = apis.get(target_api_name)
        
        # 3. 准备参考 API：如果探索的是执行类，则提供所有信息类 API 给 LLM 做前置依赖参考
        # 如果探索的是信息类，则提供部分执行类作为背景参考
        is_executive = target_api_def.get("action_type") == "Executive Action"
        ref_type = "Informational Action" if is_executive else "Executive Action"
        reference_apis = {k: v for k, v in apis.items() if k != target_api_name and v.get("action_type") == ref_type}

        # 4. 调用 LLM 生成指令
        prompt = PLAN_GENERATION_PROMPT.format(
            target_api_name=target_api_name,
            app_name=app_name,
            target_api_details=json.dumps(target_api_def, indent=2, ensure_ascii=False),
            available_info_apis=json.dumps(reference_apis, indent=2, ensure_ascii=False)
        )
        
        response = self._chat_with_retry(messages=[{"role": "user", "content": prompt}], temperature=0.7)
        if not response: return None
            
        return Task(
            task_id="intra_placeholder",
            instruction=response.content.strip(),
            metrics={"phase": "intra", "target_app": app_name, "target_api": target_api_name}
        )

    def generate_cross_task(self, app_list: List[str] = None) -> Optional[Task]:
        """
        生成跨域探索任务：从指定 APP 列表中随机采样 5+5 API 并合成泛泛的探索指令。
        """
        # 0. 默认 APP 列表
        if not app_list:
            app_list = [
                "venmo", "spotify", "phone", "file_system", "simple_note", 
                "amazon", "gmail", "splitwise", "todoist"
            ]
        
        available_apps = [app for app in app_list if app in self.api_knowledge]
        if len(available_apps) < 2:
            logger.warning("[ApiDriven] Not enough active apps for cross-domain task.")
            return None
        
        # 1. 随机选择 2 个不同的 APP
        selected_apps = random.sample(available_apps, 2)
        app_a, app_b = selected_apps[0], selected_apps[1]
        
        # 简单角色分配逻辑：优先让强信息类 App 做 Source，强执行类 App 做 Target
        if app_a not in UNIVERSAL_INFO_PROVIDERS and app_b in UNIVERSAL_INFO_PROVIDERS:
            info_app_name, exec_app_name = app_b, app_a
        else:
            info_app_name, exec_app_name = app_a, app_b

        # 2. 内部采样函数
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

        logger.info(f"[ApiDriven] Cross-Exploration: {info_app_name} (Info) -> {exec_app_name} (Exec)")

        # 3. 准备 Prompt 参数
        system_tools_hint = (
            "Available System Tools (Use these to verify logic or specs):\n"
            "- supervisor: Use to coordinate the multi-step process or report final success.\n"
            "- api_docs: Use to check parameter formats or search syntax (e.g., query operators for email) before calling tools."
        )

        # 4. 填充 Prompt 模板
        prompt = PURPOSE_SYNTHESIS_PROMPT.format(
            info_app_name=info_app_name,
            exec_app_name=exec_app_name,
            info_apis_json=json.dumps(info_apis, indent=2, ensure_ascii=False),
            exec_apis_json=json.dumps(exec_apis, indent=2, ensure_ascii=False),
            system_tools_hint=system_tools_hint
        )

        # 5. 调用 LLM
        response = self._chat_with_retry(
            messages=[{"role": "user", "content": prompt}], 
            response_format={"type": "json_object"}
        )
        if not response: return None
        
        try:
            res_json = extract_json_from_str(response.content)
            user_query = res_json.get("user_query", "")
            target_action = res_json.get("target_action_api", "")
        except Exception: return None

        return Task(
            task_id="cross_placeholder",
            instruction=user_query,
            metrics={
                "phase": "extra",
                "info_app": info_app_name,
                "exec_app": exec_app_name,
                "target_api": target_action,
                "sampled_info_apis": list(info_apis.keys()),
                "sampled_exec_apis": list(exec_apis.keys())
            }
        )

    # ================= 阶段总结逻辑 (Summarize) =================

    def summarize_intra(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        """
        单域探索总结：验证生成的指令是否真的触发了目标 API。
        如果成功触发，则将这一串 API 调用轨迹“反向归纳”成一个简洁的自然语言 Query。
        """
        target_app = task.metrics.get("target_app")
        target_api = task.metrics.get("target_api")
        
        # 1. 检查执行轨迹中是否包含目标 API 且无报错
        if not self._check_api_called(trajectory, target_api):
            logger.info(f"[Intra-Domain] Failed: Target API {target_api} not called.")
            return None
            
        logger.info(f"[Intra-Domain] Success: Back-translating...")
        
        # 2. 提取 API 调用序列（Trace）
        tool_trace = self._extract_tool_trace(trajectory)
        
        # 3. 使用 LLM 进行反向归纳 (Back-Translation)
        bt_prompt = BACK_TRANSLATION_PROMPT.format(
            tool_calls_trace=tool_trace,
            target_api_name=target_api
        )
        
        bt_response = self._chat_with_retry(messages=[{"role": "user", "content": bt_prompt}])
        if not bt_response:
            return None
            
        user_query = bt_response.content.strip()
        
        # 4. 更新探索状态：将此 App 加入“已掌握”记忆
        if target_app not in self.explored_intra_apps:
            self.explored_intra_apps.add(target_app)
            self._save_intra_memory(target_app)
            
        # 5. 补充轨迹信息，返回高质量的任务目标（用于模型训练）
        trajectory.info["synthesized_user_query"] = user_query
        trajectory.info["exploration_type"] = "intra_domain"
        
        return TaskObjective(input=user_query, output=trajectory)

    def summarize_cross(self, task: Task, trajectory: Trajectory) -> Optional[TaskObjective]:
        """
        跨域探索总结：验证信息流是否打通。
        验证点：是否使用了信息源 App，且是否调用了目标执行 API。
        """
        info_app = task.metrics.get("info_app")
        exec_app = task.metrics.get("exec_app")
        target_api = task.metrics.get("target_api")
        user_query = task.instruction 
        
        # 验证两个 App 是否都被用到
        called_info = self._check_app_usage(trajectory, info_app)
        called_exec = self._check_api_called(trajectory, target_api)
        
        if called_info and called_exec:
            logger.info(f"[Cross-Domain] Success: {info_app} -> {exec_app}")
            trajectory.info["synthesized_user_query"] = user_query
            trajectory.info["exploration_type"] = "cross_domain"
            
            # 保存跨域成功日志
            self._save_cross_memory(trajectory.info)
            return TaskObjective(input=user_query, output=trajectory)
        
        return None

    # ================= 辅助私有方法 =================

    def _chat_with_retry(self, messages: List[Dict], **kwargs) -> Optional[Any]:
        """
        带重试机制的 LLM 调用
        :param messages: 消息列表
        :param kwargs: LLM 参数（如 temperature）
        """
        for i in range(self._max_llm_retries):
            try:
                response = self.llm_client.chat(messages=messages, **kwargs)
                if response and response.content: return response
            except Exception as e:
                logger.warning(f"LLM call failed: {e}. Retry {i+1}...")
            time.sleep(2 ** i) # 指数退避重试
        return None

    def _load_sandbox_task_ids(self, path: str) -> List[str]:
        """从 JSON 文件加载可用的沙箱任务 ID"""
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
        """检查轨迹中是否存在对指定 API 的成功调用"""
        if not trajectory or not trajectory.steps: return False
        for step in trajectory.steps:
            if step.role == "tool" and not step.error:
                if api_name in step.tool_name: return True
        return False

    def _check_app_usage(self, trajectory: Trajectory, app_name: str) -> bool:
        """检查轨迹中是否存在对指定 App 下任意 API 的调用"""
        if not trajectory or not trajectory.steps: return False
        app_apis = self.api_knowledge.get(app_name, {}).get("apis", {}).keys()
        for step in trajectory.steps:
            if step.role == "tool":
                # 匹配 app 名称或 app 下的任一 API 名称
                if app_name.lower() in step.tool_name.lower(): return True
                for api in app_apis:
                    if api in step.tool_name: return True
        return False

    def _extract_tool_trace(self, trajectory: Trajectory) -> str:
        """从轨迹中提取精简的‘动作-观察’序列，方便 LLM 理解"""
        trace = []
        for step in trajectory.steps:
            if step.role == "assistant" and step.tool_calls:
                for tc in step.tool_calls:
                    trace.append(f"Action: {tc['name']} args={tc['arguments']}")
            elif step.role == "tool":
                content = str(step.content)
                # 截断过长的观察内容，避免超出 Token 限制
                if len(content) > 200: content = content[:200] + "..."
                trace.append(f"Observation: {content}")
        return "\n".join(trace)

    def _load_json(self, path: str) -> Dict:
        """通用的 JSON 文件加载方法"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {}

    def _save_intra_memory(self, app_name: str):
        """保存已成功探索的单域 App 列表到磁盘"""
        os.makedirs(os.path.dirname(self.intra_memory_path), exist_ok=True)
        current_data = self._load_json(self.intra_memory_path)
        current_apps = set(current_data.get("explored_apps", []))
        current_apps.add(app_name)
        with open(self.intra_memory_path, 'w', encoding='utf-8') as f:
            json.dump({"explored_apps": list(current_apps)}, f, indent=2)

    def _save_cross_memory(self, metadata: Dict):
        """保存成功合成的跨域任务日志"""
        os.makedirs(os.path.dirname(self.cross_memory_path), exist_ok=True)
        current_data = self._load_json(self.cross_memory_path)
        if "logs" not in current_data: current_data["logs"] = []
        current_data["logs"].append(metadata)
        with open(self.cross_memory_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2)