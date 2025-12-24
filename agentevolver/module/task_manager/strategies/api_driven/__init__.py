import json
import os
import random
import copy
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

# 引入我们分离出去的 Prompt
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
    Implements:
    1. Intra-Domain Reverse Semantic Exploration
    2. Purpose-Driven Cross-Domain Synthesis
    """

    def __init__(self, tokenizer, config, **kwargs):
        super().__init__(tokenizer, config)
        self.tokenizer = tokenizer
        self.config = config
        
        # 配置路径与参数
        self.api_knowledge_path = kwargs.get("api_knowledge_path", "data/knowledge/api_quadruplets.json")
        self.strategy_memory_path = kwargs.get("strategy_memory_path", "data/memory/api_explore_strategy.json")
        self.active_apps = set(kwargs.get("active_apps", []))
        
        # 初始化内部组件占位符 (将在运行时注入或懒加载)
        self.agent_flow: Optional[AgentFlow] = None
        self.env_worker: Optional[EnvWorker] = None
        
        # 1. 加载 API 知识库
        self.api_knowledge = self._load_json(self.api_knowledge_path)
        if not self.api_knowledge:
            logger.warning(f"API Knowledge not found at {self.api_knowledge_path}. Exploration might fail.")

        # 2. 加载策略记忆 (已探索过的 APP)
        self.memory_data = self._load_json(self.strategy_memory_path) or {"explored_apps": []}
        self.explored_apps = set(self.memory_data.get("explored_apps", []))

        # 3. 模式判定逻辑
        self.new_apps = self.active_apps - self.explored_apps
        
        if not self.explored_apps and self.active_apps:
            self.mode = "cold_start"
            self.target_apps_pool = list(self.active_apps)
            logger.info(f"[ApiDriven] Mode: COLD START. Target Pool: {self.target_apps_pool}")
        elif self.new_apps:
            self.mode = "incremental"
            self.target_apps_pool = list(self.new_apps)
            logger.info(f"[ApiDriven] Mode: INCREMENTAL. Target Pool (New Apps): {self.target_apps_pool}")
        else:
            self.mode = "maintenance" # 或者 done，这里为了持续运行设为 maintenance
            self.target_apps_pool = list(self.active_apps)
            logger.info("[ApiDriven] Mode: MAINTENANCE. Re-exploring active apps.")

    def explore(self, task: Task, data_id: str, rollout_id: str) -> List[Trajectory]:
        """
        核心入口。
        注意：这里的 task 参数通常是 TaskManager 传进来的空壳或种子，
        但在 ApiDriven 模式下，我们会自己生成具体的 Instruction。
        """
        # 懒加载执行环境 (避免在 init 时过早连接)
        self._ensure_execution_capability()

        if not self.target_apps_pool:
            return []

        # 随机策略：50% 概率做单域探索，50% 概率做跨域合成 (如果条件允许)
        # 优先保证新 APP 的单域探索
        target_app = random.choice(self.target_apps_pool)
        
        trajectories = []
        
        # === 决策逻辑 ===
        should_do_cross_domain = (len(self.active_apps) > 1) and (random.random() > 0.5)
        
        if should_do_cross_domain:
            traj = self._run_cross_domain_synthesis(target_app)
        else:
            traj = self._run_intra_domain_exploration(target_app)

        if traj:
            # 标记该 APP 已探索 (简单处理，实际可能需要达到一定成功率才标记)
            if target_app not in self.explored_apps:
                self.explored_apps.add(target_app)
                self._save_memory()
            
            trajectories.append(traj)

        return trajectories

    def summarize(self, task: Task, trajectory: Trajectory) -> List[TaskObjective]:
        """
        将探索成功的 Trajectory 转化为微调数据。
        """
        if not trajectory or not trajectory.steps:
            return []
        
        # 只有当我们在 trajectory.info 中标记了成功，并且提取了真实的用户 query 时才生成
        user_query = trajectory.info.get("synthesized_user_query")
        if user_query:
            return [TaskObjective(input=user_query, output=trajectory)]
        return []

    # ================= 阶段二：单域逆向语义探索 =================

    def _run_intra_domain_exploration(self, app_name: str) -> Optional[Trajectory]:
        logger.info(f"[Intra-Domain] Exploring App: {app_name}")
        
        app_knowledge = self.api_knowledge.get(app_name, {})
        apis = app_knowledge.get("apis", {})
        if not apis:
            return None

        # 1. 锚定终态：筛选 Action API (非 GET 且有参数)
        action_apis = [
            k for k, v in apis.items() 
            if v.get("method", "").upper() != "GET" and len(v.get("parameters", [])) > 0
        ]
        if not action_apis:
            # 如果没有明显的 Action API，随机选一个非 list 的
            action_apis = list(apis.keys())
        
        target_api_name = random.choice(action_apis)
        target_api_def = apis[target_api_name]
        
        # 筛选 Info APIs 供 LLM 参考
        info_apis = {
            k: v for k, v in apis.items() 
            if k != target_api_name and (k.startswith("get") or k.startswith("list") or k.startswith("search"))
        }

        # 2. LLM 规划 (Plan Generation)
        prompt = PLAN_GENERATION_PROMPT.format(
            target_api_name=target_api_name,
            app_name=app_name,
            target_api_details=json.dumps(target_api_def, indent=2),
            available_info_apis=json.dumps(info_apis, indent=2)
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat(messages=messages, temperature=0.7)
        instruction = response.content.strip()
        logger.info(f"[Intra-Domain] Generated Plan: {instruction}")

        # 3. 执行探索 (Execution)
        # 构造一个用于执行的 Task 对象
        exec_task = Task(
            task_id=f"intra_{app_name}_{random.randint(1000,9999)}",
            instruction=instruction,
            metrics={"target_app": app_name}
        )
        
        trajectory = self._execute_agent_loop(exec_task)

        # 4. 验证与反向归纳 (Back-Translation)
        if self._check_api_called(trajectory, target_api_name):
            logger.info(f"[Intra-Domain] Success! Back-translating trajectory...")
            tool_trace = self._extract_tool_trace(trajectory)
            
            bt_prompt = BACK_TRANSLATION_PROMPT.format(
                tool_calls_trace=tool_trace,
                target_api_name=target_api_name
            )
            bt_response = self.llm_client.chat(messages=[{"role": "user", "content": bt_prompt}])
            user_query = bt_response.content.strip()
            
            # 将生成的 User Query 注入到 Trajectory info 中
            trajectory.info["synthesized_user_query"] = user_query
            trajectory.info["exploration_type"] = "intra_domain"
            return trajectory
        else:
            logger.info(f"[Intra-Domain] Failed to call target API: {target_api_name}")
            return None

    # ================= 阶段三：跨域合成 =================

    def _run_cross_domain_synthesis(self, primary_app: str) -> Optional[Trajectory]:
        # 1. 角色定义与配对
        # 尝试将 primary_app 归类
        is_info_provider = primary_app in UNIVERSAL_INFO_PROVIDERS
        
        info_app = ""
        exec_app = ""
        
        if is_info_provider:
            info_app = primary_app
            # 找一个非 info 的 app 做 executor
            candidates = [a for a in self.active_apps if a not in UNIVERSAL_INFO_PROVIDERS]
            if not candidates: return None
            exec_app = random.choice(candidates)
        else:
            exec_app = primary_app
            # 找一个 info app
            candidates = [a for a in self.active_apps if a in UNIVERSAL_INFO_PROVIDERS]
            if not candidates: return None
            info_app = random.choice(candidates)
            
        logger.info(f"[Cross-Domain] Pair: Info({info_app}) -> Exec({exec_app})")
        
        # 获取相关描述
        info_desc = self.api_knowledge.get(info_app, {}).get("description", "No description")
        exec_desc = self.api_knowledge.get(exec_app, {}).get("description", "No description")
        exec_apis = list(self.api_knowledge.get(exec_app, {}).get("apis", {}).keys())[:10] # 限制长度

        # 2. 目的合成 (Purpose Synthesis)
        prompt = PURPOSE_SYNTHESIS_PROMPT.format(
            info_app_name=info_app,
            info_app_desc=info_desc,
            exec_app_name=exec_app,
            exec_app_desc=exec_desc,
            exec_api_list=json.dumps(exec_apis)
        )
        
        response = self.llm_client.chat(messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        
        try:
            scenario_data = extract_json_from_str(response.content)
            setup_context = scenario_data.get("setup_context", "")
            user_query = scenario_data.get("user_query", "")
            target_action = scenario_data.get("target_action_api", "")
        except Exception as e:
            logger.error(f"[Cross-Domain] JSON Parse Error: {e}")
            return None

        logger.info(f"[Cross-Domain] Scenario: {user_query}")

        # 3. 环境预设 (Data Injection)
        # 注意：这里需要 EnvWorker 支持 setup 注入。
        # 我们假设 task.metrics 或 task.extra_config 可以传递给环境的 reset/setup
        exec_task = Task(
            task_id=f"cross_{info_app}_{exec_app}_{random.randint(1000,9999)}",
            instruction=user_query,
            metrics={
                "setup_action": "inject_data",
                "app": info_app,
                "content": setup_context
            }
        )

        # 4. 执行与验证
        trajectory = self._execute_agent_loop(exec_task)
        
        # 验证：既要看了 Info App，又要调了 Exec App
        called_info = self._check_app_usage(trajectory, info_app)
        called_exec = self._check_api_called(trajectory, target_action)
        
        if called_info and called_exec:
            logger.info("[Cross-Domain] Success!")
            trajectory.info["synthesized_user_query"] = user_query
            trajectory.info["exploration_type"] = "cross_domain"
            return trajectory
        
        return None

    # ================= 辅助方法 =================

    def _ensure_execution_capability(self):
        """
        初始化 AgentFlow 和 EnvWorker。
        这里我们实例化一个新的 Worker，不影响 TaskManager 主流程的 Worker。
        """
        if self.agent_flow is None:
            # 初始化 Agent Flow
            self.agent_flow = AgentFlow(self.config, self.tokenizer)
            
        if self.env_worker is None:
            # 初始化 Env Client 和 Worker
            # 假设 env_config 在 self.config.env_config 中
            from agentevolver.client.env_client import EnvClient
            
            # 使用 config 中的配置，但可能需要覆盖 task_id 等
            env_config = self.config.get("env_config", {})
            env_client = EnvClient(**env_config)
            self.env_worker = EnvWorker(env_client=env_client)
            # 注意：真实的 EnvWorker 可能需要 start 或 connect
            # self.env_worker.start() 

    def _execute_agent_loop(self, task: Task) -> Trajectory:
        """
        运行完整的 Agent Loop：Thought -> Act -> Obs
        """
        # 重置环境
        # 注意：这里我们利用 task 对象传参来控制环境初始化（如果支持）
        obs = self.env_worker.reset(task) 
        
        # 使用 AgentFlow 运行任务
        # 这里假设 agent_flow.run 返回的是 Trajectory 对象
        try:
            trajectory = self.agent_flow.run(task, self.env_worker)
            return trajectory
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return Trajectory(steps=[])

    def _check_api_called(self, trajectory: Trajectory, api_name: str) -> bool:
        """检查轨迹中是否成功调用了指定 API (无 Error)"""
        if not trajectory or not trajectory.steps:
            return False
        
        for step in trajectory.steps:
            if step.role == "tool" and not step.error:
                if api_name in step.tool_name:
                    return True
        return False

    def _check_app_usage(self, trajectory: Trajectory, app_name: str) -> bool:
        """检查轨迹中是否调用了属于某个 App 的任意 API"""
        if not trajectory or not trajectory.steps:
            return False
        # 假设 tool_name 格式为 app.function 或类似的
        # 或者我们遍历 api_knowledge 查找该 app 下的所有 api
        app_apis = self.api_knowledge.get(app_name, {}).get("apis", {}).keys()
        
        for step in trajectory.steps:
            if step.role == "tool":
                # 简单匹配：如果 tool_name 包含 app 名，或者在 app_apis 列表中
                if app_name.lower() in step.tool_name.lower():
                    return True
                for api in app_apis:
                    if api in step.tool_name:
                        return True
        return False

    def _extract_tool_trace(self, trajectory: Trajectory) -> str:
        """将轨迹转换为文本摘要"""
        trace = []
        for step in trajectory.steps:
            if step.role == "assistant" and step.tool_calls:
                for tc in step.tool_calls:
                    trace.append(f"Action: {tc['name']} args={tc['arguments']}")
            elif step.role == "tool":
                # 截断过长的输出
                content = str(step.content)
                if len(content) > 200:
                    content = content[:200] + "...[truncated]"
                trace.append(f"Observation: {content}")
        return "\n".join(trace)

    def _load_json(self, path: str) -> Dict:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_memory(self):
        os.makedirs(os.path.dirname(self.strategy_memory_path), exist_ok=True)
        data = {"explored_apps": list(self.explored_apps)}
        with open(self.strategy_memory_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)