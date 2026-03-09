import copy
import random
import time
import json
import os
import uuid
import logging
import threading
from typing import Callable, Optional, Sequence, List, Any, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from tqdm import tqdm

# 内部模块引入
from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.module.env_manager.env_worker import EnvWorker
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig
from agentevolver.module.task_manager.agent_flow import ModifiedAgentFlow
from agentevolver.module.task_manager.base import LlmClient
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from agentevolver.client.llm_client import Mix_DashScopeClient
from typing import List
# 奖励计算与总结相关
from agentevolver.module.task_manager.rewards.integrated_reward_gt import IntegratedRewardCalculator_GT
from agentevolver.module.task_manager.strategies.common.prompts.prompt_extract_refsol import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)

from . import TaskPostFilter

# 设置标准 logging
std_logger = logging.getLogger(__name__)

class LlmFilter(TaskPostFilter):
    def __init__(self, env_url: str, llm_client: LlmClient, num_threads: int, *, tokenizer, config):
        """
        初始化 LlmFilter。强制重写为 Mix_DashScopeClient 以支持多模型回退。
        """
        self._env_client = EnvClient(env_url)
        self._num_threads = num_threads
        self._tokenizer = tokenizer
        self._config = config
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()

        # 1. 强制使用 Mix 客户端以驱动模型轮训
        try:
            logger.info("🛡️ [LlmFilter] Forcing Mix_DashScopeClient for tier polling.")
            self._llm_client = Mix_DashScopeClient(
                model_name="HY-qwen3.5-plus", 
                temperature=config.get("exploration_llm_temperature", 0.7)
            )
        except Exception as e:
            logger.error(f"Failed to init MixClient: {e}. Using passed client.")
            self._llm_client = llm_client

        # 2. 配置输出路径
        self.output_dir = os.environ.get("GEN_OUTPUT_DIR", "./data/output/filter_logs")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        self.success_file = os.path.join(self.output_dir, "post_filter_success.jsonl")
        self.rejected_file = os.path.join(self.output_dir, "post_filter_rejected.jsonl")

    def _save_to_log(self, item: TaskObjective, is_success: bool, reason: str = ""):
        """线程安全的实时追加写入"""
        target_file = self.success_file if is_success else self.rejected_file
        with self._io_lock:
            try:
                data = item.dict()
                data["filter_info"] = {
                    "is_success": is_success,
                    "reason": reason,
                    "timestamp": time.time()
                }
                with open(target_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Failed to save log: {e}")

    def filter(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        if not tasks:
            return []
        return self._filter_with_threadpool(tasks)

    def _filter_with_threadpool(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        """使用线程池并行处理，并实现实时保存"""
        from tqdm import tqdm
        res = []
        
        progress = tqdm(total=len(tasks), desc="🚀 [Post-Filter] Tier Polling & Sorting")
        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            future_to_task = {
                executor.submit(self._execute_strategy1, task): task 
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                original_task = future_to_task[future]
                progress.update(1)
                try:
                    t = future.result()
                    if t is not None:
                        res.append(t)
                        self._save_to_log(t, is_success=True)
                    else:
                        self._save_to_log(original_task, is_success=False, reason="Rejected: All model tiers failed.")
                except Exception as e:
                    logger.exception(f"Error processing task {original_task.task.task_id}: {e}")
                    self._save_to_log(original_task, is_success=False, reason=f"Exception: {str(e)}")
                    continue
        progress.close()
        return res

    def _execute_strategy1(self, task: TaskObjective) -> TaskObjective | None:
        """模型轮训核心逻辑：DeepSeek -> Qwen -> GPT-5-mini -> GPT-5"""
        # "qwen3-max-mini",
        # "qwen3-max"
        candidate_models = [
            "DeepSeek-V3-Online",
            "HY-qwen3.5-plus"
        ]
        random.shuffle(candidate_models)
        for model_name in candidate_models:
            try:
                logger.info(f"🔄 [Filter] Task {task.task.task_id} -> Trying {model_name}")
                
                # 采样参数，驱动 MixClient 路由
                sampling_params = {"model": model_name}
                if model_name in ["DeepSeek-V3-Online", "HY-qwen3.5-plus"]:
                    sampling_params.update({
                        "temperature": self._config.get("exploration_llm_temperature", 0.5),
                        "top_p": 0.9,
                        "top_k": 50
                    })

                # 1. 实例化 GT 奖励类
                reward_calculator = IntegratedRewardCalculator_GT(task=task.task)

                # 2. 构造环境与 Agent 流程
                worker = EnvWorker(
                    task.task,
                    config=self._config,
                    thread_index=0,
                    tokenizer=self._tokenizer
                )
                
                agent_flow = ModifiedAgentFlow(
                    enable_context_generator=False,
                    llm_chat_fn=self._get_llm_chat_fn(sampling_params=sampling_params),
                    tokenizer=self._tokenizer,
                    config=self._config,
                    reward_calculator=reward_calculator 
                )
                agent_flow._reward_calculator = reward_calculator

                # 3. 校验数据
                assert task.task.query is not None, "Task query is missing"

                # 4. 执行
                traj = worker.execute(
                    data_id=task.task.task_id,
                    rollout_id=f"filter_{model_name}",
                    traj_exp_config=TrajExpConfig(add_exp=False),
                    agent_flow=agent_flow,
                    tmux={'step': [0], 'token': [0]},
                    stop=[False],
                    system_prompt=make_solver_tip_prompt(task.task.query, getattr(task.task, 'ground_truth', ""))
                )

                # 5. 基于 outcome 判定 (>= 0.7)
                if traj and traj.reward:
                    score = traj.reward.outcome
                    if score >= 0.7:
                        logger.info(f"✅ [Filter] Task {task.task.task_id} PASSED with {model_name} (Score: {score})")
                        # 记录使用的模型并重写 GT
                        task.task.metadata = task.task.metadata or {}
                        task.task.metadata["filter_model"] = model_name
                        return self._rewrite_new_gt(task, traj)
                
                logger.warning(f"⏭️ [Filter] {model_name} failed for {task.task.task_id}. Falling back...")

            except Exception as e:
                logger.error(f"❌ [Filter] Exception with {model_name} on task {task.task.task_id}: {e}")
                continue

        return None

    def _validate(self, task: TaskObjective, trajectory: Trajectory) -> bool:
        """保留方法兼容性"""
        try:
            validator = TrajectoryEvaluator(self._llm_client)
            return validator.evaluate_trajectory_success(task, trajectory)
        except:
            return False

    def _rewrite_new_gt(self, task: TaskObjective, trajectory: Trajectory) -> TaskObjective:
        """任务成功后重新总结 Ground Truth"""
        sys_prompt, user_prompt = get_task_summarize_prompt([trajectory], [task], None)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        
        try:
            llm_output = self._get_llm_chat_fn()(messages)["content"]
            gt = parse_tasks_from_response(llm_output)
            if gt is not None:
                # ✅ 直接使用 Schema 中预留的 origin_ground_truth 字段来备份
                # 如果当前任务尚未设置过 origin_ground_truth，就把被覆盖前的 ground_truth 存进去
                if task.task.origin_ground_truth is None:
                    task.task.origin_ground_truth = task.task.ground_truth
                
                # ✅ 更新为新的 GT
                task.ground_truth = gt 
                # (注意: task.ground_truth 会触发 setter，自动修改 task.task.ground_truth，不需要写两遍)
                
        except Exception as e:
            logger.error(f"Failed to rewrite GT: {e}")
        
        return task
        
    def _get_llm_chat_fn(self, sampling_params: Optional[dict] = None) -> Callable:
        def llm_chat(
            messages: list[dict[str, str]],
            custom_sampling_params: Optional[dict] = None,
            request_id: Optional[str] = None,
        ) -> dict:
            updated_params = copy.deepcopy(sampling_params) if sampling_params else {}
            if custom_sampling_params:
                updated_params.update(custom_sampling_params)

            res = None
            for i in range(3):
                try:
                    res = self._llm_client.chat(messages=messages, sampling_params=updated_params)
                    if res: break
                except Exception as e:
                    time.sleep(2 ** i)

            return {"role": "assistant", "content": res if res else "Error generating response"}
        return llm_chat

class TrajectoryEvaluator:
    """Evaluate trajectory success using LLM"""
    
    def __init__(self, client: LlmClient):
        self.client = client
        self.prompts = EvaluationPrompts()

    def evaluate_trajectory_success(self, task: TaskObjective, trajectory: Trajectory) -> bool:
        """Evaluate if trajectory completed the task successfully"""
        try:
            # Create trajectory summary
            trajectory_summary = self._create_trajectory_summary(trajectory)
            
            final_observation: str | None = None
            for step in reversed(trajectory.steps):
                if final_observation is None and step['role'] != 'assistant':
                    final_observation = step['content']
                    break
                
            # Generate evaluation prompt
            assert task.objective is not None, "synthetic task must have objective"
            prompt = self.prompts.success_evaluation_prompt(
                query=task.objective,
                trajectory_summary=trajectory_summary,
                final_observation=final_observation or "[no observation]"
            )
            
            # Get LLM evaluation
            response = self.client.chat(prompt, sampling_params={})
            
            # Parse evaluation result
            success = self._parse_evaluation_response(response)
            
            logger.debug(f"Trajectory evaluation result: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to evaluate trajectory: {e}")
            return False
    
    def _create_trajectory_summary(self, traj: Trajectory) -> str:
        """
        Creates a summary of the steps in a given trajectory, with each step's content truncated to 200 characters.

        Args:
            traj (Trajectory): The trajectory object containing the steps to be summarized.

        Returns:
            str: A string summary of the trajectory steps.
        """
        summary_blocks = []
        
        for i, step in enumerate(traj.steps):
            block = f"(Step {i + 1}) {step['role']}:\n"
            block += f"{step['content'][:200]}...\n"  # ⭐ Truncate the content to 200 characters and append an ellipsis
            summary_blocks.append(block)
        
        return "\n".join(summary_blocks)
    
    def _parse_evaluation_response(self, response: str) -> bool:
        """
        Parses the evaluation response from an LLM to determine if the task was successful or not.

        Args:
            response (str): The response from the LLM.

        Returns:
            bool: True if the task is deemed successful, False otherwise.
        """
        if not response:
            return False
        
        response_lower = response.lower()
        
        # Look for explicit success/failure indicators
        if 'success: true' in response_lower or 'successful: true' in response_lower:
            return True
        elif 'success: false' in response_lower or 'successful: false' in response_lower:
            return False
        
        # Look for keywords
        success_keywords = ['success', 'completed', 'achieved', 'accomplished', 'solved']
        failure_keywords = ['failed', 'incomplete', 'unsuccessful', 'not completed', 'not achieved']
        
        success_count = sum(1 for keyword in success_keywords if keyword in response_lower)
        failure_count = sum(1 for keyword in failure_keywords if keyword in response_lower)
        
        # Default to success if more success keywords found
        return success_count > failure_count  # ⭐ Determine success based on the count of success and failure keywords
    
    

class EvaluationPrompts:
    """Prompt templates for trajectory evaluation"""
    
    def success_evaluation_prompt(self, query: str, trajectory_summary: str,
                                final_observation: str) -> list:
        """
        Generates a prompt for evaluating the success of a trajectory.

        Args:
            query (str): The task query.
            trajectory_summary (str): The summary of the trajectory.
            final_observation (str): The final observation of the trajectory.

        Returns:
            list: A list containing a single dictionary with the role and content for the prompt.
        """

# - Expected Outcome (Ground Truth API Call or Result): {ground_truth}
        messages = [
            {
            "role": "user",
            "content": f"""You are a strict task evaluation expert. Your goal is to determine whether the following multi-step agent trajectory successfully completed the assigned task.

    # Task Details
    - Query: {query}

    # Execution Summary
    - Trajectory Summary:
    {trajectory_summary}

    - Final Observation: {final_observation}

    # Evaluation Instructions

    Carefully analyze the trajectory to determine if the task was truly completed. Specifically, consider the following aspects:

    1. **API Matching**: Did the agent correctly call the required APIs according to the task requirements?
    2. **Parameter Usage**: Were the parameters used in API calls correct and sufficient?
    3. **Logical Flow**: Was the sequence of steps logical without unreasonable skips?
    4. **Final Result**: Did the final state achieve the expected outcome, reasonably solve the task, obtain all necessary information, and complete the task objectives?
    5. **Failed or Skipped Steps**: Were there any critical errors, skipped steps, or invalid code that prevented the task from being actually executed?

    # Format Your Response Strictly As:

    Success: [true/false]
    Reason: [Concise and specific explanation, referring to the above criteria.]

    Note: Do NOT mark the task as successful if the correct API was never called, the parameters were incorrect, or the result was not achieved, even if the intent seemed right.
    """
            }
        ]
        
        return messages



def make_solver_tip_prompt(query: str, gt: str):
    return f"""You are an AI assistant helping to complete tasks in an interactive environment.
Feel free to use the tips to help you complete the task. The tips include a potential step-by-step solution to the task, but I do not ensure it is correct.

Your Query:
{query}

Tips:
{gt}
"""