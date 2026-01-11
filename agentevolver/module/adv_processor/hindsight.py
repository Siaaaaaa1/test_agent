import json
import os
import re
import threading
from typing import List, Dict, Any, Sequence
from loguru import logger

from agentevolver.schema.trajectory import Trajectory, Reward
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.utils.step_parser import parse_response_ids_to_steps

# --- 引入你提供的 Prompt 和解析函数 ---

AGENT_SUMMARIZE_SYSTEM_PROMPT = """# ROLE
You are a **Real-World Task Discovery Expert**.  
Your job is to analyze an agent's API interaction history and transform it into **realistic, user-centered tasks** that could be solved using the same interaction **patterns**.

---

# OBJECTIVES
1. **Understand Capabilities** - Analyze the recorded API calls to identify the actual functional capabilities demonstrated.

2. **Think Like a Real Experienced User** - Imagine practical, everyday problems where a real person would naturally use this exact API call sequence (minus the documentation exploration).
   - Create problems that use **multiple different API calls**, not just a single call.
   - Use **clear, specific, verifiable** user requests.

3. **Abstract into Three Elements** For each realistic task, provide:
   - **query**: A natural-language request that a real user might make.
   - **confidence**: A number between `0.0` and `1.0` representing how confident you are that this is a real, common need.
   - **action_sequence**: The sequence of technical steps that directly accomplishes the task.

---

# RULES FOR SCENARIO CREATION
## 1. Focus on User Intent
- Always start from a **human goal**.
- Avoid restating the API function in technical terms—capture the **why** behind the action.

## 2. Remove Non-Essential Steps
- Do **not** include:
  - Capability exploration or debugging steps.

## 3. Specificity & Verifiability
- The query must be **precise enough** that someone can clearly judge success/failure.
- Include **concrete details**:  
  - Numbers, dates, names, locations, thresholds, item lists, etc.
- Avoid vague words like “check”, “review”, or “ensure” unless paired with measurable criteria.

## 4. Practicality
- Use **relatable, everyday** scenarios.
- Avoid tasks that are purely exploratory or only serve to test an API.

---

# OUTPUT FORMAT
For each identified task, output exactly one block in this format:

<task>
{
  "query": "[A natural, specific, verifiable user request]",
  "confidence": [0.0 - 1.0],
  "action_sequence": "[Technical sequence that directly solves the task]"
}
</task>
"""

def _get_action_observation_pair(traj: Trajectory) -> list[tuple[str, str]]:
    res = []
    # 适配 Trajectory schema
    if not traj.steps:
        return res
        
    for idx, step in enumerate(traj.steps):
        if step.get("role") == "assistant" and idx + 1 < len(traj.steps):
            next_step = traj.steps[idx + 1]
            if next_step.get("role") in ["tool", "user"]:
                observation = next_step.get("content", "")
                res.append((step.get("content", ""), observation))
    return res

def get_task_summarize_prompt(
    trajectories: Sequence[Trajectory],
    env_profile: EnvProfile | None = None,
) -> tuple[str, str]:
    x = ""
    idx = 0
    for traj in trajectories:
        pairs = _get_action_observation_pair(traj)
        x += f"## Record {idx}\n"
        x += f"### History\n"
        for step_idx, history in enumerate(pairs):
            x += f""">>> STEP {step_idx} <<<
<|ACTION|>
{history[0]}
<|END|>

<|OBSERVATION|>
{history[1]}
<|END|>
"""
        idx += 1

    profile_instruction = env_profile.get_task_preference_instruction() if env_profile else "Please follow the instructions to generate tasks."

    user_prompt = f"""Please analyze the following agent interaction sequence and abstract specific tasks from it.

# Actual Interaction History
{x}

# Task Requirements
{profile_instruction}

# Constraints for Task Abstraction
1. **Specific Grounding**: The task query MUST include **specific details** found in the environment history.
2. **High Quality & Unique**: Summarize into **1 to 3** distinct, high-quality tasks. 
3. **No Repetition**: Do not generate multiple tasks that describe the exact same action sequence.

# Now Start
Please identify the specific tasks the agent is attempting to complete in these interactions.
"""
    return AGENT_SUMMARIZE_SYSTEM_PROMPT, user_prompt

def parse_tasks_from_response(seed_task_id: str, response: str) -> list[Dict]:
    tasks = []
    try:
        task_matches = re.findall(r"<task>(.*?)</task>", response, re.DOTALL)
        for task_content in task_matches:
            try:
                t = json.loads(task_content)
            except json.JSONDecodeError:
                continue

            if "query" not in t or "action_sequence" not in t:
                continue
            
            # 生成新的任务记录格式
            tasks.append({
                "task_id": f"hindsight_{seed_task_id}_{os.urandom(4).hex()}",
                "query": t["query"],
                "ground_truth": t["action_sequence"],
                "confidence": t.get("confidence", 0.0),
                "source": "hindsight"
            })
    except Exception as e:
        logger.warning(f"Error parsing hindsight tasks: {e}")
    return tasks

# --- Manager 类 ---

class HindsightManager:
    def __init__(self, llm_client, tokenizer, save_path: str = "tasks_explored/hindsight_supplement.jsonl"):
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.save_path = save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self._lock = threading.Lock()

    def process_failed_batch(self, 
                             prompts: List[List[int]], 
                             responses: List[List[int]], 
                             scores: List[float], 
                             task_ids: List[str],
                             threshold: float = 0.0):
        """
        处理一个批次的失败样本
        """
        failed_trajectories = []
        original_task_ids = []

        for i, score in enumerate(scores):
            # 筛选低分样本（失败路径）
            if score <= threshold:
                # 重构 Trajectory 对象
                traj = self._reconstruct_trajectory(prompts[i], responses[i], task_ids[i])
                if traj:
                    failed_trajectories.append(traj)
                    original_task_ids.append(task_ids[i])

        if not failed_trajectories:
            return

        logger.info(f"[Hindsight] Found {len(failed_trajectories)} failed trajectories. Starting reverse induction...")
        
        # 异步或同步调用 LLM (建议批量处理，这里简单起见逐个或小批处理)
        # 为了不阻塞训练主循环，建议这里只是放入队列或者启动后台线程
        # 这里演示同步调用的逻辑核心：
        
        for traj, original_tid in zip(failed_trajectories, original_task_ids):
            self._run_hindsight_single(traj, original_tid)

    def _reconstruct_trajectory(self, prompt_tokens: List[int], response_tokens: List[int], data_id: str) -> Trajectory:
        """利用 step_parser 将 token 序列还原为 Trajectory 对象"""
        try:
            # 1. 解析 Steps
            parse_result = parse_response_ids_to_steps(response_tokens, self.tokenizer)
            
            # 2. 构建 steps 列表 [{"role": ..., "content": ...}]
            steps = []
            for step_data in parse_result.steps:
                # Action (Assistant)
                if step_data['action_text'].strip():
                    steps.append({"role": "assistant", "content": step_data['action_text']})
                # Observation (User/Tool)
                if step_data['observation_text'].strip():
                    # 假定观察结果是 user 或 tool 角色，具体取决于你的环境定义
                    steps.append({"role": "user", "content": step_data['observation_text']})
            
            # 3. 解码 Prompt 获取原始 query (如果有需要)
            query_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)

            return Trajectory(
                data_id=data_id,
                query=query_text,
                steps=steps,
                reward=Reward(outcome=0.0) # 标记为失败
            )
        except Exception as e:
            logger.warning(f"[Hindsight] Failed to reconstruct trajectory: {e}")
            return None

    def _run_hindsight_single(self, traj: Trajectory, original_task_id: str):
        try:
            # 1. 构造 Prompt
            # 注意：如果需要 EnvProfile，需要从外部传入或者根据 task_id 查找
            sys_prompt, user_prompt = get_task_summarize_prompt([traj], env_profile=None)
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 2. 调用 LLM
            response = self.llm_client.chat_completion(messages=messages, temperature=0.7)
            
            # 3. 解析结果
            new_tasks = parse_tasks_from_response(original_task_id, response)
            
            if new_tasks:
                self._save_tasks(new_tasks)
                
        except Exception as e:
            logger.error(f"[Hindsight] LLM processing failed: {e}")

    def _save_tasks(self, tasks: List[Dict]):
        with self._lock:
            with open(self.save_path, "a", encoding="utf-8") as f:
                for t in tasks:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")