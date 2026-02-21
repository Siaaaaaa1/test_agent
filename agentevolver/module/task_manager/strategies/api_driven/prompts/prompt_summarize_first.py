import re
import json
from typing import Sequence, Tuple, Optional
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from agentevolver.module.task_manager.env_profiles import EnvProfile

# 系统提示词：角色是代码审计员
DIRECT_VERIFY_SYSTEM_PROMPT = """# ROLE
You are a **Code Execution Auditor & Refiner**.
Your task is to evaluate whether an AI agent's execution trace successfully satisfies a specific **User Query**, and if so, extract the clean, executable Python code (Ground Truth) that achieves this.

# OBJECTIVES
1. **Verification**: Determine if the `action_sequence` in the trace actually solves the `user_query`.
   - If the agent failed, hallucinated, or didn't finish the task, mark as Invalid.
   - If the logic contradicts the query (e.g., query asks for "cheapest" but agent picked "first"), mark as Invalid.
2. **Extraction & Refinement**:
   - Extract the valid API calls and logic.
   - **CRITICAL**: Remove failed attempts, syntax errors, or redundant steps (e.g., checking the same thing twice unnecessarily, or logging in twice).
   - Ensure the code is complete (includes imports/setup if shown in trace) and executable.

# INPUT FORMAT
- **User Query**: The original request.
- **Execution Trace**: The step-by-step actions and observations from the agent.

# OUTPUT FORMAT
Return a single JSON block:
{
  "is_valid": true/false,
  "confidence": [0.0 - 1.0],
  "reason": "Short explanation of why it is valid or invalid",
  "refined_code": "The cleaned-up python code sequence. If invalid, leave empty."
}

# GOOD EXAMPLES

## Example 1: Valid & Needs Refinement (Removing Errors)
[User Query]
"Play the song 'Shape of You' on Spotify."
[Trace]
Step 0: apis.spotify.play(song='Shape of You') -> Error: User not logged in.
Step 1: apis.spotify.login(username='user', password='123') -> Success.
Step 2: apis.spotify.play(song='Shape of You') -> Success: Playing.
[Output]
{
  "is_valid": true,
  "confidence": 1.0,
  "reason": "The agent successfully played the song after logging in. The failed attempt in Step 0 is removed.",
  "refined_code": "apis.spotify.login(username='user', password='123')\\napis.spotify.play(song='Shape of You')"
}

## Example 2: Invalid (Incomplete Action)
[User Query]
"Buy the cheapest stapler."
[Trace]
Step 0: items = apis.amazon.search('stapler') -> Found 10 items.
Step 1: sorted_items = sorted(items, key=lambda x: x['price'])
Step 2: print(sorted_items[0]) -> Printed item details.
(End of Trace)
[Output]
{
  "is_valid": false,
  "confidence": 0.0,
  "reason": "The agent identified the cheapest item but never performed the 'buy' or 'order' action required by the query.",
  "refined_code": ""
}

## Example 3: Valid (Clean Logic)
[User Query]
"Count how many emails I received from 'Boss'."
[Trace]
Step 0: emails = apis.gmail.list_messages()
Step 1: count = 0
Step 2: for e in emails: if e['sender'] == 'Boss': count += 1
Step 3: print(count)
[Output]
{
  "is_valid": true,
  "confidence": 1.0,
  "reason": "The logic correctly iterates and counts messages matching the sender.",
  "refined_code": "emails = apis.gmail.list_messages()\\ncount = 0\\nfor e in emails:\\n    if e['sender'] == 'Boss':\\n        count += 1\\nprint(count)"
}
"""

def _get_action_observation_pair(traj: Trajectory) -> list[tuple[str, str]]:
    """提取动作和观察（复用逻辑）"""
    res = []
    for idx, step in enumerate(traj.steps):
        role = step.get("role", "unknown")
        if role == "assistant" and idx + 1 < len(traj.steps):
            next_step = traj.steps[idx + 1]
            next_role = next_step.get("role", "unknown")
            if next_role in ["tool", "user"]:
                raw_content = next_step.get("content", "")
                # 截断过长的 Observation 以节省 Token，保留 Action 代码
                if len(raw_content) > 1000:
                    observation = raw_content[:1000] + "...[truncated]"
                else:
                    observation = raw_content
                res.append((step.get("content", ""), observation))
    return res

def get_direct_verify_prompt(
    task: Task,
    trajectory: Trajectory,
    profile: EnvProfile | None,
) -> tuple[str, str]:
    """
    构造验证 Prompt
    """
    trace_str = ""
    pairs = _get_action_observation_pair(trajectory)
    
    for i, (action, obs) in enumerate(pairs):
        trace_str += f">>> STEP {i} <<<\n[Code]\n{action}\n[Output]\n{obs}\n\n"

    # [修改说明] 移除了关于 Outcome < 0.7 的特定判断指令，完全依赖 System Prompt 的逻辑判断
    user_prompt = f"""
# Analyze This Execution

## User Query
"{task.query}"

## Execution Trace
{trace_str}

# Instructions
1. Check if the trace logic strictly matches the User Query.
2. Generate the `refined_code` by cleaning up the trace (remove errors/redundancy).

Produce the JSON response now.
"""
    return DIRECT_VERIFY_SYSTEM_PROMPT, user_prompt

def parse_direct_verification(response: str) -> dict:
    """
    解析验证结果
    """
    try:
        # 尝试提取 JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return data
    except Exception as e:
        print(f"Error parsing verification response: {e}")
    
    return {"is_valid": False, "reason": "Parse Error", "refined_code": ""}