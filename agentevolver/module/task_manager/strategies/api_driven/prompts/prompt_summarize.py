import json
from typing import Optional, Sequence, Tuple
import re
from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory


AGENT_SUMMARIZE_SYSTEM_PROMPT = """# ROLE
You are a **Real-World Task Discovery Expert**.  
Your job is to analyze an agent's API interaction history and transform it into **realistic, user-centered tasks** that could be solved using the same interaction **patterns**.

---

# OBJECTIVES
1. **Understand Capabilities**  
   - Analyze the recorded API calls to identify the actual functional capabilities demonstrated.

2. **Think Like a Real Experienced User**  
   - Imagine practical, everyday problems where a real person would naturally use this exact API call sequence (minus the documentation exploration).
   - Create problems that use **multiple different API calls**, not just a single call.
   - Use **clear, specific, verifiable** user requests.

3. **Abstract into Three Elements**  
   For each realistic task, provide:
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

---

# GOOD EXAMPLES
<task>
{
  "query": "Do I have at least $150 in my Venmo account for this weekend's grocery shopping?",
  "confidence": 1.0,
  "action_sequence": "# step0\nbalance = apis.venmo.get_balance()\nif balance >= 150:\n    print('Yes')\nelse:\n    print('No')"
}
</task>

<task>
{
  "query": "Find red women's heels under $100 that can be delivered by next Friday",
  "confidence": 1.0,
  "action_sequence": "# step0\n[click('https://www.taobao.com')]\n# step1\n[search('red women heels price<100 delivery:2025-08-22')]"
}
</task>

---

# CHECKLIST BEFORE FINALIZING
✅ **Clear goal** – What exactly is the user trying to achieve?  
✅ **Concrete details** – Who, what, when, where, how much/many?  
✅ **Verifiable** – Can success/failure be objectively determined?  
✅ **Human-first phrasing** – Sounds like something a real person would say.
"""


def _get_action_observation_pair(traj: Trajectory) -> list[tuple[str, str]]:
    res = []
    for idx, step in enumerate(traj.steps):
        assert "role" in step, "steps must have role field"
        if step["role"] == "assistant" and idx + 1 < len(traj.steps):
            next_step = traj.steps[idx + 1]
            # As there is no standard for environments, we do not know whether it will respond as user or tool.
            if next_step["role"] == "tool":
                # get observation from tool message
                observation = next_step["content"]
            elif next_step["role"] == "user":
                # get observation from user message
                observation = next_step["content"]
            else:
                continue
            res.append((step["content"], observation))

    return res


def get_task_summarize_prompt(
    trajectories: Sequence[Trajectory],
    old_objectives: str,
    profile: EnvProfile | None,
) -> tuple[str, str]:
    x = ""
    idx = 0
    for traj in trajectories:
        pairs = _get_action_observation_pair(traj)
        x += f"## Record {idx}\n"
        x += f"### History\n"
        for step_idx, history in enumerate(pairs):
            x+=f""">>> STEP {step_idx} <<<
<|ACTION|>
{history[0]}
<|END|>

<|OBSERVATION|>
{history[1]}
<|END|>
"""
            pass
        if traj.reward is not None:
            x += f"### Reward: {traj.reward.outcome}\n{traj.reward.description}\n"
        idx += 1

    user_prompt = f"""Please analyze the following agent interaction sequence and abstract specific tasks from it.

# Actual Interaction History
{x}

# Task Requirements

{profile.get_task_preference_instruction() if profile is not None else "Please follow the instructions to generate tasks."}

# Constraints for Task Abstraction
1. **Specific Grounding**: The task query MUST include **specific details** found in the environment history (e.g., exact filenames, specific contact names, exact dollar amounts, song titles). Do NOT use generic terms like "a file" or "someone".
2. **High Quality & Unique**: Summarize into **1 to 3** distinct, high-quality tasks. 
3. **No Repetition**: Do not generate multiple tasks that describe the exact same action sequence. If the agent performed one main workflow, output only the single best description for it.

# Now Start

Please identify the specific tasks the agent is attempting to complete in these interactions, and abstract them into clear task descriptions and queries following the specified format.
"""

# # Context (Original Intent)
# The agent was originally instructed to explore or solve the following objective:
# "{old_objectives}"

# Please use this objective as a reference to understand the user's initial intent. However, your generated task must be based on the **actual successful actions** taken in the history below. If the agent deviated from the plan but successfully performed a valid action sequence, summarize what it *actually* did.
    return AGENT_SUMMARIZE_SYSTEM_PROMPT, user_prompt


def parse_tasks_from_response(seed_task: Task, response: str) -> list[TaskObjective]:
    # 建议重命名参数为 seed_task 以区分
    tasks: list[TaskObjective] = []
    try:
        task_matches = re.findall(r"<task>(.*?)</task>", response, re.DOTALL)

        for task_content in task_matches:
            try:
                t = json.loads(task_content)
            except json.JSONDecodeError:
                continue

            if (
                "query" not in t
                or "confidence" not in t
                or "action_sequence" not in t
            ):
                continue
            
            # [修正]：在循环内部进行 copy，确保每个生成的任务是独立的实例
            current_task = seed_task.copy()
            
            # 修改新实例的属性
            current_task.query = t["query"]
            current_task.open_query = True
            
            x = TaskObjective(
                task=current_task,  # 传入新实例
                confidence=t["confidence"],
                reward=None,
            )
            x.ground_truth = t["action_sequence"]
            tasks.append(x)

    except Exception as e:
        print(f"Error parsing tasks: {e}")

    return tasks