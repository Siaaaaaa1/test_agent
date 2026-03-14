import json
from typing import Optional, Sequence, Tuple
import re
from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory

# ================= 修改重点 =================
# 1. 新增了一个 "Complex Loop & Logic" (复杂循环与逻辑) 的例子。
# 2. 该例子展示了如何处理：登录 -> 翻页获取列表 -> 创建容器 -> 循环子项 -> 比较逻辑(Max) -> 写入。
# 3. 对应的 Query 使用了自然语言描述 ("My Most Played Album Songs")。
# ===========================================

AGENT_SUMMARIZE_SYSTEM_PROMPT = """# ROLE
You are a **Reverse Engineering Expert for API Interactions**.
Your job is to analyze an agent's code execution trace and precisely reconstruct the **User Intent** that would necessitate *exactly* these API calls and logic.

---

# OBJECTIVES
1. **Trace-to-Intent Mapping**: The generated `query` must be solvable **strictly** using the logic shown in the `action_sequence`.
2. **Logic Consistency**:
   - If the code filters by "price < 20", the query MUST include "under $20".
   - If the code loops until a specific song is found, the query MUST imply "find X".
   - If the code *only* checks a value but doesn't change it, the query CANNOT be "change X".
3. **Natural Abstraction**: While the logic must be exact, the *referencing* must be human (use attributes, not IDs).

---

# CRITICAL RULES (STRICT ADHERENCE REQUIRED)

## 1. Natural Referencing (No IDs)
- **NEVER** use database IDs, UUIDs, or internal codes in the `query`.
- **USE** descriptive attributes found in the logic (e.g., "the cheapest one", "emails from Mom", "my amazon cart").
- **Example**:
  - Code: `if item['id'] == 'B08X': buy()` -> Query: "Buy the item I was looking at" (Too vague) -> "Buy the specific item ID B08X" (BANNED) -> **Correct**: "Buy the specific item I selected" or describe its attributes if visible in context.

## 2. Scope & Capability Match
- The query must NOT ask for more than what was performed.
- **Example**:
  - History: Agent searches for "stapler" and adds the *first one* to cart.
  - ❌ Query: "Buy the best rated stapler." (The agent didn't sort by rating).
  - ✅ Query: "Search for a stapler and buy the first one you see."

## 3. Criteria Over Constants
- Users give **constraints**; the Agent finds the **values**.
- **Example**:
  - History: `msgs = list_messages(); target = [m for m in msgs if m['sender'] == 'Boss']`
  - ❌ Query: "Read the message with ID 999."
  - ✅ Query: "Read the last message from my Boss."

## 4. No Spoilers / Information Asymmetry
- Do NOT include the *result* of a query in the *question*.
- **Example**:
  - History: Agent checks weather -> Returns "Rainy".
  - ❌ Query: "Confirm that it is rainy."
  - ✅ Query: "Check the weather."

## 5. API Fidelity & Completeness (Code Construction)
- **The code must match the actual APIs used in the provided trace.** Do not invent new methods or properties.
- **Do not hallucinate helper functions that don't exist.** Use raw Python logic (loops, list comprehensions) and the raw API calls visible in the trace.
- **Include every single API call required (login, search, filter, iterate, action).** Do not skip setup steps, pagination, or authentication calls if they were present in the trace.

---

# OUTPUT FORMAT
For each identified task, output exactly one block in this format:

<task>
{
  "query": "[A natural user request that strictly matches the API logic]",
  "confidence": [0.0 - 1.0],
  "action_sequence": "[The COMPLETE, EXECUTABLE sequence of Python API calls derived from the history. The code must match the actual APIs used in the provided trace. Include every single API call required (login, search, filter, iterate, action) without hallucinating helper functions.]"
}
</task>

---

# GOOD EXAMPLES (Logic <-> Query Match)

<task>
{
  "query": "Make me a Spotify playlist called 'My Most Played Album Songs' containing only the most-played song from each album in my library.",
  "confidence": 0.85,
  "action_sequence": "# step0: Login and setup\\nprofile = apis.supervisor.show_profile()\\npasswords = {p['account_name']: p['password'] for p in apis.supervisor.show_account_passwords()}\\ntoken = apis.spotify.login(username=profile['email'], password=passwords['spotify'])['access_token']\\n\\n# step1: Get all albums (paginated)\\nalbums = [item for i in range(10) for item in apis.spotify.show_album_library(page_index=i, access_token=token)]\\n\\n# step2: Create target playlist\\nnew_playlist = apis.spotify.create_playlist(access_token=token, title='My Most Played Album Songs')\\n\\n# step3: Process albums and find best songs\\nfor album in albums:\\n    songs = apis.spotify.show_album(album_id=album['album_id'])['songs']\\n    # Find song with max play_count\\n    best_song = max(songs, key=lambda s: apis.spotify.show_song(song_id=s['id'])['play_count'])\\n    apis.spotify.add_song_to_playlist(access_token=token, playlist_id=new_playlist['playlist_id'], song_id=best_song['id'])\\n\\napis.supervisor.complete_task(status='success')"
}
</task>

<task>
{
  "query": "Place an order for all staplers currently in my amazon cart.",
  "confidence": 1.0,
  "action_sequence": "# step0\\ncart = apis.amazon.get_cart()\\n# step1\\n[logic iterating cart and ordering items where name contains 'stapler']..."
}
</task>

<task>
{
  "query": "How much have I paid in phone bills on Venmo this year?",
  "confidence": 1.0,
  "action_sequence": "# step0\\ntransactions = apis.venmo.list_transactions(year=2023)\\n# step1\\n[logic filtering note='phone bill' and summing amounts]..."
}
</task>

<task>
{
  "query": "Find the cheapest insect repellent that arrives within 5 days.",
  "confidence": 1.0,
  "action_sequence": "# step0\\nitems = apis.amazon.search('insect repellent')\\n# step1\\n[logic filtering delivery_days<=5]\\n# step2\\n[logic sorting by price]"
}
</task>

<task>
{
  "query": "Reply 'A gentle reminder' to everyone who hasn't replied in 96 hours.",
  "confidence": 0.95,
  "action_sequence": "# step0\\nthreads = apis.gmail.list_threads()\\n# step1\\n[logic checking last_reply_time > 96h]..."
}
</task>
"""


def _get_action_observation_pair(traj: Trajectory) -> list[tuple[str, str]]:
    """
    从轨迹中提取 (动作, 观察) 对。
    """
    res = []
    for idx, step in enumerate(traj.steps):
        role = step.get("role", "unknown")
        # 找到 Agent 的动作 (assistant 角色)
        if role == "assistant" and idx + 1 < len(traj.steps):
            next_step = traj.steps[idx + 1]
            next_role = next_step.get("role", "unknown")
            
            # 找到对应的环境反馈 (tool 或 user 角色)
            if next_role in ["tool", "user"]:
                # [优化] 截断过长的 Observation
                # 即使截断了，Action 代码本身（参数、逻辑）才是推断意图的关键。
                raw_content = next_step.get("content", "")
                if len(raw_content) > 800:
                    observation = raw_content[:800] + "...[truncated]"
                else:
                    observation = raw_content
                res.append((step.get("content", ""), observation))
            else:
                continue
    return res


def get_task_summarize_prompt(
    trajectories: Sequence[Trajectory],
    old_objectives: str,
    profile: EnvProfile | None,
) -> tuple[str, str]:
    """
    构造发送给 LLM 的最终 Prompt。
    """
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
        if traj.reward is not None:
            outcome = getattr(traj.reward, 'outcome', 'N/A')
            desc = getattr(traj.reward, 'description', '')
            x += f"### Reward: {outcome}\n{desc}\n"
        idx += 1

    # 用户侧 Prompt：强调基于路径的反向工程
    user_prompt = f"""Please analyze the following agent interaction sequence and reconstruct the user task.

# Actual Interaction History
{x}

# Task Requirements
{profile.get_task_preference_instruction() if profile is not None else "Please follow the instructions to generate tasks."}

# Constraints for Task Abstraction (STRICT)
1. **Grounded in Action**: The generated query must be solvable **using ONLY the API calls and logic shown above**. Do not hallucinate requirements that the code didn't actually check (e.g., don't say "red shoes" if the code only searched "shoes").
2. **Natural Descriptions**: Use relative terms (e.g., "my last email", "the cheapest option") derived from the code's sorting/filtering logic.
3. **No Database IDs**: Do NOT include specific IDs like `thread_123` or `product_B00X`.
4. **No Spoilers**: Do not put the answer/result in the question.
5. **API Coverage**: The query should imply the need for the specific APIs used (e.g., if `apis.weather` and `apis.calendar` were both used, the query should likely involve both).

# Now Start

Identify the user intent that strictly aligns with this execution path. Output in the specified XML/JSON format.
"""
    return AGENT_SUMMARIZE_SYSTEM_PROMPT, user_prompt


def parse_tasks_from_response(seed_task: Task, response: str) -> list[TaskObjective]:
    """
    解析 LLM 的回复，将其转换为程序可用的 Task 对象。
    """
    tasks: list[TaskObjective] = []
    try:
        # 使用正则提取 <task> 标签内的内容
        task_matches = re.findall(r"<task>(.*?)</task>", response, re.DOTALL)

        # 兼容性处理：如果没有标签，尝试直接找最外层 JSON 对象
        # 使用贪婪匹配 \{.*\} 而非 \{.*?\}，避免 action_sequence 中的 {} 导致提前截断
        if not task_matches:
            task_matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)

        for task_content in task_matches:
            try:
                # 清理 Markdown 代码块符号
                clean_content = task_content.replace("```json", "").replace("```", "").strip()
                t = json.loads(clean_content)
            except json.JSONDecodeError:
                continue

            if (
                "query" not in t
                or "action_sequence" not in t
            ):
                continue
            
            # 复制种子任务，确保新任务是独立的
            current_task = seed_task.copy()
            # 设置新生成的 Query
            current_task.query = t["query"]
            # 标记为开放式查询
            current_task.open_query = True
            
            conf = t.get("confidence", 1.0)
            try:
                conf = float(conf)
            except:
                conf = 1.0

            x = TaskObjective(
                task=current_task,
                confidence=conf,
                reward=None,
            )
            # 将 LLM 总结出的代码序列作为 Ground Truth
            x.ground_truth = t["action_sequence"]
            tasks.append(x)

    except Exception as e:
        print(f"Error parsing tasks: {e}")

    return tasks