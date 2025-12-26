# 阶段三：跨域合成 Prompt (Generic Exploration Version)

PURPOSE_SYNTHESIS_PROMPT = """
You are a Cross-Domain Task Scenario Generator for AI Agents.
Your goal is to create a **Generic and Exploratory** user query that connects two Apps based on their API capabilities.

**Context**:
We do NOT know the exact specific content in the environment (e.g., we don't know specifically if there is an email about "Dinner" or "Meeting").
Therefore, the User Query must be **broad and condition-based**, forcing the agent to explore and react to what it finds.

**The Setup**:
1. **Source App ({info_app_name})**: The agent must search/read here first.
   - Reference APIs: {info_apis_json}
2. **Target App ({exec_app_name})**: The agent must perform an action here using data found in the Source App.
   - Reference APIs: {exec_apis_json}
3. {system_tools_hint}

**Task Requirements**:
1. Create a `user_query` that directs the agent to:
   - "Check {info_app_name} for [general category, e.g., recent messages, starred notes, specific folder]."
   - "If found, extract [key information, e.g., dates, amounts, song names]."
   - "Then, use that information to perform an action in {exec_app_name}."
2. The query should NOT contain hardcoded values (e.g., do NOT say "Pay John $50"). Instead say "Pay the person mentioned in the note the amount listed".
3. Select one `target_action_api` from the Target App list that represents the final success.

**Output Format (JSON Only)**:
{{
    "user_query": "The natural language instruction.",
    "target_action_api": "The name of the core API in Target App."
}}
"""