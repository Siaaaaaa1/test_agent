# 阶段三：跨域合成 Prompt

PURPOSE_SYNTHESIS_PROMPT = """
You are a Scenario Designer for AI Agents.
We need to construct a "Complex Cross-Domain Task" involving two specific Apps.

### App Roles
1. **Source App (Info Provider):** {info_app_name}
   - Description: {info_app_desc}
   - This app contains unstructured information (emails, notes, messages).
2. **Target App (Executor):** {exec_app_name}
   - Description: {exec_app_desc}
   - Available Action APIs: {exec_api_list}

### Task Requirements
1. **Context Creation:** Create a realistic piece of content for the Source App (e.g., a specific note text or email body) that contains necessary details (time, location, item, price, etc.) for the Target App's action.
2. **User Query:** Create a user instruction that forces the agent to FIRST read the Source App to find information, and THEN perform an action in the Target App using that information.

### Output Format (JSON)
Please return a valid JSON object with exactly these keys:
{{
    "setup_context": "The specific content to be injected into the Source App (e.g., 'Note: Buy milk at 5pm').",
    "user_query": "The instruction for the agent (e.g., 'Check my notes for what to buy and add it to my shopping list').",
    "target_action_api": "The specific API name from Target App that represents success."
}}
"""