# 阶段二：单域逆向语义探索 Prompt

PLAN_GENERATION_PROMPT = """
You are an expert App Tester and Automation Engineer.
Your goal is to successfully call the target Action API: "{target_api_name}" in the App "{app_name}".

### Target API Definition
{target_api_details}

### Available Information APIs
You have access to the following 'read-only' APIs (get/list/search) in the same App. You should use them to find necessary parameters (e.g., IDs, names) for the target API.
{available_info_apis}

### Instruction
Please generate a step-by-step exploration plan.
1. Analyze the parameters required by "{target_api_name}".
2. Use your semantic intuition to decide which Information APIs might return these parameters.
3. Construct a natural language instruction that an agent can execute. The instruction must explicitly state which Information API to call first to get the data, and then use that data to call the Target API.

**Output Format:**
Just provide the actionable instruction string for the agent. Do not output any other text.
"""