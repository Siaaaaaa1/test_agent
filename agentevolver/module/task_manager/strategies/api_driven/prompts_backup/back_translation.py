# 阶段二：反向归纳 Prompt

BACK_TRANSLATION_PROMPT = """
You are a Data Annotator. You are observing a sequence of actions performed by an AI agent.
Your task is to infer the original User Query that would trigger this exact sequence of actions.

### Agent Execution Trace
{tool_calls_trace}

### Target Action Achieved
The agent successfully executed the function: "{target_api_name}".

### Instruction
Generate a natural language User Query (Prompt) that implies this intent.
The query should be specific enough to require the steps taken (e.g., "Find the user named Alice and send her 50 dollars") rather than generic (e.g., "Use the app").

**Output Format:**
Just provide the User Query string. Do not output any other text.
"""