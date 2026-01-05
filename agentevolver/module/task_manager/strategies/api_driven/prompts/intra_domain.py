import json
import re

# 阶段二：单域泛化引导 Prompt (Generic Task Generation)

# 阶段二：单域泛化引导 Prompt (Generic Task Generation)

INTRA_DOMAIN_PURPOSE_PROMPT = """
You are an expert data generator creating training data for an AI Agent in a "Black Box" environment.
Given a list of APIs for a single App, select one target API and construct a **logical, natural, and intent-clear** User Query based on it.

### Core Context
The user interacts with a **Black Box System**:
1.  The user **CANNOT** see the internal state (e.g., current volume level, specific inventory, existing filenames).
2.  The user relies on **common sense** or **fuzzy needs** to issue commands, expecting the Agent to handle the details.

### Task Goal
Generate a natural language instruction. This query must strike the **perfect balance** between "ambiguity" and "precision":
- Not too vague (leaving the Agent clueless);
- Not too specific (including parameters the user couldn't possibly know, leading to execution failure).

### Construction Guidelines

**1. Clear Intent & Valid Parameters (The Goldilocks Rule):**
   The query should contain key constraints required for the task (e.g., category, budget, purpose), but **STRICTLY AVOID** "gambling-style" specific values the user cannot confirm.
   - **Bad (Too Broad):** "Buy something." (Agent cannot execute)
   - **Bad (Too Specific/Gambling):** "Buy a fan on Amazon that costs exactly $20.50." (User cannot predict the exact price; high failure rate.)
   - **Good (Functional Constraint):** "Buy a highly-rated desktop fan on Amazon for under $25." (Delegates the filtering to the Agent, which is reasonable.)

**2. Blind Action & State-Agnostic:**
   State the desired outcome directly. Do NOT write conditional logic based on a "hypothetical current state."
   - **Bad (God View):** "If my account name is currently 'J. Doe' and I am logged in, change it to 'Jane'." (User cannot see these backend states.)
   - **Good (Direct Command):** "Update my account name to 'Jane Smith'." (Overwrite regardless of the previous value.)

**3. Attribute Filtering & Fuzzy Logic:**
   Use qualitative descriptors to simulate real human thinking.
   - **Examples:** "Play songs I liked recently", "Delete all spam messages", "Pick the highest-rated option".

### Typical High-Quality Logic (Few-Shot Logic)

**Type A: Acquisition with Soft Constraints**
*Scenario:* Shopping, Search
*Query:* "Find me an Italian restaurant with a rating above 4.5 that is closest to me."
*Logic:* User doesn't know the specific name but knows the filtering criteria.

**Type B: Batch Processing**
*Scenario:* Email, File Management
*Query:* "Move all PDF files from my 'Downloads' folder to the 'Documents' directory."
*Logic:* User doesn't list specific filenames but operates on a group via attributes (file type).

**Type C: Overwrite Settings**
*Scenario:* Personal Settings
*Query:* "Change my profile bio to 'Working hard'."
*Logic:* Execute directly without caring what the old bio was.

---

### Output Format
Output ONLY a raw JSON object. **Do NOT** use Markdown code blocks.
{{
    "user_query": "Generated natural language instruction",
    "target_api": "The primary API call_name (entry point) used to fulfill the request"
}}

### Example
App Name: Spotify
App APIs: [apis.spotify.search, apis.spotify.play, apis.spotify.add_to_queue]

Output JSON:
{{
    "user_query": "Play some upbeat rock music to wake me up.",
    "target_api": "apis.spotify.play"
}}

---

**Input Data:**

App Name: {APP_NAME}
App APIs:
{API_LIST}
"""

# INTRA_DOMAIN_PURPOSE_PROMPT = """
# You generate intra-domain AI agent training data. Given an API list for a single app, construct a logical scenario for a User Query.

# Task:
# Select a reasonable API (Action) to solve a user problem. The query should not be a simple function call; it must involve **constraints** or **context**.

# Core Logic Types:
# 1. Conditional/Filtering: "Delete emails *older than 30 days*", "Find items *under $50*".
# 2. Batch Operations: "Process *all* unread messages", "Buy *every* item in cart".
# 3. Complex Parameters: "Book a flight *using my business card* for *next Friday*".

# Rules:
# 1. Natural: Fluent, conversational English.
# 2. Specific: Include plausible details (dates, quantities, specific attributes) to make the query realistic.
# 3. Self-Contained: The user implies the necessary information exists within the app context (e.g., "my cart", "my history").

# Output Format:
# Output ONLY a raw JSON object. No Markdown.
# {{
#     "user_query": "Generated natural language instruction",
#     "target_api": "The primary API call_name used to fulfill the request"
# }}

# Example:
# App Name: Gmail
# App APIs:
# [
#     "apis.gmail.list_threads",
#     "apis.gmail.delete_thread",
#     "apis.gmail.send_message",
#     "apis.gmail.get_profile"
# ]

# Output JSON:
# {{
#     "user_query": "Delete all my archived gmail threads that are from before this calendar month.",
#     "target_api": "apis.gmail.delete_thread"
# }}

# ---

# Input Data:

# App Name: {APP_NAME}
# App APIs:
# {API_LIST}
# """

def parse_intra_purpose_from_response(response_text: str) -> dict:
    try:
        content = response_text.strip()
        
        # 1. 尝试移除 Markdown 代码块标记 ```json ... ```
        if "```" in content:
            # 提取第一个代码块的内容
            pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1)
        
        # 2. 正则提取最外层的 JSON 对象（非贪婪匹配）
        # 使用 .*? 非贪婪模式，防止匹配到多余的内容
        match = re.search(r'(\{.*?\})', content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = content

        data = json.loads(json_str)

        required_keys = ["user_query", "target_api"]
        missing_keys = [k for k in required_keys if k not in data]
        
        # [Fix] 如果缺少键，视为解析失败，返回 None
        if missing_keys:
            print(f"[Parse Warning] 结果缺少必要字段: {missing_keys}, 原文: {content[:100]}...")
            return None
            
        return data

    except json.JSONDecodeError as e:
        print(f"[Parse Error] JSON 解码失败: {e}")
        # print(f"--> 原始文本: {response_text}") # 调试时可打开
        return None
    except Exception as e:
        print(f"[Parse Error] 未知错误: {e}")
        return None