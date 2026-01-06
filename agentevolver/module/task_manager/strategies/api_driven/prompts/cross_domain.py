import json
import re

# 阶段三：跨域合成 Prompt (Condition-Based Exploration)

CROSS_DOMAIN_PURPOSE_PROMPT = """
You are an expert Data Synthetic Generator specializing in **Cross-App Automation**.
Your task is to create a realistic, logically robust user command that bridges two apps based on their provided APIs.

### The Challenge
Users often want to pipe data from a **Source App** (Data Provider) to a **Target App** (Action Performer).
However, you must ensure the connection is **Logically Valid** and **Data-Type Compatible**.

### 🚫 STRICT Constraints (Avoid these failures):
1.  **No "Magic" ID Jumps:** Do NOT assume a "Song Name" from Spotify can be directly added to an Amazon Cart. The logic must imply a **Search** step (e.g., "Find the song on Amazon and buy it", NOT "Add the song name to cart").
2.  **No Type Mismatches:**
    * Do NOT use a **File/Image** (e.g., receipt.pdf) as **Text Description**.
    * Do NOT use a **Text String** (e.g., "Song Title") as a **File Attachment**.
3.  **No Unreasonable Assumptions:** Do NOT assume "the last email" contains a phone number. Instead, use reliable metadata (Sender Name, Subject, Date, Attachment).
4.  **No Physical World Control:** Do NOT ask to "show on screen" or "unlock phone".

### ✅ Approved Logic Patterns (Use these):
1.  **Search & Act:** Source gives a Name -> Target **Searches** for that Name -> User acts on result.
    * *Ex:* "Get the name of the playing song (Spotify) -> Search for it on Amazon to buy the CD."
2.  **Notification/Sharing:** Source gives Content -> Target sends it to someone.
    * *Ex:* "Get the Amazon order number -> Text it to my wife (Phone)."
3.  **Record Keeping:** Source gives Transaction/Event -> Target notes it down.
    * *Ex:* "Get the last Venmo payment amount -> Log it in Splitwise."
4.  **File Management:** Source gives a File -> Target saves/uploads it.
    * *Ex:* "Get the attachment from the last email -> Save it to File System."

### Input Data
App 1 (Source): {APP_NAME1}
APIs: {API_LIST1}

App 2 (Target): {APP_NAME2}
APIs: {API_LIST2}

### Task
Select ONE compatible API from the Source and ONE from the Target. Construct a natural language query that a human would actually say.

### Output Format (JSON Only)
{{
    "user_query": "The natural language instruction. Must be logically sound.",
    "source_info_api": "The specific API name from App 1",
    "target_action_api": "The specific API name from App 2",
    "logic_pattern": "One of: Search & Act, Notification, Record Keeping, File Management"
}}
"""

# CROSS_DOMAIN_PURPOSE_PROMPT = """
# You are an expert data generator creating Cross-App Automation training data for an AI Agent.
# Given a few APIs for two different Apps, you need to select one API from each and construct a logical **"Pipeline Scenario"** centered around them, where information flows from the "Source App" to the "Target App" via a single User Query.

# ### Core Context (The Data Pipeline)
# The user wants to bridge two isolated apps.
# 1.  **Source App (Data Provider):** The user acts as a "Black Box" observer. They don't know the exact content (e.g., the exact tracking number or meeting ID), so they must use **Relative References** (e.g., "the last email", "my upcoming meeting") to refer to the data.
# 2.  **Target App (Action Performer):** The user wants to **use** the data retrieved from the Source to perform an action here.

# ### Task Goal
# Construct a natural language instruction that:
# 1.  **Identifies** specific info in the Source App (using relative logic).
# 2.  **Pipes** that info into the Target App to execute a task.

# ### Output Format
# Output ONLY a raw JSON object. **Do NOT** use Markdown code blocks.
# {{
#     "user_query": "Generated natural language instruction clearly connecting both apps",
#     "source_info_api": "The Source API call_name (Data Provider)",
#     "target_action_api": "The Target API call_name (Action Performer)"
# }}

# ### Example
# App 1: Amazon (Source) [apis.amazon.get_last_order]
# App 2: Gmail (Target) [apis.gmail.send_email]

# Output JSON:
# {{
#     "user_query": "Find the delivery date of my last Amazon order and email it to my boss with the subject 'Package Arrival'.",
#     "source_info_api": "apis.amazon.get_last_order",
#     "target_action_api": "apis.gmail.send_email"
# }}

# ---

# **Input Data:**

# App 1 Name (Source): {APP_NAME1}
# App 1 APIs:
# [{API_LIST1}]

# App 2 Name (Target): {APP_NAME2}
# App 2 APIs:
# [{API_LIST2}]
# """

# CROSS_DOMAIN_PURPOSE_PROMPT = """
# You generate cross-domain AI agent training data. Given API lists for two apps, construct a logical scenario connecting them via a single User Query.

# Task:
# Select one API from the Source App and one from the Target App. They must satisfy parameter dependency:
# 1. Source API: Retrieves information. Its output contains data needed by the Target API.
# 2. Target API: Performs an action using data from the Source API as input.

# Rules:
# 1. Natural: The query must be fluent and conversational.
# 2. Descriptive: Use generic terms (e.g., "the latest order") as specific IDs are unknown.
# 3. Flow: Implies logic of "Check A first, then do B with that info".

# Output Format:
# Output ONLY a raw JSON object. No Markdown blocks.
# {{
#     "user_query": "Generated natural language instruction",
#     "source_info_api": "Selected Source API call_name",
#     "target_action_api": "Selected Target API call_name"
# }}

# Example:
# App 1 Name: Amazon
# App 1 APIs:
# [
#     "apis.amazon.initiate_return",
#     "apis.amazon.show_return",
#     "apis.amazon.show_return_deliverers",
#     "apis.amazon.show_prime_plans"
# ]

# App 2 Name: Phone
# App 2 APIs:
# [
#     "apis.phone.show_alarms",
#     "apis.phone.create_alarm",
#     "apis.phone.show_alarm",
#     "apis.phone.delete_alarm",
#     "apis.phone.update_alarm"
# ]

# Output JSON:
# {{
#     "user_query": "Check the time of my latest return on Amazon, then set an alarm on my phone for that time labeled 'Check refund'.",
#     "source_info_api": "apis.amazon.show_return",
#     "target_action_api": "apis.phone.create_alarm"
# }}

# ---

# Input Data:

# App 1 Name: {APP_NAME1}
# App 1 APIs:
# [{API_LIST1}]

# App 2 Name: {APP_NAME2}
# App 2 APIs:
# [{API_LIST2}]
# """

def parse_cross_purpose_from_response(response_text: str) -> dict:
    try:
        content = response_text.strip()
        
        # 1. [新增] 尝试移除 Markdown 代码块标记 ```json ... ```
        if "```" in content:
            pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1)
        
        # 2. [优化] 正则提取最外层的 JSON 对象（使用非贪婪匹配 .*?）
        match = re.search(r'(\{.*?\})', content, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            json_str = content

        data = json.loads(json_str)

        required_keys = ["user_query", "source_info_api", "target_action_api"]
        missing_keys = [k for k in required_keys if k not in data]
        
        # [优化] 缺少 Key 时返回 None，而不是返回不完整的字典
        if missing_keys:
            print(f"[Parse Warning] 跨域结果缺少必要字段: {missing_keys}")
            # print(f"--> 原文片段: {content[:100]}...")
            return None
            
        return data

    except json.JSONDecodeError as e:
        print(f"[Parse Error] JSON 解码失败: {e}")
        # print(f"--> 原始文本: {response_text}") 
        return None
    except Exception as e:
        print(f"[Parse Error] 未知错误: {e}")
        return None