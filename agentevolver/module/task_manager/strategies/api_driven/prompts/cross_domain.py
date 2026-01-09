import json
import re

# 阶段三：跨域合成 Prompt (Condition-Based Exploration)


# CROSS_DOMAIN_PURPOSE_PROMPT = """
# You are an expert Data Synthetic Generator specializing in Cross-App Automation.
# Your task is to create a realistic, logically robust user command that bridges two apps based on their provided APIs.

# ### The Challenge
# Users often want to pipe data from a Source App (Data Provider) to a Target App (Action Performer).
# However, you must ensure the connection is Logically Valid and Data-Type Compatible.

# ### 🚫 STRICT Constraints:
# 1.  Source Uncertainty (The "Black Box" Rule):
#     * Do NOT assume the Source App definitely contains specific data (e.g., "the song 'Hello'").
#     * Instead, use queries that imply retrieval or discovery.
#     * *Bad:* "Take the song 'Hello' from Spotify..." (Assumes it exists).
#     * *Good:* "Find out what song I last liked on Spotify..." (Discovery driven).

# 2.  No "Magic" ID Jumps:
#     * Do NOT assume an entity ID or Name from App A can be directly added to App B without a search step.
#     * *Bad:* "Get the song name from Spotify and add it to Amazon Cart." (Amazon needs a product ID, not a string).
#     * *Good:* "Get the song name from Spotify, search for it on Amazon, and add the result to Cart."

# 3.  No Type Mismatches:
#     * Do NOT use a File/Image (e.g., `receipt.pdf`) as a Text String.
#     * Do NOT use a raw Text String (e.g., "Project Plan") as a File Attachment.
#     * Ensure the Target API actually accepts the data format provided by the Source API.

# 4.  Fuzzy Connection:
#     * The bridge between apps should often be semantic, requiring the Agent to extract or interpret information.
#     * *Good:* "Read the *content* of the latest note, and perform a search on Amazon based on *keywords found in it*."

# ### ✅ Approved Logic Patterns (Use these):
# 1.  Search & Act: Source gives a Name -> Target Searches for that Name -> User acts on result.
#     * *Ex:* "Get the name of the playing song (Spotify) -> Search for it on Amazon to buy the CD."
# 2.  Notification/Sharing: Source gives Content -> Target sends it to someone.
#     * *Ex:* "Get the Amazon order number -> Text it to my wife (Phone)."
# 3.  Record Keeping: Source gives Transaction/Event -> Target notes it down.
#     * *Ex:* "Get the last Venmo payment amount -> Log it in Splitwise."
# 4.  File Management: Source gives a File -> Target saves/uploads it.
#     * *Ex:* "Get the attachment from the last email -> Save it to File System."

# ### Task
# Select ONE compatible API from the Source and ONE from the Target. Construct a natural language query that a human would actually say.

# ### Few-Shot Examples
# Use these examples to understand how to construct generic, exploratory commands that don't rely on specific, hard-to-guess entities.
# #### Example 1
# user_query: Find the total cost of my most recent order on Amazon and request that exact amount from a friend on Venmo.
# #### Example 2
# user_query: Search my recent emails for song names and create a new Spotify playlist containing those tracks.
# #### Example 3
# user_query: Check the delivery status of my last Amazon order and text the tracking information to a specific contact.
# #### Example 4
# user_query: Get the list of tracks from my 'Favorites' playlist on Spotify and save the song titles to a text file in my documents folder.
# #### Example 5
# user_query: Retrieve the description of my last payment on Venmo and create a new note in Simple Note with that description as the title.

# ### Output Format (JSON Only)
# {{
#     "user_query": "The natural language instruction. Must be logically sound.",
#     "source_info_api": "The specific API name from App 1",
#     "target_action_api": "The specific API name from App 2",
#     "logic_pattern": "One of: Search & Act, Notification, Record Keeping, File Management"
# }}

# ### Input Data
# App 1 (Source): {APP_NAME1}
# APIs: {API_LIST1}

# App 2 (Target): {APP_NAME2}
# APIs: {API_LIST2}
# """


# CROSS_DOMAIN_PURPOSE_PROMPT = """
# You are an expert data generator creating Cross-App Automation training data for an AI Agent.
# Given a few APIs for two different Apps, you need to select one API from each and construct a logical "Pipeline Scenario" centered around them, where information flows from the "Source App" to the "Target App" via a single User Query.

# ### Core Context (The Data Pipeline)
# The user wants to bridge two isolated apps.
# 1.  Source App (Data Provider): The user acts as a "Black Box" observer. They don't know the exact content (e.g., the exact tracking number or meeting ID), so they must use Relative References (e.g., "the last email", "my upcoming meeting") to refer to the data.
# 2.  Target App (Action Performer): The user wants to use the data retrieved from the Source to perform an action here.

# ### Task Goal
# Construct a natural language instruction that:
# 1.  Identifies specific info in the Source App (using relative logic).
# 2.  Pipes that info into the Target App to execute a task.

# ### Output Format
# Output ONLY a raw JSON object. Do NOT use Markdown code blocks.
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

# Input Data:

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

CROSS_DOMAIN_PURPOSE_PROMPT = """
You are an expert Data Synthetic Generator for Cross-App Automation.
Create a realistic, logically robust user command bridging two apps using their APIs. Ensure the connection is Logically Valid and Data-Type Compatible.

### 🚫 STRICT Constraints
1.  Source Uncertainty (Black Box): Do NOT assume specific data exists. Use queries that imply retrieval (e.g., "Find my last...", "Search for...").
2.  No Hard-coded Containers/Files: Do not assume specific user-defined folder names (e.g., "Work") or filenames (e.g., "data.csv") exist.
3.  No "Magic" ID Jumps: Source Names are NOT valid Target IDs. Logic must imply a Search step.
4.  Type & Semantic Safety: Do NOT mix Text with Files. Ensure extracted data logically fits the Target action.

### ✅ Logic Patterns
1.  Search & Act: Source gives Name -> Target Searches Name -> Act.
2.  Notification: Source gives Content -> Target sends to Contact.
3.  Record Keeping: Source gives Transaction -> Target logs it.
4.  File Management: Source gives File -> Target saves/uploads it.
5.  Other Reasonable Patterns: Ensure logical flow and data compatibility.

### Few-Shot Examples

❌ BAD Examples (Overly specific/Hard-coded logic):
- Bad 1: Get the names of all tasks under my 'Concert Prep' section in Todoist and search Spotify for each name to add any matching live recordings to my queue. (*Reason: Assumes a specific project named 'Concert Prep' exists.*)
- Bad 2: Check if a file named 'weekly_report.pdf' exists in my Downloads directory, and if so, create a Gmail draft. (*Reason: Hard-codes a specific filename.*)
- Bad 3: Compress all files in my 'Projects' folder into a single archive. (*Reason: Assumes a specific folder named 'Projects' exists.*)

✅ GOOD Examples (Robust & Discovery-driven):
- Good 1: Find the total cost of my most recent Amazon order and request that amount from a friend on Venmo.
- Good 2: Search my recent emails for song names and create a new Spotify playlist with them.
- Good 3: Retrieve my last Venmo payment description and create a Simple Note with that title.

### Task
Select the most naturally compatible APIs from Source and Target. Generate 1 highly realistic user scenario that fits a provided Logic Pattern and addresses a genuine, everyday need. Construct a natural, intent-driven query that strictly adheres to the Source Uncertainty constraint.

### Output Format (JSON Only)
[
    {{
        "user_query": "Natural language instruction 1",
        "source_info_api": "API name from App 1",
        "target_action_api": "API name from App 2",
        "logic_pattern": "Pattern Name"
    }}
]

### Input Data
App 1 (Source): {APP_NAME1}
APIs: {API_LIST1}

App 2 (Target): {APP_NAME2}
APIs: {API_LIST2}
"""

import json
import re

def parse_cross_purpose_from_response(response_text: str) -> list:
    """
    解析 LLM 返回的 JSON 列表字符串，返回 Python list[dict]。
    如果解析失败或没有有效数据，返回空列表 []。
    """
    try:
        content = response_text.strip()
        
        # 1. [尝试移除 Markdown 代码块]
        # 兼容 ```json [ ... ] ``` 或 ``` [ ... ] ```
        if "```" in content:
            pattern = r"```(?:json)?\s*(\[.*?\])\s*```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1)
        
        # 2. [正则提取最外层的 JSON 列表]
        # 查找以 [ 开头，以 ] 结尾的内容
        match = re.search(r'(\[.*\])', content, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            # 如果正则没匹配到，尝试直接解析原文本（防止没写中括号但本身就是JSON）
            json_str = content

        parsed_data = json.loads(json_str)

        # 3. [类型检查] 确保解析出来的是列表
        if not isinstance(parsed_data, list):
            print(f"[Parse Warning] 期望得到 list，但得到了 {type(parsed_data)}")
            # 如果模型偶尔只返回了一个 dict，可以尝试在这里做一层兼容： return [parsed_data]
            return []

        # 4. [字段校验] 遍历 List，筛选有效项
        valid_items = []
        required_keys = ["user_query", "source_info_api", "target_action_api"]
        
        for index, item in enumerate(parsed_data):
            if not isinstance(item, dict):
                continue
                
            missing_keys = [k for k in required_keys if k not in item]
            
            if missing_keys:
                print(f"[Parse Warning] 第 {index+1} 项缺少必要字段: {missing_keys}")
                continue
            
            valid_items.append(item)
            
        return valid_items

    except json.JSONDecodeError as e:
        print(f"[Parse Error] JSON 解码失败: {e}")
        # print(f"--> 原始文本: {response_text}") 
        return []
    except Exception as e:
        print(f"[Parse Error] 未知错误: {e}")
        return []