import json
import re

CROSS_DOMAIN_PURPOSE_PROMPT = """
You are an expert Data Synthetic Generator for Cross-App Automation.
Create a realistic, logically robust user command bridging two apps using their APIs. 

You have access to:
1. **Global Context**: Descriptions of all available apps.
2. **App Overviews**: A list of all APIs within the two target apps (Format: `call_name: description`).
3. **Anchor API Details**: Detailed specifications (call_name, parameters, returns) for specific APIs we **MUST** focus on.

### 🚫 STRICT Constraints
1.  **Use Available Info**: Utilize the `Anchor API Details` to ensure parameter compatibility (e.g., if an API requires a `song_id`, ensure the logical source provides something equivalent or a search capability).
2.  Source Uncertainty (Black Box): Do NOT assume specific data exists. You must refer to data dynamically (e.g., "my last email", "the current song").
3.  No Hard-coded Containers/Files: Do not assume specific user-defined folder names or filenames exist.
4.  Type & Semantic Safety: Do NOT mix Text with Files. Ensure extracted data logically fits the Target action.

### ✅ Logic Patterns
1.  Search & Act: Source gives Name -> Target Searches Name -> Act.
2.  Notification: Source gives Content -> Target sends to Contact.
3.  Record Keeping: Source gives Transaction -> Target logs it.
4.  File Management: Source gives File -> Target saves/uploads it.

### Few-Shot Examples

❌ BAD Examples (Overly specific/Hard-coded logic):
- Bad 1: Get the names of all tasks under my 'Concert Prep' section in Todoist and search Spotify for each name to add any matching live recordings to my queue. (*Reason: Assumes a specific project named 'Concert Prep' exists.*)
- Bad 2: Check if a file named 'weekly_report.pdf' exists in my Downloads directory, and if so, create a Gmail draft. (*Reason: Hard-codes a specific filename.*)
- Bad 3: Compress all files in my 'Projects' folder into a single archive. (*Reason: Assumes a specific folder named 'Projects' exists.*)

✅ GOOD Examples:
- Good 1: Forward the full content of the most recent text message I received to my work email address for archiving.
- Good 2: Retrieve the total cost of my very last Amazon order and request exactly half of that amount from 'Roommate' on Venmo with the note 'Split purchase'.
- Good 3: Get the share link for the song currently playing on Spotify and text it to the last person I called on the Phone.
- Good 4: Summarize the titles of all Amazon orders placed in the last 30 days and create a new list in SimpleNote containing these names.
- Good 5: Locate the last Venmo payment I made. Extract the transaction ID and amount, and email these details to my accountant for tax tracking.
- Good 6: Find the to-do item in my Todoist that is most relevant to shopping, and purchase that item on Amazon.
- Good 7: Identify the note most relevant to "workout playlist" in SimpleNote, and search for the corresponding songs on Spotify to play them.

### Task
Select the most naturally compatible APIs from Source and Target.
You **MUST** prioritize the provided **Anchor APIs**. 
The constructed user query **MUST** utilize the functionality described in the Anchor API Details.

### Output Format (JSON Only)
[
    {{
        "user_query": "Natural language instruction 1",
        "source_info_api": "API call_name from App 1",
        "target_action_api": "API call_name from App 2",
        "logic_pattern": "Pattern Name"
    }}
]

### Input Data

#### 📱 App 1 (Source): {APP_NAME1}

**All APIs Overview:**
{APP1_API_DESCS}

**⚓ Anchor API Details (YOU MUST USE ONE OF THESE):**
{APP1_ANCHOR_DETAILS}

---

#### 📱 App 2 (Target): {APP_NAME2}

**All APIs Overview:**
{APP2_API_DESCS}

**⚓ Anchor API Details (YOU MUST USE ONE OF THESE):**
{APP2_ANCHOR_DETAILS}

### 🚨 FINAL MANDATORY REQUIREMENT
**You MUST strictly adhere to the following logic:**
1. **Identify the Anchor API(s):** Look at the "Anchor API Details" provided above for both apps.
2. **Center the Task around the Anchor:** The generated `user_query` MUST be designed to trigger the functionality of the provided Anchor API(s).
3. **Selection Verification:** The `source_info_api` OR `target_action_api` in your output JSON **MUST** be one of the APIs explicitly listed in the **Anchor API Details** sections.
"""

def parse_cross_purpose_from_response(response_text: str) -> list:
    """
    解析 LLM 返回的 JSON 列表字符串，返回 Python list[dict]。
    如果解析失败或没有有效数据，返回空列表 []。
    """
    try:
        content = response_text.strip()
        
        # 1. [尝试移除 Markdown 代码块]
        if "```" in content:
            pattern = r"```(?:json)?\s*(\[.*?\])\s*```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1)
        
        # 2. [正则提取最外层的 JSON 列表]
        match = re.search(r'(\[.*\])', content, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            json_str = content

        parsed_data = json.loads(json_str)

        # 3. [类型检查]
        if not isinstance(parsed_data, list):
            return []

        # 4. [字段校验]
        valid_items = []
        required_keys = ["user_query", "source_info_api", "target_action_api"]
        
        for index, item in enumerate(parsed_data):
            if not isinstance(item, dict):
                continue
            missing_keys = [k for k in required_keys if k not in item]
            if missing_keys:
                continue
            valid_items.append(item)
            
        return valid_items

    except Exception as e:
        print(f"[Parse Error] Cross Domain: {e}")
        return []