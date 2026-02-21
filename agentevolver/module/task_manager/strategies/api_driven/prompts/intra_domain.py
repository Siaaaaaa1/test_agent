import json
import re

INTRA_DOMAIN_PURPOSE_PROMPT = """
You are an expert Data Synthetic Generator for Single-App Automation.
Create a realistic, logically robust user command within a specific app using its APIs.

You have access to:
1. **Global Context**: Descriptions of all available apps.
2. **App Overview**: A list of ALL APIs in this app (Format: `call_name: description`).
3. **Anchor API Details**: Detailed specifications (call_name, parameters, returns) for specific APIs we **MUST** focus on.

### 🚫 STRICT Constraints
1.  **Use Available Info**: Utilize the `Anchor API Details` to ensure parameter compatibility (e.g., if an API requires a `category`, ensure the user query implies a category filter).
2.  **Source Uncertainty (Black Box)**: 
    * Do NOT assume specific data content exists (e.g., specific senders like 'from HR', specific file contents).
    * Do NOT assume specific user-defined tags/folders exist unless they are system defaults (like 'Inbox', 'Spam').
3.  **Relative over Specific**: Prefer relative references (e.g., "my last email", "the most recent order") over specific entities.
4.  **Action Oriented**: The query must imply a clear action that corresponds to the capabilities of the Anchor API.

### ✅ Logic Patterns
1.  **Relative Search**: User refers to items by time or order (e.g., "Delete the last message").
2.  **Keyword Search**: User performs a broad search where 0 results is acceptable (e.g., "Search for 'invoice'").
3.  **State Modification**: User provides a value -> App updates settings (e.g., "Turn on Do Not Disturb").
4.  **Content Creation**: User provides content -> App creates a note/list/message (e.g., "Create a shopping list with 'Milk'").

### Few-Shot Examples

❌ **BAD Examples (Overly specific/Hard-coded logic):**
- **Bad 1:** "Search for the email from 'John Doe'." (*Reason: Assumes John Doe has emailed the user.*)
- **Bad 2:** "Move 'report.pdf' to the 'Finance' folder." (*Reason: Assumes specific file and folder names exist.*)
- **Bad 3:** "Play the song 'Shape of You'." (*Reason: Assumes this specific song is in the library.*)

✅ **GOOD Examples (Robust & Natural):**
- **Good 1:** "Find the most recent email I received and mark it as important." (Robust: Relies on time, not sender).
- **Good 2:** "Set a daily alarm for 7:00 AM labeled 'Morning Run'." (Robust: Creation/Setting).
- **Good 3:** "Search for tasks containing the word 'deadline' and mark them as complete." (Robust: Search intent is valid even if no tasks match).
- **Good 4:** "Update my status to 'Away' and set the auto-reply message to 'Traveling'." (Robust: Configuration).
- **Good 5:** "Create a new note titled 'Ideas' and append the text 'Project Alpha' to it." (Robust: Creation).
- **Good 6:** "Get the details of my last order." (Robust: Relative reference).

### Task
You are provided with a set of **Anchor APIs**. You **MUST** select one API from the provided Anchor list as the target action.
Generate 1 realistic user scenario where the user's intent directly leads to calling this Anchor API.
The constructed user query **MUST** utilize the functionality described in the **Anchor API Details**.

### Output Format (JSON Only)
[
    {{
        "user_query": "Natural language instruction 1",
        "target_api": "The exact call_name of the selected anchor API"
    }}
]

### Input Data

#### 📱 Target App: {APP_NAME}

**All APIs Overview:**
{TARGET_APP_API_DESCS}

**⚓ Anchor API Details (YOU MUST USE ONE OF THESE):**
{ANCHOR_API_DETAILS}

### 🚨 FINAL MANDATORY REQUIREMENT
**You MUST strictly adhere to the following logic:**
1. **Identify the Anchor API:** Look at the "Anchor API Details" provided above.
2. **Center the Task around the Anchor:** The generated `user_query` MUST be designed to specifically trigger the functionality of this Anchor API.
3. **Selection Verification:** The `target_api` in your output JSON **MUST** be exactly one of the `call_name`s listed in the **Anchor API Details** section.
"""

def parse_intra_purpose_from_response(response_text: str) -> list:
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
        required_keys = ["user_query", "target_api"]
        
        for index, item in enumerate(parsed_data):
            if not isinstance(item, dict):
                continue
            missing_keys = [k for k in required_keys if k not in item]
            if missing_keys:
                continue
            valid_items.append(item)
            
        return valid_items

    except Exception as e:
        print(f"[Parse Error] Intra Domain: {e}")
        return []