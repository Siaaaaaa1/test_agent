import json
import re

INTRA_DOMAIN_PURPOSE_PROMPT = """
You are an expert data generator for an AI Agent in a **"Black Box" Environment**.
Given an App's APIs, select one target API and construct a **logical, natural** User Query.

### 🌑 Core Context (Black Box)
The user **CANNOT** see internal states (IDs, specific filenames, inventory). They rely on **fuzzy needs** or **common sense**, expecting the Agent to handle details.
*Goal:* Balance **Ambiguity** (don't micromanage) and **Precision** (give enough context).

### 📐 Construction Guidelines
1.  **The Goldilocks Rule (No Gambling):**
    Provide functional constraints, but NEVER guess specific values the user can't know.
    * *Bad (Gambling):* "Buy a fan that costs exactly $20.50." (User can't predict exact price).
    * *Good:* "Buy a highly-rated desktop fan for under $25." (Delegates filtering to Agent).

2.  **No Hard-coded Containers/Files (Robustness):**
    Do not assume specific user-defined folder names or filenames exist unless you are creating them.
    * *Bad:* "Compress my 'Projects' folder." (Assumes a specific folder exists).
    * *Good:* "Find all documents from last week and archive them." (Discovery based).

3.  **Blind Action (State-Agnostic):**
    State the desired outcome directly. Do NOT use conditional logic based on hidden states.
    * *Bad:* "If my name is 'J. Doe', change it to 'Jane'."
    * *Good:* "Update my account name to 'Jane Smith'."

### 🎯 Target Scenarios
* **Acquisition:** Search with soft constraints (e.g., "Find a restaurant near me rated 4.5+").
* **Batch Processing:** Operate on groups via attributes (e.g., "Move all PDF files to Documents").
* **Overwrite:** Change settings directly (e.g., "Set bio to 'Working hard'").

### Task
Select a target API. Generate 1 realistic user scenario based on fuzzy needs or functional constraints. Construct a natural, discovery-oriented user query.

### Few-Shot Examples

**❌ BAD Examples (Overly specific/Hard-coded logic):**
- **Bad 1:** Get the names of all tasks under my 'Concert Prep' section and search for them online. (*Reason: Assumes a specific section name exists.*)
- **Bad 2:** Check if a file named 'weekly_report.pdf' exists in my directory and send it to my boss. (*Reason: Hard-codes a specific filename.*)
- **Bad 3:** Move all images from my 'Projects' folder to the cloud. (*Reason: Assumes a specific folder name exists.*)

**✅ GOOD Examples (Robust & Natural):**
- **Good 1:** Find a highly-rated coffee maker under $50 and add it to my shopping cart.
- **Good 2:** Archive all emails received from 'newsletter@tech.com' in the last 30 days.
- **Good 3:** Set a wake-up alarm for 7:00 AM on weekdays, replacing any existing alarms.
- **Good 4:** Send $20 for 'Lunch' to the last person I paid.
- **Good 5:** Create a new note titled 'Grocery List' and add 'Milk' as the first line.

### Output Format (JSON Only)
[
    {{
        "user_query": "Natural language instruction 1",
        "target_api": "The primary API call_name used"
    }}
]

### Input Data
App Name: {APP_NAME}
App APIs: {API_LIST}
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
        # 查找以 [ 开头，以 ] 结尾的内容
        match = re.search(r'(\[.*\])', content, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            json_str = content

        parsed_data = json.loads(json_str)

        # 3. [类型检查] 确保解析出来的是列表
        if not isinstance(parsed_data, list):
            print(f"[Parse Warning] 期望得到 list，但得到了 {type(parsed_data)}")
            return []

        # 4. [字段校验] 遍历 List，筛选有效项
        valid_items = []
        required_keys = ["user_query", "target_api"]
        
        for index, item in enumerate(parsed_data):
            if not isinstance(item, dict):
                continue
                
            missing_keys = [k for k in required_keys if k not in item]
            
            # 如果缺少键，跳过该项并打印警告
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