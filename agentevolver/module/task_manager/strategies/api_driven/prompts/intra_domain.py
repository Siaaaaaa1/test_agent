import json
import re

# 阶段二：单域泛化引导 Prompt (Generic Task Generation)

INTRA_DOMAIN_PURPOSE_PROMPT = """
You generate intra-domain AI agent training data. Given an API list for a single app, construct a logical scenario for a User Query.

Task:
Select a reasonable API (Action) to solve a user problem. The query should not be a simple function call; it must involve **constraints** or **context**.

Core Logic Types:
1. Conditional/Filtering: "Delete emails *older than 30 days*", "Find items *under $50*".
2. Batch Operations: "Process *all* unread messages", "Buy *every* item in cart".
3. Complex Parameters: "Book a flight *using my business card* for *next Friday*".

Rules:
1. Natural: Fluent, conversational English.
2. Specific: Include plausible details (dates, quantities, specific attributes) to make the query realistic.
3. Self-Contained: The user implies the necessary information exists within the app context (e.g., "my cart", "my history").

Output Format:
Output ONLY a raw JSON object. No Markdown.
{
    "user_query": "Generated natural language instruction",
    "target_api": "The primary API call_name used to fulfill the request"
}

Example:
App Name: Gmail
App APIs:
[
    "apis.gmail.list_threads",
    "apis.gmail.delete_thread",
    "apis.gmail.send_message",
    "apis.gmail.get_profile"
]

Output JSON:
{
    "user_query": "Delete all my archived gmail threads that are from before this calendar month.",
    "target_api": "apis.gmail.delete_thread"
}

---

Input Data:

App Name: {APP_NAME}
App APIs:
{API_LIST}
"""

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
        match = re.search(r'(\{.*\})', content, re.DOTALL)
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