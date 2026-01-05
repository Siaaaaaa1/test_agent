import json
import re

# 阶段三：跨域合成 Prompt (Condition-Based Exploration)

CROSS_DOMAIN_PURPOSE_PROMPT = """
You generate cross-domain AI agent training data. Given API lists for two apps, construct a logical scenario connecting them via a single User Query.

Task:
Select one API from the Source App and one from the Target App. They must satisfy parameter dependency:
1. Source API: Retrieves information. Its output contains data needed by the Target API.
2. Target API: Performs an action using data from the Source API as input.

Rules:
1. Natural: The query must be fluent and conversational.
2. Descriptive: Use generic terms (e.g., "the latest order") as specific IDs are unknown.
3. Flow: Implies logic of "Check A first, then do B with that info".

Output Format:
Output ONLY a raw JSON object. No Markdown blocks.
{{
    "user_query": "Generated natural language instruction",
    "source_info_api": "Selected Source API call_name",
    "target_action_api": "Selected Target API call_name"
}}

Example:
App 1 Name: Amazon
App 1 APIs:
[
    "apis.amazon.initiate_return",
    "apis.amazon.show_return",
    "apis.amazon.show_return_deliverers",
    "apis.amazon.show_prime_plans"
]

App 2 Name: Phone
App 2 APIs:
[
    "apis.phone.show_alarms",
    "apis.phone.create_alarm",
    "apis.phone.show_alarm",
    "apis.phone.delete_alarm",
    "apis.phone.update_alarm"
]

Output JSON:
{{
    "user_query": "Check the time of my latest return on Amazon, then set an alarm on my phone for that time labeled 'Check refund'.",
    "source_info_api": "apis.amazon.show_return",
    "target_action_api": "apis.phone.create_alarm"
}}

---

Input Data:

App 1 Name: {APP_NAME1}
App 1 APIs:
[{API_LIST1}]

App 2 Name: {APP_NAME2}
App 2 APIs:
[{API_LIST2}]
"""

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