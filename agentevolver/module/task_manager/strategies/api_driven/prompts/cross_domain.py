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
{
    "user_query": "Generated natural language instruction",
    "source_info_api": "Selected Source API call_name",
    "target_action_api": "Selected Target API call_name"
}

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
{
    "user_query": "Check the time of my latest return on Amazon, then set an alarm on my phone for that time labeled 'Check refund'.",
    "source_info_api": "apis.amazon.show_return",
    "target_action_api": "apis.phone.create_alarm"
}

---

Input Data:

App 1 Name: {APP_NAME1}
App 1 APIs:
{API_LIST1}

App 2 Name: {APP_NAME2}
App 2 APIs:
{API_LIST2}
"""

def parse_cross_purpose_from_response(response_text: str) -> dict:
    try:
        content = response_text.strip()
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            json_str = content

        data = json.loads(json_str)

        required_keys = ["user_query", "source_info_api", "target_action_api"]
        missing_keys = [k for k in required_keys if k not in data]
        
        if missing_keys:
            print(f"[Parse Warning] 结果缺少必要字段: {missing_keys}")
            
        return data

    except json.JSONDecodeError as e:
        print(f"[Parse Error] JSON 解码失败: {e}")
        print(f"--> 原始文本: {response_text}")
        return None
    except Exception as e:
        print(f"[Parse Error] 未知错误: {e}")
        return None

# 阶段三：跨域合成 Prompt（基于条件的探索）

# PURPOSE_SYNTHESIS_PROMPT = """
# 你是一个专门用于生成跨应用（Cross-Domain）AI 代理训练数据的专家。
# 我将为你提供两个 App 的 API 列表。你的任务是构建一个合乎逻辑的场景，通过一个**用户自然语言指令（User Query）**，连接这两个 App。

# ### 任务核心 (Core Task)
# 请从提供的列表中挑选**最合理的一对 API**（一个来自源 App，一个来自目标 App），并生成一个用户指令。
# 这对 API 必须满足**参数依赖关系（Parameter Dependency）**：
# 1. **Source API (源)**：用于获取信息。其返回结果必须包含下一步所需的关键数据。
# 2. **Target API (目标)**：用于执行操作。其执行必须依赖于 Source API 查到的具体信息作为输入参数。

# ### 指令生成规则 (Rules for Query Generation)
# 1. **自然性**：指令应像普通用户对智能助手说的话，口语化、流畅且意图清晰。
# 2. **条件性**：由于具体数据（如订单号、具体人名）在当前上下文中未知，请使用泛指或描述性的语言（例如：“最近的一个订单”、“刚收到的那条短信”）。
# 3. **流程引导**：指令应隐含“先查 A，拿到信息后再做 B”的逻辑链条。

# ### 输出格式 (Strict JSON Output)
# 请仅输出一个合法的 JSON 对象，**不要**包含 Markdown 代码块或任何其他解释性文字。格式如下：
# {
#     "user_query": "生成的自然语言指令",
#     "source_info_api": "源 App 中被选中的那个 API 的 call_name",
#     "target_action_api": "目标 App 中被选中的那个 API 的 call_name"
# }

# ### 示例 (Example)
# **App 1 Name**: Amazon
# **App 1 APIs**:
# [
#     "apis.amazon.initiate_return",
#     "apis.amazon.show_return",
#     "apis.amazon.show_return_deliverers",
#     "apis.amazon.show_prime_plans"
# ]

# **App 2 Name**: Phone
# **App 2 APIs**:
# [
#     "apis.phone.show_alarms",
#     "apis.phone.create_alarm",
#     "apis.phone.show_alarm",
#     "apis.phone.delete_alarm",
#     "apis.phone.update_alarm"
# ]

# **Output JSON:**
# {
#     "user_query": "帮我查一下 Amazon 上最近的一个退货订单的发起时间，然后根据那个时间在手机上定一个闹钟，备注写'检查退款进度'。",
#     "source_info_api": "apis.amazon.show_return",
#     "target_action_api": "apis.phone.create_alarm"
# }

# ---

# ### 输入数据 (Input Data)

# **App 1 Name**: {APP_NAME1}
# **App 1 APIs**:
# {API_LIST1}

# **App 2 Name**: {APP_NAME2}
# **App 2 APIs**:
# {API_LIST2}
# """