import json
import re
from loguru import logger

# ================= 动态 Schema 字典库 =================
# 将各个 App 的 Schema 拆解为字典，便于按需动态组合，节省 Token 并提高 LLM 专注度
APP_SCHEMA_MAP = {
    "amazon": "## amazon (E-Commerce)\n- `Product`: Items available for purchase (contains price, rating, description, delivery_days).\n- `CartEntry`: Items currently in the user's shopping cart.\n- `Order`, `OrderItem`: History of completed purchases and specific items within those orders.\n- `Address`: User's delivery addresses (Home, Work, etc.).\n- `PaymentCard`: Credit/Debit cards linked to the account.\n- `PrimeSubscription`: Details about Amazon Prime membership.\n- `ProductReview`, `ProductReturn`: User reviews and return records for products.\n- `WishListEntry`: Items saved for later.",
    "venmo": "## venmo (Peer-to-Peer Payments)\n- `User`: Venmo account profiles, including current `venmo_balance`.\n- `Transaction`: Completed money transfers between users (contains amount, description, privacy status).\n- `PaymentRequest`: Pending requests for money (contains amount, description, status).\n- `Friendship`: Social connections/friends list on Venmo.\n- `Notification`: Alerts for received money or new requests.\n- `PaymentCard`: Linked bank cards for funding transactions.",
    "gmail": "## gmail (Email Services)\n- `Email`: Individual email messages (contains sender, recipient, subject, body content, date).\n- `UserEmailThread`: User's view of an email thread (contains read/unread status, starred, labels).\n- `Draft`: Unsent emails currently being composed.\n- `Attachment`: Files attached to specific emails.",
    "phone": "## phone (Device Native Apps)\n- `Contact`: Address book (contains names, phone numbers, emails, physical addresses, relationships like 'manager' or 'coworker').\n- `Alarm`: Configured device alarms (contains time, repeat_days, label, enabled status).\n- `GlobalTextMessage`, `UserTextMessage`: SMS text message history.\n- `GlobalVoiceMessage`, `UserVoiceMessage`: Voicemail history.",
    "todoist": "## todoist (Task Management)\n- `Task`, `SubTask`: To-do items (contains title, description, due_date, priority, is_completed).\n- `Project`, `Section`: Organization folders for tasks (e.g., 'Inbox', 'Grocery Shopping').\n- `Label`, `TaskLabelLink`: Tags applied to specific tasks.\n- `TaskComment`: Notes or discussions attached to a task.\n- `ProjectCollaboratorLink`: Users sharing a specific project.",
    "splitwise": "## splitwise (Shared Expenses)\n- `Group`, `GroupMember`: Groups of users sharing costs (e.g., 'Roommates', 'Trip to Japan').\n- `Expense`, `ExpenseShare`: Shared bills and how the cost is divided among members.\n- `Payment`: Settlements (records of users paying back their debts).\n- `ExpenseComment`: Chat and notes attached to a shared bill.",
    "spotify": "## spotify (Music Streaming)\n- `Song`, `Album`, `Artist`: Core music catalog entities (contains play_count, genre, duration).\n- `Playlist`, `PlaylistSong`: User-created playlists and the songs within them.\n- `MusicPlayer`: Current playback state (contains queue, is_playing, volume).\n- `SongLike`, `PlaylistLike`: User's saved/liked music.",
    "file_system": "## file_system (Local Storage)\n- `File`: Local files on the user's device (contains path, file_name, text/binary content).\n- `Directory`: Folders on the local device.",
    "simple_note": "## simple_note (Note Taking)\n- `Note`: Text notes (contains title, content, tags, pinned status). Often contains structured lists.",
    "supervisor": "## supervisor (Global/System Settings)\n- `AccountPassword`: The user's login passwords for all the apps mentioned above.\n- `Supervisor`, `Address`, `PaymentCard`: The user's primary identity, global addresses, and default bank cards."
}

def get_cross_schema(app_names: list) -> str:
    """根据传入的 app_names 列表，动态拼装仅包含这些 App 的 Schema 字符串"""
    schemas = []
    for app in app_names:
        if app in APP_SCHEMA_MAP:
            schemas.append(APP_SCHEMA_MAP[app])
    return "\n\n".join(schemas)


# ================= 第一阶段：跨域选拔 (Cross-Domain Selector) =================
CROSS_DOMAIN_SELECTOR_PROMPT = """
You are an expert API Judge for Cross-App Automation. 
Your task is to evaluate {CANDIDATE_COUNT} groups of cross-app API combinations spanning {APP_COUNT} apps, and select the ONE group that makes the most logical sense for a complex, natural cross-domain workflow.

### Apps Involved Context
{APPS_INFO_STR}

### Candidate API Groups
{CANDIDATE_GROUPS_STR}

### AppWorld Database Schema Overview
Use the following exact schema to accurately determine the `required_tables`. 
🚫 STRICT CONSTRAINT: Do NOT hallucinate or guess table names. You MUST strictly select from the entities listed below:

{DB_SCHEMA_OVERVIEW}

### Task Requirements
1. Evaluate which API combination represents the most realistic, multi-step information flow between these apps (e.g., Data Extraction -> Conditional Logic -> Action Target). Prioritize groups that allow for rich context merging.
2. Select exactly ONE winning group index.
3. Identify which underlying Database Tables (Table Names) in EACH app would realistically need to be queried.
🚫 STRICT LIMIT: You MUST select a MAXIMUM of 2 tables per App. Prioritize the most crucial tables.

### Output Format (JSON Only)
Output valid JSON only.
```json
{{
    "selected_group_index": 1,
    "selected_apis": {{
        "app_1_name": ["api_1", "api_2"],
        "app_2_name": ["api_3"]
    }},
    "reasoning": "Detailed explanation of the cross-app logical flow.",
    "required_tables": {{
        "app_1_name": ["table_1"],
        "app_2_name": ["table_2"]
    }}
}}
```
"""

def parse_cross_selector(response_text: str) -> dict:
    if not response_text:
        return {}
    try:
        content = response_text.strip()
        json_str = ""
        match = re.search(r'```(?:json)?\s*([\[\{].*?[\]\}])\s*```', content, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
        else:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = content[start_idx:end_idx+1]
            else:
                json_str = content
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"[Parse Error] Cross Selector JSON parsing failed: {e}\nRaw: {response_text}")
        return {}
    except Exception as e:
        logger.error(f"[Parse Error] Cross Selector: {e}")
        return {}


# ================= 第二阶段：跨域生成 (Cross-Domain Generator) =================
CROSS_DOMAIN_GENERATOR_PROMPT = """
You are an expert Data Synthetic Generator for Cross-App Automation. 
Create a realistic, highly DIVERSE, and EXPLORATORY user command bridging multiple apps, based on the selected APIs and Real Database Content.

### Inputs
1. **Apps Involved**: {APPS_NAMES}
2. **Selected APIs**: {SELECTED_APIS_JSON}
3. **Anchor API Details**: 
{ANCHOR_API_DETAILS}
4. **Real Database Content**: 
{DB_CONTENT}
5. **All APIs in App (Brief)**:
{ALL_APIS_BRIEF}

### 🚫 STRICT Constraints & Diversity Goals
1. **Source Uncertainty (Black Box)**: Do NOT assume specific data exists. You must refer to data dynamically via semantic descriptions, timeframes, or relationships (e.g., "the email from my manager", "the cost of my cart").
2. **No Hard-coded IDs/Filenames**: Do not assume specific user-defined folder names, exact filenames, or Exact IDs unless referencing standard system defaults (like '~/documents').
3. **Complex Logic Injection**: Push for scenarios that require conditional checks (e.g., "If the total is higher than X..."), aggregations (e.g., "cost in total"), or multi-recipient looping (e.g., "for each of my following friends...").
4. **Data Consistency**: Ensure the logic holds up based on the provided `Real Database Content`.

### Few-Shot Examples

❌ BAD Examples (Too simple, linear, or hard-coded):
- Bad 1: "Get order #123 from Amazon and request $10 from Venmo ID 456." (Reason: Uses hardcoded, unrealistic IDs instead of natural language exploration).
- Bad 2: "Find 'report.pdf' in File System and email it to John." (Reason: Too linear, assumes specific exact filenames, lacks conditional or aggregation logic).
- Bad 3: "Check my Venmo balance and buy a watch on Amazon." (Reason: The two apps are disconnected in logic. Why does the balance dictate the watch purchase? Lacks a cohesive narrative).

✅ GOOD Examples (Robust, Exploratory, Conditional, and Multi-App Chaining):
- Good 1 (Conditionals & Multi-Target): "Buy the highest rated popcorn maker available on Amazon now, one for each of my following friends: Grant, Jose, Brenda. They have to be gift wrapped. If the total delivery fee is higher than the monthly prime subscription cost, subscribe me to prime first."
- Good 2 (State Syncing & Math): "I maintain my work schedule in SimpleNote and track my tasks in Todoist. Delete the completed tasks from 'Today's Target'. Then, move the maximum number of incomplete tasks from my Inbox to 'Today's Target' assuming I work back-to-back as per my schedule."
- Good 3 (Context Merging): "Angelica asked me for my song recommendations over email. I started drafting the response email off the top of my head. Please update the email draft with all of my liked songs that are in my library. Keep the existing format of the email, making changes only to the song entries."
- Good 4 (Delegation Pipeline): "Denise has requested me to buy 'Nintendo Switch Lite' on amazon for them as their card is currently blocked. Place the order, forward the confirmation email containing the receipt to them, and make a venmo request to them for the total cost of the order."

### Task
Generate 1 realistic cross-app user scenario triggering the `Selected APIs` using the context of the `Real Database Content`. The query MUST require deep exploration and complex reasoning.

### Output Format (JSON Only)
```json
[
    {{
        "user_query": "The fuzzy, exploratory, and complex natural language instruction",
        "source_info_api": "API call_name acting as the source",
        "target_action_api": "API call_name acting as the action target",
        "logic_pattern": "Pattern Name (e.g., Conditional Chaining, Aggregation, State Sync)"
    }}
]
```
"""

def parse_cross_generator(response_text: str) -> list:
    if not response_text:
        return []
    try:
        content = response_text.strip()
        json_str = ""
        match_list = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL | re.IGNORECASE)
        match_dict = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL | re.IGNORECASE)
        
        if match_list:
            json_str = match_list.group(1)
        elif match_dict:
            json_str = f"[{match_dict.group(1)}]"
        else:
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = content[start_idx:end_idx+1]
            else:
                start_idx_dict = content.find('{')
                end_idx_dict = content.rfind('}')
                if start_idx_dict != -1 and end_idx_dict != -1 and start_idx_dict < end_idx_dict:
                    json_str = f"[{content[start_idx_dict:end_idx_dict+1]}]"
                else:
                    json_str = content
        
        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as e:
        logger.error(f"[Parse Error] Cross Generator JSON parsing failed: {e}\nRaw: {response_text}")
        return []
    except Exception as e:
        logger.error(f"[Parse Error] Cross Generator: {e}")
        return []