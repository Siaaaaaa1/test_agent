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

def get_intra_schema(app_names: list) -> str:
    """根据传入的 app_names 列表，动态拼装仅包含这些 App 的 Schema 字符串"""
    schemas = []
    for app in app_names:
        if app in APP_SCHEMA_MAP:
            schemas.append(APP_SCHEMA_MAP[app])
    return "\n\n".join(schemas)

# ================= 第一阶段：单域选拔 (Intra-Domain Selector) =================
INTRA_DOMAIN_SELECTOR_PROMPT = """
You are an expert API Judge for Single-App Automation.
Your task is to evaluate {CANDIDATE_COUNT} groups of API combinations for a specific App, and select the ONE group that makes the most logical sense for a complex, realistic daily automation task.

### App Information
Target App: {APP_NAME}
App Description: {ALL_APPS_DESC}

### All Available APIs Overview
{TARGET_APP_API_DESCS}

### Candidate API Groups
{CANDIDATE_GROUPS_STR}

### AppWorld Database Schema Overview
Use the following exact schema to accurately determine the `required_tables`. 
🚫 STRICT CONSTRAINT: Do NOT hallucinate or guess table names. You MUST strictly select from the entities listed below:

{DB_SCHEMA_OVERVIEW}

### Task Requirements
1. Evaluate which API combination allows for the most powerful single-app operations (e.g., bulk actions, aggregations, deep filtering, or time-bound queries). Avoid basic CRUD operations if more complex combinations exist.
2. Select exactly ONE winning group index.
3. Identify which underlying Database Tables (Table Names) would realistically need to be queried to provide context or parameters for this API combination.
🚫 STRICT LIMIT: You MUST select a MAXIMUM of 2 tables per App. Prioritize the most crucial tables.

### Output Format (JSON Only)
``json
{{
    "selected_group_index": 1,
    "selected_apis": ["api_name_1", "api_name_2"],
    "reasoning": "Brief explanation of why this combination supports advanced/complex tasks.",
    "required_tables": ["table_name_1", "table_name_2"]
}}
``
"""

def parse_intra_selector(response_text: str) -> dict:
    if not response_text:
        return {}
    try:
        content = response_text.strip()
        json_str = ""
        match = re.search(r'``(?:json)?\s*([\[\{].*?[\]\}])\s*``', content, re.DOTALL | re.IGNORECASE)
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
        logger.error(f"[Parse Error] Intra Selector JSON parsing failed: {e}\nRaw: {response_text}")
        return {}
    except Exception as e:
        logger.error(f"[Parse Error] Intra Selector: {e}")
        return {}


# ================= 第二阶段：单域生成 (Intra-Domain Generator) =================
INTRA_DOMAIN_GENERATOR_PROMPT = """
You are an expert Data Synthetic Generator for Single-App Automation.
Create a realistic, highly DIVERSE, and EXPLORATORY user command within a specific app using its selected APIs and Real Database Content.

### Inputs
1. **Target App**: {APP_NAME}
2. **Selected Target APIs**: {SELECTED_APIS}
3. **Anchor API Details**: 
{ANCHOR_API_DETAILS}
4. **Real Database Content**: 
{DB_CONTENT}
5. **All APIs in App (Brief)**:
{ALL_APIS_BRIEF}

### 🚫 STRICT Constraints & Diversity Goals
1. **Source Uncertainty (Black Box)**: Do NOT assume specific data content exists outside of what is explicitly shown in the `Real Database Content`.
2. **Exploratory & Fuzzy**: NEVER hardcode specific IDs, exact database keys, or over-specific parameters. The Agent must *explore* the app to find the target.
3. **Embrace Complexity**: Favor instructions that require aggregation (counting, summing), batch processing (loops), temporal logic ("last 5 days"), or filtering by attributes (ratings, read/unread status).

### Few-Shot Examples

❌ BAD Examples (Too simple, linear, or hard-coded):
- Bad 1: "Search for the email from 'John Doe'." (Reason: Assumes John Doe exists, hardcoded string, no complex filtering).
- Bad 2: "Buy order number 10024." (Reason: Hardcoded exact ID, not exploratory).
- Bad 3: "Create a task called 'Buy milk' in Todoist." (Reason: A simple 1-step creation. Lacks exploration, aggregation, or bulk context).

✅ GOOD Examples (Robust, Exploratory, Aggregation & Batching):
- Good 1 (Deep Search & Aggregation): "How many hours does the battery of ecobee SmartCamera last? Please answer as per its amazon reviews or questions/answers and only trust information from its verified purchasers."
- Good 2 (Conditional Batch Editing): "Relabel all my pr-1 and pr-2 email threads with P1 and P2, respectively, and remove all pr-3 labels."
- Good 3 (Relative State Modification): "Initiate returns via FedEx for everything in my last 3 amazon orders."
- Good 4 (Mathematical Filtering): "Move all products with over 3.7 rating from my amazon wish list to cart."
- Good 5 (Time-bound Batching): "Respond to all the emails I have received within the last 5 days (including today) that I have not replied to yet. The reply should be 'I am out of office'."
- Good 6 (Contextual Aggregation): "What is the total cost of my cable bills for this year? The bills are in '~/bills/' directory of my file system."

### Task
Generate 1 realistic user scenario that naturally triggers the `Selected Target APIs` based on the context of the `Real Database Content`. The query MUST be exploratory and complex.

### Output Format (JSON Only)
``json
[
    {{
        "user_query": "The fuzzy, complex, exploratory natural language instruction",
        "target_api": "The exact call_name of the primary API to execute"
    }}
]
``
"""

def parse_intra_generator(response_text: str) -> list:
    if not response_text:
        return []
    try:
        content = response_text.strip()
        json_str = ""
        match = re.search(r'``(?:json)?\s*(.*?)\s*``', content, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_str = match.group(1).strip()
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
                    return []

        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as e:
        logger.error(f"[Parse Error] Intra Generator JSON parsing failed: {e}\nRaw: {response_text}")
        return []
    except Exception as e:
        logger.error(f"[Parse Error] Intra Generator: {e}")
        return []