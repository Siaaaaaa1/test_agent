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