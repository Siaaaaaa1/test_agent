import re
import time
import logging
from typing import Sequence, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from loguru import logger  # 使用 loguru 以保持一致性

from agentevolver.module.task_manager.base import LlmClient
from agentevolver.schema.task import Task

# 如果你有定义 TaskPreFilter 基类，请继承它，否则作为一个独立类即可
class LlmQualityPreFilter:
    def __init__(self, llm_client: LlmClient, num_threads: int = 10, **kwargs):
        """
        初始化预过滤器。
        
        Args:
            llm_client (LlmClient): 用于判断任务质量的 LLM 客户端。
            num_threads (int): 并发线程数。
        """
        self._llm_client = llm_client
        self._num_threads = num_threads
        
        # 简单的重试配置
        self._max_retries = 3

    def filter(self, tasks: Sequence[Task]) -> List[Task]:
        """
        并行过滤任务列表。
        """
        if not tasks:
            return []
        
        logger.info(f"[PreFilter] Starting quality check for {len(tasks)} tasks...")
        
        valid_tasks = []
        
        # 使用线程池并发请求 LLM
        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            future_to_task = {
                executor.submit(self._check_single_task, task): task 
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result_task = future.result()
                    if result_task:
                        valid_tasks.append(result_task)
                except Exception as e:
                    logger.exception(f"[PreFilter] Error processing task {task.task_id}: {e}")
                    # 发生异常时，保守策略可以选择丢弃，或者保留（取决于你想不想漏掉任务）
                    # 这里选择丢弃
                    continue
        
        logger.info(f"[PreFilter] Finished. Kept {len(valid_tasks)}/{len(tasks)} tasks.")
        return valid_tasks

    def _check_single_task(self, task: Task) -> Optional[Task]:
        """
        对单个任务进行 LLM 判别。
        返回 Task 对象表示保留，返回 None 表示丢弃。
        """
        query = task.query
        
        # 1. 构造 Prompt
        prompt = self._construct_prompt(query)
        
        # 2. 调用 LLM (带重试)
        response_content = self._chat_with_retry(prompt)
        
        if not response_content:
            logger.warning(f"[PreFilter] ⚠️ LLM No Response | Task ID: {task.task_id}")
            return None

        # 3. 解析结果
        is_good = self._parse_response(response_content)
        
        # 4. 打印日志并返回结果
        if is_good:
            # 使用绿色高亮 (如果有配置) 或普通 INFO
            logger.info(f"[PreFilter] ✅ KEEP | Query: {query[:100]}...")
            return task
        else:
            # 打印被丢弃的任务，方便调试
            logger.warning(f"[PreFilter] ❌ DROP | Reason: Poor Quality | Query: {query}")
            return None

    def _construct_prompt(self, query: str) -> list[dict]:
        """
        构造用于判断任务质量的 Prompt。
        """
        content = f"""You are a Data Quality Evaluator for an AI Agent dataset. Your goal is to identify meaningful and actionable user commands. You should accept queries that have a clear intent, even if they are concise or slightly broad. REJECT only those that contain logical fallacies, unverifiable assumptions, or are fundamentally impossible to execute.
Task Query: "{query}"
Evaluation Criteria:
The query must express a clear goal. Moderate vagueness is acceptable (e.g., not specifying a brand or model) as long as the Agent can supplement those details using common sense or search capabilities.
ACCEPT (Clear Intent): Commands with a clear action and object (e.g., "Buy a fan," "Book me a flight to London").
REJECT (Logical Flaws/Over-specification): Commands that include precise values or prerequisites that the user cannot guarantee, which would likely lead to execution failure.
Unacceptable: "Buy a fan that costs exactly $20.50" (Price fluctuations make this high-risk).
Unacceptable: "If my current balance is $100, then pay my phone bill" (The user should not speculate on internal states that only the Agent can verify).
ACCEPT (Delegated Commands): Queries where the user sets a goal and lets the Agent handle the filtering/optimization.
Example: "Help me find a good desktop fan," "Update my username to 'Alex'."
Is this a valid and usable query? Return strictly in this format: <answer>True</answer> or <answer>False</answer>"""
        
        
        
#         f"""You are a strict Data Quality Auditor for an advanced AI Agent dataset.
# Your goal is to accept **ONLY** high-quality, intent-clear, and logically robust user commands, and **REJECT** vague, trivial, or queries containing unreasonable assumptions.

# Task Query: "{query}"

# ### Evaluation Criteria:
#     The query contains key information needed to execute the task (e.g., category, usage, preferences) but **avoids including overly specific values that the user cannot verify (risking execution failure)**.
#     * *Bad (Too Vague):* "Buy a fan."
#     * *Bad (Overly Specific/Guessing):* "Buy a fan on Amazon that costs exactly $20.50." (User cannot predict exact pricing; high failure risk.)
#     * *Bad (Presumed State):* "Update my account name to 'Jane Smith' only if the current name is 'J. Doe'." (User should not speculate on current state in the command; just order the rename directly.)
#     * *Good (Clear Intent):* "Help me pick a highly-rated cooling fan suitable for a desktop on Amazon." (Delegates filtering to the Agent.)
#     * *Good (Functional Constraint):* "Update my account name to 'Jane Smith'."

# Is this a high-quality query?
# Return strictly in this format: <answer>True</answer> or <answer>False</answer>
# """
        
# ### Automatic FAIL Triggers (Immediate False):
# 1.  **Hallucinated Constraints:** Contains internal IDs, hashes, or overly precise unnecessary values unknown to the user (e.g., "Buy item with ID 83920").
# 2.  **Too Trivial:** Extremely simple parameter-less operations (e.g., "Click Confirm").
# 3.  **Impossible Logic:** Asks the Agent to access physical world objects or private states outside its permissions.
# 4.  **Refusals/Non-commands:** "I can't do that", "Hello", etc.

# ### Decision Logic:
# - If the query has clear intent, reasonable parameters, and is actionable by the Agent in an unknown environment -> True.
# - If the query relies on the user guessing the environment state (e.g., guessing a specific price or specific stock count) or is too vague -> False.
        return [{"role": "user", "content": content}]

    def _parse_response(self, content: str) -> bool:
        """
        解析 LLM 返回的 <answer>True/False</answer>
        """
        # 使用正则提取，忽略大小写和空白字符
        match = re.search(r"<answer>\s*(True|False)\s*</answer>", content, re.IGNORECASE)
        
        if match:
            result_str = match.group(1).lower()
            return result_str == "true"
        
        # Fallback: 如果没有标签，尝试直接找关键字（防止模型没遵循格式）
        # 但既然你要求严格格式，也可以默认返回 False
        if "true" in content.lower() and "false" not in content.lower():
            return True
        
        return False

    def _chat_with_retry(self, messages: list[dict]) -> Optional[str]:
        """
        带重试机制的 LLM 调用
        """
        for i in range(self._max_retries):
            try:
                # 采样参数：temperature 设低一点，让判断更确定
                response = self._llm_client.chat(
                    messages=messages, 
                    sampling_params={"temperature": 0.1, "max_tokens": 50}
                )
                
                # 兼容不同 Client 返回格式 (dict 或 str 或 object)
                if isinstance(response, dict):
                    return response.get("content", "")
                elif hasattr(response, "content"):
                    return response.content
                elif isinstance(response, str):
                    return response
                
            except Exception as e:
                logger.warning(f"[PreFilter] Retry {i+1} failed: {e}")
                time.sleep(2 ** i)
        
        return None