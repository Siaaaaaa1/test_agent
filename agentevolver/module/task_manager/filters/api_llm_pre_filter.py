import re
import time
import logging
from typing import Sequence, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import json

from loguru import logger  # 使用 loguru 以保持一致性

from agentevolver.module.task_manager.base import LlmClient
from agentevolver.schema.task import Task

class LlmQualityPreFilter:
    """
    基于大模型的任务质量前置过滤器。
    在任务生成后、丢入环境执行前，通过 LLM 筛除逻辑不通或意图不明的劣质任务。
    """
    def __init__(self, llm_client: LlmClient, num_threads: int = 20, **kwargs):
        """
        初始化预过滤器。
        
        Args:
            llm_client (LlmClient): 用于判断任务质量的 LLM 客户端。
            num_threads (int): 并发线程数。修改为20以提高并发吞吐。
        """
        self._llm_client = llm_client
        self._num_threads = num_threads
        
        # 简单的重试配置 (网络问题重试5次)
        self._max_retries = 5
        
        # 线程锁用于同进程内多线程安全写入（ThreadPoolExecutor 场景）。
        # 注意：threading.Lock 仅保护同一进程内的并发，不跨进程；
        # 若改为多进程并行，需换用 multiprocessing.Lock 或 fcntl.flock。
        self._file_lock = threading.Lock()

    def filter(self, tasks: Sequence[Task]) -> List[Task]:
        """
        并行过滤任务列表。
        """
        if not tasks:
            return []
        
        logger.info(f"[PreFilter] Starting quality check for {len(tasks)} tasks...")
        
        valid_tasks = []
        
        # 使用线程池并发请求 LLM，加速评估过程
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
                    # 发生异常时，保守策略选择丢弃
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
        
        # 2. 调用 LLM (带重试，路径评估仅使用默认的 qwen3.5-plus)
        response_content = self._chat_with_retry(prompt)
        
        if not response_content:
            logger.warning(f"[PreFilter] ⚠️ LLM No Response | Task ID: {task.task_id}")
            return None

        # 3. 解析大模型是否给出了 True 的放行标签
        is_good = self._parse_response(response_content)
        
        # --- 数据准备 ---
        # 基础信息 (用于原有的 json)
        simple_task_info = {
            "query": task.query,
            "data_id": task.metadata.get('data_id', 'unknown') if task.metadata else 'unknown'
        }

        # 详细信息 (用于新的 json，增加了大模型实际输出的内容 response_content)
        detailed_task_info = {
            "query": task.query,
            "data_id": task.metadata.get('data_id', 'unknown') if task.metadata else 'unknown',
            "response_content": response_content  
        }

        # --- 保存逻辑 ---
        try:
            base_dir = "./tmp"
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)

            # 根据好坏分配到不同的流式日志文件中
            if is_good:
                file_name_std = "tasks_keep_pre_filter.json"
                file_name_detail = "tasks_keep_pre_filter_with_response.json"
            else:
                file_name_std = "tasks_drop_pre_filter.json"
                file_name_detail = "tasks_drop_pre_filter_with_response.json"

            # 内部闭包函数：安全的线程锁写入
            def _append_to_json_locked(filename, data_dict):
                full_path = os.path.join(base_dir, filename)
                # [修复] 使用线程锁替代 fcntl
                with self._file_lock:
                    try:
                        data_list = []
                        if os.path.exists(full_path):
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if content:
                                    try:
                                        data_list = json.loads(content)
                                    except json.JSONDecodeError:
                                        data_list = []
                        
                        # 追加新数据
                        data_list.append(data_dict)
                        
                        # 覆写文件
                        with open(full_path, 'w', encoding='utf-8') as f:
                            json.dump(data_list, f, ensure_ascii=False, indent=4)
                            
                    except Exception as file_e:
                        logger.error(f"[PreFilter] File write error for {filename}: {file_e}")

            # 4.1 保存原有格式 (仅 query, data_id)
            _append_to_json_locked(file_name_std, simple_task_info)
            # 4.2 保存新格式 (包含 response_content)
            _append_to_json_locked(file_name_detail, detailed_task_info)

        except Exception as e:
            logger.error(f"[PreFilter] Failed to save task logs: {e}")
        # -----------------------------------

        # 5. 打印日志并返回结果
        if is_good:
            logger.info(f"[PreFilter] ✅ KEEP | Query: {query[:100]}...")
            return task
        else:
            logger.warning(f"[PreFilter] ❌ DROP | Reason: Poor Quality | Query: {query}")
            return None

    def _construct_prompt(self, query: str) -> list[dict]:
        """
        构造用于判断任务质量的 Prompt。不修改 Prompt 内容。
        """
        content = f"""You are a **Data Quality Evaluator** for an AI Agent dataset. 
Your task is to determine whether a given user Query is "High Quality" and "Executable" by an advanced AI Agent.

#### Task Query:
"{query}"

#### Evaluation Criteria (Strictly Adhere to These):

1. **Intent Clarity:**
   - **ACCEPT**: The query must have a clear action (e.g., buy, move, reply, update) and a clear object (e.g., specific files, emails, or products).
   - **ACCEPT**: Moderate vagueness is acceptable if the Agent can fill in the gaps using common sense, search, or context (e.g., "find the cheapest option" or "based on highest ratings").

2. **Logical Sanity vs. Over-specification:**
   - **REJECT**: Commands with hyper-precise values that the user cannot guarantee (e.g., "Buy a fan that costs exactly $20.51").
   - **REJECT**: Commands based on blind speculation of internal system states (e.g., "If my bank balance is exactly $500, then..."). Users should ask the Agent to check the state first.
   - **ACCEPT**: Delegated constraints where the Agent handles the filtering (e.g., "Move items with under 4.2 rating from my cart to wish list").

3. **Complex Multi-step & Cross-App Reasoning:**
   - **ACCEPT**: Queries involving multi-step logic or cross-platform interaction (e.g., parsing an email to trigger an Amazon purchase, or managing files based on metadata).

4. **Privacy & Permission Boundaries:**
   - **ACCEPT**: Operations within the user's explicit or implied digital circle (e.g., "Email my parents," "Follow artists on my Spotify").

---

#### High-Quality Reference Examples:
* **File Management**: Add "YYYY-MM-DD_" prefix to all files in ~/downloads/ based on creation date and move old files to trash.
* **Cross-App Logistics**: Check a checklist sent via email and order the remaining missing items on Amazon with the highest ratings.
* **Social & Gifting**: Buy 2 identical earbuds; deliver one gift-wrapped to my parents' house and one to my home.
* **Workflow Automation**: Schedule a "gentle reminder" reply to every job application email I sent in the last 48 hours.
* **Media Management**: Follow all artists who have at least one song in my "Liked Songs" on Spotify.

---

#### Evaluation Process:
Analyze the query for logical traps, clarity of intent, and whether an Agent equipped with tools (Email, Browser, File System, Spotify, etc.) could realistically complete it.

**Response Format:**
Provide your thought process first (analyze intent, constraints, and executability), then conclude strictly in this format:
<answer>True</answer> (if valid and usable) or <answer>False</answer> (if invalid or logically flawed)."""

        return [{"role": "user", "content": content}]

    def _parse_response(self, content: str) -> bool:
        """
        解析 LLM 返回的 <answer>True/False</answer>
        """
        match = re.search(r"<answer>\s*(True|False)\s*</answer>", content, re.IGNORECASE)
        
        if match:
            result_str = match.group(1).lower()
            return result_str == "true"
        
        if "true" in content.lower() and "false" not in content.lower():
            return True
        
        return False

    def _chat_with_retry(self, messages: list[dict]) -> Optional[str]:
        """
        带重试机制的 LLM 调用。
        """
        for i in range(self._max_retries):
            try:
                # 强制指定使用 qwen3.5-plus
                response = self._llm_client.chat(
                    messages=messages,
                    model="qwen3.5-plus"
                )
                
                if isinstance(response, dict):
                    return response.get("content", "")
                elif hasattr(response, "content"):
                    return response.content
                elif isinstance(response, str):
                    return response
                
            except Exception as e:
                logger.warning(f"[PreFilter] Retry {i+1} failed due to network/API error: {e}")
                time.sleep(2 ** i)
        
        return None