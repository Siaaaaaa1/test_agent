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
    def __init__(self, llm_client: LlmClient, num_threads: int = 20, **kwargs):
        """
        初始化预过滤器。
        
        Args:
            llm_client (LlmClient): 用于判断任务质量的 LLM 客户端。
            num_threads (int): 并发线程数。修改为20。
        """
        self._llm_client = llm_client
        self._num_threads = num_threads
        
        # 简单的重试配置 (网络问题重试5次)
        self._max_retries = 5

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
        同时将结果分别存入：
        1. 标准日志 (tasks_keep/drop_pre_filter.json)
        2. 详细响应日志 (tasks_keep/drop_pre_filter_with_response.json)
        """
        import os
        import json
        import fcntl  # 必须导入 fcntl 以处理并发写入锁

        query = task.query
        
        # 1. 构造 Prompt
        prompt = self._construct_prompt(query)
        
        # 2. 调用 LLM (带重试，路径评估仅使用默认的 qwen3.5-plus)
        response_content = self._chat_with_retry(prompt)
        
        if not response_content:
            logger.warning(f"[PreFilter] ⚠️ LLM No Response | Task ID: {task.task_id}")
            return None

        # 3. 解析结果
        is_good = self._parse_response(response_content)
        
        # --- 数据准备 ---
        # 基础信息 (用于原有的 json)
        simple_task_info = {
            "query": task.query,
            "data_id": task.metadata.get('data_id', 'unknown') if task.metadata else 'unknown'
        }

        # 详细信息 (用于新的 json，增加了 response_content)
        detailed_task_info = {
            "query": task.query,
            "data_id": task.metadata.get('data_id', 'unknown') if task.metadata else 'unknown',
            "response_content": response_content  # <--- 新增字段
        }

        # --- 保存逻辑 ---
        try:
            base_dir = "./tmp"
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)

            # 定义两个文件名：原有文件 和 新的详细文件
            if is_good:
                file_name_std = "tasks_keep_pre_filter.json"
                file_name_detail = "tasks_keep_pre_filter_with_response.json" # 新文件
            else:
                file_name_std = "tasks_drop_pre_filter.json"
                file_name_detail = "tasks_drop_pre_filter_with_response.json" # 新文件

            # 定义一个内部函数来处理加锁写入，避免代码重复
            def _append_to_json_locked(filename, data_dict):
                full_path = os.path.join(base_dir, filename)
                with open(full_path, 'a+', encoding='utf-8') as f:
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
                        
                        # 读取现有内容
                        f.seek(0)
                        content = f.read()
                        data_list = []
                        if content:
                            try:
                                data_list = json.loads(content)
                            except json.JSONDecodeError:
                                data_list = []
                        
                        # 追加新数据
                        data_list.append(data_dict)
                        
                        # 覆写文件
                        f.seek(0)
                        f.truncate()
                        json.dump(data_list, f, ensure_ascii=False, indent=4)
                        
                    except Exception as file_e:
                        logger.error(f"[PreFilter] File write error for {filename}: {file_e}")
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁

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
        构造用于判断任务质量的 Prompt。
        """
        content = f"""You are a Data Quality Evaluator for an AI Agent dataset.
Task Query: "{query}"
Evaluation Criteria:
The query must express a clear goal. Moderate vagueness is acceptable (e.g., not specifying a brand or model) as long as the Agent can supplement those details using common sense or search capabilities.
ACCEPT (Clear Intent): Commands with a clear action and object (e.g., "Buy a fan," "Book me a flight to London").
REJECT (Logical Flaws/Over-specification): Commands that include precise values or prerequisites that the user cannot guarantee, which would likely lead to execution failure.
Unacceptable: "Buy a fan that costs exactly $20.50" (Price fluctuations make this high-risk).
Unacceptable: "If my current balance is $100, then pay my phone bill" (The user should not speculate on internal states that only the Agent can verify).
ACCEPT (Delegated Commands): Queries where the user sets a goal and lets the Agent handle the filtering/optimization.
Example: "Help me find a desktop fan under $50.", "Update my username to 'Alex'."
Is this a valid and usable query? Please think first, then reply with the result strictly in this format: <answer>True</answer> or <answer>False</answer>"""

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
        带重试机制的 LLM 调用。
        网络问题最多重试5次。由于这是路径评估/质量过滤任务，
        仅使用一次默认的 qwen3.5-plus 即可，不进行任务推理失败的模型降级切换。
        """
        for i in range(self._max_retries):
            try:
                # 调用 LLM，强制指定使用 qwen3.5-plus
                response = self._llm_client.chat(
                    messages=messages,
                    model="qwen3.5-plus"
                )
                
                # 兼容不同 Client 返回格式 (dict 或 str 或 object)
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