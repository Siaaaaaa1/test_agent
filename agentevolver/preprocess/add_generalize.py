import json
import os
import time
import threading
from typing import Any, Optional
from loguru import logger
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================
DASHSCOPE_API_KEY = "sk-25678a0b18d24afa86d3185f736fd886"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"

INPUT_PATH = "/Users/sianeko/vscode/AgentEvolver/agentevolver/preprocess/output/appworld_tool_manual.json"
OUTPUT_PATH = "/Users/sianeko/vscode/AgentEvolver/agentevolver/preprocess/output/appworld_tool_manual_with_generality.json"
REASON_OUTPUT_PATH = "/Users/sianeko/vscode/AgentEvolver/agentevolver/preprocess/output/generality_reasons.json"

MAX_WORKERS = 8       # 并行线程数
SAVE_INTERVAL = 10    # 每完成 10 个 API 保存一次文件
# ===========================================

class LlmException(Exception):
    def __init__(self, typ: str):
        self._type = typ
    @property
    def typ(self):
        return self._type

class DashScopeClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen-plus", 
                 temperature: float = 0.7, max_tokens: int = 2048, base_url: Optional[str] = None):
        raw_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not raw_key:
            raise ValueError("API key is required.")
        self.api_key = raw_key.strip().strip('"').strip("'")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(self, messages: list[dict[str, str]], **kwargs) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }
        response = requests.post(url, headers=self.headers, json=params, timeout=60)
        if not response.ok:
            response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def chat_with_retry(self, messages: list[dict[str, str]], max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                return self.chat_completion(messages)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e
        return ""

class GeneralityEvaluator:
    def __init__(self, client: DashScopeClient):
        self.client = client
        self.prompt_template = """
You are an expert AI Agent Architect specializing in Dataset Curation for Cross-App Automation.
Your task is to evaluate the **"Generality"** of a specific API function.

**"Generality"** defines how likely this API allows an agent to bridge two different apps.

### Evaluation Rubric:
1. **Very High:** Universal connectors (Search, Email, File, SMS, Text Copy).
2. **High:** Core entity bridges (Money, Products, Tasks, Songs).
3. **Medium:** Domain-specific metadata (Receipts, Reviews, Contact Details).
4. **Low:** Intra-app utilities (Volume, Sort, Archive, Delete).
5. **Very Low:** Admin/Setup (Login, Logout, Verify).

### API to Evaluate:
App: {app_name} | API: {api_name}
Description: {description}
Parameters: {parameters}
Returns: {returns}

### Output Format (Strict JSON):
{{
    "generality_level": "Very High/High/Medium/Low/Very Low",
    "reason": "explanation"
}}
"""

    def evaluate(self, app_name: str, api_name: str, api_info: dict) -> dict:
        prompt = self.prompt_template.format(
            app_name=app_name,
            api_name=api_name,
            description=api_info.get("description", ""),
            parameters=json.dumps(api_info.get("parameters", []), ensure_ascii=False),
            returns=json.dumps(api_info.get("returns", {}), ensure_ascii=False)
        )
        try:
            res = self.client.chat_with_retry([{"role": "user", "content": prompt}])
            if "```json" in res:
                res = res.split("```json")[1].split("```")[0]
            elif "```" in res:
                res = res.split("```")[1].split("```")[0]
            return json.loads(res.strip())
        except Exception as e:
            return {"generality_level": "Error", "reason": str(e)}

def save_now(data, reasons, lock):
    """线程安全的保存函数"""
    with lock:
        try:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            with open(REASON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(reasons, f, indent=4, ensure_ascii=False)
            logger.debug("--- Progress Saved to Disk ---")
        except Exception as e:
            logger.error(f"Save failed: {e}")

def main():
    # 1. 初始化
    client = DashScopeClient(api_key=DASHSCOPE_API_KEY)
    evaluator = GeneralityEvaluator(client)
    file_lock = threading.Lock()

    # 2. 读取文件
    if not os.path.exists(INPUT_PATH):
        logger.error(f"File not found: {INPUT_PATH}")
        return
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 加载已有的理由（如果存在），支持断点续传
    reasons_data = {}
    if os.path.exists(REASON_OUTPUT_PATH):
        try:
            with open(REASON_OUTPUT_PATH, 'r', encoding='utf-8') as f:
                reasons_data = json.load(f)
        except: pass

    # 3. 展平任务列表，方便并行处理
    task_list = []
    for app_id, app_content in data.items():
        if "apis" not in app_content: continue
        for api_id, api_info in app_content["apis"].items():
            # 检查是否已经处理过
            if "generality_assessment" in api_info:
                continue
            task_list.append((app_id, api_id, api_info))

    if not task_list:
        logger.info("No new APIs to process.")
        return

    logger.info(f"Starting parallel processing for {len(task_list)} APIs with {MAX_WORKERS} threads...")

    # 4. 执行多线程任务
    completed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_api = {
            executor.submit(evaluator.evaluate, app_id, api_id, api_info): (app_id, api_id, api_info) 
            for app_id, api_id, api_info in task_list
        }

        with tqdm(total=len(task_list), desc="Total Progress") as pbar:
            for future in as_completed(future_to_api):
                app_id, api_id, api_info = future_to_api[future]
                try:
                    assessment = future.result()
                    
                    # 更新内存中的数据（加锁保护）
                    with file_lock:
                        api_info["generality_assessment"] = assessment
                        reasons_data[f"{app_id}.{api_id}"] = assessment
                    
                    completed_count += 1
                    
                    # 阶段性保存
                    if completed_count % SAVE_INTERVAL == 0:
                        save_now(data, reasons_data, file_lock)
                        
                except Exception as e:
                    logger.error(f"Unexpected error for {api_id}: {e}")
                finally:
                    pbar.update(1)

    # 5. 最终保存
    save_now(data, reasons_data, file_lock)
    logger.success(f"Task finished! Evaluated {completed_count} new APIs.")

if __name__ == "__main__":
    main()