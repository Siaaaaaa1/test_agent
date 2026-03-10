import json
import os
import threading
from loguru import logger
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from agentevolver.client.llm_client import DashScopeClient

# ================= 配置区域 =================
MODEL_NAME = "qwen3.5-plus"

INPUT_PATH = "/Users/sianeko/vscode/AgentEvolver/agentevolver/preprocess/output/appworld_tool_manual.json"
OUTPUT_PATH = "/Users/sianeko/vscode/AgentEvolver/agentevolver/preprocess/output/appworld_tool_manual_with_generality.json"
REASON_OUTPUT_PATH = "/Users/sianeko/vscode/AgentEvolver/agentevolver/preprocess/output/generality_reasons.json"

MAX_WORKERS = 8       # 并行线程数
SAVE_INTERVAL = 10    # 每完成 10 个 API 保存一次文件
# ===========================================

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
    client = DashScopeClient(model_name=MODEL_NAME)
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