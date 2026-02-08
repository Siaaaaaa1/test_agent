import re
import time
import json
import os
import threading
from typing import Any, Optional, cast, Generator, List, Dict, Tuple
from loguru import logger
import requests

# 尝试加载 dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- 内部模块引入 ---
from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from . import grader_manager

# ==============================================================================
# SECTION 1: Embedded LLM Client (Azure GPT-5 Mini Specific)
# ==============================================================================

class _EmbeddedAzureClient:
    """
    内置的 Azure Client，专门用于 Reward Calculation。
    包含：速率限制、并发控制、以及增强的调试日志。
    """
    def __init__(self):
        self.model_name = "azure-gpt-5-mini" 
        self.base_url = os.getenv("AZURE_PROXY_URL") or "http://ichatproxy.devops.weread.woa.com"
        # 默认并发限制
        self._semaphore = threading.BoundedSemaphore(5) 
        self._rate_limit_lock = threading.Lock()
        self._request_timestamps = []
        self._max_rpm = 40 
        
        self.headers = {"Content-Type": "application/json"}

    def _wait_for_rate_limit(self):
        window_duration = 60.0
        while True:
            with self._rate_limit_lock:
                now = time.time()
                self._request_timestamps = [t for t in self._request_timestamps if now - t < window_duration]
                if len(self._request_timestamps) < self._max_rpm:
                    self._request_timestamps.append(now)
                    return
                wait_time = window_duration - (now - self._request_timestamps[0])
            if wait_time > 0:
                time.sleep(wait_time + 0.05)

    def chat_stream_with_retry(self, messages: list[dict[str, str]], max_retries: int = 3) -> Generator[str, None, None]:
        """流式对话，带重试机制"""
        last_error = None
        for attempt in range(max_retries):
            try:
                yield from self._do_stream_request(messages)
                return 
            except Exception as e:
                last_error = e
                logger.warning(f"[_EmbeddedClient] Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        # 重试耗尽后，抛出异常以便上层捕获
        logger.error(f"[_EmbeddedClient] All {max_retries} attempts failed.")
        raise last_error or Exception("Unknown error after retries")

    def _do_stream_request(self, messages: list[dict[str, str]]) -> Generator[str, None, None]:
        """实际执行请求，包含锁和超时设置"""
        url = f"{self.base_url.rstrip('/')}/api/chat_completions?source=emoji_agent_research"
        params = {
            "model": self.model_name,
            "messages": messages,
            "stream": True
        }
        no_proxy = {"http": None, "https": None}

        # 1. 获取信号量
        logger.info(f"🚦 [Judge-Client] Semaphore Status: {self._semaphore._value}/5 slots free. Thread {threading.get_ident()} trying to enter...")
        
        with self._semaphore:
            self._wait_for_rate_limit()
            
            # 2. 发起请求
            start_req = time.time()
            # 注意：timeout=(连接超时, 读取超时)
            response = requests.post(
                url, headers=self.headers, json=params, stream=True,
                timeout=(10, 120), proxies=no_proxy
            )
            
            # [DEBUG] 打印响应头信息
            ct = response.headers.get('Content-Type', '')
            logger.info(f"📡 [Judge-Net] Response Code: {response.status_code}, Content-Type: {ct}, Latency: {time.time()-start_req:.2f}s")

            if not response.ok:
                try:
                    err_msg = response.text[:200]
                except: err_msg = "Cannot read text"
                logger.error(f"API Error: {response.status_code} - {err_msg}")
                response.raise_for_status()

            # 3. 解析流式响应
            has_valid_data = False
            line_count = 0
            
            for line in response.iter_lines():
                if line:
                    line_count += 1
                    line_str = line.decode('utf-8').strip()
                    if not line_str: continue

                    # ---------------------------------------------------------
                    # [修复核心] 兼容两种流式格式：
                    # 1. 标准 SSE: "data: {...}"
                    # 2. 裸 JSON (NDJSON): "{...}"
                    # ---------------------------------------------------------
                    
                    json_str = None
                    if line_str.startswith('data: '):
                        temp = line_str[6:]
                        if temp == '[DONE]': break
                        json_str = temp
                    elif line_str.startswith('{') and line_str.endswith('}'):
                        # 捕获直接返回 JSON 的情况
                        json_str = line_str
                    
                    if json_str:
                        try:
                            chunk = json.loads(json_str)
                            # 提取 content
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content: 
                                    has_valid_data = True
                                    yield content
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ [Judge-Format] JSON decode failed for line: '{line_str[:50]}...'")
                            continue
                    else:
                        # 确实无法识别的格式，且不是空行
                        if line_count <= 3:
                            logger.warning(f"⚠️ [Judge-Format] Ignored line: '{line_str[:100]}...'")

            if line_count > 0 and not has_valid_data:
                logger.error(f"❌ [Judge-Logic] Received {line_count} lines but found NO valid content.")


# ==============================================================================
# SECTION 2: Prompts & Helper Functions (Updated with Reference Solution)
# ==============================================================================

USER_PROMPT = """### Role
You are an expert AI agent evaluator. Your job is to judge an agent's performance using the following inputs:

1) **User Task** — what the agent was supposed to accomplish.
2) **Reference Solution** — a correct approach/outcome to compare against (other valid solutions may exist).
3) **Agent Trajectory** — chronological steps the agent took, including actions, decisions, and outputs.

### Ground Rules
- Base your judgment strictly on the provided trajectory. Do **not** invent missing steps or assumptions.
- Treat the Reference Solution as an oracle for correctness checks and efficiency comparison, while allowing alternative correct methods.
- When citing issues, reference concrete steps or observations from the trajectory.
- Be deterministic: follow the procedure below and the scoring constraints exactly.
- “Infinite or runaway repetition” means the agent repeats essentially the same step/loop ≥3 times with no new information or progress.

---

## Evaluation Procedure

**Step 1 — Relevance Gate (0 or proceed)**
- Determine if the trajectory's steps are **materially related** to the User Task.
- If the approach is wholly unrelated → **score = 0** and stop.
- Otherwise, continue.

**Step 2 — Repetition Penalty Gate**
- Check for infinite/runaway repetition of identical or near-identical steps.
  - If such repetition exists:
    - If steps are otherwise relevant → **final score must be ≤ 20**.
    - If steps are irrelevant → **score = 0**.
- If no infinite repetition, continue.

**Step 3 — Goal Achievement (Critical Binary Check)**
- Examine **all** steps and the final result to decide if the task is actually completed **correctly**.
- **Compare** both the final answer **and** the solution path against the Reference Solution to validate correctness. Note that the Reference Solution is not the only correct solution, other equivalent solution should also be considered correct.
- Do not be misled by confident language—verify substance.
- There are some critic details you should check:
    - Some APIs are paginated, which is documented in the API doc. Agent must call the API multiple times to get all the data.


**Step 4 — Additional Deductions (respect the above ranges)**
- **Code Execution Errors:** Deduct for crashes, runtime errors, failed tool calls, or obvious bugs.
- **Efficiency & Conciseness vs. Reference:** Deduct if the trajectory is substantially more roundabout, redundant, or cluttered than the reference solution, even if it is correct. Unnecessary or irrelevant steps are also penalized. However, additional steps taken solely to consult API documentation are acceptable.
---

## Scoring Guidelines (choose a range, then adjust within it)
**If goal achieved (must be 60-100):**
- **90-100:** Exceptional — clean, efficient, equal/better than reference; no significant issues.
- **80-89:** Strong — correct with minor inefficiencies or small issues vs. the reference.
- **70-79:** Good — correct but notably less efficient or with several unnecessary steps.
- **60-69:** Adequate — correct yet with significant problems in efficiency, clarity, or execution quality.

**If goal not achieved (must be 0-40):**
- **30-40:** Poor — incorrect but generally relevant with partial progress aligned to the reference path.
- **10-29:** Very poor — incorrect with major execution issues; only weak alignment to a correct path.
- **1-9:** Minimal relevant attempt — incorrect with severe problems, but some faint relevance.
- **0:** Complete failure — irrelevant approach **or** infinite repetition of irrelevant steps.

> Note on Step 2 cap: If infinite/runaway repetition is detected and steps are otherwise relevant, the **maximum** final score is **20** (within the 0-40 band).

---

## Output Format
First, provide a **detailed reasoning analysis** that references specific steps/observations and compares against the Reference Solution (including efficiency notes and any code/error findings).
Then output a single integer score (either **0-40** or **60-100**, never 41-59) wrapped in tags:

<reward>75</reward>

---

** User Task **
{task}

** Reference Solution **
{reference_trajs}

** Agent Trajectory (STEP-ACTION-OBSERVATION) **
{trajs}
"""

def steps_to_msg(steps: list[dict[str, Any]]) -> str:
    """
    Converts a list of step dictionaries into a single coherent string message.
    """
    trajectory_text = ""
    if not steps or not isinstance(steps, list):
        return "No steps provided."

    for i, msg in enumerate(steps):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == 'assistant':
            block = f""">>> STEP {i//2} <<<
<|ACTION|>
{content}
<|END|>
"""
        elif role == "user":
            block = f"""<|OBSERVATION|>
{content}
<|END|>
"""
        else:
            block = f"""<|{role.upper()}|>
{content}
<|END|>
"""
        trajectory_text += block.strip() + "\n\n"
    return trajectory_text

# ==============================================================================
# SECTION 3: Unified Reward Calculator Class (Modified for GT)
# ==============================================================================

@grader_manager.reg("llm-binary-gt-custom")
class IntegratedRewardCalculator_GT(RewardCalculator):
    def __init__(self, task: Task):
        super().__init__(task)
        self._client = _EmbeddedAzureClient()

    def pack_message(self, trajectory: Trajectory) -> List[Dict]:
        if not trajectory.steps or len(trajectory.steps) < 2:
            task_query = "Unknown Task"
            traj_text = "No steps."
        else:
            # 假设 steps[1] 是 User Task
            task_query = trajectory.steps[1].get('content', '')
            # steps[2:] 是实际交互
            traj_text = steps_to_msg(trajectory.steps[2:])

        # 获取 Ground Truth (Reference Solution)
        # 如果 task 中没有 ground_truth，提供一个占位符
        gt_text = getattr(self.task, 'ground_truth', None) or "[No solution provided, please judge the task by yourself]"

        # 使用包含 Reference Solution 的新 Prompt 格式
        content = USER_PROMPT.format(
            task=task_query, 
            reference_trajs=gt_text,
            trajs=traj_text
        )
        return [{"role": "user", "content": content}]

    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        score, reason = self._calculate_reward_internal(trajectory)
        return {
            "score": score,
            "reason": reason
        }

    def _calculate_reward_internal(self, trajectory: Trajectory) -> Tuple[float, str]:
        task_id = getattr(self.task, 'task_id', 'unknown_task')
        
        # --- Watchdog (防止网络死锁) ---
        watchdog_done = threading.Event()
        def watchdog():
            time.sleep(120) 
            if not watchdog_done.is_set():
                logger.critical(f"[{task_id}] 🚨 WATCHDOG ALERT: Reward calculation stuck > 120s! Network congestion likely.")
        
        wd_thread = threading.Thread(target=watchdog, daemon=True)
        wd_thread.start()
        # -----------------------------

        logger.info(f"[{task_id}] 🟢 Start calculating reward (Azure-GPT-5-Mini with GT)...")
        start_time = time.time()
        response_text = ""
        
        try:
            stream = self._client.chat_stream_with_retry(
                messages=self.pack_message(trajectory), 
                max_retries=3
            )
            
            first_token_seen = False
            chunk_count = 0
            
            for chunk in stream:
                if not first_token_seen:
                    logger.info(f"[{task_id}] ⚡ First token received after {time.time()-start_time:.2f}s")
                    first_token_seen = True
                
                response_text += chunk
                chunk_count += 1
                
                if chunk_count % 100 == 0:
                    logger.debug(f"[{task_id}] ... receiving reward stream (len: {len(response_text)})")
                
        except Exception as e:
            logger.error(f"[{task_id}] ❌ Reward calculation failed (Stream Error): {e}")
            # 如果是异常，将错误信息放入 Reason，分数记为 0
            return 0.0, f"Error: {str(e)}"
        finally:
            watchdog_done.set()

        total_time = time.time() - start_time
        logger.info(f"[{task_id}] 🏁 Judge finished in {total_time:.2f}s. Response Len: {len(response_text)}")

        # --- 解析分数 ---
        score = 0.0
        if response_text:
            logger.info(f"[{task_id}] Judge Reasoning:\n{response_text}")
            
            match = re.search(r'<reward>([\d\.]+)</reward>', response_text.strip())
            if match:
                raw = float(match.group(1))
                score = max(0.0, min(100.0, raw)) / 100.0
                logger.info(f"[{task_id}] ✅ Score parsed: {score} (Raw: {raw})")
            else:
                logger.warning(f"[{task_id}] ⚠️ No score tag found in response.")
                # 简单兜底
                try:
                    fallback = re.findall(r'\b(100|[1-9]?[0-9])\b', response_text)
                    if fallback:
                        logger.warning(f"[{task_id}] (Fallback) Potential score: {fallback[-1]}. Returning 0.0 for safety.")
                except: pass
        else:
            logger.error(f"[{task_id}] ❌ Empty response from Judge.")

        return score, response_text