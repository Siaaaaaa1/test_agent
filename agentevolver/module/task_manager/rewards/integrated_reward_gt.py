import re
import time
import threading
from typing import Any, List, Dict, Tuple
from loguru import logger

# --- 内部模块引入 ---
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from . import grader_manager


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
        self._client = DashScopeClient()

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

        logger.info(f"[{task_id}] 🟢 Start calculating reward (DashScope qwen3.5-plus with GT)...")
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