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
# SECTION 2: Prompts & Helper Functions
# ==============================================================================

USER_PROMPT = """### Role
You are an expert AI agent evaluator. Your job is to judge an agent's performance using the following inputs:

1) **User Task** — what the agent was supposed to accomplish.
2) **Agent Trajectory** — chronological steps the agent took, including actions, decisions, and outputs.

### Ground Rules
- **Strictly Verify Final Content**: You must judge whether the **final submitted content** (e.g., specific numbers, code, file contents, or answers) is factually correct and fully addresses the User Task. Do not be fooled by a polite closing statement if the actual data is wrong.
- **Validate the Path**: You must ensure the intermediate steps **actually** completed the task. The agent must not "hallucinate" success; the path of actions must logically and truthfully lead to the final result.
- Base your judgment strictly on the provided trajectory. Do **not** invent missing steps or assumptions.
- Be deterministic: follow the procedure below and the scoring constraints exactly.

---

## Evaluation Procedure
1. **Relevance Gate**: If the approach is wholly unrelated → **score = 0**.
2. **Repetition Penalty**: If infinite/runaway repetition exists → **max score = 20**.
3. **Path Verification**: Examine the intermediate steps. Did the agent actually execute the necessary tools/APIs to solve the problem? Did it skip essential logic or fake the execution?
4. **Result Verification**: Examine the final submission. Is the final answer/output **correct** based on the executed steps? Does it directly satisfy the User Task requirements?
5. **Deductions**: Deduct for execution errors, inefficiency, or roundabout steps.

## Scoring Guidelines
**If goal achieved (must be 60-100):**
*CRITICAL: Only assign this range if the final answer is correct AND the path validates it.*
- **90-100:** Exceptional — clean path, perfect result.
- **80-89:** Strong — correct result, path has minor inefficiencies.
- **70-79:** Good — correct result, but path is notably inefficient.
- **60-69:** Adequate — correct result, but path had significant issues (e.g., recovered from many errors).

**If goal not achieved (must be 0-40):**
*Assign this range if the final answer is wrong, OR if the path does not support the answer (hallucinated success).*
- **30-40:** Poor — incorrect result, but approach was generally relevant.
- **10-29:** Very poor — incorrect result with major execution issues.
- **0-9:** Failure — irrelevant, infinite repetition, or fake success.

## Output Format
First, provide a **detailed reasoning analysis**. Explicitly state:
1. Whether the intermediate path truly completed the task.
2. Whether the final submitted content is correct.

Then output a single integer score (either **0-40** or **60-100**, never 41-59) wrapped in tags:

<reward>75</reward>

---

** User Task **
{task}

** Agent Trajectory (STEP-ACTION-OBSERVATION) **
{trajs}
"""

def steps_to_msg(steps: list[dict[str, Any]]) -> str:
    """
    将字典格式的步骤转化为纯文本格式供裁判分析。
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
# SECTION 3: Unified Reward Calculator Class
# ==============================================================================

@grader_manager.reg("llm-binary-no-gt-no_constraint")
class IntegratedRewardCalculator(RewardCalculator):
    def __init__(self, task: Task):
        super().__init__(task)
        self._client = DashScopeClient()

    def pack_message(self, trajectory: Trajectory) -> List[Dict]:
        """将轨迹打包进 LLM 裁判的 Prompt。"""
        if not trajectory.steps or len(trajectory.steps) < 2:
            task_query = "Unknown Task"
            traj_text = "No steps."
        else:
            task_query = trajectory.steps[1].get('content', '')
            traj_text = steps_to_msg(trajectory.steps[2:])

        content = USER_PROMPT.format(task=task_query, trajs=traj_text)
        return [{"role": "user", "content": content}]

    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        score, reason = self._calculate_reward_internal(trajectory)
        return {
            "score": score,
            "reason": reason
        }

    def _calculate_reward_internal(self, trajectory: Trajectory) -> Tuple[float, str]:
        """
        [修复] 取消了不稳定的流式读取机制，改为标准同步请求。
        优化了 Watchdog 的资源回收逻辑，避免僵尸线程堆积。
        """
        task_id = getattr(self.task, 'task_id', 'unknown_task')
        
        # --- Watchdog 防死锁 (改良版) ---
        watchdog_done = threading.Event()
        def watchdog():
            # [核心修复] 使用 wait 替代 sleep，一旦主线程发出信号立刻结束守护，不占用资源
            is_timeout = not watchdog_done.wait(timeout=120)
            if is_timeout:
                logger.critical(f"[{task_id}] 🚨 WATCHDOG ALERT: Reward calculation stuck > 120s! Network congestion likely.")
        
        wd_thread = threading.Thread(target=watchdog, daemon=True)
        wd_thread.start()
        # -----------------------------

        logger.info(f"[{task_id}] 🟢 Start calculating reward (DashScope qwen3.5-plus)...")
        start_time = time.time()
        response_text = ""
        
        try:
            # [核心修复] 作为后台裁判系统，无需使用流式 API，直接使用阻塞重试获取完整回复，极大降低解析错误概率
            response = self._client.chat_with_retry(
                messages=self.pack_message(trajectory), 
                max_retries=3
            )
            
            # 兼容返回值格式
            if isinstance(response, dict):
                response_text = response.get("content", "")
            elif hasattr(response, "content"):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
                
        except Exception as e:
            logger.error(f"[{task_id}] ❌ Reward calculation failed (Network Error): {e}")
            return 0.0, f"Error: {str(e)}"
        finally:
            # 发出事件通知，秒杀挂起的 Watchdog 线程
            watchdog_done.set()

        total_time = time.time() - start_time
        logger.info(f"[{task_id}] 🏁 Judge finished in {total_time:.2f}s. Response Len: {len(response_text)}")

        # --- 解析分数 ---
        score = 0.0
        if response_text:
            logger.info(f"[{task_id}] Judge Reasoning:\n{response_text}")
            
            # 优先精准正则匹配标签内的分值
            match = re.search(r'<reward>([\d\.]+)</reward>', response_text.strip())
            if match:
                raw = float(match.group(1))
                score = max(0.0, min(100.0, raw)) / 100.0
                logger.info(f"[{task_id}] ✅ Score parsed: {score} (Raw: {raw})")
            else:
                logger.warning(f"[{task_id}] ⚠️ No score tag found in response.")
                # 兜底：如果模型忘了输出标签，强行找最后出现的数字
                try:
                    fallback = re.findall(r'\b(100|[1-9]?[0-9])\b', response_text)
                    if fallback:
                        logger.warning(f"[{task_id}] (Fallback) Potential score: {fallback[-1]}. Returning 0.0 for safety.")
                except: pass
        else:
            logger.error(f"[{task_id}] ❌ Empty response from Judge.")

        return score, response_text