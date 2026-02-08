import re
import threading
import time
from typing import Any, Optional, cast
from loguru import logger
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

from . import grader_manager

# --------------------------------------------------------------------------------
# Modified Prompt: Removed Reference Solution / Ground Truth Dependencies
# --------------------------------------------------------------------------------
USER_PROMPT = """### Role
You are an expert AI agent evaluator. Your job is to judge an agent's performance using the following inputs:

1) **User Task** — what the agent was supposed to accomplish.
2) **Agent Trajectory** — chronological steps the agent took, including actions, decisions, and outputs.

### Ground Rules
- Base your judgment strictly on the provided trajectory. Do **not** invent missing steps or assumptions.
- Rely on your own knowledge to validate the correctness of the approach and the final result.
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
- **Evaluate** both the final answer **and** the solution path based on the task requirements and your internal knowledge.
- Do not be misled by confident language—verify substance.
- There are some critic details you should check:
    - Some APIs are paginated, which is documented in the API doc. Agent must call the API multiple times to get all the data.


**Step 4 — Additional Deductions (respect the above ranges)**
- **Code Execution Errors:** Deduct for crashes, runtime errors, failed tool calls, or obvious bugs.
- **Efficiency & Conciseness:** Deduct if the trajectory is substantially roundabout, redundant, or cluttered, even if it is correct. Unnecessary or irrelevant steps are also penalized. However, additional steps taken solely to consult API documentation are acceptable.
---

## Scoring Guidelines (choose a range, then adjust within it)
**If goal achieved (must be 60-100):**
- **90-100:** Exceptional — clean, efficient; no significant issues.
- **80-89:** Strong — correct with minor inefficiencies or small issues.
- **70-79:** Good — correct but notably less efficient or with several unnecessary steps.
- **60-69:** Adequate — correct yet with significant problems in efficiency, clarity, or execution quality.

**If goal not achieved (must be 0-40):**
- **30-40:** Poor — incorrect but generally relevant with partial progress.
- **10-29:** Very poor — incorrect with major execution issues; only weak alignment to a correct path.
- **1-9:** Minimal relevant attempt — incorrect with severe problems, but some faint relevance.
- **0:** Complete failure — irrelevant approach **or** infinite repetition of irrelevant steps.

> Note on Step 2 cap: If infinite/runaway repetition is detected and steps are otherwise relevant, the **maximum** final score is **20** (within the 0-40 band).

---

## Output Format
First, provide a **detailed reasoning analysis** that references specific steps/observations (including efficiency notes and any code/error findings).
Then output a single integer score (either **0-40** or **60-100**, never 41-59) wrapped in tags:

<reward>75</reward>

---

** User Task **
{task}

** Agent Trajectory (STEP-ACTION-OBSERVATION) **
{trajs}


---
"""

USER_PROMPT_WITH_MEAN_CONSTRAINT = USER_PROMPT + """
Over the past period of time, the average score you gave to some samples was {running_mean:.4f}.
Please note that the average score must be maintained around {mean_score:.4f} (+-0.2), or you will be penalized.
"""

def steps_to_msg(steps: list[dict[str, Any]]) -> str:
    """
    Converts a list of step dictionaries into a single coherent string message.

    Args:
        steps (list[dict[str, Any]]): A list of dictionaries, each representing a step in the agent's trajectory.

    Returns:
        str: A single string that concatenates all the steps into a coherent message.
    """
    # format the trajectory
    trajectory_text = ""
    assert steps[0]['role'] == 'assistant'
    for i, msg in enumerate(steps):
        role = msg.get("role", "unknown")
        if role == 'assistant':
            block = f""">>> STEP {i//2} <<<
<|ACTION|>
{msg['content']}
<|END|>
"""
        elif role == "user":
            block = f"""<|OBSERVATION|>
{msg['content']}
<|END|>
"""
        else:
            raise ValueError("roles in trajectory must be assistant or user")
        trajectory_text += block.strip() + "\n\n"
    return trajectory_text

@grader_manager.reg("llm-binary-no-gt")
class LlmAsJudgeBinaryRewardCalculatorNoGT(RewardCalculator):
    """
    RewardCalculator that uses LLM as judge WITHOUT Reference Solution (Ground Truth).
    """
    _running_judge_mean_fast = 0.3
    _running_judge_mean_slow = 0.3

    _alpha_fast=0.9
    _alpha_slow=0.95
    _update_lock = threading.Lock()

    def __init__(self, task: Task, model_name='qwen3-235b-a22b-instruct-2507', use_mean_constraint=True):
        super().__init__(task)

        self._client = DashScopeClient(model_name=model_name)
        self._client.max_tokens=32768
        self._use_mean_constraint = use_mean_constraint

    @classmethod
    def update_running_mean(cls, new_score: float):
        with cls._update_lock:
            cls._running_judge_mean_fast = cls._alpha_fast * cls._running_judge_mean_fast + (1-cls._alpha_fast) * new_score
            cls._running_judge_mean_slow = cls._alpha_slow * cls._running_judge_mean_slow + (1-cls._alpha_slow) * new_score

    @classmethod
    def get_running_mean(cls):
        with cls._update_lock:
            return cls._running_judge_mean_fast

    @classmethod
    def get_stable_mean(cls):
        with cls._update_lock:
            return cls._running_judge_mean_slow

    def pack_message(self, trajectory: Trajectory):
        """
        Pack trajectory into a message.
        MODIFIED: Does not require ground_truth and does not include reference_trajs in the prompt.
        """
        messages = []

        assert len(trajectory.steps) >= 2 and trajectory.steps[1]['role'] == 'user', "trajectory must start with system message and then user message"
        task_query = trajectory.steps[1]['content']

        # Removed assertion for self.task.ground_truth here.

        if self._use_mean_constraint:
            content = USER_PROMPT_WITH_MEAN_CONSTRAINT.format(
                task=task_query,
                trajs=steps_to_msg(trajectory.steps[2:]),
                running_mean=self.get_running_mean(),
                mean_score=self.get_stable_mean()
                # Removed reference_trajs argument
            )
        else:
            content = USER_PROMPT.format(
                task=task_query,
                trajs=steps_to_msg(trajectory.steps[2:])
                # Removed reference_trajs argument
            )
        messages.append(
            {
                "role": "user",
                "content": content
            }
        )
        return messages

    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        x,res = cast(tuple[float,str], self._calculate_reward(trajectory, env, eject_llm_output=True))
        return {
            "score": x,
            "reason": res
        }


    def _calculate_reward(self, trajectory: Trajectory, env: EnvClient, *, eject_llm_output: bool = False):
        """
        Calculate the reward with WATCHDOG to detect hangs.
        """
        task_id = self.task.task_id if self.task and hasattr(self.task, 'task_id') else "unknown_task"
        
        # --- [Watchdog Setup] ---
        # 启动一个后台线程，如果 30秒 后主逻辑还没结束，就打印严重警告
        # 这有助于确认是否是网络死锁
        watchdog_done_event = threading.Event()
        
        def watchdog_monitor(tid):
            time.sleep(30) # 30秒超时阈值
            if not watchdog_done_event.is_set():
                logger.critical(f"[{tid}] 🚨 WATCHDOG ALERT: Judge LLM call has been stuck for 30s! This implies a NETWORK DEADLOCK or Client Hang.")
                # 注意：我们很难从这里强制杀死 requests 线程，但至少我们知道了原因
        
        wd_thread = threading.Thread(target=watchdog_monitor, args=(task_id,), daemon=True)
        wd_thread.start()
        # ------------------------

        logger.info(f"[{task_id}] 🟢 Start calling Judge LLM (Streaming Mode)...")
        start_time = time.time()

        response = ""
        chunk_count = 0
        
        try:
            # 尝试调用流式接口
            # 注意：如果 DashScopeClient 底层 requests 没有设置 timeout，这里可能会永久卡死
            stream_iterator = self._client.chat_stream_with_retry(messages=self.pack_message(trajectory), max_retries=8)
            
            for chunk in stream_iterator:
                response += chunk
                chunk_count += 1
                
                # [LOG] 收到第一个 Token
                if chunk_count == 1:
                    first_token_time = time.time()
                    logger.info(f"[{task_id}] ⚡ First token received after {first_token_time - start_time:.2f}s")
                
                # [LOG] 心跳
                if chunk_count % 50 == 0:
                    logger.debug(f"[{task_id}] ... streaming (len: {len(response)})")
        
        except Exception as e:
            logger.error(f"[{task_id}] ❌ Error during Judge LLM stream: {e}")
        finally:
            # 无论成功失败，通知看门狗退出
            watchdog_done_event.set()

        total_time = time.time() - start_time
        logger.info(f"[{task_id}] 🏁 Judge generation finished in {total_time:.2f}s. Total Length: {len(response)}")
        
        if response:
            logger.info(f"[{task_id}] Judge Reasoning & Output:\n{response}")
        
        score = 0.0
        if response:
            import re
            reward_match = re.search(r'<reward>([\d\.]+)</reward>', response.strip())
            if reward_match:
                raw_score = float(reward_match.group(1))
                score = max(0.0, min(100.0, raw_score)) / 100.0
                logger.info(f"[{task_id}] ✅ Judge Result -> Raw: {raw_score}, Normalized: {score}")
            else:
                logger.warning(f"[{task_id}] ⚠️ Could not parse score from response. Snippet: {response[:200]}...")
                try:
                    fallback_matches = re.findall(r'\b(100|[1-9]?[0-9])\b', response)
                    if fallback_matches:
                        logger.warning(f"[{task_id}] (Fallback) Potential scores found: {fallback_matches}. Returning 0.0.")
                except: pass
                score = 0.0
        else:
            logger.error(f"[{task_id}] ❌ No response content accumulated.")
            score = 0.0

        if not eject_llm_output:
            return score
        else:
            return score, response

@grader_manager.reg("llm-binary-no-gt-no_constraint")
class LlmAsJudgeBinaryRewardCalculatorNoGTNoConstraint(LlmAsJudgeBinaryRewardCalculatorNoGT):
    def __init__(self, task: Task, model_name='qwen3-235b-a22b-instruct-2507'):
        super().__init__(task, model_name, use_mean_constraint=False)