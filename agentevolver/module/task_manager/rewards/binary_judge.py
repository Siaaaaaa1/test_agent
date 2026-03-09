import re
import threading
from typing import Any, Optional, cast
from loguru import logger
from agentevolver.client.env_client import EnvClient

# [修改] 只保留新创建的 DashScopeClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

from . import grader_manager

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


---
"""

USER_PROMPT_WITH_MEAN_CONSTRAINT=USER_PROMPT+"""
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

@grader_manager.reg("llm-binary-gt")
class LlmAsJudgeBinaryRewardCalculatorWithGT(RewardCalculator):
    """
    RewardCalculator that uses LLM as judge.
    """
    _running_judge_mean_fast = 0.3
    _running_judge_mean_slow = 0.3

    _alpha_fast=0.9
    _alpha_slow=0.95
    _update_lock = threading.Lock()

    def __init__(self, task: Task, model_name='qwen3.5-plus', use_mean_constraint=True):
        super().__init__(task)
        # [修改] 使用 DashScopeClient 替换 Mix_DashScopeClient，并支持自定义 max_tokens
        self._client = DashScopeClient(model_name=model_name, max_tokens=32768)
        self._use_mean_constraint = use_mean_constraint

    @classmethod
    def update_running_mean(cls, new_score: float):
        with cls._update_lock:
            cls._running_judge_mean_fast = cls._alpha_fast * cls._running_judge_mean_fast + (1-cls._alpha_fast) * new_score  # ⭐ Update the fast running mean
            cls._running_judge_mean_slow = cls._alpha_slow * cls._running_judge_mean_slow + (1-cls._alpha_slow) * new_score  # ⭐ Update the slow running mean

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

        Args:
            trajectory (Trajectory): The trajectory to pack.

        Returns:
            list: A list of messages prepared for the language model.
        """
        # [DEBUG] 标记开始打包消息
        logger.info(f"[{threading.get_ident()}] Start packing message...")

        messages = []

        assert len(trajectory.steps) >= 2 and trajectory.steps[1]['role'] == 'user', "trajectory must start with system message and then user message"
        task_query = trajectory.steps[1]['content']

        # the ground-truth must be provided, for now.
        assert self.task.ground_truth is not None, "ground truth must not be None for synthetic task"
        
        if self._use_mean_constraint:
            # [DEBUG] 标记开始获取锁，用于检测是否死锁
            logger.debug(f"[{threading.get_ident()}] Attempting to acquire locks for mean scores...")
            
            # 分别获取，以便确认是哪一步卡住
            running_mean = self.get_running_mean()
            stable_mean = self.get_stable_mean()
            
            logger.debug(f"[{threading.get_ident()}] Locks acquired. Means: running={running_mean}, stable={stable_mean}")

            content=USER_PROMPT_WITH_MEAN_CONSTRAINT.format(
                task=task_query,
                trajs=steps_to_msg(trajectory.steps[2:]),
                running_mean=running_mean,
                mean_score=stable_mean,
                reference_trajs=self.task.ground_truth or "[No solution provided, please judge the task by yourself]"
            )
        else:
            content=USER_PROMPT.format(
                task=task_query,
                trajs=steps_to_msg(trajectory.steps[2:]),
                reference_trajs=self.task.ground_truth or "[No solution provided, please judge the task by yourself]"
            )
        messages.append(
            {
                "role": "user",
                "content": content  # ⭐ Construct the content of the message
            }
        )
        
        # [DEBUG] 打包完成
        logger.info(f"[{threading.get_ident()}] Message packing finished.")
        return messages

    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        x,res = cast(tuple[float,str], self._calculate_reward(trajectory, env, eject_llm_output=True))
        return {
            "score": x,
            "reason": res
        }


    def _calculate_reward(self, trajectory: Trajectory, env: EnvClient, *, eject_llm_output: bool = False):
        """
        Calculate the reward for a given trajectory in a specific environment by querying an LLM.

        Args:
            trajectory (Trajectory): The trajectory for which the reward is to be calculated.
            env (EnvClient): The environment in which the trajectory was executed.
            eject_llm_output (bool, optional): If True, the function will return both the score and the LLM's response. Defaults to False.

        Returns:
            float or tuple: The calculated reward score, or a tuple containing the score and the LLM's response if `eject_llm_output` is True.
        """
        # [DEBUG] 进入函数
        logger.info(f"[{threading.get_ident()}] Entering _calculate_reward...")
        
        try:
            messages = self.pack_message(trajectory)
        except Exception as e:
            logger.error(f"[{threading.get_ident()}] Error in pack_message: {e}")
            raise e

        response = ""
        # [DEBUG] 准备发起 LLM 请求
        logger.info(f"[{threading.get_ident()}] Starting LLM chat stream (max_retries=64)...")
        
        try:
            chunk_count = 0
            # [DEBUG] 进入流式循环
            for chunk in self._client.chat_stream_with_retry(messages=messages, max_retries=64):
                response += chunk
                chunk_count += 1
                # 仅打印前几个 chunk 和每隔一定数量的 chunk，证明流还在动
                if chunk_count == 1 or chunk_count % 20 == 0:
                    logger.debug(f"[{threading.get_ident()}] Receiving chunk #{chunk_count}, current len: {len(response)}")
            
            # [DEBUG] 请求循环结束
            logger.info(f"[{threading.get_ident()}] LLM chat stream finished. Total response length: {len(response)}")
            
        except Exception as e:
            logger.error(f"[{threading.get_ident()}] Exception during LLM chat stream: {e}")
            # 记录完整堆栈以备排查
            import traceback
            logger.error(traceback.format_exc())
            
        if response:
            # [DEBUG] 开始正则解析
            logger.debug(f"[{threading.get_ident()}] Parsing response for reward...")
            reward_match = re.search(r'<reward>([\d\.]+)</reward>', response.strip())
            if reward_match:
                score = float(reward_match.group(1))
                score = max(0.0, min(100.0, score)) / 100.0  # ⭐ Normalize the score to a range of [0, 1]
                logger.info(f"[{threading.get_ident()}] Score parsed successfully: {score}")
            else:
                logger.warning(f"[{threading.get_ident()}] Could not parse score from response. Response prefix: {response[:100]}...")
                score = 0.0
        else:
            logger.error(f"[{threading.get_ident()}] No response from evaluation API (response is empty)")
            score = 0.0

        if not eject_llm_output:
            return score
        else:
            return score, response

@grader_manager.reg("llm-binary-gt-no_constraint")
class LlmAsJudgeBinaryRewardCalculatorWithGTNoConstraint(LlmAsJudgeBinaryRewardCalculatorWithGT):
    # [修改] 默认模型改为 qwen3.5-plus
    def __init__(self, task: Task, model_name='qwen3.5-plus'):
        super().__init__(task, model_name, use_mean_constraint=False)