import re
import time
from typing import Any, cast, Set, List

# 保持原有导入不变
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from . import grader_manager

# 辅助打印函数保持不变
def log_reward(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [RewardCalc] {msg}", flush=True)

# System Prompt 和 steps_to_msg 函数保持不变
USER_PROMPT="""Based on the conversation trajectory above, evaluate the task completion quality using the framework provided.

Your evaluation should address the following dimensions in order:

**Step 1: Relevance Check (0 or proceed)**
- Are the solution steps relevant to the problem? If the approach is completely unrelated to the task requirements, assign 0 points immediately.
- If relevant, proceed to other evaluation dimensions.

**Step 2: Repetition Penalty Check**
- Does the agent get stuck in infinite loops or repeat identical steps endlessly?
- If there are infinite repetitions of the same steps, consider the relevance of existing steps:
 - If steps are relevant: Maximum 20 points
 - If steps are irrelevant: 0 points

**Step 3: Goal Achievement Assessment (Critical Binary Check)**
- Examine ALL steps comprehensively to determine if the task goal is truly achieved
- Do not be misled by superficial language - verify actual completion
- Check if there is a correct final answer or if the stated objective is genuinely accomplished

**MANDATORY SCORING CONSTRAINTS:**
- If steps are relevant AND goal is achieved/answer is correct: Score MUST be 60-100
- If steps are relevant BUT goal is not achieved/answer is incorrect: Score MUST be 0-40
- FORBIDDEN: Do not assign scores between 41-59 (这是为了强制模型明确判定任务是否成功)

**Step 4: Additional Deductions (within the above constraints)**
- **Code Execution Errors**: Deduct points for runtime errors, bugs, or failed executions
- **Unnecessary/Irrelevant Steps**: Deduct points for redundant or off-topic actions

**Scoring Guidelines:**
- 90-100: Exceptional performance - goal achieved with efficient, clean execution
- 80-89: Strong performance - goal achieved with minor inefficiencies or small errors
- 70-79: Good performance - goal achieved with some unnecessary steps or code issues
- 60-69: Adequate performance - goal achieved but with notable problems
- 30-40: Poor performance - goal not achieved but relevant approach with some progress
- 10-29: Very poor performance - goal not achieved with major execution issues
- 1-9: Minimal relevant attempt - goal not achieved with severe problems
- 0: Complete failure - irrelevant approach or infinite repetition of irrelevant steps

**REMEMBER**: 
- No scores between 41-59 are allowed
- Goal achievement determines the 60+ vs 0-40 range
- Infinite repetition caps score at 20 (if steps are relevant) or 0 (if irrelevant)

Provide your detailed analysis first, explaining your reasoning for each evaluation dimension. Then assign a precise integer score following the mandatory constraints above.

First provide your detailed reasoning analysis, then output an integer score between 0-40 or 60-100 enclosed in <reward></reward> tags, e.g., <reward>75</reward>
"""

def steps_to_msg(steps: list[dict[str, Any]]) -> str:
    trajectory_text = ""
    if steps and steps[0]['role'] != 'assistant':
        pass 
        
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
            continue
        trajectory_text += block.strip() + "\n\n"
    return trajectory_text


@grader_manager.reg("api_process_llm_judge")
class APIProcessRewardCalculator(RewardCalculator):
    """
    混合型奖励计算器：
    1. 结果奖励 (Outcome Reward): 轨迹结束后，使用 LLM 作为裁判打分。
    2. 过程奖励 (Process Reward): 每一步检查是否覆盖了 Ground Truth (GT) 中的 API 调用。
    """
    def __init__(self, task: Task, model_name='DeepSeek-V3-Online-64K'):
        super().__init__(task)
        # 初始化 LLM 裁判客户端
        self._client = DashScopeClient(model_name=model_name)
        
        # 初始化过程奖励逻辑
        self.gt_apis: Set[str] = self._extract_apis(task.ground_truth)
        self.visited_apis: Set[str] = set()
        self.reward_per_api = 0.1 

    # ---------------- 过程奖励逻辑 (Process Reward Logic) ----------------
    def _extract_apis(self, code_str: str) -> Set[str]:
        if not code_str:
            return set()
        return set(re.findall(r"(apis\.\w+\.\w+)", code_str))

    def calculate_step_reward(self, step_code: str) -> float:
        if not step_code or not self.gt_apis:
            return 0.0

        step_apis = self._extract_apis(step_code)
        matched_apis = step_apis.intersection(self.gt_apis)
        newly_covered_apis = matched_apis - self.visited_apis
        
        reward = 0.0
        if newly_covered_apis:
            reward = len(newly_covered_apis) * self.reward_per_api
            self.visited_apis.update(newly_covered_apis)
            
        return reward

    # ---------------- 结果奖励逻辑 (Outcome Reward Logic - LLM) ----------------

    def pack_message(self, trajectory: Trajectory):
        messages=[]
        query = "Unknown Query"
        if len(trajectory.steps) >= 2:
            query = trajectory.steps[1].get('content', '')
        
        trajectory_text = f"Query: {query}\n"
        trajectory_text += "The following is the dialogue trace of the task execution:\n\n"
        trajectory_text += steps_to_msg(trajectory.steps[2:])
        
        messages.append({"role": "user", "content": trajectory_text})
        messages.append({"role": "user", "content": USER_PROMPT})
        return messages
    
    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        log_reward(f"Calculating final reward for Task ID: {instance_id}")
        x, reason = cast(tuple[float, str], self._calculate_reward(trajectory, env, eject_llm_output=True))
        log_reward(f"Final Reward for {instance_id}: {x}")
        return {
            "score": x,
            "reason": reason
        }

    def _calculate_reward(self, trajectory: Trajectory, env: EnvClient, *, eject_llm_output: bool = False):
        """
        内部逻辑：负责调用 LLM API 并解析结果。
        [Updated] 已从流式 chat_stream_with_retry 更改为同步 chat_with_retry
        """
        start_t = time.time()
        log_reward("Starting LLM Judge request...")
        
        response = ""
        messages = self.pack_message(trajectory)
        
        log_reward(f"Message packed. Content length ~{len(str(messages))} chars.") 

        try:
            log_reward(f"Debug - Message Count: {len(messages)}")
            if messages:
                log_reward(f"Debug - Last Message Content Preview: {messages[-1]['content'][:100]}...")

            # ================= [CHANGE START] =================
            # 使用同步的 chat_with_retry 方法，直接获取完整字符串
            # 移除了原有的 loop 和 chunk 处理逻辑
            response = self._client.chat_with_retry(
                messages=messages, 
                max_retries=3
            )
            
            log_reward("="*20 + " LLM RESPONSE START " + "="*20)
            print(response)  # 或者使用 log_reward(f"\n{response}")
            log_reward("="*20 + " LLM RESPONSE END " + "="*20)

            # 简单的 Debug 打印
            log_reward(f"[DEBUG] Response received. Length: {len(response)}")
            if hasattr(response, '__dict__'):
                # 防止意外返回对象而非字符串的情况
                 log_reward(f"[DEBUG] Response Dict: {response.__dict__}")
            # ================= [CHANGE END] =================
                    
        except Exception as e:
            log_reward(f"[ERROR] LLM Judge request failed with Exception: {e}")
            
        total_time = time.time() - start_t
        log_reward(f"LLM Judge request completed. Total Cost: {total_time:.2f}s")
            
        score = 0.0
        if response:
            reward_match = re.search(r'<reward>([\d\.]+)</reward>', response.strip())
            if reward_match:
                score_val = float(reward_match.group(1))
                score = max(0.0, min(100.0, score_val)) / 100.0
            else:
                log_reward(f"[WARNING] Could not parse score from response: {response[-200:]}...")
                score = 0.0
        else:
            log_reward("[ERROR] No response from evaluation API")
            score = 0.0
        
        if not eject_llm_output:
            return score
        else:
            return score, response