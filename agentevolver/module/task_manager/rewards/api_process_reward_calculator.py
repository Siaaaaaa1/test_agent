import re
import time
import zlib
from typing import Any, cast, Set, List, Optional, Tuple, Dict

# 保持原有导入不变
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from . import grader_manager
import math
# 辅助打印函数
def log_reward(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [RewardCalc] {msg}", flush=True)

# ================= PROMPTS =================

CONTINUOUS_SCORE_PROMPT = """Based on the conversation trajectory above, evaluate the task completion quality using the framework provided.

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

BINARY_SCORE_PROMPT = """Based on the conversation trajectory above, you are a strict evaluator tasked with determining if the user's request was **successfully resolved** using a binary scoring system.

You must follow the strict evaluation framework below to determine the final score (0 or 1). Partial completion counts as FAILURE (0).

**Evaluation Framework:**

**Step 1: Critical Filtering (Immediate Failure Check)**
If any of the following are true, the score is **0 (Failure)** immediately:
- **Irrelevance**: The agent's actions are completely unrelated to the user's request.
- **Infinite Loops/Repetition**: The agent gets stuck repeating the same invalid steps, identical code blocks, or "thinking" loops without making progress.
- **Refusal**: The agent explicitly refuses to do the task (e.g., "I cannot do this").
- **Hallucination**: The agent claims to have finished the task, but the context shows no evidence of a solution or the final answer is missing.

**Step 2: Goal Achievement Verification (The "Success" Standard)**
To assign a **1 (Success)**, the trajectory must meet ALL of the following criteria:
1.  **Final Answer Presence**: There is a clear, definitive answer or result provided at the end.
2.  **Correctness**: The answer effectively solves the user's specific problem.
3.  **Execution Validity**: If code was written, it must have been executed successfully (or the final iteration must be correct). Do not give credit for code that crashed and was never fixed.
4.  **Completeness**: All parts of the user's multi-part request (if any) are addressed.

**Step 3: Handling Imperfections (Noise Tolerance)**
- **Minor Inefficiencies**: If the agent took a few extra steps or had a small error but *fixed it* and eventually reached the correct goal, this is still a **SUCCESS (1)**.
- **Redundant Steps**: Irrelevant steps do not disqualify a success, *unless* they lead to a failure to answer the main question.

**Summary of Scoring Logic:**
- **Result: 1 (Pass)**: Goal achieved, answer correct, even with minor redundant steps.
- **Result: 0 (Fail)**: Goal not achieved, wrong answer, infinite loop, or relevant but incomplete attempt.

**Instructions for Output:**
1.  First, analyze the trajectory step-by-step. explicitly checking for Repetition, Relevance, and Final Goal Achievement.
2.  State your verdict clearly: "Goal Achieved" or "Goal Failed".
3.  Finally, output the binary score enclosed in tags.

Example Output Format:
Reasoning: The agent encountered an error in step 2 but fixed it in step 4. The final answer provided matches the user's requirement. No severe repetition found.
<reward>1</reward>
"""

def steps_to_msg(steps: list[dict[str, Any]]) -> str:
    trajectory_text = ""
    for i, msg in enumerate(steps):
        role = msg.get("role", "unknown")
        content = msg.get('content', '')
        if role == 'assistant':
            block = f">>> STEP {i//2} <<<\n<|ACTION|>\n{content}\n<|END|>\n"
        elif role == "user":
            block = f"<|OBSERVATION|>\n{content}\n<|END|>\n"
        else:
            continue
        trajectory_text += block.strip() + "\n\n"
    return trajectory_text

def log_reward(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [RewardCalc] {msg}", flush=True)

@grader_manager.reg("api_process_llm_judge")
class APIProcessRewardCalculator(RewardCalculator):
    """
    混合型奖励计算器 (增强版)：
    1. 步骤奖励：API 命中奖励 + 复读惩罚。
    2. 结果奖励：语义打分 (Outcome) + 效率打分 (Efficiency)。
    """
    def __init__(self, task: Task, model_name='DeepSeek-V3-Online-64K', 
                 reward_mode='outcome_continuous', 
                 degeneration_mode='ngram', 
                 degeneration_char_limit=100, 
                 zlib_threshold=0.1,
                 ngram_n=3,
                 ngram_threshold=0.1,
                 repetition_penalty=-1.0,
                 # [新增] 效率奖励配置
                 efficiency_lambda: float = 0.05,     # 衰减系数
                 api_cost_weight: float = 2.0         # API 调用的步骤当量
                 ): 
        super().__init__(task)
        self.model_name = model_name
        self.reward_mode = reward_mode
        self._client = DashScopeClient(model_name=model_name)
        
        # 规则检测配置
        self.deg_mode = degeneration_mode
        self.deg_char_limit = degeneration_char_limit
        self.zlib_thresh = zlib_threshold
        self.ngram_n = ngram_n
        self.ngram_thresh = ngram_threshold
        self.repetition_penalty = repetition_penalty
        
        # 效率计算参数
        self.efficiency_lambda = efficiency_lambda
        self.api_cost_weight = api_cost_weight

        # API 提取初始化
        self.gt_apis: Set[str] = self._extract_apis(task.ground_truth)
        self.visited_apis: Set[str] = set()
        
        num_gt = len(self.gt_apis)
        if num_gt > 0:
            self.reward_per_api = 1.0 / num_gt
        else:
            self.reward_per_api = 0.0 

        self.total_process_reward = 0.0 

        log_reward("-" * 40)
        log_reward(f"Init RewardCalculator [API+Repetition+Efficiency].")
        log_reward(f"Efficiency: lambda={self.efficiency_lambda}, api_weight={self.api_cost_weight}")
        log_reward("-" * 40)

    # ---------------- 辅助：单步规则检测 (Rule-Based Check) ----------------
    
    def _check_step_degeneration(self, content: str) -> Tuple[bool, str]:
        """
        检测单个步骤的内容是否发生退化（复读/死循环）。
        针对 step_code (LLM输出) 进行检测。
        """
        if not self.deg_mode:
            return False, ""
        
        # 只有内容足够长才检测，避免误伤简短回答
        if not content or len(content) < self.deg_char_limit:
            return False, ""

        # 分流算法
        if self.deg_mode == 'zlib':
            return self._check_zlib(content)
        elif self.deg_mode == 'ngram':
            return self._check_ngram(content)
        
        return False, ""

    def _check_zlib(self, content: str) -> Tuple[bool, str]:
        """算法1: Zlib 压缩比检测"""
        compressed = zlib.compress(content.encode('utf-8'))
        ratio = len(compressed) / len(content)
        if ratio < self.zlib_thresh:
            return True, f"Zlib Ratio {ratio:.4f} < {self.zlib_thresh}"
        return False, ""

    def _check_ngram(self, content: str) -> Tuple[bool, str]:
        """算法2: N-gram 重复率检测"""
        tokens = content.split()
        if len(tokens) < self.ngram_n:
            return False, ""
            
        ngrams = [tuple(tokens[i:i + self.ngram_n]) for i in range(len(tokens) - self.ngram_n + 1)]
        total_ngrams = len(ngrams)
        unique_ngrams = len(set(ngrams))
        
        if total_ngrams == 0:
            return False, ""
            
        diversity_ratio = unique_ngrams / total_ngrams
        
        if diversity_ratio < self.ngram_thresh:
            return True, f"N-gram Diversity {diversity_ratio:.4f} < {self.ngram_thresh}"
        return False, ""

    # ---------------- 核心修改：步骤奖励 (API + Repetition) ----------------

    def _extract_apis(self, code_str: str) -> Set[str]:
        if not code_str:
            return set()
        return set(re.findall(r"(apis\.\w+\.\w+)", code_str))

    def calculate_step_reward(self, step_code: str) -> Dict[str, float]:
        """
        计算单步奖励。
        
        Args:
            step_code: LLM 生成的动作代码/内容。
            
        Returns:
            dict: {
                "api_reward": float,          # API 命中得分 (正)
                "repetition_penalty": float,  # 复读惩罚扣分 (负)
                "total_score": float          # 两者之和
            }
        """
        # 预览日志
        log_reward(f"[Step Check] Analyzing LLM Action: {step_code[:100].replace(chr(10), ' ')}...")

        # 1. 计算 API 奖励 (Positive)
        api_reward = 0.0
        if step_code and self.gt_apis:
            step_apis = self._extract_apis(step_code)
            matched_apis = step_apis.intersection(self.gt_apis)
            newly_covered_apis = matched_apis - self.visited_apis
            if newly_covered_apis:
                # 这里不需要修改，因为 self.reward_per_api 已经在 __init__ 中动态计算好了
                api_reward = len(newly_covered_apis) * self.reward_per_api
                self.visited_apis.update(newly_covered_apis)

        # 2. 计算复读惩罚 (Negative)
        repetition_pen = 0.0
        is_bad, reason = self._check_step_degeneration(step_code)
        if is_bad:
            repetition_pen = self.repetition_penalty # 使用初始化的惩罚值 (如 -1.0)
            log_reward(f"[Step Degeneration] Detected: {reason}")

        # 3. 汇总
        total_score = api_reward + repetition_pen
        
        # 统计
        if total_score != 0:
             self.total_process_reward += total_score
             log_reward(f"[Step Result] API: +{api_reward:.2f}, Repetition: {repetition_pen:.2f} -> Total: {total_score:.2f}")

        # 返回字典
        return {
            "api_reward": api_reward,
            "repetition_penalty": repetition_pen,
            "total_score": total_score
        }

    # ---------------- 核心修改：结果奖励 (Only Semantic) ----------------

    def pack_message(self, trajectory: Trajectory, use_binary_prompt: bool = False):
        messages=[]
        query = "Unknown Query"
        if len(trajectory.steps) >= 2:
            query = trajectory.steps[1].get('content', '')
        
        trajectory_text = f"Query: {query}\n"
        trajectory_text += "The following is the dialogue trace of the task execution:\n\n"
        trajectory_text += steps_to_msg(trajectory.steps[2:])
        
        system_prompt = BINARY_SCORE_PROMPT if use_binary_prompt else CONTINUOUS_SCORE_PROMPT
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please evaluate this trajectory:\n\n{trajectory_text}"}
        ]
    
    def _count_trajectory_cost(self, trajectory: Trajectory) -> float:
        """[新增] 计算轨迹总成本"""
        assistant_steps = [s for s in trajectory.steps if s.get('role') == 'assistant']
        num_steps = len(assistant_steps)
        api_count = 0
        for step in assistant_steps:
            content = step.get('content', '')
            api_count += len(re.findall(r"apis\.", content))
        # total_cost = float(num_steps) + (self.api_cost_weight * float(api_count))
        total_cost = float(num_steps) # + (self.api_cost_weight * float(api_count))
        log_reward(f"[Cost] Steps:{num_steps}, APIs:{api_count} -> Total:{total_cost:.2f}")
        return total_cost

    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        """计算最终奖励：Outcome + Efficiency (存入metadata)"""
        use_binary = 'binary' in self.reward_mode
        outcome_score, _ = self._calculate_llm_outcome(trajectory, use_binary=use_binary)
        
        efficiency_score = 0.0
        # 成功门控：只有结果 > 0.6 (即成功) 才计算效率分
        if outcome_score > 0.6:
            cost = self._count_trajectory_cost(trajectory)
            efficiency_score = math.exp(-self.efficiency_lambda * cost)
            log_reward(f"[Efficiency] Success! Score={efficiency_score:.4f}")
        else:
            log_reward(f"[Efficiency] Failed. Score=0.0")

        return {
            "score": outcome_score,
            # [修改] 显式返回 efficiency_score，以便 AgentFlow 提取
            "efficiency_score": efficiency_score, 
            "reason": "LLM Semantic Evaluation",
            # 将效率分传递给 Trainer
            "metadata": {
                "efficiency_score": efficiency_score,
                "process_reward_sum": self.total_process_reward
            }
        }

    def _calculate_llm_outcome(self, trajectory: Trajectory, use_binary: bool) -> Tuple[float, str]:
        messages = self.pack_message(trajectory, use_binary_prompt=use_binary)
        try:
            response = self._client.chat_with_retry(messages=messages, max_retries=10)
        except Exception as e:
            return 0.0, f"Error: {str(e)}"
            
        score = 0.0
        if response:
            reward_match = re.search(r'<reward>([\d\.]+)</reward>', response.strip())
            if reward_match:
                score_val = float(reward_match.group(1))
                if use_binary:
                    score = 1.0 if score_val >= 0.9 else 0.0
                else:
                    score = max(0.0, min(100.0, score_val)) / 100.0
        return score, response