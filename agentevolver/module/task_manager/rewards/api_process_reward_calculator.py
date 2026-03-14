import re
import time
import zlib
from typing import Any, cast, Set, List, Optional, Tuple, Dict
import math

# 内部模块导入
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from . import grader_manager

# 辅助打印函数
def log_reward(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [RewardCalc] {msg}", flush=True)

# ================= PROMPTS =================

# 里程碑断档 (Milestone Tiering) 的连续分数评价 Prompt
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
- FORBIDDEN: Do not assign scores between 41-59 (This forces a clear distinction between success and failure)

**Step 4: Additional Deductions (within the above constraints)**
- **Code Execution Errors**: Deduct points for runtime errors, bugs, or failed executions
- **Unnecessary/Irrelevant Steps**: Deduct points for redundant or off-topic actions

**Scoring Guidelines (STRICT COMPLIANCE REQUIRED):**
- 90-100: Exceptional - goal achieved with efficient, clean execution.
- 70-89: Strong/Good - goal achieved but with minor inefficiencies or redundant steps.
- 60-69: Adequate - goal achieved but with notable problems or messy logic.

*(DO NOT assign scores between 41-59)*

- 30-40: [High Partial] Goal NOT achieved, BUT agent successfully retrieved core information (e.g., successfully searched/read the target emails/messages) and attempted the final state-changing action but failed due to syntax/logic errors.
- 15-29: [Low Partial] Goal NOT achieved. Agent made good progress in reading/searching for information, but did NOT attempt the final required action.
- 1-14: [Setup Only] Goal NOT achieved. Agent ONLY performed basic setup actions (e.g., getting passwords, logging in) without making meaningful progress on the actual query.
- 0: Complete failure, infinite loops, or totally irrelevant actions.

Provide your detailed analysis first, explaining your reasoning for each evaluation dimension. Then assign a precise integer score following the mandatory constraints above.

First provide your detailed reasoning analysis, then output an integer score between 0-40 or 60-100 enclosed in <reward></reward> tags, e.g., <reward>75</reward>
"""

# 纯二元评价 Prompt (非 1 即 0)
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
    """将步骤列表转化为适合 LLM 裁判阅读的文本流水账。"""
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

@grader_manager.reg("api_process_llm_judge")
class APIProcessRewardCalculator(RewardCalculator):
    """
    混合型奖励计算器：
    1. 步骤级：API 命中奖励 (基于语义梯队) + 复读惩罚 + 执行错误四象限惩罚。
    2. 结果级：语义打分 (Outcome) + 效率分 (Efficiency，仅供参考，不计入总分)。
    """
    def __init__(self, task: Task, model_name='qwen3.5-plus',
                 reward_mode='outcome_continuous',
                 degeneration_mode='ngram',
                 degeneration_char_limit=100,
                 zlib_threshold=0.1,
                 ngram_n=3,
                 ngram_threshold=0.1,
                 repetition_penalty=-1.0,
                 format_penalty=-1.0,
                 efficiency_lambda: float = 0.05,
                 api_cost_weight: float = 2.0
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
        self.format_penalty = format_penalty

        # 效率计算参数
        self.efficiency_lambda = efficiency_lambda
        self.api_cost_weight = api_cost_weight

        # 预加载并解析 Ground Truth 中的 API
        self.gt_apis: Set[str] = self._extract_apis(task.ground_truth)
        self.visited_apis: Set[str] = set()
        
        # 用于累加执行过程中的奖励/惩罚分数
        self.total_process_reward = 0.0 

    # ---------------- 辅助：单步规则检测 ----------------
    
    def _check_step_degeneration(self, content: str) -> Tuple[bool, str]:
        """检测输出内容是否发生退化（复读/死循环）。"""
        if not self.deg_mode:
            return False, ""
        
        if not content or len(content) < self.deg_char_limit:
            return False, ""

        if self.deg_mode == 'zlib':
            return self._check_zlib(content)
        elif self.deg_mode == 'ngram':
            return self._check_ngram(content)
        
        return False, ""

    def _check_zlib(self, content: str) -> Tuple[bool, str]:
        """使用 Zlib 压缩比检测文本信息熵，判断是否复读。"""
        compressed = zlib.compress(content.encode('utf-8'))
        ratio = len(compressed) / len(content)
        if ratio < self.zlib_thresh:
            return True, f"Zlib Ratio {ratio:.4f} < {self.zlib_thresh}"
        return False, ""

    def _check_ngram(self, content: str) -> Tuple[bool, str]:
        """使用 N-gram 词汇多样性检测是否复读。"""
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

    def _check_execution_error(self, observation: str) -> bool:
        """
        基于正则和关键字，增强检测环境反馈中是否包含致命的代码或 API 报错。
        """
        if not observation:
            return False
            
        if re.search(r"\b[a-zA-Z]*Error:\s", observation):
            return True

        if "Exception: Response status code" in observation:
            return True
        if "Traceback (most recent call last)" in observation:
            return True

        obs_lower = observation.lower()
        critical_phrases = [
            "command not found",      
            "syntax error",           
            "validation error",       
            "internal server error",  
            "access token is missing",
            "module not found",       
            "is not defined"          
        ]
        
        for phrase in critical_phrases:
            if phrase in obs_lower:
                return True
                
        return False

    # ---------------- 核心：步骤奖励 ----------------

    def _extract_apis(self, code_str: str) -> Set[str]:
        if not code_str:
            return set()
        return set(re.findall(r"(apis\.\w+\.\w+)", code_str))

    def _get_api_weight_by_category(self, api_name: str) -> float:
        """
        根据 API 的核心语义动作分配阶梯权重，鼓励高难度操作。
        """
        api_lower = api_name.lower()
        
        setup_keywords = [
            'login', 'logout', 'signup', 'account', 'profile', 
            'password', 'verification', 'verify', 'help'
        ]
        if any(kw in api_lower for kw in setup_keywords):
            return 0.2

        write_action_keywords = [
            'create', 'update', 'delete', 'add', 'remove', 'move', 'copy', 
            'compress', 'decompress', 'send', 'reply', 'forward', 'upload',
            'complete', 'apply', 'place', 'write', 'initiate', 'subscribe', 
            'record', 'attach', 'settle_up', 'post', 'like', 'unlike', 
            'follow', 'unfollow', 'review', 'play', 'pause', 'previous', 
            'next', 'seek', 'loop', 'shuffle', 'clear', 'set', 'label', 
            'unlabel', 'mark', 'withdraw', 'approve', 'deny', 'remind'
        ]
        if any(kw in api_lower for kw in write_action_keywords):
            return 1.2

        read_search_keywords = [
            'show', 'search', 'get', 'download', 'exists'
        ]
        if any(kw in api_lower for kw in read_search_keywords):
            return 0.6

        return 0.6

    def _check_format(self, content: str) -> float:
        """
        检查每步输出是否有且仅有一对 ```python...``` 代码块。
        缺少或超出 1 对时返回 format_penalty，否则返回 0.0。
        """
        matches = re.findall(r'```python[\s\S]*?```', content)
        if len(matches) == 1:
            return 0.0
        reason = "missing code block" if len(matches) == 0 else f"found {len(matches)} code blocks (expected 1)"
        log_reward(f"[Format Check] Penalty applied: {reason}")
        return self.format_penalty

    def calculate_step_reward(self, step_code: str, observation: str = "") -> Dict[str, float]:
        """
        计算单步奖励，引入四象限平滑惩罚逻辑。
        包含：API 命中奖励、复读惩罚、格式惩罚。
        """
        is_error = self._check_execution_error(observation)
        step_apis = self._extract_apis(step_code)
        matched_apis = step_apis.intersection(self.gt_apis)

        api_reward = 0.0
        api_valid = 0

        if matched_apis:
            if not is_error:
                newly_covered = matched_apis - self.visited_apis
                if newly_covered:
                    for api in newly_covered:
                        api_reward += self._get_api_weight_by_category(api)
                    api_valid = 1
                    self.visited_apis.update(newly_covered)
            else:
                api_reward = -0.1
        else:
            if not is_error:
                api_reward = 0.0
            else:
                api_reward = -0.3

        repetition_pen = 0.0
        is_bad, reason = self._check_step_degeneration(step_code)
        if is_bad:
            repetition_pen = self.repetition_penalty
            log_reward(f"[Step Degeneration] Detected: {reason}")

        format_pen = self._check_format(step_code)

        total_score = api_reward + repetition_pen + format_pen
        if total_score != 0:
            self.total_process_reward += total_score

        return {
            "api_reward": api_reward,
            "api_valid": api_valid,
            "repetition_penalty": repetition_pen,
            "format_penalty": format_pen,
            "total_score": total_score
        }

    # ---------------- 结果奖励 ----------------

    def pack_message(self, trajectory: Trajectory, use_binary_prompt: bool = False):
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
        """计算轨迹调用的总成本，用于观察记录。"""
        assistant_steps = [s for s in trajectory.steps if s.get('role') == 'assistant']
        num_steps = len(assistant_steps)
        api_count = 0
        for step in assistant_steps:
            content = step.get('content', '')
            api_count += len(re.findall(r"apis\.", content))
        
        total_cost = float(num_steps) + (self.api_cost_weight * float(api_count))
        return total_cost

    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        """计算最终奖励：严格保持原有的 Score 输出逻辑"""
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

        # 保持 score 仅返回大模型语义分数，效率分及过程分仅存放于元数据
        return {
            "score": outcome_score,
            "efficiency_score": efficiency_score, 
            "reason": "LLM Semantic Evaluation",
            "metadata": {
                "efficiency_score": efficiency_score,
                "process_reward_sum": self.total_process_reward
            }
        }

    def _calculate_llm_outcome(self, trajectory: Trajectory, use_binary: bool) -> Tuple[float, str]:
        """请求大模型作为裁判打分。"""
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
                    score = 1.0 if score_val >= 0.8 else 0.0
                else:
                    score = max(0.0, min(100.0, score_val)) / 100.0
        return score, response