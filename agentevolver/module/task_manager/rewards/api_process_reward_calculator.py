import re
from typing import Any, cast, Set, List
# 导入项目内部的模块
# EnvClient: 用于与环境交互的客户端
# DashScopeClient: 阿里模型服务的客户端（用于调用通义千问等模型）
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
# RewardCalculator: 奖励计算器的基类
# GraderResult: 定义返回结果格式的类型
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

# 导入上一段代码中定义的管理器实例，用于注册本计算器
from . import grader_manager

# =============================================================================
# 裁判模型的提示词 (System Prompt)
# 这是一个非常详细的指令，用于指导 LLM 如何评估 Agent 的表现。
# 核心逻辑：
# 1. 0-40分：任务未完成或失败。
# 2. 60-100分：任务成功完成。
# 3. 禁止打分 41-59分：强制 LLM 进行二分类判断（成/败），避免模糊的中间分。
# =============================================================================
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
    """
    辅助函数：将步骤列表转换为格式化的字符串，以便 LLM 阅读。
    格式化为类似 ReAct 的轨迹：
    <|ACTION|> (Assistant 的输出)
    <|OBSERVATION|> (User/Environment 的反馈)

    Args:
        steps (list[dict[str, Any]]): 包含 'role' 和 'content' 的字典列表。

    Returns:
        str: 拼接好的对话历史字符串。
    """
    trajectory_text = ""
    # 稍微放宽断言，防止某些特殊情况下第一条不是 assistant，只做检查不报错
    if steps and steps[0]['role'] != 'assistant':
        pass 
        
    for i, msg in enumerate(steps):
        role = msg.get("role", "unknown")
        if role == 'assistant':
            # 模型的输出被标记为 Action
            block = f""">>> STEP {i//2} <<<
<|ACTION|>
{msg['content']}
<|END|>
"""
        elif role == "user":
            # 环境的返回被标记为 Observation
            block = f"""<|OBSERVATION|>
{msg['content']}
<|END|>
"""
        else:
            # 忽略 system prompt 或其他未知角色
            continue
        trajectory_text += block.strip() + "\n\n"
    return trajectory_text


# 使用装饰器将此类注册到 grader_manager 中
# 注册名为 "api_process_llm_judge"
# 以后可以通过 grader_manager.get_calculator("api_process_llm_judge") 获取实例
@grader_manager.reg("api_process_llm_judge")
class APIProcessRewardCalculator(RewardCalculator):
    """
    混合型奖励计算器：
    1. 结果奖励 (Outcome Reward): 轨迹结束后，使用 LLM 作为裁判打分。
    2. 过程奖励 (Process Reward): 每一步检查是否覆盖了 Ground Truth (GT) 中的 API 调用。
    """
    def __init__(self, task: Task, model_name='qwen3-235b-a22b-instruct-2507'):
        """
        初始化计算器。

        Args:
            task (Task): 包含任务描述和 Ground Truth 代码的任务对象。
            model_name (str): 用于当裁判的大模型名称。
        """
        super().__init__(task)
        # --- 初始化 LLM 裁判客户端 ---
        self._client = DashScopeClient(model_name=model_name)
        
        # --- 初始化过程奖励逻辑 ---
        # 1. 从任务的标准答案(GT)中提取出所有正确的 API 调用集合
        #    task.ground_truth 是一段正确的 Python 代码
        self.gt_apis: Set[str] = self._extract_apis(task.ground_truth)
        
        # 2. 初始化一个集合，记录 Agent 已经调用并获得过奖励的 API
        #    防止同一个正确操作被重复刷分
        self.visited_apis: Set[str] = set()
        
        # 配置: 每覆盖一个新的正确 API，奖励 0.1 分
        self.reward_per_api = 0.1 

    # ---------------- 过程奖励逻辑 (Process Reward Logic) ----------------
    
    def _extract_apis(self, code_str: str) -> Set[str]:
        """
        工具方法：使用正则表达式从代码字符串中提取 API 调用。
        目标格式: apis.service.function (例如: apis.calendar.add_event)
        """
        if not code_str:
            return set()
        # 正则含义：匹配 "apis." 开头，后面跟着两个由点号分隔的单词
        return set(re.findall(r"(apis\.\w+\.\w+)", code_str))

    def calculate_step_reward(self, step_code: str) -> float:
        """
        计算单步的过程奖励。
        
        调用时机：Agent 生成代码并执行后。
        
        逻辑：
        1. 提取当前步骤代码中的 API。
        2. 判断这些 API 是否在标准答案 (GT) 中。
        3. 过滤掉之前已经奖励过的 API。
        4. 剩下的就是“新覆盖的正确 API”，给予奖励并更新 visited 记录。
        
        Args:
            step_code (str): 当前步骤 Agent 生成的代码。
            
        Returns:
            float: 本步骤获得的增量奖励值。
        """
        if not step_code or not self.gt_apis:
            return 0.0

        # 1. 提取当前步骤使用的 API
        step_apis = self._extract_apis(step_code)
        
        # 2. 交集运算：找出属于标准答案的 API
        matched_apis = step_apis.intersection(self.gt_apis)
        
        # 3. 差集运算：找出其中尚未奖励过的 (Newly Covered)
        newly_covered_apis = matched_apis - self.visited_apis
        
        reward = 0.0
        if newly_covered_apis:
            # 计算奖励：新发现数量 * 单个奖励值
            reward = len(newly_covered_apis) * self.reward_per_api
            # 4. 更新已访问集合，防止未来重复奖励
            self.visited_apis.update(newly_covered_apis)
            
        return reward

    # ---------------- 结果奖励逻辑 (Outcome Reward Logic - LLM) ----------------

    def pack_message(self, trajectory: Trajectory):
        """
        将完整的轨迹打包成 LLM 的输入消息格式。
        """
        messages=[]
        
        # 简单的防御性检查：尝试从步骤中获取具体的 Query 内容
        # 假设 trajectory.steps[1] 是 User 提出的具体问题
        query = "Unknown Query"
        if len(trajectory.steps) >= 2:
            query = trajectory.steps[1].get('content', '')
        
        # 拼接 Prompt
        trajectory_text = f"Query: {query}\n"
        trajectory_text += "The following is the dialogue trace of the task execution:\n\n"
        # 转换历史对话为字符串 (跳过前两个 setup 步骤)
        trajectory_text += steps_to_msg(trajectory.steps[2:])
        
        # 第一条 User 消息：提供轨迹上下文
        messages.append({"role": "user", "content": trajectory_text})
        # 第二条 User 消息：提供评分标准 (Prompt Engineering)
        messages.append({"role": "user", "content": USER_PROMPT})
        return messages
    
    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        """
        对外接口：计算最终奖励 (Outcome Reward)。
        通常在 Episode 结束时调用。
        """
        # 调用内部方法获取分数和理由
        x, reason = cast(tuple[float, str], self._calculate_reward(trajectory, env, eject_llm_output=True))
        return {
            "score": x,
            "reason": reason
        }

    def _calculate_reward(self, trajectory: Trajectory, env: EnvClient, *, eject_llm_output: bool = False):
        """
        内部逻辑：负责调用 LLM API 并解析结果。
        """
        response = ""
        # 使用流式 API 调用 LLM，并拼接完整的响应字符串
        # max_retries=64 表示网络不稳定时会疯狂重试，保证获取结果
        for chunk in self._client.chat_stream_with_retry(messages=self.pack_message(trajectory), max_retries=64):
            response += chunk
            
        score = 0.0
        if response:
            # 使用正则从 LLM 的回复中提取 <reward>75</reward> 标签中的数字
            reward_match = re.search(r'<reward>([\d\.]+)</reward>', response.strip())
            if reward_match:
                score_val = float(reward_match.group(1))
                # 归一化分数：将 0-100 的整数转换为 0.0-1.0 的浮点数
                # max/min 确保分数不越界
                score = max(0.0, min(100.0, score_val)) / 100.0
            else:
                # 如果 LLM 没按格式输出，打印错误并给 0 分
                print(f"Could not parse score from response: {response}")
                score = 0.0
        else:
            print("No response from evaluation API")
            score = 0.0
        
        # 根据参数决定是只返回分数，还是返回 (分数, 完整回复)
        if not eject_llm_output:
            return score
        else:
            return score, response