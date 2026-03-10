


import json
import os
import re
import threading
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Sequence, Optional, Tuple, Callable

from loguru import logger

# 假设项目中的基础 Schema 引用
from agentevolver.schema.trajectory import Trajectory, Reward
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.module.task_manager.base import LlmClient
from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.utils.step_parser import parse_response_ids_to_steps

# =============================================================================
# PART 1: Prompts & Parsing Logic (User Provided)
# =============================================================================

AGENT_SUMMARIZE_SYSTEM_PROMPT = """# ROLE
You are a **Real-World Task Discovery Expert**.  
Your job is to analyze an agent's API interaction history and transform it into **realistic, user-centered tasks** that could be solved using the same interaction **patterns**.

---

# OBJECTIVES
1. **Understand Capabilities** - Analyze the recorded API calls to identify the actual functional capabilities demonstrated.

2. **Think Like a Real Experienced User** - Imagine practical, everyday problems where a real person would naturally use this exact API call sequence (minus the documentation exploration).
   - Create problems that use **multiple different API calls**, not just a single call.
   - Use **clear, specific, verifiable** user requests.

3. **Abstract into Three Elements** For each realistic task, provide:
   - **query**: A natural-language request that a real user might make.
   - **confidence**: A number between `0.0` and `1.0` representing how confident you are that this is a real, common need.
   - **action_sequence**: The sequence of technical steps that directly accomplishes the task.

---

# RULES FOR SCENARIO CREATION
## 1. Focus on User Intent
- Always start from a **human goal**.
- Avoid restating the API function in technical terms—capture the **why** behind the action.

## 2. Remove Non-Essential Steps
- Do **not** include:
  - Capability exploration or debugging steps.

## 3. Specificity & Verifiability
- The query must be **precise enough** that someone can clearly judge success/failure.
- Include **concrete details**:  
  - Numbers, dates, names, locations, thresholds, item lists, etc.
- Avoid vague words like “check”, “review”, or “ensure” unless paired with measurable criteria.

## 4. Practicality
- Use **relatable, everyday** scenarios.
- Avoid tasks that are purely exploratory or only serve to test an API.

---

# OUTPUT FORMAT
For each identified task, output exactly one block in this format:

<task>
{
  "query": "[A natural, specific, verifiable user request]",
  "confidence": [0.0 - 1.0],
  "action_sequence": "[Technical sequence that directly solves the task]"
}
</task>

---

# GOOD EXAMPLES
<task>
{
  "query": "Joseph paid for my grocery recently as my payment cards were not working at the time. Send them the owed money with a description note 'Grocery Bill' as per my phone text conversation, and then send them a phone text message, 'Done.'",
  "confidence": 1.0,
  "action_sequence": "supervisor_profile = apis.supervisor.show_profile()\\nsupervisor_passwords = {ap['account_name']: ap['password'] for ap in apis.supervisor.show_account_passwords()}\\nphone_token = apis.phone.login(username=supervisor_profile['phone_number'], password=supervisor_passwords['phone'])['access_token']\\nfriend_contact = next(item for p in range(10) for item in apis.phone.search_contacts(page_index=p, access_token=phone_token, query='Joseph') if item['first_name'] == 'Joseph')\\nyesterday_msgs = [item for p in range(10) for item in apis.phone.search_text_messages(page_index=p, access_token=phone_token, phone_number=friend_contact['phone_number']) if pendulum.parse(item['sent_at']) > DateTime.now().subtract(days=1).start_of('day')]\\npattern = re.compile(r'\\\\$(\\\\d*)')\\nfor msg in yesterday_msgs:\\n    matches = pattern.search(msg['message'])\\n    if matches: grocery_cost = int(matches[1])\\nvenmo_token = apis.venmo.login(username=supervisor_profile['email'], password=supervisor_passwords['venmo'])['access_token']\\napis.venmo.create_transaction(access_token=venmo_token, receiver_email=friend_contact['email'], amount=grocery_cost, description='Grocery Bill')\\napis.phone.send_text_message(access_token=phone_token, phone_number=friend_contact['phone_number'], message='Done.')\\napis.supervisor.complete_task(status='success')"
}
</task>

<task>
{
  "query": "Send the following phone message to my roommates and friends, who do not have a venmo account, 'Make an account on venmo please.'",
  "confidence": 1.0,
  "action_sequence": "supervisor_profile = apis.supervisor.show_profile()\\nsupervisor_passwords = {ap['account_name']: ap['password'] for ap in apis.supervisor.show_account_passwords()}\\nphone_token = apis.phone.login(username=supervisor_profile['phone_number'], password=supervisor_passwords['phone'])['access_token']\\ncontacts = []\\nfor rel in ['roommates', 'friends']:\\n    for p in range(10): contacts += apis.phone.search_contacts(page_index=p, access_token=phone_token, query=rel, relationship=rel)\\nvenmo_token = apis.venmo.login(username=supervisor_profile['email'], password=supervisor_passwords['venmo'])['access_token']\\nfor c in contacts:\\n    prof = apis.venmo.show_profile(email=c['email'], access_token=venmo_token)\\n    if 'message' in prof and 'Account for this email does not exist.' in prof['message']:\\n        apis.phone.send_text_message(phone_number=c['phone_number'], message='Make an account on venmo please.', access_token=phone_token)\\napis.supervisor.complete_task(status='success')"
}
</task>

---

# CHECKLIST BEFORE FINALIZING
✅ **Clear goal** – What exactly is the user trying to achieve?  
✅ **Concrete details** – Who, what, when, where, how much/many?  
✅ **Verifiable** – Can success/failure be objectively determined?  
✅ **Human-first phrasing** – Sounds like something a real person would say.
"""

def _get_action_observation_pair(traj: Trajectory) -> list[tuple[str, str]]:
    res = []
    for idx, step in enumerate(traj.steps):
        # 兼容 step 是 dict 或对象的情况
        role = step.get("role") if isinstance(step, dict) else getattr(step, "role", None)
        content = step.get("content") if isinstance(step, dict) else getattr(step, "content", "")
        
        if role == "assistant" and idx + 1 < len(traj.steps):
            next_step = traj.steps[idx + 1]
            next_role = next_step.get("role") if isinstance(next_step, dict) else getattr(next_step, "role", None)
            next_content = next_step.get("content") if isinstance(next_step, dict) else getattr(next_step, "content", "")

            # As there is no standard for environments, we do not know whether it will respond as user or tool.
            if next_role == "tool":
                # get observation from tool message
                observation = next_content
            elif next_role == "user":
                # get observation from user message
                observation = next_content
            else:
                continue
            res.append((content, observation))

    return res

def get_task_summarize_prompt(
    trajectories: Sequence[Trajectory],
    old_objectives: Sequence[TaskObjective],
    profile: EnvProfile | None,
) -> tuple[str, str]:
    x = ""
    idx = 0
    for traj in trajectories:
        pairs = _get_action_observation_pair(traj)
        x += f"## Record {idx}\n"
        x += f"### History\n"
        for step_idx, history in enumerate(pairs):
            x += f""">>> STEP {step_idx} <<<
<|ACTION|>
{history[0]}
<|END|>

<|OBSERVATION|>
{history[1]}
<|END|>
"""
        if traj.reward is not None:
            # 兼容 outcome 为 float 或其他类型
            outcome = traj.reward.outcome if traj.reward else 0.0
            desc = traj.reward.description if traj.reward else ""
            x += f"### Reward: {outcome}\n{desc}\n"
        idx += 1

    objectives: list[str] = [obj.objective for obj in old_objectives if obj.objective is not None]

    user_prompt = f"""Please analyze the following agent interaction sequence and abstract specific tasks from it:

{x}

# Old Objectives
You have already explored the following objectives:

{objectives}

Please avoid repeating these objectives.

# Task Requirements

{profile.get_task_preference_instruction() if profile else "Please follow the instructions to generate tasks."}

# Now Start

Please identify the specific tasks the agent is attempting to complete in these interactions, and abstract them into clear task descriptions and queries following the specified format.
"""
    return AGENT_SUMMARIZE_SYSTEM_PROMPT, user_prompt

def parse_tasks_from_response(base_task: Task, response: str) -> list[TaskObjective]:
    """
    Args:
        base_task: Template task object to clone from
        response: LLM output string
    """
    tasks: list[TaskObjective] = []
    try:
        task_matches = re.findall(r"<task>(.*?)</task>", response, re.DOTALL)

        for task_content in task_matches:
            try:
                t = json.loads(task_content)
            except json.JSONDecodeError:
                continue

            if (
                "query" not in t
                or "confidence" not in t
                or "action_sequence" not in t
            ):
                continue
            
            # 创建新的 Task 对象
            new_task = copy.deepcopy(base_task)
            new_task.query = t["query"]
            new_task.open_query = True # 标记为开放性任务
            
            # 创建 TaskObjective
            # 注意：TaskObjective 的构造函数参数可能需要根据你的实际定义调整
            # 这里假设它接受 task, objective(query), confidence, ground_truth
            x = TaskObjective(
                task=new_task,
                confidence=t["confidence"],
                reward=None,
            )
            x.objective = t["query"] # 确保 objective 字段被赋值
            x.ground_truth = t["action_sequence"]
            tasks.append(x)

    except Exception as e:
        logger.error(f"Error parsing tasks: {e}")

    return tasks

# =============================================================================
# PART 2: Evaluator Components (Copied/Adapted for reuse)
# =============================================================================

class EvaluationPrompts:
    """Prompt templates for trajectory evaluation"""
    
    def success_evaluation_prompt(self, query: str, trajectory_summary: str,
                                final_observation: str) -> list:
        messages = [
            {
            "role": "user",
            "content": f"""You are a strict task evaluation expert. Your goal is to determine whether the following multi-step agent trajectory successfully completed the assigned task.

    # Task Details
    - Query: {query}

    # Execution Summary
    - Trajectory Summary:
    {trajectory_summary}

    - Final Observation: {final_observation}

    # Evaluation Instructions
    Carefully analyze the trajectory to determine if the task was truly completed. Specifically, consider:
    1. **Logical Flow**: Was the sequence of steps logical without unreasonable skips?
    2. **Final Result**: Did the final state achieve the expected outcome of the NEW Query?
    3. **Consistency**: Does the trajectory actually perform what the query asks for?

    # Format Your Response Strictly As:

    Success: [true/false]
    Reason: [Concise and specific explanation]
    """
            }
        ]
        return messages

class TrajectoryEvaluator:
    """Evaluate trajectory success using LLM"""
    
    def __init__(self, client: LlmClient):
        self.client = client
        self.prompts = EvaluationPrompts()

    def evaluate_trajectory_success(self, task: TaskObjective, trajectory: Trajectory) -> bool:
        """Evaluate if trajectory completed the task successfully"""
        try:
            trajectory_summary = self._create_trajectory_summary(trajectory)
            
            final_observation: str | None = None
            for step in reversed(trajectory.steps):
                # 兼容字典访问
                role = step.get('role') if isinstance(step, dict) else getattr(step, 'role', '')
                content = step.get('content') if isinstance(step, dict) else getattr(step, 'content', '')
                
                if final_observation is None and role != 'assistant':
                    final_observation = content
                    break
                
            assert task.objective is not None, "synthetic task must have objective"
            prompt = self.prompts.success_evaluation_prompt(
                query=task.objective,
                trajectory_summary=trajectory_summary,
                final_observation=final_observation or "[no observation]"
            )
            
            # Get LLM evaluation
            # 假设 client.chat 返回 {"content": "..."} 或直接返回 str
            response_dict = self.client.chat(prompt, sampling_params={"temperature": 0.0})
            response = response_dict.get("content", "") if isinstance(response_dict, dict) else str(response_dict)
            
            return self._parse_evaluation_response(response)
            
        except Exception as e:
            logger.error(f"Failed to evaluate trajectory: {e}")
            return False
    
    def _create_trajectory_summary(self, traj: Trajectory) -> str:
        summary_blocks = []
        for i, step in enumerate(traj.steps):
            role = step.get('role') if isinstance(step, dict) else getattr(step, 'role', 'unknown')
            content = step.get('content') if isinstance(step, dict) else getattr(step, 'content', '')
            
            block = f"(Step {i + 1}) {role}:\n"
            block += f"{content[:200]}...\n"
            summary_blocks.append(block)
        return "\n".join(summary_blocks)
    
    def _parse_evaluation_response(self, response: str) -> bool:
        if not response: return False
        response_lower = response.lower()
        if 'success: true' in response_lower: return True
        return False

# =============================================================================
# PART 3: Hindsight Manager (The Logic Reuser)
# =============================================================================

class HindsightManager:
    def __init__(self, 
                 llm_client: LlmClient, 
                 tokenizer, 
                 save_path: str = "tasks_explored/hindsight_supplement.jsonl",
                 num_threads: int = 4):
        
        self._llm_client = llm_client
        self._tokenizer = tokenizer
        self.save_path = save_path
        self._num_threads = num_threads
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self._lock = threading.Lock()
        
        # 内置验证器
        self._validator = TrajectoryEvaluator(llm_client)

    def process_failed_batch(self, 
                             prompts: List[List[int]], 
                             responses: List[List[int]], 
                             scores: List[float], 
                             task_ids: List[str],
                             threshold: float = 0.0):
        """
        入口函数：处理训练中的失败 Batch
        """
        failed_tasks_data = []

        # 1. 筛选
        for i, score in enumerate(scores):
            if score <= threshold:
                traj = self._reconstruct_trajectory(prompts[i], responses[i], task_ids[i])
                if traj and len(traj.steps) > 0:
                    failed_tasks_data.append((traj, task_ids[i]))

        if not failed_tasks_data:
            return

        logger.info(f"[Hindsight] Found {len(failed_tasks_data)} failed trajectories. Starting processing...")

        # 2. 并发处理 (Thread Pool)
        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            for traj, original_tid in failed_tasks_data:
                executor.submit(self._execute_hindsight_strategy, traj, original_tid)

    def _execute_hindsight_strategy(self, traj: Trajectory, original_task_id: str):
        """
        单条 Trajectory 的核心处理流程：
        Generate -> Validate -> Rewrite GT -> Save
        """
        try:
            # --- 1. Generate Candidates ---
            # 构造 dummy task 用于 old_objectives (Hindsight 不依赖旧目标，所以传空)
            dummy_old_obj = [] 
            
            sys_prompt, user_prompt = get_task_summarize_prompt(
                trajectories=[traj], 
                old_objectives=dummy_old_obj, 
                profile=None  # Hindsight 往往不需要特定 Profile，或者你可以传入全局 Profile
            )
            
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
            
            # 调用 LLM 生成建议 (Higher temp for diversity)
            resp_dict = self._get_llm_chat_fn()(messages, custom_sampling_params={"temperature": 0.7})
            response_content = resp_dict["content"]

            # 创建基础 Task 模板 (只需要 ID 等基本信息)
            base_task = Task(dataset="hindsight", id=original_task_id, query="")
            
            # 解析出 TaskObjective 列表
            candidate_tasks = parse_tasks_from_response(base_task, response_content)
            
            valid_tasks = []
            
            # --- 2. Validate Loop ---
            for task_obj in candidate_tasks:
                # 复用 _validate
                is_valid = self._validate(task_obj, traj)
                
                if is_valid:
                    # --- 3. Rewrite GT ---
                    # 复用 _rewrite_new_gt (它会调用 LLM 清洗 action_sequence)
                    final_task_obj = self._rewrite_new_gt(task_obj, traj)
                    
                    # 补全 ID 信息，防止重复
                    unique_suffix = os.urandom(4).hex()
                    final_task_obj.task.id = f"hindsight_{original_task_id}_{unique_suffix}"
                    
                    valid_tasks.append(final_task_obj)
                    logger.debug(f"[Hindsight] Captured Valid Task: {final_task_obj.objective[:30]}...")

            # --- 4. Save ---
            if valid_tasks:
                self._save_tasks(valid_tasks)

        except Exception as e:
            logger.exception(f"[Hindsight] Strategy execution failed for {original_task_id}: {e}")

    # =========================================================
    # 下面是严格复用 LLM_filter 的逻辑方法
    # =========================================================

    def _validate(self, task: TaskObjective, trajectory: Trajectory) -> bool:
        """
        Validates the success of a task's execution by evaluating the provided trajectory.
        (Logic copied from LLM_filter)
        """
        try:
            # 这里我们使用 self._validator 替代原本每次实例化 validator
            return self._validator.evaluate_trajectory_success(task, trajectory)
        except Exception as e:
            logger.exception(f"Error in _validate for task {task}: {e}")
            return False

    def _rewrite_new_gt(self, task: TaskObjective, trajectory: Trajectory) -> TaskObjective:
        """
        Rewrite ground-truth
        (Logic copied from LLM_filter)
        """
        # 注意：这里调用 get_task_summarize_prompt 可能会有些冗余，
        # 因为 parse_tasks_from_response 已经给了我们一个 action_sequence。
        # 但按照你的要求，严格复用 LLM_filter 的流程，我们会再跑一次来“清洗”或“确认” GT。
        
        sys_prompt, user_prompt = get_task_summarize_prompt([trajectory], [task], None)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        
        llm_output = self._get_llm_chat_fn()(messages)["content"]
        
        # 注意：parse_tasks_from_response 返回 list[TaskObjective]
        # 我们取第一个匹配的作为 GT
        tasks_list = parse_tasks_from_response(task.task, llm_output)
        
        if tasks_list:
            # 更新 GT 为重新生成版本
            task.ground_truth = tasks_list[0].ground_truth
        
        return task   

    def _get_llm_chat_fn(self, sampling_params: Optional[dict] = None) -> Callable:
        """
        Helper to wrap LLM client with retry logic (Copied logic style)
        """
        def llm_chat(messages: list[dict], custom_sampling_params: Optional[dict] = None) -> dict:
            params = {}
            if sampling_params: params.update(sampling_params)
            if custom_sampling_params: params.update(custom_sampling_params)

            for i in range(3):
                try:
                    res = self._llm_client.chat(messages=messages, sampling_params=params)
                    # 适配返回格式
                    if isinstance(res, dict): return res
                    return {"role": "assistant", "content": res}
                except Exception as e:
                    logger.warning(f"LLM chat attempt {i + 1} error: {e}")
                    time.sleep(2 * (2**i))
            raise RuntimeError("LLM client failed after 3 attempts")

        return llm_chat

    def _reconstruct_trajectory(self, prompt_tokens: List[int], response_tokens: List[int], data_id: str) -> Optional[Trajectory]:
        """将 token 还原为 Trajectory 对象"""
        try:
            parse_result = parse_response_ids_to_steps(response_tokens, self._tokenizer)
            steps = []
            for step_data in parse_result.steps:
                # 兼容 step_data 可能的结构
                act = step_data.get('action_text', '').strip()
                obs = step_data.get('observation_text', '').strip()
                
                if act: steps.append({"role": "assistant", "content": act})
                if obs: steps.append({"role": "user", "content": obs}) # 假设 Observation 归为 User
            
            if not steps: return None
            
            return Trajectory(
                data_id=data_id,
                query=self._tokenizer.decode(prompt_tokens, skip_special_tokens=True),
                steps=steps,
                reward=Reward(outcome=0.0)
            )
        except Exception as e:
            logger.warning(f"[Hindsight] Reconstruct failed: {e}")
            return None

    def _save_tasks(self, tasks: List[TaskObjective]):
        """保存任务到文件"""
        with self._lock:
            with open(self.save_path, "a", encoding="utf-8") as f:
                for t in tasks:
                    # 构造要保存的 JSON 结构
                    record = {
                        "task_id": t.task.id,
                        "query": t.objective,
                        "ground_truth": t.ground_truth,
                        "confidence": t.confidence,
                        "source": "hindsight_verified"
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")