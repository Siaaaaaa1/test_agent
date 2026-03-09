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
# 引用新版 DashScopeClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.task_manager.base import LlmClient
from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.utils.step_parser import parse_response_ids_to_steps

# =============================================================================
# PART 1: Prompts & Parsing Logic (已按要求更换所有例子)
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
        role = step.get("role") if isinstance(step, dict) else getattr(step, "role", None)
        content = step.get("content") if isinstance(step, dict) else getattr(step, "content", "")
        
        if role == "assistant" and idx + 1 < len(traj.steps):
            next_step = traj.steps[idx + 1]
            next_role = next_step.get("role") if isinstance(next_step, dict) else getattr(next_step, "role", None)
            next_content = next_step.get("content") if isinstance(next_step, dict) else getattr(next_step, "content", "")

            if next_role in ["tool", "user"]:
                observation = next_content
                res.append((content, observation))
    return res

def get_task_summarize_prompt(
    trajectories: Sequence[Trajectory],
    old_objectives: Sequence[TaskObjective],
    profile: EnvProfile | None,
) -> tuple[str, str]:
    x = ""
    for idx, traj in enumerate(trajectories):
        pairs = _get_action_observation_pair(traj)
        x += f"## Record {idx}\n### History\n"
        for step_idx, history in enumerate(pairs):
            x += f""">>> STEP {step_idx} <<<\n<|ACTION|>\n{history[0]}\n<|END|>\n<|OBSERVATION|>\n{history[1]}\n<|END|>\n"""
        if traj.reward:
            x += f"### Reward: {traj.reward.outcome}\n{getattr(traj.reward, 'reason', '')}\n"

    old_queries = [obj.objective for obj in old_objectives if hasattr(obj, 'objective') and obj.objective]
    user_prompt = f"""Please analyze the following agent interaction sequence and abstract specific tasks from it:

{x}

# Old Objectives
You have already explored the following objectives:

{old_queries}

Please avoid repeating these objectives.

# Task Requirements

{profile.get_task_preference_instruction() if profile else "Please follow the instructions to generate tasks."}

# Now Start

Please identify the specific tasks the agent is attempting to complete in these interactions, and abstract them into clear task descriptions and queries following the specified format.
"""
    return AGENT_SUMMARIZE_SYSTEM_PROMPT, user_prompt

def parse_tasks_from_response(base_task: Task, response: str) -> list[TaskObjective]:
    tasks: list[TaskObjective] = []
    try:
        task_matches = re.findall(r"<task>(.*?)</task>", response, re.DOTALL)
        for task_content in task_matches:
            try:
                t = json.loads(task_content)
                if not all(k in t for k in ["query", "confidence", "action_sequence"]):
                    continue
                
                new_task = copy.deepcopy(base_task)
                new_task.query = t["query"]
                new_task.open_query = True 
                
                x = TaskObjective(task=new_task, confidence=t["confidence"], reward=None)
                x.objective = t["query"]
                x.ground_truth = t["action_sequence"]
                tasks.append(x)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logger.error(f"Error parsing tasks: {e}")
    return tasks

# =============================================================================
# PART 2: Evaluator Components
# =============================================================================

class TrajectoryEvaluator:
    def __init__(self, client: DashScopeClient):
        self.client = client

    def evaluate_trajectory_success(self, task: TaskObjective, trajectory: Trajectory) -> bool:
        try:
            summary = self._create_trajectory_summary(trajectory)
            final_obs = "[no observation]"
            for step in reversed(trajectory.steps):
                role = step.get('role') if isinstance(step, dict) else getattr(step, 'role', '')
                if role != 'assistant':
                    final_obs = step.get('content') if isinstance(step, dict) else getattr(step, 'content', '')
                    break
            
            prompt = f"""You are a strict task evaluation expert. Your goal is to determine whether the following multi-step agent trajectory successfully completed the assigned task.

    # Task Details
    - Query: {task.objective}

    # Execution Summary
    - Trajectory Summary:
    {summary}

    - Final Observation: {final_obs}

    # Evaluation Instructions
    Carefully analyze the trajectory to determine if the task was truly completed.

    # Format Your Response Strictly As:

    Success: [true/false]
    Reason: [Concise explanation]
    """
            response = self.client.chat_with_retry(
                messages=[{"role": "user", "content": prompt}], 
                model="qwen3.5-plus", 
                temperature=0.0
            )
            
            if not response: return False
            match = re.search(r"Success:\s*(true|false)", response, re.IGNORECASE)
            return match.group(1).lower() == "true" if match else False
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return False
    
    def _create_trajectory_summary(self, traj: Trajectory) -> str:
        summary_blocks = []
        for i, step in enumerate(traj.steps):
            role = step.get('role') if isinstance(step, dict) else getattr(step, 'role', 'unknown')
            content = step.get('content') if isinstance(step, dict) else getattr(step, 'content', '')
            summary_blocks.append(f"(Step {i + 1}) {role}: {content[:200]}...")
        return "\n".join(summary_blocks)

# =============================================================================
# PART 3: Hindsight Manager
# =============================================================================

class HindsightManager:
    def __init__(self, 
                 llm_client: DashScopeClient, 
                 tokenizer, 
                 save_path: str = "tasks_explored/hindsight_supplement.jsonl",
                 num_threads: int = 4):
        
        self._llm_client = llm_client
        self._tokenizer = tokenizer
        self.save_path = save_path
        self._num_threads = num_threads
        
        # 强制注入 100 RPM / 20 并发限制
        self._llm_client._max_rpm = 100
        self._llm_client._semaphore = threading.BoundedSemaphore(20)
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self._lock = threading.Lock()
        self._validator = TrajectoryEvaluator(llm_client)

    def process_failed_batch(self, prompts, responses, scores, task_ids, threshold: float = 0.0):
        failed_tasks_data = []
        for i, score in enumerate(scores):
            if score <= threshold:
                traj = self._reconstruct_trajectory(prompts[i], responses[i], task_ids[i])
                if traj and len(traj.steps) > 0:
                    failed_tasks_data.append((traj, task_ids[i]))

        if not failed_tasks_data: return

        logger.info(f"[Hindsight] Processing {len(failed_tasks_data)} failed trajectories...")
        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            for traj, tid in failed_tasks_data:
                executor.submit(self._execute_hindsight_strategy, traj, tid)

    def _execute_hindsight_strategy(self, traj: Trajectory, original_task_id: str):
        try:
            # 1. 生成候选任务 (使用 qwen3.5-plus)
            sys_p, user_p = get_task_summarize_prompt(trajectories=[traj], old_objectives=[], profile=None)
            
            resp_dict = self._get_llm_chat_fn()(
                [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
                model="qwen3.5-plus",
                temperature=0.7
            )
            
            base_task = Task(task_id=original_task_id, env_type="hindsight")
            candidate_tasks = parse_tasks_from_response(base_task, resp_dict["content"])
            
            valid_results = []
            for task_obj in candidate_tasks:
                if self._validator.evaluate_trajectory_success(task_obj, traj):
                    
                    # 3. 净化并重写 GT
                    final_obj = self._rewrite_new_gt(task_obj, traj)
                    
                    uid = f"hind_{original_task_id}_{uuid.uuid4().hex[:6]}"
                    final_obj.task.task_id = uid
                    if final_obj.task.metadata is None: final_obj.task.metadata = {}
                    final_obj.task.metadata.update({
                        "source_task_id": original_task_id,
                        "generation_type": "hindsight_evolved",
                        "summary_analysis_process": resp_dict["content"]
                    })
                    valid_results.append(final_obj)

            if valid_results:
                self._save_tasks(valid_results)

        except Exception as e:
            logger.error(f"[Hindsight] Strategy failed for {original_task_id}: {e}")

    def _rewrite_new_gt(self, task: TaskObjective, trajectory: Trajectory) -> TaskObjective:
        sys_p, user_p = get_task_summarize_prompt([trajectory], [task], None)
        llm_out = self._get_llm_chat_fn()(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            model="qwen3.5-plus"
        )["content"]
        
        tasks_list = parse_tasks_from_response(task.task, llm_out)
        if tasks_list:
            task.ground_truth = tasks_list[0].ground_truth
        return task   

    def _get_llm_chat_fn(self, sampling_params: Optional[dict] = None) -> Callable:
        def llm_chat(messages: list[dict], **kwargs) -> dict:
            params = {**(sampling_params or {}), **kwargs}
            res = self._llm_client.chat_with_retry(messages=messages, **params)
            return {"role": "assistant", "content": res or ""}
        return llm_chat

    def _reconstruct_trajectory(self, prompt_tokens, response_tokens, data_id) -> Optional[Trajectory]:
        try:
            parse_result = parse_response_ids_to_steps(response_tokens, self._tokenizer)
            steps = []
            for s in parse_result.steps:
                act = s.get('action_text', '').strip()
                obs = s.get('observation_text', '').strip()
                if act: steps.append({"role": "assistant", "content": act})
                if obs: steps.append({"role": "user", "content": obs})
            if not steps: return None
            return Trajectory(
                data_id=data_id,
                query=self._tokenizer.decode(prompt_tokens, skip_special_tokens=True),
                steps=steps,
                reward=Reward(outcome=0.0)
            )
        except: return None

    def _save_tasks(self, tasks: List[TaskObjective]):
        with self._lock:
            with open(self.save_path, "a", encoding="utf-8") as f:
                for t in tasks:
                    record = {
                        "task_id": t.task.task_id,
                        "query": t.objective,
                        "ground_truth": t.ground_truth,
                        "confidence": t.confidence,
                        "metadata": t.task.metadata,
                        "timestamp": time.time()
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

import uuid