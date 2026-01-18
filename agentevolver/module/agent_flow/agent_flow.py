import time
import os
import json
import threading
from typing import Any, Dict, List, Union, Optional

from loguru import logger
from agentevolver.client.em_client import EMClient
from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.utils.utils import convert_tool_to_user_message
from agentevolver.schema.trajectory import Reward, Trajectory
from best_logger import register_logger, print_dict, print_listofdict
from agentevolver.module.context_manager.cmt_linear import Linear_CMT, ExtendedMessage
from agentevolver.module.context_manager.cmt_linear_think import LinearThinkCMT
from agentevolver.module.context_manager.cmt_context_clip import SelfContextClipCMT
from agentevolver.module.agent_flow.reward_calculator import RewardCalculator
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig, ExperienceWorker

# 全局锁，用于控制日志生成的线程安全
log_generate_lock = threading.Lock()
# 新增：用于多线程安全写入同一个 JSON 文件锁
json_save_lock = threading.Lock()

class AgentFlow(BaseAgentFlow):
    """
    AgentFlow 类：实现了具体的 Agent 与环境交互的循环逻辑 (Think-Act Loop)。
    继承自 BaseAgentFlow。
    """

    def __init__(self, reward_calculator:Optional[RewardCalculator]=None, **kwargs):
        """
        初始化 AgentFlow 实例。
        """
        super().__init__(**kwargs)
        self._reward_calculator = reward_calculator
        self.instruction_template_ids = self.tokenizer.encode("user\n")  
        self.response_template_ids = self.tokenizer.encode("assistant\n")  
        self.sparse = self.config.actor_rollout_ref.rollout.sparse  
        self.cmt: Union[Linear_CMT, LinearThinkCMT] = None
        self.console_debug_mode: bool = self.config.actor_rollout_ref.rollout.debug_llm_io
        self.exp_worker = ExperienceWorker(config=self.config)

    def _save_to_json(self, task_id: str, instance_id: str, rollout_id: str):
        """
        私有方法：将当前对话保存并追加到 ./tmp/all_trajectories.json
        """
        folder_path = "./tmp"
        file_path = os.path.join(folder_path, "all_trajectories.json")
        
        # 确保文件夹存在
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 构造本次对话的数据结构
        # 使用 self.cmt.to_role_content 将消息对象转为标准 role/content 字典列表
        current_entry = {
            "task_id": task_id,
            "instance_id": instance_id,
            "rollout_id": rollout_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reward": {
                "outcome": self.cmt.reward.outcome,
                "success_rate": self.cmt.reward.success_rate,
                "reason": self.cmt.reward.description
            },
            "trajectory": self.cmt.to_role_content(self.cmt.full_context)
        }

        # 使用锁确保多线程追加写入时不冲突
        with json_save_lock:
            data_list = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data_list = json.load(f)
                        if not isinstance(data_list, list):
                            data_list = []
                except Exception:
                    data_list = []
            
            data_list.append(current_entry)

            with open(file_path, 'w', encoding='utf-8') as f:
                # indent=4 实现格式化保存，ensure_ascii=False 保证中文正常显示
                json.dump(data_list, f, ensure_ascii=False, indent=4)

    def execute(self, context_manager, init_messages: List[dict], env: EnvClient, instance_id: str, tmux, stop, thread_index, task_id, traj_exp_config, data_id="", rollout_id="", query="", ground_truth="", **kwargs) -> Linear_CMT:
        """
        核心执行逻辑：管理 AI Agent 与环境的交互，生成轨迹、处理经验并计算奖励。
        [DEBUG MODE] 已添加详细流程日志
        """
        # [DEBUG LOG] Start
        logger.info(f"[DEBUG-FLOW] Thread-{thread_index} Task-{task_id} Rollout-{rollout_id}: Starting execute. Max steps: {self.max_steps}")

        self.cmt = context_manager
        add_nothink = self.config.actor_rollout_ref.rollout.use_qwen3 

        # 1. 初始化消息和经验
        traj_exp_config.query = query
        init_messages, traj_exp_config = self.exp_worker.manage_rollout_context(
                init_messages=init_messages,
                traj_exp_config=traj_exp_config
                )
        
        self.cmt.metadata.update({
            "task_train_exp_mode": traj_exp_config.train_mode,
            "add_exp": traj_exp_config.add_exp,
            "experience_list": traj_exp_config.experience_list
        })
        
        self.cmt.save_init_input(init_messages, add_nothink)

        request_id: str = ""
        err_in_env = False
        
        # [新增] 读取过程奖励开关 (默认开启)
        enable_gt_process_reward = self.config.actor_rollout_ref.rollout.get("enable_gt_process_reward", False)
        
        # [新增] 检查当前 RewardCalculator 是否支持过程奖励 (duck typing)
        can_calculate_process = (
            self._reward_calculator is not None and 
            hasattr(self._reward_calculator, 'calculate_step_reward')
        )

        # [新增] 初始化环境耗时统计
        total_env_time = 0.0

        # ---------------- 交互循环 (ReAct Loop) ----------------
        for act_step in range(self.max_steps):
            # [DEBUG LOG] Step start
            # logger.info(f"[DEBUG-FLOW] Thread-{thread_index} Step {act_step}: Starting...")

            tmux['step'][thread_index] = act_step
            if (stop is not None) and stop[thread_index]: 
                logger.info(f"[DEBUG-FLOW] Thread-{thread_index} Step {act_step}: Stopped externally.")
                self.cmt.discarded = True
                break

            try:
                step_input_message_arr = self.cmt.prepare_next_llm_context()  
            except Exception as e:
                print_listofdict(self.cmt.to_role_content(self.cmt.full_context), mod='exception', header="Before Crash")
                raise e

            is_safe: bool = self.cmt.check_context_token_num_safe(step_input_message_arr)  
            if not is_safe:
                logger.warning(f"Token overflow at step {act_step}.")
                # [DEBUG LOG] Token Overflow
                logger.warning(f"[DEBUG-FLOW] Thread-{thread_index} Step {act_step}: Token Overflow triggered! Breaking loop.")
                self.cmt.is_terminated = False 
                break

            # [DEBUG LOG] Before LLM
            logger.info(f"[DEBUG-FLOW] Thread-{thread_index} Step {act_step}: Requesting LLM...")
            llm_st = time.time()
            
            # --- LLM Call ---
            llm_output = self.llm_chat_fn(step_input_message_arr, request_id=request_id)  
            # ----------------
            
            logger.info(f"[DEBUG-FLOW] Thread-{thread_index} Step {act_step}: LLM responded in {time.time()-llm_st:.2f}s. Content len: {len(llm_output.get('content', ''))}")
            
            if (stop is not None) and stop[thread_index]:  
                self.cmt.discarded = True
                break

            self.cmt.save_llm_output(llm_output, input_msg_ref=step_input_message_arr)  
            tmux['token'][thread_index] += self.cmt.generated_token_cnt

            step_process_reward = 0.0

            try:
                action_content = self.cmt.prepare_world_interaction()
                
                # [修改] 增加计时逻辑
                st_time = time.time()
                # [DEBUG LOG] Before Env
                logger.info(f"[DEBUG-FLOW] Thread-{thread_index} Step {act_step}: Calling Env step...")
                
                # --- Env Call ---
                env_output = env.step(instance_id, {"content": action_content, "role": "assistant"})  
                # ----------------
                
                env_dur = time.time() - st_time
                logger.info(f"[DEBUG-FLOW] Thread-{thread_index} Step {act_step}: Env step finished in {env_dur:.2f}s.")
                
                # 累加环境执行时间
                total_env_time += env_dur
                
                assert len(env_output['state']) == 1
                env_output["state"] = env_output["state"][0]
                
                # [新增] 计算过程奖励逻辑
                # 仅当 env.step 成功，且配置开启，且计算器支持时调用
                if enable_gt_process_reward and can_calculate_process:
                    # 调用 APIProcessRewardCalculator 的特有方法
                    step_process_reward = self._reward_calculator.calculate_step_reward(action_content)
                    
                    # 将奖励累加到 env_output 中 (假设 env 可能返回 sparse reward = 0)
                    current_val = env_output.get('reward', 0.0)
                    if current_val is None: current_val = 0.0
                    env_output['reward'] = current_val + step_process_reward
                    
                    if step_process_reward > 0 and self.console_debug_mode:
                        logger.info(f"[Process Reward] Step {act_step}: +{step_process_reward}")

                if env_output["state"]["role"] == "tool":
                    env_output["state"] = convert_tool_to_user_message(env_output["state"], self.tokenizer, format="qwen")
                
                if self.console_debug_mode:
                    print_listofdict(
                        step_input_message_arr +
                        [{'role': 'llm_latest', 'content': llm_output['content']}] +
                        [{'role': 'env',        'content': env_output["state"]['content']}]
                    , mod='c')
            except Exception as e:
                logger.bind(exception=True).exception(f"call env.step error with {e}")
                err_in_env = True
                self.cmt.is_terminated = False 
                state = {"content": str(e), "role": "user"}
                env_output = {"reward": 0, "is_terminated": True, "state": state}

            state = env_output["state"]
            state.pop('tool_calls', None)
            self.cmt.save_env_output(state, input_msg_ref=step_input_message_arr, add_nothink=add_nothink)  

            self.cmt.is_terminated = env_output["is_terminated"]
            if self.cmt.is_terminated or err_in_env:
                logger.info(f"[DEBUG-FLOW] Thread-{thread_index} Step {act_step}: Terminated (Done={self.cmt.is_terminated}, Err={err_in_env}).")
                break
        
        # ---------------- 循环结束 ----------------

        tmux['step'][thread_index] = -1

        logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: Exited Loop. Calculating Reward...")

        # 10. 计算奖励
        if self._reward_calculator is not None:
            # 这里调用的是 Outcome Reward 逻辑 (LLM Judge)
            logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: Using RewardCalculator. Calling calculate_reward...") 
            try:
                # [潜在卡死点 1] LLM Judge 调用
                grader_res = self._reward_calculator.calculate_reward(self.cmt, env, instance_id)  
                logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: calculate_reward returned successfully.")
                
                score = grader_res["score"] 
                reason = grader_res["reason"] or "No reason provided."
            except Exception as e:
                logger.error(f"[DEBUG-FLOW] Thread-{thread_index}: CRITICAL ERROR in calculate_reward: {e}")
                score = 0.0
                reason = f"Error during reward calculation: {e}"
        else:
            logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: No RewardCalculator. Using env.evaluate...")
            try:
                # [潜在卡死点 2] 环境评估调用
                score = env.evaluate(instance_id, params={"sparse": self.sparse})  
                logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: env.evaluate returned successfully.")
            except Exception as e:
                logger.error(f"[DEBUG-FLOW] Thread-{thread_index}: CRITICAL ERROR in env.evaluate: {e}")
                score = 0.0
            
            reason = "Outcome 1 = success, 0 = failure."

        logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: Reward Outcome: {score}. Preparing Reward object...")

        success_rate = 1.0 if score >= 1 else 0.0
        self.cmt.reward = Reward(outcome=score, success_rate=success_rate, madness=self.cmt.compute_madness(), description=reason)  
        
        # [新增] 将统计的环境时间写入 Reward 的 metadata 中
        if self.cmt.reward.metadata is None:
            self.cmt.reward.metadata = {}
        self.cmt.reward.metadata["env_time"] = total_env_time

        self.cmt.reward = self.cmt.reward_patch(self.cmt.reward)
        self.cmt.remove_last_context()

        # 生成日志
        logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: Waiting for log_generate_lock...")
        
        # [潜在卡死点 3] 线程锁死锁
        # 建议：如果不需要生成 html/text log，可以暂时注释掉 with 块来测试
        try:
            with log_generate_lock:
                logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: Acquired lock. Generating log...")
                self.cmt.generate_log(task_id=task_id)  
                logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: Log generated. Releasing lock.")
        except Exception as e:
            logger.error(f"[DEBUG-FLOW] Thread-{thread_index}: Error inside log_generate_lock: {e}")

        # 11. 追加对话到同一个 JSON 文件中
        logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: Calling _save_to_json...")
        try:
            # [潜在卡死点 4] 文件 IO 阻塞
            self._save_to_json(task_id, instance_id, rollout_id)
            logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: _save_to_json finished.")
        except Exception as e:
            logger.error(f"[DEBUG-FLOW] Thread-{thread_index}: Error in _save_to_json: {e}")

        logger.info(f"[DEBUG-FLOW] Thread-{thread_index}: Execute Finished. Returning CMT.")
        return self.cmt