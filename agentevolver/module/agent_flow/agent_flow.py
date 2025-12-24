import time
import os

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
from typing import Any, Dict, List, Union, Optional
import threading
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig, ExperienceWorker

# 全局锁，用于控制日志生成的线程安全
log_generate_lock = threading.Lock()

class AgentFlow(BaseAgentFlow):
    """
    AgentFlow 类：实现了具体的 Agent 与环境交互的循环逻辑 (Think-Act Loop)。
    继承自 BaseAgentFlow。
    """

    def __init__(self, reward_calculator:Optional[RewardCalculator]=None, **kwargs):
        """
        初始化 AgentFlow 实例。

        Args:
            reward_calculator (Optional[RewardCalculator]): 可选的奖励计算器。如果提供，将使用它来计算最终奖励（如基于 LLM 的评分）；否则使用环境自带的评估函数。
            **kwargs: 传递给基类的其他关键字参数 (config, tokenizer, llm_chat_fn)。
        """
        super().__init__(**kwargs)  # ⭐ 调用基类构造函数，初始化配置、Tokenizer 和 LLM 接口
        self._reward_calculator = reward_calculator
        # self._enable_context_generator=self.config.experience_maker.enable_context_generator

        # 预先编码用户指令和助手回复的模板 Token，用于后续处理
        self.instruction_template_ids = self.tokenizer.encode("user\n")  
        self.response_template_ids = self.tokenizer.encode("assistant\n")  
        
        # sparse 标志：指示是否使用稀疏奖励（Sparse Reward，通常 0 或 1）
        self.sparse = self.config.actor_rollout_ref.rollout.sparse  
        
        # 上下文管理器 (CMT) 实例占位符
        self.cmt: Union[Linear_CMT, LinearThinkCMT] = None
        
        # 控制台调试模式开关
        self.console_debug_mode: bool = self.config.actor_rollout_ref.rollout.debug_llm_io
        
        # 初始化经验工作者，负责管理经验回放（Experience Replay）相关的逻辑
        self.exp_worker = ExperienceWorker(config=self.config)


    def execute(self, context_manager, init_messages: List[dict], env: EnvClient, instance_id: str, tmux, stop, thread_index, task_id, traj_exp_config, data_id="", rollout_id="", query="", **kwargs) -> Linear_CMT:
        """
        核心执行逻辑：管理 AI Agent 与环境的交互，生成轨迹、处理经验并计算奖励。

        Args:
            context_manager (ContextManager): 当前任务的上下文管理器 (负责维护 Prompt History)。
            init_messages (List[dict]): 任务的初始消息列表 (通常包含 System Prompt 和 User Query)。
            env (EnvClient): 环境客户端，用于与远程环境通信。
            instance_id (str): 当前运行的环境实例 ID。
            tmux (dict): 用于跨线程状态监控的字典 (记录 step, token 等)。
            stop (list): 停止标志列表，用于检查是否应该提前终止当前线程。
            thread_index (int): 当前线程的索引。
            task_id (str): 任务 ID。
            traj_exp_config (TrajExpConfig): 轨迹的经验配置 (控制是否插入历史经验)。
            data_id (str, optional): 数据 ID。默认为 ""。
            rollout_id (str, optional): Rollout ID。默认为 ""。
            query (str, optional): 查询字符串。默认为 ""。
            **kwargs: 其他参数。

        Returns:
            Linear_CMT: 执行完成后的上下文管理器对象 (本质上也是生成的 Trajectory)。
        """
        self.cmt = context_manager
        
        # 针对 Qwen3 模型的特殊处理：添加 /no_think 标记以禁用思维链（如果配置要求）
        add_nothink = self.config.actor_rollout_ref.rollout.use_qwen3 

        # 1. 🚀 初始化消息和经验
        # 将本次任务的 query 注入配置
        traj_exp_config.query = query
        
        # 调用 exp_worker 处理初始消息，可能会在 Prompt 中插入检索到的相关经验 (RAG / Few-shot)
        init_messages, traj_exp_config = self.exp_worker.manage_rollout_context(
                init_messages=init_messages,
                traj_exp_config=traj_exp_config
                )
        
        # 将经验配置元数据保存到轨迹中
        self.cmt.metadata["task_train_exp_mode"] = traj_exp_config.train_mode
        self.cmt.metadata["add_exp"] = traj_exp_config.add_exp
        self.cmt.metadata["experience_list"] = traj_exp_config.experience_list
        
        # 将处理后的初始消息保存到上下文管理器中
        self.cmt.save_init_input(init_messages, add_nothink)

        request_id: str = ""
        err_in_generating = False
        err_in_env = False
        
        # ---------------- 交互循环 (ReAct Loop) ----------------
        for act_step in range(self.max_steps):
            # 2. 🔄 更新线程进度
            tmux['step'][thread_index] = act_step
            # 检查是否收到外部停止信号 (例如其他线程已经找到了答案，不需要再跑了)
            if (stop is not None) and stop[thread_index]: 
                self.cmt.discarded = True
                break

            # 3. ⏮️ 准备上下文 (Prompt)
            try:
                # 获取 LLM 的输入历史
                step_input_message_arr = self.cmt.prepare_next_llm_context()  
            except Exception as e:
                # 如果构建 Prompt 失败，打印当前状态以便调试
                print_listofdict(self.cmt.to_role_content(self.cmt.full_context), mod='exception', header="Before Crash")
                raise e

            # 4. ⚠️ 检查 Token 溢出
            is_safe: bool = self.cmt.check_context_token_num_safe(step_input_message_arr)  
            if not is_safe:
                logger.warning(f"Token overflow detected at step {act_step}. Current token count exceeds the limit.")
                self.cmt.is_terminated = False # 标记为未完成
                break

            # 5. 🤖 调用 LLM (Think/Act)
            # 发送请求给 LLM，获取回复 (content)
            llm_output = self.llm_chat_fn(step_input_message_arr, request_id=request_id)  
            
            # 再次检查停止信号
            if (stop is not None) and stop[thread_index]:  
                self.cmt.discarded = True
                break

            # 6. 💾 保存 LLM 输出
            # 将 LLM 的回复记录到上下文管理器中
            self.cmt.save_llm_output(llm_output, input_msg_ref=step_input_message_arr)  
            # 更新生成的 Token 统计
            tmux['token'][thread_index] += self.cmt.generated_token_cnt

            # 7. 🌍 与环境交互 (Environment Interaction)
            try:
                # 准备发送给环境的动作 (从 LLM 输出中提取代码或指令)
                action_content = self.cmt.prepare_world_interaction()
                # 发送 step 请求给环境客户端
                env_output = env.step(instance_id, {"content": action_content, "role": "assistant"})  
                
                # 确保环境返回格式正确
                assert len(env_output['state'])==1
                env_output["state"] = env_output["state"][0]
                
                # 如果环境返回的是 Tool Role (OpenAI 格式)，转换为 User Message (Qwen/通用格式)
                if env_output["state"]["role"] == "tool":
                    env_output["state"] = convert_tool_to_user_message(env_output["state"], self.tokenizer, format="qwen")
                
                # 控制台调试输出
                if self.console_debug_mode:
                    print_listofdict(
                        step_input_message_arr +
                        [{'role': 'llm_latest', 'content': llm_output['content']}] +
                        [{'role': 'env',        'content': env_output["state"]['content']}]
                    , mod='c')
            except Exception as e:
                # 捕获环境交互异常
                logger.bind(exception=True).exception(f"call env.step error with {e}")
                err_in_env = True
                self.cmt.is_terminated = False # 发生错误，标记为未完成
                # 构造一个错误的 Observation 反馈给 Agent (或者直接终止)
                state = {"content": str(e), "role": "user"}
                env_output = {
                    "reward": 0,
                    "is_terminated": True,
                    "state": state,
                }

            # 8. 📥 保存环境输出 (Observation)
            state = env_output["state"]
            # 移除不需要的 tool_calls 字段
            state.pop('tool_calls', None)
            # 将环境的反馈记录到上下文管理器中
            self.cmt.save_env_output(state, input_msg_ref=step_input_message_arr, add_nothink=add_nothink)  

            # 9. 🔚 判断任务是否终止
            self.cmt.is_terminated = env_output["is_terminated"]
            if self.cmt.is_terminated or err_in_env:
                break
        
        # ---------------- 循环结束 ----------------

        # 标记线程状态为已完成
        tmux['step'][thread_index] = -1

        # 10. 🏆 计算奖励 (Reward Calculation)
        if self._reward_calculator is not None:
            # 如果配置了高级奖励计算器 (如 LLM-as-a-Judge)，使用它
            grader_res = self._reward_calculator.calculate_reward(self.cmt, env, instance_id)  
            score = grader_res["score"] 
            reason = grader_res["reason"] or "No reason provided."
        else:
            # 否则使用环境自带的评估函数 (通常是 Outcome Reward)
            score = env.evaluate(instance_id, params={"sparse": self.sparse})  
            reason = "Outcome 1 = success, 0 = failure."

        # 计算成功率 (通常 score >= 1 视为成功)
        if score >= 1: success_rate = 1.0
        else: success_rate = 0.0

        # 将奖励信息打包并保存到上下文管理器中
        # madness 是某种衡量 Agent 行为疯狂程度或不可控程度的指标
        self.cmt.reward = Reward(outcome=score, success_rate=success_rate, madness=self.cmt.compute_madness(), description=reason)  
        
        # 对奖励进行可能的修补或后处理
        self.cmt.reward = self.cmt.reward_patch(self.cmt.reward)
        
        # 移除最后一条上下文 (通常是为了整理数据格式，比如去掉最后的 User 回复以便于训练预测)
        self.cmt.remove_last_context()

        # 生成日志 (使用锁保证线程安全)
        with log_generate_lock:
            self.cmt.generate_log(task_id=task_id)  


        return self.cmt