import copy
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import random
import re
import os
from loguru import logger
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import (pad_sequence_to_length)

from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.module.env_manager.env_worker import EnvWorker
from agentevolver.module.trainer.ae_async_llm_server_manager import BaAsyncLLMServerManager
from agentevolver.module.task_manager.rewards import grader_manager
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory, Sample
from agentevolver.utils.step_parser import parse_response_ids_to_steps
# do not delete this line
from agentevolver.module.task_manager.rewards import LlmAsJudgeRewardCalculator,LlmAsJudgeRewardCalculatorWithGT,LlmAsJudgeBinaryRewardCalculator,LlmAsJudgeBinaryRewardCalculatorWithGT,EnvGrader, AvgBinaryGTJudge, AvgLlmJudge
from beast_logger import register_logger
from agentevolver.module.exp_manager.exp_manager import TaskExpConfig, TrajExpConfig


def init_logger(experiment_name):
    """
    初始化日志记录器，设置实验名称和日志环境。

    Args:
        experiment_name (str): 当前实验的名称，用于生成日志文件的路径。
    """
    if 'BEST_LOGGER_INIT' in os.environ: return  # 防止在 Ray 环境中重复初始化
    os.environ['BEST_LOGGER_INIT'] = '1'
    os.environ['BEST_LOGGER_WEB_SERVICE_URL'] = "http://127.0.0.1:8181/"
    from datetime import datetime
    # 构建最终的日志路径：experiments/{experiment_name}/trace_rollout/{timestamp}
    final_log_path = os.path.join( "experiments", experiment_name, "trace_rollout", datetime.now().strftime("%Y_%m_%d_%H_%M"))
    # 指定不输出到控制台的模块列表，避免控制台刷屏
    non_console_mods = ["conversation", "rollout", "token_clip", "bad_case", "env_clip"]
    # 注册日志记录器，开启 evaluation 和 exception 模块，设置基础路径
    register_logger(mods=["evaluation", "exception"], non_console_mods=non_console_mods, auto_clean_mods=[], base_log_path=final_log_path, debug=False)
    print('Run `beast_logger_go` and click the url to inspect rollout logs. Continue in 5 seconds')
    time.sleep(2.5)



class ParallelEnvManager(object):
    """
    并行环境管理器。
    负责管理多个任务的并行执行 (Rollout)，处理重试逻辑、日志记录，并使用 LLM 生成响应。
    最终将任务执行生成的轨迹 (Trajectory) 收集并转换为训练数据格式。
    """
    def __init__(self, config: DictConfig, async_rollout_manager: BaAsyncLLMServerManager, max_parallel: int,
                 max_llm_retries: int = 3, **kwargs):
        """
        初始化 ParallelEnvManager。

        Args:
            config (DictConfig): 包含所有必要参数的配置字典 (通常来自 Hydra)。
            async_rollout_manager (BaAsyncLLMServerManager): 异步 LLM 服务器管理器，负责处理 LLM 推理请求。
            max_parallel (int): 最大并行任务数，即线程池的大小。
            max_llm_retries (int, optional): LLM 请求失败时的最大重试次数。默认为 3。
            **kwargs: 其他关键字参数。
        """
        init_logger(experiment_name=config.trainer.experiment_name)  # ⭐ 初始化日志系统
        super().__init__(**kwargs)

        self.config: DictConfig = config
        self.async_rollout_manager: BaAsyncLLMServerManager = async_rollout_manager
        self.max_parallel: int = max_parallel
        self.max_llm_retries: int = max_llm_retries

        # 每个任务需要采样的轨迹数量 (通常用于 Best-of-N 或 PPO 采样)
        self.rollout_n = config.actor_rollout_ref.rollout.n
        # 获取模型名称和分词器
        self.model_name = self.async_rollout_manager.chat_scheduler.model_name
        self.tokenizer = self.async_rollout_manager.chat_scheduler.completion_callback.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.rollout_config = config.actor_rollout_ref.rollout

        # self.experience_template = config.hybrid_experience_training.experience_template
        self.llm_mode = "local" # LLM 模式："local" 使用本地 FSDP worker，"remote" 使用外部服务器
        self.current_token = 0
        self.current_token_count_time = time.time()


    def get_llm_chat_fn(self, sampling_params: dict = None) -> callable:
        """
        获取用于与 LLM 对话的可调用函数。
        根据 `llm_mode` 返回 `llm_chat` (本地/FSDP) 或 `llm_chat_remote` (远程)。

        Args:
            sampling_params (dict, optional): 默认的采样参数 (如 temperature, top_p 等)。

        Returns:
            callable: 一个接受消息列表并返回 LLM 响应的函数。
        """

        def llm_chat(messages: List[Dict[str, str]],
                     custom_sampling_params: dict = None,
                     request_id: str = None) -> dict:
            """
            发送消息到 LLM 并返回助手的回复。包含重试逻辑。

            Args:
                messages (List[Dict[str, str]]): 消息列表，每个元素包含 "role" 和 "value" (或 "content")。
                custom_sampling_params (dict, optional): 自定义的采样参数，会覆盖默认参数。
                request_id (str, optional): 请求的唯一标识符，用于追踪。

            Returns:
                dict: 输入消息列表中的最后一个消息，通常包含助手的回复内容。
            """
            # TODO: sending sampling_params to rollout server
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)
            # 强制要求返回 logprobs 和 token IDs，这是 PPO 训练所必需的
            updated_sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})  # ⭐ 更新采样参数

            input_messages = copy.deepcopy(messages)
            weighted_addresses = self.async_rollout_manager.chat_scheduler.weighted_addresses
            # logger.info(f"weighted_addresses={weighted_addresses}")
            for i in range(self.max_llm_retries):
                try:
                    # 提交异步聊天补全请求
                    self.async_rollout_manager.submit_chat_completions(messages=input_messages,
                                                                       sampling_params=updated_sampling_params,
                                                                       request_id=request_id)  # ⭐ 提交聊天请求
                    break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1) # 简单的指数退避等待

            return input_messages[-1] # 返回填充了回复内容的最后一条消息

        def llm_chat_remote(messages: List[Dict[str, str]],
                           custom_sampling_params: dict = None,
                           request_id: str = None) -> dict:
            """
            发送消息到远程 LLM 服务器。逻辑与 llm_chat 类似，但针对远程调用进行了适配。
            """
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)
            updated_sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})  # ⭐ 更新采样参数
            input_messages = copy.deepcopy(messages)
            for i in range(self.max_llm_retries):
                try:
                    output_message = self.async_rollout_manager.submit_chat_completions(messages=input_messages,
                                                                                         sampling_params=updated_sampling_params,
                                                                                         request_id=request_id)  # ⭐ 提交聊天请求
                    break
                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(2**i)
            return output_message[-1]

        if self.llm_mode == "remote":
            return llm_chat_remote
        else:
            return llm_chat

    def step_status_printer(self, tmux):
        """
        打印当前并行环境的步骤状态，监控 Rollout 进度。
        显示不同步骤范围内的线程数量以及 Token 生成速率。

        Args:
            tmux (dict): 包含当前环境状态信息的字典，主要键为 'step' (当前步骤) 和 'token' (生成的 Token 数)。
                         这个字典在所有线程间共享。
        """
        # 初始化计数器，用于统计处于每个步骤范围的线程数
        step_counter = {}

        # 计算总 Token 数和自上次统计以来的时间
        current_token = sum(tmux['token'])
        current_time = time.time()
        delta_token = current_token - self.current_token
        delta_time = current_time - self.current_token_count_time
        self.current_token = current_token
        self.current_token_count_time = current_time
        token_gen_per_sec_str = f"{delta_token/delta_time:.2f} tokens/s" if delta_time > 0 else "N/A"

        # 将步骤分箱并统计每个箱中的线程数 (例如 0-5 步, 5-10 步)
        for step in tmux['step']:
            if step == -1: # -1 表示已终止的任务
                step_counter[(-1, 'terminated')] = step_counter.get((-1, 'terminated'), 0) + 1
                continue
            else:
                start = (step // 5) * 5
                end = start + 5
                step_counter[(start, end)] = step_counter.get((start, end), 0) + 1

        # 按箱的起始值对计数器进行排序
        step_counter = dict(sorted(step_counter.items(), key=lambda x: x[0][0]))  # ⭐ 排序以保证输出顺序

        # 准备打印缓冲区
        print_buf = []
        for (start, end), count in step_counter.items():
            if start != -1:
                print_buf += [f"[{start}-{end}]:{count} threads"]
        for (start, end), count in step_counter.items():
            if start == -1:
                print_buf += [f"[finished]:{count} threads"]

        # 打印 Rollout 进度
        print(f"Rollout progress ({token_gen_per_sec_str}): " + "  //  ".join(print_buf))



    def rollout_env_worker(self, task: Task, traj_exp_config: TrajExpConfig, data_id: str, rollout_id: str, mode: Literal["sample", "validate"],
                           thread_index: int, tmux: dict, stop:list, **kwargs) -> Trajectory:
        """
        在单个线程中处理单个任务的 Rollout 逻辑，处理重试和异常。

        Args:
            task (Task): 要处理的具体任务对象。
            traj_exp_config (TrajExpConfig): 该轨迹的经验配置（如是否添加经验、训练模式等）。
            data_id (str): 数据的唯一标识符（通常对应任务索引）。
            rollout_id (str): Rollout 的 ID（如果对同一个任务进行多次采样，用此区分）。
            mode (Literal["sample", "validate"]): 操作模式，'sample' 用于训练采样，'validate' 用于验证。
            thread_index (int): 线程索引，用于在 tmux 和 stop 列表中定位状态。
            tmux (dict): 共享的状态字典，用于记录进度。
            stop (list): 停止标志列表，用于外部控制线程停止。
            **kwargs: 其他关键字参数。

        Returns:
            Trajectory: 任务执行生成的轨迹对象。
        """
        max_retry = 4
        for retry in range(max_retry):
            try:

                # 准备采样参数
                sampling_params = dict(
                    n=1,
                    max_completion_tokens=self.rollout_config.response_length,
                    temperature=self.rollout_config.temperature,
                    top_p=self.rollout_config.top_p)

                # 验证模式下使用不同的采样参数
                if mode == "validate":
                    sampling_params["temperature"] = self.rollout_config.val_kwargs.temperature
                    sampling_params["top_k"] = self.rollout_config.val_kwargs.top_k
                    sampling_params["top_p"] = self.rollout_config.val_kwargs.top_p

                # 获取 LLM 对话函数
                llm_chat_fn = self.get_llm_chat_fn(sampling_params)
                # 获取奖励计算器
                reward_caculator=grader_manager.get_calculator(task.evaluator, task=task)
                
                # 初始化 AgentFlow，负责管理 Agent 与环境的交互流
                agent_flow: BaseAgentFlow = AgentFlow(
                    reward_calculator=reward_caculator,
                    llm_chat_fn=llm_chat_fn,
                    tokenizer=self.tokenizer,
                    config=self.config,
                    **kwargs
                )

                # 初始化 EnvWorker，负责具体的环境操作
                env_worker = EnvWorker(task=task, thread_index=thread_index, config=self.config, tokenizer=self.tokenizer)
                
                # 执行任务并生成轨迹
                trajectory: Trajectory = env_worker.execute(data_id=data_id, rollout_id=rollout_id, traj_exp_config=traj_exp_config, agent_flow=agent_flow, tmux=tmux, stop=stop) # ⭐ 执行任务生成轨迹
                return trajectory

            except Exception as e:
                if retry < max_retry - 1:
                    logger.bind(exception=True).exception(f"rollout_env_worker error: {e.args}, retrying {retry + 1}/{max_retry}")
                    time.sleep(2 ** retry)
                else:
                    logger.bind(exception=True).exception(f"rollout_env_worker failed after {max_retry} retries: {e.args}")
                    raise e

    def rollout(self, tasks: List[Task], task_exp_configs: List[TaskExpConfig], mode: Literal["sample", "validate"], epoch: str) -> List[Trajectory]:
        """
        并行执行任务列表。使用线程池管理并发，并具备自动重试机制。

        Args:
            tasks (List[Task]): 要处理的任务列表。
            task_exp_configs (List[TaskExpConfig]): 每个任务对应的经验配置列表。
            mode (Literal["sample", "validate"]): 模式，'sample' 或 'validate'。
            epoch (str): 当前 Epoch 标识符，用于日志和进度条显示。

        Returns:
            List[Trajectory]: 成功完成的任务生成的轨迹对象列表，已按 ID 排序。
        """
        traj_cmt_array = []
        # 确定每个任务的采样次数
        rollout_n = self.rollout_config.val_kwargs.n if mode == "validate" else self.rollout_n
        
        # 存储 Future 对象到参数的映射，用于错误处理和重试
        future_to_params: Dict[Future, Tuple[Task, TrajExpConfig, str, str, str, int, dict, list[bool]]] = {}

        # 初始化共享状态 tmux 和停止标志 stop
        tmux = {
            'step': [0 for _ in range(len(tasks) * rollout_n)],
            'token': [0 for _ in range(len(tasks) * rollout_n)],
        }
        stop = [False for _ in range(len(tasks) * rollout_n)]

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # 2. 提交所有任务到线程池
            for data_id, (task, task_exp_config) in enumerate(zip(tasks, task_exp_configs)):
                for rollout_id in range(rollout_n):
                    thread_index = data_id * rollout_n + rollout_id
                    add_exp = task_exp_config.add_exp[rollout_id]
                    train_mode = task_exp_config.train_mode
                    
                    # 构建单个轨迹的经验配置
                    traj_exp_config = TrajExpConfig(
                        add_exp=add_exp, train_mode=train_mode, task_id=task.task_id, data_id=data_id, rollout_id=rollout_id, mode=mode)

                    params = (task, traj_exp_config, str(data_id), str(rollout_id), mode, thread_index, tmux,stop)
                    future = executor.submit(self.rollout_env_worker, *params)
                    future_to_params[future] = params

            total_rollouts = len(future_to_params)
            pbar = tqdm(total=total_rollouts, desc=f"Epoch {epoch}: Collecting rollouts")

            # 3. 等待所有任务完成
            while future_to_params:
                # 轮询完成的 Future
                for future in as_completed(future_to_params):
                    # 获取对应的参数并从字典中移除
                    params = future_to_params.pop(future)
                    self.step_status_printer(tmux) # 打印进度状态

                    # 4. 获取结果并处理错误
                    try:
                        result = future.result()  # ⭐ 获取完成任务的结果

                        # 如果结果元数据中包含错误信息，尝试恢复
                        if 'error' in result.metadata:
                            error_msg = result.metadata['error']
                            logger.warning(f"Task {params[1]}-{params[2]} failed with metadata error: {error_msg}. Retrying... \n Task: {params[0]}")
                            # 大多数错误是网络或配额问题，等待一段时间后重试
                            time.sleep(30)
                            # 重置状态并重新提交任务
                            thread_index=params[5]
                            for k in tmux: tmux[k][thread_index] = 0
                            stop[thread_index]=False
                            new_future = executor.submit(self.rollout_env_worker, *params) # type: ignore
                            future_to_params[new_future] = params
                            continue

                        # 5. 任务成功，将结果添加到列表
                        traj_cmt_array.append(result)
                        pbar.update(1) # 更新进度条

                    except Exception as e:
                        # 处理未捕获的异常
                        logger.error(f"Task {params[1]}-{params[2]} raised an exception: {e}. Retrying... \n Task: {params[0]}")
                        # 重置状态并重新提交任务
                        thread_index=params[5]
                        for k in tmux: tmux[k][thread_index] = 0
                        stop[thread_index]=False
                        new_future = executor.submit(self.rollout_env_worker, *params) # type: ignore
                        future_to_params[new_future] = params
            pbar.close()

        # 计算并更新当前批次的成功率
        task_success_rate = np.mean([cmt.reward.success_rate for cmt in traj_cmt_array])
        for cmt in traj_cmt_array:
            cmt.current_batch_success_rate = np.mean(task_success_rate)

        # 对结果按 ID 排序，保证顺序一致性
        traj_cmt_array = sorted(traj_cmt_array, key=lambda x: (int(x.data_id), int(x.rollout_id)))
        return traj_cmt_array


    # TODO: define an extra class for trajectory-dataproto converting.
    def to_dataproto(self, cmt_array) -> DataProto:
        """
        将轨迹列表转换为 DataProto 对象，这是训练框架 (Verl) 所需的数据格式。

        Args:
            cmt_array (list): 待转换的轨迹列表。

        Returns:
            DataProto: 转换后的 DataProto 对象，包含张量数据和非张量元数据。
        """
        # 第一步：将轨迹转换为样本 (Sample) 对象，进行分词处理
        samples = self.trajectories_to_samples(cmt_array)  # ⭐ 分词

        # 第二步：将样本转换为 DataProto，进行填充 (Padding) 和张量化
        dataproto = self.samples_to_dataproto(samples)  # ⭐ 填充和转换

        return dataproto


    def get_extra(self, cmt):
        """
        从轨迹元数据中提取额外信息。

        Args:
            cmt (object): 包含元数据的轨迹对象。

        Returns:
            dict: 包含 'add_exp' (是否添加经验), 'task_train_expmode' (训练模式), 'experience_list' (经验列表) 的字典。
        """
        extras = {
            "add_exp": cmt.metadata.get("add_exp", None),  # ⭐ 从元数据获取 'add_exp'
            "task_train_expmode": cmt.metadata.get("task_train_exp_mode", None),  # ⭐ 从元数据获取 'task_train_exp_mode'
            "experience_list": cmt.metadata.get("experience_list", [])  # ⭐ 从元数据获取 'experience_list'
        }
        return extras


    def trajectories_to_samples(self, cmt_array: List) -> List[Sample]:
        """
        将轨迹列表转换为样本列表，并确保样本数量能被分布式训练的总 GPU 数整除。

        Args:
            cmt_array (List): 轨迹列表。

        Returns:
            List[Sample]: 转换后的样本列表，已根据 GPU 数量进行了裁剪。
        """
        # 第一步：转换
        sample_arr_final = []
        for cmt in cmt_array:
            extras = self.get_extra(cmt)
            # 调用轨迹对象的 group_tokenize 方法进行分词
            sample_arr = cmt.group_tokenize()  # ⭐ 对轨迹进行分词生成样本
            for sample in sample_arr:
                sample.extras = extras  # ⭐ 添加额外信息到样本
            sample_arr_final += sample_arr

        # 第二步：计算需要移除多少样本，以确保总数能被 world_size 整除
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        remainder = len(sample_arr_final) % world_size
        if remainder != 0:
            import random
            # 随机选择要移除的索引
            remove_indices = random.sample(range(len(sample_arr_final)), remainder)
            # 倒序排列索引，避免 pop 时索引偏移
            remove_indices.sort(reverse=True)
            for idx in remove_indices:
                sample_arr_final.pop(idx)  # ⭐ 移除样本以适配 GPU 数量

        # 返回适配后的样本列表
        return sample_arr_final

    def samples_to_dataproto(self, samples: list[Sample]) -> DataProto:
        """
        将 Sample 对象列表转换为 DataProto 对象，进行批处理和填充。
        这里构建了模型训练所需的所有 Tensor 输入（如 input_ids, attention_mask 等）。

        Args:
            samples (list[Sample]): Sample 对象列表。

        Returns:
            DataProto: 包含批处理和填充后数据的 DataProto 对象。
        """
        # 初始化列表以存储批处理数据
        step_ids_list  = []
        steps_texts_list = []
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        prompt_exp_mask_list, response_exp_mask_list = [], []  # 用于经验回放的掩码列表
        messages = []
        reward_scores = []
        task_ids = []
        rollout_ids = []
        extras = [] # 存储轨迹的补充数据
        k_text_list = []
        for sample in samples:
            # 验证所有字段长度一致
            assert len(sample.input_ids) == len(sample.attention_mask) == len(sample.position_ids) == len(
                sample.loss_mask), f"Sample {sample.request_id} has mismatched lengths: " \
                                f"{len(sample.input_ids)=}, {len(sample.attention_mask)=}, " \
                                f"{len(sample.position_ids)=}, {len(sample.loss_mask)=}"

            task_ids.append(sample.task_id)
            rollout_ids.append(sample.rollout_id)
            # 丢弃超过 Prompt 长度限制的样本
            if len(sample.prompt_ids) > self.config.data.max_prompt_length:
                raise RuntimeError(f"Sample has prompt_ids length {len(sample.prompt_ids)} ")

            # 如果 Response 长度超过限制则警告（但仍包含在内）
            if len(sample.response_ids) > self.config.data.max_response_length:
                logger.warning(
                    f"Sample {sample.request_id} has response_ids length {len(sample.response_ids)} "
                    f"greater than max_response_length {self.config.data.max_response_length}."
                )
                raise RuntimeError(f"Sample has prompt_ids length {len(sample.prompt_ids)} ")

            # ------------- 处理步骤 ID 和文本 (用于语义评估) ------------
            resp_ids = sample.response_ids
            # 解析 Response ID 为具体的步骤
            parse_result = parse_response_ids_to_steps(resp_ids, self.tokenizer) # ⭐ 将 response IDs 解析为步骤

            step_ids_list.append(torch.tensor(parse_result.step_ids, dtype=torch.long))
            # 生成步骤文本
            steps_texts_list.append([
                {"action": s["action_text"], "observation": s["observation_text"]}
                for s in parse_result.steps
            ])

            # 将各个字段转换为 Tensor 并添加到列表
            assert len(sample.prompt_ids) != 0
            assert len(sample.response_ids) != 0
            prompt_ids.append(torch.tensor(sample.prompt_ids, dtype=torch.int))
            response_ids.append(torch.tensor(sample.response_ids, dtype=torch.int))

            prompt_attention_mask.append(torch.tensor(sample.prompt_attention_mask, dtype=torch.int))
            response_attention_mask.append(torch.tensor(sample.response_attention_mask, dtype=torch.int))

            prompt_position_ids.append(torch.tensor(sample.prompt_position_ids, dtype=torch.int))
            response_position_ids.append(torch.tensor(sample.response_position_ids, dtype=torch.int))

            prompt_loss_mask.append(torch.tensor(sample.prompt_loss_mask, dtype=torch.int))
            response_loss_mask.append(torch.tensor(sample.response_loss_mask, dtype=torch.int))

            messages.append({"messages": sample.messages})
            reward_scores.append(sample.reward_scores)
            extras.append(sample.extras)

            # 创建经验掩码 (exp_mask)：用于控制 off-policy 数据在训练中的行为
            # 如果满足 off_clip_high 条件（即 add_exp=True 且 mode="discard"），则设为 1
            if sample.extras.get("add_exp", False) and sample.extras.get("task_train_expmode", None)=="discard":
                prompt_exp_mask_list.append(torch.ones(len(sample.prompt_loss_mask), dtype=torch.int))
                response_exp_mask_list.append(torch.ones(len(sample.response_loss_mask), dtype=torch.int))
            else:
                prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
                response_exp_mask_list.append(torch.zeros(len(sample.response_loss_mask), dtype=torch.int))



        # 获取当前批次的最大 Prompt 和 Response 长度，用于 Padding
        max_prompt_length_this_batch = max([p.shape[-1] for p in prompt_ids])
        assert max_prompt_length_this_batch <= self.config.data.max_prompt_length
        max_response_length_this_batch = max([p.shape[-1] for p in response_ids])
        assert max_response_length_this_batch <= self.config.data.max_response_length

        # 对序列进行填充 (Padding)
        # ------------- 填充 step_ids ------------
        step_ids_pad = pad_sequence(
            step_ids_list, batch_first=True, padding_value=-1
        )
        step_ids_pad = pad_sequence_to_length(
            step_ids_pad, self.config.data.max_response_length, -1
        )  # ⭐ 将 step IDs 填充到最大响应长度

        # 对 Prompt 相关 Tensor 进行左填充 (Left Padding)
        prompt_ids =            pad_sequence(prompt_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left")
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        prompt_position_ids =   pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
        prompt_loss_mask =      pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        prompt_exp_mask_list =  pad_sequence(prompt_exp_mask_list, batch_first=True, padding_value=0, padding_side="left")

        # 填充到统一长度
        prompt_ids =            pad_sequence_to_length(prompt_ids, max_prompt_length_this_batch, self.pad_token_id, left_pad=True)
        prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_position_ids =   pad_sequence_to_length(prompt_position_ids, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_loss_mask =      pad_sequence_to_length(prompt_loss_mask, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_exp_mask_list =  pad_sequence_to_length(prompt_exp_mask_list, max_prompt_length_this_batch, 0, left_pad=True)

        # 对 Response 相关 Tensor 进行右填充 (Right Padding)
        response_ids =            pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        response_loss_mask =      pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        # response_exp_mask_list =  pad_sequence(response_exp_mask_list, batch_first=True, padding_value=0, padding_side="left")
        response_exp_mask_list  = pad_sequence(response_exp_mask_list,  batch_first=True, padding_value=0)        # shuchang debug: Remove padding_side="left"

        # 填充到统一长度
        response_ids =            pad_sequence_to_length(response_ids, max_response_length_this_batch, self.pad_token_id)  # ⭐ 填充 response IDs
        response_attention_mask = pad_sequence_to_length(response_attention_mask, max_response_length_this_batch, 0)  # ⭐ 填充 response attention mask
        response_loss_mask =      pad_sequence_to_length(response_loss_mask, max_response_length_this_batch, 0)  # ⭐ 填充 response loss mask
        # response_exp_mask_list =  pad_sequence_to_length(response_exp_mask_list, max_prompt_length_this_batch, 0, left_pad=True)  # ⭐ Pad response experience mask list to the maximum prompt length
        response_exp_mask_list =  pad_sequence_to_length(response_exp_mask_list, max_response_length_this_batch, 0) # shuchang debug: 应填充到 max_response_length_this_batch

        # 计算 Response 的 Position IDs (接续 Prompt 的 Position IDs)
        delta_position_id = torch.arange(1, response_ids.size(1) + 1, device=response_ids.device).unsqueeze(0).repeat(len(samples), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id  # ⭐ 计算 response 的 position IDs

        # 拼接 Prompt 和 Response 的 Tensors，形成完整的 Input
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)  # ⭐ 拼接 prompt 和 response IDs
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)  # ⭐ 拼接 attention masks
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)  # ⭐ 拼接 position IDs
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)  # ⭐ 拼接 loss masks
        
        # 构建 group_ids，用于 GRPO 等算法中的组内优势计算
        group_ids = torch.tensor([int(s.data_id) for s in samples], dtype=torch.long)  # ⭐ 构建 group IDs
        
        # 拼接 exp_mask
        exp_mask = torch.cat((prompt_exp_mask_list, response_exp_mask_list), dim=-1)  # ⭐ 拼接 exp masks

        # 验证形状一致性
        assert exp_mask.shape == loss_mask.shape, f"Shape mismatch: {exp_mask.shape} vs {loss_mask.shape}"

        # 使用 TensorDict 构建 Batch 数据
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "exp_mask": exp_mask,        # 添加 exp_mask (用于经验回放控制)
                "step_ids": step_ids_pad,
                "group_ids": group_ids,   # ★ 添加 group_ids (用于分组处理)
            },
            batch_size=len(samples),
        )

        # 返回 DataProto 对象，包含 tensor batch 和 non-tensor batch (元数据)
        return DataProto(
            batch=batch,
            non_tensor_batch={
                "task_ids": np.array(task_ids),
                "rollout_ids": np.array(rollout_ids),
                "messages": np.array(messages),
                "reward_scores": np.array(reward_scores),
                "extras": np.array(extras),
                "steps": np.array(steps_texts_list, dtype=object)
            }
        )