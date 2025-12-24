# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications copyright 2025 Alibaba Tongyi EconML Lab. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
基于 Ray 单控制器的 FSDP PPO 训练器。
该训练器支持与 HuggingFace 模型无关的模型初始化。
"""

import os
import uuid
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from pprint import pprint
from typing import List, Optional, Any
import warnings

from loguru import logger
import numpy as np
import ray
import torch
import random
import json
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from torch.utils.data import SequentialSampler,IterableDataset,Dataset,Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from agentevolver.client.env_client import EnvClient
from agentevolver.module.task_manager.task_manager import AutoReloadDataset, FullDataset
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, create_colocated_worker_cls
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from agentevolver.utils.metric_utils import (compute_data_metrics,
                                             compute_throughout_metrics,
                                             compute_timing_metrics,
                                             process_validation_metrics)
from verl.trainer.ppo.ray_trainer import (AdvantageEstimator, RayPPOTrainer, ResourcePoolManager, WorkerType,
                                          _timer, apply_kl_penalty,
                                          compute_response_mask, Role)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.metric import reduce_metrics

from agentevolver.client.llm_client import DashScopeClient
from agentevolver.client.em_client import EMClient
from agentevolver.module.env_manager.env_manager import ParallelEnvManager
from agentevolver.module.task_manager import adapter as task_adapter
from agentevolver.module.task_manager import TaskManager,NaiveTaskObjectiveRetrieval
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

from agentevolver.utils.tracking import ValidationGenerationsLogger

from agentevolver.module.adv_processor.adca_grpo_pipeline import apply_adca_grpo

from agentevolver.module.exp_manager.exp_manager import ExperienceManager


def parse_reward_from_dataproto(data: DataProto, return_dict=False) -> dict | torch.Tensor:
    """
    从数据批次中计算/提取奖励。

    Args:
        data: DataProto 对象，包含输入数据。
        return_dict: 是否返回字典形式的详细信息，还是只返回奖励张量。

    Returns:
        如果 return_dict 为 False，返回形状为 (bs, response_len) 的张量；
        否则返回包含 'reward_tensor' 和 'reward_extra_info' 的字典。
    """
    # 在 DataFlow 中，world.execute() 会传递一个浮点数分数，该分数包含在 DataProto.non_tensor_batch('reward_scores') 中

    # 初始化奖励张量
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # (bs, reslen)  # ⭐ 初始化奖励张量，默认全0
    reward_extra_info = defaultdict(list)

    # 批次级处理
    prompt_ids_batch = data.batch["prompts"]  # (bs, prompt_len)
    prompt_lengths = prompt_ids_batch.shape[-1]

    # 获取所有项的注意力掩码
    attention_masks = data.batch["attention_mask"]  # (bs, total_len)
    # 计算响应长度
    response_lengths = attention_masks[:, prompt_lengths:].sum(dim=1)  # (bs, )

    # 获取奖励分数 (Outcome Reward)
    reward_scores_list = [item["outcome"] for item in data.non_tensor_batch["reward_scores"]]
    reward_scores = torch.tensor(reward_scores_list, device=reward_tensor.device, dtype=torch.float32)  # (bs, )  # ⭐ 将奖励列表转换为张量

    # 使用高级索引将奖励分配给响应的最后一个 Token 位置
    # 这是一个稀疏奖励设置，只在序列结束时给分
    reward_tensor[torch.arange(len(data)), response_lengths - 1] = reward_scores

    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }
    else:
        return reward_tensor


def create_rl_sampler(data_config, dataset):
    """
    为数据集创建采样器。

    Arguments:
        data_config: 数据配置对象，包含是否打乱 (shuffle) 和种子 (seed) 等设置。
        dataset (Dataset): 数据集对象。

    Returns:
        sampler (Sampler): 创建的采样器。
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # 使用采样器以便更好地进行断点续训 (Checkpoint Resume)
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler

def union_gen_batch_via_task_id(tasks, batch: DataProto, gen_batch_output: DataProto):
    """
    基于 `task_id` 将生成批次输出 (`gen_batch_output`) 与原始批次 (`batch`) 合并。
    这通常用于将 Rollout 产生的新轨迹与原始的任务描述（Prompt）对齐。

    Args:
        tasks (list): 任务对象列表，每个对象包含一个 `task_id`。
        batch (DataProto): 原始数据批次。
        gen_batch_output (DataProto): 需要合并的生成批次输出。

    Returns:
        DataProto: 最终合并后的批次。
    """
    map_task_id_to_index = {t.task_id:i for i, t in enumerate(tasks)}  # ⭐ 创建 task_id 到任务索引的映射
    gen_task_task_ids = gen_batch_output.non_tensor_batch['task_ids']
    # 找到生成数据对应的原始任务索引
    indices = [map_task_id_to_index[tid] for tid in gen_task_task_ids]
    # 从原始批次中选出对应的数据
    batch_extend = batch.select_idxs(indices)
    # 将选出的原始数据与生成的数据合并
    batch_final = batch_extend.union(gen_batch_output)  # ⭐ 合并数据
    return batch_final


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    计算 GRPO (Group Relative Policy Optimization) 的优势函数 (Advantage)。
    仅针对 Outcome Reward（结果奖励）进行操作，即每个响应只有一个标量奖励。

    Args:
        token_level_rewards: `(torch.Tensor)`
            形状为 (bs, response_length)，包含 Token 级别的奖励。
        response_mask: `(torch.Tensor)`
            形状为 (bs, response_length)，响应掩码。
        index: `(np.ndarray)`
            组索引 (Group Index)，通常对应 prompt_id 或 uid，用于标识哪些样本属于同一个组。
        epsilon: (float)
            防止除零的小数值。
        norm_adv_by_std_in_grpo: (bool)
            是否对 GRPO 优势进行缩放。
            如果为 True，优势会除以组内标准差 (std)，如原始 GRPO 论文所述。
            如果为 False，不进行缩放，类似于 Dr.GRPO。

    Returns:
        advantages: `(torch.Tensor)` 形状 (bs, response_length)
        returns: `(torch.Tensor)` 形状 (bs, response_length)
    """
    # 将 Token 级别的奖励求和得到总分数 (因为是 Outcome Reward)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    if scores.dim()!=1:
        logger.warning("scores.dim()!=1")

    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 将分数按组 ID 收集
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # 计算每组的均值和标准差
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0) # 防止除以0
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        # 计算归一化后的优势 A = (r - mean) / std
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
                # 即使不除以 std，也可能需要某种形式的权重调整
                # scores[i] = scores[i] / (batch_std + epsilon)
        
        # 将标量优势扩展回 Token 级别 (广播) 并应用掩码
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores



def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    """
    计算策略优化的优势估计 (Advantage Estimates)。

    此函数支持多种优势估计器，如 GAE, GRPO, REINFORCE++ 等。

    Args:
        data (DataProto): 包含批处理模型输出和输入的数据。
        adv_estimator: 使用的优势估计器类型 (AdvantageEstimator 枚举)。
        gamma (float, optional): 折扣因子。默认为 1.0。
        lam (float, optional): GAE 的 Lambda 参数。默认为 1.0。
        num_repeat (int, optional): 重复计算次数。默认为 1。
        multi_turn (bool, optional): 数据是否来自多轮对话。默认为 False。
        norm_adv_by_std_in_grpo (bool, optional): 是否在 GRPO 中按标准差归一化。默认为 True。
        config (dict, optional): 算法设置的配置字典。默认为 None。

    Returns:
        DataProto: 更新后的数据，包含计算出的 'advantages' 和 'returns'。
    """
    # 向后兼容：如果 fit 中未计算 response_mask，则在此处计算
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    
    # 准备响应组
    if adv_estimator == AdvantageEstimator.GAE:
        # 使用广义优势估计 (GAE) 计算优势和回报
        # GAE 需要 Critic 网络 (Values)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # GRPO 不需要 Critic 网络，而是基于组内平均基线
        
        # 初始化 GRPO 计算掩码
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # 如果是多轮对话，使用 loss_mask 的相关部分
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        
        # 调用专门的 GRPO 优势计算函数
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"], # 使用 uid 作为分组依据
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )  # ⭐ 计算 GRPO 优势和回报
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # 处理除 GAE 和 GRPO 之外的其他优势估计器类型
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # 可选
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # 可选 (例如 ReMax 需要)
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # 计算优势估计
        advantages, returns = adv_estimator_fn(**adv_kwargs)  # ⭐ 计算其他估计器的优势和回报
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class AgentEvolverRayPPOTrainer(RayPPOTrainer):
    """
    AgentEvolver 的 Ray PPO 训练器。
    注意：此训练器在单个 CPU/GPU 节点的 Driver 进程上运行，通过 Ray 调度远程 Worker。
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        train_task_manager:TaskManager,
        val_task_manager:TaskManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, # type: ignore
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        collate_fn=None,
        shuffle_trainset:bool=False,
        device_name="cuda",
    ):
        """
        初始化基于 Ray 后端的分布式 PPO 训练器。

        Args:
            config: 包含各种设置的配置对象。
            tokenizer: 用于处理文本的分词器。
            role_worker_mapping (dict[Role, WorkerType]): 角色到 Worker 类型的映射 (例如 Actor -> FSDPWorker)。
            resource_pool_manager (ResourcePoolManager): 资源池管理器，管理 GPU 分配。
            train_task_manager (TaskManager): 训练任务管理器。
            val_task_manager (TaskManager): 验证任务管理器。
            ray_worker_group_cls (RayWorkerGroup, optional): Ray Worker 组类。默认为 RayWorkerGroup。
            processor (optional): 用于额外数据处理的处理器 (如多模态)。
            reward_fn (optional): 计算奖励的函数。
            val_reward_fn (optional): 计算验证奖励的函数。
            collate_fn (optional): 数据整理函数。
            shuffle_trainset (bool, optional): 是否打乱训练集。默认为 False。
            device_name (str, optional): 设备名称。默认为 "cuda"。
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"  # ⭐ 确保支持混合引擎 (vLLM + FSDP)

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"  # ⭐ 确保 ActorRollout 角色存在

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        
        # 检查是否需要 Reference Policy 和 Reward Model
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # 如果 ref_in_actor 为 True，则参考策略即为没有应用 LoRA 的 Actor
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # 定义奖励内的 KL 控制器
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        # 决定是否使用 Critic 网络
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        # 验证配置
        self._validate_config()

        self.env_manager: ParallelEnvManager | None = None
        self.thread_pool: ThreadPoolExecutor | None = None

        self.train_task_manager=train_task_manager
        self.val_task_manager=val_task_manager
        self._collate_fn=collate_fn

        # 创建数据加载器
        self._create_dataloader_from_manager(collate_fn, shuffle_trainset)  # ⭐ 从管理器创建数据加载器


    def init_workers(self):
        """
        使用 Ray 后端初始化分布式训练 Worker。

        此函数创建：
        1. Ray 资源池 (基于配置)。
        2. 每个角色的 Worker 组 (Actor, Critic 等)。

        Args:
            None

        Returns:
            None
        """
        self.resource_pool_manager.create_resource_pool()  # ⭐ 初始化资源池

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 创建 Actor 和 Rollout Worker
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # 创建 Critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # 创建 Reference Policy (如果需要)
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # 创建 Reward Model (如果 reward_fn 为 None)
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # 初始化 WorkerGroup
        all_wg = {}
        wg_kwargs = {}  # 设置 RayWorkerGroup 的参数
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        # 实例化所有 Worker 组
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls,
                                                device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # 初始化各模型的权重
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()  # ⭐ 初始化 Critic 模型

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()  # ⭐ 初始化 Ref Policy 模型

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()  # ⭐ 初始化 Reward Model

        # 最后初始化 Actor/Rollout，以便 vLLM 可以更好地估计 KV 缓存内存
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()  # ⭐ 初始化 Actor/Rollout 模型

        # 创建异步 Rollout 管理器和请求调度器
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from agentevolver.module.trainer.ae_async_llm_server_manager import BaAsyncLLMServerManager
            self.async_rollout_mode = True
            self.async_rollout_manager = BaAsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg)  # ⭐ 创建异步 Rollout 管理器

        self.reward_fn = parse_reward_from_dataproto
        self.val_reward_fn = parse_reward_from_dataproto

        # 初始化并行环境管理器 (用于与环境交互)
        self.env_manager = ParallelEnvManager(config=self.config, async_rollout_manager=self.async_rollout_manager, max_parallel=self.config.actor_rollout_ref.rollout.max_env_worker)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool.max_workers)
        self.exp_manager = ExperienceManager(config=self.config)


    def _create_dataloader_from_manager(self, collate_fn, shuffle_trainset: bool = True):
        """
        创建训练和验证数据加载器 (DataLoaders)。

        1. 检查训练和验证文件是否存在，加载本地任务。如果未给出文件，则从环境服务加载任务 (train 和 val/dev 划分)。
        2. 使用 TaskManager 为训练集生成合成任务 (Synthetic Tasks)，并加载原始验证数据集。
        3. 使用 TaskManager 混合不同来源的任务。
        4. 适配数据集并创建训练器中使用的数据加载器。

        Args:
            collate_fn (callable): 用于将数据整理成批次的函数。
            shuffle_trainset (bool, optional): 是否打乱训练集。默认为 True。

        Returns:
            None
        """
        # TODO: 我们必须确保批次大小能被 dp_size 整除
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn


        from verl.trainer.main_ppo import create_rl_dataset
        # 加载训练数据集 (从文件或环境)
        env_client=EnvClient(self.config.env_service.env_url)
        if self.config.data.train_files is not None:
            train_seed_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
            assert isinstance(train_seed_dataset,RLHFDataset), "train_dataset must be RLHFDataset"
            self.train_task_manager.load_tasks_from_dataset(train_seed_dataset,env_type=self.config.env_service.env_type)
        else:
            self.train_task_manager.load_tasks_from_environment(env_client,env_type=self.config.env_service.env_type,split="train")
        
        # 加载验证数据集
        if self.config.data.val_files is not None:
            val_seed_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
            assert isinstance(val_seed_dataset,RLHFDataset), "train_dataset must be RLHFDataset"
            self.val_task_manager.load_tasks_from_dataset(val_seed_dataset,env_type=self.config.env_service.env_type)
        else:
            num_loaded_val_tasks = 0
            if 'val_on_test' in os.environ.get("DEBUG_ARG",'') or (self.config.data.val_type == 'test_normal' and self.config.env_service.env_type == "appworld"):
                logger.warning("using test_normal as val dataset")
                num_loaded_val_tasks += self.val_task_manager.load_tasks_from_environment(env_client,env_type=self.config.env_service.env_type,split="test_normal")
            else:
                for split in ['val','dev']:
                    try:
                        num_loaded_val_tasks += self.val_task_manager.load_tasks_from_environment(env_client,env_type=self.config.env_service.env_type,split=split)
                    except:
                        logger.warning(f"failed to load val dataset from environment, split={split}. this may be *normal* if your dataset is split into train/dev")    
            
            assert num_loaded_val_tasks > 0, "failed to load val/dev dataset from environment"
        
        # 创建完整数据集 (FullDataset)，包含任务生成和混合逻辑
        self.train_dataset = FullDataset(
            self.train_task_manager,
            self.train_task_manager._mixture_strategy,
            self.train_task_manager._reward_config,
            self.config.task_manager.train_data_path,
            tokenizer=self.tokenizer,
            config=self.config.data,
            processor=self.processor,
        )
        self.val_dataset = FullDataset(
            self.val_task_manager,
            self.val_task_manager._mixture_strategy,
            self.val_task_manager._reward_config,
            cache_path=None,
            tokenizer=self.tokenizer,
            config=self.config.data,
            processor=self.processor,
        )

        assert not isinstance(self.train_dataset,AutoReloadDataset), "please disable multiple workers for AutoReloadDataset"
        assert not isinstance(self.val_dataset,AutoReloadDataset), "please disable multiple workers for AutoReloadDataset"
        
        # 创建训练 DataLoader
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data,self.train_dataset),
        )  # ⭐ 创建训练数据加载器

        val_batch_size = self.config.data.val_batch_size  # 优先使用 config 值
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset) # type: ignore

        # 创建验证 DataLoader
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )  # ⭐ 创建验证数据加载器

        # 训练 DataLoader 是动态的，所以不检查大小
        # assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        if not isinstance(self.train_dataset,IterableDataset):
            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
            print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")
        else:
            # FIXME: 需要一种优雅的方式来设置 total_training_steps
            total_training_steps = len(self.train_task_manager.seed_tasks)*self.config.trainer.total_epochs
            print(f"Size of train dataloader: unknown, Size of val dataloader: {len(self.val_dataloader)}")

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")


    def _get_attribution_config(self):
        """
        获取并验证 Attribution Driven Credit Assignment (ADCA) 的配置。
        包括设置 API 重试尝试次数。

        Returns:
            dict: 验证和更新后的配置字典。

        Raises:
            ValueError: 如果配置中缺少 'attribution_driven_credit_assignment' 块。
        """
        if not hasattr(self.config, 'attribution_driven_credit_assignment'):
            raise ValueError("attribution_driven_credit_assignment configuration block is required")

        config = self.config.attribution_driven_credit_assignment

        # 设置默认的 api_max_retries
        if not hasattr(config, 'api_max_retries'):
            config.api_max_retries = 200  # ⭐ 设置 API 默认最大重试次数为 200
            print(f"[attribution_config] Using default api_max_retries: {config.api_max_retries}")

        return config


    def _validate_config(self):
        """
        验证配置设置，确保它们一致并满足训练过程的要求。

        此函数检查：
        - GPU 总数及其分配。
        - 总批次大小及其对最小可能批次大小的可整除性。
        - 某些微批次 (micro-batch) 大小参数的互斥性。
        - Actor、Critic 和 Reward Model 配置的一致性。
        - 其他关键设置，如损失聚合模式和序列并行性。

        Raises:
            AssertionError: 如果任何配置设置不满足要求。
            ValueError: 如果互斥参数同时设置或均未设置。
        """
        config = self.config
        # GPU 总数
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. 检查数据正确性的总批次大小
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # 辅助函数：检查 "micro_batch_size" 与 "micro_batch_size_per_gpu" 是否互斥
        # 如果用户同时设置了两者，抛出错误。新约定是使用 "..._micro_batch_size_per_gpu"。
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            # Rollout 部分也有 log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # 检查 Critic 微批次大小冲突
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # 检查 Reward Model 微批次大小冲突
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor 检查
        # 检查 train_batch_size 是否大于 ppo_mini_batch_size
        # 如果不是 dynamic_bsz，必须确保：
        #    ppo_mini_batch_size 能被 ppo_micro_batch_size 整除
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size  # ⭐ 确保 train_batch_size 至少与 ppo_mini_batch_size 一样大
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0  # ⭐ 确保 ppo_mini_batch_size 能被 ppo_micro_batch_size 整除
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus  # ⭐ 确保 GPU 分配满足微批次大小和序列并行

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic 检查
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size  # ⭐ 确保 Critic 的 train_batch_size 足够大
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0  # ⭐ 确保 Critic 批次整除
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus  # ⭐ 确保 Critic GPU 分配正确

        # 检查在使用 FSDP 序列并行时是否启用了 use_remove_padding
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # 检查评估配置
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # 检查多轮对话与工具配置
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            # 0623 yunpeng comment: no need this tool_config_path
            # assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None, "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    ##################
    # ANNI
    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """
        将 Rollout 或验证生成的样本转储为 JSONL 格式。

        Args:
            inputs (list): 输入数据列表。
            outputs (list): 输出数据列表。
            scores (list): 分数列表。
            reward_extra_infos_dict (dict): 包含额外奖励信息的字典。
            dump_path (str): 保存 JSONL 文件的目录路径。

        Returns:
            None
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")  # ⭐ 创建 JSONL 文件名

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")  # ⭐ 将数据写入 JSONL 文件

        print(f"Dumped generations to {filename}")


    def _validate(self):
        """
        验证模型：生成序列，收集样本，并存储结果。

        此函数处理每一批验证数据，生成输出，并收集输入、输出和经验信息以供进一步分析。

        Args:
            None

        Returns:
            None
        """
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # 用于收集样本以供表格显示的列表
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for i, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            # 重复测试批次
            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # 我们只在基于规则的 RM 上进行验证
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "extras" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("extras")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # 填充以被 dp_size 整除
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                raise NotImplementedError

            else:
                self.async_rollout_manager.wake_up()
                # 从测试批次构建任务对象
                tasks = [Task(
                            task_id=test_gen_batch.non_tensor_batch["extras"][i]["task_id"],
                            query=test_gen_batch.non_tensor_batch["extras"][i]['new_query'],
                            env_type=self.config.env_service.env_type,
                            open_query=test_gen_batch.non_tensor_batch["extras"][i]['open_query'],
                            # evaluator=gen_batch.non_tensor_batch['extras'][i]['evaluator'], # avoid potential bugs
                          ) for i in range(len(test_gen_batch))]
                task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="validate")
                print("=" * 10 + "start validate rollout" + "=" * 10)
                # 执行验证 Rollout
                trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="validate", epoch=f"test.1.{i}")  # ⭐ 执行 Rollout 生成轨迹
                print("=" * 10 + "end validate rollout" + "=" * 10)
                test_output_gen_batch = self.env_manager.to_dataproto(trajectories)
                # test_output_gen_batch_padded = self.explorer_manager.rollout(test_gen_batch_padded)
                # test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # 去除填充
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # 存储原始输入
            input_ids = test_output_gen_batch.batch["prompts"]
            # TODO: 我们是否可以保留除 padding token 之外的特殊 token？
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # 存储生成的输出
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # 重复测试批次
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            test_batch = union_gen_batch_via_task_id(tasks, test_batch, test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            # test_batch = test_batch.union(test_output_gen_batch)

            # 使用奖励函数进行评估
            result = self.val_reward_fn(test_batch, return_dict=True)  # ⭐ 使用奖励函数评估测试批次
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # 转储生成结果
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        # val_data_dir = "experiments/validation_log"
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        # 处理并汇总验证指标
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)  # ⭐ 处理验证指标
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict
    
    def initialize_exp_pool(self):
        """
        初始化经验池。类似于验证过程，但目的是更新经验管理器。
        """
        for i, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            # 我们只在基于规则的 RM 上进行验证
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "extras" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("extras")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # 填充以被 dp_size 整除
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                raise NotImplementedError

            else:
                self.async_rollout_manager.wake_up()
                tasks = [Task(
                            task_id=test_gen_batch.non_tensor_batch["extras"][i]["task_id"],
                            query=test_gen_batch.non_tensor_batch["extras"][i]['new_query'],
                            env_type=self.config.env_service.env_type,
                            open_query=test_gen_batch.non_tensor_batch["extras"][i]['open_query'],
                            # evaluator=gen_batch.non_tensor_batch['extras'][i]['evaluator'], # avoid potential bugs
                          ) for i in range(len(test_gen_batch))]
                task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="validate")
                print("=" * 10 + "start validate rollout" + "=" * 10)
                trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="validate", epoch=f"test.1.{i}")  # ⭐ 执行 Rollout 生成轨迹
                print("=" * 10 + "end validate rollout" + "=" * 10)
                self.async_rollout_manager.sleep()

            # 批量总结：更新经验池
            self.exp_manager.summarize_in_batch(trajectories)
        
        return


    def fit(self):
        """
        PPO 训练的主循环。
        Driver 进程只需通过 RPC 调用 Worker 组的计算函数来构建 PPO 数据流。
        轻量级的优势 (Advantage) 计算在 Driver 进程上完成。
        """
        from omegaconf import OmegaConf

        from agentevolver.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # 在做任何事情之前加载检查点
        self._load_checkpoint()
        # 将参数传播到 vLLM
        self.async_rollout_manager.wake_up()
        self.async_rollout_manager.sleep()

        # 初始化经验池
        if self.config.exp_manager.get("init_exp_before_training", False):
            self.initialize_exp_pool()
            if self.config.exp_manager.get("init_exp_only", False):
                return

        # 在训练前执行验证
        # 目前，我们只支持使用 reward_function 进行验证
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()  # ⭐ 执行初始验证并获取指标
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # [0616] qingxu: 添加 `RAY_DEBUG_POST_MORTEM` 环境变量以激活断点调试
        # vscode_conditional_breakpoint()
        # breakpoint()

        # 添加进度条
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # 我们从第 1 步开始
        self.global_steps += 1
        last_val_metrics = None
        
        # 训练 Epoch 循环
        for epoch in range(self.config.trainer.total_epochs):
            for i, batch_dict in enumerate(self.train_dataloader):
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # 弹出那些用于生成的键
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "extras" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("extras")
                    batch_extras = deepcopy(batch.non_tensor_batch["extras"])
                else:
                    batch_extras = None
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # 生成一个批次 (Rollout)
                    with _timer("gen", timing_raw):
                        trajectories: List[Trajectory] = []
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            # gen_batch_output = self.explorer_manager.rollout(gen_batch)

                            # 构造 Task 列表
                            tasks = [Task(
                                        task_id=gen_batch.non_tensor_batch["extras"][i]["task_id"],
                                        query=gen_batch.non_tensor_batch["extras"][i]['new_query'],
                                        env_type=self.config.env_service.env_type,
                                        open_query=gen_batch.non_tensor_batch["extras"][i]['open_query'],
                                        evaluator=gen_batch.non_tensor_batch['extras'][i]['evaluator'],
                                        ground_truth=gen_batch.non_tensor_batch['extras'][i]['ground_truth']
                                      ) for i in range(len(gen_batch))
                                    ]
                            # 获取经验配置
                            task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="sample")
                            assert len(task_exp_configs)==len(tasks), "{len(task_exp_configs)=}, {len(gen_batch)=}"

                            # TODO enable tracing by jinli 0619
                            print("=" * 10 + "start fit rollout" + "=" * 10)
                            trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="sample", epoch=f"train.{epoch}.{i}")  # ⭐ 使用环境管理器生成轨迹
                            assert len(trajectories)>0, "{len(trajectories)=}?"
                            print("=" * 10 + "end fit rollout" + "=" * 10)
                            # 将轨迹转换为训练数据格式
                            gen_batch_output = self.env_manager.to_dataproto(trajectories)
                            
                            # 更新关于经验管理器的指标
                            exp_mask_ratio = gen_batch_output.batch["exp_mask"].float().mean()
                            metrics.update({"exp_mask_ratio": exp_mask_ratio.detach().item()})
                            context_time_cost = [x.metadata["context_time_cost"] for x in trajectories if "context_time_cost" in x.metadata]
                            if context_time_cost:
                                metrics.update({
                                  "exp_manager/context_cost_avg":   np.mean(context_time_cost),
                                  "exp_manager/context_cost_max":   np.max(context_time_cost),
                                  "exp_manager/context_cost_min":   np.min(context_time_cost),
                                })

                            print(f"gen_batch_output.info batch.keys={gen_batch_output.batch.keys()}")
                            num_term_traj = sum([traj.is_terminated  for traj in trajectories])
                            num_not_none_traj = sum([len(traj.steps)>0  for traj in trajectories])

                            # gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    # 如果使用 RE-Max，需要生成 Baseline
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)  # ⭐ 生成用于优势估计的基线序列

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor  # ⭐ 将奖励基线添加到批次中

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)  # ⭐ 为批次中的每个项目生成唯一的 UID

                    # 在新代码中，rollout 过程生成新的 extras，应该与原始 extra 合并。
                    # 目前，它们是分开存储的。
                    # assert len(gen_batch_output.non_tensor_batch["extras"].keys()&batch_extras.keys())==0, "extra of extra should not overlap with existing extra...how funny..."
                    batch.non_tensor_batch['original_extras']=batch_extras  # ⭐ 存储原始 extras
                    batch = union_gen_batch_via_task_id(tasks, batch, gen_batch_output)  # ⭐ 将生成的批次与当前批次合并

                    batch.batch["response_mask"] = compute_response_mask(batch)  # ⭐ 计算并添加响应掩码

                    # 更新经验池
                    summary_task = self.exp_manager.submit_summary_task(trajectories, self.global_steps)


                    # 平衡每个 DP Rank 上的有效 Token 数量。
                    # 注意这会打乱批次内数据的顺序。
                    # 实现基于组的优势计算 (如 GRPO 和 RLOO) 时请注意。
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)  # ⭐ 平衡批次以均匀分布有效 Token

                    # 计算全局有效 Token
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()  # ⭐ 计算并存储全局 Token 数量

                    with _timer("reward", timing_raw):
                        # 计算奖励模型分数
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)  # ⭐ 使用奖励模型计算分数
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)  # ⭐ 计算奖励和额外信息

                    # 重新计算 old_log_probs (因为在 Rollout 后模型可能已经更新，或者需要更精确的 logprobs)
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)  # ⭐ 计算旧的对数概率
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: 我们可能也想添加概率差异的指标。
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # 计算参考策略的 log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)  # ⭐ 计算参考对数概率
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # 计算 Values (如果使用 Critic)
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)  # ⭐ 使用 Critic 计算状态价值
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # 我们结合基于规则的 RM
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)  # ⭐ 从异步调用获取奖励
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # 计算最终奖励。如果可用，应用 KL 惩罚。
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)  # ⭐ 应用 KL 散度惩罚
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # 计算优势 (Advantage)，在 Driver 进程上执行
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO 优势归一化因子
                        if os.environ.get("DEBUG_ARG","").find("disable_adv_std")!=-1:
                            if epoch==0 and i==0:
                                print("DEBUG: change norm_adv_by_std_in_grpo from True to False, using batch std!")
                            norm_adv_by_std_in_grpo = False

                        # 调用原始的 compute_advantage 以兼容
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )
                        # shuchang
                        # ==================== 开始 ADCA GRPO  ====================
                        # 应用基于归因的信用分配 (Attribution-Driven Credit Assignment)
                        attribution_cfg = self._get_attribution_config()
                        if getattr(attribution_cfg, 'enable', False):
                            batch, adca_metrics = apply_adca_grpo(
                                batch=batch,
                                attribution_cfg=attribution_cfg,
                                tokenizer=self.tokenizer,
                                global_steps=self.global_steps,
                                epoch=epoch,
                                i=i,
                            )
                            metrics.update(adca_metrics)
                        # ==================== 结束 ADCA GRPO ====================
                        
                        # 调试选项：对非环境评估器（即合成数据）生成的优势应用 0.5 的衰减因子
                        if os.environ.get("DEBUG_ARG","").find("synth_decay")!=-1:
                            if epoch==0 and i==0:
                                print("DEBUG: change ratio of synthetic data from 1 to 0.5")
                            assert 'extras' in batch.non_tensor_batch
                            if 'extras' in batch.non_tensor_batch:
                                for i in range(len(batch.non_tensor_batch['extras'])):
                                    assert 'evaluator' in batch.non_tensor_batch['extras'][i]
                                    evaluator = batch.non_tensor_batch['extras'][i]['evaluator']
                                    if evaluator != 'env':
                                        batch.batch["advantages"][i] *= 0.5  # ⭐ 对合成数据应用衰减因子

                    # 更新 Critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)  # ⭐ 更新 Critic 模型
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # 实现 Critic 热身 (Warmup)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # 更新 Actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)  # ⭐ 使用新批次更新 Actor
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    
                    # 收集总结任务结果
                    if summary_task is not None:
                        time_cost = self.exp_manager.collect_summary_result(summary_task)
                        metrics.update({"exp_manager/summary": time_cost})


                    # 如果启用，记录 Rollout 生成结果
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )  # ⭐ 转储生成的经验和轨迹

                            # 保存原始轨迹
                            filename = os.path.join(rollout_data_dir, f"traj_{self.global_steps}.jsonl")
                            with open(filename, "w") as f:
                                for traj in trajectories:
                                    f.write(traj.json() + "\n")
                            # 保存任务
                            filename = os.path.join(rollout_data_dir, f"task_{self.global_steps}.jsonl")
                            with open(filename,"w") as f:
                                for task in tasks: # this must be bounded # type: ignore
                                    f.write(task.json() + "\n")

                    # 验证
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()  # ⭐ 验证模型并收集指标
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()  # ⭐ 保存当前模型状态为检查点

                # 训练指标
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "training/num_not_none_traj": num_not_none_traj,
                        "training/num_term_traj": num_term_traj
                    }
                )
                # 收集指标
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # 记录日志
                logger.log(data=metrics, step=self.global_steps)  # ⭐ 记录收集到的指标

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

            # 调试选项：随着训练进行，减少合成数据的比例
            if os.environ.get("DEBUG_ARG",'').find("ratio_decay")!=-1:
                from agentevolver.module.task_manager.data_mixture import UnifiedMixtureStrategy
                print("DEBUG: change ratio of synthetic data from 1 to 0.5")
                assert isinstance(self.train_dataset._mixture_strategy,UnifiedMixtureStrategy)
                self.train_dataset._mixture_strategy._synthetic_ratio-=1/5 # 初始为 1, 约在第 5 个 epoch (约第 30 步) 降为 0
            self.train_dataset.update()  # ⭐ 更新训练数据集 (生成新任务或混合数据) 以进行下一个 Epoch