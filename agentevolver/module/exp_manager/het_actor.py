# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor (单进程 Actor 实现)
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

__all__ = ['HETDataParallelPPOActor']

from verl.workers.actor.dp_actor import DataParallelPPOActor


class HETDataParallelPPOActor(DataParallelPPOActor):
    """
    HET (Hybrid Experience Training) 数据并行 PPO Actor。
    
    继承自 DataParallelPPOActor，主要修改了 `update_policy` 方法以支持混合经验训练。
    在计算 Loss 时，它会区分 On-Policy（探索）和 Off-Policy（经验）的 Token。
    """
    def __init__(self, **kwargs):
        """
        初始化 HETDataParallelPPOActor。
        """
        super().__init__(**kwargs)

    def update_policy(self, data: DataProto):
        """
        使用 PPO 算法更新策略模型，支持 HET 逻辑。
        
        流程包括：
        1. 准备数据（特别是 multi_turn 和 exp_mask）。
        2. 将数据切分为 Mini-batches 和 Micro-batches。
        3. 前向传播计算新的 Log Probabilities。
        4. 调用 HET 专用的 Loss 计算函数，处理 On/Off Policy 混合 loss。
        5. 反向传播与优化。

        Args:
            data (DataProto): 包含更新策略所需信息的 DataProto 对象。
        """
        # 确保 Actor 模块处于训练模式 (启用 Dropout 等)
        self.actor_module.train()  # ⭐ Ensure the actor module is in training mode

        temperature = data.meta_info["temperature"]  # 温度参数必须存在，否则可能会有静默错误
        multi_turn = data.meta_info.get("multi_turn", False) # 检查是否为多轮对话模式
        
        ##################
        # ANNI 修改部分：数据选择
        # 定义需要从 Batch 中提取的键
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            # 如果是多轮对话，需要 loss_mask 来屏蔽 Prompt 部分的 loss
            select_keys.append("loss_mask")
            # ⭐ HET 关键：需要 exp_mask 来标识哪些 token 是来自经验回放（Off-Policy）
            select_keys.append("exp_mask")
        
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob") # 如果计算 KL 散度，需要参考模型的 log prob
        
        # 从 DataProto 中选择数据构建 Batch
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        ##################

        # 1. 拆分 Mini-batch 用于更新 Actor
        # 详见 PPO 论文 https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            # 多模态数据处理逻辑
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            # 普通文本数据处理逻辑
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        # PPO Epoch 循环
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # 2. 将 Mini-batch 拆分为 Micro-batches (用于梯度累积)
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    # 动态 Batch Size (按 Token 数平衡)
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # 标准拆分
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()  # ⭐ 计算新梯度前清零

                for data in micro_batches:
                    # 硬件兼容性处理 (GPU/CPU/NPU)
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_device_id())  # 使用 Offload 时 Actor 设备可能是 CPU
                    
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    
                    # 确定计算 loss 的 mask
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    # 获取 PPO 裁剪参数
                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # 3. 前向传播：获取当前模型的 Log Probabilities 和 Entropy
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)  # ⭐ Forward pass

                    ##################
                    # ANNI 0814 - HET 核心逻辑
                    from .het_core_algos import het_compute_token_on_off_policy_loss
                    
                    off_cliprange_high = self.config.off_cliprange_high # Off-policy 数据的裁剪上限
                    
                    # 获取经验掩码 (1 表示有经验辅助/Off-policy，0 表示无经验/On-policy)
                    exp_mask = data["exp_mask"][:, -response_length:]
                    
                    # 4. 计算 HET Policy Loss
                    # 这个函数会分别计算 On-policy loss 和 Off-policy loss
                    ret_dict = het_compute_token_on_off_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        exp_mask=exp_mask,   # (bs, response_length) ANNI add: 1 w/ exp(off-policy); 0 w/o exp(on-policy)
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        off_cliprange_high=off_cliprange_high, # Off-policy 往往允许更大的 clip range 或单侧裁剪
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )  # ⭐ Compute on-policy and off-policy losses
                    
                    pg_loss = ret_dict["pg_loss"]       # 总 Policy Loss
                    pg_losses = ret_dict["pg_losses"]
                    on_pg_losses = ret_dict["on_pg_losses"]
                    off_pg_losses = ret_dict["off_pg_losses"]
                    on_pg_loss = ret_dict["on_pg_loss"]
                    off_pg_loss = ret_dict["off_pg_loss"]
                    on_pg_clipfrac = ret_dict["on_pg_clipfrac"] # On-policy 数据的裁剪比例
                    on_pg_clipfrac_lower = ret_dict["on_pg_clipfrac_lower"]
                    ppo_kl = ret_dict["ppo_kl"]
                    ##################
                    
                    # 5. 计算 Entropy Loss
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)  # ⭐ Aggregate entropy loss

                        # 最终 Policy Loss = PG Loss - Entropy Bonus
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    # 6. 计算 KL Loss (正则项)
                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # 计算 KL 散度
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)  # ⭐ Compute KL divergence penalty
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)  # ⭐ Aggregate KL divergence loss

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # 7. 反向传播
                    if self.config.use_dynamic_bsz:
                        # 动态 Batch Size 需要按比例缩放 Loss
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        # 梯度累积平均
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()  # ⭐ Backpropagate the loss

                    ##################
                    # ANNI TODO: 添加指标记录
                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/on_pg_clipfrac": on_pg_clipfrac.detach().item(), # 记录 On-policy 数据的裁剪率
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/on_pg_clipfrac_lower": on_pg_clipfrac_lower.detach().item(),
                    }
                    ##################
                    append_to_dict(metrics, data)

                # 8. 优化器步进 (包含梯度裁剪)
                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        
        self.actor_optimizer.zero_grad()
        return metrics