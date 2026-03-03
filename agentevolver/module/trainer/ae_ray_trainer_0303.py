# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications copyright 2025 Alibaba Tongyi EconML Lab. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
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
from torch.utils.data import SequentialSampler, IterableDataset, Dataset, Sampler
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
from agentevolver.module.task_manager import TaskManager, NaiveTaskObjectiveRetrieval
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

from agentevolver.utils.tracking import ValidationGenerationsLogger

from agentevolver.module.adv_processor.adca_grpo_pipeline import apply_adca_grpo

from agentevolver.module.exp_manager.exp_manager import ExperienceManager

# =============================================================================
# [新增] Debug 日志工具函数
# =============================================================================
import os
import time

DEBUG_BASE_DIR = "/mnt/cephfs/haowengao/test_agent/GEN_NEW_DATA"
try:
    os.makedirs(DEBUG_BASE_DIR, exist_ok=True)
except Exception as e:
    print(f"[Warning] Failed to create debug dir {DEBUG_BASE_DIR}: {e}")

# 创建唯一的日志文件名 (带时间戳)
DEBUG_LOG_FILE = os.path.join(DEBUG_BASE_DIR, f"debug_adv_calc.log")

def get_token_context_string(tokenizer, input_ids_tensor, batch_idx, token_idx, window=10):
    """
    提取指定 Token 前后 window 范围内的文本，并将中心 Token 高亮。
    支持完整 input_ids 维度的绝对索引。
    """
    if input_ids_tensor is None or tokenizer is None:
        return ""
    
    seq_len = input_ids_tensor.size(1)
    start_idx = max(0, token_idx - window)
    end_idx = min(seq_len, token_idx + window + 1)
    
    context_str = ""
    for i in range(start_idx, end_idx):
        token_id = input_ids_tensor[batch_idx, i].item()
        
        # 防御性编程：防止传入 -100 等忽略标签导致 decode 崩溃
        if token_id < 0:
            token_text = f"<unk_{token_id}>"
        else:
            token_text = tokenizer.decode([token_id]) if tokenizer else str(token_id)
        
        if i == token_idx:
            context_str += f"【{token_text}】"  # 高亮目标 Token
        else:
            context_str += token_text
            
    return repr(context_str) # 使用 repr 使得 \n 等转义字符可视化

def write_debug_log(msg):
    """
    将调试信息追加写入到指定文件中。
    """
    try:
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {msg}\n")
    except Exception as e:
        print(f"[LOG ERROR] Failed to write to {DEBUG_LOG_FILE}: {e}")
        print(msg)

def compute_single_component_advantage(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor, 
    uid_index: np.ndarray, 
    norm_by_std: bool = True, 
    epsilon: float = 1e-8,
    mode: str = "sparse",
    tokenizer = None,          
    input_ids = None,          
    component_name: str = "Unknown" 
):
    # 【修改】：强制 100% 打印，不再随机抽样
    do_debug_print = True
    
    if mode == "dense":
        advantage_component = token_level_rewards * response_mask
        
        if do_debug_print and input_ids is not None:
            bsz = token_level_rewards.shape[0]
            log_lines = [] 
            
            # 【修改】：不再只取第一条 (min(1, bsz))，而是遍历整个 Batch 所有的轨迹
            for i in range(bsz):
                valid_indices = torch.nonzero(response_mask[i]).squeeze(-1)
                for orig_token_idx in valid_indices:
                    rew = token_level_rewards[i, orig_token_idx].item()
                    # 只要奖励不为0，就记录下来
                    if rew != 0.0:
                        context_str = get_token_context_string(tokenizer, input_ids, i, orig_token_idx.item(), window=10)
                        log_lines.append(f"    [Traj {i:2d}] Abs_Token_Idx [{orig_token_idx.item():4d}] | Reward: {rew:8.4f} | Context: {context_str}")
            
            # 打印该 Component 的所有非0奖励
            if log_lines:
                write_debug_log(f"\n>>> [Advantage Logic - Micro] {component_name} (Dense Mode) ALL Non-zero assignments in Batch:")
                for line in log_lines:
                    write_debug_log(line)
            else:
                write_debug_log(f"\n>>> [Advantage Logic - Micro] {component_name} (Dense Mode): All rewards are 0.0 for ALL trajectories in this Batch.")
                        
        return advantage_component

    elif mode == "sparse":
        scores = token_level_rewards.sum(dim=-1) 
        id2score = defaultdict(list)
        id2mean, id2std = {}, {}
        bsz = scores.shape[0]
        
        for i in range(bsz):
            id2score[uid_index[i]].append(scores[i])

        for idx in id2score:
            vals = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in id2score[idx]]).float()
            if len(vals) > 1:
                id2mean[idx] = torch.mean(vals)
                id2std[idx] = torch.std(vals) + epsilon
            else:
                id2mean[idx] = vals[0]
                id2std[idx] = torch.tensor(1.0, device=vals.device)

        normalized_scores = torch.zeros_like(scores)
        for i in range(bsz):
            mean = id2mean[uid_index[i]].to(scores.device)
            std = id2std[uid_index[i]].to(scores.device)
            if norm_by_std:
                normalized_scores[i] = (scores[i] - mean) / std
            else:
                normalized_scores[i] = scores[i] - mean

        advantage_component = normalized_scores.unsqueeze(-1) * response_mask
        return advantage_component
    else:
        raise ValueError(f"Unknown mode: {mode}")

def compute_outcome_advantage_dual_strategy(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    uid_index: np.ndarray,
    gamma: float = 1.0,
    strategy: str = "normalize_then_decay", 
    epsilon: float = 1e-8,
    norm_adv_by_std: bool = True,
    tokenizer = None,       
    input_ids = None,       
    step_ids = None,        
    step_info: str = ""     
):
    raw_scores = token_level_rewards.sum(dim=-1) 
    lengths = response_mask.sum(dim=-1)
    bsz = raw_scores.shape[0]
    
    id2data = defaultdict(list)
    for i in range(bsz):
        id2data[uid_index[i]].append((raw_scores[i], lengths[i], i))

    dense_advantages = torch.zeros_like(token_level_rewards)

    do_debug_print = True 
    printed_group = False 

    if do_debug_print:
        write_debug_log(f"\n=========================================================")
        write_debug_log(f">>> [Advantage Logic] {step_info} | Strategy: {strategy} | Gamma: {gamma} (Step-level Decay)")

    for uid, group_items in id2data.items():
        g_scores = torch.stack([x[0] for x in group_items])
        g_lens = torch.stack([x[1] for x in group_items])
        g_idxs = [x[2] for x in group_items]
        
        if do_debug_print: 
            write_debug_log(f"\n  [Macro Log] Group UID: {uid} (Group Size: {len(g_scores)}):")
            write_debug_log(f"    - Raw Scores: {g_scores.tolist()}")
            write_debug_log(f"    - Valid Token Lengths: {g_lens.tolist()}")

        if strategy == "normalize_then_decay":
            g_mean = g_scores.mean()
            g_std = g_scores.std()
            if g_std < epsilon:
                g_adv = torch.zeros_like(g_scores)
            else:
                g_adv = (g_scores - g_mean) / (g_std + epsilon) if norm_adv_by_std else (g_scores - g_mean)
            
            if do_debug_print:
                write_debug_log(f"    - Outcome Mean: {g_mean.item():.4f}, Std: {g_std.item():.4f}")
                write_debug_log(f"    - Base Group Adv (Before Decay): {g_adv.tolist()}")
                
        elif strategy == "strict_consistency":
            discounted_returns = g_scores * (gamma ** g_lens)
            g_mean = discounted_returns.mean()
            g_std = discounted_returns.std() + epsilon
            g_adv = (discounted_returns - g_mean) / g_std if norm_adv_by_std else (discounted_returns - g_mean)
        else:
            discounted_returns = g_scores * (gamma ** g_lens)
            g_std = discounted_returns.std()
            if g_std < epsilon:
                g_adv = torch.zeros_like(g_scores)
            else:
                g_mean = discounted_returns.mean()
                g_adv = (discounted_returns - g_mean) / (g_std + epsilon) if norm_adv_by_std else (discounted_returns - g_mean)

        # === 回填到 Token (按 Step 向后衰减) ===
        for local_i, global_idx in enumerate(g_idxs):
            adv_scalar = g_adv[local_i]
            current_mask = response_mask[global_idx]
            valid_indices = torch.nonzero(current_mask).squeeze(-1)
            
            if valid_indices.numel() == 0:
                continue
                
            num_valid = valid_indices.numel()
            
            # [核心机制修改]：获取 Step IDs，计算按步数的距离
            if step_ids is not None:
                # 【终极防崩网】：强制限制 valid_indices 最大值，防止任何维度的 off-by-one 越界
                safe_valid_indices = torch.clamp(valid_indices, max=step_ids.size(1) - 1)
                traj_step_ids = step_ids[global_idx, safe_valid_indices]
                
                max_step = traj_step_ids.max()
                dist_to_end = max_step - traj_step_ids
                dist_to_end = torch.clamp(dist_to_end, min=0).float()
            else:
                dist_to_end = torch.zeros(num_valid, device=raw_scores.device)
                
            token_advs = adv_scalar * (gamma ** dist_to_end)
            dense_advantages[global_idx, valid_indices] = token_advs

            # [Micro Log]
            if do_debug_print and input_ids is not None:
                write_debug_log(f"\n  [Micro Log] Trajectory Index {global_idx} Token-level Decay Assignment:")
                write_debug_log(f"    Base Advantage (Scalar): {adv_scalar.item():.4f}")
                
                for step_idx, orig_token_idx in enumerate(valid_indices):
                    if num_valid > 20 and (5 <= step_idx < num_valid - 5):
                        if step_idx == 6:
                            write_debug_log("      ... (middle tokens skipped in log) ...")
                        continue
                    
                    context_str = get_token_context_string(tokenizer, input_ids, global_idx, orig_token_idx.item(), window=10)
                    cur_dist = dist_to_end[step_idx].item()
                    adv_val = token_advs[step_idx].item()
                    cur_step_id = traj_step_ids[step_idx].item() if step_ids is not None else 0
                    
                    write_debug_log(f"      Step_ID: {cur_step_id:3d} | Step_Dist: {int(cur_dist):3d} | Adv: {adv_val:8.4f} | Context: {context_str}")

    return dense_advantages


def parse_reward_from_dataproto(data: DataProto, return_dict=False) -> dict | torch.Tensor:
    """
    [安全版] 解析 DataProto，分离各项奖励。增加了越界防护。
    """
    device = data.batch["input_ids"].device
    full_seq_shape_tensor = data.batch["input_ids"]
    
    outcome_tensor = torch.zeros_like(full_seq_shape_tensor, dtype=torch.float32, device=device)
    api_tensor = torch.zeros_like(full_seq_shape_tensor, dtype=torch.float32, device=device)
    rep_tensor = torch.zeros_like(full_seq_shape_tensor, dtype=torch.float32, device=device)
    eff_tensor = torch.zeros_like(full_seq_shape_tensor, dtype=torch.float32, device=device)
    
    reward_extra_info = defaultdict(list)
    prompt_lengths = data.batch["prompts"].shape[-1]
    
    response_mask = data.batch["attention_mask"][:, prompt_lengths:]
    response_lengths = response_mask.sum(dim=1)
    step_ids = data.batch.get("step_ids", None)

    # --- [越界防护] 过滤长度为0的无效样本 ---
    valid_response_mask = (response_lengths > 0)
    
    last_token_indices = prompt_lengths + response_lengths - 1
    max_len = full_seq_shape_tensor.shape[1]
    last_token_indices = torch.clamp(last_token_indices, max=max_len - 1)

    batch_indices = torch.arange(len(data), device=device)

    reward_scores_obj = data.non_tensor_batch["reward_scores"]
    outcome_list = [item["outcome"] for item in reward_scores_obj]
    
    if valid_response_mask.any():
        valid_batch_idxs = batch_indices[valid_response_mask]
        valid_token_idxs = last_token_indices[valid_response_mask]
        valid_outcome_vals = torch.tensor(outcome_list, dtype=torch.float32, device=device)[valid_response_mask]
        outcome_tensor[valid_batch_idxs, valid_token_idxs] = valid_outcome_vals

    eff_list = [item.get("metadata", {}).get("efficiency_score", 0.0) for item in reward_scores_obj]
    if valid_response_mask.any():
        valid_eff_vals = torch.tensor(eff_list, dtype=torch.float32, device=device)[valid_response_mask]
        eff_tensor[valid_batch_idxs, valid_token_idxs] = valid_eff_vals

    if step_ids is not None:
        batch_size, seq_len = step_ids.shape 
        api_scores_batch = [item.get("step_api_rewards", []) for item in reward_scores_obj]
        rep_scores_batch = [item.get("step_repetition_rewards", []) for item in reward_scores_obj]

        for b in range(batch_size):
            if not valid_response_mask[b]: continue 

            valid_steps = step_ids[b]
            cur_api = api_scores_batch[b]
            cur_rep = rep_scores_batch[b]
            
            for t in range(seq_len):
                if t >= response_lengths[b]: break
                s_id = valid_steps[t].item()
                if s_id < 0: continue
                
                is_last_token_of_step = (t == seq_len - 1) or \
                                        (t + 1 < seq_len and step_ids[b, t+1].item() != s_id) or \
                                        (t + 1 >= response_lengths[b])
                
                if is_last_token_of_step:
                    write_pos = prompt_lengths + t
                    if write_pos < max_len:
                        if s_id < len(cur_api): 
                            api_tensor[b, write_pos] = cur_api[s_id]
                        if s_id < len(cur_rep): 
                            rep_tensor[b, write_pos] = cur_rep[s_id]

    data.batch["outcome_reward_tensor"] = outcome_tensor
    data.batch["api_reward_tensor"] = api_tensor
    data.batch["rep_reward_tensor"] = rep_tensor
    data.batch["eff_reward_tensor"] = eff_tensor
    total_reward_tensor = outcome_tensor + api_tensor + rep_tensor + eff_tensor

    if return_dict:
        return {"reward_tensor": total_reward_tensor, "reward_extra_info": reward_extra_info}
    else:
        return total_reward_tensor


# =============================================================================
# [缺失补充] 辅助函数 (请确保它们在 class AgentEvolverRayPPOTrainer 之前定义)
# =============================================================================

def create_rl_sampler(data_config, dataset):
    """
    为数据集创建采样器。
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler

def union_gen_batch_via_task_id(tasks, batch: DataProto, gen_batch_output: DataProto):
    """
    [终极修复版] 彻底放弃不唯一的 task_id 对齐！
    利用框架原生的 group_ids 或真实 Prompt Token 指纹进行 100% 绝对安全的对齐。
    """
    from collections import defaultdict
    import torch
    
    # =========================================================================
    # 方案 1：最优解 - 使用框架原生专用的 group_ids (最安全、最精准)
    # group_ids 记录了每一条生成轨迹对应的原始 batch 索引，完美解决同 task_id 冲突
    # =========================================================================
    if 'group_ids' in gen_batch_output.batch:
        group_ids = gen_batch_output.batch['group_ids']
        # 展平 tensor
        if group_ids.dim() > 1:
            group_ids = group_ids.squeeze(-1)
        group_ids_list = group_ids.tolist()
        
        # 严谨的防崩校验：确保索引列表长度对得上，且最大索引不越界
        if isinstance(group_ids_list, list) and len(group_ids_list) == len(gen_batch_output) and max(group_ids_list) < len(batch):
            logger.info(f"✅ Successfully aligned {len(gen_batch_output)} trajectories using native 'group_ids'.")
            batch_extend = batch.select_idxs(group_ids_list)
            return batch_extend.union(gen_batch_output)

    # =========================================================================
    # 方案 2：备用解 - 如果 group_ids 缺失，则使用 Prompt Token 内容作为指纹
    # 因为不同 Prompt 的 token ID 序列是绝对不一样的，借此完美区分不同的 prompt
    # =========================================================================
    logger.warning("⚠️ Native 'group_ids' not found or invalid. Falling back to Token-based fingerprint matching.")
    prompt_to_batch_idx = defaultdict(list)
    
    # 提取输入 Batch 的 Prompt 指纹
    for i in range(len(batch)):
        p_tensor = batch.batch['prompts'][i]
        # 过滤掉常见的 pad / 特殊 token (比如 0, 1, 2)，保留核心文字作为指纹
        core_tokens = tuple(tok for tok in p_tensor.tolist() if tok > 10)
        prompt_to_batch_idx[core_tokens].append(i)
        
    indices = []
    # 根据生成的输出去找对应的原始 Prompt
    for j in range(len(gen_batch_output)):
        p_tensor = gen_batch_output.batch['prompts'][j]
        core_tokens = tuple(tok for tok in p_tensor.tolist() if tok > 10)
        
        if core_tokens in prompt_to_batch_idx and len(prompt_to_batch_idx[core_tokens]) > 0:
            # 如果存在完全一模一样的 Prompt (指纹 100% 相同)，
            # 它们在数学上等效，统一指向第一个索引即可
            idx = prompt_to_batch_idx[core_tokens][0]
            indices.append(idx)
        else:
            error_msg = f"[CRITICAL ERROR] Generated trajectory {j} cannot find its original prompt fingerprint!"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    batch_extend = batch.select_idxs(indices)
    return batch_extend.union(gen_batch_output)


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
                stacked_scores = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(stacked_scores)
                id2std[idx] = torch.std(stacked_scores)
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

def align_dense_rewards_to_model(reward_tensor, loss_mask):
    """
    [核心修复算法]：解决 Tokenizer 边界错位导致的奖励丢失。
    将落在环境反馈区域 (loss_mask=0) 上的奖励，强制向左吸附到
    最近的一个属于模型生成的有效 Token (loss_mask=1) 身上（如 <|im_end|>）。
    """
    if reward_tensor is None:
        return None
    aligned_tensor = torch.zeros_like(reward_tensor)
    bsz, seq_len = reward_tensor.shape
    for b in range(bsz):
        last_valid_idx = -1
        for i in range(seq_len):
            if loss_mask[b, i] == 1:
                last_valid_idx = i
            
            rew = reward_tensor[b, i].item()
            if rew != 0.0:
                # 如果左侧有合法的模型 Token，吸附过去
                if last_valid_idx != -1:
                    aligned_tensor[b, last_valid_idx] += rew
                else:
                    # 极小概率边界情况：句首就没有合法 Token，原样保留
                    aligned_tensor[b, i] += rew
    return aligned_tensor

def compute_advantage(
    data: DataProto, 
    adv_estimator, 
    gamma=1.0, 
    lam=1.0, 
    num_repeat=1, 
    multi_turn=False, 
    norm_adv_by_std_in_grpo=True, 
    config=None,
    tokenizer=None,     
    global_steps=0      
):
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    
    if adv_estimator == AdvantageEstimator.GAE:
        from verl.trainer.ppo import core_algos
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
        if "loss_mask" in data.batch:
            loss_mask = data.batch["loss_mask"]
        else:
            loss_mask = data.batch["response_mask"]
            
        base_tensor = data.batch.get("outcome_reward_tensor", data.batch["token_level_rewards"])
        if loss_mask.size(1) != base_tensor.size(1):
            pad_len = base_tensor.size(1) - loss_mask.size(1)
            padded_loss_mask = torch.zeros_like(base_tensor)
            if pad_len > 0:
                padded_loss_mask[:, pad_len:] = loss_mask
                loss_mask = padded_loss_mask

        uid_index = data.non_tensor_batch["uid"]
        full_input_ids_tensor = data.batch.get("input_ids", None)
        
        # =====================================================================
        # 🛠️ [核心修正与进阶防御] 提取并精确对齐 step_ids
        # =====================================================================
        step_ids_tensor = data.batch.get("step_ids", None)
        padded_step_ids = None
        
        if step_ids_tensor is not None:
            full_seq_len = base_tensor.size(1)
            step_len = step_ids_tensor.size(1)
            
            # 1. 动态获取真实的 Prompt 长度 (这是位移的唯一真理)
            if "prompts" in data.batch:
                prompt_length = data.batch["prompts"].shape[1]
            else:
                prompt_length = full_seq_len - step_len
                
            padded_step_ids = torch.zeros(
                (step_ids_tensor.size(0), full_seq_len),
                dtype=step_ids_tensor.dtype,
                device=step_ids_tensor.device
            )
            
            # ---------------------------------------------------------
            # 🕵️‍♂️ [进阶防御] 动态嗅探 Padding 方向与硬性断言
            # ---------------------------------------------------------
            # attention_mask = data.batch.get("attention_mask", None)
            # if attention_mask is not None:
            #     first_mask = attention_mask[0]
                
            #     # 判断逻辑: 如果最后一位是 1 (有效)，且头部是 0 (Padding)，说明序列被整体推到了最右边
            #     is_left_padded = (first_mask[-1] == 1 and first_mask[0] == 0)
                
            #     assert not is_left_padded, (
            #         f"\n[Fatal Error] 🚨 触发对齐防御机制！检测到序列处于 '左侧填充 (Left Padding)' 模式！\n"
            #         f"-> 当前 AgentEvolver 的奖励对齐算法严格基于 '右侧填充 (Right Padding)' 设计。\n"
            #         f"-> 在 Left Padding 下，奖励位置会发生致命错位。\n"
            #         f"-> 请检查 Actor/vLLM 的初始化配置 (如 use_remove_padding 等参数)，确保序列以 Right Padding 生成。"
            #     )
            
            # ---------------------------------------------------------
            # 🚀 绝对安全的对齐写入 (Right Padding 模式)
            # 在 Right Padding 下，内存布局为: [Prompt, Response, Pad]
            # 所以包含 Response 步骤信息的 step_ids 必须从 prompt_length 开始向右写入
            # ---------------------------------------------------------
            if step_len < full_seq_len:
                end_idx = prompt_length + step_len
                
                # 防御: 确保写入范围绝对不会越界
                assert end_idx <= full_seq_len, (
                    f"[Alignment Error] Prompt长度({prompt_length}) + Step长度({step_len}) "
                    f"超出了张量总长度({full_seq_len})!"
                )
                
                padded_step_ids[:, prompt_length:end_idx] = step_ids_tensor
            else:
                padded_step_ids = step_ids_tensor

            # 🛡️ [最终兜底检查] 确保 step_ids 放对位置后，没有抢占 Prompt 的空间
            if loss_mask is not None:
                # 抽查第一条包含有效掩码的轨迹
                for b in range(min(4, loss_mask.size(0))):
                    valid_idx = torch.nonzero(loss_mask[b]).squeeze(-1)
                    if valid_idx.numel() > 0:
                        first_valid = valid_idx[0].item()
                        assert first_valid >= prompt_length, (
                            f"[Mask Error] 轨迹 {b} 的 loss_mask 越界进入了 Prompt 区域！\n"
                            f"-> Prompt 结束于 {prompt_length}，但 Mask 开始于 {first_valid}。"
                        )
        # =====================================================================

        # =====================================================================
        # 🛡️ [应用核心修复]：强制平移吸附 API 和 Repetition 奖励，防止落入无效区
        # =====================================================================
        api_reward_raw = data.batch.get("api_reward_tensor", None)
        if api_reward_raw is not None:
            data.batch["api_reward_tensor"] = align_dense_rewards_to_model(api_reward_raw, loss_mask)
            
        rep_reward_raw = data.batch.get("rep_reward_tensor", None)
        if rep_reward_raw is not None:
            data.batch["rep_reward_tensor"] = align_dense_rewards_to_model(rep_reward_raw, loss_mask)
        # =====================================================================

        process_mode = config.get("process_reward_mode", "dense")
        w_outcome = config.get("w_outcome", 1.0)
        w_efficiency = 0.0
        w_api = config.get("w_api", 1.0)
        w_rep = config.get("w_rep", 1.0)
        outcome_strategy = config.get("outcome_strategy", "normalize_then_decay") 

        # --- 2. Outcome Stream ---
        outcome_tensor = data.batch.get("outcome_reward_tensor", data.batch["token_level_rewards"])
        adv_out = compute_outcome_advantage_dual_strategy(
            token_level_rewards=outcome_tensor,
            response_mask=loss_mask,  
            uid_index=uid_index,
            gamma=gamma, 
            strategy=outcome_strategy,
            norm_adv_by_std=norm_adv_by_std_in_grpo,
            tokenizer=tokenizer,         
            input_ids=full_input_ids_tensor, 
            step_ids=padded_step_ids,        
            step_info=f"Step {global_steps}" 
        )
        
        # --- 3. Process Streams ---
        adv_api = compute_single_component_advantage(
            data.batch.get("api_reward_tensor", torch.zeros_like(data.batch["responses"])),
            loss_mask, uid_index, norm_adv_by_std_in_grpo,
            mode=process_mode,
            tokenizer=tokenizer,         
            input_ids=full_input_ids_tensor, 
            component_name="API Reward"  
        ) if "api_reward_tensor" in data.batch else torch.zeros_like(adv_out)
        
        adv_rep = compute_single_component_advantage(
            data.batch.get("rep_reward_tensor", torch.zeros_like(data.batch["responses"])),
            loss_mask, uid_index, norm_adv_by_std_in_grpo,
            mode=process_mode,
            tokenizer=tokenizer,         
            input_ids=full_input_ids_tensor, 
            component_name="Repetition Penalty" 
        ) if "rep_reward_tensor" in data.batch else torch.zeros_like(adv_out)

        adv_eff = compute_single_component_advantage(
            data.batch.get("eff_reward_tensor", torch.zeros_like(data.batch["responses"])),
            loss_mask, uid_index, norm_adv_by_std_in_grpo,
            mode="sparse",
            tokenizer=tokenizer,         
            input_ids=full_input_ids_tensor, 
            component_name="Efficiency Reward" 
        )

        # --- 4. 非对称门控 (Asymmetric Gating) ---
        # 🚨 强制将 api_gate 设为 1.0，释放早期探索奖励
        api_gate = 1.0

        final_advantages = (w_outcome * adv_out) + (w_efficiency * adv_eff) + \
                           (w_api * adv_api * api_gate) + (w_rep * adv_rep)

        data.batch["advantages"] = final_advantages
        data.batch["returns"] = outcome_tensor + \
                                data.batch.get("api_reward_tensor", 0) + \
                                data.batch.get("rep_reward_tensor", 0)

        # =====================================================================
        # 👑 [终极核查]：打印对齐矩阵表格 (Alignment Verification Table)
        # =====================================================================
        try:
            do_table_print = True
            if do_table_print and full_input_ids_tensor is not None and tokenizer is not None:
                api_rew = data.batch.get("api_reward_tensor")
                rep_rew = data.batch.get("rep_reward_tensor")
                
                for b in range(min(4, final_advantages.shape[0])): 
                    nz_indices = []
                    if api_rew is not None:
                        nz_indices.extend(torch.nonzero(api_rew[b]).squeeze(-1).tolist())
                    if rep_rew is not None:
                        nz_indices.extend(torch.nonzero(rep_rew[b]).squeeze(-1).tolist())
                    
                    if not nz_indices:
                        valid_idx = torch.nonzero(loss_mask[b]).squeeze(-1)
                        if valid_idx.numel() > 0:
                            nz_indices.append(valid_idx[-1].item())
                    
                    if not nz_indices: continue
                    
                    target_idx = nz_indices[0] if isinstance(nz_indices, list) else nz_indices
                    
                    start_idx = max(0, target_idx - 8)
                    end_idx = min(full_input_ids_tensor.size(1), target_idx + 9)
                    
                    log_str = f"\n👑 === [Alignment Verification Table] Step {global_steps} | Trajectory {b} === 👑\n"
                    log_str += f"{'Idx':<5} | {'Token_Text':<18} | {'StepID':<6} | {'API_Rew':<8} | {'Rep_Rew':<8} | {'Out_Adv':<8} | {'Final_Adv':<9}\n"
                    log_str += "-"*80 + "\n"
                    
                    for i in range(start_idx, end_idx):
                        tid = full_input_ids_tensor[b, i].item()
                        ttext = repr(tokenizer.decode([tid])) if tid >= 0 else f"<unk_{tid}>"
                        sid = padded_step_ids[b, i].item() if padded_step_ids is not None else 0
                        api_val = api_rew[b, i].item() if api_rew is not None else 0.0
                        rep_val = rep_rew[b, i].item() if rep_rew is not None else 0.0
                        out_val = adv_out[b, i].item()
                        fadv_val = final_advantages[b, i].item()
                        
                        flag = ">> " if i == target_idx else "   "
                        if api_val != 0 or rep_val != 0:
                            flag = "🚀 " 
                            
                        log_str += f"{flag}{i:<4} | {ttext:<18} | {sid:<6} | {api_val:>8.4f} | {rep_val:>8.4f} | {out_val:>8.4f} | {fadv_val:>9.4f}\n"
                    
                    write_debug_log(log_str)
        except Exception as e:
            write_debug_log(f"[Warning] Failed to print verification table: {e}")

    else:
        from verl.trainer.ppo import core_algos
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        advantages, returns = adv_estimator_fn(**adv_kwargs)
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
        hindsight_manager=Optional[None],
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
        
        # 初始化 LLM Client (用于 Hindsight 等 ADCA 功能)
        self.llm_client = None
        if hasattr(self.config, 'attribution_driven_credit_assignment'):
            # 假设 config 中有相关配置，或者使用默认的 DashScopeClient
            # 这里简单初始化，实际可能需要更多参数
            try:
                self.llm_client = DashScopeClient()
                logger.info("LLM Client initialized for ADCA/Hindsight.")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM Client: {e}")

        # 创建数据加载器
        self._create_dataloader_from_manager(collate_fn, shuffle_trainset)  # ⭐ 从管理器创建数据加载器
        self.hindsight_manager = hindsight_manager


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
        import time  # [Log Add] 引入 time 库

        # [Log Add] 辅助打印函数
        def val_log(msg):
            print(f"[{time.strftime('%H:%M:%S')}] [Validate] {msg}", flush=True)

        val_log(f"Starting validation at step {self.global_steps}...")

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # 用于收集样本以供表格显示的列表
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        # ================= [新增代码 1/3] 初始化合并记录列表 =================
        validation_merged_records = []
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        # ===================================================================

        for i, test_data in enumerate(self.val_dataloader):
            batch_start_time = time.time()
            val_log(f"Processing Validation Batch {i}...")

            test_batch = DataProto.from_single_dict(test_data)

            # 重复测试批次
            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # 我们只在基于规则的 RM 上进行验证
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                val_log("Skipping validation (reward model style is 'model')")
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
            # print(f"test_gen_batch meta info: {test_gen_batch.meta_info}") # Use val_log instead

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
                
                val_log(f"Batch {i}: Starting Rollout ({len(tasks)} tasks)...") # [Log Add]
                print("=" * 10 + "start validate rollout" + "=" * 10)
                
                # 执行验证 Rollout
                # >>> 容易卡住的地方 <<<
                trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="validate", epoch=f"test.1.{i}")  # ⭐ 执行 Rollout 生成轨迹
                
                print("=" * 10 + "end validate rollout" + "=" * 10)
                val_log(f"Batch {i}: Rollout Finished. Count: {len(trajectories)}") # [Log Add]

                test_output_gen_batch = self.env_manager.to_dataproto(trajectories)
                # test_output_gen_batch_padded = self.explorer_manager.rollout(test_gen_batch_padded)
                # test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # 去除填充
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            # print("validation generation end") # Use val_log instead

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

            val_log(f"Batch {i}: Computing Rewards...") # [Log Add]
            # 使用奖励函数进行评估
            result = self.val_reward_fn(test_batch, return_dict=True)  # ⭐ 使用奖励函数评估测试批次
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # ================= [新增代码 2/3] 构建全量合并记录 =================
            if val_data_dir:
                # 注意：此时 input_texts, output_texts, trajectories, tasks, scores 都是当前 batch 的变量
                # 它们在长度和顺序上是一一对应的
                for idx, (traj, task, score, inp_txt, out_txt) in enumerate(zip(trajectories, tasks, scores, input_texts, output_texts)):
                    record = {
                        "step": self.global_steps,
                        "batch_index": i,
                        "sample_index": idx,
                        
                        # --- 基础信息 ---
                        "input": inp_txt,
                        "output": out_txt,
                        "score": score,
                        
                        # --- 任务元数据 ---
                        "task_id": task.task_id,
                        "query": task.query,
                        "ground_truth": getattr(task, 'ground_truth', "N/A"),
                        
                        # --- 交互历史 (关键) ---
                        "interaction_trace": [
                            {
                                "order": step_i,
                                "action": s.action if hasattr(s, 'action') else str(s),
                                "observation": s.observation if hasattr(s, 'observation') else str(s),
                                "reward": s.reward if hasattr(s, 'reward') else 0.0,
                                "is_terminal": s.done if hasattr(s, 'done') else False
                            }
                            for step_i, s in enumerate(traj.steps)
                        ],
                        
                        # --- 错误诊断 ---
                        "error_info": traj.metadata.get("error", None) if hasattr(traj, "metadata") else None,
                        "termination_reason": "success" if traj.is_successful else "failed",
                        
                        # --- 额外奖励信息 ---
                        **{k: v[idx] for k, v in reward_extra_infos_dict.items() if len(v) > idx}
                    }
                    validation_merged_records.append(record)
            # ===================================================================

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            
            val_log(f"Batch {i} Finished. Cost: {time.time() - batch_start_time:.2f}s") # [Log Add]

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # 转储生成结果
        if val_data_dir:
            # self._dump_generations( ... )  <-- 移除旧调用
            
            # ================= [新增代码 3/3] 保存全量合并的 Validation 轨迹文件 =================
            if validation_merged_records:
                val_log("Saving merged validation traces...")
                save_path = os.path.join(val_data_dir, f"{self.global_steps}.jsonl")
                try:
                    os.makedirs(val_data_dir, exist_ok=True) # Ensure dir exists
                    with open(save_path, "w", encoding='utf-8') as f:
                        for record in validation_merged_records:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    logger.info(f"Saved merged validation traces to {save_path}") # Using Ray logger here might be fine if defined
                    val_log(f"Saved to {save_path}")
                except Exception as e:
                    print(f"Failed to save validation traces: {e}")
            # ==============================================================================

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        # 处理并汇总验证指标
        val_log("Computing validation metrics...")
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

        val_log("Validation Complete.")
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
        [修改说明] 
        1. 新增 Generation-Only Mode 检测逻辑。
        2. 如果开启，创建独立归档目录，设置环境变量，触发任务生成，并直接退出。
        """
        from omegaconf import OmegaConf
        from agentevolver.utils.tracking import Tracking
        import threading
        import uuid
        import time  # [Log Add] 引入 time 库

        # [Log Add] 辅助打印函数
        def main_log(msg):
            print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] {msg}", flush=True)

        # ================= [新增] Generation-Only Mode 逻辑 =================
        # 检查是否开启纯生成模式
        generate_task_only = self.config.task_manager.get("generate_task_only", False)
        
        if generate_task_only:
            main_log("🚀 Detected 'generate_task_only' mode. Initializing Generation Sequence...")
            
            # 1. 优先检查是否存在 GEN_OUTPUT_DIR 环境变量
            if "GEN_OUTPUT_DIR" in os.environ:
                isolation_dir = os.environ["GEN_OUTPUT_DIR"]
                # 确保目录存在（即使用户指定了路径，也需要保证文件夹被创建）
                os.makedirs(isolation_dir, exist_ok=True)
                main_log(f"📂 Using existing output directory from ENV: {isolation_dir}")
            else:
                # 2. 若不存在，则创建带时间戳的新目录
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                # 允许通过配置指定前缀，默认为 gen_
                dir_prefix = self.config.get("gen_output_prefix", "gen_")
                isolation_dir = os.path.join(os.getcwd(), f"{dir_prefix}{timestamp_str}")
                
                os.makedirs(isolation_dir, exist_ok=True)
                main_log(f"📂 Created isolation directory: {isolation_dir}")
                
                # 3. 设置环境变量，供 TaskManager 和 AgentFlow 使用
                os.environ["GEN_OUTPUT_DIR"] = isolation_dir
            # 同时也强制修改 config 中的相关路径，双重保险
            # self.config.task_manager.train_data_path = os.path.join(isolation_dir, "train_dataset_cache.json")
            
            # 3. 强制生成 (Force Execution)
            # 通过调用 train_dataset 的 reload_new_task 来触发 TaskManager 的生成逻辑
            # TaskManager 内部会检测环境变量或参数来决定是否忽略断点
            main_log("🔄 Triggering Task Generation (Force Execution)...")
            
            try:
                # 这里的 reload_new_task 会调用 TaskManager.generate_task
                self.train_dataset.reload_new_task()
                
                # 如果需要保存最终生成的 dataset cache
                self.train_dataset.save_to_file()
                
                main_log(f"✅ Generation Complete. All data saved to {isolation_dir}")
            except Exception as e:
                main_log(f"❌ Generation Failed: {e}")
                raise e
            
            # 4. 直接退出程序，跳过 PPO 训练
            main_log("🛑 Exiting program as 'generate_task_only' is active.")
            return
        # ====================================================================

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        main_log("Starting training fit process...")

        # 在做任何事情之前加载检查点
        self._load_checkpoint()
        # 将参数传播到 vLLM
        self.async_rollout_manager.wake_up()
        self.async_rollout_manager.sleep()

        # 初始化经验池
        if self.config.exp_manager.get("init_exp_before_training", False):
            main_log("Initializing experience pool...")
            self.initialize_exp_pool()
            if self.config.exp_manager.get("init_exp_only", False):
                return

        # 在训练前执行验证
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            main_log("Performing pre-train validation...")
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # 添加进度条
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # 我们从第 1 步开始
        self.global_steps += 1
        last_val_metrics = None
        
        # 训练 Epoch 循环
        for epoch in range(self.config.trainer.total_epochs):
            main_log(f"=== Starting Epoch {epoch} ===")
            
            # 动态数据注入逻辑
            if hasattr(self.train_task_manager, 'load_new_hindsight_tasks'):
                new_count = self.train_task_manager.load_new_hindsight_tasks()
                if new_count > 0:
                    main_log(f"🔄 Detected {new_count} new tasks. Refreshing DataLoader...")
                    self._create_dataloader_from_manager(collate_fn=self._collate_fn, shuffle_trainset=True)
                    progress_bar.total = self.total_training_steps
                    progress_bar.refresh()

            for i, batch_dict in enumerate(self.train_dataloader):
                step_start_time = time.time()
                main_log(f"Step {self.global_steps} (Epoch {epoch}.{i}): Started.")

                # Need Delete: hindsight data after each batch to save memory
                if self.hindsight_manager is not None:
                    self.train_dataset.update_hindsight_data()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # 弹出那些用于生成的键
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                # ... (省略中间的 pop 逻辑，保持原样) ...
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
                            main_log(f"Step {self.global_steps}: Generating sequences (Sync)...")
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            # 构造 Task 列表 (省略代码保持原样)
                            tasks = [Task(
                                        task_id=gen_batch.non_tensor_batch["extras"][i]["task_id"],
                                        query=gen_batch.non_tensor_batch["extras"][i]['new_query'],
                                        env_type=self.config.env_service.env_type,
                                        open_query=gen_batch.non_tensor_batch["extras"][i]['open_query'],
                                        evaluator=gen_batch.non_tensor_batch['extras'][i]['evaluator'],
                                        ground_truth=gen_batch.non_tensor_batch['extras'][i]['ground_truth']
                                    ) for i in range(len(gen_batch))
                                    ]
                            
                            task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="sample")
                            
                            main_log(f"Step {self.global_steps}: Generating Rollouts (Async)...") # [Log Add]
                            print("=" * 10 + "start fit rollout" + "=" * 10)
                            
                            trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="sample", epoch=f"train.{epoch}.{i}")
                            
                            assert len(trajectories)>0, "{len(trajectories)=}?"
                            print("=" * 10 + "end fit rollout" + "=" * 10)
                            main_log(f"Step {self.global_steps}: Rollout Finished. Count: {len(trajectories)}") # [Log Add]
                            
                            gen_batch_output = self.env_manager.to_dataproto(trajectories)
                            
                            # 更新关于经验管理器的指标
                            exp_mask_ratio = gen_batch_output.batch["exp_mask"].float().mean()
                            metrics.update({"exp_mask_ratio": exp_mask_ratio.detach().item()})
                            context_time_cost = [x.metadata["context_time_cost"] for x in trajectories if "context_time_cost" in x.metadata]
                            if context_time_cost:
                                metrics.update({
                                    "exp_manager/context_cost_avg":    np.mean(context_time_cost),
                                    "exp_manager/context_cost_max":    np.max(context_time_cost),
                                    "exp_manager/context_cost_min":    np.min(context_time_cost),
                                })

                            print(f"gen_batch_output.info batch.keys={gen_batch_output.batch.keys()}")
                            num_term_traj = sum([traj.is_terminated  for traj in trajectories])
                            num_not_none_traj = sum([len(traj.steps)>0  for traj in trajectories])

                            self.async_rollout_manager.sleep()

                    # 如果使用 RE-Max，需要生成 Baseline
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        main_log(f"Step {self.global_steps}: Generating REMAX baseline...")
                        # ... (REMAX logic) ...
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch.non_tensor_batch['original_extras']=batch_extras
                    batch = union_gen_batch_via_task_id(tasks, batch, gen_batch_output)

                    # batch.batch["response_mask"] = compute_response_mask(batch)

                    prompt_length = batch.batch['prompts'].shape[1]
                    attention_mask = batch.batch['attention_mask'] # 这是全长的 (Prompt + Response) 

                    response_mask = attention_mask.clone()
                    response_mask[:, :prompt_length] = 0 # 把 Prompt 区域盖住，只留 Response

                    batch.batch["response_mask"] = response_mask # 赋值回去

                    # 更新经验池
                    summary_task = self.exp_manager.submit_summary_task(trajectories, self.global_steps)

                    # 平衡批次
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # 计算全局有效 Token
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        main_log(f"Step {self.global_steps}: Computing Rewards...") # [Log Add]
                        # 计算奖励模型分数
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # 重新计算 old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        main_log(f"Step {self.global_steps}: Computing Old Log Probs...") 
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        
                        # ========================================================
                        # [ROBUST CRASH FIX] 动态对齐掩码长度 (带三重安全保险)
                        # ========================================================
                        import torch.nn.functional as F

                        ent_len = entropys.size(1)
                        mask_len = response_masks.size(1)

                        if ent_len != mask_len:
                            if ent_len < mask_len:
                                # 【保险 1：双向边界探测】
                                # 尝试两种截断方式，看底层引擎到底是裁掉了左侧还是右侧的 Padding
                                left_slice = response_masks[:, :ent_len]
                                right_slice = response_masks[:, -ent_len:]
                                
                                # 原始 Mask 中有效 Response Token 的总数
                                orig_valid_tokens = response_masks.sum()
                                
                                if left_slice.sum() == orig_valid_tokens:
                                    # 如果保留左侧，有效 Token 一个没丢，说明底层裁掉的是右侧的空白
                                    response_masks_for_loss = left_slice
                                elif right_slice.sum() == orig_valid_tokens:
                                    # 如果保留右侧，有效 Token 一个没丢，说明底层裁掉的是左侧的空白
                                    response_masks_for_loss = right_slice
                                else:
                                    # 【保险 2：致命错误硬阻断】
                                    # 如果无论怎么截取，都会把值为 1 的有效 Token 切掉，说明序列严重错位！
                                    # 此时必须抛出异常，绝不能让模型用错位的错误数据继续训练。
                                    raise RuntimeError(
                                        f"[Fatal Alignment Error] 无法安全对齐 LogProb 掩码！\n"
                                        f"-> 原始有效 Token 数: {orig_valid_tokens.item()}\n"
                                        f"-> 左截断保留数: {left_slice.sum().item()}, 右截断保留数: {right_slice.sum().item()}\n"
                                        f"-> Entropys 长度: {ent_len}, 原始 Mask 长度: {mask_len}"
                                    )
                            else:
                                # 【保险 3：反向扩充】
                                # 极小概率下，entropys 反而比 mask 长（例如底层强行 Padding 到了某个基数）
                                # 安全的做法是在 mask 右侧补 0（即视为无效区）
                                pad_len = ent_len - mask_len
                                response_masks_for_loss = F.pad(response_masks, (0, pad_len), value=0)
                        else:
                            response_masks_for_loss = response_masks
                        # ========================================================
                            
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks_for_loss, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        
                        # ... (metrics calculation) ...
                        if "rollout_log_probs" in batch.batch.keys():
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
                        with _timer("ref", timing_raw):
                            main_log(f"Step {self.global_steps}: Computing Ref Policy...") # [Log Add]
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            main_log(f"Step {self.global_steps}: Computing Critic Values...") # [Log Add]
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        main_log(f"Step {self.global_steps}: Computing Advantages...") # [Log Add]
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        if os.environ.get("DEBUG_ARG","").find("disable_adv_std")!=-1:
                            if epoch==0 and i==0:
                                print("DEBUG: change norm_adv_by_std_in_grpo from True to False, using batch std!")
                            norm_adv_by_std_in_grpo = False

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                            tokenizer=self.tokenizer,          # [新增行] 传入 tokenizer
                            global_steps=self.global_steps     # [新增行] 传入当前 step
                        )
                        # =====================================================================
                        # [CRASH FIX] 强制对齐 Advantages 和 Returns 的维度
                        # 截取掉 Prompt 部分，仅保留与 Responses 长度一致的后半部分
                        # =====================================================================
                        resp_len = batch.batch["responses"].size(1)
                        
                        if batch.batch["advantages"].size(1) != resp_len:
                            batch.batch["advantages"] = batch.batch["advantages"][:, -resp_len:]
                            
                        if "returns" in batch.batch and batch.batch["returns"].size(1) != resp_len:
                            batch.batch["returns"] = batch.batch["returns"][:, -resp_len:]
                            
                        if "token_level_rewards" in batch.batch and batch.batch["token_level_rewards"].size(1) != resp_len:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"][:, -resp_len:]
                        # =====================================================================

                        # ... (Hindsight logic & ADCA GRPO) ...
                        # ==================== Hindsight 反向归纳逻辑 ====================
                        attribution_cfg = self._get_attribution_config()
                        
                        if getattr(attribution_cfg, "enable_hindsight", False) and getattr(self, "hindsight_manager", None) is not None:
                            try:
                                main_log(f"Step {self.global_steps}: Processing Hindsight...") # [Log Add]
                                prompts = batch.batch['prompts'].tolist()
                                responses = batch.batch['responses'].tolist()
                                
                                if "extras" in batch.non_tensor_batch:
                                    task_ids = [e.get("task_id", "unknown") for e in batch.non_tensor_batch["extras"]]
                                else:
                                    task_ids = batch.non_tensor_batch.get('data_id', ["unknown"] * len(prompts))

                                sample_scores = []
                                token_rewards = batch.batch['token_level_rewards']
                                if hasattr(token_rewards, 'cpu'):
                                    token_rewards = token_rewards.cpu()
                                
                                for _idx in range(len(prompts)):
                                    score = token_rewards[_idx].sum().item()
                                    sample_scores.append(1.0 if score > 0 else 0.0)

                                threading.Thread(
                                    target=self.hindsight_manager.process_failed_batch,
                                    args=(prompts, responses, sample_scores, task_ids),
                                    kwargs={"threshold": 0.0}
                                ).start()
                                
                            except Exception as e:
                                print(f"[Warning] Hindsight logic encountered an error: {e}")

                        # ==================== 开始 ADCA GRPO (如果开启) ====================
                        if getattr(attribution_cfg, 'enable', False):
                            batch, adca_metrics = apply_adca_grpo(
                                batch=batch,
                                attribution_cfg=attribution_cfg,
                                tokenizer=self.tokenizer,
                                global_steps=self.global_steps,
                                epoch=epoch,
                                i=i,
                                llm_client=self.llm_client,
                            )
                            metrics.update(adca_metrics)
                        
                        if os.environ.get("DEBUG_ARG","").find("synth_decay")!=-1:
                            if epoch==0 and i==0:
                                print("DEBUG: change ratio of synthetic data from 1 to 0.5")
                            assert 'extras' in batch.non_tensor_batch
                            if 'extras' in batch.non_tensor_batch:
                                for i in range(len(batch.non_tensor_batch['extras'])):
                                    assert 'evaluator' in batch.non_tensor_batch['extras'][i]
                                    evaluator = batch.non_tensor_batch['extras'][i]['evaluator']
                                    if evaluator != 'env':
                                        batch.batch["advantages"][i] *= 0.5

                    # 更新 Critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            main_log(f"Step {self.global_steps}: Updating Critic Model...") # [Log Add]
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # 实现 Critic 热身 (Warmup)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # 更新 Actor
                        with _timer("update_actor", timing_raw):
                            main_log(f"Step {self.global_steps}: Updating Actor Model...") # [Log Add]
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    
                    # 收集总结任务结果
                    if summary_task is not None:
                        main_log(f"Step {self.global_steps}: Collecting Summary Task...")
                        time_cost = self.exp_manager.collect_summary_result(summary_task)
                        metrics.update({"exp_manager/summary": time_cost})


                    # 如果启用，记录 Rollout 生成结果
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        main_log(f"Step {self.global_steps}: Saving Rollout Data...")
                        # ... (Saving logic) ...
                        with _timer("dump_rollout_generations", timing_raw):
                            os.makedirs(rollout_data_dir, exist_ok=True)
                            
                            # 1. 准备数据
                            # 从 batch 中获取 PPO 计算后的最终分数 (outcome score)
                            scores_list = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            
                            # 获取 inputs/outputs 用于快速预览
                            inputs_text = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs_text = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            
                            # 2. 构建全量合并记录 (Merged Record)
                            merged_records = []
                            for idx, (traj, task, score) in enumerate(zip(trajectories, tasks, scores_list)):
                                record = {
                                    "step": self.global_steps,
                                    "sample_index": idx,
                                    "input": inputs_text[idx],
                                    "output": outputs_text[idx],
                                    "score": score,
                                    "task_id": task.task_id,
                                    "ground_truth": getattr(task, 'ground_truth', "N/A"),
                                    "query": task.query,
                                    "interaction_trace": [
                                        {
                                            "order": i,
                                            "action": s.action if hasattr(s, 'action') else str(s),
                                            "observation": s.observation if hasattr(s, 'observation') else str(s),
                                            "reward": s.reward if hasattr(s, 'reward') else 0.0,
                                            "is_terminal": s.done if hasattr(s, 'done') else False
                                        }
                                        for i, s in enumerate(traj.steps)
                                    ],
                                    "error_info": traj.metadata.get("error", None) if hasattr(traj, "metadata") else None,
                                    **{k: v[idx] for k, v in reward_extra_infos_dict.items() if len(v) > idx}
                                }
                                merged_records.append(record)

                            # 3. 写入同一个文件
                            save_path = os.path.join(rollout_data_dir, f"{self.global_steps}.jsonl")
                            with open(save_path, "w", encoding='utf-8') as f:
                                for record in merged_records:
                                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            
                            # 保存原始轨迹 (备份)
                            filename = os.path.join(rollout_data_dir, f"traj_{self.global_steps}.jsonl")
                            with open(filename, "w") as f:
                                for traj in trajectories:
                                    f.write(traj.json() + "\n")
                            # 保存任务 (备份)
                            filename = os.path.join(rollout_data_dir, f"task_{self.global_steps}.jsonl")
                            with open(filename,"w") as f:
                                for task in tasks:
                                    f.write(task.json() + "\n")

                    # 验证
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        main_log(f"Step {self.global_steps}: Running Evaluation...")
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            main_log(f"Step {self.global_steps}: Saving Checkpoint...")
                            self._save_checkpoint()

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
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # 自定义 WandB 数据提取逻辑
                if "reward_scores" in batch.non_tensor_batch:
                    reward_scores_list = batch.non_tensor_batch["reward_scores"]
                    custom_stats = defaultdict(list)
                    for r_item in reward_scores_list:
                        meta = r_item.get('metadata', {})
                        if meta:
                            for k, v in meta.items():
                                if k.startswith("metric/"):
                                    custom_stats[k].append(v)
                    for k, v_list in custom_stats.items():
                        if v_list:
                            metrics[f"rollout/{k.split('/')[-1]}"] = np.mean(v_list)
                
                # 记录日志
                logger.log(data=metrics, step=self.global_steps)
                
                step_cost = time.time() - step_start_time
                main_log(f"Step {self.global_steps} Finished. Cost: {step_cost:.2f}s") # [Log Add]

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

            if os.environ.get("DEBUG_ARG",'').find("ratio_decay")!=-1:
                from agentevolver.module.task_manager.data_mixture import UnifiedMixtureStrategy
                print("DEBUG: change ratio of synthetic data from 1 to 0.5")
                assert isinstance(self.train_dataset._mixture_strategy,UnifiedMixtureStrategy)
                self.train_dataset._mixture_strategy._synthetic_ratio-=1/5
            if self.hindsight_manager is not None:
                self.train_dataset.update_hindsight_data()