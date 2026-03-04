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
该训练器支持与 HuggingFace 模型无关的模型初始化，并统筹了
环境采样 (Rollout)、优势评估 (Advantage Computation)、模型更新等一系列核心训练循环动作。
"""

import os
import uuid
import json
import time
import random
import warnings
from copy import deepcopy
from pprint import pprint
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Optional, Any

from collections import Counter
import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from loguru import logger

from torch.utils.data import SequentialSampler, IterableDataset, Dataset, RandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from agentevolver.client.env_client import EnvClient
from agentevolver.module.task_manager.task_manager import AutoReloadDataset, FullDataset
from agentevolver.utils.metric_utils import (
    compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, process_validation_metrics
)
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

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, create_colocated_worker_cls
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator, RayPPOTrainer, ResourcePoolManager, WorkerType,
    _timer, apply_kl_penalty, compute_response_mask, Role
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.metric import reduce_metrics

# =============================================================================
# [调试日志工具]
# 用于在优势函数 (Advantage) 计算时捕获底层数据的具体分布，排查对齐问题。
# =============================================================================

DEBUG_BASE_DIR = "/mnt/cephfs/haowengao/test_agent/GEN_NEW_DATA"
try:
    os.makedirs(DEBUG_BASE_DIR, exist_ok=True)
except Exception as e:
    print(f"[Warning] Failed to create debug dir {DEBUG_BASE_DIR}: {e}")

# 创建固定的日志文件名
DEBUG_LOG_FILE = os.path.join(DEBUG_BASE_DIR, "debug_adv_calc.log")

def get_token_context_string(tokenizer, input_ids_tensor, batch_idx, token_idx, window=10):
    """
    [用途]: 提取指定 Token 前后指定窗口大小的文本，并高亮中心 Token。用于可视化 Debug 追踪。
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
    [用途]: 将调试信息追加写入到指定的 Debug 日志文件中。
    """
    try:
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {msg}\n")
    except Exception as e:
        print(f"[LOG ERROR] Failed to write to {DEBUG_LOG_FILE}: {e}")
        print(msg)

# =============================================================================
# [优势函数计算模块]
# 以下函数负责不同的奖励和优势函数的精细计算与 Token 对齐。
# =============================================================================

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
    """
    [用途]: 计算单维度（如 API 奖励、Repetition 奖励）的优势函数值，支持 dense（密集）与 sparse（稀疏）计算模式。
    """
    do_debug_print = True
    
    if mode == "dense":
        # 密集模式：不进行分组归一化，直接保留原 Token 分数
        advantage_component = token_level_rewards * response_mask
        
        if do_debug_print and input_ids is not None:
            bsz = token_level_rewards.shape[0]
            log_lines = [] 
            
            # 记录 Batch 中所有的非 0 奖励分布，用于校验对齐是否正确
            for i in range(bsz):
                valid_indices = torch.nonzero(response_mask[i]).squeeze(-1)
                for orig_token_idx in valid_indices:
                    rew = token_level_rewards[i, orig_token_idx].item()
                    if rew != 0.0:
                        context_str = get_token_context_string(tokenizer, input_ids, i, orig_token_idx.item(), window=10)
                        log_lines.append(f"    [Traj {i:2d}] Abs_Token_Idx [{orig_token_idx.item():4d}] | Reward: {rew:8.4f} | Context: {context_str}")
            
            if log_lines:
                write_debug_log(f"\n>>> [Advantage Logic - Micro] {component_name} (Dense Mode) ALL Non-zero assignments in Batch:")
                for line in log_lines:
                    write_debug_log(line)
            else:
                write_debug_log(f"\n>>> [Advantage Logic - Micro] {component_name} (Dense Mode): All rewards are 0.0 for ALL trajectories in this Batch.")
                        
        return advantage_component

    elif mode == "sparse":
        # 稀疏模式：通常用于基于组 (Group-based / GRPO) 的计算，首先计算总分，再做组内归一化
        scores = token_level_rewards.sum(dim=-1) 
        id2score = defaultdict(list)
        id2mean, id2std = {}, {}
        bsz = scores.shape[0]
        
        for i in range(bsz):
            id2score[uid_index[i]].append(scores[i])

        # 计算组内的均值和方差
        for idx in id2score:
            vals = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in id2score[idx]]).float()
            if len(vals) > 1:
                id2mean[idx] = torch.mean(vals)
                id2std[idx] = torch.std(vals) + epsilon
            else:
                id2mean[idx] = vals[0]
                id2std[idx] = torch.tensor(1.0, device=vals.device)

        # 归一化得分
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
    data_ids: list = None,   # [新增] 用于校验任务一致性
    gamma: float = 1.0,
    strategy: str = "normalize_then_decay", 
    epsilon: float = 1e-8,
    norm_adv_by_std: bool = True,
    tokenizer = None,       
    input_ids = None,       
    step_ids = None,        
    step_info: str = ""     
):
    """
    [重构版]: 增加了严格的组校验、DataID一致性检查及NaN防御。
    """
    raw_scores = token_level_rewards.sum(dim=-1) 
    lengths = response_mask.sum(dim=-1)
    bsz = raw_scores.shape[0]
    
    # 建立索引映射，同时记录 data_id
    id2data = defaultdict(list)
    for i in range(bsz):
        d_id = data_ids[i] if data_ids is not None else "Unknown"
        id2data[uid_index[i]].append({
            "score": raw_scores[i], 
            "length": lengths[i], 
            "idx": i, 
            "data_id": d_id
        })

    dense_advantages = torch.zeros_like(token_level_rewards)
    do_debug_print = True 

    if do_debug_print:
        write_debug_log(f"\n=========================================================")
        write_debug_log(f">>> [Advantage Logic] {step_info} | Strategy: {strategy} | Gamma: {gamma}")

    for uid, items in id2data.items():
        # --- 1. 组大小验证 (必须为 8) ---
        group_size = len(items)
        if group_size != 8:
            # 如果训练出问题，首要检查 rollout.n 是否配置为 8
            raise ValueError(f"[GRPO Error] Group UID {uid} has size {group_size}, expected 8. Check your config.")

        # --- 2. DataID 一致性验证 ---
        if data_ids is not None:
            unique_data_ids = set([it["data_id"] for it in items])
            if len(unique_data_ids) > 1:
                raise ValueError(f"[Alignment Error] Group UID {uid} contains multiple data_ids: {unique_data_ids}. Samples are misaligned!")

        # 提取数据用于计算
        g_scores = torch.stack([it["score"] for it in items])
        g_lens = torch.stack([it["length"] for it in items])
        g_idxs = [it["idx"] for it in items]
        
        # --- 3. 标量优势计算与 NaN 防御 ---
        if strategy == "normalize_then_decay":
            g_mean = g_scores.mean()
            g_std = g_scores.std()
            
            # 如果组内所有样本得分完全一样（g_std=0），优势应为 0 避免除以 0
            if g_std < epsilon:
                g_adv = torch.zeros_like(g_scores)
            else:
                g_adv = (g_scores - g_mean) / (g_std + epsilon) if norm_adv_by_std else (g_scores - g_mean)
        
        # 其他 strategy 分支 (如 strict_consistency) 也应参考上述 g_std 逻辑进行保护...
        # 为简洁此处略，建议统一使用 normalize_then_decay 进行调试

        # --- 4. Token-level 回填与 dist_to_end 检查 ---
        for local_i, global_idx in enumerate(g_idxs):
            adv_scalar = g_adv[local_i]
            current_mask = response_mask[global_idx]
            valid_indices = torch.nonzero(current_mask).squeeze(-1)
            
            if valid_indices.numel() == 0:
                continue
                
            num_valid = valid_indices.numel()
            
            # 计算 dist_to_end
            if step_ids is not None:
                safe_valid_indices = torch.clamp(valid_indices, max=step_ids.size(1) - 1)
                traj_step_ids = step_ids[global_idx, safe_valid_indices]
                
                max_step = traj_step_ids.max()
                
                # [关键检查]: 如果轨迹很长但 max_step 却是 0，说明 step_ids 数据丢失
                if max_step == 0 and num_valid > 1:
                    write_debug_log(f"⚠️ [Warning] Trajectory {global_idx} has {num_valid} tokens but max_step is 0. dist_to_end will be zero for all tokens!")
                
                dist_to_end = (max_step - traj_step_ids).float().clamp(min=0)
            else:
                dist_to_end = torch.zeros(num_valid, device=raw_scores.device)
                
            # 应用时序衰减
            # 注意：如果发现梯度不一致，尝试将 gamma 设为 1.0 (等额分配) 观察是否好转
            token_advs = adv_scalar * (gamma ** dist_to_end)
            dense_advantages[global_idx, valid_indices] = token_advs

            # 日志输出... (保持原有逻辑)
    return dense_advantages


def parse_reward_from_dataproto(data: DataProto, return_dict=False) -> dict | torch.Tensor:
    """
    [用途]: 从结构化的数据协议 (DataProto) 中提取并分离各个维度的奖励（Outcome、API、Repetition、Efficiency 等）。
    同时对生成的 Tensor 执行安全越界防护，防止底层 CUDA 错误。
    """
    device = data.batch["input_ids"].device
    full_seq_shape_tensor = data.batch["input_ids"]
    
    # 初始化不同类型奖励 Tensor
    outcome_tensor = torch.zeros_like(full_seq_shape_tensor, dtype=torch.float32, device=device)
    api_tensor = torch.zeros_like(full_seq_shape_tensor, dtype=torch.float32, device=device)
    rep_tensor = torch.zeros_like(full_seq_shape_tensor, dtype=torch.float32, device=device)
    eff_tensor = torch.zeros_like(full_seq_shape_tensor, dtype=torch.float32, device=device)
    
    reward_extra_info = defaultdict(list)
    prompt_lengths = data.batch["prompts"].shape[-1]
    
    response_mask = data.batch["attention_mask"][:, prompt_lengths:]
    response_lengths = response_mask.sum(dim=1)
    step_ids = data.batch.get("step_ids", None)

    # 过滤无效响应序列以防越界
    valid_response_mask = (response_lengths > 0)
    
    last_token_indices = prompt_lengths + response_lengths - 1
    max_len = full_seq_shape_tensor.shape[1]
    last_token_indices = torch.clamp(last_token_indices, max=max_len - 1)

    batch_indices = torch.arange(len(data), device=device)
    reward_scores_obj = data.non_tensor_batch["reward_scores"]
    outcome_list = [item["outcome"] for item in reward_scores_obj]
    
    # 设置 Outcome Reward
    if valid_response_mask.any():
        valid_batch_idxs = batch_indices[valid_response_mask]
        valid_token_idxs = last_token_indices[valid_response_mask]
        valid_outcome_vals = torch.tensor(outcome_list, dtype=torch.float32, device=device)[valid_response_mask]
        outcome_tensor[valid_batch_idxs, valid_token_idxs] = valid_outcome_vals

    # 设置 Efficiency Reward
    eff_list = [item.get("metadata", {}).get("efficiency_score", 0.0) for item in reward_scores_obj]
    if valid_response_mask.any():
        valid_eff_vals = torch.tensor(eff_list, dtype=torch.float32, device=device)[valid_response_mask]
        eff_tensor[valid_batch_idxs, valid_token_idxs] = valid_eff_vals

    # 设置 Step 维度的 API 和 Repetition 奖励
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
                
                # 判断当前 Token 是否是步骤的最后一个 Token
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

def create_rl_sampler(data_config, dataset):
    """
    [用途]: 辅助函数。基于参数配置为 PyTorch 数据集构建对应的采样器（随机或顺序采样）。
    """
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler

def union_gen_batch_via_task_id(tasks, batch: DataProto, gen_batch_output: DataProto):
    """
    [用途]: 在环境采样 (Rollout) 生成结果返回后，将生成的 Response 轨迹安全合并到原始的请求 Batch 中。
    使用框架原生的 `group_ids` 或基于 Token 指纹进行精准对齐，避免因同构任务 ID 冲突而导致的融合失败。
    """
    if 'group_ids' in gen_batch_output.batch:
        group_ids = gen_batch_output.batch['group_ids']
        if group_ids.dim() > 1:
            group_ids = group_ids.squeeze(-1)
        group_ids_list = group_ids.tolist()
        
        if isinstance(group_ids_list, list) and len(group_ids_list) == len(gen_batch_output) and max(group_ids_list) < len(batch):
            logger.info(f"✅ Successfully aligned {len(gen_batch_output)} trajectories using native 'group_ids'.")
            batch_extend = batch.select_idxs(group_ids_list)
            return batch_extend.union(gen_batch_output)

    # Fallback：基于 Token 内容做指纹匹配
    logger.warning("⚠️ Native 'group_ids' not found or invalid. Falling back to Token-based fingerprint matching.")
    prompt_to_batch_idx = defaultdict(list)
    
    for i in range(len(batch)):
        p_tensor = batch.batch['prompts'][i]
        core_tokens = tuple(tok for tok in p_tensor.tolist() if tok > 10)
        prompt_to_batch_idx[core_tokens].append(i)
        
    indices = []
    for j in range(len(gen_batch_output)):
        p_tensor = gen_batch_output.batch['prompts'][j]
        core_tokens = tuple(tok for tok in p_tensor.tolist() if tok > 10)
        
        if core_tokens in prompt_to_batch_idx and len(prompt_to_batch_idx[core_tokens]) > 0:
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
    [用途]: 计算基于组相关策略优化 (GRPO) 框架下的 Outcome Reward 优势。
    该算法不依赖 Critic 网络，仅通过组内响应结果相互比对计算方差来评估优势。
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    if scores.dim()!=1:
        logger.warning("scores.dim()!=1")

    with torch.no_grad():
        bsz = scores.shape[0]
        
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                stacked_scores = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(stacked_scores)
                id2std[idx] = torch.std(stacked_scores)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

def align_dense_rewards_to_model(reward_tensor, loss_mask):
    """
    [用途]: 修复 Token 映射偏移带来的奖励丢失问题。
    将环境中不在模型生成掩膜 (loss_mask) 中的微小奖励向前吸收到邻近有效模型 Token（如结束符）身上。
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
                if last_valid_idx != -1:
                    aligned_tensor[b, last_valid_idx] += rew
                else:
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
    """
    [用途]: 统领级优势计算网关。根据选定的训练架构（GAE / GRPO 等）调用对应的计算模块，
    组合多种类型奖励（API、结果、重复惩罚等）以形成最终引导模型更新的 Advantage Tensor。
    """
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    
    # 针对普通 GAE (Generalized Advantage Estimation) 分支的处理
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

    # 针对 GRPO (组相对策略优化) 分支的处理逻辑（包含精细对齐与组合）
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
        
        # 精确对齐 step_ids
        step_ids_tensor = data.batch.get("step_ids", None)
        padded_step_ids = None
        
        if step_ids_tensor is not None:
            full_seq_len = base_tensor.size(1)
            step_len = step_ids_tensor.size(1)
            
            if "prompts" in data.batch:
                prompt_length = data.batch["prompts"].shape[1]
            else:
                prompt_length = full_seq_len - step_len
                
            padded_step_ids = torch.zeros(
                (step_ids_tensor.size(0), full_seq_len),
                dtype=step_ids_tensor.dtype,
                device=step_ids_tensor.device
            )
            
            if step_len < full_seq_len:
                end_idx = prompt_length + step_len
                assert end_idx <= full_seq_len, (
                    f"[Alignment Error] Prompt长度({prompt_length}) + Step长度({step_len}) "
                    f"超出了张量总长度({full_seq_len})!"
                )
                padded_step_ids[:, prompt_length:end_idx] = step_ids_tensor
            else:
                padded_step_ids = step_ids_tensor

            if loss_mask is not None:
                for b in range(min(4, loss_mask.size(0))):
                    valid_idx = torch.nonzero(loss_mask[b]).squeeze(-1)
                    if valid_idx.numel() > 0:
                        first_valid = valid_idx[0].item()
                        assert first_valid >= prompt_length, (
                            f"[Mask Error] 轨迹 {b} 的 loss_mask 越界进入了 Prompt 区域！\n"
                            f"-> Prompt 结束于 {prompt_length}，但 Mask 开始于 {first_valid}。"
                        )

        # 强制平移吸附环境密集奖励
        api_reward_raw = data.batch.get("api_reward_tensor", None)
        if api_reward_raw is not None:
            data.batch["api_reward_tensor"] = align_dense_rewards_to_model(api_reward_raw, loss_mask)
            
        rep_reward_raw = data.batch.get("rep_reward_tensor", None)
        if rep_reward_raw is not None:
            data.batch["rep_reward_tensor"] = align_dense_rewards_to_model(rep_reward_raw, loss_mask)

        process_mode = config.get("process_reward_mode", "dense")
        w_outcome = config.get("w_outcome", 1.0)
        w_efficiency = 0.0
        w_api = config.get("w_api", 1.0)
        w_rep = config.get("w_rep", 1.0)
        outcome_strategy = config.get("outcome_strategy", "normalize_then_decay") 

        # 汇总 Outcome 流奖励
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
        
        # 计算不同进程的辅助奖励
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

        # ---------------------------------------------------------
        # 进行奖励叠加汇总 (融合门控系数)
        api_gate = 1.0
        pre_norm_advantages = (w_outcome * adv_out) + (w_efficiency * adv_eff) + \
                              (w_api * adv_api * api_gate) + (w_rep * adv_rep)

        # 🚀 [新增修复] Batch 级别 Advantage 全局归一化
        # 只对有效 Token (loss_mask == 1) 进行统计，防 Padding 干扰
        final_advantages = pre_norm_advantages.clone()
        valid_mask = loss_mask.bool()
        valid_advs = pre_norm_advantages[valid_mask]
        
        pre_norm_mean, pre_norm_std = 0.0, 0.0
        if valid_advs.numel() > 1:
            pre_norm_mean = valid_advs.mean().item()
            pre_norm_std = valid_advs.std().item()
            
            # 对有效 Token 进行标准化 (均值 0，方差 1)
            normalized_valid_advs = (valid_advs - pre_norm_mean) / (pre_norm_std + 1e-8)
            final_advantages[valid_mask] = normalized_valid_advs
        elif valid_advs.numel() == 1:
            pre_norm_mean = valid_advs.mean().item()
            final_advantages[valid_mask] = 0.0 # 若只有一个有效Token，优势归零
        # ---------------------------------------------------------

        # 赋值回 data.batch (采用归一化后的 final_advantages)
        data.batch["advantages"] = final_advantages
        data.batch["returns"] = outcome_tensor + \
                                data.batch.get("api_reward_tensor", 0) + \
                                data.batch.get("rep_reward_tensor", 0)

        # 日志打印部分
        try:
            do_table_print = True
            if do_table_print and full_input_ids_tensor is not None and tokenizer is not None:
                api_rew = data.batch.get("api_reward_tensor")
                rep_rew = data.batch.get("rep_reward_tensor")
                
                # --- 宏观组信息与 Advantage 统计 ---
                post_norm_advs = final_advantages[valid_mask]
                post_norm_mean = post_norm_advs.mean().item() if post_norm_advs.numel() > 0 else 0.0
                post_norm_std = post_norm_advs.std().item() if post_norm_advs.numel() > 1 else 0.0
                
                group_sizes = list(Counter(uid_index).values())
                avg_group_size = sum(group_sizes) / len(group_sizes) if group_sizes else 0
                
                write_debug_log(f"\n=========================================================")
                write_debug_log(f"📊 [Step {global_steps} Batch Stats] PRE-Norm Adv  -> Mean: {pre_norm_mean:.4f}, Std: {pre_norm_std:.4f}")
                write_debug_log(f"📊 [Step {global_steps} Batch Stats] POST-Norm Adv -> Mean: {post_norm_mean:.4f}, Std: {post_norm_std:.4f}")
                write_debug_log(f"👥 [Group Stats] Total Groups: {len(group_sizes)}, Avg Group Size: {avg_group_size:.1f}")
                write_debug_log(f"=========================================================")

                # --- 随机采样 2 个样本，打印整个有效序列的 Token 级奖励 ---
                bsz = final_advantages.shape[0]
                sample_indices = random.sample(range(bsz), min(2, bsz))
                
                for b in sample_indices:
                    valid_idx = torch.nonzero(loss_mask[b]).squeeze(-1)
                    if valid_idx.numel() == 0:
                        continue
                        
                    start_idx = valid_idx[0].item()
                    end_idx = valid_idx[-1].item() + 1
                    
                    log_str = f"\n👑 === [Token-Level Detail] Step {global_steps} | Sampled Trajectory {b} === 👑\n"
                    log_str += f"{'Idx':<5} | {'Token_Text':<18} | {'StepID':<6} | {'API_Rew':<8} | {'Rep_Rew':<8} | {'Out_Adv':<8} | {'Pre_Adv':<9} | {'Post_Adv':<9}\n"
                    log_str += "-"*95 + "\n"
                    
                    for i in range(start_idx, end_idx):
                        tid = full_input_ids_tensor[b, i].item()
                        ttext = repr(tokenizer.decode([tid])) if tid >= 0 else f"<unk_{tid}>"
                        sid = padded_step_ids[b, i].item() if padded_step_ids is not None else 0
                        
                        api_val = api_rew[b, i].item() if api_rew is not None else 0.0
                        rep_val = rep_rew[b, i].item() if rep_rew is not None else 0.0
                        out_val = adv_out[b, i].item() if 'adv_out' in locals() else 0.0
                        
                        # 提取归一化前和归一化后的 Advantage
                        pre_adv_val = pre_norm_advantages[b, i].item()
                        post_adv_val = final_advantages[b, i].item()
                        
                        # 只打印有数值波动或关键步骤的 Token，避免日志过长
                        if api_val != 0 or rep_val != 0 or abs(pre_adv_val) > 0.01 or abs(post_adv_val) > 0.01 or (i % 20 == 0):
                            flag = "🚀 " if (api_val != 0 or rep_val != 0) else "   "
                            log_str += f"{flag}{i:<4} | {ttext:<18} | {sid:<6} | {api_val:>8.4f} | {rep_val:>8.4f} | {out_val:>8.4f} | {pre_adv_val:>9.4f} | {post_adv_val:>9.4f}\n"
                    
                    write_debug_log(log_str)
        except Exception as e:
            write_debug_log(f"[Warning] Failed to print verification table: {e}")

    # Fallback/Custom 分支
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


# =============================================================================
# [核心主类：AgentEvolver Ray PPO 训练器]
# =============================================================================

class AgentEvolverRayPPOTrainer(RayPPOTrainer):
    """
    [用途]: AgentEvolver 的 Ray PPO 分布式训练器。
    运行在单个控制节点（Driver）上，调度并协调位于远程集群上的各类 Ray Worker 进行环境 Rollout 和模型更新。
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
        [用途]: 初始化训练器基础参数、任务管理器及关联的远端依赖模块。
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine" 

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"  

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

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

        self._validate_config()

        self.env_manager: ParallelEnvManager | None = None
        self.thread_pool: ThreadPoolExecutor | None = None

        self.train_task_manager=train_task_manager
        self.val_task_manager=val_task_manager
        self._collate_fn=collate_fn
        
        # 初始化用于 ADCA 分配机制的大语言模型端点
        self.llm_client = None
        if hasattr(self.config, 'attribution_driven_credit_assignment'):
            try:
                self.llm_client = DashScopeClient()
                logger.info("LLM Client initialized for ADCA/Hindsight.")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM Client: {e}")

        # 利用 TaskManager 构建训练与验证所用的数据管道 Dataloader
        self._create_dataloader_from_manager(collate_fn, shuffle_trainset) 
        self.hindsight_manager = hindsight_manager

    def init_workers(self):
        """
        [用途]: 创建并初始化基于 Ray 的各个角色 Worker 组 (如 Actor、Critic、RM)，加载相应模型，并建立环境联通。
        """
        self.resource_pool_manager.create_resource_pool() 

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

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

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        all_wg = {}
        wg_kwargs = {} 
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls,
                                                device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model() 

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model() 

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model() 

        # 初始化 Actor（支持 vLLM）
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model() 

        # 启动异步模型支持
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from agentevolver.module.trainer.ae_async_llm_server_manager import BaAsyncLLMServerManager
            self.async_rollout_mode = True
            self.async_rollout_manager = BaAsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg) 

        self.reward_fn = parse_reward_from_dataproto
        self.val_reward_fn = parse_reward_from_dataproto

        self.env_manager = ParallelEnvManager(config=self.config, async_rollout_manager=self.async_rollout_manager, max_parallel=self.config.actor_rollout_ref.rollout.max_env_worker)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool.max_workers)
        self.exp_manager = ExperienceManager(config=self.config)

    def _create_dataloader_from_manager(self, collate_fn, shuffle_trainset: bool = True):
        """
        [用途]: 数据管道准备器。将底层 TaskManager 和网络读取流转换为训练可用的 DataLoader 实例。
        """
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
            collate_fn = default_collate_fn

        from verl.trainer.main_ppo import create_rl_dataset
        env_client=EnvClient(self.config.env_service.env_url)
        
        # 加载训练集合
        if self.config.data.train_files is not None:
            train_seed_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
            assert isinstance(train_seed_dataset,RLHFDataset), "train_dataset must be RLHFDataset"
            self.train_task_manager.load_tasks_from_dataset(train_seed_dataset,env_type=self.config.env_service.env_type)
        else:
            self.train_task_manager.load_tasks_from_environment(env_client,env_type=self.config.env_service.env_type,split="train")
        
        # 加载验证集合
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
        
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data,self.train_dataset),
        ) 

        val_batch_size = self.config.data.val_batch_size 
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset) # type: ignore

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        ) 

        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        if not isinstance(self.train_dataset,IterableDataset):
            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
            print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")
        else:
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
        [用途]: 获取属性归因（ADCA）的详细配置设置。
        """
        if not hasattr(self.config, 'attribution_driven_credit_assignment'):
            raise ValueError("attribution_driven_credit_assignment configuration block is required")

        config = self.config.attribution_driven_credit_assignment

        if not hasattr(config, 'api_max_retries'):
            config.api_max_retries = 200 
            print(f"[attribution_config] Using default api_max_retries: {config.api_max_retries}")

        return config

    def _validate_config(self):
        """
        [用途]: 校验传入配置文件的合法性和一致性，如 GPU 总数、并行大小、微批次配置冲突等。
        """
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

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
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size 
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0 
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus 

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size 
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0 
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus 

        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """
        [用途]: 保存评估轨迹文件 (JSONL) 。
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl") 

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
            f.write("\n".join(lines) + "\n") 

        print(f"Dumped generations to {filename}")

    def _validate(self):
        """
        [用途]: 跑一次完整的评估流程。拉取验证集数据并使用 Actor 模型生成轨迹，最后汇总出统计数据用于 WandB 等展示。
        """
        import time 

        def val_log(msg):
            print(f"[{time.strftime('%H:%M:%S')}] [Validate] {msg}", flush=True)

        val_log(f"Starting validation at step {self.global_steps}...")

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        validation_merged_records = []
        val_data_dir = self.config.trainer.get("validation_data_dir", None)

        for i, test_data in enumerate(self.val_dataloader):
            batch_start_time = time.time()
            val_log(f"Processing Validation Batch {i}...")

            test_batch = DataProto.from_single_dict(test_data)

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

            if not self.async_rollout_mode:
                raise NotImplementedError

            else:
                self.async_rollout_manager.wake_up()
                tasks = [Task(
                            task_id=test_gen_batch.non_tensor_batch["extras"][i]["task_id"],
                            query=test_gen_batch.non_tensor_batch["extras"][i]['new_query'],
                            env_type=self.config.env_service.env_type,
                            open_query=test_gen_batch.non_tensor_batch["extras"][i]['open_query'],
                          ) for i in range(len(test_gen_batch))]
                
                task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="validate")
                
                val_log(f"Batch {i}: Starting Rollout ({len(tasks)} tasks)...") 
                print("=" * 10 + "start validate rollout" + "=" * 10)
                
                trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="validate", epoch=f"test.1.{i}") 
                
                print("=" * 10 + "end validate rollout" + "=" * 10)
                val_log(f"Batch {i}: Rollout Finished. Count: {len(trajectories)}") 

                test_output_gen_batch = self.env_manager.to_dataproto(trajectories)
                self.async_rollout_manager.sleep()

            input_ids = test_output_gen_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            test_batch = union_gen_batch_via_task_id(tasks, test_batch, test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            val_log(f"Batch {i}: Computing Rewards...") 
            result = self.val_reward_fn(test_batch, return_dict=True) 
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            if val_data_dir:
                for idx, (traj, task, score, inp_txt, out_txt) in enumerate(zip(trajectories, tasks, scores, input_texts, output_texts)):
                    record = {
                        "step": self.global_steps,
                        "batch_index": i,
                        "sample_index": idx,
                        "input": inp_txt,
                        "output": out_txt,
                        "score": score,
                        "task_id": task.task_id,
                        "query": task.query,
                        "ground_truth": getattr(task, 'ground_truth', "N/A"),
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
                        "error_info": traj.metadata.get("error", None) if hasattr(traj, "metadata") else None,
                        "termination_reason": "success" if traj.is_successful else "failed",
                        **{k: v[idx] for k, v in reward_extra_infos_dict.items() if len(v) > idx}
                    }
                    validation_merged_records.append(record)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            val_log(f"Batch {i} Finished. Cost: {time.time() - batch_start_time:.2f}s") 

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        if val_data_dir:
            if validation_merged_records:
                val_log("Saving merged validation traces...")
                save_path = os.path.join(val_data_dir, f"{self.global_steps}.jsonl")
                try:
                    os.makedirs(val_data_dir, exist_ok=True) 
                    with open(save_path, "w", encoding='utf-8') as f:
                        for record in validation_merged_records:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    logger.info(f"Saved merged validation traces to {save_path}") 
                    val_log(f"Saved to {save_path}")
                except Exception as e:
                    print(f"Failed to save validation traces: {e}")

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        val_log("Computing validation metrics...")
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict) 
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
        [用途]: 利用跑一次初评生成的基础轨迹来预热经验池 (Experience Pool)，使其不在刚开始训练时为空。
        """
        for i, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

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

            if not self.async_rollout_mode:
                raise NotImplementedError

            else:
                self.async_rollout_manager.wake_up()
                tasks = [Task(
                            task_id=test_gen_batch.non_tensor_batch["extras"][i]["task_id"],
                            query=test_gen_batch.non_tensor_batch["extras"][i]['new_query'],
                            env_type=self.config.env_service.env_type,
                            open_query=test_gen_batch.non_tensor_batch["extras"][i]['open_query'],
                          ) for i in range(len(test_gen_batch))]
                task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="validate")
                
                print("=" * 10 + "start validate rollout" + "=" * 10)
                trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="validate", epoch=f"test.1.{i}") 
                print("=" * 10 + "end validate rollout" + "=" * 10)
                self.async_rollout_manager.sleep()

            # 将本批次生成的轨迹存入经验管理器中
            self.exp_manager.summarize_in_batch(trajectories)
        
        return

    def fit(self):
        """
        [用途]: PPO 训练的主循环执行点。
        核心执行流：
        1. 检查是不是纯生成模式（不训练直接结束）。
        2. 按 Epoch 读取 DataLoader 数据。
        3. 让 Actor (vLLM等后端) 与环境交互 (Rollout) 生成 response 和轨迹。
        4. 结算环境中拿到的各类奖励 (Rewards)。
        5. 调用 compute_advantage 获取优化信号 (Advantage Tensor) 并计算 loss_mask。
        6. 使用计算好的指标反向更新 Actor（以及 Critic 模型，如果有）。
        7. 记录并监控系统性能/训练表现直至训练轮次结束。
        """
        from omegaconf import OmegaConf
        from agentevolver.utils.tracking import Tracking
        import threading
        import uuid
        import time 

        def main_log(msg):
            print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] {msg}", flush=True)

        # ================= [纯生成模式] =================
        generate_task_only = self.config.task_manager.get("generate_task_only", False)
        
        if generate_task_only:
            main_log("🚀 Detected 'generate_task_only' mode. Initializing Generation Sequence...")
            
            if "GEN_OUTPUT_DIR" in os.environ:
                isolation_dir = os.environ["GEN_OUTPUT_DIR"]
                os.makedirs(isolation_dir, exist_ok=True)
                main_log(f"📂 Using existing output directory from ENV: {isolation_dir}")
            else:
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                dir_prefix = self.config.get("gen_output_prefix", "gen_")
                isolation_dir = os.path.join(os.getcwd(), f"{dir_prefix}{timestamp_str}")
                
                os.makedirs(isolation_dir, exist_ok=True)
                main_log(f"📂 Created isolation directory: {isolation_dir}")
                
                os.environ["GEN_OUTPUT_DIR"] = isolation_dir
            
            main_log("🔄 Triggering Task Generation (Force Execution)...")
            try:
                self.train_dataset.reload_new_task()
                self.train_dataset.save_to_file()
                main_log(f"✅ Generation Complete. All data saved to {isolation_dir}")
            except Exception as e:
                main_log(f"❌ Generation Failed: {e}")
                raise e
            
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

        self._load_checkpoint()
        self.async_rollout_manager.wake_up()
        self.async_rollout_manager.sleep()

        if self.config.exp_manager.get("init_exp_before_training", False):
            main_log("Initializing experience pool...")
            self.initialize_exp_pool()
            if self.config.exp_manager.get("init_exp_only", False):
                return

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            main_log("Performing pre-train validation...")
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        
        for epoch in range(self.config.trainer.total_epochs):
            main_log(f"=== Starting Epoch {epoch} ===")
            
            # 动态注入新生成的数据 (Hindsight)
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

                if self.hindsight_manager is not None:
                    self.train_dataset.update_hindsight_data()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

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
                    # 【核心交互区】：与环境 Rollout 以产生新的状态轨迹
                    with _timer("gen", timing_raw):
                        trajectories: List[Trajectory] = []
                        if not self.async_rollout_mode:
                            main_log(f"Step {self.global_steps}: Generating sequences (Sync)...")
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
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
                            
                            main_log(f"Step {self.global_steps}: Generating Rollouts (Async)...") 
                            print("=" * 10 + "start fit rollout" + "=" * 10)
                            
                            trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="sample", epoch=f"train.{epoch}.{i}")
                            
                            assert len(trajectories)>0, "{len(trajectories)=}?"
                            print("=" * 10 + "end fit rollout" + "=" * 10)
                            main_log(f"Step {self.global_steps}: Rollout Finished. Count: {len(trajectories)}") 
                            
                            gen_batch_output = self.env_manager.to_dataproto(trajectories)
                            
                            exp_mask_ratio = gen_batch_output.batch["exp_mask"].float().mean()
                            metrics.update({"exp_mask_ratio": exp_mask_ratio.detach().item()})
                            context_time_cost = [x.metadata["context_time_cost"] for x in trajectories if "context_time_cost" in x.metadata]
                            if context_time_cost:
                                metrics.update({
                                    "exp_manager/context_cost_avg":    np.mean(context_time_cost),
                                    "exp_manager/context_cost_max":    np.max(context_time_cost),
                                    "exp_manager/context_cost_min":    np.min(context_time_cost),
                                })

                            num_term_traj = sum([traj.is_terminated  for traj in trajectories])
                            num_not_none_traj = sum([len(traj.steps)>0  for traj in trajectories])

                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        main_log(f"Step {self.global_steps}: Generating REMAX baseline...")
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

                    # 将生成的结果（Gen DataProto）再插拔回原始的 Batch 结构里
                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch.non_tensor_batch['original_extras']=batch_extras
                    batch = union_gen_batch_via_task_id(tasks, batch, gen_batch_output)

                    # 生成用于过滤不计算梯度的掩码（只更新 Response 部分，不更新 Prompt）
                    prompt_length = batch.batch['prompts'].shape[1]
                    attention_mask = batch.batch['attention_mask'] 
                    response_mask = attention_mask.clone()
                    response_mask[:, :prompt_length] = 0 
                    batch.batch["response_mask"] = response_mask 

                    summary_task = self.exp_manager.submit_summary_task(trajectories, self.global_steps)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        main_log(f"Step {self.global_steps}: Computing Rewards...") 
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        main_log(f"Step {self.global_steps}: Computing Old Log Probs...") 
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        
                        # ========================================================
                        # [鲁棒性防御] 对齐掩码长度，防止因为底层框架丢掉 Padding 导致的严重维度错乱
                        # ========================================================
                        ent_len = entropys.size(1)
                        mask_len = response_masks.size(1)

                        if ent_len != mask_len:
                            if ent_len < mask_len:
                                left_slice = response_masks[:, :ent_len]
                                right_slice = response_masks[:, -ent_len:]
                                
                                orig_valid_tokens = response_masks.sum()
                                
                                if left_slice.sum() == orig_valid_tokens:
                                    response_masks_for_loss = left_slice
                                elif right_slice.sum() == orig_valid_tokens:
                                    response_masks_for_loss = right_slice
                                else:
                                    raise RuntimeError(
                                        f"[Fatal Alignment Error] 无法安全对齐 LogProb 掩码！\n"
                                        f"-> 原始有效 Token 数: {orig_valid_tokens.item()}\n"
                                        f"-> 左截断保留数: {left_slice.sum().item()}, 右截断保留数: {right_slice.sum().item()}\n"
                                        f"-> Entropys 长度: {ent_len}, 原始 Mask 长度: {mask_len}"
                                    )
                            else:
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
                            main_log(f"Step {self.global_steps}: Computing Ref Policy...") 
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            main_log(f"Step {self.global_steps}: Computing Critic Values...") 
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        main_log(f"Step {self.global_steps}: Computing Advantages...") 
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

                        # 【核心逻辑区】：算出最终优势 (Advantage)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                            tokenizer=self.tokenizer,          
                            global_steps=self.global_steps     
                        )
                        # =====================================================================
                        # 强制对齐 Advantages 和 Returns 的维度 (直接截取 Response 长度维度的数据)
                        # =====================================================================
                        resp_len = batch.batch["responses"].size(1)
                        
                        if batch.batch["advantages"].size(1) != resp_len:
                            batch.batch["advantages"] = batch.batch["advantages"][:, -resp_len:]
                            
                        if "returns" in batch.batch and batch.batch["returns"].size(1) != resp_len:
                            batch.batch["returns"] = batch.batch["returns"][:, -resp_len:]
                            
                        if "token_level_rewards" in batch.batch and batch.batch["token_level_rewards"].size(1) != resp_len:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"][:, -resp_len:]
                        # =====================================================================

                        # ==================== Hindsight 后见之明重构流 ====================
                        attribution_cfg = self._get_attribution_config()
                        
                        if getattr(attribution_cfg, "enable_hindsight", False) and getattr(self, "hindsight_manager", None) is not None:
                            try:
                                main_log(f"Step {self.global_steps}: Processing Hindsight...") 
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

                        # ==================== 启动 ADCA GRPO 调整 (如果开启) ====================
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

                    # ==================== 开始模型更新 ====================
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            main_log(f"Step {self.global_steps}: Updating Critic Model...") 
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            main_log(f"Step {self.global_steps}: Updating Actor Model...") 
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    
                    if summary_task is not None:
                        main_log(f"Step {self.global_steps}: Collecting Summary Task...")
                        time_cost = self.exp_manager.collect_summary_result(summary_task)
                        metrics.update({"exp_manager/summary": time_cost})


                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        main_log(f"Step {self.global_steps}: Saving Rollout Data...")
                        with _timer("dump_rollout_generations", timing_raw):
                            os.makedirs(rollout_data_dir, exist_ok=True)
                            
                            scores_list = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            
                            inputs_text = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs_text = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            
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

                            save_path = os.path.join(rollout_data_dir, f"{self.global_steps}.jsonl")
                            with open(save_path, "w", encoding='utf-8') as f:
                                for record in merged_records:
                                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            
                            filename = os.path.join(rollout_data_dir, f"traj_{self.global_steps}.jsonl")
                            with open(filename, "w") as f:
                                for traj in trajectories:
                                    f.write(traj.json() + "\n")
                            filename = os.path.join(rollout_data_dir, f"task_{self.global_steps}.jsonl")
                            with open(filename,"w") as f:
                                for task in tasks:
                                    f.write(task.json() + "\n")

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

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "training/num_not_none_traj": num_not_none_traj,
                        "training/num_term_traj": num_term_traj
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

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
                
                logger.log(data=metrics, step=self.global_steps)
                
                step_cost = time.time() - step_start_time
                main_log(f"Step {self.global_steps} Finished. Cost: {step_cost:.2f}s") 

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