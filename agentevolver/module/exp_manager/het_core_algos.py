import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    聚合损失矩阵为标量。
    
    将每个 token 的损失值根据指定的聚合模式（如平均、求和等）聚合成一个最终的 loss 标量，用于反向传播。

    参数 (Args):
        loss_mat (torch.Tensor): 
            形状: (batch_size, response_length)
            含义: 原始的损失矩阵，每个位置对应一个 token 的损失值。
        loss_mask (torch.Tensor): 
            形状: (batch_size, response_length)
            含义: 损失掩码。1 表示该位置是有效的 response token，0 表示 padding 或 prompt 部分（不计算 loss）。
        loss_agg_mode (str): 
            含义: 聚合模式的名称。
            可选值:
                - 'token-mean': 所有有效 token 的损失求平均。
                - 'seq-mean-token-sum': 先对每个序列(sequence)内的 token 损失求和，再对所有序列求平均。
                - 'seq-mean-token-mean': 先对每个序列内的 token 损失求平均，再对所有序列求平均。
                - 'seq-mean-token-sum-norm': 类似于 'seq-mean-token-sum'，但归一化因子固定为 mask 的长度（参考 DrGRPO 论文）。

    返回 (Returns):
        loss (torch.Tensor): 
            形状: 标量 (scalar)
            含义: 聚合后的最终损失值。
    """
    if loss_agg_mode == "token-mean":
        # 计算所有有效 token loss 的平均值
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        # dim=-1 求和得到每个 sequence 的总 loss，然后对 batch 求平均
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        # 先计算每个 sequence 内部的平均 loss，再对 batch 求平均
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        # 针对 DrGRPO 的特殊归一化方式
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        # 注意：这里直接除以了 mask 的最后一维长度（response_length），假定它是常数
        loss = torch.sum(seq_losses) / loss_mask.shape[-1] 
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def het_compute_token_on_off_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    off_cliprange_high=1.0,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    计算异构（Heterogeneous）PPO 损失，同时处理 On-policy 和 Off-policy 数据。

    该函数区分数据来源（当前策略产生 vs 经验回放产生），并允许对不同来源的数据应用不同的 PPO 裁剪（Clip）参数。

    参数 (Args):
        old_log_prob (torch.Tensor): 
            形状: (bs, response_len)
            含义: 采样时（旧策略）动作的对数概率。
        log_prob (torch.Tensor): 
            形状: (bs, response_len)
            含义: 当前训练模型（新策略）动作的对数概率。
        advantages (torch.Tensor): 
            形状: (bs, response_len)
            含义: 优势函数值 (Advantage)，用于衡量动作的好坏。
        response_mask (torch.Tensor): 
            形状: (bs, response_len)
            含义: 标识哪些 token 是模型生成的回复（response），只有这些部分参与 Loss 计算。
        exp_mask (torch.Tensor): 
            形状: (bs, response_len)
            含义: **经验来源掩码**。1 表示数据来自 Off-policy（如经验池），0 表示数据来自 On-policy（当前策略）。
        cliprange (float, optional): PPO 默认裁剪范围（如 0.2）。
        cliprange_low (float, optional): PPO 裁剪下界（1 - cliprange_low）。若未指定则使用 cliprange。
        cliprange_high (float, optional): PPO 裁剪上界（1 + cliprange_high）。On-policy 数据使用此值。
        off_cliprange_high (float, optional): **Off-policy 数据的裁剪上界**。通常设置得比 On-policy 更大（如 1.0），允许 Off-policy 数据有更大的更新幅度。
        clip_ratio_c (float, optional): 裁剪系数 C，用于处理 loss 下界的保护逻辑（防止 loss 过小）。
        loss_agg_mode (str, optional): 损失聚合模式，默认为 "token-mean"。

    返回 (Returns):
        dict: 包含以下键值的字典：
            - "pg_loss": 聚合后的总策略梯度损失（标量）。
            - "pg_losses": 每个 token 的损失矩阵。
            - "on_pg_loss": On-policy 部分的平均损失。
            - "off_pg_loss": Off-policy 部分的平均损失。
            - "ppo_kl": PPO 过程中的 KL 散度近似值。
            - 其他裁剪比例统计指标 (clipfrac)。
    """
    # 计算 log(new/old) = log(new) - log(old)
    negative_approx_kl = log_prob - old_log_prob
    # 计算 KL 散度用于监控
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # 计算概率比率 r_t(\theta) = exp(log_prob - old_log_prob)
    ratio = torch.exp(negative_approx_kl)

    # 内部辅助函数：计算标准的 PPO Clip Loss
    def compute_pg_losses(cliprange_low, cliprange_high):
        # 1. 未裁剪的损失: -A * ratio
        pg_losses1 = -advantages * ratio
        # 2. 裁剪后的损失: -A * clip(ratio, 1-low, 1+high)
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
        # PPO Loss 取 max (因为是取负号后的 max，相当于原公式的 min)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        
        # 3. 下界保护逻辑 (针对 extremely bad updates)
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        
        # 如果 Advantage < 0，使用带下界保护的 loss，否则使用标准 PPO loss
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        
        # 计算裁剪发生的比例 (用于监控)
        clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
        clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)
        return pg_losses, clipfrac, clipfrac_lower

    # ================= On-policy 计算 =================
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    # 使用标准的 cliprange_high 计算
    on_pg_losses, on_pg_clipfrac, on_pg_clipfrac_lower = compute_pg_losses(cliprange_low, cliprange_high)
    # 仅保留 exp_mask == 0 的部分 (On-policy)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)

    # ================= Off-policy 计算 =================
    off_cliprange_low = cliprange_low
    # 使用专门的 off_cliprange_high (通常更大) 计算
    off_pg_losses, off_pg_clipfrac, off_pg_clipfrac_lower = compute_pg_losses(off_cliprange_low, off_cliprange_high)
    # 仅保留 exp_mask == 1 的部分 (Off-policy)
    off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
    off_pg_loss = torch.tensor(0.0) if off_pg_loss.isnan().item() else off_pg_loss

    # ================= 组合 Loss =================
    exp_mask = exp_mask.float()
    # 根据 mask 组合两种 loss
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
    # 聚合
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses": on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
    }


def bam_compute_token_on_off_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Bam 版本的混合策略损失计算 (V1)。

    特点：
    1. On-policy 部分：使用标准的 PPO 逻辑。
    2. Off-policy 部分：**不使用 PPO 裁剪**，而是使用一种特殊的比率变换 `p / (p + 0.1)`，
       这旨在稳定 Off-policy 的更新幅度，避免比率过大。

    参数 (Args):
        (同上 `het_compute_token_on_off_policy_loss`)
        exp_mask: 1 表示 Off-policy 数据，0 表示 On-policy 数据。

    返回 (Returns):
        dict: 包含计算后的各项损失指标。
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # ---------------- On-policy 处理 (标准 PPO) ----------------
    ratio = torch.exp(negative_approx_kl)   # (bs, response_length)
    on_pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    # 裁剪
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    on_clip_pg_losses1 = torch.maximum(on_pg_losses1, on_pg_losses2)
    on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses1).float(), response_mask)

    # 下界保护
    on_pg_losses3 = -advantages * clip_ratio_c
    on_clip_pg_losses2 = torch.min(on_pg_losses3, on_clip_pg_losses1)
    on_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(on_clip_pg_losses1, on_pg_losses3) * (advantages < 0).float(), response_mask)

    # 最终 On-policy loss
    on_pg_losses = torch.where(advantages < 0, on_clip_pg_losses2, on_clip_pg_losses1)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0-exp_mask) * response_mask)

    # ---------------- Off-policy 处理 (特殊比率) ----------------
    # 获取新策略的概率值 exp(log_prob)
    off_ratio = torch.exp(log_prob)     #(bs, response_length)
    # ⭐ 核心差异：重塑比率 R = P / (P + 0.1)。
    # 这个变换将输出值限制在 [0, 1) 之间，使得 Off-policy 的 loss 不会因为概率极低或极高而爆炸。
    off_ratio = off_ratio / (off_ratio + 0.1)
    off_pg_losses = -advantages * off_ratio
    off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)

    # ---------------- 聚合 ----------------
    exp_mask = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    ret_dict = {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses":  on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
    }

    return ret_dict

def bam_compute_token_on_off_policy_loss_v2(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,   # (bs, response_length)
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Bam 版本的混合策略损失计算 (V2)。

    特点：
    1. On-policy 部分：同 V1，标准 PPO。
    2. Off-policy 部分：同 V1 使用 `p/(p+0.1)` 变换，但**增加了正样本筛选**。
       只计算 Advantage >= 0 的 Off-policy 数据，忽略负样本。这是一种类似于 "只学习好的经验" 的策略。

    参数与返回同 V1。
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # ---------------- On-policy 处理 ----------------
    ratio = torch.exp(negative_approx_kl)   # (bs, response_length)
    on_pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high) 
    on_clip_pg_losses1 = torch.maximum(on_pg_losses1, on_pg_losses2) 
    on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses1).float(), response_mask)

    on_pg_losses3 = -advantages * clip_ratio_c
    on_clip_pg_losses2 = torch.min(on_pg_losses3, on_clip_pg_losses1)
    on_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(on_clip_pg_losses1, on_pg_losses3) * (advantages < 0).float(), response_mask)

    on_pg_losses = torch.where(advantages < 0, on_clip_pg_losses2, on_clip_pg_losses1)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0-exp_mask) * response_mask)

    # ---------------- Off-policy 处理 (筛选正样本) ----------------
    off_ratio = torch.exp(log_prob)     #(bs, response_length)
    off_ratio = off_ratio / (off_ratio + 0.1)   # 同样的数值稳定变换
    off_pg_losses = -advantages * off_ratio
    
    # ⭐ 核心差异：过滤逻辑
    # off_positive_mask 选中：是 Off-policy 数据 (exp_mask>0) 且 Advantage >= 0 且是有效 token
    off_positive_mask = (exp_mask > 0) & (advantages >=0) & (response_mask > 0) 
    
    # 将不满足条件的 Off-policy 样本 loss 置为 0
    adjusted_off_pg_losses = torch.where(off_positive_mask, off_pg_losses, torch.zeros_like(off_pg_losses))
    off_pg_loss = verl_F.masked_mean(off_pg_losses, off_positive_mask)
    if torch.isnan(off_pg_loss).item():
        off_pg_loss = torch.tensor(0.0)

    # ---------------- 聚合 ----------------
    exp_mask = exp_mask.float()
    # 使用筛选过的 adjusted_off_pg_losses
    pg_losses = adjusted_off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    ret_dict = {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses":  on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
    }

    return ret_dict

def bam_compute_token_on_off_policy_loss_v3(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,   # (bs, response_length)
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Bam 版本的混合策略损失计算 (V3)。

    特点：
    1. On-policy 部分：同前，标准 PPO。
    2. Off-policy 部分：**回归使用 PPO 的 Clip 逻辑**，但强制将裁剪上界 (`cliprange_high`) 设为 1.0。
       这意味着 Off-policy 数据在 Advantage > 0 时允许最大 2 倍 (1+1.0) 的概率比率更新，相比标准 PPO 通常更宽松。

    参数与返回同 V1。
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # ---------------- On-policy 处理 ----------------
    ratio = torch.exp(negative_approx_kl)   # (bs, response_length)
    on_pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    on_clip_pg_losses1 = torch.maximum(on_pg_losses1, on_pg_losses2)
    on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses1).float(), response_mask)

    on_pg_losses3 = -advantages * clip_ratio_c
    on_clip_pg_losses2 = torch.min(on_pg_losses3, on_clip_pg_losses1)
    on_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(on_clip_pg_losses1, on_pg_losses3) * (advantages < 0).float(), response_mask)

    on_pg_losses = torch.where(advantages < 0, on_clip_pg_losses2, on_clip_pg_losses1)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0-exp_mask) * response_mask)

    # ---------------- Off-policy 处理 (固定 Clip 上界) ----------------
    off_pg_losses1 = -advantages * ratio
    off_cliprange_low = cliprange_low
    off_cliprange_high = 1.0  # ⭐ 核心差异：强制 Off-policy 数据的 clip 上界为 1.0
    
    off_pg_losses2 = -advantages * torch.clamp(ratio, 1 - off_cliprange_low, 1 + off_cliprange_high)
    off_clip_pg_losses1 = torch.maximum(off_pg_losses1, off_pg_losses2)
    off_pg_clipfrac = verl_F.masked_mean(torch.gt(off_pg_losses2, off_pg_losses1).float(), response_mask)
    
    off_pg_losses3 = -advantages * clip_ratio_c
    off_clip_pg_losses2 = torch.min(off_pg_losses3, off_clip_pg_losses1)
    off_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(off_clip_pg_losses1, off_pg_losses3) * (advantages < 0).float(), response_mask)

    off_pg_losses = torch.where(advantages < 0, off_clip_pg_losses2, off_clip_pg_losses1)
    off_pg_loss = verl_F.masked_mean(off_pg_losses, (1.0-exp_mask) * response_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)

    # ---------------- 聚合 ----------------
    exp_mask = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    ret_dict = {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses":  on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
    }

    return ret_dict