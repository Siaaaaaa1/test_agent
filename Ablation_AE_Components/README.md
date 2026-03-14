# AgentEvolver 消融实验说明

> 实验路径：`Ablation_AE_Components/`
> WandB 项目：`AgentEvolver_Ablation`
> 数据目录：`GEN_DATA_AE_ABLATION/`
> 实验日志：`experiments/ae_ablation/{EXP_NAME}/`

---

## 一、消融实验列表

每个脚本的文件名即为配置的完整描述，训练脚本在启动时自动解析文件名并设置对应超参数，无需手动修改。

| 编号 | 脚本 | 关闭的组件 | WandB 实验名（自动生成） |
|------|------|-----------|------------------------|
| 0 | `0-full.sh` | 无（完整系统） | `ae_wAPI0.5_wFMT0.5_wREP1.0_ANNtrue_CURtrue_HSTtrue_ALPHA0.0` |
| 1 | `1-noHST.sh` | Hindsight 事后任务重标注 | `...HSTfalse...` |
| 2 | `2-noCUR.sh` | Curriculum 课程学习 | `...CURfalse...` |
| 3 | `3-noANN.sh` | API 奖励收敛退火 | `...ANNfalse...` |
| 4 | `4-noAPI_noANN.sh` | API 过程奖励 + 退火（均关闭） | `...wAPI0.0...ANNfalse...` |
| 5 | `5-noFMT.sh` | 格式惩罚 | `...wFMT0.0...` |
| 6 | `6-noREP.sh` | 复读惩罚 | `...wREP0.0...` |
| 7 | `7-baseline_noAPI_noFMT_noREP_noANN_noCUR_noHST.sh` | 所有组件（纯结果奖励基线） | `...wAPI0.0_wFMT0.0_wREP0.0_ANNfalse_CURfalse_HSTfalse...` |
| 8 | `8-stepCredit_ALPHA0.5.sh` | 无（完整系统 + 步骤级信用重分配） | `...ALPHA0.5` |

### 各实验具体配置对照

```
                    w_api  w_fmt  w_rep  ANN    CUR    HST    ALPHA
0-full              0.5    0.5    1.0    true   true   true   0.0
1-noHST             0.5    0.5    1.0    true   true   false  0.0
2-noCUR             0.5    0.5    1.0    true   false  true   0.0
3-noANN             0.5    0.5    1.0    false  true   true   0.0
4-noAPI_noANN       0.0    0.5    1.0    false  true   true   0.0
5-noFMT             0.5    0.0    1.0    true   true   true   0.0
6-noREP             0.5    0.5    0.0    true   true   true   0.0
7-baseline          0.0    0.0    0.0    false  false  false  0.0
8-stepCredit        0.5    0.5    1.0    true   true   true   0.5
```

> `w_outcome = 1.0` 在所有实验中恒为固定值，结果奖励始终保留。
> 实验 4 同时关闭 API 奖励和退火，因为退火以 API reward std 为监控信号，关闭 API 奖励后退火无意义。

---

## 二、核心组件与方法创新

### 2.1 多流过程奖励（Multi-stream Process Reward）

**背景**：传统 RLHF/GRPO 仅使用单一结果奖励（任务是否完成），无法对智能体的中间行为提供细粒度反馈。

**方法**：将奖励信号分解为四条独立流，每条流有独立权重，加权求和后参与 GRPO 优势计算：

| 奖励流 | 权重 | 含义 |
|--------|------|------|
| `outcome` | `w_outcome=1.0` | 任务最终结果（成功/失败），主导信号 |
| `api` | `w_api=0.5` | 步骤级 API 命中奖励：该步是否调用了 GT 参考路径中的 API |
| `fmt` | `w_fmt=0.5` | 格式惩罚：每步是否恰好包含一个 ` ```python...``` ` 代码块 |
| `rep` | `w_rep=1.0` | 复读惩罚：检测步骤间重复调用，负值惩罚无效探索 |

**实现**：四路奖励在 rollout 阶段由 `ApiProcessRewardCalculator` 实时计算，并通过 `CmtLinear` 上下文管理器与轨迹步骤对齐存储，最终在 `parse_reward_from_dataproto()` 中组装为 token 级奖励张量。

---

### 2.2 稀疏 GRPO + 步骤信用重分配（Sparse GRPO + Step Credit Alpha）

**背景**：Agent 任务轨迹较长（最多 30 步），稀疏 GRPO 将整条轨迹的奖励广播到所有 token，不区分步骤贡献。

**两种模式**（本消融的核心对比）：

- **Sparse（`ALPHA=0.0`，实验 0）**：过程奖励求和 → GRPO 组内归一化 → 广播至整条轨迹的所有 token。每条轨迹的每个 token 使用相同的优势值。
- **Step Credit（`ALPHA=0.5`，实验 8）**：在 sparse GRPO 归一化后，额外叠加步骤级信用重分配。`alpha` 控制插值强度：
  ```
  adv_final = (1 - alpha) * adv_grpo + alpha * adv_step_local
  ```
  步骤靠后的 token 携带更多本步骤的局部信号，实现更细粒度的功劳归属。

**参数**：`algorithm.step_credit_alpha`，范围 `[0.0, 1.0]`，`0.0` 退化为纯 sparse GRPO。

---

### 2.3 API 奖励收敛退火（API Reward Convergence Annealing）

**背景**：API 过程奖励在训练初期能有效引导模型学习规范 API 调用，但当模型已收敛到稳定的 API 调用模式后，组内 API reward 方差趋近于零，此时继续保持 `w_api` 只会引入冗余信号，反而可能干扰结果奖励的优化。

**方法**：
1. 每个训练步，对 batch 内所有同 UID 的 rollout 组计算组内 API reward 标准差
2. 若所有组的平均 std 连续 `min_converged_count`（默认 3）步低于 `convergence_threshold`（默认 0.05），判定 API 奖励已收敛
3. 触发线性退火：在 `anneal_steps`（默认 5）步内将 `w_api` 从当前值线性衰减至 0

```yaml
api_reward_annealing:
  enable: true
  convergence_threshold: 0.05   # 组内 std 低于此值视为收敛
  min_converged_count: 3        # 连续满足条件的步数阈值
  anneal_steps: 5               # 线性退火步数
```

**注意**：退火逻辑基于「连续收敛计数」而非固定步数触发，避免偶发噪声导致的误判。

---

### 2.4 课程学习：由 Intra 到 Cross 的渐进式任务引入（Curriculum Learning）

**背景**：跨 App 任务（cross）比单 App 任务（intra）复杂度高出数倍，训练初期直接混合 cross 任务会导致学习信号稀疏、梯度噪声大。

**方法**：
1. 在 `domain_type` 字段（`intra`/`cross`）标记每条训练样本的类型，通过 adapter 传入 batch extras
2. 使用滑动窗口（`window_size=3`）追踪近几步的 intra 任务成功率
3. 当滑动窗口均值超过 `intra_success_threshold`（默认 0.5）时，以 `ramp_rate_per_step`（默认 0.5）的速率逐步提升 cross 任务比例
4. cross 比例上限为 `max_cross_ratio`（默认 4.0），即 cross:intra = 4:1
5. 当 cross 比例累计变化超过 `rebuild_delta`（默认 0.5）时，重建 DataLoader 生效

```yaml
curriculum:
  enable: true
  intra_success_threshold: 0.5
  max_cross_ratio: 4.0
  ramp_rate_per_step: 0.5
  window_size: 3
  rebuild_delta: 0.5
```

---

### 2.5 事后任务重标注（Hindsight Task Relabeling）

**背景**：当模型在某个任务上全组（n=8）均失败时，这批 rollout 数据的梯度信号极为稀疏（GRPO 组内方差趋零）。但这些失败轨迹中，模型实际上可能完成了某些子目标——这些隐含的成功可以被重新挖掘为新的训练样本。

**方法**：
1. **触发条件**：同 UID 组的成功率 `< hindsight_success_rate_threshold`（默认 0.5），即超过一半轨迹失败时才处理
2. **轨迹选择**：从符合条件的组中随机选取一条轨迹（避免同质化）
3. **任务重标注**：将选中轨迹送入 LLM，让其从轨迹中提取模型实际完成的子任务描述
4. **质量过滤**：
   - `NaiveTaskPostFilter`：基于置信度排序 + IoU 去重，过滤低质量或重复任务
   - `LlmFilter`（可选，`hindsight_use_llm_filter=true`）：将生成任务在真实环境中执行验证，保留可完成的任务
5. **数据注入**：通过 `hindsight_save_path` 保存，下一个 epoch 开始时混入训练数据

```yaml
attribution_driven_credit_assignment:
  enable_hindsight: true
  hindsight_save_path: "tasks_explored/hindsight_supplement.jsonl"
  hindsight_success_rate_threshold: 0.5
  hindsight_use_llm_filter: false
```

---

### 2.6 格式惩罚（Format Penalty）

**背景**：AppWorld 任务要求智能体以规范的 Python 代码块格式调用 API，格式不规范（缺少代码块、多个代码块混杂）会导致环境解析失败，产生无效步骤。

**方法**：每个步骤检查 LLM 输出中是否恰好包含**一个** ` ```python...``` ` 代码块：
- 恰好 1 个 → 格式奖励 = 0.0（不惩罚）
- 0 个或 >1 个 → 格式奖励 = `format_penalty`（默认 -1.0）

格式奖励附加在步骤末尾 token 上，通过 `w_fmt` 权重参与优势计算。

---

### 2.7 复读惩罚（Repetition Penalty）

**背景**：长轨迹中模型容易陷入重复调用同一 API 的局部最优，既无法推进任务，又消耗宝贵的上下文 token 配额。

**方法**：检测当前步骤与近期步骤是否调用了相同的 API（包括参数），若检测到重复调用则施加负奖励。`w_rep` 控制惩罚强度，惩罚强度已在 `ApiProcessRewardCalculator` 内部归一化，外部权重保持为正数即可。

---

## 三、系统整体架构

```
训练循环 (ae_ray_trainer.py)
│
├── [数据层] TaskManager
│   ├── 原始 AppWorld 任务 (intra/cross 标记)
│   ├── 动态合成数据 (api_driven 探索策略)
│   └── Hindsight 补充数据 (tasks_explored/hindsight_supplement.jsonl)
│       └── CurriculumMixtureStrategy (动态调整 intra/cross 比例)
│
├── [Rollout 层] MultiTurnRollout (vLLM async)
│   └── ApiProcessRewardCalculator (每步实时计算 api/fmt/rep 奖励)
│       └── CmtLinear (上下文管理 + 步骤奖励对齐存储)
│
├── [奖励解析] parse_reward_from_dataproto()
│   └── 将 step_api_rewards / step_format_rewards / step_repetition_rewards
│       映射到 token 级奖励张量 (api_reward_tensor, fmt_reward_tensor, rep_reward_tensor)
│
├── [优势计算] compute_advantage()
│   ├── 多流奖励加权: pre_norm_adv = w_out*adv_out + w_api*adv_api + w_fmt*adv_fmt + w_rep*adv_rep
│   ├── GRPO 组内归一化 (norm_adv_by_std_in_grpo)
│   └── 步骤信用重分配 (step_credit_alpha)
│
├── [动态权重] _compute_api_annealing_weights()
│   └── 监控组内 API reward std → 触发 w_api 线性退火
│
├── [课程更新] _update_curriculum()
│   └── 追踪 intra 成功率 → 动态调整 cross 任务比例 → 按需重建 DataLoader
│
└── [Hindsight] HindsightManager.process_failed_batch()
    └── 低成功率组 → 随机选轨迹 → LLM 重标注 → NaiveTaskPostFilter → 写入 JSONL
```

---

## 四、实验目的与预期

| 对比组 | 研究问题 | 预期 |
|--------|---------|------|
| `0-full` vs `7-baseline` | 所有 AE 组件的整体增益 | Full >> Baseline |
| `0-full` vs `1-noHST` | Hindsight 的贡献 | 全组失败样本得到利用，低资源场景收益显著 |
| `0-full` vs `2-noCUR` | 课程学习的贡献 | 有课程学习时 cross 任务表现更好，收敛更稳定 |
| `0-full` vs `3-noANN` | API 退火的贡献 | 退火后 outcome 优化不受 API 信号干扰，后期表现更好 |
| `0-full` vs `4-noAPI_noANN` | API 过程奖励整体的贡献 | API 奖励在训练早期提供有效的 dense 信号 |
| `0-full` vs `5-noFMT` | 格式惩罚的贡献 | 减少无效步骤，提升样本利用率 |
| `0-full` vs `6-noREP` | 复读惩罚的贡献 | 减少轨迹中的重复调用，提升探索效率 |
| `0-full` vs `8-stepCredit` | 步骤信用重分配的贡献 | ALPHA=0.5 是否比 ALPHA=0.0 提供更细粒度的训练信号 |
