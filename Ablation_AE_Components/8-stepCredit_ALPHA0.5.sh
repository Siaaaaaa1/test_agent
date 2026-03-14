#!/bin/bash

# ============================================================
# AgentEvolver 组件消融实验
#
# 文件名编码规则（大小写不敏感）：
#   noAPI → w_api=0.0    API 过程奖励关闭
#   noFMT → w_fmt=0.0    格式惩罚关闭
#   noREP → w_rep=0.0    复读惩罚关闭
#   noANN → api_reward_annealing 关闭
#   noCUR → curriculum 关闭
#   noHST → hindsight 关闭
#   无前缀 → 对应组件开启（默认权重）
#
# w_outcome 恒为 1.0（结果奖励始终保留）
# 完整系统 (FULL): w_api=0.5 w_fmt=0.5 w_rep=1.0 ANN=true CUR=true HST=true
# ============================================================

SCRIPT_NAME=$(basename "$0")
UPPER=$(echo "$SCRIPT_NAME" | tr 'a-z' 'A-Z')

# --- 自动解析组件开关 ---
[[ $UPPER == *"NOAPI"* ]] && W_API="0.0"  || W_API="0.5"
[[ $UPPER == *"NOFMT"* ]] && W_FMT="0.0"  || W_FMT="0.5"
[[ $UPPER == *"NOREP"* ]] && W_REP="0.0"  || W_REP="1.0"
[[ $UPPER == *"NOANN"* ]] && ANN="false"   || ANN="true"
[[ $UPPER == *"NOCUR"* ]] && CUR="false"   || CUR="true"
[[ $UPPER == *"NOHST"* ]] && HST="false"   || HST="true"

# step_credit_alpha：从文件名中提取 ALPHAx.x，否则默认 0.0
if [[ $UPPER == *"ALPHA"* ]]; then
    ALPHA=$(echo "$SCRIPT_NAME" | grep -i -oE 'ALPHA[0-9.]+' | sed 's/[Aa][Ll][Pp][Hh][Aa]//' | sed 's/\.$//')
else
    ALPHA="0.0"
fi

# 所有过程奖励权重均为 0 时，关闭 enable_gt_process_reward 节省计算
if [[ "$W_API" == "0.0" ]] && [[ "$W_FMT" == "0.0" ]] && [[ "$W_REP" == "0.0" ]]; then
    GT_PROCESS="false"
else
    GT_PROCESS="true"
fi

# wandb 实验名：精确反映当前配置
EXP_NAME="ae_wAPI${W_API}_wFMT${W_FMT}_wREP${W_REP}_ANN${ANN}_CUR${CUR}_HST${HST}_ALPHA${ALPHA}"

echo "=================================================="
echo "  AgentEvolver 组件消融实验"
echo "  脚本: $SCRIPT_NAME"
echo "  w_api=${W_API} | w_fmt=${W_FMT} | w_rep=${W_REP} | w_outcome=1.0"
echo "  api_annealing=${ANN} | curriculum=${CUR} | hindsight=${HST} | step_credit_alpha=${ALPHA}"
echo "  gt_process_reward=${GT_PROCESS}"
echo "  experiment_name=${EXP_NAME}"
echo "=================================================="

# ---- 1. 清理旧进程 ----
echo "Nuking previous processes..."
ps -ef | grep -E "ray|vllm|agentevolver|env_service" | grep -v grep | awk '{print $2}' | xargs -r kill -9
fuser -k -9 8080/tcp >/dev/null 2>&1
fuser -k -9 6379/tcp >/dev/null 2>&1
rm -rf /tmp/ray/* 2>/dev/null
find ./env_service/environments/appworld/experiments/outputs -type d -depth -exec rmdir {} + 2>/dev/null
sleep 2

# ---- 2. 环境变量 ----
export GEN_OUTPUT_DIR="./GEN_DATA_AE_ABLATION"
mkdir -p "$GEN_OUTPUT_DIR"

export PYTHONUNBUFFERED=1
export VLLM_ENFORCE_EAGER=True
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export DEBUG_ARG="kl_control"

# ---- 3. 启动 Ray Head ----
echo "Starting Ray Head..."
ray start --head --port=6379 --num-gpus=8 --num-cpus=32 \
    --dashboard-host=0.0.0.0 --disable-usage-stats --block &
RAY_PID=$!
sleep 5
for i in {1..60}; do
    if ray status > /dev/null 2>&1; then echo "Ray is ready!"; break; fi
    sleep 1
done

# ---- 4. 启动 AppWorld 服务 ----
LAUNCHER_SCRIPT="./env_service/launch_script/appworld_single.sh"
chmod +x "$LAUNCHER_SCRIPT"
$LAUNCHER_SCRIPT 2>&1 | tee server.log &
SERVER_PID=$!
trap "echo 'Shutting down...'; kill $SERVER_PID; kill $RAY_PID; ray stop --force" EXIT

# ---- 5. 服务健康检查 ----
while ! curl -s --noproxy "*" "http://localhost:8080/healthz" > /dev/null; do
    sleep 1
    if ! kill -0 $SERVER_PID 2>/dev/null; then exit 1; fi
done
echo "AppWorld Server is UP!"

# ---- 6. 启动训练 ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agentevolver

CONFIG_PATH="$(pwd)/config"
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="log_ae_ablation_${EXP_NAME}_${current_time}.log"
unset CUDA_VISIBLE_DEVICES

python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    ray_init.address="auto" \
    env_service.env_url="http://localhost:8080" \
    env_service.env_type=appworld \
    seed=1 \
    debug_log=True \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.process_reward_mode=sparse \
    algorithm.outcome_strategy=normalize_then_decay \
    algorithm.gamma=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.w_outcome=1.0 \
    algorithm.w_api=$W_API \
    algorithm.w_fmt=$W_FMT \
    algorithm.w_rep=$W_REP \
    algorithm.w_efficiency=0.0 \
    algorithm.step_credit_alpha=$ALPHA \
    algorithm.api_reward_annealing.enable=$ANN \
    algorithm.api_reward_annealing.convergence_threshold=0.05 \
    algorithm.api_reward_annealing.min_converged_count=3 \
    algorithm.api_reward_annealing.anneal_steps=5 \
    algorithm.curriculum.enable=$CUR \
    algorithm.curriculum.intra_success_threshold=0.5 \
    algorithm.curriculum.max_cross_ratio=4.0 \
    algorithm.curriculum.ramp_rate_per_step=0.5 \
    algorithm.curriculum.window_size=3 \
    algorithm.curriculum.rebuild_delta=0.5 \
    \
    data.train_batch_size=32 \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.train_files=null \
    data.val_files=null \
    data.max_prompt_length=4096 \
    data.max_response_length=28672 \
    data.val_batch_size=32 \
    \
    actor_rollout_ref.model.path="./models/Qwen2.5-7B-Instruct" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.003 \
    actor_rollout_ref.actor.off_cliprange_high=0.6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.prompt_length=28672 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=81920 \
    actor_rollout_ref.rollout.enable_gt_process_reward=$GT_PROCESS \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    \
    critic.ppo_max_token_len_per_gpu=32768 \
    critic.forward_max_token_len_per_gpu=32768 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="AgentEvolver_Ablation" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=40 \
    trainer.val_before_train=false \
    trainer.validation_data_dir="experiments/ae_ablation/${EXP_NAME}/validation_log" \
    trainer.rollout_data_dir="experiments/ae_ablation/${EXP_NAME}/rollout_log" \
    \
    attribution_driven_credit_assignment.enable=false \
    attribution_driven_credit_assignment.enable_hindsight=$HST \
    attribution_driven_credit_assignment.hindsight_success_rate_threshold=0.5 \
    attribution_driven_credit_assignment.hindsight_use_llm_filter=false \
    \
    task_manager.n=32 \
    task_manager.mixture.synthetic_data_ratio=10.0 \
    task_manager.mixture.use_original_tasks=False \
    task_manager.train_data_path=$GEN_OUTPUT_DIR/tasks_explored.train.json \
    task_manager.val_data_path=$GEN_OUTPUT_DIR/tasks_explored.val.json \
    task_manager.exploration_strategy_args.a=4 \
    task_manager.exploration_strategy_args.b=8 \
    task_manager.strategy=api_driven \
    task_manager.exploration_strategy_args.active_apps="['amazon','gmail','spotify','venmo','simple_note','todoist','splitwise','phone','file_system']" \
    task_manager.exploration_strategy_args.task_labels_path="./environments/appworld/data/datasets/train.txt" \
    task_manager.llm_client="qwen3.5-plus" \
    task_manager.grader.synthetic_grader=api_process_llm_judge \
    task_manager.env_profile=./cookbook/env_profiles/appworld.json \
    thread_pool.max_workers=12 \
    task_manager.generate_task_only=false \
    2>&1 | tee "$log_file"
