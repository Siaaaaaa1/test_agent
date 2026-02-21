#!/bin/bash

# ==========================================
# 1. 环境与网络配置 (保留这些修复！)
# ==========================================

# [修复包冲突] 强制 Python 忽略用户目录(.local)
export PYTHONNOUSERSITE=1

# [修复 Ray 连接超时] 获取本机 IP 并设置不走代理
HOST_IP=$(hostname -i)
# 这一步非常关键，保留它！
export no_proxy="localhost,127.0.0.1,::1,${HOST_IP},${no_proxy}"
echo "Current Host IP: ${HOST_IP}"
echo "No Proxy Set To: ${no_proxy}"

# [vLLM/NCCL 设置]
export VLLM_LOGGING_LEVEL=INFO
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ==========================================
# 2. Conda 环境初始化
# ==========================================
if [ -z "$CONDA_EXE" ]; then
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    source "$(dirname $(dirname $CONDA_EXE))/etc/profile.d/conda.sh"
fi

# ==========================================
# 3. 启动 AppWorld 环境服务
# ==========================================
conda activate appworld
echo "Starting AppWorld Environment Service..."

pkill -f "bash env_service/launch_script/appworld.sh" || true
bash env_service/launch_script/appworld.sh > server.log 2>&1 &
SERVER_PID=$!

# 清理函数
cleanup() {
    echo "Stopping AppWorld Server (PID: $SERVER_PID)..."
    kill $SERVER_PID
    echo "Stopping Ray..."
    ray stop --force  # 脚本结束时强制清理 Ray
}
trap cleanup EXIT

echo "Waiting for server to start (PID: $SERVER_PID)..."
sleep 10

# ==========================================
# 4. 准备训练 (回退到让 Python 自动启动 Ray)
# ==========================================
conda activate agentevolver

# [关键修改]：
# 1. 确保没有残留的 Ray 进程干扰
ray stop --force 2>/dev/null
# 2. 删除 ray start --head ...
# 3. 删除 export RAY_ADDRESS ...
# 4. 取消 RAY_ADDRESS 变量，确保 Python 脚本启动本地实例
unset RAY_ADDRESS

# ==========================================
# 5. 训练参数配置
# ==========================================
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"
ENV_URL="http://127.0.0.1:8080"
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="log_${CURRENT_TIME}.log"
EXP_NAME="appworld_optimized"

echo "Starting Training..."
echo "Log file: ${LOG_FILE}"

# ==========================================
# 6. 执行训练
# ==========================================
python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    env_service.env_url=$ENV_URL \
    env_service.env_type=appworld \
    seed=1 \
    debug_log=True \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_batch_size=32 \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.train_files=null \
    data.val_files=null \
    data.max_prompt_length=4000 \
    data.max_response_length=21580 \
    data.val_batch_size=32 \
    \
    actor_rollout_ref.model.path=./models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.off_cliprange_high=0.6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.prompt_length=4000 \
    actor_rollout_ref.rollout.response_length=21580 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
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
    trainer.project_name="AgentEvolver" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.save_freq=2 \
    trainer.test_freq=5 \
    trainer.total_epochs=40 \
    trainer.val_before_train=false \
    trainer.validation_data_dir="experiments/tech_synthetic/${EXP_NAME}/validation_log" \
    trainer.rollout_data_dir="experiments/tech_synthetic/${EXP_NAME}/rollout_log" \
    \
    attribution_driven_credit_assignment.enable=false \
    attribution_driven_credit_assignment.enable_hindsight=false \
    \
    task_manager.n=256 \
    task_manager.mixture.synthetic_data_ratio=2.0 \
    task_manager.mixture.use_original_tasks=False \
    task_manager.train_data_path=./tasks_explored/tasks_explored.train.json \
    task_manager.val_data_path=./tasks_explored/tasks_explored.val.json \
    task_manager.exploration_strategy_args.a=1 \
    task_manager.exploration_strategy_args.b=4 \
    task_manager.strategy=api_driven \
    task_manager.exploration_strategy_args.active_apps="['amazon','gmail','spotify','venmo','simple_note','todoist','splitwise','phone','file_system']" \
    task_manager.exploration_strategy_args.task_labels_path="./environments/appworld/data/datasets/train.jsonl" \
    task_manager.llm_client="azure-gpt-5" \
    2>&1 | tee "$LOG_FILE"