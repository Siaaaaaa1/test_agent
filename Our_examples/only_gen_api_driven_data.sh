#!/bin/bash

# ============================================================
# AgentEvolver 数据生成专用脚本 (参数全量补齐版)
# ============================================================

# ---- 1. 基础环境配置 ----
PROJECT_ROOT="$(pwd)"
CONFIG_PATH="$PROJECT_ROOT/config"
# 数据生成输出路径
export GEN_OUTPUT_DIR="/mnt/cephfs/haowengao/test_agent/GEN_NEW_DATA"
mkdir -p "$GEN_OUTPUT_DIR"

# ---- 2. 强力清理环境 (Clean Machine) ----
echo "🧹 Nuking previous processes..."
ps -ef | grep -E "ray|vllm|agentevolver|env_service" | grep -v grep | awk '{print $2}' | xargs -r kill -9
fuser -k -9 8080/tcp >/dev/null 2>&1
fuser -k -9 6379/tcp >/dev/null 2>&1
rm -rf /tmp/ray/* 2>/dev/null
sleep 2

# ---- 3. 网络配置 ----
HOST_IP=$(hostname -I | tr ' ' '\n' | grep '^29\.' | head -n 1)
export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,$HOST_IP,.woa.com"

export PYTHONUNBUFFERED=1
export VLLM_ENFORCE_EAGER=True

# ---- 4. 启动 Local Ray Head ----
echo "🚀 Starting Local Ray Head..."
ray start --head --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats --block & 
RAY_PID=$!

echo "⏳ Waiting for Ray..."
sleep 5
for i in {1..30}; do
    if ray status > /dev/null 2>&1; then echo "✅ Ray is ready!"; break; fi
    sleep 1
done

# ---- 5. 启动 AppWorld 服务 ----
echo "🌍 Launching AppWorld Service..."
LAUNCHER_SCRIPT="./env_service/launch_script/appworld_single.sh"
chmod +x "$LAUNCHER_SCRIPT"
$LAUNCHER_SCRIPT 2>&1 | tee server.log &
SERVER_PID=$!

trap "echo '🛑 Shutting down...'; kill $SERVER_PID; kill $RAY_PID; ray stop --force" EXIT

# 健康检查
echo "⏳ Waiting for AppWorld Service..."
MAX_RETRIES=300
COUNT=0
while ! curl -s --noproxy "*" "http://localhost:8080/healthz" > /dev/null; do
    sleep 1
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then echo "❌ Timeout"; exit 1; fi
done
echo "✅ Service is UP!"

# ---- 6. 启动生成任务 (参数已完全补齐) ----
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate AgentEvolver121

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="log_gen_fullparam_${current_time}.log"
unset CUDA_VISIBLE_DEVICES

echo "🚀 Starting Data Generation with FULL Parameters..."

# 核心逻辑：
# 1. 保留 generate_task_only=true (生成模式)
# 2. 将 trainer.nnodes 强制设为 1 (单机)
# 3. 移植所有 actor_rollout_ref 及其它高级参数

python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    ray_init.address="auto" \
    env_service.env_url="http://localhost:8080" \
    env_service.env_type=appworld \
    seed=2 \
    debug_log=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=32 \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.train_files=null \
    data.val_files=null \
    data.max_prompt_length=28672 \
    data.max_response_length=4096 \
    data.val_batch_size=32 \
    actor_rollout_ref.model.path="$PROJECT_ROOT/models/Qwen2.5-7B-Instruct" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.off_cliprange_high=0.6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.prompt_length=28672 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=81920 \
    actor_rollout_ref.rollout.enable_gt_process_reward=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    critic.ppo_max_token_len_per_gpu=32768 \
    critic.forward_max_token_len_per_gpu=32768 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="AgentEvolver_Gen" \
    trainer.experiment_name="clean_machine_gen_full" \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=40 \
    trainer.val_before_train=false \
    attribution_driven_credit_assignment.enable=false \
    attribution_driven_credit_assignment.enable_hindsight=false \
    task_manager.n=32 \
    task_manager.mixture.synthetic_data_ratio=2.0 \
    task_manager.mixture.use_original_tasks=False \
    task_manager.train_data_path=${PROJECT_ROOT}/tasks_explored/tasks_explored.train.json \
    task_manager.val_data_path=${PROJECT_ROOT}/tasks_explored/tasks_explored.val.json \
    task_manager.strategy=api_driven \
    task_manager.exploration_strategy_args.a=3 \
    task_manager.exploration_strategy_args.b=8 \
    task_manager.exploration_strategy_args.active_apps="['amazon','gmail','spotify','venmo','simple_note','todoist','splitwise','phone','file_system']" \
    task_manager.exploration_strategy_args.task_labels_path="${PROJECT_ROOT}/environments/appworld/data/datasets/train.jsonl" \
    task_manager.llm_client="azure-gpt-5" \
    task_manager.grader.synthetic_grader=api_process_llm_judge \
    task_manager.env_profile=${PROJECT_ROOT}/cookbook/env_profiles/appworld.json \
    thread_pool.max_workers=10 \
    task_manager.generate_task_only=true \
    2>&1 | tee "$log_file"