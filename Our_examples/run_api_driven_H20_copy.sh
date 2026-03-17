#!/bin/bash

# ---- 1. 环境准备与 Localhost 设置 ----
# [修改] 强制指定 IP 为本地回环地址
LOCAL_IP="127.0.0.1"
env_url="http://${LOCAL_IP}:8080"
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"

# 获取本机 IP (Linux通用)
HOST_IP=$(hostname -I | awk '{print $1}')

# 将 HOST_IP 也加入 no_proxy，否则 Ray 内部通信会被代理拦截导致报错
export no_proxy="localhost,127.0.0.1,::1,.woa.com,$HOST_IP,$no_proxy"
export NO_PROXY="localhost,127.0.0.1,::1,.woa.com,$HOST_IP,$NO_PROXY"

# ---- 2. 启动环境服务 (AppWorld) ----
echo "Starting AppWorld Environment Service on $env_url (Localhost)..."

# 在子 Shell 中激活环境并启动，避免污染主脚本环境
(
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate appworld
    # 注意：这里调用的脚本内部也已被修改为监听 Localhost
    bash env_service/launch_script/appworld.sh > server.log 2>&1
) &
SERVER_PID=$!

# 注册退出陷阱：脚本结束时杀掉后台服务
trap "echo 'Shutting down server...'; kill $SERVER_PID" EXIT

# ---- 3. 健康检查：等待服务真正可用 ----
echo "Waiting for server to be ready at $env_url..."
MAX_RETRIES=30
COUNT=0
while ! curl -s "$env_url/get_env_profile" > /dev/null; do
    sleep 2
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "Error: Environment service failed to start. Last 10 lines of server.log:"
        tail -n 10 server.log
        exit 1
    fi
    echo "Still waiting... ($((COUNT*2))s)"
done
echo "Server is UP!"

# ---- 4. 启动训练 (AgentEvolver) ----
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate agentevolver

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="log_${current_time}.log"

# Ray / CUDA 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_NUM_CPUS=64

echo "Starting Training with IP: $LOCAL_IP"

python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    env_service.env_url=$env_url \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
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
    trainer.experiment_name="appworld_optimized" \
    trainer.save_freq=2 \
    trainer.test_freq=5 \
    trainer.total_epochs=40 \
    trainer.val_before_train=false \
    trainer.validation_data_dir="experiments/tech_synthetic/appworld_optimized/validation_log" \
    trainer.rollout_data_dir="experiments/tech_synthetic/appworld_optimized/rollout_log" \
    \
    attribution_driven_credit_assignment.enable=false \
    attribution_driven_credit_assignment.enable_hindsight=false \
    \
    task_manager.n=32 \
    task_manager.mixture.synthetic_data_ratio=2.0 \
    task_manager.mixture.use_original_tasks=False \
    task_manager.train_data_path=./tasks_explored/tasks_explored.train.json \
    task_manager.val_data_path=./tasks_explored/tasks_explored.val.json \
    task_manager.exploration_strategy_args.a=1 \
    task_manager.exploration_strategy_args.b=4 \
    task_manager.strategy=api_driven \
    task_manager.exploration_strategy_args.active_apps="['amazon','gmail','spotify','venmo','simple_note','todoist','splitwise','phone','file_system']" \
    task_manager.exploration_strategy_args.task_labels_path="./env_service/environments/appworld/data/datasets/train.txt" \
    task_manager.llm_client="qwen3.5-plus" \
    task_manager.grader.synthetic_grader=api_process_llm_judge \
    ray_init.num_cpus=32 \
    2>&1 | tee "$log_file"