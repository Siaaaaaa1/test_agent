#!/bin/bash

# ---- 1. 环境准备 ----
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"

export VLLM_ATTENTION_BACKEND=TORCH_SDPA
# export CUDA_LAUNCH_BLOCKING=1
export VLLM_USE_V1=1

# 获取本机 IP (Master IP)，用于让副节点访问环境
HOST_IP="29.209.104.6"
env_url="http://${HOST_IP}:8080"

# 确保代理设置透传
# export no_proxy="localhost,127.0.0.1,::1,.woa.com,$HOST_IP,29.0.0.0/8,$no_proxy"
# export NO_PROXY="localhost,127.0.0.1,::1,.woa.com,$HOST_IP,29.0.0.0/8,$NO_PROXY"
export no_proxy="localhost,127.0.0.1,::1,29.209.104.6,29.0.0.0/8,10.0.0.0/8,.woa.com"
export NO_PROXY="localhost,127.0.0.1,::1,29.209.104.6,29.0.0.0/8,10.0.0.0/8,.woa.com"
PROJECT_ROOT="$(pwd)"
# ---- 2. 启动训练 (AgentEvolver) ----
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate agentevolver

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="log_multi_${current_time}.log"

# 重要：移除 CUDA_VISIBLE_DEVICES 限制，让 Ray 看到所有卡
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

echo "Starting Distributed Training on Cluster..."
echo "Master IP (Env URL): $env_url"
echo "Log file: $log_file"

python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    \
    ray_init.address="auto" \
    \
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
    data.max_prompt_length=20480 \
    data.max_response_length=2048 \
    data.val_batch_size=32 \
    \
    actor_rollout_ref.model.path="${PROJECT_ROOT}/models/Qwen2.5-7B-Instruct" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.prompt_length=20480 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=81920 \
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
    trainer.experiment_name="appworld_distributed" \
    trainer.save_freq=2 \
    trainer.test_freq=5 \
    trainer.total_epochs=40 \
    trainer.val_before_train=false \
    trainer.validation_data_dir="${PROJECT_ROOT}/experiments/tech_synthetic/appworld_optimized/validation_log" \
    trainer.rollout_data_dir="${PROJECT_ROOT}/experiments/tech_synthetic/appworld_optimized/rollout_log" \
    \
    attribution_driven_credit_assignment.enable=false \
    attribution_driven_credit_assignment.enable_hindsight=false \
    \
    task_manager.n=32 \
    task_manager.mixture.synthetic_data_ratio=2.0 \
    task_manager.mixture.use_original_tasks=False \
    task_manager.train_data_path=${PROJECT_ROOT}/tasks_explored/tasks_explored.train.json \
    task_manager.val_data_path=${PROJECT_ROOT}/tasks_explored/tasks_explored.val.json \
    task_manager.exploration_strategy_args.a=1 \
    task_manager.exploration_strategy_args.b=4 \
    task_manager.strategy=api_driven \
    task_manager.exploration_strategy_args.active_apps="['amazon','gmail','spotify','venmo','simple_note','todoist','splitwise','phone','file_system']" \
    task_manager.exploration_strategy_args.task_labels_path="${PROJECT_ROOT}/environments/appworld/data/datasets/train.jsonl" \
    task_manager.llm_client="azure-gpt-5" \
    task_manager.grader.synthetic_grader=api_process_llm_judge \
    task_manager.env_profile=${PROJECT_ROOT}/cookbook/env_profiles/appworld.json \
    2>&1 | tee "$log_file"