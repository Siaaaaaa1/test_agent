#!/bin/bash

# ============================================================
# AgentEvolver 单机调试版 V5 (模块化调用版)
# ============================================================

# ---- 1. 强力清理 ----
echo "🧹 Nuking previous processes..."
ps -ef | grep -E "ray|vllm|agentevolver|env_service" | grep -v grep | awk '{print $2}' | xargs -r kill -9
# 清理端口
fuser -k -9 8080/tcp >/dev/null 2>&1
fuser -k -9 6379/tcp >/dev/null 2>&1
rm -rf /tmp/ray/* 2>/dev/null

sleep 2

# ---- 2. 环境变量 ----
HOST_IP=$(hostname -I | awk '{print $1}')
export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113
# 确保 localhost 和 127.0.0.1 都在 no_proxy 中
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,29.209.112.175,.woa.com"
export NO_PROXY=$no_proxy

# 关键：强制 Python 实时打印，不要缓存日志！
export PYTHONUNBUFFERED=1
# 关键：强制 vLLM 稳定模式
export VLLM_ENFORCE_EAGER=True
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export NCCL_P2P_DISABLE=1
# export OMP_NUM_THREADS=1

# ---- 3. 启动 Ray Head ----
echo "🚀 Starting Local Ray Head..."
# 限制内存使用，防止 OOM
ray start --head --port=6379 --num-gpus=8 --num-cpus=32 --dashboard-host=0.0.0.0 --disable-usage-stats --block & 
RAY_PID=$!

echo "⏳ Waiting for Ray..."
sleep 5
# 循环检查 Ray
for i in {1..60}; do
    if ray status > /dev/null 2>&1; then
        echo "✅ Ray is ready!"
        break
    fi
    sleep 1
    echo -ne "   Ray checking... ${i}s\r"
done

# ---- 4. 启动 AppWorld 服务 (调用独立脚本) ----
echo "🌍 Launching AppWorld Service Script..."

# 定义脚本路径
LAUNCHER_SCRIPT="./env_service/launch_script/appworld_single.sh"

# 赋予执行权限
chmod +x "$LAUNCHER_SCRIPT"

# [核心修改] 调用独立脚本，并将日志同时输出到屏幕和文件
# 使用 & 让其后台运行
$LAUNCHER_SCRIPT 2>&1 | tee server.log &

SERVER_PID=$!

# 退出陷阱 (杀掉脚本进程 = 杀掉 exec 启动的 Python)
trap "echo '🛑 Shutting down...'; kill $SERVER_PID; kill $RAY_PID; ray stop --force" EXIT

# ---- 5. 健康检查 (Localhost + NoProxy) ----
echo "⏳ Waiting for AppWorld (Port 8080)..."
MAX_RETRIES=300
COUNT=0

# 使用 localhost 检查
while ! curl -s --noproxy "*" "http://localhost:8080/healthz" > /dev/null; do
    sleep 1
    COUNT=$((COUNT+1))
    
    # 检查进程是否死了
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ AppWorld process died! Check logs above."
        exit 1
    fi

    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Timeout. AppWorld connected to Ray but hangs on initialization."
        exit 1
    fi
    echo -ne "   Waiting... ${COUNT}s\r"
done

echo -e "\n✅ Server is UP!"

# ---- 6. 启动训练 ----
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate AgentEvolver121

CONFIG_PATH="$(pwd)/config"
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="log_single_${current_time}.log"
unset CUDA_VISIBLE_DEVICES

echo "🚀 Starting Training..."
# 使用 localhost 连接
python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    ray_init.address="auto" \
    env_service.env_url="http://localhost:8080" \
    env_service.env_type=appworld \
    seed=1 \
    debug_log=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=32 \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.train_files=null \
    data.val_files=null \
    data.max_prompt_length=4096 \
    data.max_response_length=20480 \
    data.val_batch_size=32 \
    actor_rollout_ref.model.path="./models/Qwen2.5-7B-Instruct" \
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
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.response_length=20480 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=81920 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    critic.ppo_max_token_len_per_gpu=32768 \
    critic.forward_max_token_len_per_gpu=32768 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="AgentEvolver" \
    trainer.experiment_name="appworld_single" \
    trainer.save_freq=2 \
    trainer.test_freq=5 \
    trainer.total_epochs=40 \
    trainer.val_before_train=false \
    trainer.validation_data_dir="experiments/tech_synthetic/appworld_optimized/validation_log" \
    trainer.rollout_data_dir="experiments/tech_synthetic/appworld_optimized/rollout_log" \
    attribution_driven_credit_assignment.enable=false \
    attribution_driven_credit_assignment.enable_hindsight=false \
    task_manager.n=32 \
    task_manager.mixture.synthetic_data_ratio=2.0 \
    task_manager.mixture.use_original_tasks=False \
    task_manager.train_data_path=./tasks_explored/tasks_explored.train.json \
    task_manager.val_data_path=./tasks_explored/tasks_explored.val.json \
    task_manager.exploration_strategy_args.a=2 \
    task_manager.exploration_strategy_args.b=8 \
    task_manager.strategy=api_driven \
    task_manager.exploration_strategy_args.active_apps="['amazon','gmail','spotify','venmo','simple_note','todoist','splitwise','phone','file_system']" \
    task_manager.exploration_strategy_args.task_labels_path="./environments/appworld/data/datasets/train.jsonl" \
    task_manager.llm_client="azure-gpt-5" \
    task_manager.grader.synthetic_grader=api_process_llm_judge \
    task_manager.env_profile=./cookbook/env_profiles/appworld.json \
    2>&1 | tee "$log_file"