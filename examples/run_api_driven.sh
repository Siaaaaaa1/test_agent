# ---- Start Environment Service ----

# rm -f ./.generate_task_api.intra.json
# rm -f ./.generate_task_api.extra.json
# rm -f ./tasks_explored/tasks_explored.train.json

# 1. 确保 conda 可以在脚本中使用
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 2. 激活环境
conda activate appworld

# 3. 启动 Server 并放入后台运行 (&)
echo "Starting AppWorld Environment Service..."
bash env_service/launch_script/appworld.sh > server.log 2>&1 &
SERVER_PID=$!

# 【修改点 1】：注册退出陷阱 (Trap)
# 无论脚本因何种原因退出 (报错、Ctrl+C、正常结束)，都会执行 kill 命令
# 这样可以防止 AppWorld 服务残留
trap "kill $SERVER_PID" EXIT

# 4. 等待服务启动
echo "Waiting for server to start (PID: $SERVER_PID)..."
sleep 10

conda activate agentevolver

# 线程限制 (保持你原有的)
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export TORCH_NUM_THREADS=1

# 【修改点 2】：强制限制 Ray 可见的 CPU 数量
# 你的训练任务使用了 4 张 GPU，给 Ray 分配 32 个 CPU 核 (每卡8核) 通常足够且高效
# 这能极大减少 Ray 启动的闲置进程数，避免 'Resource temporarily unavailable'
# export RAY_NUM_CPUS=32

# ---- Start Training ----
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"
env_url=http://localhost:8080
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="log_${current_time}.log"

# 指定使用后 4 个 GPU (ID: 4,5,6,7)
# export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "Starting Training..."
# 注意：不需要在 python 参数里手动加 ray.init 了，RAY_NUM_CPUS 环境变量会自动生效
python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    env_service.env_url=$env_url \
    actor_rollout_ref.actor.off_cliprange_high=0.6 \
    attribution_driven_credit_assignment.enable=false \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=4000 \
    data.max_response_length=21580 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.use_qwen3=False \
    actor_rollout_ref.rollout.enable_request_id=False \
    actor_rollout_ref.rollout.prompt_length=20480 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.max_model_len=25580 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.model.path=./model/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=8 \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name="appworld_qwen25-7b" \
    trainer.experiment_name="appworld_qwen25-7b_baseline" \
    trainer.nnodes=1 \
    trainer.save_freq=10000 \
    trainer.test_freq=10 \
    trainer.total_epochs=40 \
    trainer.val_before_train=False \
    trainer.validation_data_dir="experiments/tech_synthetic/${experiment_name}/validation_log" \
    trainer.rollout_data_dir="experiments/tech_synthetic/${experiment_name}/rollout_log" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25580 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=25580 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=25580 \
    critic.ppo_max_token_len_per_gpu=25580 \
    critic.forward_max_token_len_per_gpu=25580 \
    data.train_files=null \
    data.val_files=null \
    env_service.env_type=appworld \
    task_manager.n=5 \
    task_manager.mixture.synthetic_data_ratio=1.0 \
    task_manager.mixture.use_original_tasks=False \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    task_manager.train_data_path=./tasks_explored/tasks_explored.train.json \
    task_manager.val_data_path=.tasks_explored/tasks_explored.val.json \
    task_manager.exploration_strategy_args.a=1 \
    task_manager.exploration_strategy_args.b=4 \
    task_manager.strategy=api_driven \
    task_manager.exploration_strategy_args.active_apps="['amazon','gmail','spotify','venmo','simple_note','todoist','splitwise','phone','file_system']" \
    task_manager.exploration_strategy_args.task_labels_path="./environments/appworld/data/datasets/train.jsonl" \
    debug_log=True \
    seed=1 \
    2>&1 | tee "$log_file"

# 脚本底部的 kill 不需要了，因为 trap 会自动处理
# 但保留也没坏处