# ---- Start Environment Service ----

# 1. 确保 conda 可以在脚本中使用
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 2. 激活环境
conda activate appworld

# 3. 启动 Server 并放入后台运行 (&)
echo "Starting AppWorld Environment Service..."
bash env_service/launch_script/appworld.sh > server.log 2>&1 &
SERVER_PID=$!

# 注册退出陷阱: 脚本退出时自动杀掉 Server 进程
trap "kill $SERVER_PID" EXIT

# 4. 等待服务启动
echo "Waiting for server to start (PID: $SERVER_PID)..."
sleep 10

conda activate agentevolver

# ---- Start Training ----
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"
env_url=http://localhost:8080
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="log_${current_time}.log"

# Ray 配置: 限制 Ray 使用的 CPU 核数，防止资源争抢
export RAY_NUM_CPUS=32
# 指定 GPU (如使用后4张)
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# 修改提示：显示新的 Context 分配
echo "Starting Training with Total Context: 100000 (Prompt: 5000, Response: 95000)..."

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
    data.max_prompt_length=5000 \
    data.max_response_length=95000 \
    \
    actor_rollout_ref.model.path=./model/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.off_cliprange_high=0.6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=100000 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.use_qwen3=False \
    actor_rollout_ref.rollout.enable_request_id=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=100000 \
    actor_rollout_ref.rollout.prompt_length=5000 \
    actor_rollout_ref.rollout.response_length=95000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=100000 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=100000 \
    \
    critic.ppo_max_token_len_per_gpu=100000 \
    critic.forward_max_token_len_per_gpu=100000 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name="AgentEvolver" \
    trainer.experiment_name="appworld_test_0110_100k" \
    trainer.save_freq=2 \
    trainer.test_freq=1 \
    trainer.total_epochs=40 \
    trainer.val_before_train=False \
    trainer.validation_data_dir="experiments/tech_synthetic/${experiment_name}/validation_log" \
    trainer.rollout_data_dir="experiments/tech_synthetic/${experiment_name}/rollout_log" \
    \
    attribution_driven_credit_assignment.enable=false \
    attribution_driven_credit_assignment.enable_hindsight=false \
    \
    task_manager.n=5 \
    task_manager.mixture.synthetic_data_ratio=1.0 \
    task_manager.mixture.use_original_tasks=False \
    task_manager.train_data_path=./tasks_explored/tasks_explored.train.json \
    task_manager.val_data_path=.tasks_explored/tasks_explored.val.json \
    task_manager.exploration_strategy_args.a=1 \
    task_manager.exploration_strategy_args.b=4 \
    task_manager.strategy=api_driven \
    task_manager.exploration_strategy_args.active_apps="['amazon','gmail','spotify','venmo','simple_note','todoist','splitwise','phone','file_system']" \
    task_manager.exploration_strategy_args.task_labels_path="./environments/appworld/data/datasets/train.jsonl" \
    2>&1 | tee "$log_file"