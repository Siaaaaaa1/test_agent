# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications copyright 2025 Alibaba Tongyi EconML Lab. and/or its affiliates
# ... (版权声明省略)
"""
注意：我们没有将此 main 函数与 ray_trainer 合并，因为 ray_trainer 可能会被其他 main 入口使用。
此脚本是运行 PPO 训练的主要入口。
"""
# from best_logger import register_logger
import torch
import os
import hydra
import ray

# 引入 AgentEvolver 特有的模块
from AgentEvolver.agentevolver.client.llm_client_backup import DashScopeClient
from agentevolver.module.task_manager.base import NaiveTaskObjectiveRetrieval
from agentevolver.module.task_manager.data_mixture import OriginalOnlyStrategy, UnifiedMixtureStrategy
from agentevolver.module.task_manager.strategies.random import LlmRandomSamplingExploreStrategy
from agentevolver.module.task_manager.task_manager import TaskManager
from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.module.trainer.ae_ray_trainer import AgentEvolverRayPPOTrainer

# 引入 verl 库的相关模块
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo import core_algos

# =============================================================================
# 1. Monkey Patching: KL 散度计算逻辑修改
# =============================================================================
# 如果环境变量 DEBUG_ARG 中包含 "kl_control"，则替换 verl 库中的 kl_penalty 函数。
# 目的：在计算 PPO 的 KL 惩罚时，引入与序列长度相关的权重，对长序列生成的偏离给予更大的惩罚。
if "kl_control" in os.environ.get("DEBUG_ARG",""):
    print("monkeypatching kl loss (正在对 KL 损失函数打补丁)")
    
    def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
        """
        计算 logprob 和 ref_logprob 之间的 KL 散度或相关惩罚项。
        
        参数:
            logprob: 当前策略的对数概率
            ref_logprob: 参考策略（Ref Model）的对数概率
            kl_penalty: 惩罚类型 ('kl', 'abs', 'mse', 'low_var_kl' 等)
        """
        # 根据配置选择基础的计算公式
        if kl_penalty in ("kl", "k1"):
            res = logprob - ref_logprob
        elif kl_penalty == "abs":
            res = (logprob - ref_logprob).abs()
        elif kl_penalty in ("mse", "k2"):
            res = 0.5 * (logprob - ref_logprob).square()
        elif kl_penalty in ("low_var_kl", "k3"):
            # 低方差 KL 近似：http://joschu.net/blog/kl-approx.html
            kl = ref_logprob - logprob
            # 数值稳定性处理
            kl = torch.clamp(kl, min=-20, max=20)
            ratio = torch.exp(kl)
            kld = (ratio - kl - 1).contiguous()
            res = torch.clamp(kld, min=-10, max=10)
        elif kl_penalty == "full":
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        assert res.dim() == 2, f"kl_penalty should be 2-dim tensor, but got {res.dim()}-dim tensor"
        
        # --- 核心修改部分 ---
        # 根据生成的长度控制 KL 惩罚。越靠后的 token（frontier tokens）会有更高的惩罚权重。
        import agentevolver.utils.utils as ut
        # 计算每个样本非 padding 部分的长度
        lengths = (res != 0).int().sum(dim=-1)
        # 获取指数衰减的权重向量 (Vectorized Exponential Decay Weights)
        weights = ut.get_batched_exponential_decay_weights_vectorized(lengths.tolist())
        # 将权重应用到 KL 结果上
        res = res * weights.unsqueeze(-1)
        return res
    
    # 替换 verl 核心算法中的函数
    core_algos.kl_penalty = kl_penalty
    print("patched (补丁应用完成)")


def get_custom_reward_fn(config):
    """
    动态加载自定义奖励函数。
    
    从配置文件中指定的路径加载 Python 文件，并提取指定的函数作为奖励函数。
    这允许用户在不修改库代码的情况下注入自定义的 Reward 逻辑。
    """
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    # 动态加载模块
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)  # 执行模块以加载定义
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    # 封装奖励函数，注入配置中的额外参数
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


# 使用 Hydra 管理配置，配置文件位于 ../config 目录，默认使用 script_config.yaml
@hydra.main(config_path="../config", config_name="script_config", version_base=None)
def main(config):
    """
    PPO 训练的主入口点。
    """
    run_ppo(config)


def run_ppo(config) -> None:
    """
    初始化 Ray 环境并启动训练任务。
    """
    # 1. 初始化 Ray 集群
    if not ray.is_initialized():
        # 本地 Ray 集群初始化，设置运行时环境变量（如 vLLM 配置、WandB 配置等）
        ray.init(
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true", 
                "NCCL_DEBUG": "WARN", 
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true", # 允许 vLLM 运行时更新 LoRA
                "VLLM_USE_V1": "1", 
                "WANDB_API_KEY": "local-e93291bd40698a593a1fcc5b99da6a71a753a383", # 示例 Key
                "WANDB_BASE_URL": "http://22.6.186.25:8080"
            }},
            num_cpus=config.ray_init.num_cpus,
        )

    # 2. 配置检查
    max_model_len: int = config.actor_rollout_ref.rollout.max_model_len
    # 确保 prompt 长度 + response 长度 不超过模型的最大上下文长度
    assert config.data.max_prompt_length + config.data.max_response_length <= max_model_len, \
        f"max_prompt_length {config.data.max_prompt_length} + max_response_length {config.data.max_response_length} should be <= max_model_len {max_model_len}"

    # 3. 启动 TaskRunner Actor
    # 使用 Ray Actor 模式运行主任务，避免在 Head 节点上运行重型任务
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


# 定义 TaskRunner Actor，指定只需要 1 个 CPU
@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """
    Ray Actor 类，负责具体的训练环境搭建和训练循环启动。
    """
    def run(self, config):
        # 打印并解析配置
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # 1. 模型下载/准备
        # 将 HDFS 或对象存储上的模型 checkpoint 下载到本地
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get('use_shm', False))

        # 2. 初始化 Tokenizer 和 Processor
        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # 用于多模态模型

        # 3. vLLM 版本检查
        # 如果使用 vLLM 作为 Rollout 引擎且启用了 LoRA，需要检查 vLLM 版本是否支持
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge
            if config.actor_rollout_ref.model.get('lora_rank', 0) > 0:
                if not is_version_ge(pkg='vllm', minver='0.7.3'):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # 4. 定义 Worker 类（Actor, Rollout, Reference）
        # 根据分布式策略（FSDP/Megatron）选择对应的 Worker 类
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            # 引入基本的 Worker
            from verl.workers.fsdp_workers import (ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker)
            
            ####################
            # AgentEvolver 特定修改 (ANNI)
            # 使用自定义的 HET (Heterogeneous) Worker 类，支持异构计算或特定的任务逻辑
            from agentevolver.module.exp_manager.het_fsdp_worker import HETAsyncActorRolloutRefWorker, HETActorRolloutRefWorker
            
            # 根据配置选择同步或异步 Rollout Worker
            actor_rollout_cls = HETAsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else HETActorRolloutRefWorker
            ####################
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # Megatron 策略支持 (代码中似乎未针对 AgentEvolver 做特殊定制，沿用 verl 原生)
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import (ActorRolloutRefWorker, CriticWorker)

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        # 5. 资源池映射 (Resource Pool Mapping)
        # 定义每个角色 (Role) 对应哪个 Worker 类
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls), # Actor 和 Rollout 通常共享模型权重
            Role.Critic: ray.remote(CriticWorker),            # Critic 独立
        }

        global_pool_id = "global_pool"
        # 定义资源规格：每个节点多少个 GPU
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # 将角色映射到资源池
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # 6. 配置 Reward Model (RM)
        # 如果启用了独立 Reward Model
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # 7. 配置 Reference Policy (Ref)
        # 如果算法需要 KL 散度（通常 PPO 都需要），则需要 Reference Model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 8. 加载奖励函数管理器
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        
        # 初始化资源池管理器
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # 9. 初始化 TaskManager (核心组件)
        # AgentEvolver 不使用静态数据集，而是使用 TaskManager 动态管理任务
        llm_client = DashScopeClient(model_name=config.task_manager.llm_client)
        
        # 训练集任务管理器
        train_task_manager = TaskManager(
            config=config,
            exploration_strategy=config.task_manager.strategy, # 探索策略
            env_profile=EnvProfile.load_from_json(config.task_manager.env_profile), # 环境配置
            exploration_strategy_args=config.task_manager.strategy_args,
            llm_client=llm_client,
            old_retrival=NaiveTaskObjectiveRetrieval(), # 任务检索器
            mixture_strategy=UnifiedMixtureStrategy(    # 数据混合策略
                use_original=config.task_manager.mixture.use_original_tasks,
                synthetic_ratio=config.task_manager.mixture.synthetic_data_ratio,
                shuffle=config.task_manager.mixture.shuffle,
                seed=42,
            ),
            reward_config=config.task_manager.grader,
            tokenizer=tokenizer,
            env_service_url=config.env_service.env_url, # 环境服务地址
            num_explore_threads=config.task_manager.num_explore_threads,
            n=config.task_manager.n,
        )
        
        # 验证集任务管理器 (通常只使用原始任务，不混合合成数据)
        val_task_manager = TaskManager(
            config=config,
            exploration_strategy=config.task_manager.strategy,
            env_profile=EnvProfile.load_from_json(config.task_manager.env_profile),
            exploration_strategy_args=config.task_manager.strategy_args,
            llm_client=llm_client,
            old_retrival=NaiveTaskObjectiveRetrieval(),
            mixture_strategy=OriginalOnlyStrategy(), # 仅使用原始数据
            reward_config=config.task_manager.grader,
            tokenizer=tokenizer,
            env_service_url=config.env_service.env_url,
            num_explore_threads=config.task_manager.num_explore_threads,
            n=config.task_manager.n,
        )

        # 10. 初始化 Trainer 并开始训练
        # 使用自定义的 AgentEvolverRayPPOTrainer
        trainer = AgentEvolverRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_task_manager=train_task_manager, # 传入 TaskManager
            val_task_manager=val_task_manager,
            collate_fn=collate_fn,
            device_name=config.trainer.device,
        )
        trainer.init_workers() # 初始化分布式 Worker
        trainer.fit()          # 开始训练循环

def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """
    创建强化学习数据集的辅助函数。
    虽然 TaskRunner 中使用了 TaskManager，但此函数可能用于兼容性或测试。
    """
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    # 支持加载自定义数据集类
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset

if __name__ == "__main__":
    import shutil
    # 以下注释代码用于调试或防止递归删除，当前被禁用
    # this will break ray's initialization
    # def _safe_guard(name:str):
    #     import traceback
    #     traceback.print_stack()
    #     print(f"{name} is overwritten")
    # shutil.rmtree=lambda *args, **kwargs: _safe_guard("rmtree")
    main()