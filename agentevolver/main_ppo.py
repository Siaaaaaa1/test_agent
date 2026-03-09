"""
注意：我们没有将此 main 函数与 ray_trainer 合并，因为 ray_trainer 可能会被其他 main 入口使用。
此脚本是运行 PPO 训练的主要入口。
"""
# from best_logger import register_logger
import torch
import os
import hydra
import ray
from agentevolver.utils.utils import seed_everything

# 引入 AgentEvolver 特有的模块
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.task_manager.base import NaiveTaskObjectiveRetrieval
from agentevolver.module.task_manager.data_mixture import OriginalOnlyStrategy, UnifiedMixtureStrategy
from agentevolver.module.task_manager.strategies.random import LlmRandomSamplingExploreStrategy
from agentevolver.module.task_manager.task_manager import TaskManager
from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.module.trainer.ae_ray_trainer import AgentEvolverRayPPOTrainer
from agentevolver.module.adv_processor.hindsight import HindsightManager  # [新增] 引入 HindsightManager

# 引入 verl 库的相关模块
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo import core_algos

# # =============================================================================
# # 1. Monkey Patching: KL 散度计算逻辑修改
# # =============================================================================
# # 如果环境变量 DEBUG_ARG 中包含 "kl_control"，则替换 verl 库中的 kl_penalty 函数。
# # 目的：在计算 PPO 的 KL 惩罚时，引入与序列长度相关的权重，对长序列生成的偏离给予更大的惩罚。
# if "kl_control" in os.environ.get("DEBUG_ARG",""):
#     print("monkeypatching kl loss (正在对 KL 损失函数打补丁)")
    
#     def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
#         """
#         计算 logprob 和 ref_logprob 之间的 KL 散度或相关惩罚项。
        
#         参数:
#             logprob: 当前策略的对数概率
#             ref_logprob: 参考策略（Ref Model）的对数概率
#             kl_penalty: 惩罚类型 ('kl', 'abs', 'mse', 'low_var_kl' 等)
#         """
#         # 根据配置选择基础的计算公式
#         if kl_penalty in ("kl", "k1"):
#             res = logprob - ref_logprob
#         elif kl_penalty == "abs":
#             res = (logprob - ref_logprob).abs()
#         elif kl_penalty in ("mse", "k2"):
#             res = 0.5 * (logprob - ref_logprob).square()
#         elif kl_penalty in ("low_var_kl", "k3"):
#             # 低方差 KL 近似：http://joschu.net/blog/kl-approx.html
#             kl = ref_logprob - logprob
#             # 数值稳定性处理
#             kl = torch.clamp(kl, min=-20, max=20)
#             ratio = torch.exp(kl)
#             kld = (ratio - kl - 1).contiguous()
#             res = torch.clamp(kld, min=-10, max=10)
#         elif kl_penalty == "full":
#             raise NotImplementedError
#         else:
#             raise NotImplementedError
        
#         assert res.dim() == 2, f"kl_penalty should be 2-dim tensor, but got {res.dim()}-dim tensor"
        
#         # --- 核心修改部分 ---
#         # 根据生成的长度控制 KL 惩罚。越靠后的 token（frontier tokens）会有更高的惩罚权重。
#         import agentevolver.utils.utils as ut
#         # 计算每个样本非 padding 部分的长度
#         lengths = (res != 0).int().sum(dim=-1)
#         # 获取指数衰减的权重向量 (Vectorized Exponential Decay Weights)
#         raw_weights = ut.get_batched_exponential_decay_weights_vectorized(lengths.tolist())
        
#         # --- 核心改进：均值归一化 (Mean Normalization) ---
#         # 确保应用权重后，该样本的总 KL 惩罚量级不会因为衰减而消失
#         # 这防止了梯度在不同 Token 位置上的“暴涨暴跌”
#         weight_means = raw_weights.sum(dim=-1, keepdim=True) / lengths.unsqueeze(-1).float()
#         normalized_weights = raw_weights / (weight_means + 1e-8)
        
#         res = res * normalized_weights
#         return res
    
#     # 替换 verl 核心算法中的函数
#     core_algos.kl_penalty = kl_penalty
#     print("patched (补丁应用完成)")


def get_custom_reward_fn(config):
    """
    动态加载自定义奖励函数。
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
        spec.loader.exec_module(module)
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


@hydra.main(config_path="../config", config_name="script_config", version_base=None)
def main(config):
    """
    PPO 训练的主入口点。
    """
    seed = config.get("seed", 42)
    seed_everything(seed)
    run_ppo(config)


def run_ppo(config) -> None:
    """
    初始化 Ray 环境并启动训练任务。
    """
    if not ray.is_initialized():
        ray_address = getattr(config.ray_init, "address", "auto")
        ray_num_cpus = getattr(config.ray_init, "num_cpus", None)

        print(f"🔌 [DEBUG] Connecting to Ray Cluster at: {ray_address}")
        
        runtime_env_config = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true", 
                "NCCL_DEBUG": "WARN", 
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                "VLLM_USE_V1": "1", 
                "WANDB_API_KEY": os.getenv("WANDB_API_KEY", "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"),
            }
        }

        try:
            ray.init(
                address=ray_address, 
                runtime_env=runtime_env_config,
                include_dashboard=False,
                ignore_reinit_error=True,
                num_cpus=ray_num_cpus 
            )
            print("✅ [DEBUG] Ray initialized successfully!")
            
        except Exception as e:
            print(f"⚠️ [DEBUG] Failed to init Ray with address={ray_address}: {e}")
            print("🔄 [DEBUG] Fallback: Trying to init generic local Ray...")
            ray.init(
                num_cpus=ray_num_cpus,
                runtime_env=runtime_env_config,
                ignore_reinit_error=True
            )

    max_model_len: int = config.actor_rollout_ref.rollout.max_model_len
    assert config.data.max_prompt_length + config.data.max_response_length <= max_model_len, \
        f"max_prompt_length {config.data.max_prompt_length} + max_response_length {config.data.max_response_length} should be <= max_model_len {max_model_len}"

    # 启动 TaskRunner Actor
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


# [调整] 取消 num_cpus=1 的限制，使 Actor 可以利用更多资源进行多线程任务
@ray.remote
class TaskRunner:
    """
    Ray Actor 类，负责具体的训练环境搭建和训练循环启动。
    """
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        seed = config.get("seed", 42)
        seed_everything(seed)

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # 1. 模型下载/准备
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get('use_shm', False))

        # 2. 初始化 Tokenizer 和 Processor
        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)

        # 3. vLLM 版本检查
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge
            if config.actor_rollout_ref.model.get('lora_rank', 0) > 0:
                if not is_version_ge(pkg='vllm', minver='0.7.3'):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # 4. 定义 Worker 类（Actor, Rollout, Reference）
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import CriticWorker
            
            # AgentEvolver 特定修改 (ANNI)
            from agentevolver.module.exp_manager.het_fsdp_worker import HETAsyncActorRolloutRefWorker, HETActorRolloutRefWorker
            
            actor_rollout_cls = HETAsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else HETActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        # 5. 资源池映射
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls), 
            Role.Critic: ray.remote(CriticWorker),            
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # 6. 配置 Reward Model (RM)
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
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker as DefaultActorRolloutRefWorker
            role_worker_mapping[Role.RefPolicy] = ray.remote(DefaultActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 8. 加载奖励函数管理器
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        from verl.utils.dataset.rl_dataset import collate_fn

        # 9. 初始化 TaskManager
        llm_client = DashScopeClient(model_name=config.task_manager.llm_client)
        
        train_task_manager = TaskManager(
            config=config,
            exploration_strategy=config.task_manager.strategy,
            env_profile=EnvProfile.load_from_json(config.task_manager.env_profile),
            exploration_strategy_args=config.task_manager.strategy_args,
            llm_client=llm_client,
            old_retrival=NaiveTaskObjectiveRetrieval(),
            mixture_strategy=UnifiedMixtureStrategy(
                use_original=config.task_manager.mixture.use_original_tasks,
                synthetic_ratio=config.task_manager.mixture.synthetic_data_ratio,
                shuffle=config.task_manager.mixture.shuffle,
                seed=seed,
            ),
            reward_config=config.task_manager.grader,
            tokenizer=tokenizer,
            env_service_url=config.env_service.env_url,
            num_explore_threads=config.task_manager.num_explore_threads,
            n=config.task_manager.n,
        )
        
        val_task_manager = TaskManager(
            config=config,
            exploration_strategy=config.task_manager.strategy,
            env_profile=EnvProfile.load_from_json(config.task_manager.env_profile),
            exploration_strategy_args=config.task_manager.strategy_args,
            llm_client=llm_client,
            old_retrival=NaiveTaskObjectiveRetrieval(),
            mixture_strategy=OriginalOnlyStrategy(),
            reward_config=config.task_manager.grader,
            tokenizer=tokenizer,
            env_service_url=config.env_service.env_url,
            num_explore_threads=config.task_manager.num_explore_threads,
            n=config.task_manager.n,
        )

        # 10. 初始化 Hindsight Manager
        hindsight_manager = None
        attribution_cfg = config.get("attribution", {})
        if attribution_cfg.get("enable_hindsight", False):
            print("[Main] Initializing HindsightManager for AgentEvolver...")
            try:
                save_path = attribution_cfg.get("hindsight_save_path", "tasks_explored/hindsight_supplement.jsonl")
                hindsight_manager = HindsightManager(
                    llm_client=llm_client,
                    tokenizer=tokenizer,  # [保留] 直接传入 tokenizer
                    saved_path=save_path,
                    num_threads=config.task_manager.get("num_explore_threads", 4)
                )
            except Exception as e:
                print(f"[Warning] Failed to initialize HindsightManager in main_ppo: {e}")

        # 11. 初始化 Trainer 并开始训练
        trainer = AgentEvolverRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_task_manager=train_task_manager,
            val_task_manager=val_task_manager,
            collate_fn=collate_fn,
            device_name=config.trainer.device,
            hindsight_manager=hindsight_manager,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """
    创建强化学习数据集的辅助函数。
    """
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

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
    # 以下注释代码用于调试或防止递归删除，当前被禁用
    # this will break ray's initialization
    # def _safe_guard(name:str):
    #     import traceback
    #     traceback.print_stack()
    #     print(f"{name} is overwritten")
    # shutil.rmtree=lambda *args, **kwargs: _safe_guard("rmtree")
    main()