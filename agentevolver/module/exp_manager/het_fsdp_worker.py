# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications copyright 2025 Alibaba Tongyi EconML Lab. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO 算法的主要入口点，定义了负责具体计算任务的 Worker 类。
"""

import json
import logging
import os
import warnings
from dataclasses import asdict
from typing import Union

import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.debug.performance import _timer, reduce_timing
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

def create_device_mesh(world_size, fsdp_size):
    """
    为分布式训练初始化设备网格 (Device Mesh)。
    
    用于定义进程组之间的通信拓扑，特别是在混合并行策略（如 FSDP + DDP）中。

    参数:
        world_size (int): 分布式训练的总进程数。
        fsdp_size (int): 用于 FSDP 分片的进程数大小。

    返回:
        torch.distributed.device_mesh.DeviceMesh: 初始化的设备网格对象。
    """
    if fsdp_size < 0 or fsdp_size >= world_size:
        # 如果 fsdp_size 无效或等于 world_size，则所有进程组成一个维度的 mesh，即全量 FSDP
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        # 否则，构建二维 mesh：(DDP 组数, FSDP 组内大小)
        # 例如：world_size=8, fsdp_size=4 -> (2, 4) mesh。
        # 第一维用于数据并行(DDP)，第二维用于模型分片(FSDP)。
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"])
    return device_mesh


def get_sharding_strategy(device_mesh):
    """
    根据设备网格的维度确定 FSDP 的分片策略 (Sharding Strategy)。

    参数:
        device_mesh (torch.distributed.device_mesh.DeviceMesh): 用于分片的设备网格。

    返回:
        ShardingStrategy: 对应的分片策略枚举值。
    """
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        # 一维 mesh 使用完全分片 (ZeRO-3)
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        # 二维 mesh 使用混合分片 (Hybrid Sharding / ZeRO-3 within group, DDP across groups)
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class HETActorRolloutRefWorker(Worker):
    """
    异构 Actor-Rollout-Reference Worker 类。
    
    这个 Worker 可以根据配置及其角色 (role) 被实例化为独立的 Actor、Rollout、Reference 策略，
    或者它们的组合（混合引擎）。它负责管理底层的模型加载、优化器配置、FSDP 包装以及并行计算逻辑。
    """

    def __init__(self, config: DictConfig, role: str):
        """
        初始化 Worker，设置分布式环境、设备网格和配置。

        参数:
            config (DictConfig): 包含模型、训练、Rollout 等设置的配置对象。
            role (str): Worker 的角色，决定了它承担的任务。
                        可选值: "actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"。
        """
        super().__init__()
        self.config = config
        import torch.distributed

        # 1. 初始化分布式进程组 (Process Group)
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}", rank=rank, world_size=world_size)

        # 2. 构建 FSDP 设备网格
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): 支持更大模型的 FSDP 混合分片
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # 3. 构建 Ulysses 序列并行 (Sequence Parallel) 设备网格
        # Ulysses 是一种针对长序列训练的高效并行方法
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self._lora_rank > 0

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        # 根据角色设置标志位
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # 4. 设置参数卸载 (Offload) 策略
        # 卸载参数或优化器状态到 CPU 以节省显存
        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: 手动卸载似乎比 FSDP 内置卸载慢
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # 5. 归一化配置 (Normalize Config)
        # 根据分布式设置调整 batch size
        if self._is_actor:
            # PPO mini_batch_size 需要乘以 rollout 的数量 (n)
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            # 再除以数据并行的路数 (总进程数 / 序列并行度)
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, f"归一化后的 ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} 必须大于 0"
            
            # 处理 micro batch size
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, \
                    f"归一化后的 ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} 应能被 ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu} 整除"

        # 归一化 rollout 配置
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # 归一化 ref 配置
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        """
        构建模型和优化器，并应用 FSDP 包装。

        参数:
            model_path (str): 模型路径。
            fsdp_config (dict): FSDP 相关配置。
            optim_config (dict): 优化器配置（学习率等）。
            override_model_config (dict): 用于覆盖默认模型配置的参数。
            use_remove_padding (bool): 是否移除 padding 以优化计算。
            use_fused_kernels (bool): 是否使用融合算子（如 FlashAttention）。
            enable_gradient_checkpointing (bool): 是否开启梯度检查点（省显存）。
            trust_remote_code (bool): 是否信任远程代码。
            use_liger (bool): 是否应用 Liger Kernel 优化。
            role (str): 模型角色 ("actor" 或 "ref")。
            enable_activation_offload (bool): 是否开启激活值卸载。

        返回:
            tuple: (FSDP模型, 优化器, 学习率调度器, 模型配置)
        """
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # 初始化 tokenizer 和 processor
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        # 确定数据类型
        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # 加载并覆盖模型配置
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")

        # 针对 Kimi-VL 模型的特殊处理
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # 获取初始化上下文管理器，处理参数分片初始化 (Meta Device)
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 判断模型类型（多模态或纯语言模型）
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                actor_module_class = AutoModelForCausalLM

            # 加载模型
            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # 应用 Liger Kernel 优化
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=actor_module)

            # 应用 Monkey Patch（移除 padding、Ulysses 并行支持等）
            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            actor_module.to(torch_dtype)

            # 开启梯度检查点
            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            # 应用 LoRA
            if self._is_lora:
                print("Applying LoRA to actor module")
                actor_module.enable_input_require_grads()
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none"
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # 配置 FSDP 混合精度
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        # 获取 FSDP 自动包装策略
        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get("wrap_policy", None), is_lora=self.config.model.get("lora_rank", 0) > 0)

        # 特殊处理：Rollout 使用 HF 时可能因 auto_wrap_policy 挂起
        if self._is_rollout and self.config.rollout.name == "hf":
            auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # 配置 CPU Offload：Reference 强制开启，Actor 强制关闭（因为梯度累积问题）
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        
        # 初始化 FSDP 模型
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # Zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.config.actor.fsdp_config.forward_prefetch,
            )
        elif fsdp_strategy == "fsdp2":
            # 使用 PyTorch 2.4+ 的 FSDP2 API
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # 初始化优化器 (Optimizer) 和 学习率调度器 (Scheduler)
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            # 配置调度器参数
            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, min_lr_ratio=min_lr_ratio, num_cycles=num_cycles)
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        """
        构建 Rollout 实例（推理引擎）及其分片管理器。

        支持 HuggingFace (HF)、vLLM、SGLang 等多种后端。

        参数:
            trust_remote_code (bool): 是否信任远程代码。

        返回:
            tuple: (rollout实例, rollout分片管理器)
        """
        from torch.distributed.device_mesh import init_device_mesh

        # 计算推理张量并行 (Tensor Parallel) 大小
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        # 创建 Rollout 的设备网格
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        
        rollout_name = self.config.rollout.name
        if rollout_name == "hf":
            # 使用 HuggingFace 进行 Rollout
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()

        elif rollout_name == "vllm":
            # 使用 vLLM 进行加速 Rollout
            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
            lora_kwargs = {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}} if self._is_lora else {}
            
            # vLLM 初始化
            if vllm_mode == "customized":
                rollout = vLLMRollout(actor_module=self.actor_module_fsdp, config=self.config.rollout, tokenizer=self.tokenizer, model_hf_config=self.actor_model_config, trust_remote_code=trust_remote_code, **lora_kwargs)
            elif vllm_mode == "spmd":
                from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

                vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
                rollout = vllm_rollout_cls(model_path=local_path, config=self.config.rollout, tokenizer=self.tokenizer, model_hf_config=self.actor_model_config, device_mesh=rollout_device_mesh, trust_remote_code=trust_remote_code, **lora_kwargs)
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            full_params = torch.distributed.get_world_size() == 1
            # 初始化 FSDP 到 vLLM 的分片管理器，负责权重转换
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name in ["sglang", "sglang_async"]:
            # 使用 SGLang 进行 Rollout
            if rollout_name == "sglang_async":
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            from verl.workers.rollout.sglang_rollout import SGLangRollout
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        初始化模型的入口函数。
        
        该方法被注册为 ONE_TO_ALL 模式，通常由控制器调用。它负责调用 _build_model_optimizer 和 _build_rollout
        来实例化 Actor、Rollout 和 Reference 模型。
        """
        from verl.workers.actor import DataParallelPPOActor

        # 导入外部库
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        # 1. 初始化 Actor 模型 (如果当前角色包含 Actor 或 Rollout)
        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # 获取原始未包装的模型（用于某些特定操作）
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            # 如果配置了卸载，初始化后立即卸载到 CPU
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        # 2. 初始化 Actor 训练类
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            ##################
            # ANNI: 使用自定义的异构数据并行 PPO Actor
            from .het_actor import HETDataParallelPPOActor
            self.actor = HETDataParallelPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)
            ##################

        # 3. 初始化 Rollout
        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        # 4. 初始化 Reference 模型 (如果需要)
        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        # 5. 初始化辅助工具：Flops计数器、Checkpoint管理器
        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # 如果只是独立的 Rollout Worker，也需要 checkpoint manager 来加载模型权重
            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=checkpoint_contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        """
        执行 PPO 算法中的 Actor 更新步骤。

        过程包括：将模型加载到 GPU，通过分片管理器处理数据，计算梯度更新参数，计算性能指标，最后（可选）将模型卸载回 CPU。

        参数:
            data (DataProto): 包含训练数据（states, actions, advantages 等）的协议对象。

        返回:
            DataProto: 包含更新后的指标（如 loss, entropy 等）的对象。
        """
        # 数据先转到 CPU，稍后在 actor.update_policy 中按 micro batch 转到 GPU
        data = data.to("cpu")

        assert self._is_actor
        # 如果启用了卸载，需要先将模型和优化器加载到 GPU
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        # 使用 Ulysses 分片管理器处理序列并行的数据
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # 执行核心训练步骤
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            
            # 记录性能指标 (FLOPs, Memory)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            # 记录并更新学习率
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.actor_lr_scheduler.step()

            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        # 训练结束后，如果启用了卸载，将模型和优化器移回 CPU
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        """
        基于提示词 (prompts) 生成序列。通常用于 Rollout 阶段。

        参数:
            prompts (DataProto): 包含输入提示词的数据对象。

        返回:
            DataProto: 包含生成序列及相关元数据的数据对象。
        """
        # 将数据移至对应设备
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        
        # 使用 Rollout 分片管理器（例如负责将 FSDP 权重转换为 vLLM 格式）
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            with _timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        # 聚合所有 rank 的耗时信息
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # 清理 KV cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        """
        使用 Actor 模型计算给定数据的对数概率 (Log Probability)。
        
        用于 PPO 算法中计算新旧策略的比率。支持 LoRA 和 FSDP。

        参数:
            data (DataProto): 包含输入 token 的数据对象。

        返回:
            DataProto: 包含计算出的 old_log_probs 和 entropy 的数据对象。
        """
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        # 如果是计算 LoRA 模型的 log_prob 但不需要 LoRA 效果（如用作 Reference），则禁用 adapter
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_device_id())
        
        # 设置计算所需的元数据
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        
        # 执行计算
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # 重新分片 FSDP 根模块以节省内存
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        """
        计算参考策略 (Reference Policy) 的对数概率。

        如果使用了 LoRA，Reference 策略通常就是禁用了 Adapter 的 Base Model。
        否则，使用独立的 Reference 模型计算。

        参数:
            data (DataProto): 输入数据。

        返回:
            DataProto: 包含 ref_log_prob 的数据对象。
        """
        if self._is_lora:
            # 如果是 LoRA 模式，调用 compute_log_prob 并标记 is_lora=True 来禁用 adapter 作为 ref
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            # 将结果中的 old_log_probs 重命名为 ref_log_prob
            data = DataProto.from_dict(tensors={"ref_log_prob": data.batch["old_log_probs"]})
            return data
        assert self._is_ref
        
        # 使用独立的 Reference 模型计算
        data = data.to(get_device_id())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """
        保存模型检查点 (Checkpoint)。支持 FSDP 模型状态和 LoRA Adapter 的保存。

        参数:
            local_path (str): 本地保存路径。
            hdfs_path (str, optional): HDFS 保存路径。
            global_step (int): 当前全局步数。
            max_ckpt_to_keep (int, optional): 最多保留的检查点数量。
        """
        from verl.utils.logger import log_with_rank

        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # 保存主模型权重
        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        dist.barrier()

        # 如果是 LoRA 模型，单独保存 Adapter 权重
        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    # 收集所有 rank 的 LoRA 参数
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True)

            dist.barrier()
            log_with_rank(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}", rank=dist.get_rank(), logger=logger, log_only_rank_0=True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        """
        加载模型检查点。

        参数:
            local_path (str): 本地路径。
            hdfs_path (str, optional): HDFS 路径。
            del_local_after_load (bool): 加载后是否删除本地文件。
        """
        assert self._is_actor or (not self._is_actor and self._is_rollout), f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got {self._is_actor} and {self._is_rollout}"

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)


# ================================= Async related workers =================================
class HETAsyncActorRolloutRefWorker(HETActorRolloutRefWorker):
    """
    异步版本的 Worker，主要用于支持异步的 Rollout 执行（如 vLLM 服务模式）。
    """
    def _build_rollout(self, trust_remote_code=False):
        """
        构建异步 Rollout。与同步版不同，这里不立即初始化完整的 Rollout 引擎，而是推迟到 AsyncvLLMServer 中处理。
        """
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout 在这里并未真正初始化，而是延迟初始化
        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # 用于 sleep/wake_up 控制
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        """
        异步 Worker 不支持同步的 generate_sequences 调用。
        """
        raise NotImplementedError("AsyncActorRolloutRefWorker does not support generate_sequences")

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """
        直接在 Rollout 实例上执行指定的方法。通常通过 RPC 调用。

        参数:
            method (str): 方法名。
            *args, **kwargs: 传递给方法的参数。
        """
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: {method if isinstance(method, str) else 'Callable'}")
        return self.rollout.execute_method(method, *args, **kwargs)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        """
        异步执行对话补全请求。

        参数:
            json_request (dict): 包含对话历史的 JSON 请求。
        
        返回:
            dict: 生成结果。
        """
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        """
        唤醒 Rollout Worker。
        """
        await self.rollout.wake_up()
        # 返回 True 以阻塞调用者，确认唤醒完成
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        """
        使 Rollout Worker 进入休眠状态，释放资源或等待。
        """
        await self.rollout.sleep()
        return True