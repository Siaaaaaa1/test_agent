import random
import re
from loguru import logger
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import List, Dict, Any, Optional, Literal, Tuple
from itertools import groupby
from concurrent.futures import as_completed, Future
from concurrent.futures.thread import ThreadPoolExecutor
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from agentevolver.client.em_client import EMClient


@dataclass
class TaskExpConfig:
    """
    任务级别的经验配置。
    决定了一个 Task 在多次 Rollout 中，每一次是否添加经验，以及生成的样本在训练时的模式。
    """
    add_exp: List[bool]             # 列表长度等于 rollout_n。True 表示该次 rollout 要注入经验。
    train_mode: str = "discard"     # "keep" (保留经验, 模仿学习) | "discard" (剔除经验, 强化内化)

@dataclass
class TrajExpConfig:
    """
    单条轨迹级别的经验配置。
    在 Rollout 运行时传递给 EnvWorker。
    """
    add_exp: bool = True            # 本次轨迹是否添加经验
    train_mode: str = "discard"     # 本次轨迹对应的训练模式
    task_id: str = ""
    data_id: str = ""
    rollout_id: str = ""
    query: str = ""
    mode: str = "sample"            # "sample" (训练) | "validate" (验证)
    experience_list: List[str] = field(default_factory=list) # 实际检索并注入的经验文本列表



class ExperienceManager(object):
    """
    经验管理器 (策略层)。
    职责：
    1. 分配策略：决定哪些任务加经验，哪些任务不加；决定训练时是 Keep 还是 Discard。
    2. 经验进化：调用后端服务，对新生成的轨迹进行总结 (Summarize)，更新经验池。
    """

    def __init__(self, config: DictConfig):
        """
        初始化经验管理器。

        Args:
            config (DictConfig): 全局配置对象。
        """
        self.config: DictConfig = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.exp_manager_config = config.exp_manager
        self.reme_config = config.exp_manager.reme  # REME (Retrieval-Enhanced Memory Engine) 配置

        # 读取策略配置
        self.val_rollout_mode = self.exp_manager_config.val_rollout_mode      # 验证集 Rollout 模式
        self.train_rollout_mode = self.exp_manager_config.train_rollout_mode  # 训练集 Rollout 模式
        self.rollout_ratio = self.exp_manager_config.rollout_ratio            # 加经验的比例
        self.train_sample_mode = self.exp_manager_config.train_sample_mode    # 训练样本处理模式 (keep/discard/hybrid)
        self.train_sample_keepratio = self.exp_manager_config.train_sample_keepratio # Hybrid 模式下 Keep 的比例

        # 用于异步提交总结任务的线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool.max_workers)
        # 负责与经验记忆服务 (Experience Maker Service) 通信的客户端
        self.em_client = EMClient(base_url=self.reme_config.base_url)
    
    def summarize_in_batch(self, trajectories: List[Trajectory]) -> None:
        """
        批量总结轨迹并更新经验池。
        这通常在验证阶段或特定的数据收集阶段调用。
        """
        # 按 task_id 排序并分组，确保同一任务的轨迹在一起处理
        trajectories_sorted = sorted(trajectories, key=lambda traj: traj.task_id)
        grouped_trajectories = [list(group) for key, group in groupby(trajectories_sorted, key=lambda traj: traj.task_id)]
        
        # 将轨迹分批
        batch_size = self.exp_manager_config.summary_batch_size
        all_batches = []
        for group in grouped_trajectories:
            for i in range(0, len(group), batch_size):
                all_batches.append(group[i:i + batch_size])
        
        # 异步提交总结请求
        futures = []
        for batch in all_batches:
            future = self.thread_pool.submit(
                self.em_client.call_summarizer,
                trajectories=batch,
                workspace_id=self.reme_config.workspace_id
            )
            futures.append(future)
        
        # 等待所有请求完成
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in summary task: {e}")
        
        return

    def submit_summary_task(self, trajectories: List[Trajectory], global_steps: int) -> Optional[Future]:
        """
        在训练循环中异步提交总结任务。
        这是经验池"自我进化"的关键步骤：将 Agent 刚刚生成的成功轨迹总结成经验存入库中。

        Args:
            trajectories (List[Trajectory]): 待总结的轨迹列表。
            global_steps (int): 当前全局训练步数。

        Returns:
            Optional[Future]: 异步任务句柄。
        """
        # 检查是否满足更新频率要求
        if not self._should_submit_summary(global_steps):
            return None
        
        try:
            summary_task = self.thread_pool.submit(
                self.em_client.call_summarizer,
                trajectories=trajectories,
                workspace_id=self.reme_config.workspace_id
            )
            print(f"[Summary] Async task submitted at step {global_steps}")
            return summary_task
        except Exception as e:
            print(f"[Summary] Failed to submit task: {e}")
            return None

    def _should_submit_summary(self, global_steps: int) -> bool:
        """
        判断当前步骤是否应该触发经验总结。
        """
        return (
            self.reme_config.enable_summarizer
            and self.reme_config.updated_freq
            and global_steps % self.reme_config.updated_freq == 0
        )
    

    def collect_summary_result(self, summary_task: Optional[Future]) -> Optional[float]:
        """
        收集异步总结任务的结果（主要是为了监控和日志，不阻塞主训练流太久）。
        """
        if summary_task is None:
            return None
        try:
            print("[Summary] Waiting for task completion...")
            summarizer_response, time_cost = summary_task.result()
            print(f"[Summary] Task completed in {time_cost:.2f}s")
            return time_cost
        except Exception as e:
            print(f"[Summary] Task failed: {e}")
            return None

    def get_complete_exp_configs(self, tasks: List[Task], mode: Literal["sample", "validate"]) -> List[TaskExpConfig]:
        """
        为一批任务生成完整的经验配置。
        这是 HET 策略的核心分配逻辑入口。

        Args:
            tasks: 任务列表。
            mode: "sample" (训练阶段) 或 "validate" (验证阶段)。

        Returns:
            List[TaskExpConfig]: 每个任务对应的配置列表。
        """
        # 1. 分配训练模式 (Keep vs Discard)
        exp_manager_configs = self.allocate_train_mode(tasks)
        # 2. 分配 Rollout 时的加经验策略 (Add vs Not Add)
        exp_manager_configs = self.allocate_add_exp(exp_manager_configs, mode)
        return exp_manager_configs

    def allocate_train_mode(self, tasks: List[Task]) -> List[TaskExpConfig]:
        """
        分配训练模式。
        根据 `train_sample_mode` 配置，决定每个任务生成的样本在训练时是保留经验 (Keep) 还是剔除经验 (Discard)。
        
        策略：
        - allkeep (EC): 全部保留。模仿学习，可以快速提升性能，但泛化性差。
        - alldiscard (EI): 全部剔除。即使 Rollout 时看了经验，训练时也假装没看，强迫模型内化能力。
        - hybrid (HET): 混合模式。一部分 Keep，一部分 Discard，平衡稳定性和泛化性。
        """
        mode_to_ratio = {
            "allkeep": 1.0,
            "alldiscard": 0.0,
            "hybrid": self.train_sample_keepratio # 从配置读取比例，例如 0.5
        }
        keep_ratio = mode_to_ratio.get(
            self.train_sample_mode, self.train_sample_keepratio
        )
        
        # 计算需要 Keep 的数量
        keep_count = int(len(tasks) * keep_ratio)
        # 生成模式列表并打乱
        exp_modes = ['keep'] * keep_count + ['discard'] * (len(tasks) - keep_count)
        random.shuffle(exp_modes)
        
        return [TaskExpConfig(add_exp=[], train_mode=exp_mode) for exp_mode in exp_modes]
    
    def allocate_add_exp(self, exp_configs: List[TaskExpConfig], mode: Literal["sample", "validate"]) -> List[TaskExpConfig]:
        """
        分配 Rollout 时的加经验策略。
        决定对于每个任务的 N 次采样 (Rollout N)，其中有几次是带着经验去跑的。

        Args:
            exp_configs: 已分配好 train_mode 的配置列表。
            mode: 当前阶段。

        策略：
        - woexp: 全不加。纯探索。
        - all: 全加。纯利用。
        - mixed: 混合。根据 `rollout_ratio` 随机决定 N 次采样中有几次加经验。
        """
        is_validate = mode == "validate"
        rollout_n = self.rollout_config.val_kwargs.n if is_validate else self.rollout_config.n
        exp_mode = self.val_rollout_mode if is_validate else self.train_rollout_mode
        
        for task_exp_config in exp_configs:
            add_exp_choices = {
                "woexp": [False] * rollout_n,
                # mixed 模式：根据比例生成 True/False 列表并随机打乱
                "mixed": sorted([i < round(rollout_n*self.rollout_ratio) for i in range(rollout_n)], key=lambda _: random.random()),
                "all": [True] * rollout_n
            }[exp_mode]
            task_exp_config.add_exp = add_exp_choices
        
        return exp_configs




class ExperienceWorker(object):
    """
    经验工作者 (执行层)。
    职责：
    1. Rollout 前：检索经验并修改 Prompt (Inject)。
    2. 训练数据生成时：根据策略移除 Prompt 中的经验 (Remove)。
    """
    def __init__(self, config: DictConfig):
        self.config: DictConfig = config
        # 经验模版，例如: "\n\nRelevant Experience:<EXP>{}</EXP>"
        self.experience_template = self.config.exp_manager.experience_template
    
    def manage_rollout_context(self, init_messages: List[dict], traj_exp_config: TrajExpConfig) -> Tuple[List[dict], TrajExpConfig]:
        """
        在 Rollout 阶段管理上下文。核心功能是 RAG (检索增强生成)。
        如果配置要求加经验，则调用 EMClient 检索相似经验，并注入到初始 Prompt 中。

        Args:
            init_messages: 初始对话消息列表。
            traj_exp_config: 轨迹经验配置。

        Returns:
            更新后的消息列表和配置对象。
        """
        # 检查是否应该处理经验 (add_exp=True 且 context_generator 启用)
        if not self._should_process_experience(traj_exp_config):
            return init_messages, traj_exp_config
        
        # 确保客户端已初始化
        self._ensure_em_client()
        
        # 构造临时的轨迹对象用于检索 (主要是为了传 query)
        trajectory = Trajectory(
            data_id=traj_exp_config.data_id,
            rollout_id=traj_exp_config.rollout_id,
            steps=init_messages,
            query=traj_exp_config.query
        )

        # 1. 调用 EMClient 检索历史经验 (Top-K)
        reme_config = self.config.exp_manager.reme
        history_experience = self.em_client.call_context_generator(
            trajectory=trajectory,
            retrieve_top_k=reme_config.retrieve_top_k,
            workspace_id=reme_config.workspace_id
        )

        # 检查是否为空
        if not history_experience:
            logger.info("Experience is empty!")
            return init_messages, traj_exp_config

        # 2. 将检索到的经验注入到 Prompt 中
        logger.info(f"Retrieved history experience: {history_experience}")
        formatted_experience = self.experience_template.format(history_experience)
        
        # 将经验文本追加到最后一条消息 (通常是 User Query) 的内容中
        new_content = formatted_experience + trajectory.steps[-1]["content"]
        trajectory.steps[-1]["content"] = new_content
        
        # 记录注入的经验，以便后续处理
        traj_exp_config.experience_list = traj_exp_config.experience_list + [formatted_experience]

        return trajectory.steps, traj_exp_config
    
    def _should_process_experience(self, traj_exp_config: TrajExpConfig) -> bool:
        """
        检查是否满足处理经验的条件。
        """
        return (traj_exp_config.add_exp and
                self.config.exp_manager.reme.enable_context_generator)
    
    def _ensure_em_client(self) -> None:
        """
        懒加载 EMClient，避免在不需要时建立连接。
        """
        if not hasattr(self, 'em_client'):
            self.em_client = EMClient(
                base_url=self.config.exp_manager.reme.base_url
            )



    def manage_training_context(self, message: str, metadata_config: Dict) -> Tuple[str, str]:
        """
        在生成训练数据阶段管理上下文。
        核心功能是 EI (Experience Improvement) 策略的实现：移除经验。

        Args:
            message: 输入的消息内容 (可能包含之前注入的经验)。
            metadata_config: 包含 train_mode 的元数据。

        Returns:
            提取出的经验字符串 和 清理后的消息内容。
        """
        experience = ""
        cleaned_message = message

        # 如果训练模式是 "discard" (剔除经验)
        if metadata_config.get("task_train_mode", "discard") == "discard": 
            # 使用正则表达式匹配模版，提取经验内容并从消息中删除
            # 例如模版是 "Hint: <EXP>{}</EXP>"，这里会匹配并删除整段 Hint
            pattern = re.escape(self.experience_template).replace(r'\{\}', '(.*?)')
            match = re.search(pattern, message, re.DOTALL)
            if match:
                experience = match.group(1) # 提取经验内容
                cleaned_message = re.sub(pattern, '', message, flags=re.DOTALL) # 删除经验部分

        # 如果模式是 "keep"，则不修改 message，模型将看到带有 Hint 的 Input 进行训练
        
        return experience, cleaned_message