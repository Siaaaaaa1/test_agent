# agentevolver/utils/smart_logger.py
import os
import json
import uuid
import re
from datetime import datetime
import threading
import fcntl
import numpy as np
import torch
from loguru import logger

class SmartLogger:
    """
    工业级大模型 RLHF/GRPO 训练智能日志器。
    支持高并发多进程安全写入、多模态 Token 清洗、Agent 轨迹去重及结构化拆分、
    细粒度对齐校验以及自动化的策略崩溃与奖励黑客(Reward Hacking)诊断预警。
    """
    def __init__(self, log_dir="./log"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 1. 生成唯一 ID (uuid4 前 4 位 + MMDD)
        uid_prefix = str(uuid.uuid4())[:4]
        date_str = datetime.now().strftime("%m%d")
        base_name = os.path.join(self.log_dir, f"log_{uid_prefix}_{date_str}")
        
        # 拆分为两个文件，避免 O(N^2) 的 I/O 阻塞
        self.step_log_file = f"{base_name}_steps.jsonl"
        self.report_log_file = f"{base_name}_reports.json"
        
        # 初始化基础结构
        self._init_file()
        
        # 用于自动化诊断的状态追踪
        self.history_response_lengths = []
        self.initial_response_length = None
        self.history_pg_loss = []
        
        # 线程锁辅助（在同一进程内双重保障）
        self._thread_lock = threading.Lock()
        logger.info(f"🚀 SmartLogger initialized.\nSteps Tracking: {self.step_log_file}\nReports Tracking: {self.report_log_file}")

    def _init_file(self):
        """初始化轻量级报告文件，如果文件不存在则创建。"""
        if not os.path.exists(self.report_log_file):
            with open(self.report_log_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _safe_write(self, step_data, analysis_report=None):
        """
        线程/进程安全的文件写入逻辑。
        使用 fcntl.flock 实现严格的排他锁 (Exclusive Lock)，防止分布式或多进程环境下的写入覆盖。
        """
        with self._thread_lock:
            try:
                # 1. O(1) 极速追加 Step 详情至 JSONL，无需读取旧文件
                if step_data:
                    with open(self.step_log_file, "a", encoding="utf-8") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        try:
                            f.write(json.dumps(step_data, ensure_ascii=False) + "\n")
                        finally:
                            fcntl.flock(f, fcntl.LOCK_UN)
                
                # 2. 报告文件较小，保留前置插入 (Prepend) 逻辑
                if analysis_report:
                    with open(self.report_log_file, "r+", encoding="utf-8") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        try:
                            # 读取现有内容
                            content = json.load(f)
                            # Prepend analysis reports
                            content.insert(0, analysis_report)
                            # 覆写文件
                            f.seek(0)
                            json.dump(content, f, ensure_ascii=False, indent=2)
                            f.truncate()
                        finally:
                            # 释放文件锁
                            fcntl.flock(f, fcntl.LOCK_UN)
            except Exception as e:
                logger.error(f"[SmartLogger] Critical error during safe write: {e}")

    def clean_multimodal_tokens(self, text: str) -> str:
        """精准剔除视觉占位符 (如 <image>, <patch_001> 等) 并将换行符转义为字面量"""
        if not isinstance(text, str):
            text = str(text)
        # 正则清理
        text = re.sub(r'<image>|<patch_\d+>', '', text)
        # 将物理换行替换为字面量 \n
        text = text.replace('\n', '\\n')
        return text

    def process_trajectories(self, trajectories, is_agent=True):
        """
        处理轨迹数据：进行去重、视觉 Token 过滤以及角色拆分。
        """
        processed_list = []
        if not trajectories:
            return processed_list

        for traj in trajectories:
            parsed_traj = {"task_id": getattr(traj, "task_id", "unknown"), "steps": []}
            
            if is_agent and hasattr(traj, "steps"):
                # Agent 轨迹上下文去重：仅依赖记录好的 step 对象，避免基于 raw text 导致重叠爆炸
                for step_idx, step in enumerate(traj.steps):
                    action_text = self.clean_multimodal_tokens(getattr(step, 'action', ''))
                    obs_text = self.clean_multimodal_tokens(getattr(step, 'observation', ''))
                    
                    parsed_traj["steps"].append({
                        "step_order": step_idx,
                        "Agent": action_text,
                        "Environment": obs_text,
                        "Reward": getattr(step, 'reward', 0.0),
                        "Is_Terminal": getattr(step, 'done', False)
                    })
            else:
                # 非 Agent 或普通文本生成
                raw_text = getattr(traj, "text", str(traj))
                parsed_traj["steps"].append({"content": self.clean_multimodal_tokens(raw_text)})
                
            processed_list.append(parsed_traj)
            
        return processed_list

    def _check_grpo_alignment(self, batch):
        """
        严格对齐检查：验证在 Step 之外（不应该加奖励的特殊 Token 或 Padding 上）是否被错误地赋予了奖励/优势。
        """
        if not batch or "response_mask" not in batch.batch:
            return "No batch or mask provided."
            
        try:
            mask = batch.batch["response_mask"].bool()
            rewards = batch.batch.get("token_level_rewards", None)
            advantages = batch.batch.get("advantages", None)
            
            issues = []
            if rewards is not None:
                invalid_rewards = rewards[~mask]
                if (invalid_rewards != 0).any():
                    issues.append("CRITICAL: Found non-zero rewards on padding/masked tokens.")
            if advantages is not None:
                invalid_advs = advantages[~mask]
                if (invalid_advs != 0).any():
                    issues.append("CRITICAL: Found non-zero advantages on padding/masked tokens.")
                    
            return "Alignment Passed" if not issues else " | ".join(issues)
        except Exception as e:
            return f"Alignment check failed: {str(e)}"

    def _extract_grpo_stats(self, batch):
        """
        基于 GRPO 分组统计当前 Batch 的成功率及优势聚合信息。
        """
        stats = {"success_rate": 0.0, "total_trajectories": 0, "alignment_check": "N/A"}
        if not batch:
            return stats
            
        try:
            stats["alignment_check"] = self._check_grpo_alignment(batch)
            
            # 正确率统计：基于 outcome_reward_tensor 或 token_level_rewards
            rewards = batch.batch.get("outcome_reward_tensor", batch.batch.get("token_level_rewards", None))
            if rewards is not None:
                # 按照行求和得到整条轨迹的总分
                traj_scores = rewards.sum(dim=-1).cpu().numpy()
                stats["total_trajectories"] = len(traj_scores)
                # 大于 0 视为触发了成功奖励
                success_count = (traj_scores > 0).sum()
                stats["success_rate"] = float(success_count) / len(traj_scores) if len(traj_scores) > 0 else 0.0
                
        except Exception as e:
            logger.warning(f"[SmartLogger] Failed to extract GRPO stats: {e}")
            
        return stats

    def run_diagnostics(self, step, metrics):
        """
        自动化诊断逻辑与风险预警。
        分析指标突变，识别 Length Collapse, Mode Collapse, Policy Collapse 和 Divergence。
        """
        report = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 提取核心指标
        resp_len = metrics.get("response_length/mean", 0.0)
        entropy = metrics.get("actor/entropy_loss", 1.0)
        grad_norm = metrics.get("actor/grad_norm", 0.0)
        pg_loss = metrics.get("actor/pg_loss", 0.0)
        ppo_kl = metrics.get("actor/ppo_kl", 0.0)
        
        # 记录历史状态
        if resp_len > 0:
            if self.initial_response_length is None:
                self.initial_response_length = resp_len
            self.history_response_lengths.append(resp_len)
            if len(self.history_response_lengths) > 5:
                self.history_response_lengths.pop(0)
                
        self.history_pg_loss.append(pg_loss)
        if len(self.history_pg_loss) > 5:
            self.history_pg_loss.pop(0)

        # ⚠️ Length Collapse 诊断
        if len(self.history_response_lengths) == 5:
            avg_last_5 = np.mean(self.history_response_lengths)
            if resp_len < avg_last_5 * 0.6 or (self.initial_response_length and resp_len < self.initial_response_length * 0.5):
                report.append(f"[CRITICAL] Response length collapsed to {resp_len:.1f}. Potential Reward Hacking detected: Agent might be 'shutting up' to avoid penalties.")

        # ⚠️ Mode Collapse 诊断
        if entropy < 0.05:
            report.append(f"[WARNING] Mode Collapse Risk: actor/entropy_loss ({entropy:.4f}) is dangerously low. Model is losing output diversity.")

        # ⚠️ Policy Collapse 诊断
        if grad_norm > 30.0:
            report.append(f"[CRITICAL] Policy Collapse Risk: actor/grad_norm ({grad_norm:.2f}) exploded (> 30).")
        if len(self.history_pg_loss) == 5:
            loss_variance = np.var(self.history_pg_loss)
            if loss_variance > 10.0:  # 阈值可调
                report.append(f"[WARNING] Unstable Policy: High pg_loss variance ({loss_variance:.2f}) over last 5 steps.")

        # ⚠️ Divergence 诊断
        if ppo_kl > 0.15:
            report.append(f"[CRITICAL] Divergence Risk: actor/ppo_kl ({ppo_kl:.4f}) exceeded 0.15. Model is deviating severely from reference policy; gibberish generation likely.")

        if not report:
            return None
            
        return {
            "timestamp": timestamp,
            "step": step,
            "risks_detected": report
        }

    def log_step(self, step, metrics, batch=None, trajectories=None, is_agent=True):
        """
        外部调用的主入口，收集当前步的数据并写入系统。
        """
        # 1. 提取需关注的 Core Metrics
        core_metrics = {k: v for k, v in metrics.items() if k in [
            "actor/pg_loss", "actor/entropy_loss", "actor/ppo_kl", 
            "actor/grad_norm", "critic/score/mean", "response_length/mean"
        ]}
        
        # 2. 轨迹处理
        clean_trajectories = self.process_trajectories(trajectories, is_agent=is_agent)
        
        # 3. GRPO 分组统计与细粒度奖励校验
        grpo_stats = self._extract_grpo_stats(batch)
        
        # 4. 组装本步写入数据
        step_data = {
            "step": step,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": core_metrics,
            "grpo_stats": grpo_stats,
            "trajectories": clean_trajectories
        }
        
        # 5. 执行自动化诊断
        analysis_report = self.run_diagnostics(step, metrics)
        
        # 6. 安全执行磁盘写入
        self._safe_write(step_data=step_data, analysis_report=analysis_report)