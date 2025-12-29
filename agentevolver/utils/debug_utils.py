# agentevolver/utils/debug_utils.py

import os
import json
import time
import threading
from typing import Any, Dict

# 尝试导入 OmegaConf 以处理配置对象的序列化
try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except ImportError:
    DictConfig = ListConfig = OmegaConf = None

_log_lock = threading.Lock()

def _json_serializer(obj):
    """
    辅助函数：处理 json.dumps 无法默认序列化的对象
    """
    # 处理 OmegaConf 配置对象
    if OmegaConf is not None and isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    
    # 处理其他可能的非基本类型（兜底方案，转为字符串）
    try:
        return str(obj)
    except Exception:
        return "<Unserializable Object>"

def debug_log(config: Any, log_name: str, data: Dict[str, Any]):
    """
    将调试信息记录到 ./logs/{log_name}_{date}.jsonl
    
    Args:
        config: 配置对象 (DictConfig 或 dict)，需包含 debug_log 开关
        log_name: 日志文件名标识 (如 'api_gen_intra')
        data: 要记录的数据字典
    """
    # 1. 检查配置开关
    enabled = False
    try:
        if isinstance(config, dict):
            enabled = config.get("debug_log", False)
        else:
            # 假设是 OmegaConf 或类似对象
            enabled = getattr(config, "debug_log", False)
    except Exception:
        pass

    if not enabled:
        return

    # 2. 准备目录和文件名
    log_dir = "./logs"
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"[DebugLog Error] Failed to create log dir: {e}")
        return
    
    date_str = time.strftime("%Y-%m-%d")
    filename = os.path.join(log_dir, f"{log_name}_{date_str}.jsonl")
    
    # 3. 构造日志条目
    entry = {
        "timestamp": time.time(),
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        **data
    }
    
    # 4. 线程安全写入
    with _log_lock:
        try:
            with open(filename, "a", encoding="utf-8") as f:
                # 关键修复：使用 default=_json_serializer 处理 DictConfig
                f.write(json.dumps(entry, default=_json_serializer, ensure_ascii=False) + "\n")
        except Exception as e:
            # 打印错误但不中断程序，防止日志系统搞挂主流程
            print(f"[DebugLog Error] Write failed: {e}")