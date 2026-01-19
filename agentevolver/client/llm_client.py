from abc import ABC, abstractmethod
import json
import os
import time
import threading
import random
from typing import Any, Optional, Protocol, Iterator, Generator, cast, List, Dict

from loguru import logger
import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

class LlmException(Exception):
    def __init__(self, typ: str):
        self._type = typ
    
    @property
    def typ(self):
        return self._type

class DashScopeClient:
    """
    Modified DashScopeClient to support dynamic scheduling for internal Hunyuan/DeepSeek Proxy.
    Supports load balancing and constraints (Token limit & Concurrency).
    """

    # --- [修改 1] 定义模型池、并发限制和约束 ---
    # 格式: "模型名": {"limit": 并发数, "max_input_tokens": 最大输入Token限制(None表示无限制)}
    MODEL_CONFIGS = {
        "HY-Qwen3-235B-A22B-Instruct-2507": {"limit": 5, "max_input_tokens": 30000},
        "DeepSeek-R1-Online":               {"limit": 3, "max_input_tokens": None},
        "DeepSeek-V3-Online-64K":           {"limit": 3, "max_input_tokens": None},
    }

    # 为每个模型初始化独立的信号量
    _model_semaphores: Dict[str, threading.BoundedSemaphore] = {
        name: threading.BoundedSemaphore(value=conf["limit"]) 
        for name, conf in MODEL_CONFIGS.items()
    }

    def __init__(self, api_key: Optional[str] = None, model_name: str = "auto-balanced", 
                 temperature: float = 0.7, max_tokens: int = 2048):
        
        if load_dotenv:
            load_dotenv()

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy_key"
        # 默认模式下，我们使用动态调度，因此 self.model_name 只是一个初始值
        # 实际请求时会由 _schedule_model 覆盖
        self.model_name = model_name 
        self.temperature = temperature
        self.max_tokens = 2048
        self.base_url = os.getenv("AZURE_PROXY_URL") or "http://ichatproxy.devops.weread.woa.com"
        
        self.headers = {
            "Content-Type": "application/json"
        }

    def set_model(self, model_name: str):
        self.model_name = model_name

    # --- [新增] Token 简单估算 ---
    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        简单估算 Token 数。这里使用 1 token ≈ 3 chars 的保守估计（中文环境）。
        如果需要更精准，可根据实际 tokenizer 调整。
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            total_chars += len(content)
        # 英文通常 4 chars/token，中文 1-2 chars/token。
        # 取平均 2.5 chars/token 估算，或者简单按字符数处理。
        # 这里为了安全起见（确保 < 3w），我们可以估算得稍微大一点
        return int(total_chars / 2.5)

    # --- [新增] 动态调度逻辑 ---
    def _schedule_model(self, messages: list[dict[str, str]]) -> tuple[str, threading.BoundedSemaphore]:
        """
        根据输入消息和并发状态，选择一个最佳模型。
        返回: (选中的模型名称, 对应的信号量)
        """
        input_tokens = self._estimate_tokens(messages)
        candidates = []

        # 1. 筛选符合 Token 限制的模型
        for name, config in self.MODEL_CONFIGS.items():
            limit = config["max_input_tokens"]
            if limit is not None and input_tokens > limit:
                continue # 超出限制，跳过
            candidates.append(name)
        
        if not candidates:
            # 如果没有符合条件的模型（极其罕见，除非全部模型都有严格限制且输入超长）
            # 降级策略：默认返回 DeepSeek-V3
            fallback = "DeepSeek-V3-Online-64K"
            return fallback, self._model_semaphores[fallback]

        # 2. 调度策略：优先使用有空闲并发位的模型 (Try-Lock Strategy)
        # 打乱顺序，防止所有请求都倾向于列表第一个
        random.shuffle(candidates)

        # 尝试寻找一个可以直接获得锁的模型（非阻塞）
        for name in candidates:
            sem = self._model_semaphores[name]
            # 注意：BoundedSemaphore 在 Python 中没有直接公开的 `locked()` 或 `value` 供判断
            # 我们尝试非阻塞获取
            if sem.acquire(blocking=False):
                # 获取成功，立即释放（因为实际获取是在 _handle 方法中进行的 context manager）
                # 这里只是为了探测谁有空闲
                sem.release()
                return name, sem

        # 3. 如果所有模型都忙，随机选择一个符合条件的模型进行排队（阻塞）
        selected_model = random.choice(candidates)
        return selected_model, self._model_semaphores[selected_model]

    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any] = None, **kwargs) -> str:
        params = sampling_params.copy() if sampling_params else {}
        params.update(kwargs)
        result = self.chat_completion(messages, stream=False, **params)
        if isinstance(result, str):
            return result
        return ""

    def chat_stream(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> Generator[str, None, None]:
        return self.chat_stream_with_retry(messages, **sampling_params)

    def chat_completion(self, messages: list[dict[str, str]], stream: bool = False, **kwargs) -> str | Generator[str, None, None]:
        base = self.base_url.rstrip('/')
        url = f"{base}/hunyuan/deepseek/chat_completions?source=exp"
        
        # [修改 2] 调用调度器选择模型和信号量
        selected_model, selected_semaphore = self._schedule_model(messages)
        
        # logger.info(f"Scheduled to model: {selected_model}") # Debug用

        params = {
            "model": selected_model, # 使用调度选出的模型
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
            **kwargs
        }
        
        try:
            # [修改 3] 将选定的信号量传递给处理函数
            if stream:
                return self._handle_stream_response(url, params, selected_semaphore)
            else:
                return self._handle_normal_response(url, params, selected_semaphore)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return "" if not stream else (x for x in [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return "" if not stream else (x for x in [])
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            return "" if not stream else (x for x in [])

    def _handle_normal_response(self, url: str, params: dict, semaphore: threading.BoundedSemaphore) -> str:
        """
        Handles the non-streaming (normal) response.
        """
        no_proxy = {"http": None, "https": None}

        # [修改 4] 使用传入的具体模型的信号量
        with semaphore:
            response = requests.post(
                url, 
                headers=self.headers, 
                json=params, 
                timeout=300, 
                proxies=no_proxy 
            )
            
            if not response.ok:
                try:
                    error_json = response.json().get('error', {})
                    message = error_json.get('message', '') if isinstance(error_json, dict) else str(error_json)
                    if "inappropriate content" in message:
                        raise LlmException("inappropriate content")
                    if "limit" in message:
                        raise LlmException("hit limit")
                except LlmException as e:
                    raise
                except:
                    logger.error(f"API request failed: {response.status_code} {response.text}")
                    response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"Unexpected response format: {result}")
                return ""

    def _handle_stream_response(self, url: str, params: dict, semaphore: threading.BoundedSemaphore) -> Generator[str, None, None]:
        """
        Handles the streaming response.
        """
        no_proxy = {"http": None, "https": None}

        # [修改 5] 手动获取传入的信号量
        semaphore.acquire()
        
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                json=params, 
                stream=True, 
                timeout=600,
                proxies=no_proxy
            )
            
            if not response.ok:
                try:
                    error_json = response.json().get('error', {})
                    message = error_json.get('message', '') if isinstance(error_json, dict) else str(error_json)
                    if "inappropriate content" in message:
                        raise LlmException("inappropriate content")
                    if "limit" in message:
                        raise LlmException("hit limit")
                except LlmException as e:
                    raise
                except:
                    logger.error(f"API request failed: {response.status_code} {response.text}")
                    response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                choice = chunk["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    if content:
                                        yield content
                        except json.JSONDecodeError:
                            continue
        finally:
            # [修改 6] 释放传入的信号量
            semaphore.release()

    def chat_with_retry(self, messages: list[dict[str, str]], max_retries: int = 3, 
                       retry_delay: float = 1.0, **kwargs) -> str:
        for attempt in range(max_retries):
            try:
                result = cast(str, self.chat_completion(messages, stream=False, **kwargs))
                if result:
                    return result
            
            except LlmException as e:
                if e.typ == 'inappropriate content':
                    logger.warning(f"llm return inappropriate content, which is blocked by the remote")
                    return "[inappropriate content]"
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
        
        logger.error(f"All {max_retries} attempts failed")
        return ""

    def chat_stream_with_retry(self, messages: list[dict[str, str]], max_retries: int = 3, 
                              retry_delay: float = 10.0, **kwargs) -> Generator[str, None, None]:
        for attempt in range(max_retries):
            try:
                stream_generator = cast(Generator[str, None, None], self.chat_completion(messages, stream=True, **kwargs))
                first_chunk = next(stream_generator, None)
                if first_chunk is not None:
                    yield first_chunk
                    for chunk in stream_generator:
                        yield chunk
                    return
            except LlmException as e:
                if e.typ == 'inappropriate content':
                    logger.warning(f"llm return inappropriate content, which is blocked by the remote")
                    yield "[inappropriate content]"
                    return
            except Exception as e:
                logger.warning(f"Stream attempt {attempt + 1} failed: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
        
        logger.error(f"All {max_retries} stream attempts failed")
        return