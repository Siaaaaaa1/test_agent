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
    Updated for SERIAL execution per model (Limit = 1).
    """

    # --- [修改 1] 将 limit 全部设为 1，实现单模型串行 ---
    # 格式: "模型名": {"limit": 1, "max_input_tokens": ...}
    MODEL_CONFIGS = {
        "HY-Qwen3-235B-A22B-Instruct-2507": {"limit": 40, "max_input_tokens": 32768},
        # "DeepSeek-R1-Online":               {"limit": 1, "max_input_tokens": None},
        "DeepSeek-V3-Online":           {"limit": 40, "max_input_tokens": 32768},
    }

    # --- [修改 2] 使用 threading.Lock 替代 Semaphore ---
    # Lock 是实现串行化（互斥）最标准的原语
    _model_locks: Dict[str, threading.Lock] = {
        name: threading.Lock() 
        for name, conf in MODEL_CONFIGS.items()
    }

    def __init__(self, api_key: Optional[str] = None, model_name: str = "auto-balanced", 
                 temperature: float = 0.7, max_tokens: int = 2048):
        
        if load_dotenv:
            load_dotenv()

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy_key"
        self.model_name = model_name 
        self.temperature = temperature
        self.max_tokens = 2048
        self.base_url = os.getenv("AZURE_PROXY_URL") or "http://ichatproxy.devops.weread.woa.com"
        
        self.headers = {
            "Content-Type": "application/json"
        }

    def set_model(self, model_name: str):
        self.model_name = model_name

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        简单估算 Token 数。
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            total_chars += len(content)
        return int(total_chars / 2.5)

    # --- [修改 3] 返回类型改为 Lock ---
    def _schedule_model(self, messages: list[dict[str, str]]) -> tuple[str, threading.Lock]:
        """
        根据输入消息和锁的状态，选择一个最佳模型。
        优先选择当前未被锁定的模型（即非阻塞）。
        """
        input_tokens = self._estimate_tokens(messages)
        candidates = []

        # 1. 筛选符合 Token 限制的模型
        for name, config in self.MODEL_CONFIGS.items():
            limit = config["max_input_tokens"]
            if limit is not None and input_tokens > limit:
                continue 
            candidates.append(name)
        
        if not candidates:
            # 降级策略
            fallback = "DeepSeek-V3-Online-64K"
            return fallback, self._model_locks[fallback]

        # 2. 调度策略：优先使用未被占用的模型 (Try-Lock Strategy)
        random.shuffle(candidates)

        for name in candidates:
            lock = self._model_locks[name]
            # --- [修改 4] 尝试非阻塞获取锁 ---
            if lock.acquire(blocking=False):
                # 获取成功说明该模型空闲，立即释放（因为实际锁定要在请求处理时进行）
                lock.release()
                return name, lock

        # 3. 如果所有模型都忙，随机选择一个符合条件的模型进行排队（阻塞）
        selected_model = random.choice(candidates)
        return selected_model, self._model_locks[selected_model]

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
        
        # 获取选定的模型和对应的锁
        selected_model, selected_lock = self._schedule_model(messages)
        
        params = {
            "model": selected_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
            **kwargs
        }
        
        try:
            # --- [修改 5] 传递 Lock 对象 ---
            if stream:
                return self._handle_stream_response(url, params, selected_lock)
            else:
                return self._handle_normal_response(url, params, selected_lock)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return "" if not stream else (x for x in [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return "" if not stream else (x for x in [])
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            return "" if not stream else (x for x in [])

    # --- [修改 6] 参数类型改为 threading.Lock ---
    def _handle_normal_response(self, url: str, params: dict, lock: threading.Lock) -> str:
        """
        Handles the non-streaming (normal) response.
        """
        no_proxy = {"http": None, "https": None}

        # 使用 with lock 确保请求期间互斥
        with lock:
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

    # --- [修改 7] 参数类型改为 threading.Lock ---
    def _handle_stream_response(self, url: str, params: dict, lock: threading.Lock) -> Generator[str, None, None]:
        """
        Handles the streaming response.
        """
        no_proxy = {"http": None, "https": None}

        # 获取锁：在此处阻塞，直到轮到该请求执行
        # 注意：对于生成器，这个代码块在用户开始迭代（调用 next）时才会执行
        lock.acquire()
        
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
            # 无论流是否正常结束（或调用方提前 break），都必须释放锁
            lock.release()

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