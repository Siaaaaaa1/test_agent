from abc import ABC, abstractmethod
import json
import os
import time
import threading
from typing import Any, Optional, Protocol, Iterator, Generator, cast, Dict, List
from collections import defaultdict

from loguru import logger
import requests

# Try to import load_dotenv to read from .env file
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

class Mix_DashScopeClient:
    """
    Unified DashScopeClient supporting dynamic model selection per request.
    
    Features:
    - Auto-routing: Selects URL based on model name (Azure vs. Hunyuan/DeepSeek).
    - Isolation: Each model has its own Concurrency Limit (20) and Rate Limit (30 RPM).
    - Dynamic: 'model' can be passed in chat/chat_completion arguments.
    - Statistics: Tracks usage per model and reports every 100 global calls.
    """
    
    # Azure 系模型集合，用于判断 API 路径
    AZURE_MODELS = {"azure-gpt-5-mini", "azure-gpt-5"}
    
    # 统一限制配置
    MAX_CONCURRENCY = 20
    MAX_RPM = 30
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "azure-gpt-5-mini", 
                 temperature: float = 0.7, max_tokens: int = 2048):
        
        if load_dotenv:
            load_dotenv()

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy_key"
        
        # 这里的 model_name 仅作为“如果不传参数时的默认值”
        self.default_model_name = model_name
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = os.getenv("AZURE_PROXY_URL") or "http://ichatproxy.devops.weread.woa.com"
        
        self.headers = {
            "Content-Type": "application/json"
        }

        # 状态存储：{ "model_name": { "semaphore": ..., "rate_lock": ..., "timestamps": [] } }
        self._model_states: Dict[str, Dict[str, Any]] = {}
        self._state_init_lock = threading.Lock()
        
        # --- 统计相关变量 ---
        self._stats_lock = threading.Lock()          
        self._model_call_counts = defaultdict(int)   
        self._total_call_count = 0                   
        
        logger.info(f"Initialized DashScopeClient. Default model: {self.default_model_name}")

    def set_default_model(self, model_name: str):
        """
        Update the default model name.
        """
        self.default_model_name = model_name

    def _get_model_state(self, model_name: str) -> Dict[str, Any]:
        """
        动态获取指定模型的并发/限流状态对象。如果该模型是第一次被调用，则自动初始化其状态。
        """
        if model_name not in self._model_states:
            with self._state_init_lock:
                if model_name not in self._model_states:
                    self._model_states[model_name] = {
                        # 每个模型独立的并发控制 (20)
                        "semaphore": threading.BoundedSemaphore(self.MAX_CONCURRENCY),
                        # 每个模型独立的速率限制锁
                        "rate_lock": threading.Lock(),
                        # 每个模型独立的请求时间戳记录
                        "timestamps": []
                    }
        return self._model_states[model_name]

    def _wait_for_rate_limit(self, model_name: str):
        """
        针对特定模型进行速率限制检查 (30 RPM)。
        """
        state = self._get_model_state(model_name)
        lock = state["rate_lock"]
        timestamps = state["timestamps"]
        window_duration = 60.0  # 60 seconds
        
        tid = threading.get_ident() # [LOG] 获取线程ID

        while True:
            wait_time = 0
            with lock:
                now = time.time()
                # 移除滑窗外的时间戳
                valid_timestamps = [t for t in timestamps if now - t < window_duration]
                timestamps[:] = valid_timestamps
                
                # 检查当前窗口内的请求数
                if len(timestamps) < self.MAX_RPM:
                    timestamps.append(now)
                    return  # 成功获取限额，直接返回
                
                # 计算等待时间 (取最早的一个时间戳计算剩余时间)
                if timestamps:
                    oldest_timestamp = timestamps[0]
                    wait_time = window_duration - (now - oldest_timestamp)
                else:
                    # 理论上不会走到这里，除非 MAX_RPM <= 0
                    wait_time = 1.0
            
            # [重要] 在锁外部休眠，避免阻塞其他线程
            if wait_time > 0:
                # [LOG] 打印限流等待信息
                logger.warning(f"[{tid}] Rate limit hit for {model_name} ({len(timestamps)}/{self.MAX_RPM}). Sleeping {wait_time:.2f}s...")
                time.sleep(wait_time + 0.05) #稍微多睡一点，避免边界误差

    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any] = None, **kwargs) -> str:
        """
        Wrapper for chat_completion. 
        """
        params = sampling_params.copy() if sampling_params else {}
        params.update(kwargs)
        
        result = self.chat_completion(messages, stream=False, **params)
        
        if isinstance(result, str):
            return result
        return ""

    def chat_stream(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> Generator[str, None, None]:
        return self.chat_stream_with_retry(messages, **sampling_params)

    def chat_completion(self, messages: list[dict[str, str]], stream: bool = False, **kwargs) -> str | Generator[str, None, None]:
        """
        Sends a request to the chat completion API.
        Dynamically selects URL and Resource Locks based on the 'model' parameter.
        """
        base = self.base_url.rstrip('/')
        
        # 优先从参数获取 model，否则使用默认值
        target_model = kwargs.get("model", self.default_model_name)
        
        # --- 记录调用与统计 ---
        report_stats = False
        stats_snapshot = None
        current_step = 0

        with self._stats_lock:
            self._model_call_counts[target_model] += 1
            self._total_call_count += 1
            current_step = self._total_call_count      
            # 检查是否满足汇报条件 (每 100 次)
            if self._total_call_count % 100 == 0:
                report_stats = True
                stats_snapshot = dict(self._model_call_counts)

        if report_stats and stats_snapshot:
            logger.info(f"====== Model Usage Statistics [Step {current_step}] ======")
            sorted_stats = sorted(stats_snapshot.items(), key=lambda item: item[1], reverse=True)
            for model, count in sorted_stats:
                logger.info(f"Model '{model}': {count} calls")
            logger.info("========================================================")
        # --------------------

        # 1. 根据 model 名称动态选择 URL
        if target_model in self.AZURE_MODELS:
            url = f"{base}/api/chat_completions?source=emoji_agent_research"
            params = {
            "model": target_model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        else:
            url = f"{base}/hunyuan/deepseek/chat_completions?source=exp"
            params = {
            "model": target_model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
            **kwargs
            }
        
        # 2. 获取该特定模型的状态对象
        state = self._get_model_state(target_model)
        semaphore = state["semaphore"]

        # 3. 分流处理：流式与非流式
        # [关键修改] 分离逻辑以确保流式 generator 在生命周期内持有锁
        if stream:
            return self._stream_with_lock(url, params, semaphore, target_model)
        else:
            return self._normal_with_lock(url, params, semaphore, target_model)

    def _normal_with_lock(self, url, params, semaphore, model_name) -> str:
        """
        非流式请求处理：获取锁 -> 限流等待 -> 请求 -> 释放锁
        """
        tid = threading.get_ident() # [LOG]
        try:
            # [LOG]
            logger.info(f"[{tid}] [Normal] Waiting for semaphore for {model_name}...")
            with semaphore:
                # [LOG]
                logger.info(f"[{tid}] [Normal] Acquired semaphore. Checking rate limit...")
                self._wait_for_rate_limit(model_name)
                # [LOG]
                logger.info(f"[{tid}] [Normal] Rate limit passed. Sending request...")
                return self._handle_normal_response(url, params)
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for model {model_name}: {e}")
            return ""
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response for model {model_name}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in API call for model {model_name}: {e}")
            return ""

    def _stream_with_lock(self, url, params, semaphore, model_name) -> Generator[str, None, None]:
        """
        流式请求处理：
        生成器持续持有锁，直到迭代完成或异常中断。
        """
        tid = threading.get_ident() # [LOG]
        try:
            # [LOG]
            logger.info(f"[{tid}] [Stream] Waiting for semaphore for {model_name}...")
            with semaphore:
                # [LOG]
                logger.info(f"[{tid}] [Stream] Acquired semaphore. Checking rate limit...")
                self._wait_for_rate_limit(model_name)
                # [LOG]
                logger.info(f"[{tid}] [Stream] Rate limit passed. Sending request...")
                # 使用 yield from 将底层生成器的数据透传出来
                yield from self._handle_stream_response(url, params)
                # [LOG]
                logger.info(f"[{tid}] [Stream] Finished successfully (Releasing Lock).")
        except requests.exceptions.RequestException as e:
            logger.error(f"Stream API request failed for model {model_name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in stream call for model {model_name}: {e}")

    def _handle_normal_response(self, url: str, params: dict) -> str:
        no_proxy = {"http": None, "https": None}
        
        response = requests.post(
            url, 
            headers=self.headers, 
            json=params, 
            timeout=300, 
            proxies=no_proxy 
        )
        
        if not response.ok:
            self._raise_for_error(response)
        
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"Unexpected response format: {result}")
            return ""

    def _handle_stream_response(self, url: str, params: dict) -> Generator[str, None, None]:
        no_proxy = {"http": None, "https": None}
        tid = threading.get_ident() # [LOG]

        # [LOG] 打印请求发起前
        logger.debug(f"[{tid}] Calling requests.post(stream=True) to {url}...")
        
        response = requests.post(
            url, 
            headers=self.headers, 
            json=params, 
            stream=True, 
            timeout=600,
            proxies=no_proxy
        )
        
        # [LOG] 打印请求发起后
        logger.debug(f"[{tid}] Connection established. Status: {response.status_code}")

        if not response.ok:
            self._raise_for_error(response)
        
        chunk_idx = 0
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
                                    # [LOG] 仅打印首包，避免日志爆炸
                                    if chunk_idx == 0:
                                        logger.debug(f"[{tid}] Received First Chunk.")
                                    chunk_idx += 1
                                    
                                    yield content
                    except json.JSONDecodeError:
                        continue

    def _raise_for_error(self, response):
        try:
            error_json = response.json().get('error', {})
            message = error_json.get('message', '') if isinstance(error_json, dict) else str(error_json)
            if "inappropriate content" in message:
                raise LlmException("inappropriate content")
            if "limit" in message:
                raise LlmException("hit limit")
        except LlmException:
            raise
        except:
            logger.error(f"API request failed: {response.status_code} {response.text}")
            response.raise_for_status()

    def chat_with_retry(self, messages: list[dict[str, str]], max_retries: int = 3, 
                       retry_delay: float = 1.0, **kwargs) -> str:
        """
        Supports passing 'model' in kwargs for retry logic.
        """
        for attempt in range(max_retries):
            try:
                # kwargs (including model) are passed down to chat_completion
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
        """
        Supports passing 'model' in kwargs for streaming retry logic.
        """
        tid = threading.get_ident() # [LOG]
        logger.info(f"[{tid}] Entering chat_stream_with_retry...") # [LOG]

        for attempt in range(max_retries):
            try:
                stream_generator = cast(Generator[str, None, None], self.chat_completion(messages, stream=True, **kwargs))
                
                # [LOG] 最关键的卡点检查
                logger.debug(f"[{tid}] Attempt {attempt+1}: Waiting for generator first next()...")
                
                # 尝试获取第一个 chunk，如果失败则触发异常并重试
                first_chunk = next(stream_generator, None)
                
                if first_chunk is not None:
                    # [LOG]
                    logger.debug(f"[{tid}] First chunk received. Yielding stream.")
                    yield first_chunk
                    # 继续产出剩余部分
                    for chunk in stream_generator:
                        yield chunk
                    return
                else:
                    logger.warning(f"[{tid}] Stream returned None (empty) on attempt {attempt+1}")

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