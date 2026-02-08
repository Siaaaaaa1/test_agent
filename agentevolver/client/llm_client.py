from abc import ABC, abstractmethod
import json
import os
import time
import threading  # [修改 1/4] 引入 threading 模块
from typing import Any, Optional, Protocol, Iterator, Generator, cast

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

class DashScopeClient:
    """
    Modified DashScopeClient to support internal Azure GPT-5 Mini Proxy.
    Updated for emoji_agent_research context.
    With Rate Limiting (40 RPM).
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "azure-gpt-5-mini", 
                 temperature: float = 0.7, max_tokens: int = 2048):
        
        if load_dotenv:
            load_dotenv()

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy_key"
        
        # [修改 2/4] 初始化并发控制信号量，限制为5个并发
        self._semaphore = threading.BoundedSemaphore(5)

        # ----------------------------------------------------------------------
        # [修改 - 新增速率限制 1/3] 初始化速率限制相关的锁和存储
        # ----------------------------------------------------------------------
        self._rate_limit_lock = threading.Lock()
        self._request_timestamps = [] # 用于存储请求发生的时间戳
        self._max_rpm = 40            # 限制每分钟40次请求

        # ----------------------------------------------------------------------
        # Modification 1: Set default model to azure-gpt-5-mini
        # ----------------------------------------------------------------------
        self.model_name = "azure-gpt-5-mini"
        # if model_name:
        #     self.model_name = model_name
        # else:
        #     self.model_name = "azure-gpt-5"
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # ----------------------------------------------------------------------
        # Modification 2: Base URL matches the provided curl command domain
        # ----------------------------------------------------------------------
        self.base_url = os.getenv("AZURE_PROXY_URL") or "http://ichatproxy.devops.weread.woa.com"
        
        self.headers = {
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized DashScopeClient (Proxy) with model: {self.model_name}, base_url: {self.base_url}")
    
    # ----------------------------------------------------------------------
    # [修改 - 新增速率限制 2/3] 实现速率限制检查逻辑
    # ----------------------------------------------------------------------
    def _wait_for_rate_limit(self):
        """
        Blocks until a request token is available ensuring max 40 requests per 60 seconds.
        Uses a sliding window approach.
        """
        window_duration = 60.0  # 60 seconds
        
        while True:
            with self._rate_limit_lock:
                now = time.time()
                
                # 1. 移除滑窗外（超过60秒前）的时间戳
                # 保留所有 (now - t < 60) 的记录
                self._request_timestamps = [t for t in self._request_timestamps if now - t < window_duration]
                
                # 2. 检查当前窗口内的请求数
                if len(self._request_timestamps) < self._max_rpm:
                    # 如果未达到限制，记录当前时间并允许通过
                    self._request_timestamps.append(now)
                    return
                
                # 3. 如果达到限制，计算需要等待的时间
                # 必须等待列表中最早的一个请求过期
                oldest_timestamp = self._request_timestamps[0]
                wait_time = window_duration - (now - oldest_timestamp)
                
            # 在锁外等待，避免阻塞其他线程的检查（虽然逻辑上都会被卡住，但更安全）
            if wait_time > 0:
                # 稍微多睡一点点(0.01s)，确保下次检查时刚好过期
                time.sleep(wait_time + 0.01)

    def set_model(self, model_name: str):
        """
        Sets the model name for the DashScopeClient instance.
        """
        self.model_name = model_name

    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any] = None, **kwargs) -> str:
        """
        Sends a chat request to the LLM.
        """
        params = sampling_params.copy() if sampling_params else {}
        params.update(kwargs)
        
        result = self.chat_completion(messages, stream=False, **params)
        
        if isinstance(result, str):
            return result
        return ""

    def chat_stream(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> Generator[str, None, None]:
        """
        Initiates a streaming chat session.
        """
        return self.chat_stream_with_retry(messages, **sampling_params)

    def chat_completion(self, messages: list[dict[str, str]], stream: bool = False, **kwargs) -> str | Generator[str, None, None]:
        """
        Sends a request to the chat completion API.
        Modified to use the specific path and query parameters for the GPT-5 Proxy API.
        """
        base = self.base_url.rstrip('/')
        
        # ----------------------------------------------------------------------
        # Modification 3: Update path to /api/chat_completions and source to emoji_agent_research
        # ----------------------------------------------------------------------
        url = f"{base}/api/chat_completions?source=emoji_agent_research"
        
        # Merge parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            # "temperature": self.temperature, # Optional: include if API supports it
            # "max_tokens": self.max_tokens,   # Optional: include if API supports it
            "stream": stream,
            **kwargs
        }
        
        try:
            if stream:
                return self._handle_stream_response(url, params)
            else:
                return self._handle_normal_response(url, params)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return "" if not stream else (x for x in [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return "" if not stream else (x for x in [])
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            return "" if not stream else (x for x in [])

    def _handle_normal_response(self, url: str, params: dict) -> str:
        """
        Handles the non-streaming (normal) response.
        """
        # ----------------------------------------------------------------------
        # Modification 4: Explicitly bypass proxies for internal API calls (WOA)
        # ----------------------------------------------------------------------
        no_proxy = {
            "http": None,
            "https": None
        }

        # [修改 3/4] 使用信号量控制普通请求的并发
        with self._semaphore:
            # [修改 - 新增速率限制 3/3] 在实际发送请求前，检查速率限制
            self._wait_for_rate_limit()
            
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
        
        # Handle standard OpenAI-compatible response format
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"Unexpected response format: {result}")
            return ""

    def _handle_stream_response(self, url: str, params: dict) -> Generator[str, None, None]:
        """
        Handles the streaming response.
        """
        no_proxy = {
            "http": None,
            "https": None
        }

        # [修改 4/4] 使用信号量控制流式请求的并发
        with self._semaphore:
            # [修改 - 新增速率限制 3/3] 在实际发送请求前，检查速率限制
            self._wait_for_rate_limit()

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
                                # Handle standard streaming delta
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    if content:
                                        yield content
                        except json.JSONDecodeError:
                            continue

    def chat_with_retry(self, messages: list[dict[str, str]], max_retries: int = 3, 
                       retry_delay: float = 1.0, **kwargs) -> str:
        """
        Sends a chat completion request with retry mechanism.
        """
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
        """
        Attempts to establish a streaming chat completion with retry mechanism.
        """
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