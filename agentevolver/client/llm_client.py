from abc import ABC, abstractmethod
import json
import os
import time
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
    Modified DashScopeClient to support internal Azure Proxy.
    Class name and method signatures are preserved.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen-plus", 
                 temperature: float = 0.7, max_tokens: int = 2048):
        # ----------------------------------------------------------------------
        # Modification: Implementation changed for new Azure Proxy API
        # ----------------------------------------------------------------------
        
        if load_dotenv:
            load_dotenv()

        # 1. API Key is not required by the new curl, but we keep the interface.
        # We store it if provided, but won't raise error if missing.
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy_key"
        
        # 2. Model Name
        # User defaults to qwen-plus in signature, but new API uses azure-gpt-5.
        # We respect the passed arg, but checking env is good practice.
        self.model_name = os.getenv("LLM_MODEL") or os.getenv("MODEL_NAME") or model_name
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 3. Base URL - Hardcoded to the new proxy address provided
        # The curl URL is http://ichatproxy.devops.weread.woa.com/api/chat_completions?source=exp
        # We set the base part here.
        self.base_url = os.getenv("AZURE_PROXY_URL") or "http://ichatproxy.devops.weread.woa.com"
        
        # 4. Headers - Removed Authorization as per curl command
        self.headers = {
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized DashScopeClient (Proxy) with model: {self.model_name}, base_url: {self.base_url}")
    
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
        Modified to use the specific path and query parameters for the new Proxy API.
        """
        base = self.base_url.rstrip('/')
        
        # Modification: Change URL path and add query parameter ?source=exp
        url = f"{base}/api/chat_completions?source=exp"
        
        # Merge parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            # Note: The new API might or might not strictly support these at top level,
            # but we keep them as per standard OpenAI/DashScope format usually forwarded by proxies.
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
        # Note: headers=self.headers no longer contains Authorization
        response = requests.post(url, headers=self.headers, json=params, timeout=600)
        
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

    def _handle_stream_response(self, url: str, params: dict) -> Generator[str, None, None]:
        """
        Handles the streaming response.
        """
        response = requests.post(url, headers=self.headers, json=params, stream=True, timeout=600)
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