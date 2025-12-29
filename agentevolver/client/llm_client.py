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
    """Aliyun DashScope API Client (Modified for Custom API & .env Support)"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen-plus", 
                 temperature: float = 0.7, max_tokens: int = 2048):
        # ----------------------------------------------------------------------
        # Modification: Load config from .env and support Aliyun API
        # ----------------------------------------------------------------------
        
        # 1. Load environment variables from .env file
        if load_dotenv:
            load_dotenv()
        else:
            logger.warning("python-dotenv not installed. Relying on system environment variables.")

        # 2. Get API Key from environment variables or argument
        # Prioritize DASHSCOPE_API_KEY, then OPENAI_API_KEY (compatible with example.env)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            # Fallback check for keys with quotes in .env if not parsed correctly
            raise ValueError("API key is required. Please set DASHSCOPE_API_KEY or OPENAI_API_KEY in .env or environment.")
        
        # Clean up key if it was loaded with quotes
        if self.api_key.startswith('"') and self.api_key.endswith('"'):
            self.api_key = self.api_key[1:-1]

        # 3. Get Model Name from environment or argument
        # Checks LLM_MODEL or MODEL_NAME in env, otherwise uses the passed default
        self.model_name = os.getenv("LLM_MODEL") or os.getenv("MODEL_NAME") or model_name
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 4. Get Base URL from environment or use Aliyun default
        # Checks OPENAI_BASE_URL (from example.env), DASHSCOPE_BASE_URL, or defaults to Aliyun
        default_aliyun_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or default_aliyun_url
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized DashScopeClient with model: {self.model_name}, base_url: {self.base_url}")
    
    def set_model(self, model_name: str):
        """
        Sets the model name for the DashScopeClient instance.

        Args:
            model_name (str): The name of the model to be used for API interactions.
        """
        self.model_name = model_name

    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any] = None, **kwargs) -> str:
        """
        Sends a chat request to the LLM.
        Modified to support optional sampling_params and additional kwargs (like response_format).
        """
        # 1. 合并参数：优先使用传入的 kwargs，同时兼容旧的 sampling_params
        params = sampling_params.copy() if sampling_params else {}
        params.update(kwargs)
        
        # 2. 直接调用 chat_completion 的非流式模式
        # 这样可以将 response_format 等参数直接透传给 API
        result = self.chat_completion(messages, stream=False, **params)
        
        # 3. 确保返回字符串 (chat_completion 在出错时可能返回 generator 或空字符串)
        if isinstance(result, str):
            return result
        return ""

    def chat_stream(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> Generator[str, None, None]:
        """
        Initiates a streaming chat session and returns a generator that yields the response as it is being generated.

        Args:
            messages (list[dict[str, str]]): A list of message objects, each containing 'role' and 'content'.
            sampling_params (dict[str, Any]): Parameters for controlling the sampling behavior of the model.

        Returns:
            Generator[str, None, None]: A generator that yields the response text as it is being generated.
        """
        return self.chat_stream_with_retry(messages, **sampling_params)

    def chat_completion(self, messages: list[dict[str, str]], stream: bool = False, **kwargs) -> str | Generator[str, None, None]:
        """
        Sends a request to the chat completion API, supporting both non-streaming and streaming modes, and handles various exceptions.

        Args:
            messages (list[dict[str, str]]): A list of message objects, each containing 'role' and 'content'.
            stream (bool, optional): If True, the response will be streamed. Defaults to False.
            **kwargs: Additional parameters to be passed to the API.

        Returns:
            str | Generator[str, None, None]: The full response text if not streaming, or a generator yielding the response text if streaming.
        """
        # Ensure base_url does not end with / to avoid double slashes if needed, though standard usually omits it
        base = self.base_url.rstrip('/')
        url = f"{base}/chat/completions"
        
        # Merge parameters
        params = {
            "model": self.model_name,
            "messages": messages,
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
        Handles the non-streaming (normal) response from the API.

        Args:
            url (str): The URL to which the POST request is sent.
            params (dict): The parameters to be included in the JSON body of the POST request.

        Returns:
            str: The content of the first choice's message in the response, or an empty string if the response format is unexpected.
        """
        response = requests.post(url, headers=self.headers, json=params, timeout=600)
        if not response.ok:
            # check inappropriate content
            try:
                error_json = response.json().get('error', {})
                message = error_json.get('message', '')
                if "inappropriate content" in message:
                    raise LlmException("inappropriate content")
                if "limit" in message:
                    raise LlmException("hit limit")
            except LlmException as e:
                raise
            except:
                logger.error(f"API request failed: {response.text}")
                response.raise_for_status()
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"Unexpected response format: {result}")
            return ""

    def _handle_stream_response(self, url: str, params: dict) -> Generator[str, None, None]:
        """
        Handles the streaming response from a POST request to the specified URL.

        Args:
            url (str): The URL to which the POST request is sent.
            params (dict): The parameters to be sent with the POST request.

        Yields:
            str: The content of the response, if it meets the specified conditions.
        """
        response = requests.post(url, headers=self.headers, json=params, stream=True, timeout=600)
        if not response.ok:
            # check inappropriate content
            try:
                error_json = response.json().get('error', {})
                message = error_json.get('message', '')
                if "inappropriate content" in message:
                    raise LlmException("inappropriate content")
                if "limit" in message:
                    raise LlmException("hit limit")
            except LlmException as e:
                raise
            except:
                logger.error(f"API request failed: {response.text}")
                response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # remove the prefix 'data: '
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
                        continue  # skip the bad line

    def chat_with_retry(self, messages: list[dict[str, str]], max_retries: int = 3, 
                       retry_delay: float = 1.0, **kwargs) -> str:
        """
        Sends a chat completion request to the LLM with a retry mechanism.

        Args:
            messages (list[dict[str, str]]): A list of message dictionaries for the chat.
            max_retries (int, optional): Maximum number of retries. Defaults to 3.
            retry_delay (float, optional): Initial delay between retries in seconds. Defaults to 1.0.
            **kwargs: Additional keyword arguments to be passed to the `chat_completion` method.

        Returns:
            str: The response from the LLM or a predefined message if all attempts fail.
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
        Attempts to establish a streaming chat completion with a retry mechanism.

        Args:
            messages (list[dict[str, str]]): A list of message dictionaries, each containing 'role' and 'content'.
            max_retries (int, optional): The maximum number of retry attempts. Defaults to 3.
            retry_delay (float, optional): The initial delay in seconds before the first retry. Defaults to 10.0.
            **kwargs: Additional keyword arguments to pass to the chat_completion method.

        Yields:
            str: Chunks of the streaming response.
        """
        for attempt in range(max_retries):
            try:
                stream_generator = cast(Generator[str, None, None], self.chat_completion(messages, stream=True, **kwargs))
                # try to fetch the first chunk to verify the connection
                first_chunk = next(stream_generator, None)
                if first_chunk is not None:
                    yield first_chunk
                    # yield the rest chunks
                    for chunk in stream_generator:
                        yield chunk
                    return  # success
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


# demo
if __name__ == "__main__":
    # Ensure .env is loaded if running directly
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Initialize client (will pick up defaults from env if not passed)
    # You can set LLM_MODEL in .env to change the default model
    client = DashScopeClient()
    
    messages = [
        {"role": "user", "content": "Write a poem about Spring."}
    ]
    
    print(f"Using model: {client.model_name}")
    print(f"Using API URL: {client.base_url}")

    # print("=== request ===")
    # response = client.chat_completion(messages)
    # print(response)
    
    print("\n=== streaming ===")
    for chunk in client.chat_completion(messages, stream=True):
        print(chunk, end='', flush=True)
    
    print("\n\n=== streaming with retry ===")
    for chunk in client.chat_stream(messages, {}):
        print(chunk, end='', flush=True)