from typing import Any, List, Dict, Union
import torch
from loguru import logger
import json
import re

# apply chat_template to a message, and then convert back to message
def convert_tool_to_user_message(tool_message, format="qwen"):
    assert format == "qwen"

    if tool_message["role"] == "user":
        return tool_message
    elif tool_message["role"] == "tool" and len(tool_message["tool_calls"])>0:
        assert len(tool_message["tool_calls"])==1
        return {
            "role": "user",
            "content": str(tool_message["tool_calls"][0]['result'])
        }


def _trim_to_safe_utf8_boundary(tokens, tokenizer):
    """
    裁掉末尾的裸字节 token，直到边界处于完整 UTF-8 字符处。

    Byte-level BPE（如 Qwen）会把多字节字符拆成若干 <0xXX> token。
    若按 token 数截断时末尾恰好是某个多字节字符的中间字节，
    tokenizer.decode() 的 Rust 层会因收到残缺 UTF-8 而 panic（无法被 try/except 捕获）。
    最多退 4 个 token（UTF-8 最大字节数），代价极小。
    """
    MAX_TRIM = 4
    for trim in range(MAX_TRIM + 1):
        candidate = tokens if trim == 0 else tokens[:-trim]
        if len(candidate) == 0:
            break
        last_token_str = tokenizer.convert_ids_to_tokens([candidate[-1].item()])[0]
        # 裸字节 token 形如 <0xe4>、<0xb8> 等
        if not (last_token_str.startswith("<0x") and last_token_str.endswith(">")):
            return candidate
    return tokens[:-MAX_TRIM] if len(tokens) > MAX_TRIM else tokens[:1]


def clip_state_content_correctly(tokenizer, state_content: str, max_env_len: int) -> str:
    """
    Correctly truncate state_content, ensuring token boundaries are not broken

    Args:
        tokenizer: Tokenizer
        state_content: Content to be truncated
        max_env_len: Maximum allowed token length

    Returns:
        Truncated content string
    """
    # First tokenize to check length
    tokens = tokenizer(state_content, return_tensors="pt", padding=False)["input_ids"][0]

    if len(tokens) <= max_env_len:
        return state_content

    # Truncate to max_env_len, then trim back to a safe UTF-8 character boundary
    # to prevent byte-level BPE partial-character panic in the Rust tokenizer
    truncated_tokens = _trim_to_safe_utf8_boundary(tokens[:max_env_len], tokenizer)

    if hasattr(tokenizer, 'decode'):
        try:
            truncated_content = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
            return truncated_content
        except:
            try:
                truncated_content = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                return truncated_content
            except:
                pass

    # Fallback: gradually reduce token count
    for i in range(min(10, max_env_len)):
        try:
            test_tokens = tokens[:max_env_len - i]
            truncated_content = tokenizer.decode(test_tokens, skip_special_tokens=False)
            logger.warning(f"Had to reduce token count by {i} to successfully decode")
            return truncated_content
        except:
            continue

    logger.error("All token-based truncation methods failed, falling back to character truncation")
    return state_content[:max_env_len]


def get_batched_exponential_decay_weights_vectorized(
    lens: list[int],
    start_val: float = 10.0,
    end_val: float = 1.0,
    decay_reach_percent: float = 0.85,
    padding_value: float = 0.0,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Efficiently generate exponential decay weights for a batch of lengths in one go.
    This version is fully vectorized, avoiding Python loops.

    Args:
        lens (list[int]): A list containing multiple lengths.
        start_val (float): Weight value at position 0.
        end_val (float): The final value that weights decay towards.
        decay_reach_percent (float): The point where weights decay close to the final value, as a percentage of total length.
        padding_value (float): Value used to pad invalid positions.
        device: The desired device of the output tensor.

    Returns:
        torch.Tensor: A 2D weight tensor with shape (len(lens), max(lens)).
    """
    if not lens:
        return torch.empty(0, 0, device=device)

    # 1. Preparation: Get batch size, maximum length, and convert lens to tensor
    batch_size = len(lens)
    max_len = max(lens)
    lens_tensor = torch.tensor(lens, dtype=torch.float32, device=device)

    # 2. Vectorized computation of decay rate `decay_rate` for each sequence
    # Note: Each variable here is a vector with length batch_size
    amplitude = start_val - end_val
    # Subtract 1 to get the correct index range [0, length-1]
    # Use clamp(min=1) to avoid division by zero when length is 1
    decay_end_index = (lens_tensor - 1).clamp(min=1) * decay_reach_percent

    # decay_rate is a tensor with shape (batch_size,)
    decay_rate = -torch.log(torch.tensor(0.01, device=device)) / decay_end_index

    # 3. Create 2D indices and decay_rate to leverage broadcasting mechanism
    # indices shape: (max_len,) -> [0, 1, ..., max_len-1]
    indices = torch.arange(max_len, device=device)

    # decay_rate shape: (batch_size,) -> (batch_size, 1)
    # This allows it to broadcast with indices [max_len,] resulting in shape (batch_size, max_len)
    exponent = -decay_rate.unsqueeze(1) * indices

    # 4. Compute all weights at once (this is a complete BxL matrix)
    calculated_weights = amplitude * torch.exp(exponent) + end_val

    # 5. Create mask to set values at invalid positions to padding_value
    # mask shape: (batch_size, max_len)
    mask = indices < lens_tensor.unsqueeze(1)

    # 6. Apply mask to keep only weight values within valid length
    final_weights = torch.where(mask, calculated_weights, torch.tensor(padding_value, device=device))
    
    return final_weights


def extract_json_from_str(content: str) -> Union[Dict, List, Any]:
    """
    从字符串中提取 JSON 对象。
    支持提取 Markdown 代码块中的 JSON (```json ... ```) 或直接查找最外层的 {}/[]。
    """
    if not content:
        return {}

    content = content.strip()

    # 1. 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. 尝试提取 Markdown 代码块 (```json ... ```)
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 尝试提取通用代码块 (``` ... ```)
    pattern = r"```\s*([\s\S]*?)\s*```"
    match = re.search(pattern, content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 4. 尝试寻找最外层的 '{' 和 '}' 或 '[' 和 ']'
    # 这是一个启发式方法，用于处理 LLM 有时会在 JSON 前后输出闲聊内容的情况
    try:
        start_brace = content.find('{')
        start_bracket = content.find('[')
        
        start = -1
        end = -1
        
        # 确定是对象还是数组，优先匹配出现较早的
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            start = start_brace
            end = content.rfind('}')
        elif start_bracket != -1:
            start = start_bracket
            end = content.rfind(']')
            
        if start != -1 and end != -1 and end > start:
            json_str = content[start : end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    logger.warning(f"Failed to extract JSON from string: {content[:100]}...")
    return {}

def seed_everything(seed: int = 42):
    """
    设置全局随机种子，确保实验可复现。
    涵盖 Python, Numpy, PyTorch (CPU & GPU).
    """
    import random
    import numpy as np
    import os
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")