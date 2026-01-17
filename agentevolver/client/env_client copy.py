# env_client.py
from typing import Dict, List, Any

import requests
from loguru import logger


class EnvClient:
    """
    环境客户端，用于与远程环境服务（Environment Service）进行通信。
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        # 初始化客户端，设置基础 URL 和超时时间
        self.base_url = base_url.rstrip("/")
        self.timeout = 300.0

    def _make_request(
        self,
        endpoint: str,
        env_type: str = "default",
        task_id: str = None,
        instance_id: str = None,
        messages: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict:
        """
        处理向指定 API 端点发送 POST 请求的通用方法。

        参数 (Args):
            endpoint (str): 要发送请求的 API 端点路径（例如 "create" 或 "step"）。
            env_type (str, optional): 环境的类型（例如 "appworld", "webshop"）。默认为 "default"。
            task_id (str, optional): 任务 ID，用于标识具体的环境配置（如特定的用户角色）。默认为 None。
            instance_id (str, optional): 实例 ID，用于标识正在运行的具体会话。默认为 None。
            messages (Dict[str, Any], optional): 要发送的消息内容（例如 Agent 的动作或对话历史）。默认为 None。
            params (Dict[str, Any], optional): 额外的请求参数。默认为 None。

        返回 (Returns):
            Dict: 来自 API 的 JSON 响应数据。
        """
        url = f"{self.base_url}/{endpoint}"  # ⭐ 构建完整 URL
        
        # 封装所有请求参数到数据字典中
        data = {
            "env_type": env_type,
            "task_id": task_id,
            "instance_id": instance_id,
            "messages": messages or {},
            "params": params or {},
            **kwargs,
        }
        try:
            # ⭐ 发送 POST 请求
            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()  # 检查 HTTP 错误
            return response.json()       # ⭐ 返回解析后的 JSON
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}, data: {data}")
            raise

    def get_env_profile(
        self, env_type: str, split: str = "train", params: dict | None = None
    ) -> List[str]:
        """
        根据指定的环境类型和数据划分，获取任务 ID 列表。

        参数 (Args):
            env_type (str): 环境的类型。
            split (str, optional): 数据集划分（例如 "train", "test"）。默认为 "train"。
            params (dict | None, optional): 请求的额外参数。默认为 None。

        返回 (Returns):
            List[str]: 一个包含可用任务 ID 的列表（例如 ["Bob", "Alice"]）。
        """
        payload: dict = {"env_type": env_type}
        if params:
            payload["params"] = params
            
        # ⭐ 发送请求获取环境概况
        response = self._make_request(
            endpoint="/get_env_profile", env_type=env_type, params={"split": split}
        )
        logger.debug(f"get_env_profile split: {split}")
        return response["data"]

    def get_tools_info(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """
        获取特定环境实例中的工具信息。

        参数 (Args):
            instance_id (str): 环境实例的 ID。
            messages (Dict, optional): 随请求发送的额外消息。默认为 {}。
            params (Dict, optional): 随请求发送的额外参数。默认为 {}。

        返回 (Returns):
            float: API 响应中的数据部分（注意：原代码标注返回 float，实际可能返回字典或工具列表）。
        """
        response = self._make_request(
            endpoint="get_info",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )
        return response["data"]

    def create_instance(
        self, env_type: str, task_id: str, instance_id: str = None, params: Dict = None
    ) -> dict:
        """
        通过发送请求来创建一个新的环境实例。

        参数 (Args):
            env_type (str): 要创建的环境类型。
            task_id (str): 任务的唯一标识符（环境配置 ID）。
            instance_id (str, optional): 实例的唯一标识符（如果需要手动指定）。默认为 None。
            params (Dict, optional): 环境创建时的额外配置参数。默认为 None。

        返回 (Returns):
            dict: 包含新创建实例信息的字典（通常包含 instance_id 和初始状态）。
        """
        response = self._make_request(  # ⭐ 请求创建实例
            endpoint="create",
            env_type=env_type,
            task_id=task_id,
            instance_id=instance_id,
            params=params,
        )
        return response["data"]

    def step(self, instance_id: str, action: Dict = {}, params: Dict = {}) -> dict:
        """
        向环境 API 发送请求，在指定实例中执行一步操作。

        参数 (Args):
            instance_id (str): 环境实例的 ID。
            action (Dict, optional): 要执行的动作（例如 Agent 的代码或回复）。默认为 {}。
            params (Dict, optional): 动作的额外参数。默认为 {}。

        返回 (Returns):
            dict: 执行步骤后环境返回的数据（通常包含观察结果 Observation）。
        """
        response = self._make_request(  # ⭐ 发送动作并获取反馈
            endpoint="step", instance_id=instance_id, messages=action, params=params
        )
        return response["data"]

    def evaluate(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """
        发送请求以评估指定的环境实例，并返回评估结果。

        参数 (Args):
            instance_id (str): 要评估的环境实例 ID。
            messages (Dict, optional): 评估所需的额外消息。默认为 {}。
            params (Dict, optional): 评估所需的额外参数。默认为 {}。

        返回 (Returns):
            float: 评估结果（通常是分数或成功率）。
        """
        response = self._make_request(  # ⭐ 请求评分
            endpoint="evaluate",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )
        return response["data"]

    def release_instance(self, instance_id: str) -> bool:
        """
        发送请求以释放（关闭）指定的环境实例。

        参数 (Args):
            instance_id (str): 要释放的环境实例 ID。

        返回 (Returns):
            bool: 如果释放操作成功返回 True，否则返回 False。
        """
        response = self._make_request(endpoint="release", instance_id=instance_id)  # ⭐ 请求释放资源
        return response["success"]


def main():
    """
    演示 EnvClient 用法的脚本：
    - 获取指定环境类型的可用任务
    - 基于获取的任务创建一个实例
    - 在创建的实例中执行一步操作
    - 评估实例
    - 释放实例
    """
    client = EnvClient()

    env_type = "appworld"
    # 1. 获取任务列表
    task_ids = client.get_env_profile(env_type)
    print(f"Available tasks: {task_ids}")

    # 2. 初始化环境实例
    task_id = task_ids[0]
    init_response = client.create_instance(env_type, task_id)
    print("init state", init_response)
    
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    # 3. 执行动作 (Action)
    action = {"role": "assistant", "content": "print('hello appworld!!')"}
    result = client.step(instance_id, action)
    print(f"Step result: {result}")

    # 4. 评估 (Evaluate)
    score = client.evaluate(instance_id)
    print(f"Evaluation score: {score}")

    # 5. 释放资源 (Release)
    success = client.release_instance(instance_id)
    print(f"Instance released: {success}")


if __name__ == "__main__":
    main()