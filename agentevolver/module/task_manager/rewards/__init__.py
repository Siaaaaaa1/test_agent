import functools
from dataclasses import dataclass
from typing import Type, Dict, Callable, Any, Optional

# 定义一个数据类，用于存储注册表中的条目信息
# 这是一个轻量级的容器，保存了：类对象本身、是否单例模式、以及单例的缓存实例
@dataclass
class _RegEntry:
    cls: Type                  # 注册的类（即计算器类本身）
    singleton: bool = False    # 标记该类是否应该作为单例运行（全局只实例化一次）
    instance: Optional[Any] = None # 如果是单例，这里存储实例化后的对象缓存

class RewardCalculatorManager:
    """
    奖励计算器管理器类。
    这是一个单例类（Singleton），用于管理和实例化不同的奖励计算器。
    它充当了一个“工厂”，根据名称创建对象。
    """
    _instance = None  # 用于存储 Manager 自身的单例实例
    _registry: Dict[str, _RegEntry] = {}  # 核心注册表字典：Key是字符串别名，Value是_RegEntry对象

    def __new__(cls, *args, **kwargs):
        """
        实现 Manager 自身的单例模式。
        确保整个程序运行期间，RewardCalculatorManager 只有一个实例。
        """
        if not cls._instance:
            # 如果实例不存在，则创建一个父类的新实例
            cls._instance = super().__new__(cls)
        # 如果已存在，直接返回现有实例
        return cls._instance

    def reg(self, name: str) -> Callable:
        """
        注册装饰器（Decorator）。
        用于将一个具体的计算器类关联到一个字符串名称上。

        注意：虽然代码逻辑中包含了 singleton 的处理逻辑（见 get_calculator），
        但当前的 reg 函数签名缺少 singleton 参数，且内部硬编码了 singleton=False。
        如果需要启用单例注册，需要修改此函数签名。

        用法示例:
            @grader_manager.reg("my_calc")
            class MyCalc: ...

        参数:
            name: 注册的名称（别名）。

        返回:
            Callable: 装饰器函数。
        """
        def decorator(calculator_cls: Type) -> Type:
            # 检查名称是否重复注册，如果重复则打印警告
            if name in self._registry:
                print(f"'{name}' has been registered, will be overwritten by '{calculator_cls.__name__}'")
            
            # 将类信息存入注册表字典
            # 目前此处硬编码了 singleton=False，意味着默认每次都会创建新对象
            self._registry[name] = _RegEntry(cls=calculator_cls, singleton=False, instance=None)
            return calculator_cls
        return decorator

    def get_calculator(self, name: str, *args, **kwargs) -> Any:
        """
        工厂方法：根据注册名获取计算器实例。

        - 如果注册时 singleton=False（默认）：每次调用都会创建一个新实例。
        - 如果注册时 singleton=True：第一次调用创建实例并缓存，后续调用直接返回缓存的实例。

        参数:
            name (str): 注册时使用的字符串名称。
            args: 传递给计算器类构造函数的位置参数。
            kwargs: 传递给计算器类构造函数的关键字参数。

        返回:
            Any: 计算器类的实例。

        抛出:
            ValueError: 如果提供的名称没有被注册。
        """
        # 从注册表中查找条目
        entry = self._registry.get(name)
        if not entry:
            # 如果没找到，抛出错误，并列出当前所有可用的计算器名称
            raise ValueError(f"no reward calculator named '{name}'. available calculators: {list(self._registry.keys())}")

        # 处理单例逻辑
        if entry.singleton:
            if entry.instance is None:
                # ⭐ 第一次调用单例：实例化并缓存
                entry.instance = entry.cls(*args, **kwargs)
            # 返回缓存的实例（注意：后续调用时 args/kwargs 将被忽略）
            return entry.instance

        # ⭐ 非单例模式：每次都使用传入的参数创建一个新的类实例
        return entry.cls(*args, **kwargs)

# 实例化全局唯一的管理器对象
# 其他模块将使用这个对象来进行注册（@grader_manager.reg）和获取实例
grader_manager = RewardCalculatorManager()

# ---------------------------------------------------------------------
# 以下是导入部分
# 这里的导入非常关键：
# Python 执行 import 时会运行对应文件中的代码。
# 这些文件中包含了 @grader_manager.reg(...) 的代码。
# 因此，这些 import 语句的副作用就是将各个计算器类自动“注册”到 grader_manager 中。
# ---------------------------------------------------------------------

from .judge_with_gt import LlmAsJudgeRewardCalculatorWithGT
from .reward import LlmAsJudgeRewardCalculator
from .binary_judge import LlmAsJudgeBinaryRewardCalculator
from .binary_judge_gt import LlmAsJudgeBinaryRewardCalculatorWithGT
from .avg_judge import AvgBinaryGTJudge,AvgLlmJudge
from .env_grader import EnvGrader
# [新增] 导入新的计算器以触发注册，使其可以通过 get_calculator("api_process_reward") 等方式调用
from .api_process_reward_calculator import APIProcessRewardCalculator

# 定义模块的公共接口
__all__=[
    "LlmAsJudgeRewardCalculatorWithGT",
    "LlmAsJudgeRewardCalculator",
    "LlmAsJudgeBinaryRewardCalculator",
    "LlmAsJudgeBinaryRewardCalculatorWithGT",
    "AvgBinaryGTJudge",
    "AvgLlmJudge",
    "EnvGrader",
    "APIProcessRewardCalculator",
    "grader_manager", # 导出管理器实例，供外部使用
]