from abc import ABC, abstractmethod
from mint.datatypes import State, Action


class BaseEnv(ABC):
    """
    环境抽象基类，所有环境类都需继承并实现。
    约定必须实现 step 和 reset 两个方法。
    """

    @abstractmethod
    def step(self, action: Action) -> State:
        """
        执行一步环境交互，输入为 Agent 的 Action，输出为新的 State。
        """
        pass

    @abstractmethod
    def reset(self) -> State:
        """
        重置环境，返回初始 State。
        """
        pass
