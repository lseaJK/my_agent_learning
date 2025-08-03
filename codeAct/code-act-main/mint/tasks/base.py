# e:\my_agent_learning\codeAct\code-act-main\mint\tasks\base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

# 定义任务的基类，所有具体任务都需要继承自它
class Task(ABC):
    """
    任务基类，所有具体任务都需继承。
    约定必须实现 get_prompt 方法。
    属性：
    - name: 任务名称。
    - description: 任务描述，说明任务目标。
    - stop_conditions: 停止条件列表，用于判断任务是否完成。
    """
    name: str
    description: str
    stop_conditions: List[str]

    def __init__(
        self,
        name: str,
        description: str,
        stop_conditions: List[str],
        **kwargs: Dict[str, Any],
    ):
        """
        初始化任务。
        参数：
        - name: 任务名称。
        - description: 任务描述。
        - stop_conditions: 停止条件列表。
        """
        self.name = name
        self.description = description
        self.stop_conditions = stop_conditions

    @abstractmethod
    def get_prompt(self) -> str:
        """
        返回任务的 prompt（初始指令）。
        """
        # 抽象方法，子类必须实现，用于生成任务的初始提示
        raise NotImplementedError

    @abstractmethod
    def get_ground_truth(self) -> Any:
        """
        Returns the ground truth for the task.
        """
        # 抽象方法，子类必须实现，用于获取任务的正确答案或目标状态
        raise NotImplementedError

    def __del__(self) -> None:
        # 析构函数，用于在任务结束时执行清理操作
        pass