from abc import ABC, abstractmethod
from typing import Dict, Any


# 定义工具的基类，所有具体工具都需要继承自它
class Tool(ABC):
    """
    Tool base class.
    """
    # 工具的名称，必须是唯一的
    name: str
    # 工具的描述，用于向模型说明工具的功能
    description: str
    # 工具的输入参数，使用 JSON Schema 格式定义
    args_schema: Dict[str, Any]

    def __init__(self, name: str, description: str, args_schema: Dict[str, Any]):
        """
        Initializes the tool.
        Args:
            name: The name of the tool. Must be unique.
            description: A description of the tool's purpose.
            args_schema: The arguments for the tool, as a JSON schema.
        """
        self.name = name
        self.description = description
        self.args_schema = args_schema

    @abstractmethod
    def __call__(self, **kwargs: Any) -> str:
        """
        Executes the tool.
        Returns:
            A string containing the tool's output.
        """
        # 这是一个抽象方法，子类必须实现这个方法来执行具体操作
        pass

    def reset(self) -> None:
        """Reset the tool to its initial state."""
        pass
