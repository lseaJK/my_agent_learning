from typing import Dict, Any
from ..utils.exec import PythonExecutor
from .base import Tool

# Python REPL 工具的描述，告诉模型这个工具能做什么
TOOL_DESCRIPTION = """
A Python REPL. Use this to execute python code. The last line of your code should be a print statement.
The code will be executed in a stateful environment.
"""

# 定义 Python REPL 工具
class PythonTool(Tool):
    """
    A tool for executing Python code.
    """

    def __init__(self, python_executor: PythonExecutor):
        """
        Initializes the Python tool.
        Args:
            python_executor: The Python executor to use.
        """
        # 调用父类的构造函数，设置工具的名称、描述和参数结构
        super().__init__(
            name="python",
            description=TOOL_DESCRIPTION,
            args_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. The last line should be a print statement.",
                    }
                },
                "required": ["code"],
            },
        )
        # 保存 Python 执行器的实例
        self.python_executor = python_executor

    def __call__(self, code: str) -> str:
        """


        Executes the Python code.
        Args:
            code: The Python code to execute.
        Returns:
            The output of the code.
        """
        # 调用 Python 执行器来运行代码，并返回结果
        return self.python_executor.run(code)
