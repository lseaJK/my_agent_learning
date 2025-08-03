
# 本文件定义了 AlfworldEnv 环境类，继承自 GeneralEnv，
# 用于在 MINT 框架下对 AlfWorld 任务进行环境交互和工具调用。
# 主要功能包括：环境初始化、工具集管理、任务成功判定、工具调用处理。

from mint.envs.general_env import GeneralEnv
from mint.datatypes import StepOutput, Action
from mint.tools import Tool
from mint.tools.alfworld import ALFWORLD_TOOL_CLS
from mint.tasks.alfworld import AlfWorldTask
from typing import Any, Dict, List, Mapping



# AlfworldEnv：用于处理 AlfWorld 任务的环境类，继承自 GeneralEnv。
class AlfworldEnv(GeneralEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        tool_set: List[Tool],
        feedback_config: Dict[str, Any],
        environment_config: Dict[str, Any],
    ):
        """
        初始化 AlfworldEnv 环境。
        - task: AlfWorldTask 实例，包含任务和底层环境。
        - tool_set: 传入的工具列表。
        - feedback_config: 反馈相关配置。
        - environment_config: 环境相关配置。
        功能：
        1. 绑定底层环境 self.env。
        2. 初始化 action_results 用于记录工具调用结果。
        3. 构建 AlfWorld 专属工具集，并合并到 GeneralEnv 的工具集。
        4. 调用父类初始化。
        """
        self.env = task.env
        self.action_results: List[StepOutput] = []
        self.tool_set = [
            tool_cls(self.env, callback_fn=self.action_results.append)
            for tool_cls in ALFWORLD_TOOL_CLS
        ]  # AlfWorld 专属工具集，后续与 GeneralEnv 工具集合并
        super().__init__(task, tool_set, feedback_config, environment_config)


    def check_task_success(self, *args, **kwargs) -> bool:
        """
        判断任务是否成功。
        AlfWorld 环境下，通过检查最近一次工具调用结果是否成功来判定任务是否完成。
        返回：True 表示任务成功，False 表示未成功。
        """
        if len(self.action_results) == 0:
            # 没有与 AlfWorld 交互则任务不可能成功
            return False
        for result in reversed(self.action_results):
            if result.success:
                return True
        return False


    def handle_tool_call(self, action: Action):
        """
        处理工具调用。
        1. 调用父类 GeneralEnv 的工具调用方法。
        2. 检查本次工具调用后任务是否成功，若成功则更新状态并终止任务。
        3. 返回工具调用的观察结果。
        """
        obs = super().handle_tool_call(action)
        # AlfWorld 环境下，正确的工具调用即可完成任务
        self.state.success = self.check_task_success()
        if self.state.success:
            self.state.finished = True
            self.state.terminate_reason = "task_success"
        return obs
