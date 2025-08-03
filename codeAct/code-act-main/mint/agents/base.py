import logging
from typing import List, Dict, Any, Mapping

LOGGER = logging.getLogger("MINT")

from mint.datatypes import Action, State


class LMAgent:
    """
    智能体基类，所有大语言模型 Agent 的父类。
    负责定义通用接口和行为规范。
    """

    def __init__(self, config: Mapping[str, Any]):
        """
        初始化 Agent。
        参数 config 为配置字典，包含模型参数等。
        stop_words 用于限制模型输出，防止生成环境/反馈标签。
        """
        self.config = config
        LOGGER.info(f"Initialized {self.__class__.__name__} with config: {config}")
        self.stop_words = ["\nObservation:", "\nExpert feedback:", "\nTask:", "\n---"]

    def lm_output_to_action(self, lm_output: str) -> Action:
        """
        将大模型输出字符串转换为 Action。
        如果输出包含 <solution>，则认为是最终答案，否则为工具调用。
        """
        propose_solution = bool("<solution>" in lm_output)
        return Action(lm_output, not propose_solution)

    def act(self, state: State) -> Action:
        """
        Agent 的核心决策方法。
        输入为当前环境状态 State，输出为 Action。
        需在子类实现具体推理逻辑。
        历史格式为多轮对话列表。
        """
        raise NotImplementedError

    def add_system_message(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        将用户消息中的 system prompt 拆分出来，作为系统消息。
        用于 prompt 工程，提升模型指令理解。
        """
        first_msg = messages[0]
        assert first_msg["role"] == "user"
        system, examples, task = first_msg["content"].split("\n---\n")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": examples + "\n---\n" + task},
        ] + messages[1:]
        return messages
