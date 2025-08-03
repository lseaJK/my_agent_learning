import re
import logging
from .openai_lm_agent import OpenAILMAgent
from mint.datatypes import State, Action
from mint.prompt import FeedbackPromptTemplate
import logging
from mint.datatypes import Action
import backoff
import requests
import os
import json

LOGGER = logging.getLogger("MINT")

url = "https://api.anthropic.com/v1/complete"
headers = {
    "accept": "application/json",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
    "x-api-key": os.environ.get("ANTHROPIC_API_KEY", None),
}


class ClaudeFeedbackAgent(OpenAILMAgent):
    """
    Claude 反馈型 Agent，专用于生成评估或反馈。
    继承自 OpenAILMAgent，主要用于辅助主流程评测闭环。
    """
    def __init__(self, config):
        super().__init__(config)
        # 反馈 Agent 不应生成 Assistant 消息，只输出反馈内容。
        self.stop_words = ["\nObservation:", "\nTask:", "\nAssistant:"]
        self.feedback_prompt = FeedbackPromptTemplate()

    def lm_output_to_action(self, lm_output, form) -> Action:
        """
        将模型输出转换为反馈 Action。
        支持文本型和二元型反馈。
        """
        if form == "textual":
            feedback = lm_output
        elif form == "binary":
            # 提取首句作为反馈，判断 GOOD/BAD。
            first_sent = re.findall(r"([^.]*\.)", lm_output)[0]
            if "GOOD" in first_sent:
                feedback = "This is GOOD."
            elif "BAD" in first_sent:
                feedback = "This is BAD."
            else:
                raise ValueError(f"Cannot find GOOD or BAD in feedback: {feedback}")
        return Action(feedback, use_tool=False)

    # ClaudeFeedbackAgent 的 act 方法只负责生成反馈意见（如评价、建议等），并不直接判断反馈是否正确。
    # 反馈的正确与否通常由主流程（如主 Agent 或评测模块）在后续环节进行判断。
    # 具体来说，主流程会根据 ClaudeFeedbackAgent 生成的反馈内容，结合 ground truth 或其他标准，来评估反馈的质量和正确性。
    # 判断反馈正确与否的时机一般是在主流程收集所有反馈后，进行统一评测或打分。
    def act(
        self,
        state: State,
        observation: str,
        form: str,
        gt,
        task_in_context_example: str,
        tool_desc: str,
    ) -> Action:
        """
        生成反馈 Action。
        输入为当前状态 State、观察结果 observation、反馈形式 form、正确答案 gt、上下文示例 task_in_context_example 和工具描述 tool_desc。

        统计 token 用量

        把本次反馈相关的 token 用量累加到 state.token_counter。
        解析模型输出为 Action

        用 self.lm_output_to_action(lm_output, form) 方法，把模型输出转换为最终的反馈 Action。
        日志记录

        用不同颜色在日志里记录 prompt 和最终 Action，方便调试。
        返回 Action

        返回生成的反馈 Action。
        注意点：

        轨迹和上下文示例都做了裁剪，避免混淆初始指令和实际任务内容。
        正确答案不会直接泄露给助手，只作为生成反馈的参考。
        反馈的具体内容由 feedback_prompt 方法和语言模型共同决定。
        """
        # 1.准备正确答案描述
        # 如果传入了 gt（正确答案），就生成一段提示（但要求不要泄露给助手）。如果没有，则标记为 "NOT GIVEN"。
        gt_solution = (
            (
                f"Correct solution (please DO NOT disclose the correct solution to the assistant): {str(gt).strip()}\n"
            )
            if gt
            else "Correct solution (please DO NOT disclose the correct solution to the assistant): NOT GIVEN\n"
        )

        # 2.构建任务轨迹（trajectory）


        # 1）从 state.history 中提取历史内容，去掉最前面的指令部分，只保留与任务相关的内容。
        trajectory = "---\n".join(state.history[0]["content"].split("---\n")[2:]) + "\n"
        trajectory += "\n".join([x["content"] for x in state.history[1:]])

        # 2）把当前观察结果 observation 加入轨迹。
        trajectory += "\n" + observation

        # 3）准备反馈提示（prompt）
        # 用 self.feedback_prompt 方法，结合上下文示例、轨迹、正确答案和工具描述，生成完整的提示内容。
        # 这个内容会作为消息传递给语言模型。
        trajectory = trajectory[
            trajectory.find("Task:") :
        ]  # Get rid of the initial instruction to avoid confusion
        messages = [
            {
                "role": "user",
                "content": self.feedback_prompt(
                    in_context_example=task_in_context_example[
                        task_in_context_example.find("Task:") :
                    ],  # This is to get rid of the initial instruction to avoid confusion
                    trajectory=trajectory,
                    correct_solution=gt_solution,
                    tool_desc=tool_desc,
                ),
            }
        ]

        # log in yellow
        LOGGER.debug(
            "Feedback Agent Prompt:\n" + "\033[93m" + messages[0]["content"] + "\033[0m"
        )

        # 4）调用语言模型
        # 用 self.call_lm(messages) 方法，把准备好的消息发给语言模型，得到模型输出和 token 使用统计。
        lm_output, token_usage = self.call_lm(messages)
        for usage_type, count in token_usage.items():
            state.token_counter["feedback_" + usage_type] += count

        # 5）解析模型输出为 Action
        action = self.lm_output_to_action(lm_output, form)
        # log in red
        LOGGER.debug("Feedback Agent Action:\n" + "\033[91m" + action.value + "\033[0m")
        return action

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
    )
    def call_lm(self, messages):
        # Prepend the prompt with the system message
        data = {
            "model": self.config["model_name"],
            "prompt": f"\n\nHuman: {messages[0]['content']}\n\n",
            "max_tokens_to_sample": self.config.get("max_tokens", 512),
            "temperature": self.config.get("temperature", 0),
            "stop_sequences": self.stop_words,
        }
        assert len(messages) == 1, "message length must be 1"

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            pass
        else:
            raise requests.exceptions.RequestException(
                "Request failed with status code:", response.status_code
            )

        return json.loads(response.text)["completion"], {}
