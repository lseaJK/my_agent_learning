from .base import LMAgent
import logging
import traceback
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


class ClaudeLMAgent(LMAgent):
    """
    Claude 大模型 Agent，负责与 Anthropic Claude API 交互。
    继承自 LMAgent，支持自定义停用词和模型参数。
    """
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        self.stop_words = [
            "Observation:",
            "Expert feedback:",
            "Task:",
            "---",
            "\n\nHuman:",
        ]

    """
    @backoff.on_exception( backoff.expo, requests.exceptions.RequestException, )
    这是 Python 的装饰器注释，用于为 call_lm 方法添加“自动重试”机制。

    当 call_lm 方法执行时，如果遇到 requests.exceptions.RequestException（即网络请求异常），会自动进行重试。
    重试的时间间隔采用指数递增（expo，即 exponential backoff），每次失败后等待时间会越来越长，直到成功或达到最大重试次数。
    这种写法常用于网络请求、API调用等容易因临时故障失败的场景，可以提升程序的健壮性和容错能力。
    """
    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
    )
    def call_lm(self, messages):
        """
        调用 Anthropic Claude API，获取模型回复。
        支持自定义模型和停用词。
        """
        # Prepend the prompt with the system message
        data = {
            "model": self.config["model_name"],
            "prompt": "",
            "max_tokens_to_sample": self.config.get("max_tokens", 512),
            "temperature": self.config.get("temperature", 0),
            "stop_sequences": self.stop_words,
        }
        for message in messages:
            if message["role"] == "user":
                data["prompt"] += f"\n\nHuman: {message['content']}"
            else:
                data["prompt"] += f"\n\nAssistant: {message['content']}"
        assert len(messages) % 2 == 1, "messages must be odd length"
        data["prompt"] += "\n\nAssistant:"

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            pass
        else:
            raise requests.exceptions.RequestException(
                "Request failed with status code:", response.status_code
            )

        return json.loads(response.text)["completion"], {}

    def act(self, state):
        messages = state.history
        lm_output, token_usage = self.call_lm(messages)
        for usage_type, count in token_usage.items():
            state.token_counter[usage_type] += count
        action = self.lm_output_to_action(lm_output)
        return action
        # except Exception as e:
        #     tb = traceback.format_exc()
        #     return Action(None, False, error="Unknown error")
