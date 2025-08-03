
# 本文件定义了 GeneralEnv 通用环境类，继承自 BaseEnv，
# 用于在 MINT 框架下实现通用任务的环境交互、工具调用、反馈机制等。
# 主要功能包括：环境初始化、工具集管理、动作解析、反馈获取、任务判定、状态管理、工具调用、答案判定、迭代控制等。

import re
import logging
import traceback
from typing import Any, Dict, List, Mapping, Tuple, Optional

LOGGER = logging.getLogger("MINT")

from mint import agents
from mint.envs.base import BaseEnv
from mint.prompt import ToolPromptTemplate
from mint.datatypes import State, Action, StepOutput, FeedbackType
from mint.tools import Tool, get_toolset_description
from mint.tasks import Task
from mint.tools.python_tool import PythonREPL
from mint.utils.exception import ParseError


INVALID_INPUT_MESSAGE = (
    "I don't understand your input. \n"
    "If you want to execute code, please use <execute> YOUR_CODE_HERE </execute>.\n"
    "If you want to give me an answer, please use <solution> YOUR_SOLUTION_HERE </solution>.\n"
    "For example: The answer to the question is <solution> 42 </solution>. \n"
)



# GeneralEnv：通用环境类，支持工具调用、反馈机制、任务判定等。
class GeneralEnv(BaseEnv):
    """
    通用环境类，支持工具调用、反馈机制、任务判定等。
    主要功能：环境初始化、工具集管理、动作解析、反馈获取、任务判定、状态管理、工具调用、答案判定、迭代控制。
    """
    def __init__(
        self,
        task: Task,
        tool_set: List[Tool],
        feedback_config: Dict[str, Any],
        environment_config: Dict[str, Any],
    ):
        """
        初始化 GeneralEnv 环境。
        参数说明：
        - task: 任务对象。
        - tool_set: 工具列表。
        - feedback_config: 反馈相关配置。
        - environment_config: 环境相关配置。
        功能：
        1. 绑定任务和工具集。
        2. 初始化环境状态。
        3. 配置反馈 agent 和反馈类型。
        4. 初始化 Python REPL。
        """
        self.task: Task = task
        self.tool_set: List[Tool] = tool_set + getattr(self, "tool_set", [])

        self.state = State()
        self.config = environment_config

        # 反馈相关配置与 agent 初始化
        self.feedback_config = feedback_config
        feedback_agent_config = feedback_config["feedback_agent_config"]
        if feedback_config["pseudo_human_feedback"] in ["GT", "no_GT"]:
            self.feedback_agent: agents = getattr(
                agents, feedback_agent_config["agent_class"]
            )(feedback_agent_config)
        else:
            self.feedback_agent = None
        # 设置反馈类型
        if self.feedback_config["pseudo_human_feedback"] == "None":
            self.feedback_type = FeedbackType.NO_FEEDBACK
        elif self.feedback_config["pseudo_human_feedback"] == "no_GT":
            self.feedback_type = FeedbackType.FEEDBACK_WO_GT
        elif self.feedback_config["pseudo_human_feedback"] == "GT":
            self.feedback_type = FeedbackType.FEEDBACK_WITH_GT
        else:
            raise ValueError(
                f"Invalid feedback type {self.feedback_config['pseudo_human_feedback']}"
            )

        self.env_outputs: List[StepOutput] = []  # 环境输出记录
        LOGGER.info(
            f"{len(self.tool_set)} tools loaded: {', '.join([t.name for t in self.tool_set])}"
        )

        # 初始化 Python REPL，支持代码执行型工具
        user_ns = {tool.name: tool.__call__ for tool in self.tool_set}
        user_ns.update(task.user_ns)
        self.python_repl = PythonREPL(
            user_ns=user_ns,
        )


    def parse_action(self, action: Action) -> Tuple[str, Dict[str, Any]]:
        """
        解析 agent 的动作。
        - 若为答案型动作，提取 <solution> 标签内容。
        - 若为工具调用型动作，提取 <execute> 标签内容。
        返回：解析后的内容字典。
        """
        lm_output = "\n" + action.value + "\n"
        output = {}
        try:
            if not action.use_tool:
                answer = "\n".join(
                    [
                        i.strip()
                        for i in re.findall(
                            r"<solution>(.*?)</solution>", lm_output, re.DOTALL
                        )
                    ]
                )
                if answer == "":
                    raise ParseError("No answer found.")
                output["answer"] = answer
            else:
                env_input = "\n".join(
                    [
                        i.strip()
                        for i in re.findall(
                            r"<execute>(.*?)</execute>", lm_output, re.DOTALL
                        )
                    ]
                )
                if env_input == "":
                    raise ParseError("No code found.")
                output["env_input"] = env_input
        except Exception as e:
            raise ParseError(e)
        return output


    def get_feedback(self, observation: str) -> Tuple[str, FeedbackType]:
        """
        获取反馈。
        根据反馈类型，调用反馈 agent 获取反馈内容。
        返回：反馈内容。
        """
        if self.feedback_type == FeedbackType.NO_FEEDBACK:
            return ""
        elif self.feedback_type == FeedbackType.FEEDBACK_WO_GT:
            gt = None
        else:
            gt = self.task.reference

        feedback = self.feedback_agent.act(
            self.state,
            observation=observation,
            form=self.feedback_config["feedback_form"],
            gt=gt,
            task_in_context_example=self.task.in_context_example(
                use_tool=self.config["use_tools"],
                with_feedback=True,
            ),
            tool_desc=get_toolset_description(self.tool_set),
        )
        return feedback.value


    def check_task_success(self, answer: str) -> bool:
        """
        检查任务是否成功。
        比较 agent 给出的答案与参考答案。
        返回：True 表示任务成功。
        """
        LOGGER.info(f"REFERENCE ANSWER: {self.task.reference}")
        return self.task.success(answer)


    def log_output(self, output: StepOutput) -> None:
        """
        记录环境输出到状态。
        包括历史、最新输出内容。
        """
        if self.state.finished:
            return
        content = output.to_str()
        self.state.history.append({"role": "user", "content": content})
        self.state.latest_output = output.to_dict()
        self.state.latest_output["content"] = content


    def handle_tool_call(self, action: Action) -> str:
        """
        处理工具调用。
        1. 解析工具调用动作。
        2. 执行工具（如 PythonREPL），获取观察结果。
        3. 记录输出，更新计数。
        4. 异常时返回错误信息。
        """
        try:
            parsed = self.parse_action(action)
            env_input = parsed["env_input"]
            obs = self.python_repl(env_input).strip()
            self.env_outputs.append(StepOutput(observation=obs))
            self.state.agent_action_count["use_tool"] += 1
            return obs
        except ParseError:
            self.state.agent_action_count["invalid_action"] += 1
            return INVALID_INPUT_MESSAGE
        except Exception as e:
            error_traceback = traceback.format_exc()
            return f"{error_traceback}"


    def handle_propose_solution(self, action: Action) -> Optional[str]:
        """
        处理答案型动作。
        1. 解析答案。
        2. 检查任务是否成功，若成功则终止任务。
        3. 异常时返回错误信息。
        """
        self.state.agent_action_count["propose_solution"] += 1
        try:
            parsed = self.parse_action(action)
            task_success = self.check_task_success(parsed["answer"])
            if task_success:
                self.state.finished = True
                self.state.success = True
                self.state.terminate_reason = "task_success"
                # NOTE: should not return the function now, because we need to log the output
                # Set state.finished = True will terminate the episode
        except ParseError:
            return INVALID_INPUT_MESSAGE
        except Exception as e:
            error_traceback = traceback.format_exc()
            return f"{error_traceback}"


    def check_max_iteration(self):
        """
        检查 agent 是否达到最大迭代次数。
        若达到最大步数或最大答案次数，则终止任务。
        """
        if self.state.finished:
            # 已结束则忽略
            return

        if (
            # propose solution > max output solution
            self.state.agent_action_count["propose_solution"]
            >= self.config["max_propose_solution"]
        ):
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_propose_steps"
        elif (
            # (propose_solution + use_tool) > max iteration limit
            sum(self.state.agent_action_count.values())
            >= self.config["max_steps"]
        ):
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"


    def step(self, action: Action, loaded=None) -> State:
        """
        环境一步交互。
        1. 记录 agent 动作。
        2. 根据动作类型（工具调用/答案）处理并获取观察结果。
        3. 检查迭代次数是否超限。
        4. 获取反馈。
        5. 记录输出并返回当前状态。
        """
        assert (
            not self.state.finished
        ), "Expecting state.finished == False for env.step()."

        # 记录 agent 动作
        if action.value:
            assistant_action = (
                "Assistant:\n" + action.value
                if not action.value.lstrip().startswith("Assistant:")
                else action.value
            )
            self.state.history.append(
                {"role": "assistant", "content": assistant_action + "\n"}
            )

        if action.error:
            # 错误处理（如超长）
            observation = f"An error occurred. {action.error}"
            self.state.finished = True
            self.state.success = False
            self.state.error = action.error
            self.state.terminate_reason = "error"
            LOGGER.error(f"Error:\n{action.error}")
        elif action.use_tool:
            observation = self.handle_tool_call(action)
        else:
            # 答案型动作
            observation = self.handle_propose_solution(action)

        # 检查是否达到最大迭代次数
        self.check_max_iteration()

        # 记录回合信息
        if self.config["count_down"]:
            turn_info = (
                self.config["max_steps"] - sum(self.state.agent_action_count.values()),
                self.config["max_propose_solution"]
                - self.state.agent_action_count["propose_solution"],
            )
        else:
            turn_info = None

        # 获取反馈
        if loaded != None:
            feedback = loaded["feedback"]
            LOGGER.info(f"Loaded feedback: {feedback}")
        elif not self.state.finished:
            # 生成反馈 agent 所需 observation
            tmp_output = StepOutput(
                observation=observation,
                success=self.state.success,
                turn_info=turn_info,
            )
            feedback = self.get_feedback(observation=tmp_output.to_str())
        else:
            feedback = ""

        # 记录输出
        output = StepOutput(
            observation=observation,
            feedback=feedback,
            feedback_type=self.feedback_type,
            success=self.state.success,
            turn_info=turn_info,
        )

        self.log_output(output)
        return self.state


    def reset(self) -> State:
        """
        环境重置。
        1. 构建用户提示（包含工具描述、上下文示例、任务描述等）。
        2. 初始化状态、计数器。
        3. 工具集重置。
        返回：初始状态。
        """
        use_tool: bool = self.config["use_tools"]
        if use_tool and len(self.tool_set) > 0:
            LOGGER.warning(
                (
                    "No tool is provided when use_tools is True.\n"
                    "Ignore this if you are running code generation."
                )
            )

        user_prompt = ToolPromptTemplate(use_tool=use_tool)(
            max_total_steps=self.config["max_steps"],
            max_propose_solution=self.config["max_propose_solution"],
            tool_desc=get_toolset_description(self.tool_set),
            in_context_example=self.task.in_context_example(
                use_tool=use_tool,
                with_feedback=self.feedback_type != FeedbackType.NO_FEEDBACK,
            ),
            task_prompt="Task:\n" + self.task.prompt,
        )
        self.state.history = [{"role": "user", "content": user_prompt}]
        self.state.latest_output = {"content": user_prompt}
        self.state.agent_action_count = {
            "propose_solution": 0,
            "use_tool": 0,
            "invalid_action": 0,
        }

        if use_tool:
            # 工具集重置
            for tool in self.tool_set:
                tool.reset()
        return self.state


    # 析构函数，清理任务资源
    def __del__(self):
        self.task.cleanup()
