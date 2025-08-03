import os
import yaml
import logging
from typing import Iterable
import alfworld
import alfworld.agents.environment as envs
import json
import os

LOGGER = logging.getLogger("MINT")

from mint.tasks.base import Task

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


class AlfWorldTask(Task):
    """
    AlfWorld 任务类，继承自 Task。
    用于封装 AlfWorld 环境下的任务实例。
    属性：
    - task_name: 任务类型。
    - id: 任务唯一标识。
    - prompt: 任务指令。
    - reference: 参考答案。
    - env: AlfWorld 环境实例。
    - task_type: 任务类型字符串。
    """

    task_name = "alfworld"

    def __init__(
        self,
        id: str,
        prompt: str,
        reference: str,
        env: envs.AlfredTWEnv,
        task_type: str,
        **kwargs,
    ):
        """
        初始化 AlfWorldTask。
        参数：
        - id: 任务唯一标识。
        - prompt: 任务指令。
        - reference: 参考答案。
        - env: AlfWorld 环境实例。
        - task_type: 任务类型字符串。
        """
        self.task_name = f"alfworld/{task_type}"  # 用于加载正确的 ICL 示例
        super().__init__(**kwargs)
        self._id = id
        self._prompt = prompt.strip()
        self._reference = reference
        self.metadata["task_type"] = task_type

        # NOTE: AlfWorld is different -> no reference solution
        self._env = env

    @property
    def env(self) -> envs.AlfredTWEnv:
        """Stateful environment of the task.

        Specific for AlfWorld.
        """
        return self._env

    def success(self, solution: str) -> bool:
        """This checks whether the given solution can complete the current task."""
        # Task success is checked at the environment level, not at the solution level.
        raise NotImplementedError


    """这个函数的作用是加载 AlfWorld 环境中的任务数据，并生成每个任务的 Task 实例。它的主要流程如下：

    1. 读取配置文件：加载 base_config.yaml，获取环境配置。

    2. 选择任务集：根据 split 参数（如 "train" 或 "eval_out_of_distribution"），决定加载哪一部分任务，并确定任务数量。

    3. 初始化环境：根据配置创建并初始化 AlfWorld 环境对象。

    4. 筛选任务：如果提供了 ids_to_run_file，只加载指定的任务。

    任务生成器：
    - 循环生成每个任务：
    - 重置环境，获取初始观察和任务信息。
    - 处理观察文本，提取任务文件路径。
    - 如果不是训练集，加载任务的参考轨迹（ground truth）。
    - 解析任务名称，判断是否需要跳过（筛选）。
    - 根据任务名称前缀确定任务类型。
    - 构造任务提示（prompt）。
    - 生成并返回一个 Task 实例，包含任务的各种信息。
    返回：返回任务生成器和任务总数。

    关键点/易错点
    env.reset() 返回的是一个元组，注意解包和处理。
    任务筛选通过 ids_to_run 实现，只有在列表中的任务才会被加载。
    PREFIXES 用于根据任务名称前缀判断任务类型，若找不到会抛出断言错误。
    该函数返回的是一个生成器对象（可迭代），而不是一次性加载所有任务。
    
    假如你只想加载 eval_out_of_distribution 集的部分任务，可以传入 split 和 ids_to_run_file 参数，函数会只生成这些任务的 Task 实例。"""
    @classmethod
    def load_tasks(cls, path: str = "./data/raw/alfworld", **kwargs) -> Iterable["Task"]:
        """Load alfworld data and prompts from a directory."""

        with open(os.path.join(path, "base_config.yaml")) as f:
            config = yaml.safe_load(f)

        # Split following ReAct
        # https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb
        SPLIT_TO_N_TASKS = {
            "eval_out_of_distribution": 134,
            "train": 3553
        }

        split = kwargs.get("split", "eval_out_of_distribution")
        LOGGER.info(f"Loading {split} split of AlfWorld data.")

        env = getattr(alfworld.agents.environment, config["env"]["type"])(
            config, train_eval=split
        )
        assert isinstance(env, alfworld.agents.environment.AlfredTWEnv)
        env = env.init_env(batch_size=1)

        N_TASKS = SPLIT_TO_N_TASKS[split]

        ids_to_run_file = kwargs.get("ids_to_run_file", None)
        if ids_to_run_file is not None:
            with open(ids_to_run_file) as f:
                ids_to_run = set(f.read().splitlines())
            LOGGER.info(f"Running only {len(ids_to_run)} tasks.")
        else:
            ids_to_run = None


        def generator():

            for _ in range(N_TASKS):
                ob, info = env.reset()
                ob = "\n".join(ob[0].split("\n\n")[1:])
                game_file = info["extra.gamefile"][0]

                if split != "train":
                    # Load ground truth reference for evaluation
                    # This is only used for +GT ablation
                    gt_reference_file = os.path.join(
                        os.path.dirname(game_file), "gt_traj.txt"
                    )

                    with open(gt_reference_file) as f:
                        gt_reference = f.read()
                else:
                    gt_reference = None

                name = "/".join(game_file.split("/")[-3:-1])
                if ids_to_run is not None and name not in ids_to_run:
                    LOGGER.info(f"Skipping task {name} from {split} split since not in ids_to_run.")
                    continue
                else:
                    LOGGER.info(f"Loaded task {name} from {split} split.")

                task_type = None
                for _, (k, v) in enumerate(PREFIXES.items()):
                    if name.startswith(k):
                        task_type = v
                        break
                assert task_type is not None, f"Task type not found for {name}"

                prompt = "Interact with a household to solve a task. \n" + ob
                yield cls(
                    id=name,
                    prompt=prompt,
                    reference=gt_reference,
                    env=env,
                    task_type=task_type,
                    loaded_history=None,
                )

        return generator(), N_TASKS
