import logging
from dataclasses import dataclass

import numpy as np

from srl.base.run.context import RunContext
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callbacks.print.base import PrintBase
from srl.runner.runner import Runner
from srl.utils.util_str import to_str_info, to_str_reward, to_str_time

logger = logging.getLogger(__name__)


@dataclass
class PrintProgress(PrintBase):
    """時間に対して少しずつ長く表示、学習し始めたら長くしていく"""

    # mode: str = "simple"
    start_time: int = 1
    interval_limit: int = 60 * 10
    single_line: bool = True
    progress_env_info: bool = False
    progress_train_info: bool = True
    progress_worker_info: bool = True
    progress_worker: int = 0
    progress_max_actor: int = 5

    def __post_init__(self):
        assert self.start_time > 0
        assert self.interval_limit >= self.start_time
        if self.start_time > self.interval_limit:
            logger.info(f"change start_time: {self.start_time}s -> {self.interval_limit}s")
            self.start_time = self.interval_limit

        self.progress_step_count = 0
        self.progress_episode_count = 0
        self.step_count = 0
        self.episode_count = 0
        self.history_step = []
        self.history_episode = []
        self.history_episode_start_idx = 0

    def on_runner_start(self, runner: Runner) -> None:
        d = super().on_runner_start(runner)
        s = f"### env: {0}, rl: {1}".format(d["env"], d["rl"])
        if "max_episodes" in d:
            s += ", max episodes: {}".format(d["max_episodes"])
        if "timeout" in d:
            s += ", timeout: {}".format(to_str_time(float(d["timeout"])))
        if "max_steps" in d:
            s += ", max steps: {}".format(d["max_steps"])
        if "max_train_count" in d:
            s += ", max train: {}".format(d["max_train_count"])
        if "max_memory" in d:
            s += ", max memory: {}".format(d["max_memory"])
        print(s)

    # -----------------------------------------

    def _print_actor(self, context: RunContext, state: RunStateActor):
        d = super()._print_actor(context, state)

        # [TIME] [actor] [elapsed time]
        s = "{0}{1}:{2}".format(d["time"], " actor" + str(d.get("actor", "")), to_str_time(float(d["elapsed_time"])))

        # [remain]
        if d["remain"] == np.inf:
            s += "(     - left)"
        else:
            s += "({} left)".format(to_str_time(float(d["remain"])))

        # [step time]
        _c = float(d["_c"])
        if _c < 10:
            s += f",{_c:6.2f}st/s"
        else:
            s += f",{int(_c):6d}st/s"

        # [all step]
        s += ",{:7d}st".format(d["all_step"])

        # [memory]
        if state.memory is not None:
            s += ",{:6d}mem".format(d["memory"])

        # [sync]
        if context.distributed:
            s += ", Q {0:4d}send/s({1:8d})".format(d["Q"], d["actor_send_q"])
            s += ", {:4d} recv Param".format(d["sync_actor"])

        # [all episode] [train]
        s += ", {:3d}ep".format(state.episode_count)
        if state.trainer is not None:
            s += ", {:5d}tr".format(state.trainer.get_train_count())

        if float(d["diff_episode"]) <= 0:
            if float(d["diff_step"]) <= 0:
                # --- no info
                s += "1 step is not over."
            else:
                # --- steps info
                # [episode step]
                s += ", {}st".format(d["step_num"])

                # [reward]
                s += " [{}]reward".format(d["reward"])

        else:
            # --- episode info
            # [mean episode step]
            s += ", {:3d}st".format(d["mean_episode_step"])

            # [reward]
            s += ",{0} {1} {2} re".format(
                to_str_reward(float(d["reward_min"])),
                to_str_reward(float(d["reward_mid"]), check_skip=True),
                to_str_reward(float(d["reward_max"])),
            )

            # [eval reward]
            s += self._eval_str(context, state.parameter)

        # [system]
        if context.actor_id == 0:
            if "MEM" in d:
                s += "[CPU{0:3.0f}%,M{1:2.0f}%]".format(d["CPU"], d["MEM"])
            else:
                s += "[CPU Nan%]"

            if "GPU" in d:
                s += str(d["GPU"])
            else:
                s += "[GPU Nan%]"
        else:
            s += "[CPU{:3.0f}%]".format(d["CPU"])

        # [info] , 速度優先して一番最新の状態をそのまま表示
        s_info = ""
        env_types = state.env.info_types
        rl_types = context.rl_config.info_types
        if self.progress_env_info:
            s_info += to_str_info(state.env.info, env_types)
        if self.progress_worker_info:
            s_info += to_str_info(state.workers[self.progress_worker].info, rl_types)
        if self.progress_train_info:
            if state.trainer is not None:
                s_info += to_str_info(state.trainer.train_info, rl_types)

        if self.single_line:
            print(s + s_info)
        elif s_info == "":
            print(s)
        else:
            print(s)
            print("  " + s_info)

    def _print_trainer(self, context: RunContext, state: RunStateTrainer):
        d = super()._print_trainer(context, state)

        # [TIME] [actor] [elapsed time]
        s = "{0}{1} {2}".format(
            d["time"],
            " trainer:" if context.distributed else "",
            to_str_time(float(d["elapsed_time"])),
        )

        # [remain]
        if d["remain"] == np.inf:
            s += "(     - left)"
        else:
            s += "({} left)".format(to_str_time(float(d["remain"])))

        # [train time]
        _c = float(d["_c"])
        if _c < 10:
            s += f",{_c:6.2f}tr/s"
        else:
            s += f",{int(_c):6d}tr/s"

        # [train count]
        s += ",{:7d}tr".format(d["train_count"])

        # [memory]
        if state.memory is not None:
            s += ",{:6d}mem".format(d["memory"])

        # [distributed]
        if context.distributed:
            s += ", Q {0:4d}recv/s({1:8d})".format(d["Q"], d["trainer_recv_q"])
            s += ", {:4d} send Param".format(d["sync_trainer"])

        if float(d["train_count"]) == 0:
            # no info
            s += " 1train is not over."
        else:
            # [eval]
            s += self._eval_str(context, state.parameter)

        # [system]
        if context.actor_id == 0:
            if "MEM" in d:
                s += "[CPU{0:3.0f}%,M{1:2.0f}%]".format(d["CPU"], d["MEM"])
            else:
                s += "[CPU Nan%]"

            if d["GPU"] != np.NaN:
                s += str(d["GPU"])
            else:
                s += "[GPU Nan%]"
        else:
            s += "[CPU{:3.0f}%]".format(d["CPU"])

        # [info] , 速度優先して一番最新の状態をそのまま表示
        s_info = ""
        if self.progress_train_info:
            if state.trainer is not None:
                s_info += to_str_info(state.trainer.train_info, context.rl_config.info_types)

        if self.single_line:
            print(s + s_info)
        elif s_info == "":
            print(s)
        else:
            print(s)
            print("  " + s_info)
