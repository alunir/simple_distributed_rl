import logging
import time
import traceback
from dataclasses import dataclass

import numpy as np

from srl.base.exception import UndefinedError
from srl.base.run.callback import RunCallback, TrainerCallback
from srl.base.run.context import RunContext
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callback import RunnerCallback
from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.callbacks.history_viewer import HistoryViewer
from srl.runner.runner import Runner
from srl.utils.common import summarize_info_from_dictlist

logger = logging.getLogger(__name__)


@dataclass
class HistoryOnMemory(RunnerCallback, RunCallback, TrainerCallback, Evaluate):
    interval: int = 1
    interval_mode: str = "time"

    def on_runner_end(self, runner: Runner) -> None:
        runner.history_viewer = HistoryViewer()
        runner.history_viewer.set_history_on_memory(self, runner)

    def _read_stats(self):
        assert self.runner is not None
        if not self.runner.config.enable_stats:
            return {}

        d = {}

        try:
            memory_percent, cpu_percent = self.runner.read_psutil()
            if memory_percent != np.NaN:
                d["system_memory"] = memory_percent
                d["cpu"] = cpu_percent
        except Exception:
            logger.debug(traceback.format_exc())

        try:
            gpus = self.runner.read_nvml()
            # device_id, rate.gpu, rate.memory
            for device_id, gpu, memory in gpus:
                d[f"gpu{device_id}"] = gpu
                d[f"gpu{device_id}_memory"] = memory
        except Exception:
            logger.debug(traceback.format_exc())

        return d

    def _add_info(self, info, prefix, dict_):
        if dict_ is None:
            return
        for k, v in dict_.items():
            k = f"{prefix}_{k}"
            if k not in info:
                info[k] = [v]
            else:
                info[k].append(v)

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, context: RunContext, state: RunStateActor):
        assert not context.distributed
        self.logs = []
        self.t0 = time.time()

        # 分散の場合はactor_id=0のみevalをする
        if context.distributed:
            self.enable_eval = self.enable_eval and (context.actor_id == 0)

    def on_episode_begin(self, context: RunContext, state: RunStateActor):
        self.episode_infos = {}
        self.t0_episode = time.time()
        self.step_count = 0

        if state.env is not None:
            self._add_info(self.episode_infos, "env", state.env.info)

    def on_step_end(self, context: RunContext, state: RunStateActor):
        self.step_count += 1

        if state.env is not None:
            self._add_info(self.episode_infos, "env", state.env.info)
        if state.trainer is not None:
            self._add_info(self.episode_infos, "trainer", state.trainer.train_info)
        [self._add_info(self.episode_infos, f"worker{i}", w.info) for i, w in enumerate(state.workers)]

    def on_episode_end(self, context: RunContext, state: RunStateActor):
        d = {
            "name": f"actor{context.actor_id}",
            "time": time.time() - self.t0,
            "episode": state.episode_count,
            "episode_time": time.time() - self.t0_episode,
            "episode_step": self.step_count,
            "sync": state.sync_actor,
        }
        if state.env is not None:
            for i, r in enumerate(state.env.episode_rewards):
                d[f"reward{i}"] = r

        trainer = state.trainer
        if trainer is not None:
            d["train"] = trainer.get_train_count()
            memory = state.memory
            d["memory"] = 0 if memory is None else memory.length()
            if state.trainer is not None:
                for k, v in state.trainer.train_info.items():
                    d[f"trainer_{k}"] = v

        assert self.runner is not None
        if self.setup_eval_runner(self.runner):
            eval_rewards = self.run_eval(state.parameter)
            if eval_rewards is not None:
                for i, r in enumerate(eval_rewards):
                    d[f"eval_reward{i}"] = r

        d.update(summarize_info_from_dictlist(self.episode_infos))
        d.update(self._read_stats())

        self.logs.append(d)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, context: RunContext, state: RunStateTrainer):
        self.t0 = time.time()
        self.t0_train = time.time()
        self.t0_train_count = 0
        self.train_infos = {}
        self.logs = []

        if self.interval_mode == "time":
            self.interval0 = self.t0
        elif self.interval_mode == "step":
            self.interval0 = 0
        else:
            raise UndefinedError(self.interval_mode)

        # eval, 分散の場合はevalをしない
        if context.distributed:
            self.enable_eval = False

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer):
        self._save_trainer_log(context, state)

    def on_trainer_loop(self, context: RunContext, state: RunStateTrainer):
        self._add_info(self.train_infos, "trainer", state.trainer.train_info)

        if self.interval_mode == "time":
            _time = time.time()
            if _time - self.interval0 > self.interval:
                self._save_trainer_log(context, state)
                self.interval0 = _time  # last
        elif self.interval_mode == "step":
            self.interval0 += 1
            if self.interval0 > self.interval:
                self._save_trainer_log(context, state)
                self.interval0 = 0
        else:
            raise UndefinedError(self.interval_mode)

    def _save_trainer_log(self, context: RunContext, state: RunStateTrainer):
        train_count = state.trainer.get_train_count()
        if train_count - self.t0_train_count > 0:
            train_time = (time.time() - self.t0_train) / (train_count - self.t0_train_count)
        else:
            train_time = np.inf
        d = {
            "name": "trainer",
            "time": time.time() - self.t0,
            "train": train_count,
            "train_time": train_time,
            "sync": state.sync_trainer,
        }
        memory = state.memory
        d["memory"] = 0 if memory is None else memory.length()

        d.update(summarize_info_from_dictlist(self.train_infos))
        d.update(self._read_stats())

        self.logs.append(d)

        self.t0_train = time.time()
        self.t0_train_count = state.trainer.get_train_count()
        self.train_infos = {}
