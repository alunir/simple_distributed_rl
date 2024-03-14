import logging
from dataclasses import dataclass

import wandb
from srl.base.run.context import RunContext
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callbacks.print.base import PrintBase
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


class CallbackAlertWandB:
    def title(self, runner: Runner) -> str:
        raise NotImplementedError

    def text(self, runner: Runner) -> str:
        raise NotImplementedError


@dataclass
class PrintWandB(PrintBase):
    def __init__(
        self,
        wandb_key: str = "",
        wandb_project: str = "",
        wandb_name: str | None = None,
        save_code: bool = True,
        alert: CallbackAlertWandB | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert wandb_key != "", "Wandb key is required."
        assert wandb_project != "", "Wandb project name is required."
        assert alert is not None, "Wandb alert is required."

        wandb.login(key=wandb_key)

        self._wandb_project = wandb_project
        self._wandb_name = wandb_name
        self._save_code = save_code
        self._alert = alert

    def on_runner_start(self, runner: Runner) -> None:
        self._wandb = wandb.init(
            project=self._wandb_project,
            name=self._wandb_name,
            save_code=self._save_code,
        )
        d = super().on_runner_start(runner)
        self._wandb.log(dict(d))

    def on_trainer_start(self, context: RunContext, state: RunStateTrainer) -> None:
        super().on_trainer_start(context, state)
        self._wandb.config.update(state.parameter.config.to_dict())

    def on_runner_end(self, runner: Runner) -> None:
        super().on_runner_end(runner)

        if runner.rl_config.parameter_path != "":
            # パラメータを保存
            runner.parameter.save(runner.rl_config.parameter_path)

            # アーティファクトとして保存
            artifact = wandb.Artifact("model", type="model")
            artifact.add(runner.rl_config.parameter_path)
            self._wandb.log_artifact(artifact)

        if runner.rl_config.memory_path != "":
            # パラメータを保存
            runner.memory.save(runner.rl_config.memory_path)

            # アーティファクトとして保存
            artifact = wandb.Artifact("memory", type="memory")
            artifact.add(runner.rl_config.memory_path)
            self._wandb.log_artifact(artifact)

        if self._alert:
            self._wandb.alert(
                title=self._alert.title(runner),
                text=self._alert.text(runner),
            )

        self._wandb.finish()

    # -----------------------------------------

    def _print_actor(self, context: RunContext, state: RunStateActor):
        d = super()._print_actor(context, state)
        self._wandb.log(dict(d))

    def _print_trainer(self, context: RunContext, state: RunStateTrainer):
        d = super()._print_trainer(context, state)
        self._wandb.log(dict(d))
