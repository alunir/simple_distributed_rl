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
    def title(self, context: RunContext, state: RunStateTrainer) -> str:
        raise NotImplementedError

    def text(self, context: RunContext, state: RunStateTrainer) -> str:
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

        wandb.login(key=wandb_key)

        self._wandb = wandb.init(
            project=wandb_project,
            name=wandb_name,
            save_code=save_code,
        )

        self._alert = alert

    def on_trainer_start(self, context: RunContext, state: RunStateTrainer) -> None:
        super().on_trainer_start(context, state)
        self._wandb.config.update(state.parameter.config.to_dict())

    def on_runner_start(self, runner: Runner) -> None:
        d = super().on_runner_start(runner)
        self._wandb.log(d)

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer) -> None:
        super().on_trainer_end(context, state)

        if state.parameter.config.parameter_path != "":
            # パラメータを保存
            state.trainer.parameter.save(state.parameter.config.parameter_path)

            # アーティファクトとして保存
            artifact = wandb.Artifact("model", type="model")
            artifact.add(state.parameter.config.parameter_path)
            self._wandb.log_artifact(artifact)

        if state.parameter.config.memory_path != "":
            # パラメータを保存
            state.trainer.parameter.save(state.parameter.config.memory_path)

            # アーティファクトとして保存
            artifact = wandb.Artifact("memory", type="memory")
            artifact.add(state.parameter.config.memory_path)
            self._wandb.log_artifact(artifact)

        if self._alert:
            self._wandb.alert(
                self._alert.title(context, state),
                self._alert.text(context, state),
            )

        self._wandb.finish()

    # -----------------------------------------

    def _print_actor(self, context: RunContext, state: RunStateActor):
        d = super()._print_actor(context, state)
        self._wandb.log(d)

    def _print_trainer(self, context: RunContext, state: RunStateTrainer):
        d = super()._print_trainer(context, state)
        self._wandb.log(d)
