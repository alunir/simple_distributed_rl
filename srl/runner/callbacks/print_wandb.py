from dataclasses import dataclass

import wandb
from srl.base.rl.base import RLParameter
from srl.base.run.context import RunContext
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callbacks.print_progress import PrintProgress


class CallbackAlertWandB:
    def title(self, context: RunContext, state: RunStateTrainer) -> str:
        raise NotImplementedError

    def text(self, context: RunContext, state: RunStateTrainer) -> str:
        raise NotImplementedError


@dataclass
class PrintWandB(PrintProgress):
    def __init__(
        self,
        wandb_key: str = "",
        wandb_project: str = "",
        wandb_name: str = "",
        save_code: bool = True,
        alert: CallbackAlertWandB | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert wandb_key != "", "Wandb key is required."
        assert wandb_project != "", "Wandb project name is required."
        assert wandb_name != "", "Wandb name is required."

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

    def _eval_str(self, context: RunContext, parameter: RLParameter) -> str:
        eval_rewards = self.run_eval(parameter)
        if eval_rewards is not None:
            self._wandb.log({"eval_reward": eval_rewards[self.progress_worker]})
        return ""

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
