import pickle

import srl
from srl.algorithms import ql
from srl.base.run.context import RunContext
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callbacks.print.print_wandb import CallbackAlertWandB, PrintWandB

wandb_key = ""  # your wandb api key
wandb_project = ""  # your wandb project
wandb_name = ""  # your wandb experiment name


class MyCallbackAlertWandB(CallbackAlertWandB):
    def title(self, context: RunContext, state: RunStateTrainer) -> str:
        return f"WandB Notification [{context.env_config.name}]"

    def text(self, context: RunContext, state: RunStateTrainer) -> str:
        return f"Training is finished.\n{state.trainer.train_info}"


def test_pickle():
    callback = PrintWandB(
        wandb_key=wandb_key,
        wandb_project=wandb_project,
        wandb_name=None,
        alert=MyCallbackAlertWandB(),
    )

    pickle.loads(pickle.dumps(callback))


def test_callback():
    runner = srl.Runner("Grid", ql.Config())

    callback = PrintWandB(
        wandb_key=wandb_key,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        alert=MyCallbackAlertWandB(),
        start_time=1,
        progress_env_info=True,
        enable_eval=True,
    )

    runner.train(timeout=15, enable_progress=True, callbacks=[callback])
