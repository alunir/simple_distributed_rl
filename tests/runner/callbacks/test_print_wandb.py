import pickle

import srl
from srl.algorithms import ql
from srl.runner.callbacks.print.print_wandb import CallbackAlertWandB, PrintWandB
from srl.runner.runner import Runner

wandb_key = ""  # your wandb api key
wandb_project = ""  # your wandb project
wandb_name = ""  # your wandb experiment name


class MyCallbackAlertWandB(CallbackAlertWandB):
    def title(self, runner: Runner) -> str:
        return f"WandB Notification [{runner.env_config.name}]"

    def text(self, runner: Runner) -> str:
        metrics = runner.trainer.train_info if runner.trainer else {}
        return "Training is finished.\n" + "\n".join([f"{k}: {v}" for k, v in metrics.items()])


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
