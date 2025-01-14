import numpy as np

import srl
from srl.algorithms import dqn  # algorithm load
from srl.base.define import EnvObservationTypes
from srl.rl.processors.image_processor import ImageProcessor
from srl.utils import common

common.set_logger()

_parameter_path = "_sample_atari_parameter.dat"
_history_path = "_sample_atari_history"

TRAIN_COUNT = 500_000


def _create_runner():
    # --- Atari env
    # Run "pip install gymnasium pygame" and also see the URL below.
    # https://gymnasium.farama.org/environments/atari/
    env_config = srl.EnvConfig(
        "ALE/Pong-v5",
        kwargs=dict(frameskip=7, repeat_action_probability=0, full_action_space=False),
    )
    rl_config = dqn.Config(
        batch_size=32,
        target_model_update_interval=10_000,
        discount=0.99,
        lr=0.00025,
        enable_reward_clip=False,
        enable_double_dqn=True,
        enable_rescale=False,
    )
    rl_config.memory.warmup_size = 1_000
    rl_config.epsilon.set_linear(TRAIN_COUNT, 1.0, 0.1)
    rl_config.memory.capacity = 10_000
    rl_config.image_block.set_r2d3_image()
    rl_config.hidden_block.set_mlp((512,))
    rl_config.window_length = 4

    # カスタムしたprocessorを追加
    rl_config.processors = [
        ImageProcessor(
            image_type=EnvObservationTypes.GRAY_2ch,
            trimming=(30, 0, 210, 160),
            resize=(84, 84),
            enable_norm=True,
        )
    ]
    rl_config.use_rl_processor = False  # アルゴリズムデフォルトのprocessorを無効にする

    runner = srl.Runner(env_config, rl_config)
    return runner


def train():
    runner = _create_runner()

    # (option) print tensorflow model
    runner.model_summary()

    # --- train
    runner.set_history_on_file(_history_path, interval=5)
    runner.train_mp(actor_num=1, max_train_count=TRAIN_COUNT)
    runner.save_parameter(_parameter_path)


def plot_history():
    history = srl.Runner.load_history(_history_path)
    history.plot()


def evaluate():
    runner = _create_runner()
    runner.load_parameter(_parameter_path)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=10)
    print(f"reward 10 episode mean: {np.mean(rewards)}")

    # --- animation
    runner.animation_save_gif("_atari.gif")


if __name__ == "__main__":
    train()
    plot_history()
    evaluate()
