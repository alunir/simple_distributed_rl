import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, cast

import numpy as np

from srl.base.define import EnvObservationTypes, RLBaseTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.rl.functions.common import create_epsilon_list, inverse_rescaling, render_discrete_action, rescaling
from srl.rl.memories.priority_experience_replay import PriorityExperienceReplay, PriorityExperienceReplayConfig
from srl.rl.models.dueling_network import DuelingNetworkConfig
from srl.rl.models.framework_config import FrameworkConfig
from srl.rl.models.image_block import ImageBlockConfig
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.scheduler import SchedulerConfig

"""
・Paper
Rainbow: https://arxiv.org/abs/1710.02298
Double DQN: https://arxiv.org/abs/1509.06461
Priority Experience Replay: https://arxiv.org/abs/1511.05952
Dueling Network: https://arxiv.org/abs/1511.06581
Multi-Step learning: https://arxiv.org/abs/1703.01327
Retrace: https://arxiv.org/abs/1606.02647
Noisy Network: https://arxiv.org/abs/1706.10295
Categorical DQN: https://arxiv.org/abs/1707.06887

DQN
    window_length          : -
    Fixed Target Q-Network : o
    Error clipping     : o
    Experience Replay  : o
    Frame skip         : -
    Annealing e-greedy : o (config selection)
    Reward clip        : o (config selection)
    Image preprocessor : -
Rainbow
    Double DQN                  : o (config selection)
    Priority Experience Replay  : o (config selection)
    Dueling Network             : o (config selection)
    Multi-Step learning(retrace): o (config selection)
    Noisy Network               : o (config selection)
    Categorical DQN             : x

Other
    Value function rescaling : o (config selection)
    invalid_actions : o

"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, PriorityExperienceReplayConfig):
    """<:ref:`PriorityExperienceReplay`>"""

    test_epsilon: float = 0

    actor_epsilon: float = 0.4
    actor_alpha: float = 7.0

    epsilon: float = 0.1  # type: ignore , type OK
    lr: float = 0.001  # type: ignore , type OK

    # --- model
    framework: FrameworkConfig = field(init=False, default_factory=lambda: FrameworkConfig())
    image_block: ImageBlockConfig = field(init=False, default_factory=lambda: ImageBlockConfig())
    discount: float = 0.99
    target_model_update_interval: int = 1000
    enable_reward_clip: bool = False

    # double dqn
    enable_double_dqn: bool = True

    # DuelingNetwork
    dueling_network: DuelingNetworkConfig = field(init=False, default_factory=lambda: DuelingNetworkConfig())

    # Multi-step learning
    multisteps: int = 3
    retrace_h: float = 1.0

    # noisy dense
    enable_noisy_dense: bool = False

    # other
    enable_rescale: bool = False

    def __post_init__(self):
        super().__post_init__()

        self.epsilon: SchedulerConfig = SchedulerConfig(cast(float, self.epsilon))
        self.lr: SchedulerConfig = SchedulerConfig(cast(float, self.lr))

        self.memory.set_proportional_memory()
        self.dueling_network.set((512,), True)

    def set_config_by_actor(self, actor_num: int, actor_id: int) -> None:
        e = create_epsilon_list(actor_num, epsilon=self.actor_epsilon, alpha=self.actor_alpha)[actor_id]
        self.epsilon.set_constant(e)

    # 論文のハイパーパラメーター
    def set_atari_config(self):
        # Annealing e-greedy
        self.epsilon.set_linear(1_000_000, 1.0, 0.1)
        # model
        self.image_block.set_dqn_image()
        self.dueling_network.set((512,), True, "average")

        self.discount = 0.99
        self.lr.set_constant(0.0000625)
        self.batch_size = 32
        self.target_model_update_interval = 32000
        self.enable_reward_clip = True

        self.enable_double_dqn = True

        # memory
        self.memory.warmup_size = 80_000
        self.memory.capacity = 1_000_000
        self.memory.set_proportional_memory(
            alpha=0.5,
            beta_initial=0.4,
            beta_steps=1_000_000,
        )

        # Multi-step learning
        self.multisteps = 3
        self.retrace_h = 1.0

        # noisy dense
        self.enable_noisy_dense = True

        # other
        self.enable_rescale = False

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationTypes.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    @property
    def base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS

    def get_use_framework(self) -> str:
        return self.framework.get_use_framework()

    def getName(self) -> str:
        if self.multisteps == 1:
            return f"Rainbow_no_multisteps:{self.get_use_framework()}"
        else:
            return f"Rainbow:{self.get_use_framework()}"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()
        assert self.multisteps > 0

    @property
    def info_types(self) -> dict:
        return {
            "loss": {},
            "sync": {"type": int, "data": "last"},
            "epsilon": {"data": "last"},
            "lr": {"data": "last"},
        }


# ------------------------------------------------------
# Memory
# ------------------------------------------------------
class Memory(PriorityExperienceReplay):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class CommonInterfaceParameter(RLParameter, ABC):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.multi_discounts = np.array([self.config.discount**n for n in range(self.config.multisteps)])

    @abstractmethod
    def predict_q(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def predict_target_q(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def calc_target_q(self, batchs, training: bool):
        batch_size = len(batchs)
        multi_discounts = np.tile(self.multi_discounts, (batch_size, 1))

        # (batch, multistep, shape)
        states_list, onehot_actions_list, rewards, dones, _ = zip(*batchs)
        states_list = np.asarray(states_list)
        onehot_actions_list = np.asarray(onehot_actions_list, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # 1step目はretraceで使わない、retraceで使うのは 2step以降
        states = states_list[:, 0, :]
        n_states = states_list[:, 1:, :]
        onehot_actions = onehot_actions_list[:, 0, :]
        n_onehot_actions = onehot_actions_list[:, 1:, :]

        """
        - actionQ (online) 1step ～ N step
        - nextQ
          double_dqn
            (online) 1step ～ N+1 step
            (target) 1step ～ N+1 step
          no double_dqn
            (online) no use
            (target) 1step ～ N+1 step
        """
        if self.config.enable_double_dqn:
            online_states = n_states
            target_states = n_states
        else:
            online_states = n_states[:, :-1, :]
            target_states = n_states

        # (batch, multistep, shape) -> (batch * multistep, shape)
        online_shape1 = online_states.shape[1]
        online_states = np.reshape(online_states, (batch_size * online_shape1,) + online_states.shape[2:])
        target_states = np.reshape(target_states, (batch_size * self.config.multisteps,) + target_states.shape[2:])

        q_online = self.predict_q(online_states)
        q_target = self.predict_target_q(target_states)

        # (batch * multistep, shape) -> (batch, multistep, shape)
        q_online = np.reshape(q_online, (batch_size, online_shape1) + q_online.shape[1:])
        q_target = np.reshape(q_target, (batch_size, self.config.multisteps) + q_target.shape[1:])

        # --- action Q
        q = np.sum(q_online[:, : self.config.multisteps - 1, :] * n_onehot_actions, axis=2)
        # 1step目は学習側で計算するので0をいれる
        q = np.insert(q, 0, 0, axis=1)

        # --- calc TD error
        # ファンシーインデックス
        idx1 = [i for i, b in enumerate(batchs) for e in b[4] for e2 in e]
        idx2 = [i for b in batchs for i, e in enumerate(b[4]) for e2 in e]
        idx3 = [e2 for b in batchs for e in b[4] for e2 in e]
        if self.config.enable_double_dqn:
            q_online[idx1, idx2, idx3] = -np.inf
            n_act_idx = np.argmax(q_online, axis=2)
        else:
            q_target[idx1, idx2, idx3] = -np.inf
            n_act_idx = np.argmax(q_target, axis=2)
        maxq = np.take_along_axis(q_target, np.expand_dims(n_act_idx, axis=2), axis=2)
        maxq = np.squeeze(maxq, axis=2)

        if self.config.enable_rescale:
            maxq = inverse_rescaling(maxq)

        gains = rewards + dones * self.config.discount * maxq

        if self.config.enable_rescale:
            gains = rescaling(gains)

        td_errors = gains - q

        # --- calc retrace
        # 各batchで最大のアクションを選んでるかどうか
        # greedyな方策なので、最大アクションなら確率1.0 #[1]
        pi_probs = np.argmax(n_onehot_actions, axis=2) == n_act_idx[:, 1:]

        #  (batch, multistep, shape) -> (multistep, batch, shape)
        # mu_probs = np.transpose(mu_probs, (1, 0)) #[1]
        pi_probs = np.transpose(pi_probs, (1, 0))

        retrace_list = [np.ones((batch_size,))]  # 0stepはretraceなし
        retrace = np.ones((batch_size,))
        for n in range(self.config.multisteps - 1):
            # #[1]
            # pi_probs は 0 or 1 で mu_probs は1以下なので必ず 0 or 1 になる
            # retrace *= self.config.retrace_h * np.minimum(1, pi_probs[n] / mu_probs[n])
            retrace *= self.config.retrace_h * pi_probs[n]
            retrace_list.append(retrace.copy())

        # (multistep, batch, shape) ->  (batch, multistep, shape)
        retrace_list = np.asarray(retrace_list).transpose((1, 0))

        target_q = np.sum(td_errors * multi_discounts * retrace_list, axis=1, dtype=np.float32)

        if training:
            return target_q, states, onehot_actions
        else:
            return target_q


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: CommonInterfaceParameter = self.parameter

        self.dummy_state = np.full(self.config.observation_shape, self.config.dummy_state_val, dtype=np.float32)
        self.onehot_arr = np.identity(self.config.action_num, dtype=int)

        self.epsilon_sch = self.config.epsilon.create_schedulers()

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        self._recent_states = [self.dummy_state for _ in range(self.config.multisteps + 1)]
        self._recent_actions = [
            self.onehot_arr[random.randint(0, self.config.action_num - 1)] for _ in range(self.config.multisteps)
        ]
        # self._recent_probs = [1.0 / self.config.action_num for _ in range(self.config.multisteps)] #[1]
        self._recent_rewards = [0.0 for _ in range(self.config.multisteps)]
        self._recent_done = [1 for _ in range(self.config.multisteps)]
        self._recent_invalid_actions = [[] for _ in range(self.config.multisteps)]

        self._recent_states.pop(0)
        self._recent_states.append(state)

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.state = state

        if self.config.enable_noisy_dense:
            self.q = self.parameter.predict_q(state[np.newaxis, ...])[0]
            self.q[invalid_actions] = -np.inf
            self.action = int(np.argmax(self.q))
            # self.prob = 1.0 #[1]
            return self.action, {}

        if self.training:
            epsilon = self.epsilon_sch.get_and_update_rate(self.total_step)
        else:
            epsilon = self.config.test_epsilon

        # valid_action_num = self.config.action_num - len(invalid_actions)
        if random.random() < epsilon:
            self.action = random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
            self.q = None
            # self.prob = epsilon / valid_action_num #[1]
        else:
            self.q = self.parameter.predict_q(state[np.newaxis, ...])[0]
            self.q[invalid_actions] = -np.inf

            # 最大値を選ぶ（複数はほぼないとして無視）
            self.action = int(np.argmax(self.q))
            # self.prob = epsilon / valid_action_num + (1 - epsilon) #[1]

        return self.action, {"epsilon": epsilon}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ):
        self._recent_states.pop(0)
        self._recent_states.append(next_state)

        if not self.training:
            return {}

        # reward clip
        if self.config.enable_reward_clip:
            if reward < 0:
                reward = -1
            elif reward > 0:
                reward = 1
            else:
                reward = 0

        self._recent_actions.pop(0)
        self._recent_actions.append(self.onehot_arr[self.action])
        # self._recent_probs.pop(0)
        # self._recent_probs.append(self.prob) #[1]
        self._recent_rewards.pop(0)
        self._recent_rewards.append(reward)
        self._recent_done.pop(0)
        self._recent_done.append(int(not done))
        self._recent_invalid_actions.pop(0)
        self._recent_invalid_actions.append(next_invalid_actions)
        priority = self._add_memory(None)

        if done:
            # 残りstepも追加
            for _ in range(len(self._recent_rewards) - 1):
                self._recent_states.pop(0)
                self._recent_states.append(self.dummy_state)
                self._recent_actions.pop(0)
                self._recent_actions.append(self.onehot_arr[random.randint(0, self.config.action_num - 1)])
                # self._recent_probs.pop(0)
                # self._recent_probs.append(1.0) #[1]
                self._recent_rewards.pop(0)
                self._recent_rewards.append(0.0)
                self._recent_done.pop(0)
                self._recent_done.append(0)
                self._recent_invalid_actions.pop(0)
                self._recent_invalid_actions.append([])
                self._add_memory(priority)

        return {}

    def _add_memory(self, priority):
        """
        [
            states,
            onehot_actions,
            # probs,
            rewards,
            dones,
            invalid_actions,
        ]
        """
        batch = [
            self._recent_states[:],
            self._recent_actions[:],
            # self._recent_probs[:], #[1]
            self._recent_rewards[:],
            self._recent_done[:],
            self._recent_invalid_actions[:],
        ]

        if priority is None:
            if not self.distributed:
                priority = None
            elif not self.config.memory.requires_priority():
                priority = None
            else:
                if self.q is None:
                    self.q = self.parameter.predict_q(self.state[np.newaxis, ...])[0]
                select_q = self.q[self.action]
                target_q = self.parameter.calc_target_q([batch], training=False)[0]
                priority = abs(target_q - select_q)

        self.memory.add(batch, priority)
        return priority

    def render_terminal(self, worker, **kwargs) -> None:
        if self.q is None:
            q = self.parameter.predict_q(self.state[np.newaxis, ...])[0]
        else:
            q = self.q
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(maxa, worker.env, self.config, _render_sub)
