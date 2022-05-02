import logging
from abc import abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from srl.base.define import RLActionType, RLObservationType
from srl.base.rl.base import RLConfig, RLWorker

logger = logging.getLogger(__name__)


class TableConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.DISCRETE

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.DISCRETE

    def _set_config_by_env(self, env: "srl.base.rl.env_for_rl.EnvForRL") -> None:
        self._nb_actions = env.action_space.n

    @property
    def nb_actions(self) -> int:
        return self._nb_actions


class TableWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> None:
        raise NotImplementedError()

    def on_reset(
        self,
        state: np.ndarray,
        player_index: int,
        env: "srl.base.rl.env_for_rl.EnvForRL",
    ) -> None:
        self.call_on_reset(state, env.get_invalid_actions(player_index))

    @abstractmethod
    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> int:
        raise NotImplementedError()

    def policy(
        self,
        state: np.ndarray,
        player_index: int,
        env: "srl.base.env.env_for_rl.EnvForRL",
    ) -> Any:
        return self.call_policy(state, env.get_invalid_actions(player_index))

    @abstractmethod
    def call_on_step(
        self,
        next_state: Any,
        reward: float,
        done: bool,
        invalid_actions: List[int],
    ) -> Dict[str, Union[float, int]]:  # info
        raise NotImplementedError()

    def on_step(
        self,
        next_state: Any,
        reward: float,
        done: bool,
        player_index: int,
        env: "srl.base.env.env_for_rl.EnvForRL",
    ) -> Dict[str, Union[float, int]]:  # info
        return self.call_on_step(next_state, reward, done, env.get_invalid_actions(player_index))


if __name__ == "__main__":
    pass
