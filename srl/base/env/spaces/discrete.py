from typing import Any, List, Tuple

import numpy as np

from srl.base.define import ContinuousAction, DiscreteAction, RLActionType, RLObservation

from .space import SpaceBase


class DiscreteSpace(SpaceBase[int]):
    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def n(self) -> int:
        return self._n

    def sample(self, invalid_actions: List[int] = []) -> int:
        assert len(invalid_actions) < self.n, f"No valid actions. {invalid_actions}"
        return int(np.random.choice([a for a in range(self.n) if a not in invalid_actions]))

    def convert(self, val: Any) -> int:
        if isinstance(val, list):
            return int(np.round(val[0]))
        elif isinstance(val, tuple):
            return int(np.round(val[0]))
        return int(np.round(val))

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, int):
            return False
        if val < 0:
            return False
        if val >= self.n:
            return False
        return True

    @property
    def base_action_type(self) -> RLActionType:
        return RLActionType.DISCRETE

    def __eq__(self, o: object) -> bool:
        return self.n == o.n

    def __str__(self) -> str:
        return f"Discrete({self.n})"

    # --- action discrete
    def get_action_discrete_info(self) -> int:
        return self.n

    def action_discrete_encode(self, val: int) -> DiscreteAction:
        return val

    def action_discrete_decode(self, val: DiscreteAction) -> int:
        return val

    # --- action continuous
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        return 1, np.array([0]), np.array([self.n - 1])

    # def action_continuous_encode(self, val: int) -> ContinuousAction:
    #    return [float(val)]

    def action_continuous_decode(self, val: ContinuousAction) -> int:
        return int(np.round(val[0]))

    # --- observation
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (1,)

    def observation_discrete_encode(self, val: int) -> RLObservation:
        return np.array([val], dtype=int)

    def observation_continuous_encode(self, val: int) -> RLObservation:
        return np.array([val], dtype=np.float32)
