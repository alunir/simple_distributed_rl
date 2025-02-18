from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from srl.base.exception import UndefinedError


@dataclass
class ImageBlockConfig:
    _name: str = field(init=False, default="DQN")
    _kwargs: Dict[str, Any] = field(init=False, default_factory=lambda: {})

    def set_dqn_image(
        self,
        filters: int = 32,
        activation: str = "relu",
    ):
        """画像の入力に対してDQNで採用されたLayersを使用します。

        Args:
            filters (int, optional): 基準となるfilterの数です. Defaults to 32.
            activation (str, optional): activation function. Defaults to "relu".
        """
        self._name = "DQN"
        self._kwargs = dict(
            filters=filters,
            activation=activation,
        )

    def set_r2d3_image(
        self,
        filters: int = 16,
        activation: str = "relu",
    ):
        """画像の入力に対してR2D3で採用されたLayersを使用します。

        Args:
            filters (int, optional): 基準となるfilterの数です. Defaults to 32.
            activation (str, optional): activation function. Defaults to "relu".
        """
        self._name = "R2D3"
        self._kwargs = dict(
            filters=filters,
            activation=activation,
        )

    def set_custom_block(self, entry_point: str, kwargs: dict):
        self._name = "custom"
        self._kwargs = dict(
            entry_point=entry_point,
            kwargs=kwargs,
        )

    # ---------------------

    def create_block_tf(self, enable_time_distributed_layer: bool = False):
        if self._name == "DQN":
            from .tf import dqn_image_block

            return dqn_image_block.DQNImageBlock(
                enable_time_distributed_layer=enable_time_distributed_layer,
                **self._kwargs,
            )
        if self._name == "R2D3":
            from .tf import r2d3_image_block

            return r2d3_image_block.R2D3ImageBlock(
                enable_time_distributed_layer=enable_time_distributed_layer,
                **self._kwargs,
            )

        if self._name == "custom":
            from srl.utils.common import load_module

            return load_module(self._kwargs["entry_point"])(**self._kwargs["kwargs"])

        raise UndefinedError(self._name)

    def create_block_torch(
        self,
        in_shape: Tuple[int, ...],
        enable_time_distributed_layer: bool = False,
    ):
        if self._name == "DQN":
            from .torch_ import dqn_image_block

            return dqn_image_block.DQNImageBlock(
                in_shape,
                enable_time_distributed_layer=enable_time_distributed_layer,
                **self._kwargs,
            )
        if self._name == "R2D3":
            from .torch_ import r2d3_image_block

            return r2d3_image_block.R2D3ImageBlock(
                in_shape,
                enable_time_distributed_layer=enable_time_distributed_layer,
                **self._kwargs,
            )

        if self._name == "custom":
            from srl.utils.common import load_module

            return load_module(self._kwargs["entry_point"])(**self._kwargs["kwargs"])

        raise UndefinedError(self._name)
