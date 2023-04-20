import numpy as np
import pytest

from srl.rl.models.alphazero_image_block_config import AlphaZeroImageBlockConfig
from srl.utils.common import is_package_installed


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
def test_call_tf():
    config = AlphaZeroImageBlockConfig()
    batch_size = 16
    x = np.ones((batch_size, 96, 72, 3), dtype=np.float32)

    block = config.create_block_tf()
    y = block(x).numpy()

    assert y.shape == (batch_size, 96, 72, 256)


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
def test_call_torch():
    import torch

    config = AlphaZeroImageBlockConfig()
    batch_size = 16
    x = np.ones((batch_size, 3, 96, 72), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:])
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 256, 96, 72)
    assert block.out_shape == (256, 96, 72)