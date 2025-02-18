import numpy as np
import pytest


def test_call_tf():
    pytest.importorskip("tensorflow")

    action_num = 5
    dense_units = 32
    batch_size = 16

    from srl.rl.models.dueling_network import DuelingNetworkConfig

    config = DuelingNetworkConfig()
    config.set((dense_units,), True)

    block = config.create_block_tf(action_num)

    x = np.ones((batch_size, 128), dtype=np.float32)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, action_num)


def test_noisy():
    pytest.importorskip("tensorflow")

    action_num = 5
    dense_units = 32
    batch_size = 16

    from srl.rl.models.dueling_network import DuelingNetworkConfig

    config = DuelingNetworkConfig()
    config.set((dense_units,), True)

    block = config.create_block_tf(action_num, enable_noisy_dense=True)

    x = np.ones((batch_size, 128), dtype=np.float32)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, action_num)


def test_disable():
    pytest.importorskip("tensorflow")

    action_num = 5
    dense_units = 32
    batch_size = 16

    from srl.rl.models.dueling_network import DuelingNetworkConfig

    config = DuelingNetworkConfig()
    config.set((dense_units,), False)

    block = config.create_block_tf(action_num)

    x = np.ones((batch_size, 128), dtype=np.float32)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, action_num)
