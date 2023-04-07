import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    import srl.envs.ox  # noqa F401
    from srl.algorithms import dqn_torch
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("torch"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = dqn_torch.Config()

    def test_Pendulum(self):
        self.rl_config.hidden_layer_sizes = (64, 64)
        self.rl_config.enable_double_dqn = False
        self.tester.verify_1player("Pendulum-v1", self.rl_config, 200 * 100)

    def test_Pendulum_mp(self):
        self.rl_config.hidden_layer_sizes = (64, 64)
        self.tester.verify_1player("Pendulum-v1", self.rl_config, 200 * 100, is_mp=True)

    def test_Pendulum_DDQN(self):
        self.rl_config.hidden_layer_sizes = (64, 64)
        self.tester.verify_1player("Pendulum-v1", self.rl_config, 200 * 70)

    def test_Pendulum_window(self):
        self.rl_config.window_length = 4
        self.rl_config.hidden_layer_sizes = (64, 64)
        self.tester.verify_1player("Pendulum-v1", self.rl_config, 200 * 70)

    def test_OX(self):
        self.rl_config.hidden_layer_sizes = (128,)
        self.rl_config.epsilon = 0.5
        self.tester.verify_2player("OX", self.rl_config, 10000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_Pendulum_window", verbosity=2)
