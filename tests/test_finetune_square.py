"""drones-finetune-square: how a fit reshapes the training setup."""
import pytest

pytest.importorskip('crazyflow')

from drones.rl.finetune_square import finetuned_configs
from drones.rl.shac import SHACConfig
from drones.rl.sysid import SysIdResult
from drones.sim.square_env import SquareConfig


def test_the_fit_centres_the_randomisation_and_slows_learning():
    result = SysIdResult(thrust_gain=0.87, latency=1, residual={}, report={})
    env, shac = finetuned_configs(SquareConfig(num_envs=4096, device='gpu'), SHACConfig(),
                                  result, iterations=200, num_envs=256, device='cpu')
    assert env.thrust_gain == 0.87 and env.thrust_gain_range == pytest.approx(0.05)
    assert env.latency_steps == (1,)
    assert env.num_envs == 256 and env.device == 'cpu'
    assert shac.iterations == 200
    assert shac.actor_lr == pytest.approx(SHACConfig().actor_lr / 4)
    assert shac.critic_lr == pytest.approx(SHACConfig().critic_lr / 4)
