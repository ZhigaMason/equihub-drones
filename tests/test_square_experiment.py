"""Square experiment configs: defaults, presets, overrides, and a round trip through YAML."""
import pytest

pytest.importorskip('crazyflow')

from drones.rl import square_experiment
from drones.rl.shac import SHACConfig
from drones.sim.square_env import SquareConfig


def test_the_default_config_is_sized_for_a_gpu():
    env, shac = square_experiment.resolve()
    assert isinstance(env, SquareConfig) and isinstance(shac, SHACConfig)
    assert env.num_envs == 4096 and env.side == 1.0 and env.latency_steps == (0, 1, 2)


def test_presets_and_overrides_apply_in_order():
    env, shac = square_experiment.resolve(preset='cpu-test',
                                          assignments=['shac.horizon=16', 'env.lap_time=[8, 8]'])
    assert env.num_envs == 32 and shac.iterations == 20
    assert shac.horizon == 16 and env.lap_time == (8.0, 8.0)


def test_hover_sections_are_refused(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('ppo:\n  total_steps: 5\n')
    with pytest.raises(ValueError, match='unknown sections'):
        square_experiment.resolve(path)


def test_dumped_config_resolves_to_the_same_thing(tmp_path):
    env, shac = square_experiment.resolve(preset='cpu', device='cpu')
    square_experiment.dump(env, shac, tmp_path / 'config.yaml')
    assert square_experiment.resolve(tmp_path / 'config.yaml') == (env, shac)
