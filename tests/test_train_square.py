"""drones-train-square end to end on a tiny run, then drones-eval-square's pieces on its output."""
import pytest

pytest.importorskip('crazyflow')

import jax

from drones.policy.runtime import Policy, SquareSpec
from drones.rl.evaluate_square import evaluate_square, load_square_run
from drones.rl.train_square import main


def test_a_tiny_run_writes_everything_and_evaluates(tmp_path):
    main(['--preset', 'cpu-test', '--set', 'env.num_envs=4', '--set', 'shac.iterations=2',
          '--set', 'shac.horizon=4', '--set', 'shac.critic_minibatches=2',
          '--runs', str(tmp_path), '--name', 'tiny'])
    run = tmp_path / 'tiny'
    for name in ('config.yaml', 'config.json', 'metrics.csv', 'params.msgpack'):
        assert (run / name).exists(), name
    assert isinstance(Policy.load(run / 'policy').spec, SquareSpec)

    env, agent, params = load_square_run(run, num_envs=4, device='cpu')
    result = evaluate_square(env, jax.jit(lambda o: agent.act(params['actor'], o)),
                             jax.random.key(0))
    assert set(result) == {'crash_rate', 'survived_seconds', 'laps', 'pos_rmse_m',
                           'max_error_m', 'return'}
    assert 0.0 <= result['crash_rate'] <= 1.0
