"""SHAC pieces in isolation, a hand-checked rollout loss, and learning on the square task."""
import pytest

pytest.importorskip('crazyflow')

import jax
import jax.numpy as jnp
import numpy as np

from drones.rl.shac import SHAC, SHACConfig, td_lambda_returns
from drones.sim.square_env import SquareConfig, SquareEnv


def test_td_lambda_one_is_the_discounted_return():
    ret = td_lambda_returns(jnp.array([[1.0], [2.0], [3.0]]), jnp.array([[0.0], [0.0], [4.0]]),
                            jnp.zeros((3, 1), bool), jnp.zeros((3, 1), bool), gamma=0.5, lam=1.0)
    np.testing.assert_allclose(ret[:, 0], [3.25, 4.5, 5.0])


def test_td_lambda_zero_is_one_step_td():
    ret = td_lambda_returns(jnp.array([[1.0], [2.0]]), jnp.array([[10.0], [20.0]]),
                            jnp.zeros((2, 1), bool), jnp.zeros((2, 1), bool), gamma=0.5, lam=0.0)
    np.testing.assert_allclose(ret[:, 0], [6.0, 12.0])


@pytest.mark.parametrize('crashed, expected', [(True, 1.0), (False, 5.0)])
def test_a_crash_has_no_future_and_a_timeout_does(crashed, expected):
    done = jnp.array([[True], [False]])
    ret = td_lambda_returns(jnp.array([[1.0], [1.0]]), jnp.array([[8.0], [8.0]]), done,
                            done & crashed, gamma=0.5, lam=0.95)
    assert float(ret[0, 0]) == pytest.approx(expected)


class LineEnv:
    """One world on a line: the action moves it and the reward is where it is. Each episode ends
    after two steps, in a crash or a timeout as chosen."""
    num_envs, action_size = 1, 1

    def __init__(self, crash):
        self.crash = crash

    @staticmethod
    def _obs(x):
        return {'policy': x[:, None], 'critic': x[:, None]}

    def reset(self, key):
        state = {'x': jnp.zeros(1), 't': jnp.zeros(1, jnp.int32)}
        return state, self._obs(state['x'])

    def step(self, state, action):
        x, t = state['x'] + action[:, 0], state['t'] + 1
        done = t >= 2
        crashed = done & self.crash
        zero = jnp.zeros(1)
        info = {'crashed': crashed, 'truncated': done & ~crashed, 'final_critic': x[:, None],
                'episode_return': zero, 'episode_length': jnp.zeros(1, jnp.int32),
                'pos_error': zero, 'speed': zero, 'tilt': zero}
        reward = x
        x, t = jnp.where(done, 0.0, x), jnp.where(done, 0, t)
        return {'x': x, 't': t}, self._obs(x), reward, done, info


@pytest.mark.parametrize('crash, expected_loss', [(True, -0.5 / 3), (False, -0.75 / 3)])
def test_rollout_bootstraps_after_a_timeout_but_not_a_crash(crash, expected_loss):
    # Zero actions, so every reward is 0 and the loss is the bootstrapped values alone. The target
    # critic returns 1 everywhere. With gamma 0.5 over 3 steps and an episode ending at step 2:
    # a crash leaves only the window-end value, 0.5 * 1; a timeout adds 0.25 * 1 for its end.
    # critic_minibatches=1 is a no-op here: this test only calls `_rollout`, never `_fit_critic`,
    # but SHAC.__init__ unconditionally checks divisibility against the default of 4, which
    # LineEnv's batch size of num_envs(1) * horizon(3) = 3 does not satisfy.
    agent = SHAC(LineEnv(crash), SHACConfig(horizon=3, gamma=0.5, hidden=(4,), remat=False,
                                            critic_minibatches=1))
    state = agent.init(jax.random.key(0))
    path_str = jax.tree_util.keystr
    actor = jax.tree_util.tree_map_with_path(
        lambda p, x: jnp.full_like(x, -30.0) if 'log_std' in path_str(p) else jnp.zeros_like(x),
        state.actor)
    target = jax.tree_util.tree_map_with_path(
        lambda p, x: (jnp.ones_like(x) if "'Dense_1'" in path_str(p) and "'bias'" in path_str(p)
                      else jnp.zeros_like(x)), state.critic)
    loss, _ = agent._rollout(actor, target, state.env_state, state.obs, jax.random.key(1))
    assert float(loss) == pytest.approx(expected_loss, abs=1e-6)


@pytest.fixture(scope='module')
def square():
    return SquareEnv(SquareConfig(num_envs=8))


def test_one_iteration_runs_and_updates(square):
    agent = SHAC(square, SHACConfig(iterations=2, horizon=4, critic_epochs=2,
                                    critic_minibatches=2, hidden=(16, 16)))
    state = agent.init(jax.random.key(0))
    new, stats = agent.iterate(state)
    assert all(bool(jnp.isfinite(v)) for k, v in stats.items()
               if k not in ('episode_return', 'episode_length', 'crash_rate'))
    assert float(stats['skipped']) == 0.0
    changed = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), new.actor, state.actor)
    assert any(jax.tree.leaves(changed))
    assert int(new.iteration) == 1


def test_shac_learns_to_track_the_square():
    # Iteration 0 sees only the first 0.64 s of each episode, before a hover-like policy drifts off
    # the square, so the baseline is the early-training window, not iteration 0.
    env = SquareEnv(SquareConfig(num_envs=32))
    # The learning-rate schedule spans 80 iterations; only the first 40 are run.
    agent = SHAC(env, SHACConfig(iterations=80, horizon=32, hidden=(64, 64)))
    state = agent.init(jax.random.key(0))
    rewards, errors = [], []
    for _ in range(40):
        state, stats = agent.iterate(state)
        rewards.append(float(stats['reward']))
        errors.append(float(stats['pos_error']))
    assert np.mean(rewards[-5:]) > np.mean(rewards[5:15]) + 0.2
    assert np.mean(errors[-5:]) < 0.2
