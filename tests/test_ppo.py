"""PPO pieces in isolation, plus a two-iteration smoke run on the real hover task."""
import pytest

pytest.importorskip('crazyflow')

import jax
import jax.numpy as jnp
import numpy as np

from drones.rl.networks import ActorCritic
from drones.rl.ppo import (PPO, PPOConfig, config_from_dict, config_to_json, gae, load_params,
                           save_params)
from drones.sim.hover_env import HoverConfig, HoverEnv
from drones.sim.sensors import SensorConfig


def test_gae_without_episode_ends_is_the_discounted_return():
    rewards = jnp.array([[1.0], [2.0], [3.0]])
    values = jnp.zeros((3, 1))
    adv, ret = gae(rewards, values, jnp.zeros((3, 1), bool), jnp.array([4.0]), gamma=0.5, lam=1.0)
    # Backwards from the bootstrap value 4: 3 + .5*4 = 5, 2 + .5*5 = 4.5, 1 + .5*4.5 = 3.25.
    np.testing.assert_allclose(ret[:, 0], [3.25, 4.5, 5.0])
    np.testing.assert_allclose(adv, ret)


def test_gae_does_not_bootstrap_across_an_episode_end():
    rewards = jnp.array([[1.0], [5.0]])
    values = jnp.array([[0.0], [10.0]])
    done = jnp.array([[True], [False]])
    adv, _ = gae(rewards, values, done, jnp.array([0.0]), gamma=0.9, lam=0.95)
    assert float(adv[0, 0]) == pytest.approx(1.0)   # the next episode's value is not used


@pytest.mark.parametrize('use_image', [False, True])
def test_actor_runs_on_deployable_inputs_only(use_image):
    model = ActorCritic(4, (32, 32), use_image=use_image)
    obs = {'policy': jnp.zeros((5, 57)), 'critic': jnp.zeros((5, 70))}
    if use_image:
        obs['image'] = jnp.zeros((5, 24, 32, 3))
    params = model.init(jax.random.key(0), obs)
    mean, log_std, value = model.apply(params, obs)
    assert mean.shape == (5, 4) and value.shape == (5,) and log_std.shape == (4,)
    deployable = {k: v for k, v in obs.items() if k != 'critic'}
    action = model.apply(params, deployable, method=ActorCritic.act)
    np.testing.assert_allclose(action, mean)
    assert float(jnp.abs(action).max()) < 0.1, 'should start near zero action (hover)'


def test_config_round_trips_through_json():
    config = HoverConfig(num_envs=7, room_size=(2.0, 3.0),
                         sensors=SensorConfig(enabled=('multiranger', 'camera'),
                                              camera_resolution=(16, 12)))
    import json
    assert config_from_dict(HoverConfig, json.loads(config_to_json(config))) == config


@pytest.fixture(scope='module')
def trained():
    env = HoverEnv(HoverConfig(num_envs=8))
    ppo = PPO(env, PPOConfig(total_steps=8 * 16 * 2, rollout_steps=16, minibatches=2,
                             epochs=2, hidden=(32, 32)))
    state = ppo.init(jax.random.key(0))
    initial = state.params
    history = []
    for _ in range(ppo.iterations):
        state, stats = ppo.iterate(state)
        history.append(jax.device_get(stats))
    return ppo, initial, state, history


def test_training_iterations_are_finite_and_move_the_weights(trained):
    ppo, initial, state, history = trained
    assert ppo.iterations == 2 and int(state.iteration) == 2
    for stats in history:
        for name in ('policy_loss', 'value_loss', 'approx_kl', 'reward'):
            assert np.isfinite(stats[name]), name
    moved = jax.tree.map(lambda a, b: float(jnp.abs(a - b).max()), initial, state.params)
    assert max(jax.tree.leaves(moved)) > 0


def test_params_round_trip_through_a_checkpoint(trained, tmp_path):
    _, _, state, _ = trained
    save_params(tmp_path / 'p.msgpack', state.params)
    restored = load_params(tmp_path / 'p.msgpack', state.params)
    jax.tree.map(np.testing.assert_array_equal, state.params, restored)
