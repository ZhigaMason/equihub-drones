"""The numpy policy artifact: format, runner, and agreement with the trained flax actor."""
import numpy as np
import pytest

from drones.policy.interface import BASELINE
from drones.policy.runtime import Policy, PolicyRunner, PolicySpec

SPEC = PolicySpec(sensors=BASELINE, history=3, control_freq=50, target_height=(0.5, 1.5),
                  range_max=4.0, flow_gain=0.488, max_tilt=0.35, max_yaw_rate=1.5,
                  hover_thrust=0.44, thrust_min=0.085, thrust_max=0.8)
READING = dict(flow_rate=[0.2, -0.1], zrange=1.0, ranges=[1, 2, 3, 4, 4], gyro=[0, 0, 0],
               gravity=[0, 0, -1])


def random_policy(spec=SPEC, hidden=16, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [spec.observation_size, hidden, hidden, 4]
    return Policy(spec, [(rng.normal(size=(a, b)) * 0.3, rng.normal(size=b) * 0.1)
                         for a, b in zip(sizes, sizes[1:])])


def test_artifact_round_trips(tmp_path):
    policy = random_policy()
    policy.save(tmp_path / 'policy')
    loaded = Policy.load(tmp_path / 'policy')
    assert loaded.spec == SPEC
    obs = np.random.default_rng(1).normal(size=(5, SPEC.observation_size))
    np.testing.assert_array_equal(loaded.act(obs), policy.act(obs))


def test_wrong_input_width_is_rejected():
    with pytest.raises(ValueError, match='inputs'):
        Policy(SPEC, [(np.zeros((7, 4)), np.zeros(4))])


def test_camera_policies_are_refused():
    spec = PolicySpec(**{**SPEC.__dict__, 'sensors': ('multiranger', 'camera')})
    with pytest.raises(ValueError, match='camera'):
        Policy(spec, [(np.zeros((spec.observation_size, 4)), np.zeros(4))])


def test_runner_seeds_history_with_the_first_frame_and_feeds_back_actions():
    runner = PolicyRunner(random_policy(), target_height=1.0)
    first = runner.step(**READING)
    assert runner.history.shape == (3, SPEC.frame_size)
    np.testing.assert_array_equal(runner.history[0], runner.history[2])
    runner.step(**READING)
    np.testing.assert_allclose(runner.history[-1, -4:], first, rtol=1e-6)


def test_exported_actor_matches_the_flax_actor(tmp_path):
    pytest.importorskip('crazyflow')
    import jax
    import jax.numpy as jnp

    from drones.rl.export import export_policy
    from drones.rl.networks import ActorCritic
    from drones.sim.hover_env import HoverConfig, HoverEnv

    env = HoverEnv(HoverConfig(num_envs=2))
    model = ActorCritic(4, (32, 32))
    _, obs = env.reset(jax.random.key(0))
    params = model.init(jax.random.key(3), obs)
    # Make the actor output non-trivial so the comparison means something.
    params = jax.tree.map(lambda x: x * 30 if x.ndim == 2 and x.shape[1] == 4 else x, params)
    policy = Policy.load(export_policy(env, params, tmp_path / 'policy'))

    assert policy.spec.sensors == BASELINE and policy.spec.hover_thrust == env.hover_thrust
    expected = np.clip(np.asarray(model.apply(params, obs, method=ActorCritic.act)), -1, 1)
    assert np.abs(expected).max() > 0.05
    np.testing.assert_allclose(policy.act(np.asarray(obs['policy'])), expected, atol=1e-5)
    assert not bool(jnp.isnan(obs['policy']).any())
