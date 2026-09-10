"""The hover task: episodes, action scaling, termination and autoreset."""
import pytest

pytest.importorskip('crazyflow')

import jax
import jax.numpy as jnp
import numpy as np

from drones.sim.hover_env import CRASH_REWARD, PRIVILEGED_SIZE, HoverConfig, HoverEnv
from drones.sim.sensors import SensorConfig

N = 8


@pytest.fixture(scope='module')
def env():
    return HoverEnv(HoverConfig(num_envs=N))


def hold_still(state, z=1.0):
    """Level, motionless, at the room centre and height z, facing +x."""
    s = state.sim.states
    states = s.replace(
        pos=jnp.zeros_like(s.pos).at[..., 2].set(z), vel=jnp.zeros_like(s.vel),
        ang_vel=jnp.zeros_like(s.ang_vel), quat=jnp.zeros_like(s.quat).at[..., 3].set(1.0))
    return state.replace(sim=state.sim.replace(states=states), yaw_cmd=jnp.zeros(N))


def fly(env, state, action, steps):
    for _ in range(steps):
        state, obs, reward, done, info = env.step(state, jnp.broadcast_to(action, (N, 4)))
    return state, obs, reward, done, info


def test_reset_gives_finite_observations_of_the_right_size(env):
    state, obs = env.reset(jax.random.key(0))
    assert env.frame_size == 13, 'baseline: flow deck (3) + multi-ranger (5) + task (5)'
    assert obs['policy'].shape == (N, 3 * env.frame_size)
    assert obs['critic'].shape == (N, 3 * env.frame_size + PRIVILEGED_SIZE)
    assert 'image' not in obs
    assert all(bool(jnp.isfinite(v).all()) for v in obs.values())


def test_episodes_start_inside_the_room_below_the_ceiling(env):
    state, _ = env.reset(jax.random.key(1))
    pos = state.sim.states.pos[:, 0]
    assert bool((jnp.abs(pos[:, :2]) < state.room[:, :2]).all())
    assert bool((pos[:, 2] < state.room[:, 2]).all())
    assert bool((state.target <= state.room[:, 2] - env.config.ceiling_margin + 1e-6).all())


def test_same_key_same_episode(env):
    _, a = env.reset(jax.random.key(7))
    _, b = env.reset(jax.random.key(7))
    np.testing.assert_array_equal(a['policy'], b['policy'])


def test_zero_action_is_calibrated_hover(env):
    assert env.thrust_min < env.hover_thrust < env.thrust_max
    state, _ = env.reset(jax.random.key(2))
    state, *_ = fly(env, hold_still(state), jnp.zeros(4), 50)   # one second
    z = state.sim.states.pos[:, 0, 2]
    assert float(jnp.abs(z - 1.0).max()) < 0.05


@pytest.mark.parametrize('thrust, direction', [(0.5, 1), (-0.5, -1)])
def test_thrust_action_climbs_and_descends(env, thrust, direction):
    state, _ = env.reset(jax.random.key(3))
    state, *_ = fly(env, hold_still(state), jnp.array([0.0, 0.0, 0.0, thrust]), 10)
    dz = state.sim.states.pos[:, 0, 2] - 1.0
    assert bool((direction * dz > 0.02).all())


def test_thrust_extremes_hit_the_motor_limits(env):
    quat = jnp.zeros((N, 4)).at[:, 3].set(1.0)
    for a, limit in [(1.0, env.thrust_max), (-1.0, env.thrust_min)]:
        cmd, _ = env._command(jnp.full((N, 4), a), jnp.zeros(N), quat)
        assert float(cmd[0, 0, 3]) == pytest.approx(limit)


def test_crash_into_the_floor_restarts_that_world_in_the_same_step(env):
    state, _ = env.reset(jax.random.key(4))
    state = hold_still(state)
    s = state.sim.states
    state = state.replace(sim=state.sim.replace(states=s.replace(pos=s.pos.at[0, 0, 2].set(0.05))))
    state, obs, reward, done, info = env.step(state, jnp.zeros((N, 4)))
    assert bool(done[0]) and bool(info['crashed'][0])
    assert float(reward[0]) == CRASH_REWARD
    assert int(state.steps[0]) == 0, 'crashed world should already be on a new episode'
    assert float(state.sim.states.pos[0, 0, 2]) >= 0.25
    assert not bool(done[1:].any())


def test_flying_always_beats_crashing():
    # The failure this guards against: per-step penalties outweighing the crash penalty, so
    # training learned to crash early. Every flying state, however bad, must score >= 0.
    key = jax.random.split(jax.random.key(0), 6)
    n = 4096
    reward = HoverEnv._reward(
        pos=jax.random.uniform(key[0], (n, 3), minval=-3, maxval=3),
        vel=jax.random.uniform(key[1], (n, 3), minval=-5, maxval=5),
        ang_vel=jax.random.uniform(key[2], (n, 3), minval=-20, maxval=20),
        tilt=jax.random.uniform(key[3], (n,), minval=0, maxval=1.0),
        side_gap=jax.random.uniform(key[4], (n,), minval=0, maxval=2),
        target=jnp.full(n, 1.0),
        action=jax.random.uniform(key[5], (n, 4), minval=-1, maxval=1),
        prev_action=-jnp.ones((n, 4)),
        crashed=jnp.zeros(n, bool))
    assert float(reward.min()) >= 0.0
    assert float(HoverEnv._reward(jnp.zeros((1, 3)), jnp.zeros((1, 3)), jnp.zeros((1, 3)),
                                  jnp.zeros(1), jnp.ones(1), jnp.ones(1), jnp.zeros((1, 4)),
                                  jnp.zeros((1, 4)), jnp.ones(1, bool))[0]) == CRASH_REWARD < 0


def test_episodes_truncate_at_the_time_limit():
    env = HoverEnv(HoverConfig(num_envs=4, episode_seconds=0.1))
    state, _ = env.reset(jax.random.key(5))
    for step in range(env.config.episode_steps):
        state, _, _, done, info = env.step(hold_still(state).replace(yaw_cmd=jnp.zeros(4))
                                           if step == 0 else state, jnp.zeros((4, 4)))
    assert bool(done.all()) and bool(info['truncated'].all())
    assert bool((info['episode_length'] == env.config.episode_steps).all())


def test_rollout_compiles_under_scan_and_stays_finite(env):
    state, _ = env.reset(jax.random.key(6))

    def body(carry, key):
        state = carry
        action = jax.random.uniform(key, (N, 4), minval=-1, maxval=1)
        state, obs, reward, done, _ = env.step(state, action)
        return state, (obs['policy'], reward)

    _, (obs, reward) = jax.lax.scan(body, state, jax.random.split(jax.random.key(0), 30))
    assert bool(jnp.isfinite(obs).all()) and bool(jnp.isfinite(reward).all())


def test_every_sensor_including_the_camera():
    enabled = ('multiranger', 'optical_flow', 'imu', 'camera')
    env = HoverEnv(HoverConfig(num_envs=2, sensors=SensorConfig(enabled=enabled,
                                                                 camera_resolution=(8, 6))))
    state, obs = env.reset(jax.random.key(0))
    assert env.image_shape == (6, 8, 3)
    assert obs['policy'].shape == (2, 3 * (3 + 5 + 6 + 5))
    assert obs['image'].shape == (2, 6, 8, 3)
    _, obs, *_ = env.step(state, jnp.zeros((2, 4)))
    assert float(obs['image'].min()) >= 0.0 and float(obs['image'].max()) <= 1.0
