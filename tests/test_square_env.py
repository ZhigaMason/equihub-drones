"""The square task: episodes, observations, differentiability, randomisation, residual hook."""
import dataclasses

import pytest

pytest.importorskip('crazyflow')

import jax
import jax.numpy as jnp
import numpy as np

from drones.policy.square import OBS_SIZE
from drones.sim.geometry import euler_to_quat, yaw_from_quat
from drones.sim.residual import FORCE_SCALE, init_residual
from drones.sim.square_env import PRIVILEGED_SIZE, SquareConfig, SquareEnv

N = 8
QUIET = SquareConfig(num_envs=N, thrust_gain_range=0.0, latency_steps=(0,), pos_noise=0.0,
                     pos_drift=0.0, vel_noise=0.0, attitude_noise=0.0)


@pytest.fixture(scope='module')
def env():
    return SquareEnv(SquareConfig(num_envs=N))


@pytest.fixture(scope='module')
def quiet():
    return SquareEnv(QUIET)


def hold_still(state, z=1.0):
    """Level, motionless at (0, 0, z), facing +x, with an empty action buffer."""
    s = state.sim.states
    states = s.replace(pos=jnp.zeros_like(s.pos).at[..., 2].set(z), vel=jnp.zeros_like(s.vel),
                       ang_vel=jnp.zeros_like(s.ang_vel),
                       quat=jnp.zeros_like(s.quat).at[..., 3].set(1.0))
    return state.replace(sim=state.sim.replace(states=states), yaw_cmd=jnp.zeros(N),
                         ref_yaw=jnp.zeros(N), actions=jnp.zeros_like(state.actions))


def fly(env, state, action, steps):
    for _ in range(steps):
        state, obs, reward, done, info = env.step(state, jnp.broadcast_to(action, (N, 4)))
    return state, obs, reward, done, info


def test_observations_have_the_right_size(env):
    _, obs = env.reset(jax.random.key(0))
    assert obs['policy'].shape == (N, OBS_SIZE)
    assert obs['critic'].shape == (N, OBS_SIZE + PRIVILEGED_SIZE)
    assert all(bool(jnp.isfinite(v).all()) for v in obs.values())


def test_episodes_start_on_the_reference_at_the_origin(quiet):
    state, _ = quiet.reset(jax.random.key(1))
    ref, _ = quiet.reference_at(state, state.phase)
    np.testing.assert_allclose(ref[:, :2], 0.0, atol=1e-5)
    assert float(jnp.abs(state.sim.states.pos[:, 0] - ref).max()) <= 0.1 + 1e-6


def test_randomisation_stays_in_its_ranges(env):
    state, _ = env.reset(jax.random.key(2))
    assert bool(((state.thrust_gain >= 0.9) & (state.thrust_gain <= 1.1)).all())
    assert set(np.asarray(state.latency).tolist()) <= {0, 1, 2}
    assert bool(((state.lap_time >= 6.0) & (state.lap_time <= 10.0)).all())
    assert set(np.asarray(state.direction).tolist()) <= {-1.0, 1.0}
    assert bool(((state.origin[:, 2] >= 0.8) & (state.origin[:, 2] <= 1.2)).all())


def test_zero_action_from_the_start_stays_near_the_reference(quiet):
    state, _ = quiet.reset(jax.random.key(3))
    for _ in range(25):   # 0.5 s
        state, _, _, done, info = quiet.step(state, jnp.zeros((N, 4)))
        assert not bool(done.any())
        assert float(info['pos_error'].max()) < 0.5


def test_rollout_gradient_matches_finite_differences(quiet):
    state, _ = quiet.reset(jax.random.key(4))

    def total_reward(action):
        def body(s, _):
            s, _, reward, _, _ = quiet.step(s, jnp.broadcast_to(action, (N, 4)))
            return s, reward
        _, rewards = jax.lax.scan(body, state, None, length=16)
        return rewards.sum()

    a0 = jnp.array([0.05, -0.05, 0.02, 0.03])
    grad = jax.grad(total_reward)(a0)
    assert bool(jnp.isfinite(grad).all())
    eps = 1e-3
    numeric = jnp.array([(total_reward(a0.at[i].add(eps)) - total_reward(a0.at[i].add(-eps)))
                         / (2 * eps) for i in range(4)])
    np.testing.assert_allclose(grad, numeric, rtol=0.05, atol=0.05)


def test_a_restarted_world_carries_no_gradient_from_its_last_episode(quiet):
    state, _ = quiet.reset(jax.random.key(5))

    def world0_height_after_step(z0):
        pos = state.sim.states.pos.at[0, 0, 2].set(z0)
        s = state.replace(sim=state.sim.replace(states=state.sim.states.replace(pos=pos)))
        s, _, _, done, _ = quiet.step(s, jnp.zeros((N, 4)))
        return s.sim.states.pos[0, 0, 2], done[0]

    assert bool(world0_height_after_step(0.05)[1]), 'below min_height: crashed and restarted'
    assert float(jax.grad(lambda z: world0_height_after_step(z)[0])(0.05)) == 0.0
    assert float(jax.grad(lambda z: world0_height_after_step(z)[0])(1.0)) == pytest.approx(1.0,
                                                                                          abs=0.01)


def test_thrust_gain_scales_the_lift(quiet):
    state = hold_still(quiet.reset(jax.random.key(6))[0])
    level, *_ = fly(quiet, state, jnp.zeros(4), 10)
    strong, *_ = fly(quiet, state.replace(thrust_gain=jnp.full(N, 1.2)), jnp.zeros(4), 10)
    assert float(jnp.abs(level.sim.states.vel[:, 0, 2]).max()) < 0.02
    assert float(strong.sim.states.vel[:, 0, 2].min()) > 0.3


def test_latency_delays_the_action():
    env = SquareEnv(dataclasses.replace(QUIET, latency_steps=(2,)))
    state = hold_still(env.reset(jax.random.key(7))[0])
    roll = jnp.array([0.5, 0.0, 0.0, 0.0])
    state, *_ = fly(env, state, roll, 2)
    assert float(jnp.abs(state.sim.states.ang_vel).max()) < 1e-6, 'still flying the old zeros'
    state, *_ = fly(env, state, roll, 1)
    assert float(jnp.abs(state.sim.states.ang_vel).max()) > 0.1


def test_the_residual_force_pushes_the_drone():
    def set_bias(path, x):
        name = jax.tree_util.keystr(path)
        return jnp.array([1.0, 0, 0, 0, 0, 0]) if "'out'" in name and "'bias'" in name else x
    params = jax.tree_util.tree_map_with_path(set_bias, init_residual(jax.random.key(0)))
    env = SquareEnv(QUIET, residual=params)
    state = hold_still(env.reset(jax.random.key(8))[0])
    state, *_ = fly(env, state, jnp.zeros(4), 10)   # 0.2 s
    expected = FORCE_SCALE / env.mass * 0.2
    np.testing.assert_allclose(state.sim.states.vel[:, 0, 0], expected, rtol=0.1)


def test_holding_a_far_heading_does_not_spin(quiet):
    """so_rpy's fitted yaw model is not unit-gain (see SquareEnv.yaw_gain); without compensating
    for it, a heading far from zero drives the yaw-setpoint band to its limit and spins up."""
    heading = 2.5
    state = hold_still(quiet.reset(jax.random.key(9))[0])
    quat = euler_to_quat(jnp.zeros(N), jnp.zeros(N), jnp.full(N, heading))[:, None]
    state = state.replace(sim=state.sim.replace(states=state.sim.states.replace(quat=quat)),
                          yaw_cmd=jnp.full(N, heading), ref_yaw=jnp.full(N, heading))
    state, *_ = fly(quiet, state, jnp.zeros(4), 100)   # 2 s
    yaw = yaw_from_quat(state.sim.states.quat[:, 0])
    assert float(jnp.abs(yaw - heading).max()) < 0.05
    assert float(jnp.abs(state.sim.states.ang_vel[:, 0, 2]).max()) < 0.1
