"""System identification against flights synthesised in the simulator with known errors."""
import math

import pytest

pytest.importorskip('crazyflow')

import jax
import jax.numpy as jnp
import numpy as np

from drones.missions.fly_square import square_recorder
from drones.rl.sysid import SysIdConfig, identify, load_flight, load_flights
from drones.sim.square_env import SquareConfig, SquareEnv

TRUE_GAIN, TRUE_LATENCY = 0.85, 1
TRUE_WORLD = SquareConfig(num_envs=1, thrust_gain=TRUE_GAIN, thrust_gain_range=0.0,
                          latency_steps=(TRUE_LATENCY,), pos_noise=0.0, pos_drift=0.0,
                          vel_noise=0.0, attitude_noise=0.0, episode_seconds=60.0)


def pd_action(pos, vel, yaw, ref_pos, ref_vel, ref_yaw):
    """A plain PD position controller in the policy's action units."""
    acc = 4.0 * (ref_pos - pos) + 3.0 * (ref_vel - vel)
    c, s = math.cos(yaw), math.sin(yaw)
    forward, left = c * acc[0] + s * acc[1], -s * acc[0] + c * acc[1]
    roll, pitch = -left / 9.81, forward / 9.81       # +roll moves right, +pitch forward
    thrust = 0.2 + 1.2 * acc[2] / 9.81               # 0.2 is about hover at a gain of 0.85
    yaw_rate = 2.0 * math.remainder(ref_yaw - yaw, 2 * math.pi) / 1.5
    return np.clip([roll / 0.35, pitch / 0.35, yaw_rate, thrust], -1.0, 1.0)


def synthesise_flight(env, path, seed, seconds=12.0):
    """Fly the square in the simulator with a weak thrust and one step of latency, under the PD
    controller plus exploration noise, and log it through drones-fly-square's own recorder."""
    state, _ = env.reset(jax.random.key(seed))
    rng = np.random.default_rng(seed)
    record, close = square_recorder(path)
    for i in range(int(seconds * 50)):
        s = state.sim.states
        pos, vel, quat, ang_vel = (np.asarray(x[0, 0], float) for x in
                                   (s.pos, s.vel, s.quat, s.ang_vel))
        ref_pos, ref_vel = (np.asarray(x[0]) for x in
                            env.reference_at(state, state.phase + state.steps / 50))
        yaw = math.atan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]),
                         1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
        action = np.clip(pd_action(pos, vel, yaw, ref_pos, ref_vel, float(state.ref_yaw[0]))
                         + rng.normal(0.0, 0.1, 4), -1.0, 1.0)
        record(i / 50, 'firmware', {'pos': pos, 'vel': vel, 'quat': quat, 'gyro': ang_vel},
               ref_pos, ref_vel, action, None, 38000.0)
        state, _, _, done, _ = env.step(state, jnp.asarray(action, jnp.float32)[None])
        assert not bool(done[0]), 'the synthetic flight crashed'
    close()


@pytest.fixture(scope='module')
def flights(tmp_path_factory):
    directory = tmp_path_factory.mktemp('flights')
    env = SquareEnv(TRUE_WORLD)
    paths = [directory / f'flight{seed}.csv' for seed in range(3)]
    for seed, path in enumerate(paths):
        synthesise_flight(env, path, seed)
    return paths


def test_logs_load_as_whole_segments(flights):
    segments = load_flights(flights)
    assert len(segments) == 3
    assert all(len(s.pos) == 600 and s.action.shape == (600, 4) for s in segments)
    np.testing.assert_allclose(np.linalg.norm(segments[0].quat, axis=-1), 1.0, atol=1e-5)


def test_gaps_split_a_flight(flights, tmp_path):
    lines = flights[0].read_text().splitlines()
    # Drop rows 100-109: a 0.2 s gap.
    (tmp_path / 'gap.csv').write_text('\n'.join(lines[:102] + lines[112:]) + '\n')
    assert [len(s.pos) for s in load_flight(tmp_path / 'gap.csv')] == [100, 490]


def test_a_log_of_another_kind_is_refused(tmp_path):
    (tmp_path / 'hover.csv').write_text('time,phase\n0,policy\n')
    with pytest.raises(ValueError, match='square flight log'):
        load_flight(tmp_path / 'hover.csv')


def test_the_fit_finds_the_gain_and_latency_and_halves_the_error(flights):
    result = identify(load_flights(flights), SquareConfig(),
                      SysIdConfig(gain_steps=150, residual_steps=150, batch=64,
                                  learning_rate=1e-2), log=lambda *args: None)
    assert result.latency == TRUE_LATENCY
    assert result.thrust_gain == pytest.approx(TRUE_GAIN, rel=0.02)
    report = result.report
    assert report['full']['score_horizon'] <= 0.5 * report['uncorrected']['score_horizon']
    assert result.improved
