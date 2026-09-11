"""The square reference and the square policy's observation, with numpy alone."""
import math

import numpy as np
import pytest

from drones.policy.square import (LOOKAHEAD, OBS_SIZE, encode_square_obs, heading, path_length,
                                  square_reference)

PARAMS = dict(side=1.0, lap_time=8.0, corner_radius=0.15, direction=1.0, rotation=0.0,
              origin=np.zeros(3))


def ref(t, **changes):
    return square_reference(np, np.asarray(t, float), **{**PARAMS, **changes})


def test_path_is_closed_and_starts_at_the_origin():
    p0, _ = ref(0.0)
    p1, _ = ref(8.0)
    np.testing.assert_allclose(p0, [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(p1, p0, atol=1e-9)


def test_perimeter_and_constant_speed():
    assert path_length(1.0, 0.15) == pytest.approx(4 * 0.7 + 2 * math.pi * 0.15)
    pos, vel = ref(np.linspace(0.0, 8.0, 4001))
    np.testing.assert_allclose(np.linalg.norm(vel, axis=-1), path_length(1.0, 0.15) / 8.0,
                               rtol=1e-9)
    travelled = np.linalg.norm(np.diff(pos, axis=0), axis=-1).sum()
    assert travelled == pytest.approx(path_length(1.0, 0.15), rel=1e-4)


def test_fits_a_one_metre_square():
    pos, _ = ref(np.linspace(0.0, 8.0, 4001))
    assert np.ptp(pos[:, 0]) == pytest.approx(1.0, abs=1e-6)
    assert np.ptp(pos[:, 1]) == pytest.approx(1.0, abs=1e-6)


def test_position_and_velocity_are_continuous():
    t = np.linspace(0.0, 8.0, 80001)
    pos, vel = ref(t)
    dt, speed = t[1] - t[0], path_length(1.0, 0.15) / 8.0
    assert np.abs(np.diff(pos, axis=0)).max() <= speed * dt * 1.0001
    # The heading turns at most speed / radius: no jumps at the corners or at the lap end.
    assert np.abs(np.diff(vel, axis=0)).max() <= speed ** 2 / 0.15 * dt * 1.01


def test_velocity_is_the_derivative_of_position():
    t, h = np.linspace(0.01, 7.99, 500), 1e-5
    (p_plus, _), (p_minus, _), (_, vel) = ref(t + h), ref(t - h), ref(t)
    np.testing.assert_allclose((p_plus - p_minus) / (2 * h), vel, atol=1e-4)


def test_ccw_turns_left_and_cw_mirrors_it():
    t = np.linspace(0.0, 8.0, 401)
    ccw, _ = ref(t)
    cw, _ = ref(t, direction=-1.0)
    np.testing.assert_allclose(cw[:, 0], ccw[:, 0], atol=1e-12)
    np.testing.assert_allclose(cw[:, 1], -ccw[:, 1], atol=1e-12)
    assert ccw[:, 1].max() == pytest.approx(1.0) and ccw[:, 1].min() == pytest.approx(0.0)


def test_rotation_and_origin_move_the_square():
    t = np.linspace(0.0, 8.0, 101)
    base, _ = ref(t)
    moved, moved_vel = ref(t, rotation=math.pi / 2, origin=np.array([1.0, 2.0, 1.2]))
    np.testing.assert_allclose(moved[:, 0], 1.0 - base[:, 1], atol=1e-12)
    np.testing.assert_allclose(moved[:, 1], 2.0 + base[:, 0], atol=1e-12)
    np.testing.assert_allclose(moved[:, 2], 1.2)
    np.testing.assert_allclose(moved_vel[:, 2], 0.0)


def test_per_drone_parameters_broadcast():
    t = np.array([[0.0, 1.0], [2.0, 3.0]])     # 2 drones, 2 times each
    pos, vel = square_reference(np, t, side=1.0, corner_radius=0.15,
                                lap_time=np.array([[8.0], [6.0]]),
                                direction=np.array([[1.0], [-1.0]]),
                                rotation=np.zeros((2, 1)), origin=np.zeros((2, 1, 3)))
    assert pos.shape == vel.shape == (2, 2, 3)
    single, _ = ref(3.0, lap_time=6.0, direction=-1.0)
    np.testing.assert_allclose(pos[1, 1], single)


def test_heading_of_a_quarter_turn():
    half = math.sqrt(0.5)
    assert heading(np, np.array([0.0, 0.0, half, half])) == pytest.approx(math.pi / 2)


def test_observation_is_heading_invariant():
    rng = np.random.default_rng(0)
    pos, vel, ref_pos, ref_vel = (rng.normal(size=(1, 3)) for _ in range(4))
    ahead = rng.normal(size=(1, LOOKAHEAD, 3))

    def obs(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return encode_square_obs(np, pos_est=pos @ rot.T, vel_est=vel @ rot.T,
                                 yaw_est=np.array([yaw]), gravity=np.array([[0.0, 0.0, -1.0]]),
                                 ref_pos=ref_pos @ rot.T, ref_vel=ref_vel @ rot.T,
                                 lookahead_pos=ahead @ rot.T, ref_yaw=np.array([yaw + 0.3]),
                                 prev_action=np.zeros((1, 4)))

    assert obs(0.0).shape == (1, OBS_SIZE)
    np.testing.assert_allclose(obs(1.1), obs(0.0), atol=1e-12)
