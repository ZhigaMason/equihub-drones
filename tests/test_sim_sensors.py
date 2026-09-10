"""Simulated sensors against hand-computed geometry and physics."""
import pytest

pytest.importorskip('crazyflow')

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from drones.sim import sensors
from drones.sim.geometry import euler_to_quat, quat_to_matrix, yaw_from_quat
from drones.sim.hover_env import HoverConfig, HoverEnv
from drones.sim.sensors import DOWN, RANGER_NAMES, SensorConfig

NOISELESS = SensorConfig(range_noise_abs=0.0, range_noise_rel=0.0, flow_noise=0.0,
                         gyro_noise=0.0, gravity_noise=0.0)
KEY = jax.random.key(0)


@pytest.fixture(scope='module')
def env():
    return HoverEnv(HoverConfig(num_envs=2))


@pytest.fixture(scope='module')
def room(env):
    """Both worlds in a 3 m x 2 m room, 2.5 m high: walls at x = +/-1.5, y = +/-1.0."""
    return env._place_walls(env.sim.mjx_data, jnp.array([[1.5, 1.0, 2.5]] * 2))


def pose(roll=0.0, pitch=0.0, yaw=0.0, n=2):
    return euler_to_quat(jnp.full(n, roll), jnp.full(n, pitch), jnp.full(n, yaw))


def ranges(env, room, pos, quat):
    distances = sensors.ranger_distances(env.mjx_model, room, jnp.array([pos] * 2), quat)
    return dict(zip(RANGER_NAMES, np.asarray(distances[0])))


@pytest.mark.parametrize('angles', [(0.1, -0.2, 0.3), (0.5, 0.4, -2.0), (-0.3, 0.0, 3.0)])
def test_euler_convention_matches_scipy(angles):
    ours = quat_to_matrix(euler_to_quat(*map(jnp.asarray, angles)))
    theirs = Rotation.from_euler('xyz', angles).as_matrix()
    np.testing.assert_allclose(ours, theirs, atol=1e-6)


def test_yaw_round_trips():
    assert float(yaw_from_quat(euler_to_quat(0.2, -0.1, 1.3))) == pytest.approx(1.3, abs=1e-6)


def test_rangers_measure_the_room(env, room):
    got = ranges(env, room, (0.5, -0.2, 1.0), pose())
    expected = dict(front=1.0, back=2.0, left=1.2, right=0.8, up=1.5, down=1.0)
    for name, value in expected.items():
        assert got[name] == pytest.approx(value, abs=1e-4), name


def test_yawed_drone_sees_the_room_rotated(env, room):
    # Nose along +y: front looks at y = +1, left looks at x = -1.5.
    got = ranges(env, room, (0.5, -0.2, 1.0), pose(yaw=np.pi / 2))
    expected = dict(front=1.2, back=0.8, left=2.0, right=1.0)
    for name, value in expected.items():
        assert got[name] == pytest.approx(value, abs=1e-4), name


def test_tilted_down_ranger_reads_the_slant_distance(env, room):
    got = ranges(env, room, (0.0, 0.0, 1.0), pose(roll=0.3))
    assert got['down'] == pytest.approx(1.0 / np.cos(0.3), abs=1e-4)


def test_readings_clip_to_the_sensor_range():
    distances = jnp.array([[0.5, 3.9, 4.5, jnp.inf, 1.0, 2.0]])
    readings = sensors.ranger_readings(distances, KEY, NOISELESS)
    np.testing.assert_allclose(readings[0], [0.5, 3.9, 4.0, 4.0, 1.0, 2.0])


def test_range_noise_grows_with_distance():
    near = sensors.ranger_readings(jnp.full((4000, 6), 0.5), KEY, SensorConfig())
    far = sensors.ranger_readings(jnp.full((4000, 6), 3.0), KEY, SensorConfig())
    assert float(jnp.std(far)) > 2 * float(jnp.std(near))


def flow(vel=(0, 0, 0), ang_vel=(0, 0, 0), quat=None, down=1.0):
    quat = pose(n=1) if quat is None else quat
    out = sensors.optical_flow(jnp.array([vel], float), jnp.array([ang_vel], float), quat,
                               jnp.array([down]), KEY, NOISELESS)
    return np.asarray(out[0]) / NOISELESS.flow_gain  # back to rad/s of apparent motion


def test_flow_from_translation_is_velocity_over_height():
    np.testing.assert_allclose(flow(vel=(0.5, 0, 0), down=1.0), [0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(flow(vel=(0.5, 0, 0), down=2.0), [0.25, 0.0], atol=1e-6)
    np.testing.assert_allclose(flow(vel=(0, 0.3, 0), down=1.0), [0.0, 0.3], atol=1e-6)


def test_flow_from_rotation_follows_the_firmware_model():
    # predictedNX ~ (v/h - omega_y), predictedNY ~ (v/h + omega_x)
    np.testing.assert_allclose(flow(ang_vel=(0, 0.2, 0)), [-0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(flow(ang_vel=(0.2, 0, 0)), [0.0, 0.2], atol=1e-6)


def test_flow_is_in_the_body_frame():
    # Nose along +y and moving along world +x is moving to the drone's right: -y in the body.
    np.testing.assert_allclose(flow(vel=(0.5, 0, 0), quat=pose(yaw=np.pi / 2, n=1)),
                               [0.0, -0.5], atol=1e-6)


def test_flow_height_is_clamped_like_the_firmware():
    np.testing.assert_allclose(flow(vel=(0.1, 0, 0), down=0.01), [1.0, 0.0], atol=1e-6)


def test_imu_gravity_direction():
    gyro, gravity = sensors.imu(pose(), jnp.zeros((2, 3)), KEY, NOISELESS)
    np.testing.assert_allclose(gravity, [[0, 0, -1]] * 2, atol=1e-6)
    _, gravity = sensors.imu(pose(roll=0.3, pitch=-0.2), jnp.zeros((2, 3)), KEY, NOISELESS)
    expected = Rotation.from_euler('xyz', [0.3, -0.2, 0]).as_matrix().T @ [0, 0, -1]
    np.testing.assert_allclose(gravity[0], expected, atol=1e-6)


def test_camera_sees_the_wall_ahead_and_the_floor_below(env, room):
    config = SensorConfig(enabled=('camera',), camera_resolution=(8, 6), camera_pitch=0.5)
    image = sensors.render_camera(env.mjx_model, room, jnp.array([[0.0, 0.0, 1.0]] * 2), pose(),
                                  env._geom_colours, env.floor_geom, config)
    assert image.shape == (2, 6, 8, 3)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0

    # The top-centre ray, from the drone at the origin facing +x, hits the wall at x = 1.5.
    d = np.asarray(sensors.camera_directions(config)[0, 4])
    reach = 1.5 / d[0]
    assert 0 < 1.0 + reach * d[2] < 2.5, 'test geometry: the ray should hit the wall'
    wall = env.sim.mj_model.geom_rgba[env.sim.mj_model.body('wall_px').geomadr[0], :3]
    np.testing.assert_allclose(image[0, 0, 4], wall * (0.35 + 0.65 * np.exp(-reach / 6)), atol=1e-4)

    # The bottom row looks at the floor, which is one of the two checker colours.
    floor = image[0, -1, 4] / (0.35 + 0.65 * np.exp(-1.0 / 6))
    assert np.allclose(floor, sensors._FLOOR_LIGHT, atol=0.1) or \
        np.allclose(floor, sensors._FLOOR_DARK, atol=0.1)
