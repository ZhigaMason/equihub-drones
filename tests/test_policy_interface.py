"""The shared observation and action contract, with numpy alone."""
import numpy as np
import pytest

from drones.policy import interface
from drones.policy.interface import BASELINE, decode_action, encode_frame, frame_size, validate


def frame(enabled, n=2):
    return encode_frame(
        np, enabled, flow_rate=np.full((n, 2), 1.0), zrange=np.full(n, 2.0),
        ranges=np.tile([0.4, 0.8, 1.2, 1.6, 2.0], (n, 1)), gyro=np.full((n, 3), 2.5),
        gravity=np.tile([0.0, 0.0, -1.0], (n, 1)), target=np.full(n, 1.0),
        prev_action=np.tile([0.1, 0.2, 0.3, 0.4], (n, 1)), range_max=4.0)


@pytest.mark.parametrize('enabled, size', [
    (BASELINE, 3 + 5 + 5),
    (('optical_flow',), 3 + 5),
    (('multiranger',), 5 + 5),
    (('multiranger', 'optical_flow', 'imu'), 3 + 5 + 6 + 5),
    (('multiranger', 'optical_flow', 'imu', 'camera'), 3 + 5 + 6 + 5),  # camera is not in the frame
])
def test_frame_size_follows_the_sensor_selection(enabled, size):
    assert frame_size(enabled) == size
    assert frame(enabled).shape == (2, size)


def test_baseline_frame_layout_and_scaling():
    np.testing.assert_allclose(frame(BASELINE)[0], [
        0.5, 0.5, 0.5,                        # flow / 2 rad/s, z-range / 4 m
        0.1, 0.2, 0.3, 0.4, 0.5,              # rangers / 4 m
        0.5, 0.1, 0.2, 0.3, 0.4,              # target / 2 m, previous action
    ])


def test_block_order_is_fixed_whatever_order_sensors_are_listed_in():
    np.testing.assert_array_equal(frame(('optical_flow', 'multiranger', 'imu')),
                                  frame(('imu', 'multiranger', 'optical_flow')))


def test_unknown_or_repeated_sensors_are_rejected():
    with pytest.raises(ValueError, match='unknown sensors'):
        validate(['multiranger', 'lidar'])
    with pytest.raises(ValueError, match='twice'):
        validate(['imu', 'imu'])


def test_thrust_is_hover_at_zero_and_the_motor_limits_at_the_ends():
    kw = dict(max_tilt=0.35, max_yaw_rate=1.5, hover_thrust=0.44, thrust_min=0.085,
              thrust_max=0.8)
    actions = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, -1], [1, -1, 0.5, 5]], float)
    roll, pitch, yaw_rate, thrust = decode_action(np, actions, **kw)
    np.testing.assert_allclose(thrust, [0.44, 0.8, 0.085, 0.8])
    np.testing.assert_allclose([roll[3], pitch[3], yaw_rate[3]], [0.35, -0.35, 0.75])


def test_jax_and_numpy_encode_identically():
    jnp = pytest.importorskip('jax.numpy')
    enabled = ('multiranger', 'optical_flow', 'imu')
    args = dict(flow_rate=np.random.rand(3, 2), zrange=np.random.rand(3),
                ranges=np.random.rand(3, 5), gyro=np.random.rand(3, 3),
                gravity=np.random.rand(3, 3), target=np.random.rand(3),
                prev_action=np.random.rand(3, 4))
    ours = encode_frame(np, enabled, range_max=4.0, **args)
    theirs = encode_frame(jnp, enabled, range_max=4.0,
                          **{k: jnp.asarray(v) for k, v in args.items()})
    np.testing.assert_allclose(ours, np.asarray(theirs), rtol=1e-6)
    assert interface.ACTION_SIZE == 4
