"""The contract between a hover policy and the drone: observation layout and action scaling.

Defined once and used by both sides: the simulator builds observations from jax arrays, the deploy
script from numpy arrays of live cflib logs. A trained policy therefore sees the same numbers in
both places, and its actions mean the same thing. Pass the array module as `xp`. Nothing here
imports JAX, flax or cflib.
"""

SENSORS = ('optical_flow', 'multiranger', 'imu', 'camera')
BASELINE = ('multiranger', 'optical_flow')

# Vector blocks of one observation frame, in order, with their widths:
#   optical_flow  flow x, flow y (rad/s of apparent ground motion), down range  - the Flow deck
#   multiranger   front, back, left, right, up ranges                            - the Multi-ranger
#   imu           gyro x, y, z, and the gravity direction in the body frame
# Every frame ends with the target height and the previous action. The camera is an image input of
# its own, not part of the frame.
BLOCKS = (('optical_flow', 3), ('multiranger', 5), ('imu', 6))
TASK_SIZE = 1 + 4
ACTION_SIZE = 4

FLOW_SCALE = 2.0    # rad/s
GYRO_SCALE = 5.0    # rad/s
HEIGHT_SCALE = 2.0  # m


def validate(enabled):
    """Check a sensor selection; returns it as a tuple."""
    enabled = tuple(enabled)
    unknown = sorted(set(enabled) - set(SENSORS))
    if unknown:
        raise ValueError(f'unknown sensors {unknown}; choose from {list(SENSORS)}')
    if len(set(enabled)) != len(enabled):
        raise ValueError(f'sensor listed twice: {list(enabled)}')
    return enabled


def frame_size(enabled):
    return sum(width for name, width in BLOCKS if name in enabled) + TASK_SIZE


def encode_frame(xp, enabled, *, flow_rate, zrange, ranges, gyro, gravity, target, prev_action,
                 range_max):
    """One observation frame per drone, shape (n, frame_size(enabled)).

    Args, all batched over n drones:
        flow_rate: (n, 2) apparent ground motion in rad/s, i.e. flow pixels over the sensor gain.
        zrange: (n,) down range in metres, clipped to the sensor's range.
        ranges: (n, 5) front, back, left, right and up ranges in metres, clipped likewise.
        gyro: (n, 3) body rates in rad/s.
        gravity: (n, 3) unit gravity direction in the body frame.
        target: (n,) target height in metres.
        prev_action: (n, 4) the previous normalised action.
        range_max: the rangers' maximum reading, in metres.
    """
    parts = []
    if 'optical_flow' in enabled:
        parts += [flow_rate / FLOW_SCALE, zrange[:, None] / range_max]
    if 'multiranger' in enabled:
        parts.append(ranges / range_max)
    if 'imu' in enabled:
        parts += [gyro / GYRO_SCALE, gravity]
    parts += [target[:, None] / HEIGHT_SCALE, prev_action]
    return xp.concatenate(parts, axis=-1)


def decode_action(xp, action, *, max_tilt, max_yaw_rate, hover_thrust, thrust_min, thrust_max):
    """Normalised actions (..., 4) -> roll, pitch (rad), yaw rate (rad/s), collective thrust (N).

    Angles are in the right-handed body frame (+x forward, +y left, +z up): +roll moves the drone
    right, +pitch moves it forward, +yaw rate turns it left. Thrust is piecewise linear: 0 is hover,
    +/-1 are the motor limits.
    """
    a = xp.clip(action, -1.0, 1.0)
    thrust = xp.where(a[..., 3] >= 0,
                      hover_thrust + a[..., 3] * (thrust_max - hover_thrust),
                      hover_thrust + a[..., 3] * (hover_thrust - thrust_min))
    return a[..., 0] * max_tilt, a[..., 1] * max_tilt, a[..., 2] * max_yaw_rate, thrust
