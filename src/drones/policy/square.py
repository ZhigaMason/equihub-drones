"""The contract between a square-flying policy and the drone: reference path and observation.

Used by both sides, like drones.policy.interface: the simulator calls it with jax arrays, the deploy
script with numpy arrays of the firmware's state estimate. Both therefore compute the same reference
and feed the policy the same numbers. Pass the array module as `xp`. Nothing here imports JAX, flax
or cflib.

The reference is a square of side `side` with corners rounded to `corner_radius`, flown at constant
speed, one lap every `lap_time` seconds. At t = 0 it is at `origin`, the start of the first edge.
The first edge runs along the heading `rotation`. With `direction` +1 the path turns left
(counter-clockwise from above); -1 mirrors it. `origin`'s z is the flight height. Rounded corners
keep position and velocity continuous, so the reference and the reward built on it stay smooth
enough to differentiate.
"""
import math

LOOKAHEAD = 5          # future reference points in the observation
LOOKAHEAD_DT = 0.2     # s between them: the policy sees a second ahead
POS_SCALE = 0.5        # m
VEL_SCALE = 1.0        # m/s
ACTION_SIZE = 4
# Position errors now and ahead (3 each), velocity error (3), velocity (3), gravity direction in the
# body frame (3), heading error as sin and cos (2), previous action (4).
OBS_SIZE = 3 * (1 + LOOKAHEAD) + 3 + 3 + 3 + 2 + ACTION_SIZE


def path_length(side, corner_radius):
    """Length of one lap: four straights and four quarter circles."""
    return 4 * (side - 2 * corner_radius) + 2 * math.pi * corner_radius


def square_reference(xp, t, *, side, lap_time, corner_radius, direction, rotation, origin):
    """Reference position and velocity, each (..., 3), at times t (...) in seconds.

    Every parameter broadcasts against t, and `origin` against t with a trailing axis of 3, so a
    batch of drones can each fly their own square.
    """
    r = corner_radius
    straight = side - 2 * r
    edge = straight + 0.5 * math.pi * r            # one straight and the corner after it
    perimeter = 4 * edge
    speed = perimeter / lap_time
    u = xp.mod(t / lap_time, 1.0) * perimeter
    k = xp.clip(xp.floor(u / edge), 0, 3)          # which edge
    w = u - k * edge                               # distance along it
    # Edge 0 starts at (0, 0) heading +x, then turns left round the centre (straight, r).
    on_arc = w > straight
    theta = xp.clip((w - straight) / r, 0.0, 0.5 * math.pi)
    px = xp.where(on_arc, straight + r * xp.sin(theta), w)
    py = xp.where(on_arc, r - r * xp.cos(theta), 0.0)
    hx = xp.where(on_arc, xp.cos(theta), 1.0)
    hy = xp.where(on_arc, xp.sin(theta), 0.0)
    # Edge k is edge 0 turned by k quarter turns, starting where edge k - 1 ended. Edge 0 ends at
    # d = (straight + r, r), so edge k starts at the sum of d turned by 0 .. k - 1 quarter turns.
    dx, dy = straight + r, r
    sx = xp.where(k >= 1, dx, 0.0) + xp.where(k >= 2, -dy, 0.0) + xp.where(k >= 3, -dx, 0.0)
    sy = xp.where(k >= 1, dy, 0.0) + xp.where(k >= 2, dx, 0.0) + xp.where(k >= 3, -dy, 0.0)
    c, s = xp.cos(k * 0.5 * math.pi), xp.sin(k * 0.5 * math.pi)
    x, y = sx + c * px - s * py, sy + s * px + c * py
    vx, vy = speed * (c * hx - s * hy), speed * (s * hx + c * hy)
    y, vy = direction * y, direction * vy
    cr, sr = xp.cos(rotation), xp.sin(rotation)
    zero = xp.zeros_like(x)
    pos = xp.stack([cr * x - sr * y, sr * x + cr * y, zero], -1) + origin
    vel = xp.stack([cr * vx - sr * vy, sr * vx + cr * vy, zero], -1)
    return pos, vel


def heading(xp, quat):
    """Yaw of quaternions (..., 4), scalar-last."""
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return xp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_yaw_frame(xp, vec, yaw):
    """World-frame vectors (n, ..., 3) in the frame of heading yaw (n,): +x along the heading."""
    yaw = yaw.reshape(yaw.shape + (1,) * (vec.ndim - 2))
    c, s = xp.cos(yaw), xp.sin(yaw)
    x, y = vec[..., 0], vec[..., 1]
    return xp.stack([c * x + s * y, -s * x + c * y, vec[..., 2]], -1)


def encode_square_obs(xp, *, pos_est, vel_est, yaw_est, gravity, ref_pos, ref_vel, lookahead_pos,
                      ref_yaw, prev_action):
    """One observation per drone, shape (n, OBS_SIZE).

    Args, all batched over n drones:
        pos_est, vel_est: (n, 3) estimated position and velocity, world frame.
        yaw_est: (n,) estimated heading.
        gravity: (n, 3) unit gravity direction in the body frame.
        ref_pos, ref_vel: (n, 3) the reference now.
        lookahead_pos: (n, LOOKAHEAD, 3) the reference LOOKAHEAD_DT, 2 * LOOKAHEAD_DT, ... ahead.
        ref_yaw: (n,) the heading to hold.
        prev_action: (n, 4) the previous normalised action.

    Vectors are expressed in the heading frame, so the policy does not depend on which way the
    drone or the square faces.
    """
    n = pos_est.shape[0]
    targets = xp.concatenate([ref_pos[:, None], lookahead_pos], 1)
    errors = to_yaw_frame(xp, targets - pos_est[:, None], yaw_est).reshape(n, -1) / POS_SCALE
    vel_error = to_yaw_frame(xp, ref_vel - vel_est, yaw_est) / VEL_SCALE
    velocity = to_yaw_frame(xp, vel_est, yaw_est) / VEL_SCALE
    yaw_error = ref_yaw - yaw_est
    return xp.concatenate([errors, vel_error, velocity, gravity, xp.sin(yaw_error)[:, None],
                           xp.cos(yaw_error)[:, None], prev_action], -1)
