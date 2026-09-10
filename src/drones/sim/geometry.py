"""Rotation helpers in JAX. Quaternions are scalar-last ``[x, y, z, w]``, as in CrazyFlow."""
import jax.numpy as jnp


def quat_to_matrix(quat):
    """Body-to-world rotation matrices (..., 3, 3) from quaternions (..., 4)."""
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return jnp.stack([
        jnp.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], -1),
        jnp.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], -1),
        jnp.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], -1),
    ], -2)


def euler_to_quat(roll, pitch, yaw):
    """Quaternion for R = Rz(yaw) @ Ry(pitch) @ Rx(roll), i.e. scipy's extrinsic 'xyz'."""
    cr, sr = jnp.cos(roll / 2), jnp.sin(roll / 2)
    cp, sp = jnp.cos(pitch / 2), jnp.sin(pitch / 2)
    cy, sy = jnp.cos(yaw / 2), jnp.sin(yaw / 2)
    return jnp.stack([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], -1)


def yaw_from_quat(quat):
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_angle(angle):
    """Wrap to [-pi, pi)."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
