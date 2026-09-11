"""A learned correction to CrazyFlow's dynamics, fitted to real flights by drones.rl.sysid.

A small network maps the drone's state and action to a force and a torque. They act through
CrazyFlow's disturbance inputs, `states.force` and `states.torque` (world frame), which so_rpy adds to
its fitted dynamics, so the corrected simulator stays differentiable. Features and outputs are in the
body frame, so the correction does not depend on where the drone is or which way it faces.
"""
import flax.linen as nn
import jax.numpy as jnp

from drones.sim.geometry import quat_to_matrix

FEATURE_SIZE = 13      # body velocity (3), gravity direction (3), body rates (3), action (4)
FORCE_SCALE = 0.1      # N per unit output: about a quarter of the drone's weight
TORQUE_SCALE = 1e-3    # N m per unit output


class Residual(nn.Module):
    hidden: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, features):
        x = features
        for size in self.hidden:
            x = nn.tanh(nn.Dense(size)(x))
        # Zero output weights and bias: an unfitted residual changes nothing.
        return nn.Dense(6, kernel_init=nn.initializers.zeros, name='out')(x)


def init_residual(key):
    return Residual().init(key, jnp.zeros((1, FEATURE_SIZE)))


def residual_features(vel, quat, ang_vel, action):
    """(n, FEATURE_SIZE) from world velocity, attitude, body rates and the applied action."""
    rot = quat_to_matrix(quat)
    body_vel = jnp.einsum('nji,nj->ni', rot, vel)
    gravity = -rot[:, 2, :]
    return jnp.concatenate([body_vel, gravity, ang_vel, action], -1)


def residual_wrench(model, params, vel, quat, ang_vel, action):
    """World-frame force (n, 3) and torque (n, 3) for CrazyFlow's disturbance inputs."""
    out = model.apply(params, residual_features(vel, quat, ang_vel, action))
    rot = quat_to_matrix(quat)
    force = jnp.einsum('nij,nj->ni', rot, out[:, :3] * FORCE_SCALE)
    torque = jnp.einsum('nij,nj->ni', rot, out[:, 3:] * TORQUE_SCALE)
    return force, torque
