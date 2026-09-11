"""Hover-thrust calibration, shared by the simulated tasks."""
import jax
import jax.numpy as jnp
from crazyflow.sim.functional import attitude_control

GRAVITY = 9.81


def calibrate_hover_thrust(sim_step, default, num_envs, mass, sim_freq):
    """Collective thrust that holds altitude, found by simulation rather than assumed.

    The fitted so_rpy model does not turn a command of m*g into exactly m*g of lift, so zero action
    is calibrated to true hover with a few secant steps on the climb rate.
    """
    level = default.replace(states=default.states.replace(
        pos=default.states.pos.at[..., 2].set(1.0)))
    steps = int(0.2 * sim_freq)

    @jax.jit
    def climb_rate(thrust):
        cmd = jnp.zeros((num_envs, 1, 4)).at[..., 3].set(thrust)
        return sim_step(attitude_control(level, cmd), n_steps=steps).states.vel[0, 0, 2]

    weight = mass * GRAVITY
    lo, hi = 0.9 * weight, 1.1 * weight
    f_lo, f_hi = float(climb_rate(lo)), float(climb_rate(hi))
    for _ in range(4):
        # Lift is linear in the command for so_rpy, so one step usually lands on it exactly; stop
        # before the next step divides by zero.
        if abs(f_hi) < 1e-6 or f_hi == f_lo:
            break
        lo, f_lo, hi = hi, f_hi, hi - f_hi * (hi - lo) / (f_hi - f_lo)
        f_hi = float(climb_rate(hi))
    return hi
