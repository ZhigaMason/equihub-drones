"""Hover-thrust calibration, shared by the simulated tasks."""
import pytest

pytest.importorskip('crazyflow')

import jax.numpy as jnp
from crazyflow.sim.functional import attitude_control

from drones.sim.calibration import calibrate_hover_thrust
from drones.sim.hover_env import HoverConfig, HoverEnv


def test_calibrated_thrust_holds_altitude():
    env = HoverEnv(HoverConfig(num_envs=2))
    step = env.sim.build_step_fn()
    default = env.sim.default_data
    thrust = calibrate_hover_thrust(step, default, 2, env.mass, 500)
    assert thrust == pytest.approx(env.hover_thrust)
    level = default.replace(states=default.states.replace(
        pos=default.states.pos.at[..., 2].set(1.0)))
    cmd = jnp.zeros((2, 1, 4)).at[..., 3].set(thrust)
    after = step(attitude_control(level, cmd), n_steps=500)
    assert abs(float(after.states.vel[0, 0, 2])) < 1e-3
