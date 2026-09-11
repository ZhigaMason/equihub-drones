# Square Flight with SHAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the Crazyflie 2.1 Brushless to fly a 1 x 1 m square: in simulation with SHAC, on the real drone, and again after fitting the simulator to real flight logs.

**Architecture:**
- **Shared contract:** a numpy-safe module (`policy/square.py`) defines the reference and the observation, used by both the simulator and the drone.
- **Simulator:** `sim/square_env.py` is a differentiable CrazyFlow task. SHAC (`rl/shac.py`) backpropagates short rollouts through it.
- **Drone:** `missions/fly_square.py` flies the square and logs each flight.
- **System ID:** `rl/sysid.py` fits a thrust gain, a latency and a residual wrench (`sim/residual.py`) to those logs. `rl/finetune_square.py` continues SHAC in the corrected simulator.

**Tech Stack:** Python 3.14, JAX 0.11, flax, optax, CrazyFlow 0.3.2 (`so_rpy` dynamics), numpy, cflib, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-10-square-shac-design.md`

## Global Constraints

- Everything importing JAX, flax, optax or CrazyFlow lives under `src/drones/sim/` or `src/drones/rl/` (the `sim` extra). `src/drones/policy/` and `src/drones/missions/` import numpy and cflib only.
- Any module that imports CrazyFlow does `import crazyflow  # noqa: F401` before anything that imports scipy, as `sim/hover_env.py` does.
- Style follows the existing code: single quotes, lines up to 100 columns, and module docstrings that explain the why.
- Quaternions are scalar-last `[x, y, z, w]`. World frame: +x forward at take-off, +y left, +z up. Action signs: +roll moves right (−y), +pitch moves forward (+x), +yaw rate turns left.
- Tests are hermetic. Any simulator test module starts with `pytest.importorskip('crazyflow')`. Run tests with `uv run --extra sim pytest …`.
- Actions are the hover task's: four values in [-1, 1], decoded by `drones.policy.interface.decode_action`.
- The square: side 1.0 m, corner radius 0.15 m, lap time 6–10 s, height 0.8–1.2 m, 50 Hz control, 500 Hz physics.
- Commit after each task on branch `square-shac`. Never `git add -A`: add the files the task names.

---

### Task 0: Branch

- [ ] **Step 1: Check the tree is clean and branch**

Run: `git status --short`
Expected: only `?? docs/superpowers/` and `?? flight.gif`. If anything else shows, stop and ask the user.

```bash
git switch -c square-shac
git add docs/superpowers/specs/2026-09-10-square-shac-design.md docs/superpowers/plans/2026-09-11-square-shac.md
git commit -m "docs: square flight with SHAC design and plan"
```

---

### Task 1: The square reference and observation contract

**Files:**
- Create: `src/drones/policy/square.py`
- Test: `tests/test_square_reference.py`

**Interfaces:**
- Produces:
  - `square_reference(xp, t, *, side, lap_time, corner_radius, direction, rotation, origin) -> (pos (..., 3), vel (..., 3))`
  - `path_length(side, corner_radius) -> float`
  - `heading(xp, quat) -> yaw`
  - `to_yaw_frame(xp, vec, yaw)`
  - `encode_square_obs(xp, *, pos_est, vel_est, yaw_est, gravity, ref_pos, ref_vel, lookahead_pos, ref_yaw, prev_action) -> (n, OBS_SIZE)`
  - constants `LOOKAHEAD = 5`, `LOOKAHEAD_DT = 0.2`, `POS_SCALE = 0.5`, `VEL_SCALE = 1.0`, `OBS_SIZE = 33`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_square_reference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'drones.policy.square'`

- [ ] **Step 3: Implement `src/drones/policy/square.py`**

```python
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
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_square_reference.py tests/test_architecture.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/drones/policy/square.py tests/test_square_reference.py
git commit -m "feat: square reference path and observation contract"
```

---

### Task 2: Shared hover-thrust calibration

**Files:**
- Create: `src/drones/sim/calibration.py`
- Modify: `src/drones/sim/hover_env.py` (line 37 `GRAVITY`, line 123, and delete `_calibrate_hover_thrust` at lines 233–258)
- Test: `tests/test_calibration.py`

**Interfaces:**
- Produces: `calibrate_hover_thrust(sim_step, default, num_envs, mass, sim_freq) -> float` (N), and `GRAVITY = 9.81`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run --extra sim pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'drones.sim.calibration'`

- [ ] **Step 3: Create `src/drones/sim/calibration.py`**

```python
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
```

- [ ] **Step 4: Make `HoverEnv` use it**

In `src/drones/sim/hover_env.py`:
- Add `from drones.sim.calibration import GRAVITY, calibrate_hover_thrust` to the imports and delete the local `GRAVITY = 9.81` line.
- Replace `self.hover_thrust = self._calibrate_hover_thrust(self.sim.default_data)` with:

```python
        self.hover_thrust = calibrate_hover_thrust(self._sim_step, self.sim.default_data,
                                                   config.num_envs, self.mass, config.sim_freq)
```

- Delete the whole `_calibrate_hover_thrust` method.

- [ ] **Step 5: Run the new and existing hover tests**

Run: `uv run --extra sim pytest tests/test_calibration.py tests/test_hover_env.py tests/test_ppo.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/drones/sim/calibration.py src/drones/sim/hover_env.py tests/test_calibration.py
git commit -m "refactor: share hover-thrust calibration between simulated tasks"
```

---

### Task 3: Residual wrench network

**Files:**
- Create: `src/drones/sim/residual.py`
- Test: `tests/test_residual.py`

**Interfaces:**
- Produces:
  - `Residual(hidden=(64, 64))`, a flax module; its output layer is named `out`
  - `FEATURE_SIZE = 13`, `FORCE_SCALE = 0.1` (N), `TORQUE_SCALE = 1e-3` (N m)
  - `residual_features(vel, quat, ang_vel, action) -> (n, 13)`
  - `residual_wrench(model, params, vel, quat, ang_vel, action) -> (force_world (n, 3), torque_world (n, 3))`
  - `init_residual(key) -> params`

- [ ] **Step 1: Write the failing tests**

```python
"""The residual wrench: zero until fitted, body-frame in, world-frame out."""
import math

import pytest

pytest.importorskip('crazyflow')

import jax
import jax.numpy as jnp
import numpy as np

from drones.sim.residual import (FORCE_SCALE, Residual, init_residual, residual_features,
                                 residual_wrench)

HALF = math.sqrt(0.5)
YAWED_LEFT = jnp.array([[0.0, 0.0, HALF, HALF]])   # facing +y


def with_output_bias(params, bias):
    def set_bias(path, x):
        name = jax.tree_util.keystr(path)
        return jnp.asarray(bias, x.dtype) if "'out'" in name and "'bias'" in name else x
    return jax.tree_util.tree_map_with_path(set_bias, params)


def test_an_unfitted_residual_changes_nothing():
    params = init_residual(jax.random.key(0))
    force, torque = residual_wrench(Residual(), params, jnp.ones((3, 3)),
                                    jnp.tile(YAWED_LEFT, (3, 1)), jnp.ones((3, 3)),
                                    jnp.ones((3, 4)))
    np.testing.assert_array_equal(force, 0.0)
    np.testing.assert_array_equal(torque, 0.0)


def test_body_frame_force_is_rotated_into_the_world():
    params = with_output_bias(init_residual(jax.random.key(0)), [1.0, 0, 0, 0, 0, 0])
    force, _ = residual_wrench(Residual(), params, jnp.zeros((1, 3)), YAWED_LEFT,
                               jnp.zeros((1, 3)), jnp.zeros((1, 4)))
    np.testing.assert_allclose(force[0], [0.0, FORCE_SCALE, 0.0], atol=1e-6)


def test_features_are_in_the_body_frame():
    features = residual_features(jnp.array([[0.0, 1.0, 0.0]]), YAWED_LEFT, jnp.zeros((1, 3)),
                                 jnp.zeros((1, 4)))
    np.testing.assert_allclose(features[0, :3], [1.0, 0.0, 0.0], atol=1e-6)   # moving forward
    np.testing.assert_allclose(features[0, 3:6], [0.0, 0.0, -1.0], atol=1e-6)  # level
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --extra sim pytest tests/test_residual.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'drones.sim.residual'`

- [ ] **Step 3: Implement `src/drones/sim/residual.py`**

```python
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
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run --extra sim pytest tests/test_residual.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/drones/sim/residual.py tests/test_residual.py
git commit -m "feat: residual wrench network for correcting CrazyFlow from flight data"
```

---

### Task 4: The square environment

**Files:**
- Create: `src/drones/sim/square_env.py`
- Test: `tests/test_square_env.py`

**Interfaces:**
- Consumes:
  - `square_reference`, `encode_square_obs`, `LOOKAHEAD`, `LOOKAHEAD_DT`, `OBS_SIZE`, `POS_SCALE` and `VEL_SCALE` (Task 1)
  - `calibrate_hover_thrust` (Task 2)
  - `Residual` and `residual_wrench` (Task 3)
- Produces:
  - `SquareConfig`, a frozen dataclass with the fields below and an `episode_steps` property
  - `SquareState`, a flax struct
  - constants `CRASH_REWARD = -10.0` and `PRIVILEGED_SIZE = 13`
  - `SquareEnv(config, residual=None)` with:
    - attributes `num_envs`, `action_size = 4`, `policy_size = OBS_SIZE`, `critic_size = OBS_SIZE + 13`, `image_shape = None`, `hover_thrust`, `thrust_min`, `thrust_max`, `mass`, `residual_model`, `residual`
    - `reset(key) -> (state, obs)`
    - `step(state, action) -> (state, obs, reward, done, info)`, where `info` has `crashed`, `truncated`, `episode_return`, `episode_length`, `pos_error`, `speed`, `tilt` and `final_critic`
    - `advance(sim, action, yaw_cmd, thrust_gain, residual=None) -> (sim, yaw_cmd)`
    - `with_states(sim, pos, vel, quat, ang_vel) -> sim`
    - `reference_at(state, t) -> (pos, vel)`
    - `policy_spec() -> dict`
  - `obs` is `{'policy': (n, 33), 'critic': (n, 46)}`

- [ ] **Step 1: Write the failing tests**

```python
"""The square task: episodes, observations, differentiability, randomisation, residual hook."""
import dataclasses

import pytest

pytest.importorskip('crazyflow')

import jax
import jax.numpy as jnp
import numpy as np

from drones.policy.square import OBS_SIZE
from drones.sim.residual import FORCE_SCALE, init_residual
from drones.sim.square_env import PRIVILEGED_SIZE, SquareConfig, SquareEnv

N = 8
QUIET = SquareConfig(num_envs=N, thrust_gain_range=0.0, latency_steps=(0,), pos_noise=0.0,
                     pos_drift=0.0, vel_noise=0.0, attitude_noise=0.0)


@pytest.fixture(scope='module')
def env():
    return SquareEnv(SquareConfig(num_envs=N))


@pytest.fixture(scope='module')
def quiet():
    return SquareEnv(QUIET)


def hold_still(state, z=1.0):
    """Level, motionless at (0, 0, z), facing +x, with an empty action buffer."""
    s = state.sim.states
    states = s.replace(pos=jnp.zeros_like(s.pos).at[..., 2].set(z), vel=jnp.zeros_like(s.vel),
                       ang_vel=jnp.zeros_like(s.ang_vel),
                       quat=jnp.zeros_like(s.quat).at[..., 3].set(1.0))
    return state.replace(sim=state.sim.replace(states=states), yaw_cmd=jnp.zeros(N),
                         ref_yaw=jnp.zeros(N), actions=jnp.zeros_like(state.actions))


def fly(env, state, action, steps):
    for _ in range(steps):
        state, obs, reward, done, info = env.step(state, jnp.broadcast_to(action, (N, 4)))
    return state, obs, reward, done, info


def test_observations_have_the_right_size(env):
    _, obs = env.reset(jax.random.key(0))
    assert obs['policy'].shape == (N, OBS_SIZE)
    assert obs['critic'].shape == (N, OBS_SIZE + PRIVILEGED_SIZE)
    assert all(bool(jnp.isfinite(v).all()) for v in obs.values())


def test_episodes_start_on_the_reference_at_the_origin(quiet):
    state, _ = quiet.reset(jax.random.key(1))
    ref, _ = quiet.reference_at(state, state.phase)
    np.testing.assert_allclose(ref[:, :2], 0.0, atol=1e-5)
    assert float(jnp.abs(state.sim.states.pos[:, 0] - ref).max()) <= 0.1 + 1e-6


def test_randomisation_stays_in_its_ranges(env):
    state, _ = env.reset(jax.random.key(2))
    assert bool(((state.thrust_gain >= 0.9) & (state.thrust_gain <= 1.1)).all())
    assert set(np.asarray(state.latency).tolist()) <= {0, 1, 2}
    assert bool(((state.lap_time >= 6.0) & (state.lap_time <= 10.0)).all())
    assert set(np.asarray(state.direction).tolist()) <= {-1.0, 1.0}
    assert bool(((state.origin[:, 2] >= 0.8) & (state.origin[:, 2] <= 1.2)).all())


def test_zero_action_from_the_start_stays_near_the_reference(quiet):
    state, _ = quiet.reset(jax.random.key(3))
    for _ in range(25):   # 0.5 s
        state, _, _, done, info = quiet.step(state, jnp.zeros((N, 4)))
        assert not bool(done.any())
        assert float(info['pos_error'].max()) < 0.5


def test_rollout_gradient_matches_finite_differences(quiet):
    state, _ = quiet.reset(jax.random.key(4))

    def total_reward(action):
        def body(s, _):
            s, _, reward, _, _ = quiet.step(s, jnp.broadcast_to(action, (N, 4)))
            return s, reward
        _, rewards = jax.lax.scan(body, state, None, length=16)
        return rewards.sum()

    a0 = jnp.array([0.05, -0.05, 0.02, 0.03])
    grad = jax.grad(total_reward)(a0)
    assert bool(jnp.isfinite(grad).all())
    eps = 1e-3
    numeric = jnp.array([(total_reward(a0.at[i].add(eps)) - total_reward(a0.at[i].add(-eps)))
                         / (2 * eps) for i in range(4)])
    np.testing.assert_allclose(grad, numeric, rtol=0.05, atol=0.05)


def test_a_restarted_world_carries_no_gradient_from_its_last_episode(quiet):
    state, _ = quiet.reset(jax.random.key(5))

    def world0_height_after_step(z0):
        pos = state.sim.states.pos.at[0, 0, 2].set(z0)
        s = state.replace(sim=state.sim.replace(states=state.sim.states.replace(pos=pos)))
        s, _, _, done, _ = quiet.step(s, jnp.zeros((N, 4)))
        return s.sim.states.pos[0, 0, 2], done[0]

    assert bool(world0_height_after_step(0.05)[1]), 'below min_height: crashed and restarted'
    assert float(jax.grad(lambda z: world0_height_after_step(z)[0])(0.05)) == 0.0
    assert float(jax.grad(lambda z: world0_height_after_step(z)[0])(1.0)) == pytest.approx(1.0,
                                                                                          abs=0.01)


def test_thrust_gain_scales_the_lift(quiet):
    state = hold_still(quiet.reset(jax.random.key(6))[0])
    level, *_ = fly(quiet, state, jnp.zeros(4), 10)
    strong, *_ = fly(quiet, state.replace(thrust_gain=jnp.full(N, 1.2)), jnp.zeros(4), 10)
    assert float(jnp.abs(level.sim.states.vel[:, 0, 2]).max()) < 0.02
    assert float(strong.sim.states.vel[:, 0, 2].min()) > 0.3


def test_latency_delays_the_action():
    env = SquareEnv(dataclasses.replace(QUIET, latency_steps=(2,)))
    state = hold_still(env.reset(jax.random.key(7))[0])
    roll = jnp.array([0.5, 0.0, 0.0, 0.0])
    state, *_ = fly(env, state, roll, 2)
    assert float(jnp.abs(state.sim.states.ang_vel).max()) < 1e-6, 'still flying the old zeros'
    state, *_ = fly(env, state, roll, 1)
    assert float(jnp.abs(state.sim.states.ang_vel).max()) > 0.1


def test_the_residual_force_pushes_the_drone():
    def set_bias(path, x):
        name = jax.tree_util.keystr(path)
        return jnp.array([1.0, 0, 0, 0, 0, 0]) if "'out'" in name and "'bias'" in name else x
    params = jax.tree_util.tree_map_with_path(set_bias, init_residual(jax.random.key(0)))
    env = SquareEnv(QUIET, residual=params)
    state = hold_still(env.reset(jax.random.key(8))[0])
    state, *_ = fly(env, state, jnp.zeros(4), 10)   # 0.2 s
    expected = FORCE_SCALE / env.mass * 0.2
    np.testing.assert_allclose(state.sim.states.vel[:, 0, 0], expected, rtol=0.1)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --extra sim pytest tests/test_square_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'drones.sim.square_env'`

- [ ] **Step 3: Implement `src/drones/sim/square_env.py`**

```python
"""Flying a 1 x 1 m square on CrazyFlow, as a batch of differentiable JAX functions.

The policy commands attitude and collective thrust, as in the hover task. It observes the firmware's
state estimate: position, velocity and attitude as the Flow deck's Kalman filter delivers them,
modelled as the truth plus noise and a slow horizontal drift. It sees its error to a time-parametrised
reference (drones.policy.square) now and a second ahead, so it can anticipate the corners.

Unlike HoverEnv, `step` is differentiable in the state and the action, so SHAC (drones.rl.shac) can
backpropagate through the physics. No rays are cast, every reward term is smooth, and metrics are
gradient-stopped. Worlds that finish restart inside `step`, and their new state carries no gradient
from the old.

Domain randomisation covers two sim-to-real gaps the hover task left open: thrust-to-weight and
action latency. so_rpy's lift is cmd_f_coef * thrust / mass with no offset, so a thrust gain covers
mass too. A residual wrench fitted to real flights (drones.rl.sysid) enters through CrazyFlow's
disturbance force and torque.
"""
from dataclasses import dataclass

import crazyflow  # noqa: F401  Must precede scipy, see drones.sim.
import jax
import jax.numpy as jnp
from crazyflow import Control, Sim
from crazyflow.drones import load_params
from crazyflow.sim.data import SimData
from crazyflow.sim.functional import attitude_control
from crazyflow.utils import leaf_replace
from flax import struct

from drones.policy.interface import GYRO_SCALE, decode_action
from drones.policy.square import (LOOKAHEAD, LOOKAHEAD_DT, OBS_SIZE, POS_SCALE, VEL_SCALE,
                                  encode_square_obs, square_reference)
from drones.sim.calibration import calibrate_hover_thrust
from drones.sim.geometry import euler_to_quat, quat_to_matrix, wrap_angle, yaw_from_quat
from drones.sim.hover_env import YAW_BAND
from drones.sim.residual import Residual, residual_wrench

DRONE = 'cf21B_500'
CRASH_REWARD = -10.0
# Critic-only extras, on top of the policy observation: true position error (3), true velocity (3),
# angular velocity (3), reference phase as sin and cos (2), lap time (1), thrust gain (1).
PRIVILEGED_SIZE = 13


@dataclass(frozen=True)
class SquareConfig:
    num_envs: int = 64
    device: str = 'cpu'
    dynamics: str = 'so_rpy'
    sim_freq: int = 500
    control_freq: int = 50
    episode_seconds: float = 16.0
    # The square: side and corner rounding in metres; lap time and height are sampled per episode.
    side: float = 1.0
    corner_radius: float = 0.15
    lap_time: tuple[float, float] = (6.0, 10.0)
    height: tuple[float, float] = (0.8, 1.2)
    # Start error around the reference.
    start_pos_error: float = 0.1
    start_vel_error: float = 0.2
    start_tilt: float = 0.1
    # Domain randomisation: thrust gain within +/- thrust_gain_range of thrust_gain (it stands in for
    # mass too), and a latency of one of latency_steps control steps, per episode.
    thrust_gain: float = 1.0
    thrust_gain_range: float = 0.1
    latency_steps: tuple[int, ...] = (0, 1, 2)
    # State-estimate errors.
    pos_noise: float = 0.01          # m
    pos_drift: float = 0.01          # m/sqrt(s), horizontal random walk; height comes from the ranger
    vel_noise: float = 0.03          # m/s
    attitude_noise: float = 0.01     # rad
    # Action scaling, as the hover task.
    max_tilt: float = 0.35
    max_yaw_rate: float = 1.5
    # Termination.
    min_height: float = 0.1
    max_tilt_terminate: float = 1.0
    max_error: float = 1.0

    def __post_init__(self):
        if not 0.0 < self.corner_radius < self.side / 2:
            raise ValueError(f'corner_radius must be between 0 and side / 2, got '
                             f'{self.corner_radius}')
        if not self.latency_steps or min(self.latency_steps) < 0:
            raise ValueError(f'latency_steps must be whole steps >= 0, got {self.latency_steps}')

    @property
    def episode_steps(self):
        return int(round(self.episode_seconds * self.control_freq))


@struct.dataclass
class SquareState:
    sim: SimData
    default_sim: SimData
    phase: jax.Array           # (n,) reference time at the episode start, s
    lap_time: jax.Array        # (n,)
    direction: jax.Array       # (n,) +1 counter-clockwise, -1 clockwise
    rotation: jax.Array        # (n,)
    origin: jax.Array          # (n, 3)
    ref_yaw: jax.Array         # (n,) the heading to hold
    yaw_cmd: jax.Array         # (n,) absolute yaw setpoint integrated from yaw-rate commands
    thrust_gain: jax.Array     # (n,)
    latency: jax.Array         # (n,) control steps
    actions: jax.Array         # (n, max latency + 1, 4) policy actions, newest first
    bias: jax.Array            # (n, 3) state-estimate position drift
    steps: jax.Array           # (n,)
    episode_return: jax.Array  # (n,)
    key: jax.Array


class SquareEnv:
    """Vectorised square task. `reset(key)` and `step(state, action)` are jitted; `step` is
    differentiable."""

    action_size = 4
    policy_size = OBS_SIZE
    critic_size = OBS_SIZE + PRIVILEGED_SIZE
    image_shape = None

    def __init__(self, config: SquareConfig = SquareConfig(), residual=None):
        if config.sim_freq % config.control_freq:
            raise ValueError('sim_freq must be a multiple of control_freq')
        self.config = config
        self.num_envs = config.num_envs
        self.substeps = config.sim_freq // config.control_freq
        self.sim = Sim(n_worlds=config.num_envs, n_drones=1, drone=DRONE,
                       dynamics=config.dynamics, control=Control.attitude,
                       freq=config.sim_freq, device=config.device)
        self._sim_step = self.sim.build_step_fn()
        self._sim_reset = self.sim.build_reset_fn()

        params = load_params(DRONE)
        self.thrust_min = 4 * params['thrust_min']
        self.thrust_max = 4 * params['thrust_max']
        self.mass = float(self.sim.data.params.mass.ravel()[0])
        self.hover_thrust = calibrate_hover_thrust(self._sim_step, self.sim.default_data,
                                                   config.num_envs, self.mass, config.sim_freq)
        self.residual_model = Residual()
        self.residual = residual
        self._buffer = max(config.latency_steps) + 1
        self._reset_jit = jax.jit(self._reset)
        self.step = jax.jit(self._step)

    def policy_spec(self):
        """Fields of drones.policy.runtime.SquareSpec: what a deployed copy of the policy needs."""
        cfg = self.config
        return dict(control_freq=cfg.control_freq, side=cfg.side, corner_radius=cfg.corner_radius,
                    lap_time=cfg.lap_time, height=cfg.height, max_tilt=cfg.max_tilt,
                    max_yaw_rate=cfg.max_yaw_rate, hover_thrust=float(self.hover_thrust),
                    thrust_min=float(self.thrust_min), thrust_max=float(self.thrust_max))

    # ------------------------------------------------------------------ API
    def reset(self, key):
        """Start a fresh episode in every world. Returns (state, observation)."""
        return self._reset_jit(key, self.sim.default_data)

    def _reset(self, key, default):
        n = self.num_envs
        key, k_episode, k_obs = jax.random.split(key, 3)
        zeros = jnp.zeros(n)
        state = SquareState(
            sim=default, default_sim=default, phase=zeros, lap_time=jnp.ones(n),
            direction=jnp.ones(n), rotation=zeros, origin=jnp.zeros((n, 3)), ref_yaw=zeros,
            yaw_cmd=zeros, thrust_gain=jnp.ones(n), latency=jnp.zeros(n, jnp.int32),
            actions=jnp.zeros((n, self._buffer, 4)), bias=jnp.zeros((n, 3)),
            steps=jnp.zeros(n, jnp.int32), episode_return=zeros, key=key)
        state = self._reset_worlds(state, jnp.ones(n, bool), k_episode)
        return state, self._observe(state, k_obs)

    def _step(self, state, action):
        """Advance every world one control period. Returns (state, obs, reward, done, info).

        Worlds that finish restart in the same call; their observation is already the first of the
        next episode. `info['final_critic']` is the critic observation of the state each world
        reached, before any restart, which SHAC bootstraps from after a timeout.
        """
        cfg, n = self.config, self.num_envs
        action = jnp.clip(action, -1.0, 1.0)
        actions = jnp.concatenate([action[:, None], state.actions[:, :-1]], 1)
        applied = jnp.take_along_axis(actions, state.latency[:, None, None], 1)[:, 0]
        sim, yaw_cmd = self.advance(state.sim, applied, state.yaw_cmd, state.thrust_gain,
                                    self.residual)
        steps = state.steps + 1
        s = sim.states
        pos, vel, quat, ang_vel = s.pos[:, 0], s.vel[:, 0], s.quat[:, 0], s.ang_vel[:, 0]

        ref_pos, ref_vel = self.reference_at(state, state.phase + steps / cfg.control_freq)
        e2 = jnp.sum(jnp.square(pos - ref_pos), -1)
        ev2 = jnp.sum(jnp.square(vel - ref_vel), -1)
        up = quat_to_matrix(quat)[:, 2, 2]
        tilt = jnp.arccos(jnp.clip(jax.lax.stop_gradient(up), -1.0, 1.0))
        crashed = ((pos[:, 2] < cfg.min_height) | (tilt > cfg.max_tilt_terminate)
                   | (e2 > cfg.max_error ** 2))
        truncated = (steps >= cfg.episode_steps) & ~crashed
        done = crashed | truncated
        reward = self._reward(e2, ev2, up, ang_vel, action, state.actions[:, 0], crashed)
        episode_return = state.episode_return + reward

        key, k_reset, k_drift, k_obs = jax.random.split(state.key, 4)
        drift = cfg.pos_drift * jnp.sqrt(1.0 / cfg.control_freq) * jax.random.normal(k_drift, (n, 3))
        state = state.replace(sim=sim, yaw_cmd=yaw_cmd, actions=actions, steps=steps,
                              episode_return=episode_return, key=key,
                              bias=state.bias + drift * jnp.array([1.0, 1.0, 0.0]))
        final = self._observe(state, k_obs)
        info = jax.lax.stop_gradient({
            'crashed': crashed,
            'truncated': truncated,
            'episode_return': jnp.where(done, episode_return, 0.0),
            'episode_length': jnp.where(done, steps, 0),
            'pos_error': jnp.sqrt(e2),
            'speed': jnp.linalg.norm(vel, axis=-1),
            'tilt': tilt,
        })
        info['final_critic'] = final['critic']
        # Most steps restart a few worlds, but skip the work entirely when none do.
        state = jax.lax.cond(done.any(), lambda st: self._reset_worlds(st, done, k_reset),
                             lambda st: st, state)
        return state, self._observe(state, k_obs), reward, done, info

    # ------------------------------------------------------------------ physics
    def advance(self, sim, action, yaw_cmd, thrust_gain, residual=None):
        """One control period of physics under `action`, the one the drone acts on now.

        Differentiable in everything. Returns (sim, yaw_cmd). drones.rl.sysid replays logged flights
        through it.
        """
        cfg = self.config
        quat = sim.states.quat[:, 0]
        yaw = yaw_from_quat(quat)
        yaw_cmd = yaw_cmd + action[:, 2] * cfg.max_yaw_rate / cfg.control_freq
        # The integrated yaw setpoint stays near the actual heading, so it cannot wind up.
        yaw_cmd = yaw + jnp.clip(wrap_angle(yaw_cmd - yaw), -YAW_BAND, YAW_BAND)
        roll, pitch, _, thrust = decode_action(
            jnp, action, max_tilt=cfg.max_tilt, max_yaw_rate=cfg.max_yaw_rate,
            hover_thrust=self.hover_thrust, thrust_min=self.thrust_min,
            thrust_max=self.thrust_max)
        cmd = jnp.stack([roll, pitch, yaw_cmd, thrust * thrust_gain], -1)[:, None]
        sim = attitude_control(sim, cmd)
        if residual is not None:
            s = sim.states
            force, torque = residual_wrench(self.residual_model, residual, s.vel[:, 0], quat,
                                            s.ang_vel[:, 0], action)
            # so_rpy adds these as disturbances every substep, and nothing clears them: they hold
            # for the whole control period.
            sim = sim.replace(states=s.replace(force=force[:, None], torque=torque[:, None]))
        return self._sim_step(sim, n_steps=self.substeps), yaw_cmd

    def with_states(self, sim, pos, vel, quat, ang_vel):
        """`sim` with every world's drone put in the given state; arrays (n, 3) or (n, 4)."""
        return sim.replace(states=sim.states.replace(
            pos=pos[:, None], vel=vel[:, None], quat=quat[:, None], ang_vel=ang_vel[:, None]))

    def reference_at(self, state, t):
        """Each world's reference at times t, shape (n,) or (n, k). Returns (pos, vel)."""
        extra = (1,) * (t.ndim - 1)

        def per_world(x):
            return x.reshape(x.shape[:1] + extra + x.shape[1:])

        cfg = self.config
        return square_reference(jnp, t, side=cfg.side, corner_radius=cfg.corner_radius,
                                lap_time=per_world(state.lap_time),
                                direction=per_world(state.direction),
                                rotation=per_world(state.rotation), origin=per_world(state.origin))

    # ------------------------------------------------------------------ episodes
    def _sample_episodes(self, key, n):
        cfg = self.config
        k = jax.random.split(key, 11)

        def uniform(k, lo, hi, shape=(n,)):
            return jax.random.uniform(k, shape, minval=lo, maxval=hi)

        lap_time = uniform(k[0], *cfg.lap_time)
        height = uniform(k[1], *cfg.height)
        direction = jnp.where(jax.random.bernoulli(k[2], 0.5, (n,)), 1.0, -1.0)
        rotation = uniform(k[3], -jnp.pi, jnp.pi)
        phase = uniform(k[4], 0.0, 1.0) * lap_time
        ref_yaw = uniform(k[5], -jnp.pi, jnp.pi)
        # Place each square so its reference passes through the world origin as the episode starts.
        start, start_vel = square_reference(jnp, phase, side=cfg.side,
                                            corner_radius=cfg.corner_radius, lap_time=lap_time,
                                            direction=direction, rotation=rotation,
                                            origin=jnp.zeros((n, 3)))
        origin = jnp.stack([-start[:, 0], -start[:, 1], height], -1)
        squash = jnp.array([1.0, 1.0, 0.5])
        pos = (jnp.zeros((n, 3)).at[:, 2].set(height)
               + uniform(k[6], -1.0, 1.0, (n, 3)) * cfg.start_pos_error * squash)
        vel = start_vel + uniform(k[7], -1.0, 1.0, (n, 3)) * cfg.start_vel_error * squash
        roll_pitch = uniform(k[8], -cfg.start_tilt, cfg.start_tilt, (n, 2))
        quat = euler_to_quat(roll_pitch[:, 0], roll_pitch[:, 1], ref_yaw)
        gain = cfg.thrust_gain * (1.0 + uniform(k[9], -cfg.thrust_gain_range,
                                                cfg.thrust_gain_range))
        latency = jax.random.choice(k[10], jnp.array(cfg.latency_steps, jnp.int32), (n,))
        return dict(lap_time=lap_time, direction=direction, rotation=rotation, phase=phase,
                    ref_yaw=ref_yaw, origin=origin, pos=pos, vel=vel, quat=quat, gain=gain,
                    latency=latency)

    def _reset_worlds(self, state, mask, key):
        """Start new episodes in the worlds selected by `mask`; leave the others untouched.

        The new episodes are sampled independently of the old state, and `jnp.where` passes no
        gradient to the side it does not pick, so restarted worlds carry no gradient across.
        """
        n = self.num_envs
        e = self._sample_episodes(key, n)
        sim = self._sim_reset(state.sim, state.default_sim, mask)
        sim = sim.replace(states=leaf_replace(sim.states, mask, pos=e['pos'][:, None],
                                              vel=e['vel'][:, None], quat=e['quat'][:, None],
                                              ang_vel=jnp.zeros((n, 1, 3))))

        def pick(new, old):
            return jnp.where(mask.reshape(mask.shape + (1,) * (old.ndim - 1)), new, old)

        return state.replace(
            sim=sim, phase=pick(e['phase'], state.phase),
            lap_time=pick(e['lap_time'], state.lap_time),
            direction=pick(e['direction'], state.direction),
            rotation=pick(e['rotation'], state.rotation), origin=pick(e['origin'], state.origin),
            ref_yaw=pick(e['ref_yaw'], state.ref_yaw), yaw_cmd=pick(e['ref_yaw'], state.yaw_cmd),
            thrust_gain=pick(e['gain'], state.thrust_gain),
            latency=pick(e['latency'], state.latency),
            actions=pick(jnp.zeros_like(state.actions), state.actions),
            bias=pick(jnp.zeros_like(state.bias), state.bias),
            steps=pick(jnp.zeros_like(state.steps), state.steps),
            episode_return=pick(jnp.zeros_like(state.episode_return), state.episode_return))

    # ------------------------------------------------------------------ observations
    def _observe(self, state, key):
        """The policy and critic observations of the current state.

        The estimate noise is drawn from `key` and does not depend on the state, so under
        differentiation it is a constant offset: the gradient of the estimate is the gradient of
        the truth.
        """
        cfg, n = self.config, self.num_envs
        s = state.sim.states
        pos, vel, quat, ang_vel = s.pos[:, 0], s.vel[:, 0], s.quat[:, 0], s.ang_vel[:, 0]
        k_pos, k_vel, k_att, k_yaw = jax.random.split(key, 4)
        pos_est = pos + state.bias + cfg.pos_noise * jax.random.normal(k_pos, (n, 3))
        vel_est = vel + cfg.vel_noise * jax.random.normal(k_vel, (n, 3))
        gravity = -quat_to_matrix(quat)[:, 2, :]    # body-frame gravity direction
        gravity = gravity + cfg.attitude_noise * jax.random.normal(k_att, (n, 3))
        gravity = gravity / jnp.linalg.norm(gravity, axis=-1, keepdims=True)
        yaw_est = yaw_from_quat(quat) + cfg.attitude_noise * jax.random.normal(k_yaw, (n,))

        t = state.phase + state.steps / cfg.control_freq
        ref_pos, ref_vel = self.reference_at(state, t)
        ahead, _ = self.reference_at(
            state, t[:, None] + LOOKAHEAD_DT * jnp.arange(1, LOOKAHEAD + 1))
        policy = encode_square_obs(jnp, pos_est=pos_est, vel_est=vel_est, yaw_est=yaw_est,
                                   gravity=gravity, ref_pos=ref_pos, ref_vel=ref_vel,
                                   lookahead_pos=ahead, ref_yaw=state.ref_yaw,
                                   prev_action=state.actions[:, 0])
        angle = 2 * jnp.pi * jnp.mod(t / state.lap_time, 1.0)
        privileged = jnp.concatenate([
            (pos - ref_pos) / POS_SCALE,
            vel / VEL_SCALE,
            ang_vel / GYRO_SCALE,
            jnp.sin(angle)[:, None],
            jnp.cos(angle)[:, None],
            state.lap_time[:, None] / 10.0,
            state.thrust_gain[:, None] - 1.0,
        ], -1)
        return {'policy': policy, 'critic': jnp.concatenate([policy, privileged], -1)}

    # ------------------------------------------------------------------ reward
    @staticmethod
    def _reward(e2, ev2, up, ang_vel, action, prev_action, crashed):
        """About 1 to 2.5 per step while flying; CRASH_REWARD for a crash.

        Smooth wherever it is differentiated. The exponentials reward tight tracking, the quadratic
        keeps a gradient far from the reference, and the constant 1 keeps the reward near 0 even at
        the 1 m crash limit, so crashing never pays. Tilt squared is 2 * (1 - R_zz), which avoids
        arccos's infinite slope when level.
        """
        tilt2 = 2.0 * (1.0 - up)
        reward = (1.0 + jnp.exp(-e2 / 0.05 ** 2) + 0.5 * jnp.exp(-ev2 / 0.25 ** 2) - e2
                  - 0.1 * tilt2 - 0.01 * jnp.sum(jnp.square(ang_vel), -1)
                  - 0.05 * jnp.sum(jnp.square(action - prev_action), -1))
        return jnp.where(crashed, CRASH_REWARD, reward)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run --extra sim pytest tests/test_square_env.py -v`
Expected: all PASS. If the finite-difference test fails only on tolerance, rerun it with `eps = 3e-3` before changing any code. Float32 roundoff on a sum of 16 rewards is about 2e-3 per unit of `eps`.

- [ ] **Step 5: Commit**

```bash
git add src/drones/sim/square_env.py tests/test_square_env.py
git commit -m "feat: differentiable square-flying task on CrazyFlow"
```

---

### Task 5: SHAC

**Files:**
- Create: `src/drones/rl/shac.py`
- Test: `tests/test_shac.py`

**Interfaces:**
- Consumes:
  - `drones.rl.networks.MLP(hidden, out, out_scale)`
  - any env with `num_envs`, `action_size`, `reset(key)`, and a differentiable `step` whose `info` has `crashed`, `truncated`, `final_critic`, `episode_return`, `episode_length`, `pos_error`, `speed` and `tilt` (Task 4's `SquareEnv`)
- Produces:
  - `SHACConfig`
  - `Actor`, whose params are `{'params': {'net': {'Dense_0', …}, 'log_std'}}`
  - `Critic`, whose params are `{'params': {'net': {'Dense_0', …}}}`
  - `SHACState` (`actor`, `critic`, `target`, `actor_opt`, `critic_opt`, `env_state`, `obs`, `key`, `iteration`)
  - `td_lambda_returns(rewards, next_values, dones, crashed, gamma, lam)`
  - `SHAC(env, config)` with `.init(key) -> SHACState`, `.iterate(state) -> (state, stats)` (jitted), `.act(actor_params, obs) -> action`, `.batch_size`, `.config`, `._rollout(actor, target, env_state, obs, key)`
  - `params_of(state) -> {'actor', 'critic', 'target'}`

- [ ] **Step 1: Write the failing tests**

```python
"""SHAC pieces in isolation, a hand-checked rollout loss, and learning on the square task."""
import pytest

pytest.importorskip('crazyflow')

import jax
import jax.numpy as jnp
import numpy as np

from drones.rl.shac import SHAC, SHACConfig, td_lambda_returns
from drones.sim.square_env import SquareConfig, SquareEnv


def test_td_lambda_one_is_the_discounted_return():
    ret = td_lambda_returns(jnp.array([[1.0], [2.0], [3.0]]), jnp.array([[0.0], [0.0], [4.0]]),
                            jnp.zeros((3, 1), bool), jnp.zeros((3, 1), bool), gamma=0.5, lam=1.0)
    np.testing.assert_allclose(ret[:, 0], [3.25, 4.5, 5.0])


def test_td_lambda_zero_is_one_step_td():
    ret = td_lambda_returns(jnp.array([[1.0], [2.0]]), jnp.array([[10.0], [20.0]]),
                            jnp.zeros((2, 1), bool), jnp.zeros((2, 1), bool), gamma=0.5, lam=0.0)
    np.testing.assert_allclose(ret[:, 0], [6.0, 12.0])


@pytest.mark.parametrize('crashed, expected', [(True, 1.0), (False, 5.0)])
def test_a_crash_has_no_future_and_a_timeout_does(crashed, expected):
    done = jnp.array([[True], [False]])
    ret = td_lambda_returns(jnp.array([[1.0], [1.0]]), jnp.array([[8.0], [8.0]]), done,
                            done & crashed, gamma=0.5, lam=0.95)
    assert float(ret[0, 0]) == pytest.approx(expected)


class LineEnv:
    """One world on a line: the action moves it and the reward is where it is. Each episode ends
    after two steps, in a crash or a timeout as chosen."""
    num_envs, action_size = 1, 1

    def __init__(self, crash):
        self.crash = crash

    @staticmethod
    def _obs(x):
        return {'policy': x[:, None], 'critic': x[:, None]}

    def reset(self, key):
        state = {'x': jnp.zeros(1), 't': jnp.zeros(1, jnp.int32)}
        return state, self._obs(state['x'])

    def step(self, state, action):
        x, t = state['x'] + action[:, 0], state['t'] + 1
        done = t >= 2
        crashed = done & self.crash
        zero = jnp.zeros(1)
        info = {'crashed': crashed, 'truncated': done & ~crashed, 'final_critic': x[:, None],
                'episode_return': zero, 'episode_length': jnp.zeros(1, jnp.int32),
                'pos_error': zero, 'speed': zero, 'tilt': zero}
        reward = x
        x, t = jnp.where(done, 0.0, x), jnp.where(done, 0, t)
        return {'x': x, 't': t}, self._obs(x), reward, done, info


@pytest.mark.parametrize('crash, expected_loss', [(True, -0.5 / 3), (False, -0.75 / 3)])
def test_rollout_bootstraps_after_a_timeout_but_not_a_crash(crash, expected_loss):
    # Zero actions, so every reward is 0 and the loss is the bootstrapped values alone. The target
    # critic returns 1 everywhere. With gamma 0.5 over 3 steps and an episode ending at step 2:
    # a crash leaves only the window-end value, 0.5 * 1; a timeout adds 0.25 * 1 for its end.
    agent = SHAC(LineEnv(crash), SHACConfig(horizon=3, gamma=0.5, hidden=(4,), remat=False))
    state = agent.init(jax.random.key(0))
    path_str = jax.tree_util.keystr
    actor = jax.tree_util.tree_map_with_path(
        lambda p, x: jnp.full_like(x, -30.0) if 'log_std' in path_str(p) else jnp.zeros_like(x),
        state.actor)
    target = jax.tree_util.tree_map_with_path(
        lambda p, x: (jnp.ones_like(x) if "'Dense_1'" in path_str(p) and "'bias'" in path_str(p)
                      else jnp.zeros_like(x)), state.critic)
    loss, _ = agent._rollout(actor, target, state.env_state, state.obs, jax.random.key(1))
    assert float(loss) == pytest.approx(expected_loss, abs=1e-6)


@pytest.fixture(scope='module')
def square():
    return SquareEnv(SquareConfig(num_envs=8))


def test_one_iteration_runs_and_updates(square):
    agent = SHAC(square, SHACConfig(iterations=2, horizon=4, critic_epochs=2,
                                    critic_minibatches=2, hidden=(16, 16)))
    state = agent.init(jax.random.key(0))
    new, stats = agent.iterate(state)
    assert all(bool(jnp.isfinite(v)) for k, v in stats.items()
               if k not in ('episode_return', 'episode_length', 'crash_rate'))
    assert float(stats['skipped']) == 0.0
    changed = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), new.actor, state.actor)
    assert any(jax.tree.leaves(changed))
    assert int(new.iteration) == 1


def test_reward_improves_on_the_square():
    env = SquareEnv(SquareConfig(num_envs=32))
    agent = SHAC(env, SHACConfig(iterations=20, horizon=32, hidden=(64, 64)))
    state = agent.init(jax.random.key(0))
    rewards = []
    for _ in range(20):
        state, stats = agent.iterate(state)
        rewards.append(float(stats['reward']))
    assert np.mean(rewards[-5:]) > np.mean(rewards[:5])
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --extra sim pytest tests/test_shac.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'drones.rl.shac'`

- [ ] **Step 3: Implement `src/drones/rl/shac.py`**

```python
"""Short-Horizon Actor-Critic, compiled end to end with JAX.

Xu et al., "Accelerated Policy Learning with Parallel Differentiable Simulation", ICLR 2022. The
actor is trained by backpropagating the return of a short rollout, `horizon` steps, through the
simulator, with a critic's value at the end of the window standing in for the rest of the episode.
Short windows keep the gradients from exploding through long rollouts; the critic keeps the policy
from being short-sighted. The critic is trained on TD(lambda) targets from the same rollout,
bootstrapped from a slowly updated target copy.

Works with any env that has `num_envs`, `action_size`, `reset(key)` and a differentiable
`step(state, action)` reporting `crashed`, `truncated` and `final_critic` (the critic observation of
the state an episode ended in) in its info, as drones.sim.square_env.SquareEnv does.
"""
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct

from drones.rl.networks import MLP


@dataclass(frozen=True)
class SHACConfig:
    iterations: int = 2000
    horizon: int = 32
    gamma: float = 0.99
    td_lambda: float = 0.95
    actor_lr: float = 2e-3
    critic_lr: float = 5e-4
    lr_decay: bool = True
    betas: tuple[float, float] = (0.7, 0.95)
    max_grad_norm: float = 1.0
    critic_epochs: int = 16
    critic_minibatches: int = 4
    target_alpha: float = 0.2      # target <- alpha * target + (1 - alpha) * critic
    init_log_std: float = -1.0
    hidden: tuple[int, ...] = (128, 128)
    remat: bool = True             # recompute env steps in the backward pass to save memory
    seed: int = 0


class Actor(nn.Module):
    action_size: int
    hidden: tuple[int, ...]
    init_log_std: float = -1.0

    def setup(self):
        # A small output scale starts the policy near zero action, which is calibrated hover.
        self.net = MLP(self.hidden, self.action_size, out_scale=0.01)
        self.log_std = self.param('log_std', nn.initializers.constant(self.init_log_std),
                                  (self.action_size,))

    def __call__(self, obs):
        """(action mean, log std) from the policy observation."""
        return self.net(obs), self.log_std


class Critic(nn.Module):
    hidden: tuple[int, ...]

    @nn.compact
    def __call__(self, obs):
        return MLP(self.hidden, 1, name='net')(obs)[..., 0]


@struct.dataclass
class SHACState:
    actor: dict
    critic: dict
    target: dict
    actor_opt: optax.OptState
    critic_opt: optax.OptState
    env_state: object
    obs: dict
    key: jax.Array
    iteration: jax.Array


def params_of(state):
    """What a checkpoint holds: the actor, the critic and the target critic."""
    return {'actor': state.actor, 'critic': state.critic, 'target': state.target}


def td_lambda_returns(rewards, next_values, dones, crashed, gamma, lam):
    """TD(lambda) targets over time-major arrays (T, n).

    `next_values[t]` is the value of the state reached by step t, before any restart. `dones[t]`
    means the episode ended there, and `crashed[t]` that it ended in a crash, which has no future.
    The last step bootstraps fully from its next value.
    """
    def step(next_return, x):
        reward, next_value, done, crash = x
        blended = (1.0 - lam) * next_value + lam * next_return
        ret = reward + gamma * jnp.where(done, next_value * (1.0 - crash), blended)
        return ret, ret

    _, returns = jax.lax.scan(step, next_values[-1],
                              (rewards, next_values, dones, crashed.astype(rewards.dtype)),
                              reverse=True)
    return returns


class SHAC:
    def __init__(self, env, config: SHACConfig = SHACConfig()):
        self.env = env
        self.config = config
        self.actor = Actor(env.action_size, config.hidden, config.init_log_std)
        self.critic = Critic(config.hidden)
        self.batch_size = env.num_envs * config.horizon
        if self.batch_size % config.critic_minibatches:
            raise ValueError('num_envs * horizon must divide into critic_minibatches')
        critic_updates = config.iterations * config.critic_epochs * config.critic_minibatches
        actor_lr, critic_lr = config.actor_lr, config.critic_lr
        if config.lr_decay:
            actor_lr = optax.linear_schedule(actor_lr, 0.0, config.iterations)
            critic_lr = optax.linear_schedule(critic_lr, 0.0, critic_updates)
        b1, b2 = config.betas
        self.actor_opt = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                                     optax.adam(actor_lr, b1=b1, b2=b2))
        self.critic_opt = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                                      optax.adam(critic_lr, b1=b1, b2=b2))
        self._env_step = jax.checkpoint(env.step) if config.remat else env.step
        self.iterate = jax.jit(self._iterate)

    def init(self, key):
        key, k_env, k_actor, k_critic = jax.random.split(key, 4)
        env_state, obs = self.env.reset(k_env)
        actor = self.actor.init(k_actor, obs['policy'])
        critic = self.critic.init(k_critic, obs['critic'])
        return SHACState(actor=actor, critic=critic, target=critic,
                         actor_opt=self.actor_opt.init(actor),
                         critic_opt=self.critic_opt.init(critic), env_state=env_state, obs=obs,
                         key=key, iteration=jnp.zeros((), jnp.int32))

    def act(self, actor_params, obs):
        """Deterministic action, the policy mean. Reads only obs['policy']: deployable."""
        return self.actor.apply(actor_params, obs['policy'])[0]

    def _rollout(self, actor_params, target_params, env_state, obs, key):
        """The actor loss of one window, differentiable in `actor_params`.

        Returns (loss, (env_state, obs, trajectory)); the trajectory is gradient-stopped.
        Each world's return is summed from the window start or from its latest restart. When an
        episode ends in the window, the target critic's value of the state it ended in closes it,
        unless it crashed. The window's end is closed the same way.
        """
        cfg, n = self.config, self.env.num_envs

        def value(o):
            return self.critic.apply(target_params, o)

        def body(carry, eps):
            env_state, obs, discount, running, total = carry
            mean, log_std = self.actor.apply(actor_params, obs['policy'])
            action = mean + jnp.exp(log_std) * eps
            env_state, next_obs, reward, done, info = self._env_step(env_state, action)
            crashed = info['crashed'].astype(reward.dtype)
            next_value = value(info['final_critic'])
            running = running + discount * reward
            discount = discount * cfg.gamma
            closed = running + discount * next_value * (1.0 - crashed)
            total = total + jnp.sum(jnp.where(done, closed, 0.0))
            running = jnp.where(done, 0.0, running)
            discount = jnp.where(done, 1.0, discount)
            record = {'critic_obs': obs['critic'], 'reward': reward, 'done': done,
                      'crashed': info['crashed'], 'next_value': next_value,
                      'info': {k: v for k, v in info.items() if k != 'final_critic'}}
            return (env_state, next_obs, discount, running, total), record

        noise = jax.random.normal(key, (cfg.horizon, n, self.env.action_size))
        init = (env_state, obs, jnp.ones(n), jnp.zeros(n), jnp.zeros(()))
        (env_state, obs, discount, running, total), traj = jax.lax.scan(body, init, noise)
        total = total + jnp.sum(running + discount * value(obs['critic']))
        loss = -total / (n * cfg.horizon)
        return loss, (env_state, obs, jax.lax.stop_gradient(traj))

    def _iterate(self, ts):
        """One iteration: a differentiable window, an actor step, then critic fitting."""
        cfg = self.config
        key, k_noise, k_critic = jax.random.split(ts.key, 3)
        # The window starts from where the last one ended, but no gradient flows back across it.
        env_state = jax.lax.stop_gradient(ts.env_state)
        obs = jax.lax.stop_gradient(ts.obs)
        (loss, (env_state, obs, traj)), grads = jax.value_and_grad(self._rollout, has_aux=True)(
            ts.actor, ts.target, env_state, obs, k_noise)

        grad_norm = optax.global_norm(grads)
        finite = jnp.isfinite(grad_norm)
        updates, actor_opt = self.actor_opt.update(grads, ts.actor_opt, ts.actor)
        actor = optax.apply_updates(ts.actor, updates)

        def keep_if_finite(new, old):
            return jax.tree.map(lambda a, b: jnp.where(finite, a, b), new, old)

        actor, actor_opt = keep_if_finite(actor, ts.actor), keep_if_finite(actor_opt, ts.actor_opt)

        returns = td_lambda_returns(traj['reward'], traj['next_value'], traj['done'],
                                    traj['crashed'], cfg.gamma, cfg.td_lambda)
        critic, critic_opt, critic_loss = self._fit_critic(ts.critic, ts.critic_opt,
                                                           traj['critic_obs'], returns, k_critic)
        target = jax.tree.map(lambda t, c: cfg.target_alpha * t + (1 - cfg.target_alpha) * c,
                              ts.target, critic)

        info, done = traj['info'], traj['done']
        finished = done.sum()

        def per_episode(x):
            return jnp.where(finished > 0, x.sum() / jnp.maximum(finished, 1), jnp.nan)

        stats = {
            'actor_loss': loss,
            'grad_norm': grad_norm,
            'skipped': (~finite).astype(jnp.float32),
            'critic_loss': critic_loss,
            'episodes': finished,
            'episode_return': per_episode(info['episode_return']),
            'episode_length': per_episode(info['episode_length'].astype(jnp.float32)),
            'crash_rate': per_episode(info['crashed'].astype(jnp.float32)),
            'reward': traj['reward'].mean(),
            'pos_error': info['pos_error'].mean(),
            'speed': info['speed'].mean(),
            'tilt': info['tilt'].mean(),
            'action_std': jnp.exp(actor['params']['log_std']).mean(),
        }
        return ts.replace(actor=actor, critic=critic, target=target, actor_opt=actor_opt,
                          critic_opt=critic_opt, env_state=env_state, obs=obs, key=key,
                          iteration=ts.iteration + 1), stats

    def _fit_critic(self, params, opt_state, obs, returns, key):
        """Regress the critic onto the TD(lambda) targets: epochs of shuffled minibatches."""
        cfg = self.config
        batch = (obs.reshape(self.batch_size, -1), returns.reshape(-1))

        def epoch(carry, key):
            order = jax.random.permutation(key, self.batch_size)
            shuffled = jax.tree.map(
                lambda x: x[order].reshape(cfg.critic_minibatches, -1, *x.shape[1:]), batch)

            def minibatch(carry, mb):
                params, opt_state = carry
                o, target = mb
                loss, grads = jax.value_and_grad(
                    lambda p: jnp.mean(jnp.square(self.critic.apply(p, o) - target)))(params)
                updates, opt_state = self.critic_opt.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), loss

            carry, losses = jax.lax.scan(minibatch, carry, shuffled)
            return carry, losses.mean()

        (params, opt_state), losses = jax.lax.scan(epoch, (params, opt_state),
                                                   jax.random.split(key, cfg.critic_epochs))
        return params, opt_state, losses[-1]
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run --extra sim pytest tests/test_shac.py -v`
Expected: all PASS. The last test compiles a 32-step backward pass and takes a few minutes on a CPU.

- [ ] **Step 5: Commit**

```bash
git add src/drones/rl/shac.py tests/test_shac.py
git commit -m "feat: SHAC, short-horizon actor-critic through differentiable CrazyFlow"
```

---

### Task 6: The square policy artifact

**Files:**
- Modify: `src/drones/policy/runtime.py`
- Modify: `src/drones/rl/export.py`
- Test: `tests/test_square_policy_runtime.py`

**Interfaces:**
- Consumes: `drones.policy.square` (Task 1); `SquareEnv.policy_spec()` and `reference_at` (Task 4); `Actor` params layout `{'params': {'net': {...}, 'log_std'}}` (Task 5)
- Produces:
  - `FORMAT_VERSION = 2`; version 1 artifacts still load, as hover
  - `PolicySpec` gains `task: str = 'hover'`
  - `SquareSpec(control_freq, side, corner_radius, lap_time, height, max_tilt, max_yaw_rate, hover_thrust, thrust_min, thrust_max, task='square')` with `.observation_size` and `.decode(action)`
  - `Policy.load` returns a `Policy` whose `.spec` is a `PolicySpec` or a `SquareSpec`
  - `SquareRunner(policy, *, origin, ref_yaw, lap_time, direction=1.0, rotation=0.0, side=None)` with `.reference(t)`, `.observe(t, *, pos, vel, yaw, gravity)`, `.step(t, *, pos, vel, yaw, gravity) -> action` and `.ref_yaw`
  - in `drones.rl.export`: `mlp_layers(tree)` and `export_square_policy(env, actor_params, directory)`

- [ ] **Step 1: Write the failing tests**

```python
"""The square policy artifact: save, load, and observe exactly as the simulator does."""
import json
import math

import numpy as np
import pytest

from drones.policy.runtime import Policy, PolicySpec, SquareRunner, SquareSpec
from drones.policy.square import OBS_SIZE, heading

SPEC = SquareSpec(control_freq=50, side=1.0, corner_radius=0.15, lap_time=(6.0, 10.0),
                  height=(0.8, 1.2), max_tilt=0.35, max_yaw_rate=1.5, hover_thrust=0.44,
                  thrust_min=0.085, thrust_max=0.8)


def constant_policy(action):
    """A one-layer policy that ignores its input and outputs `action`."""
    return Policy(SPEC, [(np.zeros((OBS_SIZE, 4)), np.asarray(action, float))])


def test_square_artifact_round_trips(tmp_path):
    constant_policy([0.1, 0.2, 0.3, 0.4]).save(tmp_path)
    loaded = Policy.load(tmp_path)
    assert loaded.spec == SPEC
    assert json.loads((tmp_path / 'policy.json').read_text())['task'] == 'square'


def test_version_1_hover_artifacts_still_load(tmp_path):
    spec = dict(sensors=['multiranger', 'optical_flow'], history=3, control_freq=50,
                target_height=[0.5, 1.5], range_max=4.0, flow_gain=0.488, max_tilt=0.35,
                max_yaw_rate=1.5, hover_thrust=0.44, thrust_min=0.085, thrust_max=0.8,
                format_version=1)
    (tmp_path / 'policy.json').write_text(json.dumps(spec))
    size = 3 * 13
    np.savez(tmp_path / 'actor.npz', w0=np.zeros((size, 4)), b0=np.zeros(4))
    loaded = Policy.load(tmp_path)
    assert isinstance(loaded.spec, PolicySpec) and loaded.spec.task == 'hover'


def test_unknown_versions_are_refused(tmp_path):
    constant_policy([0, 0, 0, 0]).save(tmp_path)
    data = json.loads((tmp_path / 'policy.json').read_text())
    (tmp_path / 'policy.json').write_text(json.dumps({**data, 'format_version': 99}))
    with pytest.raises(ValueError, match='format 99'):
        Policy.load(tmp_path)


def test_runner_feeds_back_its_previous_action():
    runner = SquareRunner(constant_policy([0.1, -0.2, 0.3, 2.0]), origin=[0.0, 0.0, 1.0],
                          ref_yaw=0.0, lap_time=8.0)
    state = dict(pos=np.array([0.0, 0.0, 1.0]), vel=np.zeros(3), yaw=0.0,
                 gravity=np.array([0.0, 0.0, -1.0]))
    action = runner.step(0.0, **state)
    np.testing.assert_allclose(action, [0.1, -0.2, 0.3, 1.0], atol=1e-6)   # clipped to 1
    np.testing.assert_allclose(runner.observe(0.02, **state)[-4:], action)


def test_runner_rejects_a_square_too_small_for_its_corners():
    with pytest.raises(ValueError, match='side'):
        SquareRunner(constant_policy([0, 0, 0, 0]), origin=[0, 0, 1], ref_yaw=0.0, lap_time=8.0,
                     side=0.25)


def test_runner_observes_what_the_simulator_does():
    pytest.importorskip('crazyflow')
    import jax

    from drones.missions.fly_policy import gravity_and_tilt
    from drones.sim.square_env import SquareConfig, SquareEnv

    env = SquareEnv(SquareConfig(num_envs=2, pos_noise=0.0, pos_drift=0.0, vel_noise=0.0,
                                 attitude_noise=0.0))
    state, obs = env.reset(jax.random.key(0))
    s = state.sim.states
    runner = SquareRunner(Policy(SquareSpec(**env.policy_spec()),
                                 [(np.zeros((OBS_SIZE, 4)), np.zeros(4))]),
                          origin=np.asarray(state.origin[0]), ref_yaw=float(state.ref_yaw[0]),
                          lap_time=float(state.lap_time[0]),
                          direction=float(state.direction[0]),
                          rotation=float(state.rotation[0]))
    quat = np.asarray(s.quat[0, 0], float)
    gravity, _ = gravity_and_tilt(quat)
    got = runner.observe(float(state.phase[0]), pos=np.asarray(s.pos[0, 0]),
                         vel=np.asarray(s.vel[0, 0]), yaw=float(heading(np, quat)),
                         gravity=gravity)
    np.testing.assert_allclose(got, np.asarray(obs['policy'][0]), atol=1e-4)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --extra sim pytest tests/test_square_policy_runtime.py -v`
Expected: FAIL with `ImportError: cannot import name 'SquareRunner'`

- [ ] **Step 3: Extend `src/drones/policy/runtime.py`**

Update the module docstring's first line to `"""Run an exported policy, hover or square, with numpy alone.`. Then make these changes:

- Change the imports to `from drones.policy import interface, square`.
- Replace `FORMAT_VERSION = 1` with:

```python
FORMAT_VERSION = 2
# 1: hover artifacts from before the square task, without a `task` field.
READABLE_VERSIONS = (1, 2)
TUPLE_FIELDS = ('sensors', 'target_height', 'lap_time', 'height')


def _decode(spec, action):
    """Normalised action(s) -> roll, pitch (rad), yaw rate (rad/s), thrust (N)."""
    return interface.decode_action(
        np, np.asarray(action, np.float64), max_tilt=spec.max_tilt,
        max_yaw_rate=spec.max_yaw_rate, hover_thrust=spec.hover_thrust,
        thrust_min=spec.thrust_min, thrust_max=spec.thrust_max)
```

- In `PolicySpec`, add `task: str = 'hover'` just above `format_version`, and replace the body of `decode` with `return _decode(self, action)`.
- After `PolicySpec`, add:

```python
@dataclass(frozen=True)
class SquareSpec:
    """A square-flying policy: what it observes (drones.policy.square) and how its actions scale."""
    control_freq: int
    side: float
    corner_radius: float
    lap_time: tuple[float, float]
    height: tuple[float, float]
    max_tilt: float
    max_yaw_rate: float
    hover_thrust: float
    thrust_min: float
    thrust_max: float
    task: str = 'square'
    format_version: int = FORMAT_VERSION

    @property
    def observation_size(self):
        return square.OBS_SIZE

    def decode(self, action):
        return _decode(self, action)


SPECS = {'hover': PolicySpec, 'square': SquareSpec}
```

- In `Policy.__init__`, change `if 'camera' in spec.sensors:` to `if 'camera' in getattr(spec, 'sensors', ()):`.
- Replace `Policy.load` with:

```python
    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        data = json.loads((directory / 'policy.json').read_text())
        version = data.get('format_version')
        if version not in READABLE_VERSIONS:
            raise ValueError(f'{directory}: artifact format {version}, this code reads '
                             f'{list(READABLE_VERSIONS)}')
        task = data.get('task', 'hover')
        if task not in SPECS:
            raise ValueError(f'{directory}: unknown task {task!r}')
        data = {k: tuple(v) if k in TUPLE_FIELDS else v for k, v in data.items()}
        spec = SPECS[task](**{**data, 'task': task, 'format_version': FORMAT_VERSION})
        with np.load(directory / 'actor.npz') as arrays:
            count = sum(1 for name in arrays.files if name.startswith('w'))
            layers = [(arrays[f'w{i}'], arrays[f'b{i}']) for i in range(count)]
        return cls(spec, layers)
```

- At the end of the file, add:

```python
class SquareRunner:
    """Feeds the firmware's state estimate to a square policy exactly as the simulator does.

    The reference is at `origin` (x, y and the flight height) when `t` is 0. Its first edge runs
    along `rotation`, and the policy holds heading `ref_yaw`. `side` defaults to the trained side;
    a smaller one is useful on first flights.
    """

    def __init__(self, policy, *, origin, ref_yaw, lap_time, direction=1.0, rotation=0.0,
                 side=None):
        spec = policy.spec
        side = float(side or spec.side)
        if not 2 * spec.corner_radius < side:
            raise ValueError(f'side {side} m is too small for {spec.corner_radius} m corners')
        self.policy = policy
        self.params = dict(side=side, corner_radius=spec.corner_radius, lap_time=float(lap_time),
                           direction=float(direction), rotation=float(rotation),
                           origin=np.asarray(origin, np.float64))
        self.ref_yaw = float(ref_yaw)
        self.prev_action = np.zeros(interface.ACTION_SIZE, np.float32)

    def reference(self, t):
        """Reference (position, velocity) at t seconds since the square started."""
        return square.square_reference(np, np.asarray(t, np.float64), **self.params)

    def observe(self, t, *, pos, vel, yaw, gravity):
        ref_pos, ref_vel = self.reference(t)
        ahead, _ = self.reference(t + square.LOOKAHEAD_DT * np.arange(1, square.LOOKAHEAD + 1))

        def row(x):
            return np.asarray(x, np.float32)[None]

        return square.encode_square_obs(
            np, pos_est=row(pos), vel_est=row(vel), yaw_est=row(yaw), gravity=row(gravity),
            ref_pos=row(ref_pos), ref_vel=row(ref_vel), lookahead_pos=row(ahead),
            ref_yaw=row(self.ref_yaw), prev_action=self.prev_action[None])[0]

    def step(self, t, *, pos, vel, yaw, gravity):
        action = self.policy.act(self.observe(t, pos=pos, vel=vel, yaw=yaw,
                                              gravity=gravity)[None])[0]
        self.prev_action = action.astype(np.float32)
        return action
```

- [ ] **Step 4: Export square actors in `src/drones/rl/export.py`**

- Change the runtime import to `from drones.policy.runtime import Policy, PolicySpec, SquareSpec`.
- Replace the body of `export_policy` with the version below, and add the two new functions:

```python
def mlp_layers(tree):
    """(kernel, bias) pairs of an MLP's Dense_0, Dense_1, ... in order, as numpy arrays."""
    names = sorted((n for n in tree if n.startswith('Dense_')),
                   key=lambda name: int(name.rsplit('_', 1)[1]))
    return [(np.asarray(tree[n]['kernel']), np.asarray(tree[n]['bias'])) for n in names]


def export_policy(env, params, directory):
    """Write the actor of `params`, with `env`'s observation and action spec, to `directory`."""
    if env.image_shape is not None:
        raise ValueError('camera policies cannot be exported for the drone yet')
    layers = mlp_layers(params['params']['actor'])
    return Policy(PolicySpec(**env.policy_spec()), layers).save(directory)


def export_square_policy(env, actor_params, directory):
    """Write a SHAC actor for a drones.sim.square_env.SquareEnv to `directory`."""
    layers = mlp_layers(actor_params['params']['net'])
    return Policy(SquareSpec(**env.policy_spec()), layers).save(directory)
```

- [ ] **Step 5: Run the new and existing runtime tests**

Run: `uv run --extra sim pytest tests/test_square_policy_runtime.py tests/test_policy_runtime.py tests/test_policy_interface.py tests/test_fly_policy.py tests/test_architecture.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/drones/policy/runtime.py src/drones/rl/export.py tests/test_square_policy_runtime.py
git commit -m "feat: square policy artifact and runner, artifact format 2"
```

---

### Task 7: Square experiments, `drones-train-square` and `drones-eval-square`

**Files:**
- Modify: `src/drones/rl/experiment.py` (`load` and `override` take the allowed sections)
- Create: `src/drones/rl/square_experiment.py`, `configs/square/shac.yaml`, `src/drones/rl/train_square.py`, `src/drones/rl/evaluate_square.py`
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `tests/test_square_experiment.py`, `tests/test_train_square.py`

**Interfaces:**
- Consumes:
  - `experiment.merge`, `experiment._build` and `experiment._plain`
  - `SquareConfig`, `SquareEnv` (Task 4); `SHAC`, `SHACConfig`, `params_of` (Task 5); `export_square_policy` (Task 6)
  - `ppo.save_params`, `ppo.load_params`, `ppo.config_to_json`, `ppo.config_from_dict`
- Produces:
  - `square_experiment.resolve(path=None, preset=None, assignments=(), device=None) -> (SquareConfig, SHACConfig)`, plus `dump(env, shac, path)` and `PRESETS`
  - `train_square.train(agent, state, run, log_every=10, save_every=100) -> state`, `train_square.write_config(run, env_config, shac_config)`, `train_square.main(argv)`
  - `evaluate_square.evaluate_square(env, act, key) -> dict` with `crash_rate`, `survived_seconds`, `laps`, `pos_rmse_m`, `max_error_m`, `return`
  - `evaluate_square.load_square_run(run, num_envs, device, corrected=True) -> (env, agent, params)`
  - the scripts `drones-train-square` and `drones-eval-square`

- [ ] **Step 1: Write the failing tests**

`tests/test_square_experiment.py`:

```python
"""Square experiment configs: defaults, presets, overrides, and a round trip through YAML."""
import pytest

pytest.importorskip('crazyflow')

from drones.rl import square_experiment
from drones.rl.shac import SHACConfig
from drones.sim.square_env import SquareConfig


def test_the_default_config_is_sized_for_a_gpu():
    env, shac = square_experiment.resolve()
    assert isinstance(env, SquareConfig) and isinstance(shac, SHACConfig)
    assert env.num_envs == 4096 and env.side == 1.0 and env.latency_steps == (0, 1, 2)


def test_presets_and_overrides_apply_in_order():
    env, shac = square_experiment.resolve(preset='cpu-test',
                                          assignments=['shac.horizon=16', 'env.lap_time=[8, 8]'])
    assert env.num_envs == 32 and shac.iterations == 20
    assert shac.horizon == 16 and env.lap_time == (8.0, 8.0)


def test_hover_sections_are_refused(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('ppo:\n  total_steps: 5\n')
    with pytest.raises(ValueError, match='unknown sections'):
        square_experiment.resolve(path)


def test_dumped_config_resolves_to_the_same_thing(tmp_path):
    env, shac = square_experiment.resolve(preset='cpu', device='cpu')
    square_experiment.dump(env, shac, tmp_path / 'config.yaml')
    assert square_experiment.resolve(tmp_path / 'config.yaml') == (env, shac)
```

`tests/test_train_square.py`:

```python
"""drones-train-square end to end on a tiny run, then drones-eval-square's pieces on its output."""
import pytest

pytest.importorskip('crazyflow')

import jax

from drones.policy.runtime import Policy, SquareSpec
from drones.rl.evaluate_square import evaluate_square, load_square_run
from drones.rl.train_square import main


def test_a_tiny_run_writes_everything_and_evaluates(tmp_path):
    main(['--preset', 'cpu-test', '--set', 'env.num_envs=4', '--set', 'shac.iterations=2',
          '--set', 'shac.horizon=4', '--set', 'shac.critic_minibatches=2',
          '--runs', str(tmp_path), '--name', 'tiny'])
    run = tmp_path / 'tiny'
    for name in ('config.yaml', 'config.json', 'metrics.csv', 'params.msgpack'):
        assert (run / name).exists(), name
    assert isinstance(Policy.load(run / 'policy').spec, SquareSpec)

    env, agent, params = load_square_run(run, num_envs=4, device='cpu')
    result = evaluate_square(env, jax.jit(lambda o: agent.act(params['actor'], o)),
                             jax.random.key(0))
    assert set(result) == {'crash_rate', 'survived_seconds', 'laps', 'pos_rmse_m',
                           'max_error_m', 'return'}
    assert 0.0 <= result['crash_rate'] <= 1.0
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --extra sim pytest tests/test_square_experiment.py tests/test_train_square.py -v`
Expected: FAIL with `ImportError: cannot import name 'square_experiment'`

- [ ] **Step 3: Let `experiment.load` and `experiment.override` take their sections**

In `src/drones/rl/experiment.py`:

```python
def load(path, sections=SECTIONS):
    """Read a config file into a nested dict, resolving any `extends` chain."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a mapping at the top level')
    parent = data.pop('extends', None)
    if parent is not None:
        data = merge(load(path.parent / parent, sections), data)
    unknown = sorted(set(data) - set(sections))
    if unknown:
        raise ValueError(f'{path}: unknown sections {unknown}; expected {list(sections)}')
    return data
```

```python
def override(data, assignments, sections=SECTIONS):
    """Apply `section.key=value` assignments; each value is parsed as YAML."""
    for assignment in assignments:
        path, sep, raw = assignment.partition('=')
        section, dot, key = path.partition('.')
        if not sep or not dot or section not in sections or not key:
            raise ValueError(f'expected SECTION.KEY=VALUE with SECTION one of {list(sections)}, '
                             f'got {assignment!r}')
        data = merge(data, {section: {key: yaml.safe_load(raw)}})
    return data
```

- [ ] **Step 4: Create `src/drones/rl/square_experiment.py`**

```python
"""Square experiments as YAML: the task and the SHAC settings.

    extends: shac.yaml     # optional, relative to this file; merged key by key
    env:                   # drones.sim.square_env.SquareConfig
      num_envs: 4096
    shac:                  # drones.rl.shac.SHACConfig
      iterations: 4000

The rules are drones.rl.experiment's: unknown keys are errors, and values are coerced to the field's
type.
"""
from dataclasses import asdict
from pathlib import Path

import yaml

from drones.rl import experiment
from drones.rl.shac import SHACConfig
from drones.sim.square_env import SquareConfig

CONFIG_DIR = experiment.CONFIG_DIR.parent / 'square'
DEFAULT_CONFIG = CONFIG_DIR / 'shac.yaml'
SECTIONS = ('env', 'shac')

# Scale overlays: the same experiment, sized for the machine it runs on.
PRESETS = {
    # Checks that everything runs end to end on a laptop; will not learn much.
    'cpu-test': {'env': {'num_envs': 32}, 'shac': {'iterations': 20}},
    # Enough to see the square taking shape on a multi-core CPU.
    'cpu': {'env': {'num_envs': 256}, 'shac': {'iterations': 1000}},
    # A full run on one GPU.
    'gpu': {'env': {'num_envs': 4096}, 'shac': {'iterations': 4000}},
}


def build(data):
    """(SquareConfig, SHACConfig) from a loaded config dict."""
    env = experiment._build(SquareConfig, data.get('env') or {}, 'env')
    shac = experiment._build(SHACConfig, data.get('shac') or {}, 'shac')
    return env, shac


def resolve(path=None, preset=None, assignments=(), device=None):
    """Load a config, then apply a preset, `--set` assignments and a device, in that order."""
    data = experiment.load(path or DEFAULT_CONFIG, SECTIONS)
    if preset:
        data = experiment.merge(data, PRESETS[preset])
    data = experiment.override(data, assignments, SECTIONS)
    if device:
        data = experiment.merge(data, {'env': {'device': device}})
    return build(data)


def dump(env, shac, path):
    """Write the resolved config in file form; loading it back gives the same dataclasses."""
    data = {'env': experiment._plain(asdict(env)), 'shac': experiment._plain(asdict(shac))}
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))
```

- [ ] **Step 5: Create `configs/square/shac.yaml`**

```yaml
# Fly a 1 x 1 m square with SHAC, from the firmware's state estimate.
#
# Scaled for a GPU. Shrink it on a CPU with --preset cpu-test or --preset cpu.

env:
  num_envs: 4096
  control_freq: 50
  episode_seconds: 16.0
  side: 1.0
  corner_radius: 0.15
  lap_time: [6.0, 10.0]
  height: [0.8, 1.2]
  thrust_gain_range: 0.1
  latency_steps: [0, 1, 2]

shac:
  iterations: 4000
  horizon: 32
  actor_lr: 2.0e-3
  critic_lr: 5.0e-4
  hidden: [128, 128]
```

- [ ] **Step 6: Create `src/drones/rl/train_square.py`**

```python
"""Train a square-flying policy with SHAC on CrazyFlow, from a YAML experiment config.

    uv run --extra sim drones-train-square --preset cpu-test           # a quick check
    uv run --extra sim drones-train-square --preset cpu
    uv run --extra sim --extra gpu drones-train-square --device gpu
    uv run --extra sim drones-train-square --set shac.horizon=16 --set env.lap_time=[8,8]

The config defaults to configs/square/shac.yaml. Each run writes to runs/<name>/:
    config.yaml     the fully resolved config; pass it back in to rerun exactly
    config.json     the same, read by drones-eval-square and drones-finetune-square
    metrics.csv     training curves
    params.msgpack  actor, critic and target critic (every --save-every iterations and at the end)
    policy/         the numpy artifact drones-fly-square flies
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

PRESET_NAMES = ('cpu-test', 'cpu', 'gpu')
LOGGED = ('episode_return', 'episode_length', 'crash_rate', 'pos_error', 'speed', 'tilt',
          'reward', 'action_std', 'actor_loss', 'critic_loss', 'grad_norm', 'skipped')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('config', nargs='?', type=Path,
                        help='experiment YAML (default: configs/square/shac.yaml)')
    parser.add_argument('--preset', choices=PRESET_NAMES, help='resize the experiment')
    parser.add_argument('--device', help='cpu or gpu (default: from the config, else cpu)')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='SECTION.KEY=VALUE', help='override one setting; repeatable')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--name', help='run directory name; defaults to a timestamp')
    parser.add_argument('--log-every', type=int, default=10, help='iterations between log lines')
    parser.add_argument('--save-every', type=int, default=100, help='iterations between checkpoints')
    return parser.parse_args(argv)


def write_config(run, env_config, shac_config):
    from drones.rl import square_experiment
    from drones.rl.ppo import config_to_json

    square_experiment.dump(env_config, shac_config, run / 'config.yaml')
    (run / 'config.json').write_text(json.dumps({
        'env': json.loads(config_to_json(env_config)),
        'shac': json.loads(config_to_json(shac_config)),
    }, indent=2))


def train(agent, state, run, log_every=10, save_every=100):
    """Run the agent's configured iterations, logging to run/metrics.csv and checkpointing to
    run/params.msgpack. Returns the final state."""
    import jax

    from drones.rl.ppo import save_params
    from drones.rl.shac import params_of

    iterations = agent.config.iterations
    with open(run / 'metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('iteration', 'env_steps', 'seconds') + LOGGED)
        start = time.time()
        for i in range(iterations):
            state, stats = agent.iterate(state)
            last = i == iterations - 1
            if i % log_every == 0 or last:
                stats = jax.device_get(stats)
                elapsed = time.time() - start
                steps = (i + 1) * agent.batch_size
                writer.writerow((i, steps, round(elapsed, 1))
                                + tuple(float(stats[k]) for k in LOGGED))
                f.flush()
                print(f'it {i:5d} | {steps / max(elapsed, 1e-9):8.0f} steps/s | '
                      f'return {stats["episode_return"]:7.1f} | '
                      f'crash {stats["crash_rate"]:4.2f} | '
                      f'err {stats["pos_error"]:.3f} m | reward {stats["reward"]:.3f} | '
                      f'|g| {stats["grad_norm"]:.2f}', flush=True)
            if i % save_every == 0 or last:
                save_params(run / 'params.msgpack', params_of(state))
    return state


def main(argv=None):
    args = parse_args(argv)
    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
    except ImportError:
        sys.exit('Training needs the sim extra:  uv sync --extra sim')
    import jax

    from drones.rl import square_experiment
    from drones.rl.export import export_square_policy
    from drones.rl.shac import SHAC
    from drones.sim.square_env import SquareEnv

    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f'shac.seed={args.seed}')
    try:
        env_config, shac_config = square_experiment.resolve(args.config, args.preset, overrides,
                                                            args.device)
    except (ValueError, OSError) as exc:
        sys.exit(f'Config error: {exc}')

    with jax.default_device(jax.devices(env_config.device)[0]):
        run = args.runs / (args.name or time.strftime('square-%Y%m%d-%H%M%S'))
        run.mkdir(parents=True, exist_ok=True)
        write_config(run, env_config, shac_config)
        print(f'Building {env_config.num_envs} worlds on {env_config.device} ...', flush=True)
        env = SquareEnv(env_config)
        agent = SHAC(env, shac_config)
        state = agent.init(jax.random.key(shac_config.seed))
        print(f'Hover thrust {env.hover_thrust:.4f} N | observation {env.policy_size} values | '
              f'{shac_config.iterations} iterations of {agent.batch_size} steps | run dir {run}',
              flush=True)
        state = train(agent, state, run, args.log_every, args.save_every)
        print(f'Saved {run / "params.msgpack"}')
        print(f'Policy artifact: {export_square_policy(env, state.actor, run / "policy")}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 7: Create `src/drones/rl/evaluate_square.py`**

```python
"""Evaluate a trained square policy against open-loop hover.

    uv run --extra sim drones-eval-square runs/<name>
    uv run --extra sim drones-eval-square runs/<name>-ft --uncorrected   # without the fitted residual

Runs every world for one full episode with deterministic actions, then again with zero action, and
prints both. Tracking error is measured after the first second, once the start error is recovered.
"""
import argparse
import json
import sys
from pathlib import Path


def evaluate_square(env, act, key):
    """Run each world for one episode. `act(obs)` returns actions. Returns summary numbers."""
    import jax
    import jax.numpy as jnp

    state, obs = env.reset(key)
    lap_time = state.lap_time

    def body(carry, _):
        state, obs, alive = carry
        state, obs, reward, done, info = env.step(state, act(obs))
        # Only the first episode of each world counts: stop recording once it ends.
        record = alive
        return (state, obs, alive & ~done), (record, reward, info['crashed'], info['pos_error'])

    steps, freq = env.config.episode_steps, env.config.control_freq
    _, (record, reward, crashed, error) = jax.lax.scan(
        body, (state, obs, jnp.ones(env.num_envs, bool)), None, length=steps)
    mask = record & (jnp.arange(steps)[:, None] >= freq)
    survived = record.sum(0) / freq
    return {
        'crash_rate': float((crashed & record).any(0).mean()),
        'survived_seconds': float(survived.mean()),
        'laps': float((survived / lap_time).mean()),
        'pos_rmse_m': float(jnp.sqrt((jnp.square(error) * mask).sum()
                                     / jnp.maximum(mask.sum(), 1))),
        'max_error_m': float(jnp.where(mask, error, 0.0).max()),
        'return': float((reward * record).sum(0).mean()),
    }


def load_square_run(run, num_envs, device, corrected=True):
    """The env, SHAC agent and trained parameters of a run directory.

    A run finetuned by drones-finetune-square carries residual.msgpack; the env uses it unless
    `corrected` is False. Call under jax.default_device.
    """
    import jax

    from drones.rl.ppo import config_from_dict, load_params
    from drones.rl.shac import SHAC, SHACConfig, params_of
    from drones.sim.residual import init_residual
    from drones.sim.square_env import SquareConfig, SquareEnv

    run = Path(run)
    saved = json.loads((run / 'config.json').read_text())
    env_config = config_from_dict(SquareConfig, {**saved['env'], 'num_envs': num_envs,
                                                 'device': device})
    residual = None
    if corrected and (run / 'residual.msgpack').exists():
        residual = load_params(run / 'residual.msgpack', init_residual(jax.random.key(0)))
    env = SquareEnv(env_config, residual=residual)
    agent = SHAC(env, config_from_dict(SHACConfig, saved['shac']))
    params = load_params(run / 'params.msgpack', params_of(agent.init(jax.random.key(0))))
    return env, agent, params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path, help='run directory written by drones-train-square')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--uncorrected', action='store_true',
                        help='ignore a fitted residual (runs from drones-finetune-square)')
    args = parser.parse_args(argv)
    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
    except ImportError:
        sys.exit('Evaluation needs the sim extra:  uv sync --extra sim')
    import jax
    import jax.numpy as jnp

    with jax.default_device(jax.devices(args.device)[0]):
        env, agent, params = load_square_run(args.run, args.num_envs, args.device,
                                             corrected=not args.uncorrected)
        policy = jax.jit(lambda o: agent.act(params['actor'], o))
        key = jax.random.key(args.seed)
        results = {
            'policy': evaluate_square(env, policy, key),
            'open-loop hover': evaluate_square(env, lambda o: jnp.zeros((env.num_envs, 4)), key),
        }
    print_table(results)


def print_table(results):
    names = list(next(iter(results.values())))
    print(f'{"":18s}' + ''.join(f'{n:>17s}' for n in names))
    for label, row in results.items():
        print(f'{label:18s}' + ''.join(f'{row[n]:17.3f}' for n in names))


if __name__ == '__main__':
    main()
```

- [ ] **Step 8: Register the scripts**

In `pyproject.toml` `[project.scripts]`, after `drones-export-policy`, add:

```toml
drones-train-square = "drones.rl.train_square:main"
drones-eval-square = "drones.rl.evaluate_square:main"
```

Then run `uv sync --extra sim` to reinstall the entry points.

- [ ] **Step 9: Run the tests**

Run: `uv run --extra sim pytest tests/test_square_experiment.py tests/test_train_square.py tests/test_experiment.py -v`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add src/drones/rl/experiment.py src/drones/rl/square_experiment.py configs/square/shac.yaml \
        src/drones/rl/train_square.py src/drones/rl/evaluate_square.py pyproject.toml \
        tests/test_square_experiment.py tests/test_train_square.py
git commit -m "feat: drones-train-square and drones-eval-square"
```

---

### Task 8: `drones-fly-square`

**Files:**
- Modify: `src/drones/missions/fly_policy.py`:
  - `SensorLog.__init__` (line 153) gains `blocks`
  - extract `take_off` out of `fly` (lines 209–217)
  - rename `_land` (line 244) to `land`
- Create: `src/drones/missions/fly_square.py`
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `tests/test_fly_square.py`

**Interfaces:**
- Consumes:
  - `SquareSpec`, `SquareRunner`, `Policy` (Task 6); `heading` (Task 1)
  - from `fly_policy`: `DEFAULT_SIGNS`, `RANGERS`, `THRUST_COMMAND_MIN`, `THRUST_COMMAND_MAX`, `gravity_and_tilt`, `ticks` and `to_setpoint`
- Produces:
  - in `fly_policy`: `take_off(commander, log, options, freq, now, sleep) -> hover_command`, `land(commander, log, options, freq, now, sleep)`, and `SensorLog(scf, now=..., blocks=None)`
  - in `fly_square`:
    - constants `LOG_FORMAT = '# format: square-log v1'`, `COLUMNS` and `LOG_BLOCKS`
    - `SquareLimits` and `SquareOptions`
    - `read_state(latest) -> (state dict, tilt_deg)`, where `state` has `pos`, `vel`, `quat`, `yaw`, `gravity`, `gyro` (rad/s, body), `zrange` and `ranges`
    - `firmware_action(latest, spec, hover_command, signs) -> action`
    - `square_abort_reason(state, tilt, log_age, ref_pos, limits)`
    - `square_recorder(path) -> (record, close)`, where `record(time, phase, state, ref_pos, ref_vel, action, setpoint, hover_command)`
    - `start_square(policy, latest, options) -> SquareRunner`
    - `fly_square(scf, policy, log, options, record=None, now=..., sleep=...) -> reason` and `main(argv)`

- [ ] **Step 1: Write the failing tests**

```python
"""drones-fly-square against a fake Crazyflie: conversions, sequencing, safety and the log."""
import csv
import math

import numpy as np
import pytest

from drones.missions.fly_policy import DEFAULT_SIGNS, to_setpoint
from drones.missions.fly_square import (COLUMNS, LOG_FORMAT, SquareLimits, SquareOptions,
                                        firmware_action, fly_square, read_state,
                                        square_abort_reason, square_recorder)
from drones.policy.runtime import Policy, SquareSpec
from drones.policy.square import OBS_SIZE

SPEC = SquareSpec(control_freq=50, side=1.0, corner_radius=0.15, lap_time=(6.0, 10.0),
                  height=(0.8, 1.2), max_tilt=0.35, max_yaw_rate=1.5, hover_thrust=0.44,
                  thrust_min=0.085, thrust_max=0.8)
HOVERING = {
    'stateEstimate.x': 0.0, 'stateEstimate.y': 0.0, 'stateEstimate.z': 1.0,
    'stateEstimate.vx': 0.0, 'stateEstimate.vy': 0.0, 'stateEstimate.vz': 0.0,
    'stateEstimate.qx': 0.0, 'stateEstimate.qy': 0.0, 'stateEstimate.qz': 0.0,
    'stateEstimate.qw': 1.0, 'gyro.x': 0.0, 'gyro.y': 0.0, 'gyro.z': 0.0,
    'controller.cmd_thrust': 38000.0, 'controller.roll': 0.0, 'controller.pitch': 0.0,
    'controller.yawRate': 0.0, 'range.zrange': 1000, 'range.front': 2000, 'range.back': 2000,
    'range.left': 2000, 'range.right': 2000, 'range.up': 8190,
}


def constant_policy(action=(0.0, 0.0, 0.0, 0.0)):
    return Policy(SPEC, [(np.zeros((OBS_SIZE, 4)), np.asarray(action, float))])


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, args))


class FakeLink:
    def __init__(self):
        self.cf = type('CF', (), {})()
        self.cf.commander = Recorder()
        self.cf.supervisor = Recorder()


class FakeLog:
    def __init__(self, **changes):
        self.latest = {**HOVERING, **changes}

    def age(self):
        return 0.0


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def run(policy=None, log=None, record=None, **options):
    """Fly 0.1 of a 6 s lap: short enough that a drone standing still stays on the square."""
    link, clock = FakeLink(), Clock()
    options = {'lap_time': 6.0, 'laps': 0.1, **options}
    reason = fly_square(link, policy or constant_policy(), log or FakeLog(),
                        SquareOptions(**options), record=record, now=clock.now, sleep=clock.sleep)
    return reason, link.cf.commander.calls, link.cf.supervisor.calls


def names(calls):
    return [name for name, _ in calls]


# ---------------------------------------------------------------------- conversions
def test_read_state_converts_units_and_heading():
    half = math.sqrt(0.5)
    state, tilt = read_state({**HOVERING, 'stateEstimate.qz': half, 'stateEstimate.qw': half,
                              'gyro.z': 90.0, 'stateEstimate.vx': 0.3})
    assert state['yaw'] == pytest.approx(math.pi / 2)
    np.testing.assert_allclose(state['gyro'], [0.0, 0.0, math.pi / 2])
    np.testing.assert_allclose(state['vel'], [0.3, 0.0, 0.0])
    np.testing.assert_allclose(state['gravity'], [0.0, 0.0, -1.0], atol=1e-12)
    assert state['zrange'] == pytest.approx(1.0) and tilt == pytest.approx(0.0)


@pytest.mark.parametrize('action', [[0.2, -0.4, 0.5, 0.3], [-0.6, 0.1, -0.2, -0.5]])
def test_firmware_action_inverts_to_setpoint(action):
    roll, pitch, yaw_rate, thrust = to_setpoint(np.array(action), SPEC, 38000.0, DEFAULT_SIGNS)
    latest = {'controller.roll': roll, 'controller.pitch': pitch, 'controller.yawRate': yaw_rate,
              'controller.cmd_thrust': float(thrust)}
    np.testing.assert_allclose(firmware_action(latest, SPEC, 38000.0, DEFAULT_SIGNS), action,
                               atol=1e-3)


def test_abort_when_off_the_square():
    state, tilt = read_state({**HOVERING, 'stateEstimate.x': 0.6})
    reason = square_abort_reason(state, tilt, 0.0, np.array([0.0, 0.0, 1.0]), SquareLimits())
    assert 'off the square' in reason
    state, tilt = read_state(HOVERING)
    assert square_abort_reason(state, tilt, 0.0, np.array([0.1, 0.0, 1.0]), SquareLimits()) is None


# ---------------------------------------------------------------------- flights
def test_policy_flight_unlocks_takes_off_flies_and_lands():
    reason, commander, supervisor = run()
    assert reason == 'square done'
    assert commander[0] == ('send_setpoint', (0, 0, 0, 0))
    assert names(supervisor) == ['send_arming_request', 'send_arming_request']
    flown = [args for name, args in commander[1:] if name == 'send_setpoint']
    assert 29 <= len(flown) <= 31   # 0.6 s at 50 Hz, give or take the float clock's last tick
    assert names(commander)[-2:] == ['send_stop_setpoint', 'send_notify_setpoint_stop']


def test_firmware_mode_flies_position_setpoints_along_the_square():
    reason, commander, _ = run(firmware=True)
    assert reason == 'square done'
    points = [args for name, args in commander if name == 'send_position_setpoint']
    assert points and points[0][:3] == pytest.approx((0.0, 0.0, 1.0))
    assert points[-1][0] > 0.2 and abs(points[-1][1]) < 1e-9   # first edge: straight ahead
    assert not [args for name, args in commander[1:] if name == 'send_setpoint']


def test_falling_behind_the_square_hands_back_and_lands():
    # The square starts where the drone is; a drone that never moves is left 0.5 m behind in
    # under a second.
    reason, commander, _ = run(laps=0.5)
    assert 'off the square' in reason
    assert 'send_stop_setpoint' in names(commander)


def test_the_log_holds_what_was_applied(tmp_path):
    path = tmp_path / 'flight.csv'
    record, close = square_recorder(path)
    run(policy=constant_policy([0.4, 0.0, 0.0, 0.2]), record=record, authority=0.5)
    close()
    lines = path.read_text().splitlines()
    assert lines[0] == LOG_FORMAT
    rows = list(csv.DictReader(lines[1:]))
    assert tuple(rows[0]) == COLUMNS and 29 <= len(rows) <= 31
    assert float(rows[0]['a_roll']) == pytest.approx(0.2)   # the policy's 0.4 at half authority
    times = [float(r['time']) for r in rows]
    assert times == sorted(times)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_fly_square.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'drones.missions.fly_square'`

- [ ] **Step 3: Refactor `src/drones/missions/fly_policy.py`**

- `SensorLog.__init__(self, scf, now=time.monotonic)` becomes `def __init__(self, scf, now=time.monotonic, blocks=None):`, and its loop becomes `for name, variables in (blocks or LOG_BLOCKS).items():`. Add a line to its docstring: `blocks` defaults to the hover policy's LOG_BLOCKS.
- Add this function just above `def fly(`:

```python
def take_off(commander, log, options, freq, now, sleep):
    """Climb to `options.height` on the firmware's hover controller, then measure the thrust command
    that holds this drone up there: the median over the settled second half of the hover."""
    steps = options.takeoff_seconds * freq
    for i in ticks(options.takeoff_seconds, freq, now, sleep):
        commander.send_hover_setpoint(0, 0, 0, options.height * min(1.0, (i + 1) / steps))
    samples = []
    for _ in ticks(options.calibrate_seconds, freq, now, sleep):
        commander.send_hover_setpoint(0, 0, 0, options.height)
        samples.append(float(log.latest.get('controller.cmd_thrust', 0.0)))
    return float(np.median(samples[len(samples) // 2:]))
```

- In `fly`, replace everything from `steps = options.takeoff_seconds * freq` through `hover_command = float(np.median(samples[len(samples) // 2:]))  # second half: settled` with:

```python
        hover_command = take_off(commander, log, options, freq, now, sleep)
```

- Rename `def _land(` to `def land(`, and its call in `fly`'s `finally` to `land(commander, log, options, freq, now, sleep)`.

Run: `uv run pytest tests/test_fly_policy.py -v`
Expected: all PASS, with behaviour unchanged.

- [ ] **Step 4: Create `src/drones/missions/fly_square.py`**

```python
"""Fly an exported square policy on the real Crazyflie, and log flights for system identification.

    uv run drones-fly-square runs/<name>/policy --dry-run            # motors off: estimate and policy live
    uv run drones-fly-square runs/<name>/policy --firmware           # the firmware flies the square; log it
    uv run drones-fly-square runs/<name>/policy --side 0.5 --authority 0.3   # first policy flights
    uv run drones-fly-square runs/<name>/policy

Take-off, hover calibration and landing are drones-fly-policy's: the firmware climbs and hovers on its
Flow-deck controller, and the thrust command that holds this drone up is measured. Then the square
starts where the drone hovers: the first edge straight ahead, turning left (--clockwise turns right).
The policy flies it through the attitude commander, from the firmware's state estimate. With
--firmware, the firmware's own position controller flies the same reference instead, so system-ID
data can be logged before any policy is trusted with the motors.

Every flight is logged to runs/<name>/flights/<stamp>-square*.csv, the format drones-finetune-square
reads. Rows hold the action the drone acted on: the policy's scaled by --authority, or the firmware
controller's command converted to the policy's units.

Needs only numpy and cflib.
"""
import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drones.missions.fly_policy import (DEFAULT_SIGNS, RANGERS, THRUST_COMMAND_MAX,
                                        THRUST_COMMAND_MIN, SensorLog, gravity_and_tilt, land,
                                        take_off, ticks, to_setpoint)
from drones.policy.runtime import Policy, SquareRunner
from drones.policy.square import heading

logger = logging.getLogger(__name__)

LOG_FORMAT = '# format: square-log v1'
COLUMNS = ('time', 'phase', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'qx', 'qy', 'qz', 'qw',
           'gyro_x', 'gyro_y', 'gyro_z', 'ref_x', 'ref_y', 'ref_z', 'ref_vx', 'ref_vy', 'ref_vz',
           'a_roll', 'a_pitch', 'a_yaw', 'a_thrust', 'roll_deg', 'pitch_deg', 'yaw_rate_deg',
           'thrust_cmd', 'hover_command')

# Log blocks, each within cflib's 26-byte packet.
LOG_BLOCKS = {
    'square_pos_vel': [(f'stateEstimate.{a}', 'float') for a in ('x', 'y', 'z', 'vx', 'vy', 'vz')],
    'square_attitude': [(f'stateEstimate.q{a}', 'float') for a in 'xyzw'],
    'square_gyro_thrust': [('gyro.x', 'float'), ('gyro.y', 'float'), ('gyro.z', 'float'),
                           ('controller.cmd_thrust', 'float')],
    # The firmware controller's attitude and yaw-rate command, logged for --firmware flights.
    'square_controller': [('controller.roll', 'float'), ('controller.pitch', 'float'),
                          ('controller.yawRate', 'float')],
    'square_ranges': [('range.zrange', 'uint16_t')] + [(f'range.{r}', 'uint16_t')
                                                       for r in RANGERS],
}


@dataclass(frozen=True)
class SquareLimits:
    max_tilt_deg: float = 30.0
    min_range: float = 0.2               # any Multi-ranger reading closer than this aborts
    min_height: float = 0.15
    max_height_above_target: float = 0.5
    max_error: float = 0.5               # m from the reference
    max_log_age: float = 0.25            # seconds without fresh sensor data aborts


@dataclass(frozen=True)
class SquareOptions:
    height: float = 1.0
    side: float | None = None            # default: the side the policy trained on
    lap_time: float = 8.0
    laps: float = 2.0
    clockwise: bool = False
    authority: float = 1.0
    firmware: bool = False
    dry_run: bool = False
    signs: dict = field(default_factory=lambda: dict(DEFAULT_SIGNS))
    limits: SquareLimits = field(default_factory=SquareLimits)
    takeoff_seconds: float = 2.0
    calibrate_seconds: float = 3.0

    @property
    def duration(self):
        return self.laps * self.lap_time


# ---------------------------------------------------------------------- pure conversions
def read_state(latest):
    """Latest cflib log values -> (state estimate in the simulator's frame and units, tilt in deg).

    The firmware's world frame is the simulator's: +x forward at take-off, +y left, +z up. The gyro
    is in the body frame, as CrazyFlow keeps angular velocity.
    """
    quat = np.array([latest[f'stateEstimate.q{a}'] for a in 'xyzw'], float)
    gravity, tilt = gravity_and_tilt(quat)
    return {
        'pos': np.array([latest[f'stateEstimate.{a}'] for a in 'xyz'], float),
        'vel': np.array([latest[f'stateEstimate.v{a}'] for a in 'xyz'], float),
        'quat': quat,
        'yaw': float(heading(np, quat)),
        'gravity': gravity,
        'gyro': np.radians([latest['gyro.x'], latest['gyro.y'], latest['gyro.z']]),
        'zrange': latest['range.zrange'] / 1000.0,
        'ranges': np.array([latest[f'range.{r}'] / 1000.0 for r in RANGERS]),
    }, tilt


def firmware_action(latest, spec, hover_command, signs=DEFAULT_SIGNS):
    """The firmware controller's command as the policy's normalised action: to_setpoint inverted.

    This assumes the logged controller.roll/pitch/yawRate follow the setpoint conventions. Check it
    on the first --firmware flight: the logged a_pitch should be positive while the drone
    accelerates forward.
    """
    roll = math.radians(signs['roll'] * latest['controller.roll'])
    pitch = math.radians(signs['pitch'] * latest['controller.pitch'])
    yaw_rate = math.radians(signs['yaw_rate'] * latest['controller.yawRate'])
    thrust = spec.hover_thrust * latest['controller.cmd_thrust'] / hover_command
    if thrust >= spec.hover_thrust:
        a_thrust = (thrust - spec.hover_thrust) / (spec.thrust_max - spec.hover_thrust)
    else:
        a_thrust = (thrust - spec.hover_thrust) / (spec.hover_thrust - spec.thrust_min)
    return np.clip([roll / spec.max_tilt, pitch / spec.max_tilt, yaw_rate / spec.max_yaw_rate,
                    a_thrust], -1.0, 1.0)


def square_abort_reason(state, tilt, log_age, ref_pos, limits):
    """Why the square must be handed back to the firmware right now, or None."""
    if log_age > limits.max_log_age:
        return f'sensor data {log_age:.2f} s old'
    if tilt > limits.max_tilt_deg:
        return f'tilt {tilt:.0f} deg'
    nearest = float(np.min(state['ranges']))
    if nearest < limits.min_range:
        return f'obstacle {nearest:.2f} m away'
    height = state['zrange']
    if height < limits.min_height or height > ref_pos[2] + limits.max_height_above_target:
        return f'height {height:.2f} m'
    error = float(np.linalg.norm(state['pos'] - ref_pos))
    if error > limits.max_error:
        return f'{error:.2f} m off the square'
    return None


def square_recorder(path):
    """A CSV writer for square flights; returns (record function, close function)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'w', newline='')
    f.write(LOG_FORMAT + '\n')
    writer = csv.writer(f)
    writer.writerow(COLUMNS)

    def record(time, phase, state, ref_pos, ref_vel, action, setpoint, hover_command):
        writer.writerow([round(time, 4), phase, *state['pos'], *state['vel'], *state['quat'],
                         *state['gyro'], *ref_pos, *ref_vel, *action,
                         *(setpoint or ('', '', '', '')), hover_command])

    return record, f.close


def start_square(policy, latest, options):
    """A runner whose square starts where the drone hovers now, first edge straight ahead."""
    state, _ = read_state(latest)
    origin = np.array([state['pos'][0], state['pos'][1], options.height])
    return SquareRunner(policy, origin=origin, ref_yaw=state['yaw'], lap_time=options.lap_time,
                        direction=-1.0 if options.clockwise else 1.0, rotation=state['yaw'],
                        side=options.side)


# ---------------------------------------------------------------------- the drone
def fly_square(scf, policy, log, options, record=None, now=time.monotonic, sleep=time.sleep):
    """Take off on the firmware, fly the square, land on the firmware.

    Returns why the square ended. Landing runs in a `finally`, so it happens whatever goes wrong
    after arming, exceptions included.
    """
    if options.dry_run:
        return _dry_run(policy, log, options, record, now, sleep)
    spec, freq = policy.spec, policy.spec.control_freq
    cf = scf.cf
    commander = cf.commander
    commander.send_setpoint(0, 0, 0, 0)  # zero thrust releases the legacy commander's thrust lock
    cf.supervisor.send_arming_request(True)
    sleep(1.0)
    try:
        hover_command = take_off(commander, log, options, freq, now, sleep)
        if not THRUST_COMMAND_MIN < hover_command < THRUST_COMMAND_MAX:
            return f'implausible hover thrust command {hover_command:.0f}'
        logger.info('Hover thrust command %.0f', hover_command)

        runner = start_square(policy, log.latest, options)
        phase = 'firmware' if options.firmware else 'policy'
        start, reason = now(), None
        try:
            for i in ticks(options.duration, freq, now, sleep):
                t = i / freq
                state, tilt = read_state(log.latest)
                ref_pos, ref_vel = runner.reference(t)
                reason = square_abort_reason(state, tilt, log.age(), ref_pos, options.limits)
                if reason:
                    break
                if options.firmware:
                    commander.send_position_setpoint(*ref_pos, math.degrees(runner.ref_yaw))
                    applied = firmware_action(log.latest, spec, hover_command, options.signs)
                    setpoint = None
                else:
                    action = runner.step(t, pos=state['pos'], vel=state['vel'], yaw=state['yaw'],
                                         gravity=state['gravity'])
                    applied = np.clip(action, -1.0, 1.0) * options.authority
                    setpoint = to_setpoint(action, spec, hover_command, options.signs,
                                           options.authority)
                    commander.send_setpoint(*setpoint)
                if record:
                    record(now() - start, phase, state, ref_pos, ref_vel, applied, setpoint,
                           hover_command)
        except KeyboardInterrupt:
            reason = 'interrupted'
        return reason or 'square done'
    finally:
        land(commander, log, options, freq, now, sleep)
        cf.supervisor.send_arming_request(False)


def _dry_run(policy, log, options, record, now, sleep):
    """Everything but the motors: read the estimate, run the policy, print what it would command."""
    spec, freq = policy.spec, policy.spec.control_freq
    runner = start_square(policy, log.latest, options)
    start = now()
    for i in ticks(options.duration, freq, now, sleep):
        t = i / freq
        state, tilt = read_state(log.latest)
        ref_pos, ref_vel = runner.reference(t)
        action = runner.step(t, pos=state['pos'], vel=state['vel'], yaw=state['yaw'],
                             gravity=state['gravity'])
        applied = np.clip(action, -1.0, 1.0) * options.authority
        if record:
            record(now() - start, 'dry-run', state, ref_pos, ref_vel, applied, None, float('nan'))
        if i % max(1, freq // 2) == 0:
            reason = square_abort_reason(state, tilt, log.age(), ref_pos, options.limits)
            roll, pitch, yaw_rate, thrust = spec.decode(applied)
            pos = state['pos']
            print(f'pos {pos[0]:+5.2f} {pos[1]:+5.2f} {pos[2]:4.2f} m | '
                  f'ref {ref_pos[0]:+5.2f} {ref_pos[1]:+5.2f} | tilt {tilt:4.1f} | '
                  f'-> roll {math.degrees(roll):+5.1f} pitch {math.degrees(pitch):+5.1f} '
                  f'yaw {math.degrees(yaw_rate):+6.1f}/s '
                  f'thrust {thrust / spec.hover_thrust:4.2f}x hover'
                  + (f' | would abort: {reason}' if reason else ''), flush=True)
    return 'dry run'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('artifact', type=Path, help='policy directory, e.g. runs/<name>/policy')
    parser.add_argument('--dry-run', action='store_true',
                        help='read the estimate and run the policy with the motors off')
    parser.add_argument('--firmware', action='store_true',
                        help='the firmware position controller flies the square; log it')
    parser.add_argument('--height', type=float, default=1.0, help='flight height, m')
    parser.add_argument('--side', type=float, help='side of the square, m (default: as trained)')
    parser.add_argument('--lap-time', type=float, default=8.0, help='seconds per lap')
    parser.add_argument('--laps', type=float, default=2.0)
    parser.add_argument('--clockwise', action='store_true')
    parser.add_argument('--authority', type=float, default=1.0,
                        help='0..1: scale the policy towards plain hover for first flights')
    for axis in DEFAULT_SIGNS:
        parser.add_argument(f'--{axis.replace("_", "-")}-sign', type=float, choices=(-1.0, 1.0),
                            default=DEFAULT_SIGNS[axis])
    parser.add_argument('--yes', action='store_true', help='skip the confirmation before arming')
    args = parser.parse_args(argv)

    try:
        policy = Policy.load(args.artifact)
    except (OSError, ValueError) as exc:
        sys.exit(f'Cannot load {args.artifact}: {exc}')
    spec = policy.spec
    if spec.task != 'square':
        sys.exit(f'{args.artifact} is a {spec.task} policy; drones-fly-square flies square ones')
    for name, value, (low, high) in (('--height', args.height, spec.height),
                                     ('--lap-time', args.lap_time, spec.lap_time)):
        if not low <= value <= high:
            sys.exit(f'{name} {value} is outside the {low}-{high} the policy trained on')
    if args.side is not None and not 2 * spec.corner_radius < args.side <= spec.side:
        sys.exit(f'--side must be over {2 * spec.corner_radius} m and at most {spec.side} m')
    if not 0.0 <= args.authority <= 1.0:
        sys.exit('--authority must be between 0 and 1')
    signs = {axis: getattr(args, f'{axis}_sign') for axis in DEFAULT_SIGNS}
    options = SquareOptions(height=args.height, side=args.side, lap_time=args.lap_time,
                            laps=args.laps, clockwise=args.clockwise, authority=args.authority,
                            firmware=args.firmware, dry_run=args.dry_run, signs=signs)

    import cflib.crtp

    from drones import config
    from drones.crazyflie.link import check_decks, open_link, resolve_uri

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('cflib').setLevel(logging.ERROR)
    cflib.crtp.init_drivers()
    stamp = (time.strftime('%Y%m%d-%H%M%S') + '-square' + ('-firmware' if args.firmware else '')
             + ('-dry' if args.dry_run else ''))
    log_path = args.artifact.parent / 'flights' / f'{stamp}.csv'

    uri = resolve_uri(config.URI)
    who = 'the firmware position controller' if args.firmware else \
        f'the policy at {args.authority:.0%} authority'
    print(f'Connecting to {uri} | {options.laps:g} laps of {options.lap_time:g} s')
    with open_link(uri) as scf:
        check_decks(scf)
        with SensorLog(scf, blocks=LOG_BLOCKS) as log:
            log.wait()
            if not args.dry_run and not args.yes:
                answer = input(f'Motors will spin: take off to {args.height} m and fly the square '
                               f'with {who}. Type "fly": ')
                if answer.strip() != 'fly':
                    sys.exit('Not flying.')
            record, close = square_recorder(log_path)
            try:
                reason = fly_square(scf, policy, log, options, record)
            finally:
                close()
    print(f'Square ended: {reason}. Flight log: {log_path}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Register the script**

In `pyproject.toml` `[project.scripts]`, after `drones-fly-policy`, add `drones-fly-square = "drones.missions.fly_square:main"`. Then run `uv sync --extra sim`.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_fly_square.py tests/test_fly_policy.py tests/test_architecture.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/drones/missions/fly_policy.py src/drones/missions/fly_square.py pyproject.toml \
        tests/test_fly_square.py
git commit -m "feat: drones-fly-square, with firmware-flown logging for system ID"
```

---

### Task 9: System identification

**Files:**
- Create: `src/drones/rl/sysid.py`
- Test: `tests/test_sysid.py`

**Interfaces:**
- Consumes:
  - `LOG_FORMAT` and `square_recorder` (Task 8)
  - `SquareConfig`, and `SquareEnv` with `advance`, `with_states` and `reference_at` (Task 4)
  - `init_residual` (Task 3)
  - `yaw_from_quat` (`sim/geometry.py`)
- Produces:
  - `Segment`, `load_flight(path, control_freq=50) -> [Segment]` and `load_flights(paths, control_freq=50) -> [Segment]`
  - `split_holdout(segments, fraction) -> (train, test)`
  - `Windows`, a flax struct with `.take(index)` and `len()`, and `make_windows(segments, horizon, max_latency) -> Windows`
  - `window_errors(predicted, windows)`
  - `SysIdConfig`
  - `SysIdResult(thrust_gain, latency, residual, report)` with an `.improved` property
  - `identify(segments, env_config=None, config=SysIdConfig(), log=print) -> SysIdResult`
  - `report` has the keys `uncorrected`, `gain_and_latency`, `full`, `thrust_gain`, `latency`, `horizon` and `note`. Each model entry has `one_step`, `horizon_step`, `score_one_step` and `score_horizon`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --extra sim pytest tests/test_sysid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'drones.rl.sysid'`

- [ ] **Step 3: Implement `src/drones/rl/sysid.py`**

```python
"""Fit CrazyFlow to real flights: a thrust gain, the action latency, and a residual wrench.

Flights come from drones-fly-square's CSV logs, from the policy or firmware phase. They are cut into
windows of `horizon` control steps. The model starts each window from its logged state, replays the
logged actions through the simulator, and is scored on how far it drifts from the logged states.
Three models are compared on held-out flights:
    uncorrected       the simulator as it is
    gain_and_latency  a fitted thrust gain, at the best-fitting latency
    full              the same plus the residual network (drones.sim.residual)

so_rpy's lift is cmd_f_coef * thrust / mass with no offset, so mass cannot be told apart from the
thrust gain; the gain carries both. The logged states are the firmware's Kalman estimate, not ground
truth, so the fit matches the simulator to what the drone believed, estimator bias included.
"""
import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import crazyflow  # noqa: F401  Must precede scipy, see drones.sim.
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from drones.missions.fly_square import LOG_FORMAT
from drones.sim.geometry import yaw_from_quat
from drones.sim.residual import init_residual
from drones.sim.square_env import SquareConfig, SquareEnv

FLIGHT_PHASES = ('policy', 'firmware')
# Error scales: a window drifting this far scores 1 in each term.
POS_SCALE, VEL_SCALE, ANGLE_SCALE = 0.05, 0.1, 0.05


@dataclass(frozen=True)
class Segment:
    """A stretch of flight logged without gaps, one row per control step."""
    flight: str
    pos: np.ndarray       # (T, 3) world
    vel: np.ndarray       # (T, 3) world
    quat: np.ndarray      # (T, 4) scalar-last
    ang_vel: np.ndarray   # (T, 3) body rad/s
    action: np.ndarray    # (T, 4) the normalised action the drone acted on


def load_flight(path, control_freq=50):
    """The segments of one drones-fly-square log, split wherever a control step is missing."""
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines or lines[0] != LOG_FORMAT:
        raise ValueError(f'{path}: not a square flight log (expected {LOG_FORMAT!r} first)')
    rows = [r for r in csv.DictReader(lines[1:]) if r['phase'] in FLIGHT_PHASES]
    groups, current, last = [], [], None
    for row in rows:
        t = float(row['time'])
        if current and t - last > 1.5 / control_freq:
            groups.append(current)
            current = []
        current.append(row)
        last = t
    if current:
        groups.append(current)
    return [_segment(path.stem, group) for group in groups]


def _segment(flight, rows):
    def columns(*names):
        return np.array([[float(r[n]) for n in names] for r in rows])

    return Segment(flight, columns('x', 'y', 'z'), columns('vx', 'vy', 'vz'),
                   columns('qx', 'qy', 'qz', 'qw'), columns('gyro_x', 'gyro_y', 'gyro_z'),
                   columns('a_roll', 'a_pitch', 'a_yaw', 'a_thrust'))


def load_flights(paths, control_freq=50):
    segments = [s for path in paths for s in load_flight(path, control_freq)]
    if not segments:
        raise ValueError('no policy or firmware phase in these flights')
    return segments


def _slice(segment, start, stop):
    return dataclasses.replace(segment, **{name: getattr(segment, name)[start:stop]
                                           for name in ('pos', 'vel', 'quat', 'ang_vel', 'action')})


def split_holdout(segments, fraction):
    """Hold out whole flights, the last `fraction` of them and at least one. With a single flight,
    hold out the last `fraction` of each of its segments instead."""
    flights = sorted({s.flight for s in segments})
    if len(flights) > 1:
        held = set(flights[-max(1, round(fraction * len(flights))):])
        return ([s for s in segments if s.flight not in held],
                [s for s in segments if s.flight in held])
    train, test = [], []
    for s in segments:
        cut = int(round(len(s.pos) * (1 - fraction)))
        train.append(_slice(s, 0, cut))
        test.append(_slice(s, cut, len(s.pos)))
    return train, test


@struct.dataclass
class Windows:
    pos: np.ndarray       # (W, K + 1, 3) logged states; index 0 is where each replay starts
    vel: np.ndarray
    quat: np.ndarray
    ang_vel: np.ndarray
    action: np.ndarray    # (W, K + max latency, 4); index max_latency + j was commanded at state j

    def __len__(self):
        return self.pos.shape[0]

    def take(self, index):
        return jax.tree.map(lambda x: x[index], self)


def make_windows(segments, horizon, max_latency):
    """Every window of `horizon` steps that has `max_latency` earlier actions to replay."""
    pieces = {name: [] for name in ('pos', 'vel', 'quat', 'ang_vel', 'action')}
    for seg in segments:
        for s in range(max_latency, len(seg.pos) - horizon):
            for name in ('pos', 'vel', 'quat', 'ang_vel'):
                pieces[name].append(getattr(seg, name)[s:s + horizon + 1])
            pieces['action'].append(seg.action[s - max_latency:s + horizon])
    if not pieces['pos']:
        raise ValueError(f'no flight segment is longer than {horizon + max_latency} control steps')
    return Windows(**{k: np.stack(v).astype(np.float32) for k, v in pieces.items()})


def window_errors(predicted, windows):
    """Scaled squared error at every step of every window, (W, K)."""
    pos, vel, quat = predicted
    dp = jnp.sum(jnp.square(pos - windows.pos[:, 1:]), -1) / POS_SCALE ** 2
    dv = jnp.sum(jnp.square(vel - windows.vel[:, 1:]), -1) / VEL_SCALE ** 2
    # 4 (1 - <q, q'>^2) is the squared angle between them near zero, and smooth at zero.
    dot = jnp.sum(quat * windows.quat[:, 1:], -1)
    da = 4.0 * (1.0 - jnp.square(dot)) / ANGLE_SCALE ** 2
    return dp + dv + da


@dataclass(frozen=True)
class SysIdConfig:
    horizon: int = 10
    latencies: tuple[int, ...] = (0, 1, 2)
    gain_steps: int = 500          # thrust gain alone, per latency candidate
    residual_steps: int = 3000     # gain and residual together, at the chosen latency
    batch: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    holdout: float = 0.2
    seed: int = 0


@dataclass
class SysIdResult:
    thrust_gain: float
    latency: int
    residual: dict
    report: dict

    @property
    def improved(self):
        """Whether the corrected simulator beats the uncorrected one on held-out flights."""
        r = self.report
        return r['full']['score_horizon'] < r['uncorrected']['score_horizon']


class Replay:
    """Replays logged actions through the simulator, corrected by a model, from logged states."""

    def __init__(self, env_config, batch):
        self.env = SquareEnv(dataclasses.replace(env_config, num_envs=batch))
        self.batch = batch
        self._rollout = jax.jit(self.rollout, static_argnums=1)

    def rollout(self, model, latency, windows):
        """Predicted (pos, vel, quat), each (batch, K, ...), for exactly `batch` windows."""
        env = self.env
        horizon = windows.pos.shape[1] - 1
        first = windows.action.shape[1] - horizon      # the max latency the windows allow for
        sim = env.with_states(env.sim.default_data, windows.pos[:, 0], windows.vel[:, 0],
                              windows.quat[:, 0], windows.ang_vel[:, 0])
        gain = jnp.exp(model['log_gain']) * jnp.ones(self.batch)

        def body(carry, j):
            sim, yaw_cmd = carry
            action = jax.lax.dynamic_index_in_dim(windows.action, first + j - latency, 1,
                                                  keepdims=False)
            sim, yaw_cmd = env.advance(sim, action, yaw_cmd, gain, model['residual'])
            s = sim.states
            return (sim, yaw_cmd), (s.pos[:, 0], s.vel[:, 0], s.quat[:, 0])

        _, predicted = jax.lax.scan(body, (sim, yaw_from_quat(windows.quat[:, 0])),
                                    jnp.arange(horizon))
        return tuple(x.swapaxes(0, 1) for x in predicted)

    def fit(self, model, latency, windows, trainable, steps, config, log):
        """Adam on the mean window error, training only the entries of `model` named in
        `trainable`."""
        frozen = {k: v for k, v in model.items() if k not in trainable}
        params = {k: v for k, v in model.items() if k in trainable}
        optimizer = optax.adam(config.learning_rate)

        def loss(params, batch):
            value = jnp.mean(window_errors(self.rollout({**frozen, **params}, latency, batch),
                                           batch))
            if 'residual' in params:
                value = value + config.weight_decay * sum(
                    jnp.sum(jnp.square(x)) for x in jax.tree.leaves(params['residual']))
            return value

        @jax.jit
        def update(params, opt_state, batch):
            value, grads = jax.value_and_grad(loss)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, value

        rng = np.random.default_rng(config.seed)
        opt_state = optimizer.init(params)
        for step in range(steps):
            batch = windows.take(rng.integers(0, len(windows), self.batch))
            params, opt_state, value = update(params, opt_state, batch)
            if step % 100 == 0 or step == steps - 1:
                log(f'  latency {latency} | fitting {" + ".join(trainable)} | step {step:4d} | '
                    f'loss {float(value):.4f}')
        return {**frozen, **params}

    def evaluate(self, model, latency, windows):
        """Errors after one step and after the whole window, averaged over every window."""
        n = len(windows)
        chunks = []
        for start in range(0, n, self.batch):
            # The last chunk wraps round to fill the batch; the extra rows are trimmed below.
            batch = windows.take(np.arange(start, start + self.batch) % n)
            chunks.append(jax.device_get(self._rollout(model, latency, batch)))
        pos, vel, quat = (np.concatenate([c[i] for c in chunks])[:n] for i in range(3))
        pos_err = np.linalg.norm(pos - windows.pos[:, 1:], axis=-1)
        vel_err = np.linalg.norm(vel - windows.vel[:, 1:], axis=-1)
        dot = np.abs(np.sum(quat * windows.quat[:, 1:], -1))
        angle = 2 * np.arccos(np.clip(dot, 0.0, 1.0))
        score = np.asarray(window_errors((pos, vel, quat), windows))

        def at(i):
            return {'pos_m': float(pos_err[:, i].mean()), 'vel_m_s': float(vel_err[:, i].mean()),
                    'angle_rad': float(angle[:, i].mean())}

        return {'one_step': at(0), 'horizon_step': at(-1),
                'score_one_step': float(score[:, 0].mean()),
                'score_horizon': float(score[:, -1].mean())}


def identify(segments, env_config=None, config=SysIdConfig(), log=print):
    """Fit the thrust gain, latency and residual to `segments`, and report held-out errors."""
    env_config = env_config or SquareConfig()
    train_segments, test_segments = split_holdout(segments, config.holdout)
    max_latency = max(config.latencies)
    train = make_windows(train_segments, config.horizon, max_latency)
    test = make_windows(test_segments, config.horizon, max_latency)
    log(f'System ID: {len(train)} training windows, {len(test)} held out')
    replay = Replay(env_config, config.batch)
    start = {'log_gain': jnp.zeros(()), 'residual': init_residual(jax.random.key(config.seed))}

    report = {'uncorrected': replay.evaluate(start, 0, test)}
    candidates = {}
    for latency in config.latencies:
        model = replay.fit(start, latency, train, ('log_gain',), config.gain_steps, config, log)
        candidates[latency] = model, replay.evaluate(model, latency, test)
    latency = min(candidates, key=lambda k: candidates[k][1]['score_horizon'])
    gain_model, report['gain_and_latency'] = candidates[latency]
    full = replay.fit(gain_model, latency, train, ('log_gain', 'residual'),
                      config.residual_steps, config, log)
    report['full'] = replay.evaluate(full, latency, test)

    gain = float(jnp.exp(full['log_gain']))
    report.update(thrust_gain=gain, latency=latency, horizon=config.horizon,
                  note='fitted to the firmware state estimate, not ground truth')
    return SysIdResult(thrust_gain=gain, latency=latency, residual=full['residual'],
                       report=report)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run --extra sim pytest tests/test_sysid.py -v`
Expected: all PASS. The fit test takes a few minutes: four fits, each compiling a 10-step backward pass.

- [ ] **Step 5: Commit**

```bash
git add src/drones/rl/sysid.py tests/test_sysid.py
git commit -m "feat: system identification of thrust gain, latency and residual from square logs"
```

---

### Task 10: `drones-finetune-square`

**Files:**
- Create: `src/drones/rl/finetune_square.py`
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `tests/test_finetune_square.py`

**Interfaces:**
- Consumes:
  - `load_flights`, `identify`, `SysIdResult` (Task 9)
  - `load_square_run`, `evaluate_square`, `print_table` (Task 7); `train`, `write_config` (Task 7)
  - `SHAC` (Task 5), `SquareEnv` (Task 4), `export_square_policy` (Task 6), `save_params`
- Produces: `finetuned_configs(env_config, shac_config, result, iterations, num_envs, device) -> (SquareConfig, SHACConfig)`, `main(argv)`, and the script `drones-finetune-square`

- [ ] **Step 1: Write the failing test**

```python
"""drones-finetune-square: how a fit reshapes the training setup."""
import pytest

pytest.importorskip('crazyflow')

from drones.rl.finetune_square import finetuned_configs
from drones.rl.shac import SHACConfig
from drones.rl.sysid import SysIdResult
from drones.sim.square_env import SquareConfig


def test_the_fit_centres_the_randomisation_and_slows_learning():
    result = SysIdResult(thrust_gain=0.87, latency=1, residual={}, report={})
    env, shac = finetuned_configs(SquareConfig(num_envs=4096, device='gpu'), SHACConfig(),
                                  result, iterations=200, num_envs=256, device='cpu')
    assert env.thrust_gain == 0.87 and env.thrust_gain_range == pytest.approx(0.05)
    assert env.latency_steps == (1,)
    assert env.num_envs == 256 and env.device == 'cpu'
    assert shac.iterations == 200
    assert shac.actor_lr == pytest.approx(SHACConfig().actor_lr / 4)
    assert shac.critic_lr == pytest.approx(SHACConfig().critic_lr / 4)
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run --extra sim pytest tests/test_finetune_square.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'drones.rl.finetune_square'`

- [ ] **Step 3: Implement `src/drones/rl/finetune_square.py`**

```python
"""Correct the simulator with real square flights, then finetune a trained policy in it.

    uv run --extra sim drones-finetune-square runs/<name> --flights runs/<name>/flights/*-square*.csv

1. System identification (drones.rl.sysid) fits the thrust gain, latency and a residual wrench to the
   flights, and compares the result with the uncorrected simulator on held-out flights. If the
   corrected model is not better, it stops there.
2. SHAC continues from runs/<name> in the corrected simulator: the fitted thrust gain and latency,
   half the randomisation, the residual on, and learning rates at a quarter.
3. Everything goes to runs/<name>-ft/: sysid.json, residual.msgpack, and all a training run writes.
   The finetuned policy is then evaluated in the corrected and the uncorrected simulator.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path


def finetuned_configs(env_config, shac_config, result, iterations, num_envs, device):
    """The env and SHAC settings to finetune with, given a system-ID result."""
    env = dataclasses.replace(env_config, num_envs=num_envs, device=device,
                              thrust_gain=result.thrust_gain,
                              thrust_gain_range=env_config.thrust_gain_range / 2,
                              latency_steps=(result.latency,))
    shac = dataclasses.replace(shac_config, iterations=iterations,
                               actor_lr=shac_config.actor_lr / 4,
                               critic_lr=shac_config.critic_lr / 4)
    return env, shac


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path, help='run directory written by drones-train-square')
    parser.add_argument('--flights', type=Path, nargs='+', required=True,
                        help='square flight logs from drones-fly-square')
    parser.add_argument('--out', type=Path, help='run directory to write (default: RUN-ft)')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=50)
    args = parser.parse_args(argv)
    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
    except ImportError:
        sys.exit('Finetuning needs the sim extra:  uv sync --extra sim')
    import jax

    from drones.rl.evaluate_square import evaluate_square, load_square_run, print_table
    from drones.rl.export import export_square_policy
    from drones.rl.ppo import save_params
    from drones.rl.shac import SHAC
    from drones.rl.sysid import identify, load_flights
    from drones.rl.train_square import train, write_config
    from drones.sim.square_env import SquareEnv

    out = args.out or args.run.with_name(args.run.name + '-ft')
    out.mkdir(parents=True, exist_ok=True)
    with jax.default_device(jax.devices(args.device)[0]):
        base_env, base_agent, params = load_square_run(args.run, args.num_envs, args.device)
        try:
            segments = load_flights(args.flights)
        except (OSError, ValueError) as exc:
            sys.exit(f'Cannot read the flights: {exc}')
        result = identify(segments, base_env.config)
        (out / 'sysid.json').write_text(json.dumps(result.report, indent=2))
        print(f'Thrust gain {result.thrust_gain:.3f} | latency {result.latency} steps | '
              f'held-out error {result.report["uncorrected"]["score_horizon"]:.2f} -> '
              f'{result.report["full"]["score_horizon"]:.2f}')
        if not result.improved:
            sys.exit(f'The corrected simulator is no better than the uncorrected one on held-out '
                     f'flights; not finetuning. See {out / "sysid.json"}')
        save_params(out / 'residual.msgpack', result.residual)

        env_config, shac_config = finetuned_configs(base_env.config, base_agent.config, result,
                                                    args.iterations, args.num_envs, args.device)
        write_config(out, env_config, shac_config)
        env = SquareEnv(env_config, residual=result.residual)
        agent = SHAC(env, shac_config)
        state = agent.init(jax.random.key(args.seed))
        state = state.replace(actor=params['actor'], critic=params['critic'],
                              target=params['target'])
        state = train(agent, state, out, args.log_every, args.save_every)
        print(f'Policy artifact: {export_square_policy(env, state.actor, out / "policy")}')

        policy = jax.jit(lambda o: agent.act(state.actor, o))
        key = jax.random.key(args.seed + 1)
        print_table({'corrected sim': evaluate_square(env, policy, key),
                     'uncorrected sim': evaluate_square(base_env, policy, key)})


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Register the script**

In `pyproject.toml` `[project.scripts]`, after `drones-eval-square`, add `drones-finetune-square = "drones.rl.finetune_square:main"`. Then run `uv sync --extra sim`.

- [ ] **Step 5: Run the test**

Run: `uv run --extra sim pytest tests/test_finetune_square.py -v`
Expected: PASS.

- [ ] **Step 6: Try the whole pipeline once by hand**

This uses Task 9's synthetic flights, via a throwaway script in the scratchpad rather than the repo:

```bash
uv run --extra sim drones-train-square --preset cpu-test --name ft-check
uv run --extra sim python - <<'EOF'
import sys; sys.path.insert(0, 'tests')
from pathlib import Path
from test_sysid import TRUE_WORLD, synthesise_flight
from drones.sim.square_env import SquareEnv
env = SquareEnv(TRUE_WORLD)
for seed in range(3):
    synthesise_flight(env, Path(f'runs/ft-check/flights/synthetic{seed}-square.csv'), seed)
EOF
uv run --extra sim drones-finetune-square runs/ft-check --flights runs/ft-check/flights/*.csv \
    --iterations 5 --num-envs 16
```

Expected:
- a gain near 0.85 and a latency of 1
- `held-out error X -> Y` with Y well below X
- 5 logged iterations and a corrected/uncorrected table
- `runs/ft-check-ft/` containing `sysid.json`, `residual.msgpack`, `params.msgpack` and `policy/`

`runs/` is git-ignored, so nothing needs cleaning up.

- [ ] **Step 7: Commit**

```bash
git add src/drones/rl/finetune_square.py pyproject.toml tests/test_finetune_square.py
git commit -m "feat: drones-finetune-square, from real flights to a corrected simulator"
```

---

### Task 11: README and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the commands to the top table**

In the table under the introduction, after the `drones-fly-policy` row, add:

```markdown
| `uv run --extra sim drones-train-square [config.yaml]` | Train a policy to fly a 1 × 1 m square with SHAC |
| `uv run --extra sim drones-eval-square runs/<name>` | Evaluate a square policy: crash rate, laps, tracking error |
| `uv run drones-fly-square runs/<name>/policy` | Fly the square on the real drone, or let the firmware fly it and log (`--firmware`) |
| `uv run --extra sim drones-finetune-square runs/<name> --flights …` | Fit the simulator to real flights, then finetune the policy in it |
```

- [ ] **Step 2: Add the new files to the Layout tree**

- Under `missions/`, add: `fly_square.py       drones-fly-square: fly a square policy, or log the firmware flying one`
- Under `policy/`, add: `square.py         the square's reference path and observation, shared by sim and drone`
- Under `sim/`, add:
  - `square_env.py     the square task: differentiable, for SHAC`
  - `residual.py       a learned force and torque correcting the dynamics`
  - `calibration.py    hover-thrust calibration`
- Under `rl/`, add:
  - `shac.py           short-horizon actor-critic through the simulator`
  - `sysid.py          fit thrust gain, latency and residual to flight logs`
  - `train_square.py`, `evaluate_square.py`, `finetune_square.py`, each with its command name
- Add `configs/square/` next to `configs/hover/`.

- [ ] **Step 3: Add a "Flying a square" section before "## Development"**

```markdown
## Flying a square

A second task: fly a 1 × 1 m square, corners rounded to 0.15 m, at a steady speed and height. It is
trained with SHAC (short-horizon actor-critic, Xu et al. 2022). Instead of estimating gradients from
returns as PPO does, SHAC backpropagates 32-step windows through CrazyFlow's dynamics, with a
critic's value closing each window.

The square policy observes the firmware's state estimate: the Flow deck's Kalman filter's position,
velocity and attitude. The hover policy does not. The policy sees its error to the reference now and
a second ahead, all in its own heading frame.

```bash
uv run --extra sim drones-train-square --preset cpu-test     # quick check
uv run --extra sim drones-train-square --preset cpu          # a few minutes on a laptop
uv run --extra sim drones-eval-square runs/<name>
```

**The task** (`sim/square_env.py`, configured by `configs/square/shac.yaml`):
- Each episode samples:
  - a lap time of 6–10 s and a height of 0.8–1.2 m
  - a direction and orientation for the square
  - a starting point along it, and a start error of up to 0.1 m
- The estimate carries noise and a 1 cm/√s horizontal drift.
- Thrust gain (±10%) and action latency (0–2 control steps) are randomised: two of the sim-to-real
  gaps the hover task left open.
- The reward runs from about 1 to 2.5 per step, for staying up and tracking position and velocity,
  less small tilt, rate and jerk costs. A crash costs 10: below 0.1 m, tilted past 57°, or 1 m off
  the reference.
- `step` is differentiable end to end. Metrics are gradient-stopped, and restarted worlds carry no
  gradient from their last episode.

**On the drone** (`drones-fly-square`): take-off, hover calibration and landing are
`drones-fly-policy`'s. The square starts where the drone hovers, first edge straight ahead.
- `--side 0.5 --authority 0.3` for first flights.
- `--clockwise`, `--lap-time` and `--laps` shape the flight.
- It aborts on `drones-fly-policy`'s limits, and when the drone is more than 0.5 m off the square.
- `--firmware` lets the firmware's own position controller fly the same square. The attitude and
  thrust it commands are logged in the policy's action units, so system-ID data can be collected
  before a policy has flown. On the first `--firmware` flight, check that `a_pitch` is positive while
  the drone accelerates forward. The sign of `controller.pitch` in the firmware log comes from reading
  the source, not from flying.

Every flight is logged to `runs/<name>/flights/<stamp>-square*.csv`.

**From real flights back to the simulator** (`drones-finetune-square`):

```bash
uv run drones-fly-square runs/<name>/policy --firmware --laps 3   # a few of these
uv run --extra sim drones-finetune-square runs/<name> --flights runs/<name>/flights/*-square*.csv
```

1. It replays the logged actions from logged states through the simulator, and fits three things to
   10-step windows:
   - a thrust gain (mass and thrust gain act only as a ratio in the fitted model, so one number
     covers both)
   - the action latency
   - a small residual network producing a force and torque, applied through CrazyFlow's disturbance
     inputs
2. It reports held-out error for the uncorrected simulator, gain and latency alone, and the full
   correction, in `runs/<name>-ft/sysid.json`. If the correction does not beat the uncorrected
   simulator on held-out flights, it stops there.
3. Otherwise it continues SHAC in the corrected simulator: randomisation centred on the fit and
   halved, learning rates at a quarter. The finetuned policy is evaluated in both simulators.

The logged states are the firmware's estimate, not ground truth, so the correction matches the
simulator to what the drone believed, estimator drift included.
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run --extra sim pytest`
Expected: all tests PASS, apart from skips where no EGL context exists (the render tests). Report the counts.

- [ ] **Step 5: Run a CPU training long enough to judge learning, and report the numbers**

Run: `uv run --extra sim drones-train-square --preset cpu --name square-cpu`, then `uv run --extra sim drones-eval-square runs/square-cpu`.

Report the policy row against open-loop hover: crash rate, laps, position RMSE. Do not claim the square is learned unless the policy completes at least one lap on average with a crash rate under 10%. If it falls short, say so with the numbers.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: flying a square with SHAC, and finetuning from real flights"
```

