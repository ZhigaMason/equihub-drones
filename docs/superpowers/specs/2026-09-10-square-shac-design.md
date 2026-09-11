# Square flight with SHAC, finetuned from real flights

Teach the Crazyflie 2.1 Brushless to fly a 1 x 1 m square in simulation with SHAC (Short-Horizon
Actor-Critic, Xu et al. 2022), fly it on the real drone, and use the real flight logs to correct the
simulator and finetune the policy.

Built in three phases, each tested before the next:

1. Square task and SHAC, simulation only.
2. Flying the square on the drone, and logging for system identification.
3. System identification from the logs, and finetuning through the corrected model.

## Facts this design rests on

Checked in this repository before writing:

- Gradients flow through CrazyFlow's `so_rpy` step. A 32-step rollout's gradient with respect to
  the action matched central finite differences to three significant figures.
- `mjx.ray` gives zero gradient with respect to position. The square task casts no rays.
- `so_rpy`'s lift is `cmd_f_coef * thrust / mass`, with `acc_coef = 0` for `cf21B_500`. Mass and
  thrust gain therefore only ever act as their ratio, and flight data cannot tell them apart. The
  design randomises and identifies a single thrust gain.
- CrazyFlow keeps angular velocity in the body frame, as the gyro reports it.
- `so_rpy` adds `states.force` and `states.torque` as disturbances (`dist_f`, `dist_t`), and
  nothing in the attitude pipeline clears them. A force set once per control step is therefore held
  for that step's 10 physics substeps. This is how the learned residual enters the simulator.

## Phase 1: the square task and SHAC

### Shared contract: `src/drones/policy/square.py` (numpy-safe, takes `xp`)

Like `policy/interface.py`, this module is used by both the simulator and the drone, so both sides
compute the same reference and the same observation. It imports neither JAX nor cflib.

- `square_reference(xp, t, *, side, lap_time, corner_radius, direction, rotation, origin)` gives the
  reference position and velocity at time `t`.
  - The path is a 1 m square with rounded corners (`corner_radius` 0.15 m), so position and
    velocity are continuous and acceleration is bounded.
  - Speed along the path is constant, set by the path length over `lap_time`.
  - `origin` is the first corner. `direction` is +1 (ccw) or -1 (cw), and `rotation` is the yaw of
    the square.
- `encode_square_obs(xp, *, pos_est, vel_est, gravity, yaw_err, ref_pos, ref_vel, lookahead_pos,
  prev_action)` builds one observation:
  - position error to the reference now and at `LOOKAHEAD = 5` points spaced 0.2 s apart
  - velocity error to the reference velocity
  - estimated velocity
  - body-frame gravity direction and yaw error
  - the previous action
  Positions and velocities are rotated into the drone's yaw frame, so the policy is invariant to
  heading. Fixed scales, as in `interface.py`.
- Actions are the hover task's: `interface.decode_action`, reused unchanged.

### Environment: `src/drones/sim/square_env.py`

`SquareEnv` exposes `num_envs`, `action_size`, `policy_size`, `critic_size`, `reset(key)` and
`step(state, action)`, the same interface as `HoverEnv`. It adds one requirement: `step` is
differentiable in `state` and `action`.

- **Drone:** `cf21B_500`, `so_rpy`, attitude control, 500 Hz physics and 50 Hz policy.
  - The hover thrust is calibrated with `HoverEnv`'s secant method, moved into a shared helper.
  - The scene is CrazyFlow's default open floor. No room, no rays.
- **Per episode:**
  - height 0.8–1.2 m
  - `lap_time` 6–10 s
  - direction cw or ccw
  - `rotation` uniform in [-pi, pi)
  - starting phase uniform around the lap
  - start pose within 0.1 m of the reference, 0.2 m/s of velocity error, 0.1 rad of tilt
  - `origin` placed so the start is at the world origin
- **Domain randomisation, each configurable and able to be switched off:**
  - thrust gain ±10%, scaling the commanded thrust. This also stands in for mass, which only acts
    through the same ratio.
  - action latency of 0, 1 or 2 control steps, from a small action buffer in the env state
- **State estimate** (what the policy sees, modelling the Flow-deck Kalman filter):
  - position = true position + random-walk bias (0.01 m/√s) + white noise (0.01 m)
  - velocity = true velocity + white noise (0.03 m/s)
  - gravity and yaw from the true attitude with 0.01 noise
  - noise is drawn from the key in `EnvState`, treated as constant under differentiation
- **Critic observation:** the policy observation plus the true position error, true velocity,
  angular velocity, the sin and cos of the reference phase, the lap time and the thrust gain.
- **Reward:** smooth everywhere it is differentiated.

  ```
  e = |p - p_ref|,  ev = |v - v_ref|
  r = 1 + exp(-e^2 / 0.05^2) + 0.5 * exp(-ev^2 / 0.25^2)
        - e^2 - 0.1 * tilt^2 - 0.01 * |w|^2 - 0.05 * |a - a_prev|^2
  ```

  - The exponentials give sharp tracking near the reference.
  - The quadratic term keeps the gradient alive far from it. With a weight of 1, the reward is still
    about 0, not negative, at the 1 m crash limit.
  - `tilt^2` is computed as `2 * (1 - R_zz)`, which avoids arccos's infinite slope when level.
  - Metrics in `info` are gradient-stopped.
  - The constant 1 rewards staying up, so crashing is never attractive.
- **Termination:**
  - A crash ends the episode: height under 0.1 m, tilt over 1.0 rad, or position error over 1.0 m.
    A crash replaces the reward with `CRASH_REWARD = -10` and carries no gradient.
  - Otherwise the episode is truncated at 16 s.
  - `info` reports `crashed` and `truncated` separately, because SHAC bootstraps on truncation.
- **Resets:** worlds that finish are reset inside `step`, with a mask as `HoverEnv` does. The reset
  worlds' new state carries no gradient from the old one: `jnp.where` against a `stop_gradient`
  sample.
- **Residual hook:** `SquareEnv(config, residual=None)` takes the parameters of the network in
  `sim/residual.py`. Before each control step, the env sets `states.force` and `states.torque`
  (world frame) to that network's body-frame output, rotated into the world frame. This is Phase 3's
  correction, and it stays differentiable.

### SHAC: `src/drones/rl/shac.py`

- **Networks:**
  - actor: `networks.MLP` on `obs['policy']`, output scale 0.01, so it starts at hover
  - learned state-independent `log_std`, initialised at -1.0
  - critic: `networks.MLP` on `obs['critic']`
  - target critic: a Polyak copy, `target <- alpha * target + (1 - alpha) * critic` with
    `alpha = 0.2`, updated after each iteration's critic training. This is SHAC's published
    setting.
- **One jitted `iterate(ts)` per iteration:**
  1. Detach the environment state and observation carried from the last window
     (`jax.lax.stop_gradient`).
  2. Roll out `horizon = 32` steps with reparameterised actions `a = mu + sigma * eps`, in
     `jax.lax.scan` inside `jax.value_and_grad` of the actor loss.
  3. Compute the actor loss:

     ```
     L = -1 / (N * h) * sum_n [ sum_{t<T_n} gamma^t r_t  +  gamma^{T_n} * V_target(s_{T_n}) * bootstrap_n ]
     ```

     - `T_n` is the first episode end in the window, or `h`.
     - `bootstrap_n` is 0 after a crash and 1 after truncation or at the window end.
     - Rewards after an episode ends in the window count towards the new episode's sum, discounted
       from its own start. Implemented with a running discount that resets on `done`.
  4. Clip the actor gradient to global norm 1.0. If any gradient is not finite, skip the update
     and count it in the metrics.
  5. Compute TD(lambda) targets from the detached rewards, dones and target-critic values
     (`lambda = 0.95`, `gamma = 0.99`).
  6. Train the critic on them for `critic_epochs = 16` epochs of `critic_minibatches = 4`, using
     MSE loss. Then update the target critic.
- **Optimisers:** Adam with betas (0.7, 0.95), as in SHAC. Actor learning rate 2e-3 and critic
  5e-4, both linearly decayed to 0 over the run.
- **Checkpoints and outputs:** the same `TrainState` shape and `save_params`/`load_params` as
  `ppo.py`. Metrics per iteration: episode return, crash rate, tracking RMSE, actor gradient norm,
  skipped updates and critic loss.

### Configuration and commands

- `configs/square/shac.yaml` with sections `env` (`SquareConfig`) and `shac` (`SHACConfig`).
- `experiment.py`'s loader (`load`, `merge`, `override`, `_build`) is generalised to take a task.
  Hover keeps its sections and presets unchanged. Square gets its own sections and presets:
  - `cpu-test`: 32 worlds
  - `cpu`: 256 worlds
  - `gpu`: 4096 worlds
- `drones-train-square [config] [--preset] [--set] [--device] [--save-every]` writes
  `runs/<name>/`: `config.yaml`, `metrics.csv`, `params.msgpack` and `policy/`, as hover does.
- `drones-eval-square runs/<name>`: runs 256 fresh episodes with deterministic actions and reports
  crash rate, laps completed, position RMSE and maximum error. For scale, it compares against a
  zero-action open-loop baseline.

## Phase 2: flying the square, and logging

### Artifact

- `policy/runtime.py` gains `SquareSpec`: control frequency, action scaling, hover thrust, square
  parameters and the observation scales.
- `Policy` is generalised to any spec with an `observation_size`.
- `policy.json` gains `task: hover | square` and `FORMAT_VERSION` goes to 2. Version 1 artifacts
  still load, as hover.
- `SquareRunner` mirrors `PolicyRunner`: reference time plus state estimate in, action out.

### `src/drones/missions/fly_square.py` (`drones-fly-square`)

Follows `fly_policy.py`'s structure and safety pattern:

- confirmation before arming, and `--dry-run`
- `--authority` and the sign flags
- take off on the firmware and calibrate the hover command
- hand over at 50 Hz through `send_setpoint`
- land in a `finally`

The square's first corner is the take-off hover point, and `--side` shrinks it for first flights.

- **Abort limits:** `fly_policy`'s tilt, stale-data and height limits, plus position error to the
  reference over 0.5 m. Rangers under 0.2 m also abort when the Multi-ranger is present.
- **`--firmware` mode:** the firmware's own position controller flies the same reference through
  `send_position_setpoint`. This collects system-ID data before any policy is trusted with the
  motors. The "action" logged is the firmware's attitude and thrust command
  (`controller.roll/pitch/yawRate/cmd_thrust`), converted to normalised actions with the inverse of
  `to_setpoint`.
- **Log blocks at 100 Hz**, each within 26 bytes:
  - `stateEstimate.x/y/z`
  - `stateEstimate.vx/vy/vz`
  - `stateEstimate.qx..qw`
  - gyro and `controller.cmd_thrust`
  - `controller.roll/pitch/yawRate`
- **CSV per flight** at `runs/<name>/flights/<stamp>-square[-firmware].csv`, one row per control
  step, with columns:
  - time and phase
  - position, velocity and quaternion estimates
  - gyro
  - reference position and velocity
  - normalised action
  - setpoint sent
  - `thrust_cmd` and the hover command

  The header carries a `# format: square-log v1` comment line. `rl/sysid.py` reads exactly this.

## Phase 3: system identification and finetuning

### `src/drones/rl/sysid.py`

- **Loading:** turn each flight CSV into sequences of `(state, action)` at 50 Hz.
  - Only the `policy` or `firmware` phase is used.
  - Rows with stale data are dropped, and sequences are split at gaps.
  - The state is position, velocity, quaternion and angular velocity. Angular velocity is the gyro
    reading in rad/s. It is body frame, as CrazyFlow keeps it.
  - Actions are converted back through the logged hover command.
- **Model:** the CrazyFlow `so_rpy` step (one `Sim` with a world per training sequence), with:
  - a learned `log_thrust_gain`. Mass cannot be identified separately; see "Facts".
  - latency of 0, 1 or 2 steps, chosen by grid search as the best held-out error after fitting
    each
  - a residual MLP (2 x 64, tanh, output initialised to zero) from body-frame velocity, attitude
    (gravity direction), angular velocity and action to a body-frame force (3) and torque (3),
    applied through `states.force` and `states.torque` and rotated to world frame
- **Loss:** multi-step error.
  - Start from a logged state, roll the model forward `k = 10` steps under the logged actions, and
    compare against the logged states.
  - Position, velocity and attitude errors are normalised by fixed scales: 0.05 m, 0.1 m/s and
    0.05 rad.
  - Attitude error is `4 * (1 - <q_pred, q_log>^2)`: the squared angle near zero, without arccos's
    infinite slope.
  - The thrust gain is fitted first for each latency candidate, then gain and residual together at
    the chosen latency.
  - L2 weight decay on the residual (1e-4). Adam, 3000 steps.
- **Split:** whole flights are held out, 20% with at least one flight. With a single flight, the
  last 20% of it is held out.
- **Report:** `sysid.json` with the parameters, chosen latency, and one-step and 10-step held-out
  error for three models: the uncorrected simulator, parameters only, and parameters plus residual.
  If the full model does not beat the uncorrected simulator on held-out 10-step error, the command
  says so and exits non-zero without finetuning.

### `drones-finetune-square runs/<name> --flights <csv>... [--iterations N]`

1. Run system ID on the flights and write `runs/<name>-ft/sysid.json` and `residual.msgpack`.
2. Build `SquareEnv` with:
   - the fitted thrust gain as the centre of randomisation
   - randomisation ranges halved
   - the fitted latency fixed
   - the residual hook set
3. Load the actor and critic from `runs/<name>/params.msgpack`, then continue SHAC:
   - learning rates at 0.25x
   - default 200 iterations
4. Write a full run directory to `runs/<name>-ft/`, flight artifact included, and evaluate it in
   both the corrected and the uncorrected simulator.

### Known limitation

The logged states are the firmware's Kalman estimates, not ground truth. The fit partly absorbs
estimator bias and drift. That is acceptable for a 1 m square flown for tens of seconds, and the
report says it is fitting to the estimate.

## Testing

All hermetic, following the existing suite's style. Simulator tests skip without the `sim` extra.

- `test_square_reference.py` (numpy only):
  - the reference is closed and has the right perimeter and lap time
  - position and velocity are continuous at corners
  - speed is constant
  - cw and ccw mirror each other
- `test_square_env.py`:
  - observation and critic shapes
  - `step` gradients with respect to the action over a 16-step rollout are finite and match
    finite differences
  - a reset world's state has zero gradient with respect to the previous state
  - thrust gain and latency act as configured
  - a constant `states.force` residual changes the trajectory as expected
  - zero action from the reference start stays near it for 0.5 s
- `test_shac.py`:
  - one `iterate` runs and returns finite metrics
  - bootstrap masking is correct on a hand-built crash and truncation
  - TD(lambda) matches a direct computation
  - on the `cpu-test` preset, return improves over 20 iterations (seeded)
- `test_square_policy_runtime.py`:
  - `SquareRunner` gives the same action as the simulator's observation path for the same state
  - version 1 hover artifacts still load
- `test_fly_square.py`: with fake cflib objects, following `test_fly_policy.py`:
  - abort on position error
  - landing in `finally`
  - firmware-mode action conversion inverts `to_setpoint`
  - the CSV header and format
- `test_sysid.py`, the key test:
  - synthesise flights in the simulator with thrust gain 0.85 and 1 step of latency, flown by a PD
    controller with exploration noise, and write them through the real CSV writer
  - the fit recovers the gain within 2% and the latency exactly
  - it reduces held-out 10-step error by at least 50% against the uncorrected simulator

## Out of scope

Rendering the square: an easy follow-up with `sim/render.py`. Also out of scope: camera or
Multi-ranger observations for the square, and obstacle avoidance.
