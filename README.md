# drones

Flight software for a **Crazyflie 2.1 Brushless** with a **Flow deck v2** and a
**Multi-ranger deck**: a hardware-free control law, a real-drone backend, a
phone control page and autonomous missions — laid out so simulators and RL can
reuse the exact control law the drone flies.

| Command | What it does |
| --- | --- |
| `uv run drones-web` | Phone control page: fly/turn joystick, height slider, live ranger telemetry, emergency stop |
| `uv run drones-wall-avoid` | Headless flight: take off, hold height, avoid walls, land on a detected ceiling |
| `uv run --extra sim drones-train-hover [config.yaml]` | Train a hover-stabilising policy in the CrazyFlow simulator |
| `uv run --extra sim drones-eval-hover runs/<name>` | Evaluate a trained policy against open-loop hover |
| `uv run --extra sim drones-render-hover runs/<name>` | Film a trained policy flying in the simulator (MP4 or GIF) |
| `uv run drones-fly-policy runs/<name>/policy` | Fly a trained policy on the real drone |
| `uv run --extra sim drones-train-square [config.yaml]` | Train a policy to fly a 1 × 1 m square with SHAC |
| `uv run --extra sim drones-eval-square runs/<name>` | Evaluate a square policy: crash rate, laps, tracking error |
| `uv run drones-fly-square runs/<name>/policy` | Fly the square on the real drone, or let the firmware fly it and log (`--firmware`) |
| `uv run --extra sim drones-finetune-square runs/<name> --flights …` | Fit the simulator to real flights, then finetune the policy in it |
| `uv run --extra sim pytest` | Test suite, including closed-loop stability checks and the simulator |

## Setup

```bash
uv sync
cp .env.example .env
```

`.env` is git-ignored and holds per-drone tuning. It is loaded by explicit path
from the repository root, so a notebook or REPL opened anywhere still picks it
up.

`CFLIB_URI` defaults to `auto`, which scans and uses whichever single interface
is present — a Crazyradio dongle (`radio://…`) or the drone plugged straight in
over USB (`usb://0`). Set it explicitly if you have more than one drone or
radio in range. If the configured URI is not available, the error names what
the scan *did* find.

On Linux the Crazyradio needs udev rules, otherwise nothing can open the
dongle — see the [Bitcraze USB permissions guide](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_permissions/).

## Layout

```
src/drones/
  config.py           settings from .env
  control/            the flight control law: pure Python, never imports cflib
    avoidance.py      ranger distances -> push away from walls
    mixer.py          Command + Ranges -> Setpoint, one 100 ms step at a time
    safety.py         debounced ceiling detection
  crazyflie/          the real-drone backend: everything that imports cflib
    link.py           URI resolution, deck checks, sensor reads, battery log
    controller.py     flight state machine on one thread: watchdog, e-stop, landing
  teleop/web/         phone control page (FastAPI + WebSocket) and its static UI
  missions/
    wall_avoid.py     headless autonomous flight
    fly_policy.py     drones-fly-policy: fly a trained policy on the real drone
    fly_square.py     drones-fly-square: fly a square policy, or log the firmware flying one
  policy/             what a trained policy needs at flight time, numpy only
    interface.py      observation layout and action scaling, shared by sim and drone
    runtime.py        the exported artifact, and a runner that feeds it live readings
    square.py         the square's reference path and observation, shared by sim and drone
  sim/                CrazyFlow simulation (sim extra): scene, sensor models, tasks
    assets/room.xml   floor, four walls and a ceiling, re-placed per world every episode
    sensors.py        Multi-ranger, Flow deck, IMU and a colour camera, as batched JAX
    hover_env.py      the hover-stabilisation task, pure functions of an EnvState
    render.py         offscreen video of one world: chase and top-down cameras, flight-path trail
    square_env.py     the square task: differentiable, for SHAC
    residual.py       a learned force and torque correcting the dynamics
    calibration.py    hover-thrust calibration
  rl/                 PPO in JAX (sim extra)
    networks.py       actor-critic; only the critic sees privileged simulator state
    ppo.py            rollout + GAE + updates compiled into one jitted call
    experiment.py     YAML experiment configs
    train_hover.py    drones-train-hover
    evaluate.py       drones-eval-hover
    render.py         drones-render-hover
    export.py         drones-export-policy: trained params -> flight artifact
    shac.py           short-horizon actor-critic through the simulator
    sysid.py          fit thrust gain, latency and residual to flight logs
    square_experiment.py square experiments as YAML: presets and --set
    train_square.py   drones-train-square
    evaluate_square.py drones-eval-square
    finetune_square.py drones-finetune-square
configs/hover/        experiment configs: baseline, imu, camera
configs/square/       experiment config: shac.yaml
tests/                pytest, hermetic: runs on the code defaults, not your .env
```

The key seam is `control.mixer.Mixer`. It holds the only state the control law
needs and advances one `UPDATE_PERIOD` per `step()` — no clock, no threads, no
hardware. The real drone, the auto mode on the phone page and the headless
mission all fly through it.

### Where new work goes

- **New simulated tasks** → beside `sim/hover_env.py`, reusing `sim/sensors.py`
  and the room scene. A task that should fly through the teleop control law can
  step a `control.mixer.Mixer` and send its `Setpoint` to CrazyFlow's `state`
  control instead of `attitude`.
- **Other learning algorithms** → `src/drones/rl/`. Decide deliberately what a
  policy outputs: attitude commands (as the hover task does) replace the
  firmware's position and velocity loops; a `Command` would keep the teleop
  safety layer underneath it. `runs/`, `checkpoints/` and `wandb/` are
  git-ignored.
- **Heavy dependencies** → the `sim` extra, or a new one, so the machine running
  the phone page does not need them.
- **Other manual control** (gamepad, keyboard) → `teleop/<name>/`. Drive
  `DroneController.set_control()` and `submit()`, the same API the web page
  uses, and the watchdog protects it too.
- Keep `control/` free of cflib and web imports — `tests/test_architecture.py`
  fails if anything in it pulls them in.

## Phone control page

```bash
uv run drones-web
```

It prints the LAN URL to open, e.g. `http://192.168.1.20:8000/`. Your phone must
be on the same network as this machine. Add the page to your home screen for a
full-screen control panel. The UI is served with `Cache-Control: no-store`, so
a restarted server always means a fresh page.

- **Height slider** (left) — an absolute target height in metres: drag it and
  the drone flies there and holds. The dashed "now" line is the measured
  height, so you can see it tracking.
- **Fly / turn joystick** (right) — up and down is forward/back, left and right
  turns on the spot. Springs back to centre. Proportional up to
  `MAX_MANUAL_SPEED` and `MAX_YAW_RATE`.
- **No sideways control.** Point the nose where you want to go; lateral motion
  comes only from the avoidance vector.
- **Auto wall-avoid** — hands the flight to the autonomous behaviour; the
  controls grey out while it runs.
- **Avoidance on/off** — on by default, blending the repulsion vector under
  your stick input. Off gives full manual authority.
- **Emergency stop** — cuts the motors. It is drained ahead of every other
  queued command, so no further setpoint is sent after it.
- **Recover after stop** — clears the supervisor lock left by an emergency stop
  or a tumble, so you can arm again without rebooting the drone.

`WEB_TOKEN` is empty by default, which means **anyone on your network can fly
the drone**. Set it and both the page and the WebSocket require
`?token=<value>`; the printed URL includes it.

## How avoidance works

Each horizontal ranger reading below `AVOID_DISTANCE` pushes the drone away from
that side. The push ramps from 0 at `AVOID_DISTANCE` to the full
`MAX_AVOID_SPEED` by `AVOID_HARD_DISTANCE` and stays at full strength closer
in. Opposing sensors cancel, so a corridor centres the drone. The result is
low-pass filtered every 100 ms step.

With the shipped defaults:

| Wall distance | Push |
| --- | --- |
| 0.9 m and beyond | 0 (ignored) |
| 0.7 m | 0.22 m/s |
| 0.5 m | 0.44 m/s |
| 0.35 m and closer | 0.60 m/s (full) |

Under manual control the push is *added* to the stick. The default
`MAX_AVOID_SPEED` (0.6) exceeds `MAX_MANUAL_SPEED` (0.4), so a full-forward
stick into a close wall still backs off. The ceiling check needs three
consecutive readings under `CEILING_DISTANCE`, so one bad measurement cannot end
the flight; it is active in every mode.

**Tuning and damping.** The ramp's steepness,
`MAX_AVOID_SPEED / (AVOID_DISTANCE - AVOID_HARD_DISTANCE)`, sets how hard the
drone reacts per centimetre. At the defaults the closed-loop simulation settles
within 10 s in corridors from 0.6 to 2.0 m, even with a sluggish 0.6 s velocity
lag, and a 1.5 m/s drift at a wall from 0.9 m out is stopped with 0.41–0.62 m
to spare. A much steeper ramp is still stable but underdamped: on a sluggish
drone in a narrow corridor it rocks for tens of seconds before settling.

## Safety behaviour

- **Watchdog.** The page sends control state 20×/s. After `STICK_TIMEOUT`
  (0.7 s) of silence the drone stops moving but *keeps holding its height*;
  after `LINK_TIMEOUT` (3 s) it lands. Auto mode needs no client.
- **Altitude envelope.** Height targets are clamped to
  `MIN_ALTITUDE`…`MAX_ALTITUDE` and tracked with a climb rate capped at
  `MAX_CLIMB_SPEED`.
- **Landing is in a `finally`.** Any exception in the control loop still brings
  the drone down; if landing itself fails, an emergency stop is sent.
- **Pre-flight checks.** Refuses to fly if either deck is missing, if the
  Multi-ranger produces no data, or if a ceiling is already detected.

## Configuration

Everything below lives in `.env`; the defaults are in `src/drones/config.py`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CFLIB_URI` | `auto` | `auto`, or `radio://<radio>/<channel>/<rate>/<address>`, or `usb://0` |
| `TAKEOFF_HEIGHT` | `1.0` | Hover height after take-off, m |
| `TAKEOFF_VELOCITY` | `0.3` | Take-off / landing speed, m/s |
| `MIN_ALTITUDE` / `MAX_ALTITUDE` | `0.2` / `2.0` | Altitude envelope, m |
| `AVOID_DISTANCE` | `0.9` | A wall closer than this starts pushing the drone away, m |
| `AVOID_HARD_DISTANCE` | `0.35` | At or below this the push is at full strength, m |
| `MAX_AVOID_SPEED` | `0.6` | Cap on the avoidance speed, m/s |
| `CEILING_DISTANCE` | `0.5` | Something above within this triggers landing, m |
| `MAX_MANUAL_SPEED` | `0.4` | Forward/back speed at full stick, m/s |
| `MAX_CLIMB_SPEED` | `0.3` | Cap on the climb/descent rate, m/s |
| `MAX_YAW_RATE` | `90.0` | Turn rate at full stick, deg/s |
| `MAX_FLIGHT_TIME` | `60.0` | Auto mode lands after this many seconds |
| `WEB_HOST` / `WEB_PORT` | `0.0.0.0` / `8000` | Server bind address |
| `WEB_TOKEN` | *(empty)* | If set, required as `?token=…` |
| `STICK_TIMEOUT` / `LINK_TIMEOUT` | `0.7` / `3.0` | Watchdog thresholds, s |

Set `DRONES_NO_DOTENV=1` to ignore `.env` entirely, as the test suite does.

## Simulation and training

Everything here needs the `sim` extra: CrazyFlow, JAX, MuJoCo, flax and optax.

```bash
uv sync --extra sim                  # laptop or CPU node
uv sync --extra sim --extra gpu      # CUDA node: adds jax[cuda12]
```

`uv sync` removes packages from extras you did not name, so always list every
extra you want. The `uv run --extra …` form in the commands below keeps them.

```bash
uv run --extra sim drones-train-hover --preset cpu-test               # baseline, quick check
uv run --extra sim drones-train-hover configs/hover/imu.yaml --preset cpu
uv run --extra sim --extra gpu drones-train-hover configs/hover/baseline.yaml --device gpu
uv run --extra sim drones-train-hover --set sensors.enabled=[multiranger] --set ppo.total_steps=2e6
uv run --extra sim drones-eval-hover runs/<name>                      # policy vs open-loop hover
uv run --extra sim drones-render-hover runs/<name>                    # video of the policy flying
```

A run is described by a YAML config (default `configs/hover/baseline.yaml`).
`--preset cpu-test|cpu|gpu` resizes it for the machine, then each
`--set SECTION.KEY=VALUE` overrides one setting. Each run writes to
`runs/<name>/`:
- `config.yaml`: fully resolved, so passing it back reruns the experiment exactly
- `config.json`
- `metrics.csv`
- `params.msgpack`
- `policy/`: the flight artifact

### Choosing sensors

`sensors.enabled` picks what the policy observes:

| Name | Hardware | Values per frame |
| --- | --- | --- |
| `optical_flow` | Flow deck v2: PMW3901 flow and the downward z-ranger | 3 |
| `multiranger` | Multi-ranger deck: front, back, left, right, up | 5 |
| `imu` | Gyro rates, and the gravity direction from the attitude estimate | 6 |
| `camera` | Forward colour camera (simulation only, for now) | an image |

Configs in `configs/hover/`:
- `baseline.yaml` (the default) has `multiranger` and `optical_flow`, the decks on the drone.
- `imu.yaml` adds `imu`.
- `camera.yaml` adds `imu` and `camera`.

A config can `extends:` another and override only what differs. Unknown keys and
unknown sensor names are errors, not silent defaults:

```yaml
extends: baseline.yaml
sensors:
  enabled: [multiranger, optical_flow, imu]
ppo:
  learning_rate: 1.0e-4
```

### What is simulated

- **The drone** is CrazyFlow's `cf21B_500`, the 2.1 Brushless with the 500 mAh
  battery, on the `so_rpy` dynamics fitted to real flight data. Physics runs at
  500 Hz and the policy at 50 Hz.
- **The scene** (`sim/assets/room.xml`) is a floor, four walls and a ceiling.
  The walls and ceiling are mocap bodies, so every world samples its own room
  each episode: 1.5–5 m across, 1.6–3 m high. Anything added to the XML is seen
  by the sensors.
- **The sensors** (`sim/sensors.py`) report what the hardware reports:

| Sensor | Model |
| --- | --- |
| Multi-ranger deck | Five VL53L1x rays (front, back, left, right, up), 4 m range, noise growing with distance; out of range reads 4 m |
| Flow deck z-ranger | The same, looking down |
| Flow deck PMW3901 | Counts per 10 ms frame from the firmware's own flow model (`mm_flow.c`): body velocity over ground distance, less rotation |
| IMU | Gyro rates, plus the gravity direction the onboard roll/pitch estimate implies |
| Colour camera (optional) | Raycast, flat-shaded, AI-deck geometry (forward-facing, 70° vertical FOV) |

The camera has the right geometry but is not photoreal. CrazyFlow's
gaussian-splat camera (`crazyflow.sim.sensors.splat`) is the drop-in upgrade on
a GPU once you have a `.ply` capture of your room.

### The task

Hover stabilisation: from a disturbed start (up to 0.5 m/s, 14° of tilt, 1 rad/s
of rotation), hold a target height between 0.5 and 1.5 m and stop drifting,
**without ever observing position**. Height comes from the z-ranger and drift
from optical flow, as on the real drone.

The policy sees three stacked frames. Each frame holds the enabled sensors, the
target height and the previous action: 13 values for the baseline. The critic
also gets privileged state (true position, velocity, room size), which is only
needed in training. Flying earns between 0 and 2 per step: 1 for staying up, up to 1 more
for holding the target height without moving, less capped penalties for tilt,
rotation, jerky actions and hugging walls. A crash costs 10. The floor at 0
matters: with uncapped penalties, early policies learned to crash on purpose to
escape them. An episode ends on a crash (within 10 cm of a surface, or tilted
past 57°) or after 10 s.

### Actions, and the real drone

The policy outputs four values in [-1, 1]:

| Action | Maps to |
| --- | --- |
| roll, pitch | ±0.35 rad attitude setpoints |
| yaw rate | ±1.5 rad/s |
| thrust | 0 is hover, calibrated by simulation (0.4395 N, 1.033 × m·g for the fitted model); ±1 are the motor limits, 0.085 and 0.800 N |

Measured in the simulator from a still hover: **+roll moves the drone right
(−y), +pitch moves it forward (+x), +yaw rate turns it left.**

See [Flying a trained policy](#flying-a-trained-policy) for how these reach the
real drone.

### Flying a trained policy

Training exports `runs/<name>/policy/`:
- `policy.json`: sensors, observation layout, action scaling and the calibrated hover thrust
- `actor.npz`: the network weights

The artifact runs with numpy alone, so the laptop at the radio needs only the
base install. `drones-export-policy runs/<name>` re-exports an existing run.

```bash
uv run drones-fly-policy runs/<name>/policy --dry-run          # motors off: sensors and policy live
uv run drones-fly-policy runs/<name>/policy --authority 0.3    # first flights: 30% of the policy
uv run drones-fly-policy runs/<name>/policy --height 1.0 --duration 10
```

A flight goes like this:

1. Release the legacy commander's thrust lock with one zero-thrust setpoint, on
   the ground. Arm, then take off on the firmware's own Flow-deck hover controller.
2. Hover for 3 s, reading `controller.cmd_thrust`: the command that holds *this*
   drone up. The policy's thrust maps through its ratio to the simulator's hover
   thrust, anchored on that value.
3. Hand attitude control to the policy at 50 Hz through `commander.send_setpoint`.
4. Hand back to the firmware and land when:
   - the time is up
   - you press Ctrl-C
   - a limit trips: tilt over 30°, any Multi-ranger reading under 0.2 m, height
     outside 0.15 m to target + 1 m, or sensor data older than 0.25 s

   Landing runs in a `finally`, so an exception lands too.

`--dry-run` connects, streams the sensors and runs the policy with the motors
off, printing what it would command and when it would abort. Move the drone by
hand over the floor to check the sensor mapping before anything spins. Every
run, dry or not, is logged to `runs/<name>/flights/<timestamp>.csv`, and the
script asks for confirmation before arming (`--yes` skips it).

The conventions come from the firmware source rather than assumptions:

| | Simulator | On the drone |
| --- | --- | --- |
| Optical flow | pixels per 10 ms frame, +x forward | `x = -0.1 × motion.deltaY`, `y = -0.1 × motion.deltaX`: the axis swap in `flowdeck_v1v2.c`, `FLOW_RESOLUTION` in `mm_flow.c` |
| Attitude | quaternion | `stateEstimate.qx..qw`, sidestepping the legacy inverted Euler pitch |
| Roll setpoint | +roll moves right | sent as is |
| Pitch setpoint | +pitch moves forward | negated: legacy CF2 pitch is inverted |
| Yaw-rate setpoint | +yaw rate turns left | negated: "legacy rate input is inverted" (`crtp_commander_rpyt.c`) |

These come from reading the firmware, not from flying. **Check every sign on a
tethered drone before a free flight.** `--roll-sign`, `--pitch-sign` and
`--yaw-rate-sign` flip them. Camera policies cannot be flown yet: the artifact
carries no image encoder, and the AI deck streams over Wi-Fi rather than the
radio.

Sim-to-real gaps that remain:
- no domain randomisation of mass or motor response
- no sensor latency
- flow isn't quantised
- the thrust mapping assumes lift is proportional to the thrust command near hover

### Measured here

`configs/hover/baseline.yaml --preset cpu` (Multi-ranger and Flow deck, no IMU;
256 worlds, 5 M steps) took about 3 minutes on a 16-core CPU at roughly 27k
steps/s. Return rose from 30 to about 400 of a possible ~1000, and the crash
rate fell from 100% to about 3%. Evaluated on 256 fresh episodes:

| | Crash rate | Survived (of 10 s) | Height error | Drift speed |
| --- | --- | --- | --- | --- |
| Baseline policy | 0.8% | 10.0 s | 0.19 m | 0.47 m/s |
| Open-loop hover (zero action) | 95% | 3.8 s | 0.40 m | 0.31 m/s |

A larger sample of 800 episodes puts the crash rate nearer 2%, and it comes from narrow rooms. The
quarter of rooms under 2 m across held 15 of the 17 crashes, each one a drift into a wall.

It has learned to stay up from the decks alone. Height hold and drift are still
loose, which is what a longer `gpu` run is for. The open-loop drift figure looks
better only because it averages over the few worlds that had not crashed yet.
The exported artifact loads and steps with JAX, flax and the simulator blocked
from importing, as it will on the flying laptop.

### Watching a policy fly

```bash
uv run --extra sim drones-render-hover runs/<name>                          # renders/chase-seed0.mp4
uv run --extra sim drones-render-hover runs/<name> --camera top --episodes 3 --seed 7
uv run --extra sim drones-render-hover runs/<name> --open-loop              # the zero-action baseline
uv run --extra sim drones-render-hover runs/<name> --out flight.gif --width 320 --height 240
```

This flies one world with the policy's deterministic actions, as `drones-eval-hover` does, and films
it through CrazyFlow's MuJoCo renderer:
- The flight path is drawn as an orange trail from a green start marker.
- The corner shows time, height against the target, speed, and how the episode ended.
  `--font-scale 100|150|200` sets the text size in percent. The default, 100, is MuJoCo's smallest,
  and the text keeps a fixed pixel size, so a larger `--width`/`--height` makes it smaller relative
  to the picture.
- Each episode gets its own room and start from `--seed`. The last frame is held briefly, so a crash is visible.
- The video goes to `runs/<name>/renders/` unless `--out` names a file, and the extension picks the
  format. MP4 uses the ffmpeg bundled with the `sim` extra. A GIF holds every frame in memory until
  it is written, so keep GIFs small.

It renders `runs/<name>/params.msgpack`, which always holds the latest iterate. Training overwrites
it every `--save-every` iterations (100 by default) and at the end. The write is atomic, so rendering
a run that is still training is safe, and it shows the policy as of the last save. Train with
`--save-every 10` to see it closer to the current iteration.

There are two cameras:
- `chase` (default) follows the drone from the room-centre side, and moves in when a wall or the
  ceiling would come between them. MuJoCo's default camera starts outside the room, where the wall
  slabs hide everything.
- `top` looks straight down on the whole room with the ceiling hidden: +x to the right, +y up. Use it
  to judge drift.

Headless nodes render through EGL: the script sets `MUJOCO_GL=egl` unless you have already set it,
and `glfw` works on a desktop. `osmesa` fails with the `sim` extra's PyOpenGL. The render tests skip
where no EGL context can be created.

### Performance notes

- With the camera on, the rollout buffer holds every image:
  `num_envs × rollout_steps × H × W × 3` floats, which is 2.4 GB at the `gpu`
  preset and 32×24. Lower `--num-envs` for bigger images.
- Time-limit truncation is treated as an episode end without bootstrapping, a
  standard simplification.
- Walls only move at reset, so the scene geometry is recomputed then and the
  per-step rays reuse it. A task with moving obstacles would need to refresh it
  every step.

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
- `--firmware` lets the firmware's own position controller fly the same square. The roll, pitch
  and thrust it commands are logged in the policy's action units, so system-ID data can be
  collected before a policy has flown. On the first `--firmware` flight, check that `a_pitch` is
  positive while the drone accelerates forward. The sign of `controller.pitch` in the firmware log
  comes from reading the source, not from flying. The yaw action is logged as 0, because the
  firmware holds a constant heading through the flight rather than commanding a yaw rate.

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

## Development

```bash
uv run --extra sim pytest    # without the sim extra, the simulator tests skip
```

The suite is hermetic: `tests/conftest.py` sets `DRONES_NO_DOTENV`, so results
do not depend on whose drone tuning is in the local `.env`. The web tests run a
real uvicorn server and a real WebSocket handshake rather than Starlette's
`TestClient`, which fakes the transport. The controller tests drive the flight
state machine against fake cflib objects; nothing in the suite needs a drone.

## Before the first flight

- Fly in open space over a textured, non-reflective floor — the Flow deck needs
  surface texture to hold position.
- The Crazyflie is armed automatically on take-off. Keep clear of the props.
- Check the battery reading on the page; below ~3.2 V it turns red and you
  should land.
