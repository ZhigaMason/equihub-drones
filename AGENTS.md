# Working in this repository

Flight software for a real Crazyflie 2.1 Brushless, plus the simulator and RL stack that trains
policies for it. Code here ends up commanding a physical aircraft, so some of the rules below are
about safety rather than taste.

`README.md` is the reference for *what* everything does and holds the measured results. This file
covers what you need to know to change the code without breaking something you cannot see from the
file you are editing.

## Document what you change, here

**Every change an agent makes must leave this documentation correct.** Updating `AGENTS.md` is part
of the change, not follow-up work — a task is not finished while this file still describes the old
behaviour.

Update it in the same commit as the code whenever you:

- add, rename or remove an entry point, a command or a config file;
- change a shared contract (`policy/`), an observation layout, an action scaling or an artifact
  format;
- change a convention, an invariant or a safety limit recorded here;
- add or change a hook, a skill or anything else under `.claude/`;
- discover a fact that cost you real effort — a firmware quirk, a silent NaN, a gradient that
  vanishes, a dependency that must be imported first. Write it down so the next agent does not pay
  for it twice. That is why the "Conventions you cannot infer from one file" section exists.

Two rules for what goes where:

- **Facts about the code belong in a docstring or a comment**, next to the code. Put it here only
  when it is something an agent must know *before* opening the file it applies to.
- **`README.md` is for humans** — what the project does, how to use it, measured results. Keep
  numbers and usage there; keep working rules here. When a change makes both stale, fix both.

Do not let this file grow into a second README. If a section is only ever read as reference, link
to the source instead.

## Never do these

- **Never fly the real drone.** `drones-fly-policy`, `drones-fly-square`, `drones-wall-avoid`,
  `drones-web` and `drones-fpv` all spin motors on hardware a person is standing next to. Only a human starts them,
  including with `--dry-run`, which still connects to the drone. Ask; do not run them yourself.
  `drones-camera` spins nothing but connects to the drone's AI-deck, so the same hook blocks it.
- **Never `git add -A` or `git add .`** Add the files you changed, by name. The repo root collects
  large untracked artefacts (`flight*.gif` is tens of megabytes) that must not land in git.
- **Never commit `runs/`, `checkpoints/`, `wandb/` or `.env`.** They are git-ignored; keep it that
  way. `.env` holds per-drone tuning and is machine-local.
- **Never edit the same contract twice.** See the shared-contract seam below.

## Layering, and the guard that enforces it

Dependencies only ever point downwards. `tests/test_architecture.py` fails if they do not:

| Package | May import | Notes |
| --- | --- | --- |
| `control/` | stdlib, `drones.config` | The flight control law. No cflib, no web, no JAX. |
| `policy/` | numpy | Runs on the flying laptop, which has neither JAX nor the simulator. |
| `crazyflie/`, `missions/` | + cflib, numpy | Everything that talks to real hardware. `crazyflie/camera.py` also uses OpenCV (the `camera` extra), imported inside functions only. |
| `sim/`, `rl/` | + JAX, flax, optax, CrazyFlow | The `sim` extra. Never imported by the above. |

If you need something from `sim/` in `missions/`, that is the signal it belongs in `policy/`
instead.

## The shared-contract seam

`policy/interface.py` (hover) and `policy/square.py` (square) define the observation layout, action
scaling and reference path **once**, for both the simulator and the drone. They take the array
module as an argument:

```python
encode_square_obs(jnp, ...)   # sim/square_env.py, under jit, differentiable
encode_square_obs(np,  ...)   # missions/fly_square.py, on live cflib logs
```

A trained policy therefore sees the same numbers in training and in flight. **Changing one of these
functions changes what every previously exported artifact means.** If you change an observation
layout or an action scaling:

1. change it in `policy/`, never in a caller;
2. use only operations both `jnp` and `np` provide;
3. bump `FORMAT_VERSION` in `policy/runtime.py` and handle the old version, or old `runs/*/policy/`
   artifacts silently decode wrong.

## Conventions you cannot infer from one file

- **Frames.** World: +x forward at take-off, +y left, +z up. Body: +x forward, +y left, +z up.
- **Quaternions are scalar-last**, `[x, y, z, w]`, everywhere.
- **Action signs**, measured in the simulator: +roll moves right (−y), +pitch moves forward (+x),
  +yaw rate turns left. The drone-side inversions in `missions/fly_policy.py` (`DEFAULT_SIGNS`) come
  from reading the Crazyflie firmware source, **not from flying** — treat them as unverified.
- **Angular velocity is body-frame**, as the gyro reports it.
- **`import crazyflow  # noqa: F401` must precede anything that imports scipy** in any module that
  touches CrazyFlow. Existing modules do this at the top of the import block; copy the comment.
- **Reverse-mode autodiff only.** `jax.jvp` / `jacfwd` through CrazyFlow's `so_rpy` return NaN, from
  a `where` in scipy's quaternion normalisation. Use `jax.grad` / `value_and_grad`.
- **NaN must be removed before it reaches a differentiable op**, not just masked out of the output:
  a `where` that drops a NaN downstream still backpropagates a NaN cotangent (`0 * NaN = NaN`). See
  `sim/square_env.py:_step` for the pattern to follow.
- **Nothing may hold up the landing on shutdown.** uvicorn runs the lifespan shutdown —
  `controller.stop()`, which lands the drone — only after waiting for open connections, without
  limit by default, and the `drones-fpv` MJPEG stream never ends by itself. On uvicorn 0.52.4 and
  Starlette 1.6, shutdown was measured to end the stream anyway, landing in 0.2 s. Start the server
  through `teleop/web/server.py:uvicorn_config` all the same: its `SHUTDOWN_GRACE` is the backstop,
  and `test_an_open_video_stream_does_not_hold_up_the_landing` guards the behaviour across upgrades.
- **An AI-deck stream that freezes is not a bug in `crazyflie/camera.py`.** It was expensive to
  find: the laptop's Wi-Fi power saving makes the deck's ESP buffer frames until its ~48 KB heap
  runs out and it deadlocks. The fix is `802-11-wireless.powersave 2` on the drone's network
  profile; three rounds of ESP buffer tuning did not help. The ESP also serves one camera client
  at a time. The deck runs patched firmware (colour JPEG, 162×122) and the latest runs were
  made on it; README, "The deck on this drone", lists the changes. Read it before touching the
  stream code or the deck.

## Style

- Single quotes, 100 columns. `ruff check` enforces this and is clean — keep it that way.
- Module docstrings explain **why**, not what. Comments record facts that were expensive to
  discover (a firmware quirk, a fitted coefficient, a gradient that silently vanishes). This is the
  repo's most valuable property; match it.
- Imports use aligned continuations, not ruff's isort wrapping. Import sorting is deliberately not
  in the enabled rule set — do not reformat import blocks.

## Tests

```bash
uv run --extra sim pytest                     # everything, ~6 min
uv run --extra sim pytest tests/test_mixer.py # one module
```

Without the `sim` extra the simulator tests skip rather than fail. The fast, non-simulator subset
runs in about 6 seconds and is what the Stop hook checks:

```bash
uv run pytest $(grep -L importorskip tests/test_*.py)
```

- The suite is **hermetic**: `tests/conftest.py` sets `DRONES_NO_DOTENV`, so results never depend on
  the local `.env`. Keep it that way — no test may read operator tuning.
- Simulator test modules start with `pytest.importorskip('crazyflow')`.
- Nothing in the suite needs a drone. Controller tests drive the state machine against fake cflib
  objects; web tests run a real uvicorn server and a real WebSocket handshake.
## Where new work goes

- **A new simulated task** → beside `sim/hover_env.py` and `sim/square_env.py`, reusing
  `sim/sensors.py` and the room scene. If it must be differentiable (for SHAC), cast no rays and
  keep every reward term smooth.
- **A new learning algorithm** → `src/drones/rl/`. Decide deliberately what the policy outputs:
  attitude commands replace the firmware's position and velocity loops; emitting a
  `control.mixer.Command` instead keeps the teleop safety layer underneath.
- **A new contract between sim and drone** → `policy/`, numpy-safe, `xp`-parameterised.
- **Heavy dependencies** → the `sim` extra or a new one, so the machine running the phone page stays
  light.
- **Other manual control** (gamepad, keyboard) → `teleop/<name>/`, driving
  `DroneController.set_control()` and `submit()` so the watchdog still protects it.

## What is set up in `.claude/`

Committed, so it applies to anyone working in this repository.

| | What it does |
| --- | --- |
| `skills/train` | training, evaluating and rendering a policy — configs, presets, what a run writes |
| `skills/fly-check` | the pre-flight checklist, sign verification and abort limits |
| `hooks/no-live-drone.sh` | **blocks** any command that flies or connects to the drone |
| `hooks/no-blind-add.sh` | **blocks** `git add -A` / `git add .` |
| `hooks/ruff-edited-file.sh` | lints each edited `.py` and reports back (advisory) |
| `hooks/fast-tests.sh` | runs the ~6 s non-simulator suite when a turn ends (advisory) |

The two blocking hooks guard things that are expensive to undo: a damaged aircraft, and a
multi-megabyte artefact in git history. If one fires, it is not an obstacle to route around — stop
and tell the user.

Changing a hook means re-testing it. Feed it a JSON event on stdin and check the exit code
(`0` allow, `2` block/report):

```bash
printf '{"tool_input":{"command":"uv run drones-web"}}' | .claude/hooks/no-live-drone.sh; echo $?
```

Check both directions — what must be blocked *and* what must still be allowed. Both of these hooks
had false positives on the first attempt (`grep drones-fly-policy README.md` was blocked;
`git -C . add -A` was not).

`no-live-drone.sh` judges every line of a command, so a Bash heredoc or script whose *text*
mentions an entry point (a README edit via `python3 - <<EOF`) is blocked too. Edit docs with the
Edit tool instead.

## Commands

Entry points are defined in `pyproject.toml`. The `train` and `fly-check` skills in `.claude/skills/`
cover the two workflows with real sequencing to get right.

```bash
uv sync --extra sim                  # CPU
uv sync --extra sim --extra gpu      # adds jax[cuda12]
uv sync --extra camera               # OpenCV, for drones-camera and drones-fpv (AI-deck video)
```

`uv sync` drops extras you do not name, so list every one you want each time.
