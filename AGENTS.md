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
| `vlm/` | pydantic, numpy, `drones.control`, `drones.config` | The VLM pilot. torch, transformers and PIL (the `vlm` extra) are imported inside functions only. `vlm/agent.py` alone imports `sim/`. |
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
- **Scanned scenes (`sim/scenes.py`) are scenery, never physics.** `scenes.attach` swaps only
  `sim.mj_model` / `sim.mj_data`, the model `Sim.render` draws; the MJX model the dynamics and
  sensors run on never sees the scan. Keep it that way. A 500k-triangle non-convex mesh does not
  belong in MJX, and a policy must fly the same with or without a backdrop. Where a scan must
  stop something, do it outside the physics, as `sim/explore.py` stops its setpoint with
  `mj_rayMesh`. Cast rays at the scan's geoms by id: `mj_ray` also hits the drone's own collision
  sphere and CrazyFlow's hidden floor plane. The scan's geoms are static world geoms, so qpos and
  the mocap bodies still line up for `Sim.render`'s copy. Two things that cost time here: MuJoCo's `usertexcoord` has v running *down* the image, the same as
  glTF, so do not flip it (a flip samples the unused, black part of a Gibson texture). And Gibson
  textures are 16k² JPEGs, 768 MB once decoded, so decode through PIL's `draft`.
- **HM3D, and the EQA benchmarks on it (`sim/eqa.py`), are in habitat-sim's frame.** HM3D meshes
  are z-up like ours, but benchmark poses are habitat's y-up. A habitat point `(x, y, z)` is
  `(x, -z, y)` here, and a heading θ about habitat's +y is yaw θ + π/2 (the agent faces −z at
  rest). Use `eqa.habitat_point` / `habitat_yaw`. Mirroring y instead puts benchmark paths
  through walls, yet a "nearest floor below the path" check slightly favoured the mirror. Test a
frame guess against walls, not floors. HM3D's
  textures are Basis Universal, decoded by `sim/basis.py` (basisu's WebAssembly build under
  `wasmtime`, since no Python package reads .basis). Its images come out **upside down** relative to
  glTF: `scenes.debasis_glb` flips them, and without that a scan renders as confetti.
  IndoorUAV's own `posture.json` rows are `[x, y, height, yaw°]`, which is habitat's (x, z, y), so
  here they are `(x, -y, height)` with yaw `90° - yaw`. Its JSON is GBK-encoded, not UTF-8.
- **`recordings/intrinsics.json` is read by `sim/lens.py` and written by
  `vision/calibrate.py`** (on the `neural-sandbox` branch). It uses OpenCV's pinhole convention:
  pixel centres at integers, distortion `k1 k2 p1 p2` with k3 fixed at 0. Change the format in both
  places and bump `version` in both. Films of the deck view go through
  `rl/render.py:open_writer`, which keeps the frame size: imageio would otherwise rescale
  324×244 to a multiple of 16 and break the calibration.
- **Every simulated camera image goes through `sim/lens.py:LensCamera`**, whatever the model:
  `DeckCamera` (a drone in an env) and `SceneView` (a dataset scan alone) only choose the model
  and the pose. `SceneView` poses are in the scan *file's* frame, the frame benchmark poses are in.
  That is not the shifted world of `scenes.load` + `attach`, where the open floor is the origin.
- **mujoco's OpenGL backend is settled by `import drones.sim`, not by `main()`.** mujoco reads
  `MUJOCO_GL` once, on first import, and takes GLFW when it is unset. CrazyFlow imports mujoco and
  `drones/sim/__init__.py` imports CrazyFlow, so a console script under `drones.sim` has the
  backend fixed before its `main()` runs: `drones-render-agent` set EGL there for months without
  effect and died on a cluster node with "X11: The DISPLAY environment variable is missing". The
  choice now lives in `drones/sim/__init__.py` (EGL when there is no display). `drones.rl` does not
  import mujoco on import, so its render commands may still choose in `main()`. A laptop is a
  poor test of this: with `DISPLAY` unset GLFW still finds the Wayland socket, so check the
  backend itself (`mujoco.GLContext.__module__`), as `tests/test_agents.py` does.
- **torch and an EGL context in one process: keep triton out.** torch imports triton when it is
  installed (on Linux it always is), and triton's bundled LLVM segfaults on import once Mesa has
  made an EGL context, with no message, only exit code 139. torch imports it lazily
  (`torch._dynamo`, `torch.utils.flop_counter`), well after `import torch`, so importing torch
  before the renderer does not help; only importing triton itself first does.
  `vlm/backend.py:_load` sets `sys.modules['triton'] = None` before importing torch, which torch
  reads as triton not being installed, and logs "triton not found" for. Anything else that loads
  torch beside the simulator's renderer needs the same. Find such a crash with
  `python -X faulthandler`.
- **lm-format-enforcer 0.11.3 (the VLM pilot's constrained decoding) has three traps.** Its
  transformers integration module cannot be imported under transformers 5 (it imports
  `PreTrainedTokenizerBase` from `transformers.tokenization_utils`, which no longer exists), so
  `vlm/backend.py:_vocabulary` does that module's small job with the library's core. A JSON
  schema `const` that is not a string (`"done": false`) crashes its parser. And `maxItems: 0`
  admits one item. That is why `vlm/actions.py:json_schema` always asks for exactly the chunk's
  size in actions, done or not, and why the range of a number stays with pydantic alone. Check a
  grammar change character by character, as `tests/test_vlm_constrained.py` does; the library's
  own errors are rarely clear.
- **transformers' image-text-to-text pipeline takes generation options only in
  `generate_kwargs`.** Any other keyword goes to the processor and is dropped with a warning, so
  `do_sample=False` passed directly still samples, as Gemma's own generation config says to.
- **Text in the explorer: `mjr_overlay` silently stops at 500 characters (`mjMAXOVERLAY`).**
  Longer text goes through `Explorer._draw_panel` (`mjr_rectangle` + `mjr_text`). `mjr_text`'s
  (x, y) are relative to the viewport of the *previous* `mjr_` call, not the window. MuJoCo's fonts
  have ASCII only, so pass prompts through `explore.ascii_text`.

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
- **Benchmark inference** (an agent answering EQA questions or following IndoorUAV instructions
  from images) → `sim/scene_view.py:SceneView`, with poses from `eqa.start_pose` /
  `eqa.path_poses`. It builds no CrazyFlow Sim; step a Sim only when the dynamics matter. An
  agent is anything with `reset(question, pose)` and `act(observation) -> Pose | None`
  (`sim/agents.py`). Run it with `agents.episode`, or film it with `drones-render-agent --agent
  package.module:factory`. Do not make agents subclass anything, and keep the agent code itself
  out of this repo's `sim/` unless it is a baseline. An agent may also have `answer` and
  `caption` (a list of lines the film shows under the question); both are read by attribute, so
  `sim/` still imports no agent. The film writes each frame through `agents.decided`, after the
  agent has acted on it: a caption read any earlier describes the previous frame, and the frame
  the agent stopped on would never show why.
- **A VLM that flies** → `src/drones/vlm/`. The schema a model must reply in is
  `vlm/actions.py`, and it is the contract: change the fields, the moves or `CHUNK` there, and
  update the action-space description in `vlm/prompt.py` with them. The prompt shows the reply
  format as a template with numbered slots, never as a finished chunk: given one complete
  example, Gemma 3n E2B returned that example for every frame. `tests/test_vlm_prompt.py` fails
  if any line of the prompt would pass as a reply. The local model's decoding is held to
  `actions.json_schema` (the pydantic schema with the count pinned) unless `constrain=0`; the
  pilot still validates every reply, because the grammar cannot hold a number's range. Another
  model or a server is a new `Backend` (`generate(prompt, image) -> str`), not a change to
  `Pilot`, selected by `make(backend=...)`. `Pilot` does not catch a backend's exceptions: a
  backend that raises stops the run. `ClaudeCodeBackend` (`backend=claude-code`) runs
  `claude -p` per chunk on the user's subscription. That rules out `--bare`, which reads only
  `ANTHROPIC_API_KEY`, so the CLI is stripped flag by flag instead: an image goes in only as a
  stream-json message, which needs stream-json out and `--verbose`; `--tools ''` still leaves
  the claude.ai MCP connectors in, and only `--strict-mcp-config` takes them out. Its tests use a
  fake `claude`; a real call spends the user's usage, so ask before making one. A real-drone
  adapter
  would call `Pilot.step` and pass each Command to `DroneController.set_control`; none exists yet,
  and writing one does not make it something an agent may run. The size of one action comes from
  `drones.config`, so a local `.env` changes how far the simulated drone moves per action and what
  the prompt tells the model; the tests run at the shipped defaults.
- **A new learning algorithm** → `src/drones/rl/`. Decide deliberately what the policy outputs:
  attitude commands replace the firmware's position and velocity loops; emitting a
  `control.mixer.Command` instead keeps the teleop safety layer underneath.
- **A new contract between sim and drone** → `policy/`, numpy-safe, `xp`-parameterised.
- **Heavy dependencies** → the `sim` extra or a new one, so the machine running the phone page stays
  light.
- **Other manual control** (gamepad, keyboard) → `teleop/<name>/`, driving
  `DroneController.set_control()` and `submit()` so the watchdog still protects it. Manual control
  of the *simulated* drone is different: it lives in `sim/` (`sim/explore.py`), because `teleop/`
  may not import the simulator.

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
uv sync --extra sim --extra vlm      # torch + transformers, for the VLM pilot's local model
```

`uv sync` drops extras you do not name, so list every one you want each time. That includes the
sync that registers a new entry point: `uv sync --extra sim` alone uninstalls OpenCV.

`drones-download-scenes` (sim extra) puts IndoorUAV scans in `scenes/`, git-ignored like `runs/`.
Never commit them. `drones-explore-scene` flies the *simulated* drone around them from the keyboard
in a GLFW window. It needs a desktop session, so leave it for the user to run, and to exercise it
yourself use `Explorer(..., visible=False)` as `tests/test_explore.py` does. Use one `Explorer` per
process: a second GLFW window, opened after the first was terminated, reads back black.
`--benchmark hm-eqa|mt-hm3d|express-bench|a-eqa|indoor-uav` shows EQA questions or IndoorUAV's
instructions. Code that picks scenes goes through `eqa.busiest` / `prepare` / `locate`, never
`eqa.load` alone: IndoorUAV's prompts arrive per scene, so `load` returns only prepared scenes.
Benchmark files are cached in `scenes/benchmarks/`, HM3D scenes in `scenes/hm3d/`, and the pinned Basis decoder in
`~/.cache/drones/`.

`drones-render-agent` renders offscreen (EGL) and touches no hardware, so you may run it; it
downloads a question's scene if missing. Tests must not depend on `scenes/`:
`tests/test_agents.py` swaps `scene_view.load_scene` for a synthetic box and runs `main()` on it.

The VLM pilot has no entry point of its own: it is `drones-render-agent --agent
drones.vlm.agent:make --agent-arg action_space=discrete --fps 16 --steps 48`. One step is 1/16 s,
so 16 fps is real time. A run loads a local model (`google/gemma-3n-E2B-it`, gated) and is slow
without a GPU: about a minute per model call, one call per 16 steps, so always pass `--steps` (the
default 500 is half an hour or more). `--agent-arg backend=claude-code` asks Claude through the
`claude` CLI instead (a few seconds a call, no `vlm` extra, the user's subscription usage).
`--agent-arg chunk=N` (1 to 32) sets the actions per call;
an action is always 1/16 s (`actions.STEP`), so N is how long the model flies open loop, not how
fast. `CHUNK` is only the default size: pass the size through (`parse_chunk`, `json_schema`,
`build_prompt`, `Pilot(size=)`) rather than reading `CHUNK`. The tests script the replies
instead and load nothing. The
agent assumes its start is `start_altitude` (1.0 m) above the floor, which is wrong for
`indoor-uav` and after `--eye-height`; pass the real height. With `--scene`, `--ask TEXT` gives
the agent a question of your own. The film's panel under the question holds `AGENT_LINES` lines
(`sim/render_agent.py`); `tests/test_vlm_agent.py` checks the pilot's longest caption fits it, so
change the two together.
