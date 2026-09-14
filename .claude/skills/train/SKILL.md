---
name: train
description: Use when training, evaluating or rendering a policy in the simulator - the hover task (PPO) or the square task (SHAC). Covers choosing a config and preset, what a run writes, and how to read the result.
---

# Training a policy

Two tasks, two algorithms. Pick by what the policy must do.

| | Hover | Square |
| --- | --- | --- |
| Goal | Hold a height, stop drifting | Fly a 1 × 1 m square |
| Algorithm | PPO (`rl/ppo.py`) | SHAC (`rl/shac.py`), backprop through the simulator |
| Observes | Deck sensors only, never position | The firmware's state estimate |
| Config | `configs/hover/*.yaml` | `configs/square/shac.yaml` |
| Env | `sim/hover_env.py` | `sim/square_env.py` (differentiable) |

Everything needs the `sim` extra. `uv sync` drops extras you do not name, so keep listing them:

```bash
uv sync --extra sim                  # CPU
uv sync --extra sim --extra gpu      # adds jax[cuda12], Linux only
```

## Before you start a long run

**Always run the `cpu-test` preset first.** It is sized to prove the whole path works end to end in
under a minute and will not learn anything useful. A config error that surfaces 40 minutes into a
GPU run is the thing this avoids.

```bash
uv run --extra sim drones-train-hover --preset cpu-test
uv run --extra sim drones-train-square --preset cpu-test
```

## Running it

```bash
# hover
uv run --extra sim drones-train-hover --preset cpu                       # minutes on a laptop
uv run --extra sim drones-train-hover configs/hover/imu.yaml --preset cpu
uv run --extra sim --extra gpu drones-train-hover configs/hover/baseline.yaml --device gpu

# square
uv run --extra sim drones-train-square --preset cpu
uv run --extra sim --extra gpu drones-train-square --device gpu
```

A run is one YAML config, resized by `--preset cpu-test|cpu|gpu`, then adjusted by any number of
`--set SECTION.KEY=VALUE` overrides:

```bash
uv run --extra sim drones-train-hover --set sensors.enabled=[multiranger] --set ppo.total_steps=2e6
```

Unknown keys and unknown sensor names are errors, not silent defaults. A config can `extends:`
another and override only what differs.

Sensor choices for the hover task (`sensors.enabled`): `optical_flow` (3 values), `multiranger`
(5), `imu` (6), `camera` (an image, simulation only — **a camera policy cannot be flown**, the
artifact carries no image encoder).

## What a run writes

`runs/<name>/`:

| File | What it is |
| --- | --- |
| `config.yaml` | the fully resolved config — pass it back to rerun exactly |
| `config.json` | the same, as JSON |
| `metrics.csv` | one row per logged iteration |
| `params.msgpack` | the latest iterate, written atomically every `--save-every` (default 100) |
| `policy/` | the flight artifact: `policy.json` + `actor.npz`, numpy-only |

Because `params.msgpack` is written atomically, you can evaluate or render a run **while it is still
training**; you see the policy as of the last save. Use `--save-every 10` to watch it closely.

## Reading the result

```bash
uv run --extra sim drones-eval-hover  runs/<name>     # policy vs open-loop hover
uv run --extra sim drones-eval-square runs/<name>     # crash rate, laps, tracking error
```

In `metrics.csv`, watch `episode_return` rising and `crash_rate` falling. For the square,
`pos_error` is the number that matters — a good CPU run reaches ~0.03–0.04 m with `crash_rate` 0.

For reference, `runs/square-cpu-v2` (CPU, 8 M steps, ~8 min) ended at `pos_error` 0.041,
`crash_rate` 0.0, `episode_return` 1668.

Note `episode_return`, `episode_length` and `crash_rate` are `nan` on iterations where no episode
finished. That is expected, not a bug — they are per-episode averages.

## Watching it fly

```bash
uv run --extra sim drones-render-hover  runs/<name>                    # renders/chase-seed0.mp4
uv run --extra sim drones-render-square runs/<name> --camera top --episodes 3
uv run --extra sim drones-render-hover  runs/<name> --open-loop        # the zero-action baseline
uv run --extra sim drones-render-square runs/<name> --out square.gif --width 320 --height 240
```

- `chase` follows the drone; `top` looks straight down and is the one for judging drift.
- The square renderer draws the reference in blue with the current target point marked, and the
  flown path as an orange trail.
- A GIF holds every frame in memory until written — keep GIFs small. MP4 uses the bundled ffmpeg.
- Headless nodes render through EGL; the script sets `MUJOCO_GL=egl` unless you already did.
  `osmesa` does not work with this extra's PyOpenGL.

## Exporting for the drone

Training exports `runs/<name>/policy/` automatically. To re-export an existing run:

```bash
uv run --extra sim drones-export-policy runs/<name>
```

The artifact runs on numpy alone, so the laptop at the radio needs only the base install.

**Do not fly it yourself.** Flying is a human action — see the `fly-check` skill for what the
operator needs to do first.
