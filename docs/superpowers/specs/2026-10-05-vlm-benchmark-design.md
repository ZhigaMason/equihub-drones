# Benchmarking a VLM pilot overnight: `drones-benchmark`

Date: 2026-10-05. Status: approved in conversation, awaiting review of this document.

## Goal

A command the user starts in the evening that flies an agent (in practice the VLM pilot on Claude,
`--agent drones.vlm.agent:make --agent-arg backend=claude-code`) over the scanned-scene benchmarks
and leaves, by morning:

1. **results** scored the way FAST-EQA (arXiv 2602.15813, Table 1) scores the EQA benchmarks, and
   IndoorUAV (arXiv 2512.19024) its VLN set;
2. **distillation data**: every model call's image, prompt, raw reply and parsed actions, so a
   smaller model can be trained on Claude's behaviour later.

What the user decided:

- One benchmark is run **in full**, over as many nights as it takes; `--benchmark` takes one name,
  several, or `all`.
- At a **usage limit** (or any other agent failure) the run **stops cleanly** and resumes on the
  next start. It does not wait for the limit to reset.
- The open-answer **judge is Claude through the CLI**, in a separate `score` pass.
- **Approach A**: an agent-agnostic `drones-benchmark` in `sim/`, so the built-in `look-around` and
  `follow-path` agents are baselines for free.

## Command line

```bash
uv run --extra sim drones-benchmark run --benchmark hm-eqa \
    --agent drones.vlm.agent:make --agent-arg backend=claude-code --agent-arg action_space=discrete
uv run --extra sim drones-benchmark run --benchmark all --agent ...      # every benchmark, in order
uv run --extra sim drones-benchmark run --benchmark hm-eqa a-eqa --questions 1-50 --agent ...
uv run --extra sim drones-benchmark score runs/benchmarks/*
```

`run` takes the camera options `drones-render-agent` takes (`--intrinsics`, `--fov`, `--width`,
`--height`, `--no-mount`, `--eye-height`), `--agent`, `--agent-arg`, `--questions A-B` and `--out`
(default `runs/benchmarks/`). `score` takes run folders and `--judge-model` (default `sonnet`).

The entry point is `drones-benchmark = "drones.sim.benchmark:main"`. Metrics are pure numpy in
`drones/sim/metrics.py`, so they are tested without the simulator.

## `run`

### Which questions, in which order

- `--benchmark`: one or more of `hm-eqa mt-hm3d express-bench a-eqa indoor-uav`, or `all`, which
  is all five in that order. Each benchmark is its own run folder and resumes on its own; a stop in
  one stops the whole command.
- Every question of the benchmark, except `indoor-uav`, which uses only its `test_seen` and
  `test_unseen` trajectories (the rest is training data). Its split is the start of
  `Question.category`.
- `--questions A-B` keeps those question numbers (the benchmark's own, 1-based), for a trial.
- Questions are grouped by scene (`eqa.by_scene`). A scene is downloaded when it is first needed
  (`scenes.download`, `eqa.prepare`) and loaded once, as one `SceneView`, for all its questions.
  `indoor-uav` scenes are found through `eqa.busiest`/`prepare`, never `eqa.load` alone (see
  AGENTS.md).

### Start pose

As in `drones-render-agent --benchmark`: `eqa.start_pose(question, view.scene.origin,
eye_height)`, eye height 1.0 m by default.

### Step budget

The budget B is counted in **decisions**, one per model call for the VLM pilot. An agent's
`chunk_size` (below) turns it into a step limit: `max_steps = B * chunk_size`.

| Benchmark | B |
| --- | --- |
| hm-eqa, mt-hm3d, express-bench, a-eqa | `int(sqrt(A) * 3)`, Explore-EQA's `max_step_room_size_ratio` |
| indoor-uav | `ceil(2 * L_R / reach)` |

- `A` is the floor area in m²: the x-y bounding box of the scan's vertices between 0.1 m and
  2.0 m above the start's floor. Explore-EQA uses habitat's navmesh bounds on the floor, which we
  do not have; this is the stand-in, and the report says so.
- `L_R` is the length of the reference path; `reach` is the furthest one decision can fly, which
  the runner reads from the agent as `reach` (m) if it has one, else 0.4 m (16 discrete
  `forward` at the default limits). The VLM agent reports `chunk_size * MAX_MANUAL_SPEED * STEP`.

### An episode

`agents.episode(view, agent, start, question, max_steps)`. It ends when the agent returns None
(the VLM pilot's `done`, or three failed chunks) or at `max_steps`. **At the budget**, for an EQA
benchmark, an agent with `conclude` is called once with the last observation and its answer is
taken. The stop reason is recorded: `done`, `failed` (the pilot gave up), `budget`.

### What is written

```
runs/benchmarks/<benchmark>-<agent>/          # agent: --agent with ':' as '.'
  config.json                                 # the run's arguments; a resume must match
  results.jsonl                               # one line per finished question
  episodes/<number>/
    trajectory.npz                            # pos (N, 3), yaw (N,) at every step
    calls.jsonl                               # one line per model call
    calls/<k>.png                             # the frame of call k, exactly as the model saw it
```

A `results.jsonl` line: `benchmark, number, scene, category, question, choices, truth, answer,
stop, start ('benchmark' or 'origin'), decisions, budget, steps, chunk_size, path_length,
final_pos, final_yaw, goal,
reference_length, seconds, stats` (the pilot's first/retry/failed counts, if it has them).
`path_length` is the sum of the flown step lengths (straight segments).

A `calls.jsonl` line: `k, step, elapsed, altitude, pos, yaw, prompt, reply, attempt, valid, error,
chunk, seconds, final` (`final` true for a `conclude` call), and `image`, the PNG's path. This is
the distillation data format, documented in the README.

### Resume and stop

- On start, questions already in `results.jsonl` are skipped. `config.json` is written on the
  first start; a later start whose arguments differ (agent, agent args, camera, eye height) is
  refused with the difference, rather than mixing two configurations in one result file.
- A finished question is written to `results.jsonl` last, after its episode folder, with a flush,
  so a folder without a results line is unfinished.
- An exception from the agent (the Claude backend's RuntimeError at a usage limit, a SystemExit
  when the CLI is missing) or Ctrl-C: the unfinished episode's folder is deleted, the message is
  printed, and the command exits non-zero. Leftover unfinished folders are deleted at start too.
- Progress goes to stdout, one line per question: number, stop reason, decisions/budget, answer,
  truth, time.

## Agent attributes read by the runner

Read by attribute, like `answer` and `caption`, so `sim/` imports no agent:

| Attribute | Meaning | When missing |
| --- | --- | --- |
| `chunk_size` | steps per decision | 1 |
| `reach` | m one decision can fly at most | 0.4 |
| `calls` | the model calls of this episode, a list of dicts (above, with `image` an array) | no distillation data |
| `conclude(observation)` | a forced final answer at the budget; sets `answer` | the answer it has |

## Changes to `vlm/`

- `Pilot.calls`: reset by `reset`; every `backend.generate` appends a record with the step,
  elapsed time, altitude, prompt, raw reply, attempt (1 or 2), valid, error, the chunk as
  `model_dump()`, seconds, `final`, and the image array.
- `Pilot.conclude(image, altitude)`: one more model call with `prompt.final_prompt(...)`. A reply
  that is not a valid chunk with `done` true and an answer is retried once, as `_ask` does; after
  that the answer stays as it was. Recorded in `calls` with `final` true.
- `prompt.final_prompt(space, question, choices, altitude, elapsed, size)`: `build_prompt`'s text
  with a last section saying this is the last look, to reply with `done` true and the answer. It
  shows the format only as the numbered-slot template; `tests/test_vlm_prompt.py` covers it.
- `VLMAgent`: `chunk_size` (`pilot.size`), `reach`, `calls` (the pilot's, with the pose added at
  each record's step), `conclude(observation)`.

## `score`

Reads `results.jsonl`, writes `scores.json` (per question) and `report.md` (totals, then per
category; with several folders, one table across them).

| Benchmark | Metric | Definition |
| --- | --- | --- |
| hm-eqa, mt-hm3d | SR | % of answers whose letter is the truth's |
| | Normalized steps | mean of decisions / B |
| express-bench | LLM-Score C* | mean of σ/5, × 100 |
| | E_path | mean of σ/5 · l/max(p, l), × 100; l the reference length, p the flown length |
| | d_T | mean distance from the final position to the goal, m |
| a-eqa | LLM-Match | mean of (σ − 1)/4, × 100 |
| indoor-uav | SR | % ending within 2 m of the goal |
| | NE | mean distance from the final position to the goal, m |
| | OSR | % passing within 2 m of the goal at any step |
| | nDTW | mean of exp(−DTW(R, P) / (L_R · 10)), positions only |

Every table also gives seconds per call, valid replies (first try / after a retry / failed) and the
share of questions that ran out of budget.

- **Letters.** The answer is read as a letter from "B", "B)", "B) red" or "(B)", or matched to an
  option's text; anything else is wrong.
- **Judge.** σ from 1 to 5, with OpenEQA's `mmbench.txt` prompt verbatim, asked through the
  `claude` CLI with no image (`ClaudeCodeBackend`'s flags, text only) and `--judge-model`
  (default sonnet). The score is parsed as OpenEQA's `parse_score` does (a bare digit, or after
  "Your mark:"). Judgements are cached in `judged.jsonl` keyed by (number, answer, judge model): a
  rerun judges only what is new, and another judge model re-judges everything. A judge failure
  stops `score` cleanly, as in `run`.
- **No answer** (None, or a failed episode): σ = 1, wrong for multiple choice; still counted.
- **Deviations from the papers**, printed in `report.md`: the judge is Claude, not GPT-4 or
  GPT-4o-mini; EXPRESS-Bench's grounding δ is not computed, so C* is reported and E_path uses
  δ = 1; distances are straight lines in the scan, not geodesics on a navmesh (no habitat-sim, and
  the drone flies through walls); a step is one model call, which flies at most `reach` (0.4 m),
  where Explore-EQA's step moves up to 3 m; A-EQA has no reference path here, so no E_path; the
  floor area A is a vertex bounding box, not the navmesh's.

## Errors

- No scene for a question (download failure): the run stops cleanly like any other failure, so it
  is retried next time rather than silently skipped.
- A question whose start pose is missing (some mt-hm3d rows) starts at the scene's open floor, as
  `eqa.start_pose` already does, and is flagged `start: origin` in its results line.

## Layering

`sim/benchmark.py` and `sim/metrics.py` import only `sim/` and the standard stack. The judge
needs the CLI call in `vlm/backend.py` (stdlib and numpy only); `sim/benchmark.py` imports it
inside the `score` function, so importing `drones.sim` never imports `drones.vlm`. AGENTS.md's
layering note records this.

## Testing

No test needs a scene, a model or the CLI.

- `tests/test_benchmark_metrics.py` (fast suite): letter parsing; C*, LLM-Match, E_path, d_T on
  hand-worked numbers; nDTW against a hand-computed DTW; SR/NE/OSR; normalized steps; the budget
  from a synthetic vertex box; judge-score parsing ("5", "Your mark: 3").
- `tests/test_benchmark.py` (sim): `run` on `test_agents.py`'s synthetic box with a scripted agent
  and a stub question list — results lines, trajectories, `calls.jsonl` and PNGs written; the
  budget stops the agent and `conclude` is called; an exception mid-episode deletes only that
  folder and exits non-zero, and a rerun resumes there; a changed config is refused; `all` runs in
  order; `score` with a fake judge, and its cache.
- `vlm` tests: `Pilot.calls` records (retries included); `conclude` valid, retried and failed;
  `final_prompt` holds no line that would pass as a reply; `VLMAgent`'s new attributes.

## Documentation

- README: "Benchmarking a pilot" — usage, each metric, the deviations, the output layout and the
  distillation data format.
- AGENTS.md: the entry point; the four optional agent attributes; that `drones-benchmark run` with
  a Claude agent spends the user's usage, so an agent asks before starting one; the layering note.
