# drones-benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `drones-benchmark run` flies any agent (the VLM pilot on Claude in practice) over one, several or all scanned-scene benchmarks, resumably, saving every model call for distillation; `drones-benchmark score` scores the runs the way FAST-EQA and IndoorUAV do.

**Architecture:** Pure-numpy metrics in `sim/metrics.py`; the flying loop in `sim/benchmark.py`, written against an injected `open_view(scene)` so it is tested in-process with a fake view; scoring and the Claude judge in `sim/scoring.py`. The VLM pilot gains a per-call record (`Pilot.calls`), a forced last answer (`Pilot.conclude`), and the agent attributes the runner reads by name (`chunk_size`, `reach`, `calls`, `stats`, `conclude`).

**Tech Stack:** Python 3, numpy, pydantic (vlm), PIL (PNG frames), MuJoCo via `SceneView`, the `claude` CLI through `ClaudeCodeBackend`, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-05-vlm-benchmark-design.md`

## Global Constraints

- Single quotes, 100 columns; `uv run ruff check src tests` stays clean. Do not reformat import blocks.
- Module docstrings explain why; comments record facts that were expensive to find. Match `src/drones/sim/eqa.py` and `src/drones/vlm/backend.py`.
- `sim/` never imports `drones.vlm` at module level. `scoring.py` imports `drones.vlm.backend` inside a function only.
- Any module under `drones.sim` imports CrazyFlow through `drones/sim/__init__.py`, so its tests start with `pytest.importorskip('crazyflow')`. (The spec put the metric tests in the fast suite; that is not possible under `drones.sim`, so they are simulator tests.)
- No test needs a scene file, a model, the network or the real `claude` CLI.
- Never run `drones-benchmark run` with a Claude agent without asking the user: it spends their subscription usage.
- Never `git add -A` / `git add .`; add files by name. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Benchmarks and order for `all`: `hm-eqa mt-hm3d express-bench a-eqa indoor-uav` (`eqa.BENCHMARKS`).
- Budget: `int(sqrt(A) * 3)` for EQA, A from vertices 0.1 to 2.0 m above the floor; `ceil(2 * L_R / reach)` for indoor-uav, reach default 0.4 m; never below 1.
- IndoorUAV: success radius 2.0 m, nDTW d_th = 10, `test_seen` and `test_unseen` only.
- Judge: OpenEQA's `mmbench.txt` prompt verbatim, Claude `sonnet` by default, cached in `judged.jsonl`.

## Review Focus

1. **A run killed mid-write** (power cut, `kill -9`) leaves a half-written last line in `results.jsonl` or an `episodes/<n>.part` folder; the next start must ignore the broken line, delete the leftovers and fly that question again — tested in Task 3 (`test_a_torn_results_line_is_flown_again`).
2. **Multiple-choice answers in other shapes**: `"b"`, `"(B)"`, `"B."`, the option's text, `"Blue"` (must not read as B) — tested in Task 1 (`test_choice_letter_reads_the_shapes_a_model_answers_in`).
3. **An agent with none of the optional attributes** (the built-in `look-around`) must still run and score, with an empty `calls.jsonl` — tested in Task 3 (`test_a_builtin_agent_runs_without_any_optional_attribute`).
4. **A scan with no vertices in the floor band** (or a tiny room) must still give a budget of at least 1 — tested in Task 1 (`test_budgets_are_never_below_one`).
5. **A judge reply that is not a mark** is retried once and then stops `score` cleanly, keeping every mark already cached — tested in Task 4 (`test_an_unreadable_judge_reply_stops_scoring_and_keeps_the_cache`).

---

### Task 1: Metrics

**Files:**
- Create: `src/drones/sim/metrics.py`
- Test: `tests/test_benchmark_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces (all in `drones.sim.metrics`):
  - constants `EQA_STEP_RATIO = 3`, `FLOOR_BAND = (0.1, 2.0)`, `DEFAULT_REACH = 0.4`, `SUCCESS_RADIUS = 2.0`, `NDTW_THRESHOLD = 10.0`
  - `floor_area(vertices, floor_z, band=FLOOR_BAND) -> float`
  - `eqa_budget(area: float) -> int`
  - `path_budget(reference_length: float, reach: float = DEFAULT_REACH) -> int`
  - `path_length(points) -> float`
  - `choice_letter(answer: str | None, choices: Sequence[str]) -> str | None`
  - `parse_mark(text: str) -> int` (raises ValueError)
  - `llm_score(marks) -> float`, `llm_match(marks) -> float`, `e_path(marks, reference_lengths, flown_lengths) -> float`
  - `dtw(a, b) -> float`, `ndtw(reference, path, threshold=NDTW_THRESHOLD) -> float`
  - `navigation(path, goal, radius=SUCCESS_RADIUS) -> dict` with keys `success: bool`, `ne: float`, `oracle: bool`

- [ ] **Step 1: Write the failing tests**

`tests/test_benchmark_metrics.py`:

```python
"""The benchmark metrics on hand-worked numbers: what FAST-EQA, EXPRESS-Bench, OpenEQA and
IndoorUAV define, as drones.sim.metrics computes them."""
import math

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones.sim import metrics

CHOICES = ('A) red', 'B) blue', 'C) green')


def test_floor_area_is_the_box_of_the_vertices_in_the_band_above_the_floor():
    v = np.array([[0.0, 0, 1.0], [4, 0, 1.0], [0, 3, 1.5], [9, 9, 0.05], [9, 9, 2.5]])
    # 0.05 m and 2.5 m above the floor are outside 0.1 to 2.0, so only the first three count.
    assert metrics.floor_area(v, floor_z=0.0) == pytest.approx(12.0)
    assert metrics.floor_area(v + [0, 0, 5], floor_z=5.0) == pytest.approx(12.0)


def test_the_eqa_budget_is_explore_eqas():
    assert metrics.eqa_budget(16.0) == 12            # int(4 * 3)
    assert metrics.eqa_budget(50.0) == int(math.sqrt(50) * 3)


def test_the_path_budget_is_twice_the_reference_in_reaches():
    assert metrics.path_budget(2.0, reach=0.4) == 10
    assert metrics.path_budget(2.1, reach=0.4) == 11


def test_budgets_are_never_below_one():
    assert metrics.floor_area(np.zeros((0, 3)), 0.0) == 0.0
    assert metrics.eqa_budget(0.0) == 1
    assert metrics.path_budget(0.0) == 1


def test_path_length_sums_the_straight_segments():
    assert metrics.path_length([[0, 0, 0], [3, 4, 0], [3, 4, 2]]) == pytest.approx(7.0)
    assert metrics.path_length([[1, 1, 1]]) == 0.0
    assert metrics.path_length(np.zeros((0, 3))) == 0.0


@pytest.mark.parametrize('answer, letter', [
    ('B', 'B'), ('b', 'B'), ('B)', 'B'), ('(B)', 'B'), ('B.', 'B'), ('B) blue', 'B'),
    ('blue', 'B'), ('Blue.', 'B'), ('  C  ', 'C'),
    ('Blue sofa', None), ('D', None), ('', None), (None, None), ('I think B', None),
])
def test_choice_letter_reads_the_shapes_a_model_answers_in(answer, letter):
    assert metrics.choice_letter(answer, CHOICES) == letter


def test_the_truth_reads_as_its_own_letter():
    assert metrics.choice_letter('B) blue', CHOICES) == 'B'


@pytest.mark.parametrize('text, mark', [('5', 5), (' 3\n', 3), ('Your mark: 4', 4),
                                        ('Thinking.\nYour mark: 2\nBecause.', 2)])
def test_parse_mark_reads_openeqa_replies(text, mark):
    assert metrics.parse_mark(text) == mark


@pytest.mark.parametrize('text', ['six', 'Your mark: 7', '0', ''])
def test_parse_mark_refuses_anything_else(text):
    with pytest.raises(ValueError):
        metrics.parse_mark(text)


def test_the_llm_scores():
    marks = [5, 3, 1]
    assert metrics.llm_score(marks) == pytest.approx(100 * (1 + 0.6 + 0.2) / 3)     # C*
    assert metrics.llm_match(marks) == pytest.approx(100 * (1 + 0.5 + 0) / 3)       # OpenEQA
    assert math.isnan(metrics.llm_score([]))


def test_e_path_weights_each_mark_by_path_efficiency():
    # 5 over a path twice the reference: 1 * 0.5; 3 over a shorter path: 0.6 * 1.
    value = metrics.e_path([5, 3], reference_lengths=[2.0, 4.0], flown_lengths=[4.0, 1.0])
    assert value == pytest.approx(100 * (0.5 + 0.6) / 2)
    assert metrics.e_path([5], [0.0], [0.0]) == pytest.approx(100.0)   # stayed, needed to


def test_dtw_on_a_hand_worked_pair():
    a = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    b = np.array([[0.0, 0, 0], [2, 0, 0]])
    # (a0,b0)=0, (a1,b0)=1 or (a1,b1)=1, (a2,b1)=0: the cheapest warping costs 1.
    assert metrics.dtw(a, b) == pytest.approx(1.0)
    assert metrics.dtw(a, a) == 0.0


def test_ndtw_is_one_on_the_reference_and_falls_with_distance():
    ref = np.array([[0.0, 0, 1], [1, 0, 1], [2, 0, 1]])
    assert metrics.ndtw(ref, ref) == pytest.approx(1.0)
    off = ref + [0, 3, 0]
    # Three matched points 3 m off: DTW 9, over 3 points x d_th 10.
    assert metrics.ndtw(ref, off) == pytest.approx(math.exp(-9 / 30))


def test_navigation_success_error_and_oracle():
    path = np.array([[0.0, 0, 1], [5, 0, 1], [10, 0, 1]])
    goal = np.array([5.5, 0, 1])
    assert metrics.navigation(path, goal) == {'success': False, 'ne': pytest.approx(4.5),
                                              'oracle': True}
    assert metrics.navigation(path, np.array([9.0, 0, 1]))['success'] is True
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run --extra sim pytest tests/test_benchmark_metrics.py -q`
Expected: collection error, `ImportError: cannot import name 'metrics'`.

- [ ] **Step 3: Implement `src/drones/sim/metrics.py`**

```python
"""The numbers a benchmark run is scored by, as the papers that define them compute them.

FAST-EQA (arXiv 2602.15813, Table 1) scores HM-EQA and MT-HM3D by success rate and normalized
steps, EXPRESS-Bench by LLM-Score and E_path, and OpenEQA's A-EQA by LLM-Match. It defines none
of these itself: the step budget is Explore-EQA's (`int(sqrt(scene area) * 3)`, its
`max_step_room_size_ratio`), C* and E_path are EXPRESS-Bench's (arXiv 2503.11117), LLM-Match
is OpenEQA's. IndoorUAV (arXiv 2512.19024) scores its long flights, the VLN set, by SR within
2 m, NE, OSR and nDTW with d_th = 10.

Where this differs from them, and why, is in drones.sim.scoring.DEVIATIONS: distances here are
straight lines in the scan, not geodesics on a navmesh, and the floor area is a vertex box.
"""
import math
import re

import numpy as np

EQA_STEP_RATIO = 3          # Explore-EQA's max_step_room_size_ratio
FLOOR_BAND = (0.1, 2.0)     # m above the floor: the vertices whose box stands in for the navmesh
DEFAULT_REACH = 0.4         # m one decision flies at most: 16 discrete forwards at 2.5 cm
SUCCESS_RADIUS = 2.0        # m: IndoorUAV's VLN success
NDTW_THRESHOLD = 10.0       # IndoorUAV's d_th for VLN

# A letter alone, or opening the answer: B, (B), B), B., B) blue. Not the B of "Blue".
_LETTER = re.compile(r'^\(?([A-H])\)?(?:[).:,]|\s|$)')


def floor_area(vertices, floor_z, band=FLOOR_BAND):
    """m² of the x-y box around the scan's `vertices` between band[0] and band[1] above
    `floor_z`. Explore-EQA takes this from habitat's navmesh bounds on the floor; there is no
    navmesh here, and vertices a little above the floor are the walls and furniture that bound
    the same space."""
    v = np.asarray(vertices, float).reshape(-1, 3)
    near = v[(v[:, 2] > floor_z + band[0]) & (v[:, 2] < floor_z + band[1])]
    if not len(near):
        return 0.0
    return float(np.ptp(near[:, 0]) * np.ptp(near[:, 1]))


def eqa_budget(area):
    """Explore-EQA's step budget for a floor of `area` m², at least 1."""
    return max(1, int(math.sqrt(area) * EQA_STEP_RATIO))


def path_budget(reference_length, reach=DEFAULT_REACH):
    """Decisions to fly twice a reference path of `reference_length` m, `reach` m each."""
    return max(1, math.ceil(2 * reference_length / reach))


def path_length(points):
    """m along `points` (N, 3), straight from each to the next."""
    p = np.asarray(points, float).reshape(-1, 3)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def choice_letter(answer, choices):
    """The letter of the option `answer` picks among `choices` ('A) red', ...), or None. A
    letter alone may be lower case; otherwise it must open the answer in upper case, so that
    "Blue sofa" is not read as B. An option's text alone also picks it."""
    if answer is None:
        return None
    text = str(answer).strip()
    letters = [c.split(')', 1)[0].strip() for c in choices]
    if len(text) == 1:
        text = text.upper()
    match = _LETTER.match(text)
    if match and match.group(1) in letters:
        return match.group(1)
    options = {c.split(')', 1)[1].strip().lower(): c.split(')', 1)[0].strip() for c in choices}
    return options.get(text.lower().rstrip('.'))


def parse_mark(text):
    """The judge's mark, 1 to 5, read as OpenEQA's parse_score reads it: the whole reply, or
    the number after "Your mark:"."""
    text = text.strip()
    found = text if text.isdigit() else None
    if found is None:
        match = re.search(r'Your mark:\s*(\d+)', text)
        found = match.group(1) if match else None
    if found is None or not 1 <= int(found) <= 5:
        raise ValueError(f'the judge gave no mark from 1 to 5: {text[:200]!r}')
    return int(found)


def _mean(values):
    values = np.asarray(list(values), float)
    return float(values.mean()) if len(values) else math.nan


def llm_score(marks):
    """EXPRESS-Bench's C*: the mean of mark / 5, in %."""
    return 100 * _mean(np.asarray(marks, float) / 5)


def llm_match(marks):
    """OpenEQA's LLM-Match: the mean of (mark - 1) / 4, in %."""
    return 100 * _mean((np.asarray(marks, float) - 1) / 4)


def _efficiency(reference, flown):
    longest = max(reference, flown)
    return 1.0 if longest == 0 else reference / longest


def e_path(marks, reference_lengths, flown_lengths):
    """EXPRESS-Bench's E_path with its grounding term at 1: mark / 5 x l / max(p, l), in %."""
    return 100 * _mean(m / 5 * _efficiency(l, p)
                       for m, l, p in zip(marks, reference_lengths, flown_lengths, strict=True))


def dtw(a, b):
    """Dynamic time warping between point sequences `a` (N, 3) and `b` (M, 3), Euclidean."""
    a, b = np.asarray(a, float).reshape(-1, 3), np.asarray(b, float).reshape(-1, 3)
    cost = np.linalg.norm(a[:, None] - b[None], axis=2)
    acc = np.full((len(a) + 1, len(b) + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        # The diagonal and the step down are known for the whole row; the step along it is not.
        best = np.minimum(acc[i - 1, :-1], acc[i - 1, 1:]) + cost[i - 1]
        row = acc[i]
        for j in range(1, len(b) + 1):
            row[j] = min(best[j - 1], row[j - 1] + cost[i - 1, j - 1])
    return float(acc[-1, -1])


def ndtw(reference, path, threshold=NDTW_THRESHOLD):
    """exp(-DTW / (|R| d_th)), |R| the reference's point count, as nDTW was defined for VLN
    (IndoorUAV writes it L_R). Positions only: the yaw term is IndoorUAV's VLA set's."""
    reference = np.asarray(reference, float).reshape(-1, 3)
    if not len(reference):
        return math.nan
    return math.exp(-dtw(reference, path) / (len(reference) * threshold))


def navigation(path, goal, radius=SUCCESS_RADIUS):
    """IndoorUAV's SR, NE and OSR for one flight `path` (N, 3) to `goal`."""
    distance = np.linalg.norm(np.asarray(path, float).reshape(-1, 3) - goal, axis=1)
    return {'success': bool(distance[-1] <= radius), 'ne': float(distance[-1]),
            'oracle': bool((distance <= radius).any())}
```

- [ ] **Step 4: Run the tests**

Run: `uv run --extra sim pytest tests/test_benchmark_metrics.py -q && uv run ruff check src tests`
Expected: all pass; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/drones/sim/metrics.py tests/test_benchmark_metrics.py
git commit -m "feat(sim): benchmark metrics as FAST-EQA, EXPRESS-Bench, OpenEQA and IndoorUAV define them

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The pilot records its calls and can be made to answer

**Files:**
- Modify: `src/drones/vlm/prompt.py` (add `FINAL`, `final_prompt` after `build_prompt`)
- Modify: `src/drones/vlm/pilot.py` (`reset`, `_plan`, `_ask`, new `conclude`)
- Modify: `src/drones/vlm/agent.py` (`VLMAgent`: `act`, new `conclude`, `_tag`, properties `chunk_size`, `reach`, `calls`, `stats`; module docstring)
- Test: `tests/test_vlm_prompt.py`, `tests/test_vlm_pilot.py`, `tests/test_vlm_agent.py`

**Interfaces:**
- Consumes: `actions.STEP`, `actions.parse_chunk`, `prompt.build_prompt`.
- Produces:
  - `prompt.final_prompt(space, question=None, choices=(), altitude=0.0, elapsed=0.0, size=CHUNK) -> str`
  - `Pilot.calls: list[dict]`, each with keys `step:int, elapsed:float, altitude:float, prompt:str, reply:str, attempt:int, valid:bool, error:str|None, chunk:dict|None, seconds:float, final:bool, image`
  - `Pilot.conclude(image, altitude) -> str | None`
  - `VLMAgent.chunk_size -> int`, `VLMAgent.reach -> float` (m), `VLMAgent.calls -> list[dict]` (the pilot's, each also with `pos: list[float]`, `yaw: float`), `VLMAgent.stats -> dict`, `VLMAgent.conclude(observation) -> str | None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vlm_prompt.py` (it already imports `pytest`, `json`-free helpers; check its imports and add `from drones.vlm.prompt import build_prompt, final_prompt` and `from drones.vlm.actions import parse_chunk` if missing):

```python
@pytest.mark.parametrize('space', ['discrete', 'continuous'])
def test_the_last_look_asks_for_an_answer_and_still_shows_no_reply(space):
    from drones.vlm.actions import parse_chunk
    from drones.vlm.prompt import build_prompt, final_prompt

    text = final_prompt(space, 'Where is the sofa?', ('A) kitchen', 'B) lounge'), 1.0, 30.0)
    assert text.startswith(build_prompt(space, 'Where is the sofa?',
                                        ('A) kitchen', 'B) lounge'), 1.0, 30.0))
    assert 'LAST LOOK' in text and 'done' in text
    for line in text.splitlines():
        with pytest.raises(ValueError):
            parse_chunk(line, space)
```

Append to `tests/test_vlm_pilot.py`:

```python
def test_every_model_call_is_recorded_with_what_the_model_saw_and_said():
    backend = FakeBackend(BAD, FORWARD, DONE)
    pilot = started(backend)
    fly(pilot, CHUNK + 1, altitude=1.25)
    calls = pilot.calls
    assert [c['attempt'] for c in calls] == [1, 2, 1]
    assert [c['valid'] for c in calls] == [False, True, True]
    assert [c['step'] for c in calls] == [0, 0, CHUNK]
    assert [c['image'] for c in calls] == [0, 0, CHUNK]
    assert [c['prompt'] for c in calls] == [prompt for prompt, _ in backend.calls]
    assert calls[0]['reply'] == BAD and 'exactly' in calls[0]['error']
    assert calls[0]['chunk'] is None
    assert calls[1]['chunk']['actions'] == ['forward'] * CHUNK
    assert calls[2]['chunk']['done'] is True
    assert calls[2]['elapsed'] == pytest.approx(1.0)
    assert {c['altitude'] for c in calls} == {1.25}
    assert not any(c['final'] for c in calls)
    assert all(c['seconds'] >= 0 for c in calls)


def test_reset_forgets_the_calls():
    pilot = started(FakeBackend(DONE))
    fly(pilot, 1)
    pilot.reset('Again.')
    assert pilot.calls == []


def test_conclude_asks_once_more_and_takes_the_answer():
    backend = FakeBackend(FORWARD, DONE)
    pilot = started(backend)
    fly(pilot, CHUNK)
    assert pilot.conclude('last', 1.0) == 'B'
    assert pilot.answer == 'B'
    final = pilot.calls[-1]
    assert final['final'] is True and final['image'] == 'last'
    assert 'LAST LOOK' in final['prompt']
    assert pilot.step(0, 1.0) is None               # the episode is over


def test_conclude_retries_a_reply_that_does_not_finish():
    pilot = started(FakeBackend(FORWARD, FORWARD, DONE))
    fly(pilot, CHUNK)
    assert pilot.conclude('last', 1.0) == 'B'
    assert [c['attempt'] for c in pilot.calls if c['final']] == [1, 2]
    assert 'last look' in pilot.calls[-2]['error']


def test_conclude_that_fails_keeps_the_answer_it_had():
    early = json.dumps({'actions': ['forward'] * CHUNK, 'answer': 'C'})
    pilot = started(FakeBackend(early, BAD, BAD))
    fly(pilot, CHUNK)
    assert pilot.conclude('last', 1.0) == 'C'
    assert pilot.stats['failed'] == 1
```

Append to `tests/test_vlm_agent.py`:

```python
def test_the_agent_tells_a_benchmark_its_chunk_reach_calls_and_stats():
    agent = vlm_agent.make(action_space='discrete', chunk=8,
                           backend=FakeBackend(json.dumps({'actions': ['forward'] * 8}), DONE))
    assert agent.chunk_size == 8
    assert agent.reach == pytest.approx(8 * config.MAX_MANUAL_SPEED / 16)
    start = agents.Pose(np.array([1.0, 2.0, 1.0]), yaw=0.0)
    agent.reset(None, start)
    observation = agents.Observation(np.zeros((2, 2, 3), np.uint8), start, 0, None)
    agent.act(observation)
    (call,) = agent.calls
    assert call['pos'] == [1.0, 2.0, 1.0] and call['yaw'] == 0.0
    assert agent.stats == {'first': 1, 'retry': 0, 'failed': 0}


def test_the_agent_concludes_from_the_last_observation():
    agent = vlm_agent.make(action_space='discrete', backend=FakeBackend(DONE))
    start = agents.Pose(np.array([0.0, 0.0, 1.5]))
    agent.reset(None, start)
    last = agents.Observation(np.zeros((2, 2, 3), np.uint8), start, 40, None)
    assert agent.conclude(last) == 'B'
    assert agent.calls[-1]['final'] is True and agent.calls[-1]['pos'] == [0.0, 0.0, 1.5]
    assert '1.00 m' in agent.calls[-1]['prompt']       # 1.5 m start, 1.0 m start_altitude
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --extra sim pytest tests/test_vlm_prompt.py tests/test_vlm_pilot.py tests/test_vlm_agent.py -q`
Expected: the new tests fail (`ImportError: final_prompt`, `AttributeError: 'Pilot' object has no attribute 'calls'`, `... 'chunk_size'`).

- [ ] **Step 3: Add `final_prompt` to `src/drones/vlm/prompt.py`**

After `build_prompt`:

```python
# The last section of the prompt when a benchmark's budget has run out: the model must answer
# now. The reply format is the template shown above it, so this holds no JSON of its own.
FINAL = ('THIS IS YOUR LAST LOOK. The flight is over: answer now, from this image and the task. '
         'Reply in the second form above, with "done" set to true and your answer.')


def final_prompt(space, question=None, choices=(), altitude=0.0, elapsed=0.0, size=CHUNK):
    """`build_prompt`'s prompt with a last section that asks for the answer now."""
    return f'{build_prompt(space, question, choices, altitude, elapsed, size)}\n\n{FINAL}'
```

- [ ] **Step 4: Record calls and add `conclude` in `src/drones/vlm/pilot.py`**

Change the import to `from drones.vlm.prompt import build_prompt, final_prompt, retry_prompt`.

In the module docstring, after the paragraph about failures, add:

```
Every backend call is kept in `calls`, retries and failures included, with the exact prompt,
image and reply: a benchmark saves them as data to distil a smaller pilot from. `conclude` is
for a benchmark whose budget ran out: one more look, and the model must answer.
```

In `reset`, after `self.stats = ...`, add:

```python
        self.calls = []         # one record per backend call, for a benchmark to save
```

In `_plan`, change `chunk = self._ask(prompt, image)` to `chunk = self._ask(prompt, image, altitude)`.

Replace `_ask` with:

```python
    def _ask(self, prompt, image, altitude, final=False):
        """A valid chunk for `image`, or None after ATTEMPTS invalid replies. A `final` reply
        must also set done and give an answer."""
        asked = prompt
        for attempt in range(ATTEMPTS):
            started = time.perf_counter()
            reply = self.backend.generate(asked, image)
            seconds = time.perf_counter() - started
            self.seconds += seconds
            self.queries += 1
            record = {'step': self._steps, 'elapsed': self._steps * STEP, 'altitude': altitude,
                      'prompt': asked, 'reply': reply, 'attempt': attempt + 1, 'valid': False,
                      'error': None, 'chunk': None, 'seconds': seconds, 'final': final,
                      'image': image}
            self.calls.append(record)
            try:
                chunk = parse_chunk(reply, self.space, self.size)
                if final and not (chunk.done and chunk.answer is not None):
                    raise ValueError('this was your last look: set "done" to true and give '
                                     'your answer')
            except ValueError as exc:
                self.error = record['error'] = str(exc)
                logger.warning('reply %d rejected: %s', self.queries, self.error)
                asked = retry_prompt(prompt, reply, self.error)
                continue
            record['valid'], record['chunk'] = True, chunk.model_dump()
            self.stats['retry' if attempt else 'first'] += 1
            return chunk
        return None

    def conclude(self, image, altitude):
        """The answer after one last look at `image`, with the drone at `altitude` m, for an
        episode whose budget ran out. A model that gives none keeps the answer it had. The
        episode is over afterwards."""
        prompt = final_prompt(self.space, self.question, self.choices, altitude,
                              self._steps * STEP, self.size)
        chunk = self._ask(prompt, image, altitude, final=True)
        if chunk is None:
            self.stats['failed'] += 1
        else:
            self.answer, self.error = chunk.answer, None
        self.chunk, self.played = chunk, 0
        self._actions.clear()
        self._over = True
        return self.answer
```

- [ ] **Step 5: Agent attributes in `src/drones/vlm/agent.py`**

Change the import `from drones.vlm.actions import CHUNK, STEP, ContinuousAction, chunk_size, json_schema` (unchanged names; `STEP` and `config` are already imported).

In the module docstring, before "Each model call is reported", add:

```
For drones-benchmark the agent also has `chunk_size`, `reach` (m one chunk can fly at most),
`calls` (the pilot's records, with the pose of each), `stats` and `conclude`, which asks for
an answer when the benchmark's budget has run out. drones.sim reads them by name.
```

Replace `VLMAgent.act` with:

```python
    def act(self, observation):
        pose = observation.pose
        altitude = float(pose.pos[2]) - self._floor
        pilot = self.pilot
        stats, seconds, before = dict(pilot.stats), pilot.seconds, len(pilot.calls)
        command = pilot.step(observation.image, altitude)
        self._tag(before, pose)
        if pilot.stats != stats:
            self._report(observation.step, stats, pilot.seconds - seconds, command is None)
        if command is None:
            self._over = True
            return None
        return integrate(pose, altitude, command)

    def conclude(self, observation):
        """The pilot's answer after one last look at `observation`, when a benchmark's budget
        has run out."""
        pilot = self.pilot
        stats, seconds, before = dict(pilot.stats), pilot.seconds, len(pilot.calls)
        answer = pilot.conclude(observation.image, float(observation.pose.pos[2]) - self._floor)
        self._tag(before, observation.pose)
        self._over = True
        self._report(observation.step, stats, pilot.seconds - seconds, True)
        return answer

    def _tag(self, since, pose):
        """Put `pose` on the pilot's call records from index `since` on."""
        for call in self.pilot.calls[since:]:
            call['pos'] = [float(x) for x in pose.pos]
            call['yaw'] = float(pose.yaw)
```

Add properties after `error`:

```python
    @property
    def chunk_size(self):
        return self.pilot.size

    @property
    def reach(self):
        """m one chunk flies at most: all of it forward at full speed."""
        return self.pilot.size * STEP * config.MAX_MANUAL_SPEED

    @property
    def calls(self):
        return self.pilot.calls

    @property
    def stats(self):
        return dict(self.pilot.stats)
```

- [ ] **Step 6: Run the tests**

Run: `uv run --extra sim pytest tests/test_vlm_prompt.py tests/test_vlm_pilot.py tests/test_vlm_agent.py tests/test_architecture.py -q && uv run ruff check src tests`
Expected: all pass (old and new); ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/drones/vlm/prompt.py src/drones/vlm/pilot.py src/drones/vlm/agent.py tests/test_vlm_prompt.py tests/test_vlm_pilot.py tests/test_vlm_agent.py
git commit -m "feat(vlm): record every model call and answer on demand, for benchmarks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Flying a benchmark (`sim/benchmark.py`, the run loop)

**Files:**
- Create: `src/drones/sim/benchmark.py`
- Modify: `src/drones/sim/eqa.py` (add `split_scenes`)
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `metrics.*` (Task 1); `agents.episode`, `agents.Pose`; `eqa.Question`, `eqa.start_pose`, `eqa.path_poses`, `eqa.by_scene`, `eqa.EYE_HEIGHT`, `eqa.INDOOR_UAV_SPLITS`. Agents' optional attributes from Task 2.
- Produces (in `drones.sim.benchmark`):
  - `RUNS_DIR = Path('runs/benchmarks')`, `EQA = ('hm-eqa', 'mt-hm3d', 'express-bench', 'a-eqa')`, `TEST_SPLITS = ('test seen', 'test unseen')`, `COMPARED = ('benchmark', 'agent', 'agent_args', 'camera', 'eye_height')`
  - `class Stopped(Exception)` — the message says why and where
  - `select(benchmark, questions, numbers=None) -> list[eqa.Question]`; `numbers` is `(first, last)` inclusive
  - `run(benchmark, questions, agent, folder, open_view, config, eye_height=eqa.EYE_HEIGHT, log=print) -> None` — raises `Stopped`
  - `finished(results_path) -> dict[int, dict]`
  - Files: `folder/config.json`, `folder/results.jsonl`, `folder/episodes/<n>/{trajectory.npz, calls.jsonl, calls/<k>.png}`. `trajectory.npz` arrays: `pos (N,3)`, `yaw (N,)`, `reference (M,3)` (lifted reference path, empty when none).
  - Results line keys: `benchmark, number, scene, category, question, choices, truth, answer, stop, start, decisions, budget, steps, chunk_size, path_length, final_pos, final_yaw, goal, reference_length, seconds, model_calls, model_seconds, stats`
- `eqa.split_scenes(splits, dest=SCENES_DIR) -> list[str]`

- [ ] **Step 1: Write the failing tests**

`tests/test_benchmark.py`:

```python
"""drones-benchmark's run loop on a fake view: what it writes, where it stops, how it resumes.
No scene, model or renderer: `open_view` hands back a stand-in that numbers its frames."""
import json
from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones.sim import agents, benchmark, eqa

CHOICES = ('A) red', 'B) blue')


class Room:
    """A scenes.Scene stand-in: a 4 x 4 m floor, walls 1 m up, its origin at the world's."""
    origin = np.zeros(3)
    vertices = np.array([[x, y, 1.0] for x in (-2.0, 2.0) for y in (-2.0, 2.0)])


class FakeView:
    def __init__(self, scene):
        self.scene = Room()
        self.opened = scene

    def render_matrix(self, pos, rotation):
        return np.full((4, 6, 3), 7, np.uint8)


@contextmanager
def fake_open(scene):
    yield FakeView(scene)


class Scripted:
    """Asks every `chunk_size` steps (recording a call), flies 0.1 m ahead a step, and says
    'B' and stops at `stop_after`, if given. Raises at question `fail_at`."""
    chunk_size, reach = 4, 0.4

    def __init__(self, stop_after=None, fail_at=None):
        self.stop_after, self.fail_at = stop_after, fail_at
        self.questions = []

    def reset(self, question, pose):
        self.questions.append(question.number)
        self.calls, self.answer, self.error, self.concluded = [], None, None, False

    def act(self, observation):
        if observation.question.number == self.fail_at:
            raise RuntimeError('claude failed: usage limit reached')
        if observation.step % self.chunk_size == 0:
            self.calls.append({'step': observation.step, 'prompt': 'fly', 'reply': '{}',
                               'seconds': 0.5, 'image': observation.image,
                               'pos': [float(x) for x in observation.pose.pos]})
        if self.stop_after is not None and observation.step >= self.stop_after:
            self.answer = 'B'
            return None
        return agents.Pose(observation.pose.pos + [0.1, 0.0, 0.0], observation.pose.yaw)

    def conclude(self, observation):
        self.concluded = True
        self.answer = 'A'
        return 'A'


def question(number, bench='hm-eqa', **kwargs):
    fields = dict(benchmark=bench, number=number, scene='room', text='What colour?',
                  answer='B) blue', category='colour', choices=CHOICES,
                  start=np.zeros(3), yaw=0.0)
    fields.update(kwargs)
    return eqa.Question(**fields)


def fly(tmp_path, agent, questions, bench='hm-eqa', config=None):
    benchmark.run(bench, questions, agent, tmp_path, fake_open,
                  config or {'benchmark': bench, 'agent': 'scripted'}, log=lambda *a: None)
    return [json.loads(line) for line in (tmp_path / 'results.jsonl').read_text().splitlines()]


def test_a_finished_question_writes_its_line_trajectory_and_calls(tmp_path):
    (row,) = fly(tmp_path, Scripted(stop_after=5), [question(1)])
    assert row['number'] == 1 and row['answer'] == 'B' and row['truth'] == 'B) blue'
    assert row['stop'] == 'done' and row['start'] == 'benchmark'
    assert row['steps'] == 5 and row['decisions'] == 2 and row['chunk_size'] == 4
    assert row['budget'] == 12                            # int(sqrt(16) * 3)
    assert row['path_length'] == pytest.approx(0.5)
    assert row['final_pos'] == pytest.approx([0.5, 0.0, 1.0])
    assert row['model_calls'] == 2 and row['model_seconds'] == pytest.approx(1.0)
    episode = tmp_path / 'episodes' / '1'
    trajectory = np.load(episode / 'trajectory.npz')
    assert trajectory['pos'].shape == (6, 3) and trajectory['yaw'].shape == (6,)
    calls = [json.loads(line) for line in (episode / 'calls.jsonl').read_text().splitlines()]
    assert [c['k'] for c in calls] == [0, 1] and calls[1]['step'] == 4
    assert calls[0]['image'] == 'calls/000.png' and (episode / 'calls/000.png').is_file()
    assert json.loads((tmp_path / 'config.json').read_text())['agent'] == 'scripted'


def test_the_budget_stops_the_agent_and_asks_it_to_conclude(tmp_path):
    agent = Scripted()
    (row,) = fly(tmp_path, agent, [question(1)])
    assert agent.concluded
    assert row['stop'] == 'budget' and row['decisions'] == 12 and row['steps'] == 48
    assert row['answer'] == 'A'


def test_a_navigation_flight_is_not_asked_for_an_answer(tmp_path):
    path = np.array([[0.0, 0, 1], [0.4, 0, 1]])
    q = question(1, 'indoor-uav', start=path[0], path=path, goal=path[-1],
                 category='traj_1, test seen, easy', choices=())
    agent = Scripted()
    (row,) = fly(tmp_path, agent, [q], bench='indoor-uav')
    assert not agent.concluded
    assert row['budget'] == 2                             # ceil(2 * 0.4 / 0.4)
    assert row['reference_length'] == pytest.approx(0.4)
    assert row['goal'] == pytest.approx([0.4, 0.0, 1.0])  # IndoorUAV is at flight height already
    reference = np.load(tmp_path / 'episodes' / '1' / 'trajectory.npz')['reference']
    assert reference.shape == (2, 3)


def test_a_failure_stops_the_run_keeps_what_finished_and_resumes_there(tmp_path):
    questions = [question(n) for n in (1, 2, 3)]
    with pytest.raises(benchmark.Stopped, match='question 2.*usage limit'):
        fly(tmp_path, Scripted(stop_after=1, fail_at=2), questions)
    assert [r['number'] for r in benchmark.finished(tmp_path / 'results.jsonl').values()] == [1]
    assert sorted(p.name for p in (tmp_path / 'episodes').iterdir()) == ['1']
    again = Scripted(stop_after=1)
    rows = fly(tmp_path, again, questions)
    assert again.questions == [2, 3]
    assert [r['number'] for r in rows] == [1, 2, 3]


def test_a_torn_results_line_is_flown_again(tmp_path):
    fly(tmp_path, Scripted(stop_after=1), [question(1)])
    with open(tmp_path / 'results.jsonl', 'a') as f:
        f.write('{"number": 2, "benchm')                  # killed mid-write
    (tmp_path / 'episodes' / '2.part').mkdir()
    again = Scripted(stop_after=1)
    rows = fly(tmp_path, again, [question(1), question(2)])
    assert again.questions == [2]
    assert [r['number'] for r in rows] == [1, 2]
    assert not (tmp_path / 'episodes' / '2.part').exists()


def test_other_settings_in_the_same_folder_are_refused(tmp_path):
    fly(tmp_path, Scripted(stop_after=1), [question(1)])
    with pytest.raises(benchmark.Stopped, match='agent'):
        fly(tmp_path, Scripted(stop_after=1), [question(2)],
            config={'benchmark': 'hm-eqa', 'agent': 'another'})


def test_a_builtin_agent_runs_without_any_optional_attribute(tmp_path):
    (row,) = fly(tmp_path, agents.LookAround(steps=3), [question(1)])
    assert row['stop'] == 'done' and row['answer'] is None and row['chunk_size'] == 1
    assert row['decisions'] == 4 and row['model_calls'] == 0 and row['stats'] is None
    assert (tmp_path / 'episodes' / '1' / 'calls.jsonl').read_text() == ''


def test_a_question_without_a_start_begins_at_the_open_floor(tmp_path):
    (row,) = fly(tmp_path, Scripted(stop_after=0), [question(1, 'a-eqa', start=None, yaw=None,
                                                            choices=())], bench='a-eqa')
    assert row['start'] == 'origin'
    assert row['final_pos'] == pytest.approx([0.0, 0.0, 1.0])


def test_select_keeps_the_numbers_and_indoor_uavs_test_splits():
    qs = [question(1), question(2), question(3)]
    assert [q.number for q in benchmark.select('hm-eqa', qs, (2, 3))] == [2, 3]
    nav = [question(1, 'indoor-uav', category='traj_1, test seen, easy'),
           question(2, 'indoor-uav', category='traj_2, train, hard'),
           question(3, 'indoor-uav', category='traj_3, test unseen, easy'),
           question(4, 'indoor-uav', category='traj_4, unsplit')]
    assert [q.number for q in benchmark.select('indoor-uav', nav)] == [1, 3]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --extra sim pytest tests/test_benchmark.py -q`
Expected: collection error, `ImportError: cannot import name 'benchmark'`.

- [ ] **Step 3: Add `eqa.split_scenes`**

In `src/drones/sim/eqa.py`, after `locate` (in the "all benchmarks" section):

```python
def split_scenes(splits, dest=SCENES_DIR):
    """Our ids of the scenes IndoorUAV's trajectories in `splits` (file names from
    INDOOR_UAV_SPLITS) fly through, for `prepare`: load() returns only prepared scenes."""
    from drones.sim.scenes import hm3d_index

    folder, ids, index = fetch('indoor-uav', dest), hm3d_index(dest), indoor_uav_index(dest)
    keys = set()
    for split in splits:
        for row in csv.DictReader(open(folder / split, newline='')):
            keys.add('/'.join(row['traj_path'].strip('/').split('/')[:2]))
    return sorted(_scene_id(key, ids) for key in keys if key in index)
```

- [ ] **Step 4: Implement `src/drones/sim/benchmark.py`**

```python
"""Fly an agent over every question of a benchmark, overnight, and keep what it did.

    uv run --extra sim drones-benchmark run --benchmark hm-eqa \\
        --agent drones.vlm.agent:make --agent-arg backend=claude-code
    uv run --extra sim drones-benchmark score runs/benchmarks/*

A run is meant to take nights: one benchmark in full is hundreds of questions and thousands of
model calls. So it resumes. Each finished question is one line of results.jsonl, written after
its episode folder, and a start skips those lines' questions and deletes anything else it
finds, so a run stopped anywhere -- a usage limit, Ctrl-C, a power cut -- flies the
interrupted question again from its start next time. It stops at the first failure rather than
skipping the question: a question that cannot be flown tonight (a usage limit, a scene that
would not download) is flown tomorrow, not dropped from the score.

The budget is counted in decisions, one per model call for the VLM pilot. EQA questions get
Explore-EQA's int(sqrt(floor area) * 3); IndoorUAV's flights twice their reference length in
`reach`es. An EQA agent that has `conclude` is asked for its answer when the budget runs out,
as Explore-EQA always answers at the end.

Every model call an agent records (`calls`, see drones.vlm.agent) is saved with the frame it
saw, as data to distil a smaller pilot from. Agents are read by attribute, so this imports
none: `chunk_size` (steps per decision, 1 without it), `reach` (m one decision flies at most),
`calls`, `stats`, `answer`, `error` and `conclude(observation)` are all optional.

Scoring is drones.sim.scoring; the metrics themselves are drones.sim.metrics.
"""
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from drones.sim import agents, eqa, metrics

RUNS_DIR = Path('runs/benchmarks')
EQA = ('hm-eqa', 'mt-hm3d', 'express-bench', 'a-eqa')
TEST_SPLITS = ('test seen', 'test unseen')      # how eqa writes IndoorUAV's split names
# Settings that change what a result means; a resume must keep them. --questions may change.
COMPARED = ('benchmark', 'agent', 'agent_args', 'camera', 'eye_height')


class Stopped(Exception):
    """The run stopped; the message says where and why. Everything finished is kept."""


def select(benchmark, questions, numbers=None):
    """`questions` numbered within `numbers` (first, last), and for indoor-uav only its test
    splits: its other trajectories are training data."""
    keep = [q for q in questions
            if numbers is None or numbers[0] <= q.number <= numbers[1]]
    if benchmark == 'indoor-uav':
        # eqa writes the category as 'traj_3, test seen, easy', or 'traj_3, unsplit'.
        keep = [q for q in keep if q.category.split(', ')[1:2] in ([s] for s in TEST_SPLITS)]
    return keep


def finished(results):
    """{number: results line} of the questions in `results`. A line cut off by a kill is not
    a finished question."""
    done = {}
    if not Path(results).exists():
        return done
    for line in Path(results).read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[row['number']] = row
    return done


def _plain(value):
    """`value` as JSON can hold it."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _check_config(folder, config):
    path = folder / 'config.json'
    config = _plain(config)
    if not path.exists():
        path.write_text(json.dumps(config, indent=2))
        return
    old = json.loads(path.read_text())
    changed = [f'{key}: {old.get(key)!r} then, {config.get(key)!r} now'
               for key in COMPARED if old.get(key) != config.get(key)]
    if changed:
        raise Stopped(f'{folder} was run with other settings ({"; ".join(changed)}). Pass '
                      '--name for a new run, or delete the folder to start again.')


def _clean(folder, done):
    """Delete the episode folders of unfinished questions, and fix a torn last line."""
    episodes = folder / 'episodes'
    for path in episodes.iterdir():
        if not path.name.isdigit() or int(path.name) not in done:
            shutil.rmtree(path)
    results = folder / 'results.jsonl'
    if results.exists():
        results.write_text(''.join(json.dumps(row) + '\n' for row in done.values()))


def _lift(question, eye_height):
    """IndoorUAV's paths are at flight height; habitat's are on the floor."""
    return 0.0 if question.benchmark == 'indoor-uav' else eye_height


def _reference(question, eye_height):
    """(M, 3) the question's reference path at the height the drone flies it; (0, 3) without."""
    poses = eqa.path_poses(question, eye_height)
    return np.array([p for p, _ in poses], float).reshape(-1, 3)


def _budget(question, view, reference, reach):
    if question.benchmark == 'indoor-uav':
        return metrics.path_budget(metrics.path_length(reference), reach)
    scene = view.scene
    floor = (question.start if question.start is not None else scene.origin)[2]
    # The scene's vertices are moved so its origin is the world's; benchmark poses are not.
    return metrics.eqa_budget(metrics.floor_area(scene.vertices + scene.origin, floor))


def _fly(view, agent, question, start, budget):
    """Fly one episode; (stop reason, decisions, poses, last observation)."""
    chunk = int(getattr(agent, 'chunk_size', 1))
    poses, last = [], None
    # Only poses are kept: the frames of a long episode would fill the memory.
    for observation in agents.episode(view, agent, start, question, budget * chunk):
        poses.append(observation.pose)
        last = observation
    if last.step == budget * chunk:
        stop = 'budget'
        if question.benchmark in EQA and hasattr(agent, 'conclude'):
            agent.conclude(last)
    else:
        stop = 'failed' if getattr(agent, 'error', None) else 'done'
    decisions = budget if stop == 'budget' else last.step // chunk + 1
    return stop, decisions, chunk, poses, last


def _write_episode(folder, number, pos, yaw, reference, calls):
    """The episode's folder, written aside and moved into place whole."""
    from PIL import Image

    part = folder / 'episodes' / f'{number}.part'
    shutil.rmtree(part, ignore_errors=True)
    (part / 'calls').mkdir(parents=True)
    np.savez_compressed(part / 'trajectory.npz', pos=pos, yaw=yaw, reference=reference)
    with open(part / 'calls.jsonl', 'w') as f:
        for k, call in enumerate(calls):
            row = {'k': k, **{key: _plain(v) for key, v in call.items() if key != 'image'}}
            image = call.get('image')
            if isinstance(image, np.ndarray) and image.ndim == 3:
                row['image'] = f'calls/{k:03d}.png'
                Image.fromarray(np.ascontiguousarray(image, np.uint8)).save(part / row['image'])
            f.write(json.dumps(row) + '\n')
    final = folder / 'episodes' / str(number)
    shutil.rmtree(final, ignore_errors=True)
    part.rename(final)


def _question(view, agent, question, eye_height):
    """Fly `question` in `view`: (results line, flown positions, yaws, reference, calls)."""
    started = time.perf_counter()
    pos, yaw = eqa.start_pose(question, view.scene.origin, eye_height)
    start = agents.Pose(np.asarray(pos, float), float(yaw))
    reference = _reference(question, eye_height)
    reach = float(getattr(agent, 'reach', metrics.DEFAULT_REACH))
    budget = _budget(question, view, reference, reach)
    stop, decisions, chunk, poses, last = _fly(view, agent, question, start, budget)
    flown = np.array([p.pos for p in poses], float)
    yaws = np.array([p.yaw for p in poses], float)
    calls = list(getattr(agent, 'calls', None) or [])
    goal = None
    if question.goal is not None:
        goal = np.asarray(question.goal, float) + [0.0, 0.0, _lift(question, eye_height)]
    row = {
        'benchmark': question.benchmark, 'number': question.number, 'scene': question.scene,
        'category': question.category, 'question': question.text,
        'choices': list(question.choices), 'truth': question.answer,
        'answer': getattr(agent, 'answer', None), 'stop': stop,
        'start': 'benchmark' if question.start is not None else 'origin',
        'decisions': decisions, 'budget': budget, 'steps': last.step, 'chunk_size': chunk,
        'path_length': metrics.path_length(flown), 'final_pos': flown[-1].tolist(),
        'final_yaw': float(yaws[-1]), 'goal': None if goal is None else goal.tolist(),
        'reference_length': metrics.path_length(reference) if len(reference) else None,
        'seconds': time.perf_counter() - started, 'model_calls': len(calls),
        'model_seconds': float(sum(c.get('seconds', 0.0) for c in calls)),
        'stats': _plain(getattr(agent, 'stats', None)),
    }
    return row, flown, yaws, reference, calls


def run(benchmark, questions, agent, folder, open_view, config, eye_height=eqa.EYE_HEIGHT,
        log=print):
    """Fly `agent` over `questions` of `benchmark` not yet in `folder`/results.jsonl, a scene at
    a time through `open_view(scene)` (a context manager giving a SceneView). Raises Stopped
    at the first failure, with the interrupted question's folder removed."""
    folder = Path(folder)
    (folder / 'episodes').mkdir(parents=True, exist_ok=True)
    _check_config(folder, config)
    done = finished(folder / 'results.jsonl')
    _clean(folder, done)
    todo = [q for q in questions if q.number not in done]
    log(f'{benchmark}: {len(done)} done, {len(todo)} to fly, into {folder}')
    current = None
    try:
        for scene, group in eqa.by_scene(todo).items():
            current = group[0]
            with open_view(scene) as view:
                for question in group:
                    current = question
                    row, flown, yaws, reference, calls = _question(view, agent, question,
                                                                   eye_height)
                    # The folder first, then the line: a line means its folder is whole.
                    _write_episode(folder, question.number, flown, yaws, reference, calls)
                    with open(folder / 'results.jsonl', 'a') as f:
                        f.write(json.dumps(_plain(row)) + '\n')
                        f.flush()
                    log(f'{benchmark} {question.number}: {row["stop"]}, '
                        f'{row["decisions"]}/{row["budget"]} decisions, answer '
                        f'{row["answer"]!r}, truth {row["truth"]!r} ({row["seconds"]:.0f} s)')
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        if isinstance(exc, Stopped):
            raise
        if current is not None:
            shutil.rmtree(folder / 'episodes' / f'{current.number}.part', ignore_errors=True)
        where = f'{benchmark} question {current.number}' if current else benchmark
        reason = 'interrupted' if isinstance(exc, KeyboardInterrupt) else str(exc) or repr(exc)
        raise Stopped(f'stopped at {where}: {reason}. Run the same command again to resume '
                      'there.') from exc
```

- [ ] **Step 5: Run the tests**

Run: `uv run --extra sim pytest tests/test_benchmark.py -q && uv run ruff check src tests`
Expected: all pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/drones/sim/benchmark.py src/drones/sim/eqa.py tests/test_benchmark.py
git commit -m "feat(sim): fly an agent over a benchmark, resumably, saving every model call

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Scoring and the judge (`sim/scoring.py`)

**Files:**
- Create: `src/drones/sim/scoring.py`
- Modify: `src/drones/vlm/backend.py` (`ClaudeCodeBackend`: `system=` parameter, `image=None` allowed)
- Test: `tests/test_benchmark_scoring.py`, `tests/test_vlm_claude_code.py`

**Interfaces:**
- Consumes: `metrics.*` (Task 1); `benchmark.finished`, `benchmark.EQA`, `benchmark.Stopped`, the folder layout (Task 3).
- Produces (in `drones.sim.scoring`):
  - `JUDGE_PROMPT: str` (OpenEQA mmbench, `{question}`, `{answer}`, `{prediction}`), `JUDGE_MODEL = 'sonnet'`, `MULTIPLE_CHOICE = ('hm-eqa', 'mt-hm3d')`, `OPEN = ('express-bench', 'a-eqa')`, `DEVIATIONS: list[str]`
  - `class Judge(backend, model, cache_path)` with `mark(number, question, truth, answer) -> int`
  - `score(folder, judge) -> dict` — writes `scores.json` and `report.md`; returns the summary `{'benchmark', 'folder', 'questions', 'metrics': {name: value}, 'categories': {category: {name: value}}}`
  - `report(summaries) -> str` (markdown for one or several runs)
  - `claude_judge(folder, model=JUDGE_MODEL) -> Judge`
- In `drones.vlm.backend`: `ClaudeCodeBackend(model=CLAUDE_MODEL, executable='claude', timeout=CLAUDE_TIMEOUT, system=CLAUDE_SYSTEM)`; `generate(prompt, image=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vlm_claude_code.py`:

```python
def test_a_text_only_call_sends_no_image_and_its_own_system_prompt(tmp_path):
    backend = ClaudeCodeBackend(executable=fake_claude(tmp_path), system='You grade answers.')
    assert backend.generate('Your mark?', None) == ' {"done": true} '
    message = json.loads(call(tmp_path)['stdin'])
    assert message['message']['content'] == [{'type': 'text', 'text': 'Your mark?'}]
    argv = call(tmp_path)['argv']
    assert argv[argv.index('--system-prompt') + 1] == 'You grade answers.'
```

`tests/test_benchmark_scoring.py`:

```python
"""drones-benchmark score on hand-made run folders, with a scripted judge."""
import json

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones.sim import benchmark, scoring


class FakeJudge:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, image):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def make_run(folder, bench, rows, trajectories=None):
    (folder / 'episodes').mkdir(parents=True)
    (folder / 'config.json').write_text(json.dumps({'benchmark': bench}))
    base = {'benchmark': bench, 'scene': 's', 'question': 'Q?', 'choices': [], 'truth': 't',
            'answer': 'a', 'stop': 'done', 'start': 'benchmark', 'decisions': 3, 'budget': 12,
            'steps': 40, 'chunk_size': 16, 'path_length': 1.0, 'final_pos': [0, 0, 1],
            'final_yaw': 0.0, 'goal': None, 'reference_length': None, 'seconds': 9.0,
            'model_calls': 3, 'model_seconds': 6.0,
            'stats': {'first': 2, 'retry': 1, 'failed': 0}}
    with open(folder / 'results.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps({**base, **row}) + '\n')
    for number, (pos, reference) in (trajectories or {}).items():
        (folder / 'episodes' / str(number)).mkdir()
        np.savez(folder / 'episodes' / str(number) / 'trajectory.npz', pos=pos,
                 yaw=np.zeros(len(pos)), reference=reference)
    return folder


def judge(folder, *replies, model='sonnet'):
    backend = FakeJudge(*replies)
    return scoring.Judge(backend, model, folder / 'judged.jsonl'), backend


def test_multiple_choice_success_rate_and_normalized_steps(tmp_path):
    choices = ['A) red', 'B) blue']
    folder = make_run(tmp_path / 'run', 'hm-eqa', [
        {'number': 1, 'choices': choices, 'truth': 'B) blue', 'answer': 'B', 'category': 'c1',
         'decisions': 6, 'budget': 12},
        {'number': 2, 'choices': choices, 'truth': 'A) red', 'answer': 'blue', 'category': 'c2',
         'decisions': 12, 'budget': 12, 'stop': 'budget'},
        {'number': 3, 'choices': choices, 'truth': 'A) red', 'answer': None, 'category': 'c2',
         'decisions': 3, 'budget': 12, 'stop': 'failed'}])
    summary = scoring.score(folder, judge(folder)[0])
    m = summary['metrics']
    assert m['SR'] == pytest.approx(100 / 3)
    assert m['Normalized steps'] == pytest.approx((0.5 + 1 + 0.25) / 3)
    assert m['Out of budget %'] == pytest.approx(100 / 3)
    assert m['Seconds per call'] == pytest.approx(2.0)
    assert m['Valid at once %'] == pytest.approx(100 * 6 / 9)
    assert summary['categories']['c2']['SR'] == 0.0
    rows = json.loads((folder / 'scores.json').read_text())
    assert [r['correct'] for r in rows] == [True, False, False]
    assert 'SR' in (folder / 'report.md').read_text()


def test_a_eqa_is_judged_with_openeqas_prompt_and_cached(tmp_path):
    folder = make_run(tmp_path / 'run', 'a-eqa', [
        {'number': 1, 'question': 'What is on the bed?', 'truth': 'a cat', 'answer': 'a dog',
         'category': 'objects'},
        {'number': 2, 'answer': None, 'category': 'objects'}])
    first, backend = judge(folder, 'Your mark: 3')
    summary = scoring.score(folder, first)
    (prompt,) = backend.prompts                       # the unanswered one is not asked
    assert prompt.startswith('You are an AI assistant who will help me to evaluate')
    assert prompt.endswith('Question: What is on the bed?\nAnswer: a cat\nResponse: a dog')
    assert summary['metrics']['LLM-Match'] == pytest.approx(100 * (0.5 + 0) / 2)
    again, backend = judge(folder)                    # no replies: it must not ask
    assert scoring.score(folder, again)['metrics']['LLM-Match'] == pytest.approx(25.0)
    other, backend = judge(folder, '5', model='opus')
    assert scoring.score(folder, other)['metrics']['LLM-Match'] == pytest.approx(50.0)


def test_express_bench_llm_score_e_path_and_distance_to_target(tmp_path):
    folder = make_run(tmp_path / 'run', 'express-bench', [
        {'number': 1, 'answer': 'x', 'reference_length': 2.0, 'path_length': 4.0,
         'goal': [3.0, 4.0, 1.0], 'final_pos': [0.0, 0.0, 1.0], 'category': 'c'}])
    summary = scoring.score(folder, judge(folder, '5')[0])
    m = summary['metrics']
    assert m['LLM-Score C*'] == pytest.approx(100.0)
    assert m['E_path'] == pytest.approx(50.0)
    assert m['d_T (m)'] == pytest.approx(5.0)


def test_indoor_uav_navigation_metrics_from_the_trajectories(tmp_path):
    reference = np.array([[0.0, 0, 1], [4, 0, 1]])
    folder = make_run(tmp_path / 'run', 'indoor-uav', [
        {'number': 1, 'goal': [4.0, 0, 1], 'category': 'traj_1, test seen, easy',
         'answer': None},
        {'number': 2, 'goal': [4.0, 0, 1], 'category': 'traj_2, test unseen, hard',
         'answer': None}], trajectories={
        1: (np.array([[0.0, 0, 1], [3, 0, 1]]), reference),
        2: (np.array([[0.0, 0, 1], [4, 0, 1], [0, 0, 1]]), reference)})
    summary = scoring.score(folder, judge(folder)[0])
    m = summary['metrics']
    assert m['SR'] == pytest.approx(50.0)
    assert m['OSR'] == pytest.approx(100.0)
    assert m['NE (m)'] == pytest.approx((1 + 4) / 2)
    assert 0 < m['nDTW'] <= 100
    assert set(summary['categories']) == {'test seen, easy', 'test unseen, hard'}


def test_an_unreadable_judge_reply_stops_scoring_and_keeps_the_cache(tmp_path):
    folder = make_run(tmp_path / 'run', 'a-eqa', [{'number': 1}, {'number': 2}])
    first, _ = judge(folder, '4', 'no idea', 'still no idea')
    with pytest.raises(benchmark.Stopped, match='question 2'):
        scoring.score(folder, first)
    cached = [json.loads(line) for line in (folder / 'judged.jsonl').read_text().splitlines()]
    assert [c['number'] for c in cached] == [1]


def test_the_report_lists_every_run_and_the_deviations(tmp_path):
    a = make_run(tmp_path / 'a', 'hm-eqa', [{'number': 1, 'choices': ['A) x'], 'truth': 'A) x',
                                             'answer': 'A', 'category': 'c'}])
    b = make_run(tmp_path / 'b', 'a-eqa', [{'number': 1, 'category': 'c'}])
    summaries = [scoring.score(a, judge(a)[0]), scoring.score(b, judge(b, '2')[0])]
    text = scoring.report(summaries)
    assert 'hm-eqa' in text and 'a-eqa' in text
    for deviation in scoring.DEVIATIONS:
        assert deviation in text
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --extra sim pytest tests/test_benchmark_scoring.py tests/test_vlm_claude_code.py -q`
Expected: `ImportError: cannot import name 'scoring'`; the backend test fails on `system`.

- [ ] **Step 3: Text-only calls and a system prompt in `ClaudeCodeBackend`**

In `src/drones/vlm/backend.py`, change `ClaudeCodeBackend.__init__` and the start of `generate`, and `command`'s system line:

```python
    def __init__(self, model=CLAUDE_MODEL, executable='claude', timeout=CLAUDE_TIMEOUT,
                 system=CLAUDE_SYSTEM):
        self.model, self.executable, self.timeout = model, executable, float(timeout)
        self.system = system
```

In `command`, replace `'--system-prompt', CLAUDE_SYSTEM,` with `'--system-prompt', self.system,`.

In `generate`, replace the `message = ...` statement with:

```python
        content = [{'type': 'text', 'text': prompt}]
        if image is not None:       # a judge asks in words alone
            content.insert(0, {'type': 'image', 'source': {
                'type': 'base64', 'media_type': 'image/png',
                'data': base64.b64encode(png(image)).decode()}})
        message = {'type': 'user', 'message': {'role': 'user', 'content': content}}
```

and change the signature to `def generate(self, prompt, image=None):`. Add to the class docstring: "`system` replaces the pilot's system prompt, for another use of the same call (a judge); `image` may then be None."

- [ ] **Step 4: Implement `src/drones/sim/scoring.py`**

```python
"""Score benchmark runs: drones-benchmark score runs/benchmarks/*

Reads what drones.sim.benchmark wrote and computes FAST-EQA's metrics per benchmark (see
drones.sim.metrics for where each comes from), then writes scores.json (a row per question)
and report.md beside results.jsonl.

Open answers (EXPRESS-Bench, A-EQA) are marked 1 to 5 by a judge with OpenEQA's own prompt,
word for word. The papers used GPT-4 or GPT-4o-mini; here it is Claude through the claude CLI,
the same subscription that flew. Marks are cached in judged.jsonl by question, answer and judge
model, so scoring again judges only what is new, and another --judge-model judges everything
again. A judge that fails stops scoring as a run stops, with every mark so far kept.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from drones.sim import metrics
from drones.sim.benchmark import Stopped, finished

JUDGE_MODEL = 'sonnet'
JUDGE_SYSTEM = 'You mark answers. Follow the instructions in the message.'
MULTIPLE_CHOICE = ('hm-eqa', 'mt-hm3d')
OPEN = ('express-bench', 'a-eqa')
# OpenEQA's prompts/mmbench.txt at commit cfa3fce, verbatim.
JUDGE_PROMPT = '''You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.

Example 1:
Question: Is it overcast?
Answer: no
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Response: {prediction}'''   # noqa: E501
DEVIATIONS = [
    'The judge is Claude through the claude CLI, not GPT-4 (OpenEQA) or GPT-4o-mini '
    '(EXPRESS-Bench).',
    "EXPRESS-Bench's grounding term is not computed: its C* is reported, and E_path takes the "
    'grounding as 1.',
    'Distances are straight lines in the scan, not geodesics on a navmesh: there is no '
    'habitat-sim here, and the drone flies through walls.',
    "A step is one model call, which flies at most `reach` (0.4 m by default); Explore-EQA's "
    'step moves up to 3 m.',
    "The floor area behind the EQA budget is the box of the scan's vertices 0.1 to 2.0 m above "
    "the start's floor, not habitat's navmesh bounds.",
    'A-EQA has no reference path here, so it has no E_path.',
]


class Judge:
    """Marks answers 1 to 5 through `backend` (generate(prompt, image) -> str), as `model`,
    caching each mark in `cache_path`."""

    def __init__(self, backend, model, cache_path):
        self.backend, self.model, self.cache_path = backend, model, Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.cache[(row['number'], row['answer'], row['model'])] = row['mark']

    def mark(self, number, question, truth, answer):
        """The mark for `answer`; 1 without asking for no answer at all."""
        if answer is None or not str(answer).strip():
            return 1
        key = (number, answer, self.model)
        if key in self.cache:
            return self.cache[key]
        prompt = JUDGE_PROMPT.format(question=question, answer=truth, prediction=answer)
        error = None
        for _ in range(2):      # one retry, as the pilot gets
            try:
                mark = metrics.parse_mark(self.backend.generate(prompt, None))
                break
            except ValueError as exc:
                error = exc
        else:
            raise Stopped(f'the judge failed on question {number}: {error}')
        self.cache[key] = mark
        with open(self.cache_path, 'a') as f:
            f.write(json.dumps({'number': number, 'answer': answer, 'model': self.model,
                                'mark': mark}) + '\n')
        return mark


def claude_judge(folder, model=JUDGE_MODEL):
    """A Judge asking Claude `model` through the claude CLI, caching in `folder`."""
    from drones.vlm.backend import ClaudeCodeBackend

    return Judge(ClaudeCodeBackend(model, system=JUDGE_SYSTEM), model,
                 Path(folder) / 'judged.jsonl')


def _category(benchmark, category):
    """IndoorUAV's categories begin with the trajectory's own name; group by split and
    difficulty."""
    if benchmark == 'indoor-uav':
        return category.split(', ', 1)[1] if ', ' in category else category
    return category


def _row(folder, benchmark, result, judge):
    row = {key: result[key] for key in ('number', 'category', 'stop', 'decisions', 'budget',
                                        'answer', 'truth')}
    row['category'] = _category(benchmark, result['category'])
    if benchmark in MULTIPLE_CHOICE:
        truth = metrics.choice_letter(result['truth'], result['choices'])
        picked = metrics.choice_letter(result['answer'], result['choices'])
        row['correct'] = truth is not None and picked == truth
    if benchmark in OPEN:
        try:
            row['mark'] = judge.mark(result['number'], result['question'], result['truth'],
                                     result['answer'])
        except Stopped:
            raise
        except Exception as exc:
            raise Stopped(f'the judge failed on question {result["number"]}: {exc}') from exc
    if result.get('goal') is not None:
        row['distance'] = float(np.linalg.norm(np.subtract(result['final_pos'],
                                                           result['goal'])))
    if benchmark == 'indoor-uav':
        trajectory = np.load(Path(folder) / 'episodes' / str(result['number']) /
                             'trajectory.npz')
        row.update(metrics.navigation(trajectory['pos'], np.asarray(result['goal'], float)))
        row['ndtw'] = metrics.ndtw(trajectory['reference'], trajectory['pos'])
    return row


def _metrics(benchmark, rows, results):
    """The benchmark's metrics over `rows` (from _row) and their `results` lines."""
    m = {}
    if benchmark in MULTIPLE_CHOICE:
        m['SR'] = 100 * float(np.mean([r['correct'] for r in rows]))
        m['Normalized steps'] = float(np.mean([r['decisions'] / r['budget'] for r in rows]))
    if benchmark == 'express-bench':
        marks = [r['mark'] for r in rows]
        m['LLM-Score C*'] = metrics.llm_score(marks)
        m['E_path'] = metrics.e_path(marks, [x['reference_length'] or 0.0 for x in results],
                                     [x['path_length'] for x in results])
        m['d_T (m)'] = float(np.mean([r.get('distance', math.nan) for r in rows]))
    if benchmark == 'a-eqa':
        m['LLM-Match'] = metrics.llm_match([r['mark'] for r in rows])
    if benchmark == 'indoor-uav':
        m['SR'] = 100 * float(np.mean([r['success'] for r in rows]))
        m['OSR'] = 100 * float(np.mean([r['oracle'] for r in rows]))
        m['NE (m)'] = float(np.mean([r['ne'] for r in rows]))
        m['nDTW'] = 100 * float(np.mean([r['ndtw'] for r in rows]))
    m['Out of budget %'] = 100 * float(np.mean([r['stop'] == 'budget' for r in rows]))
    calls = sum(x['model_calls'] for x in results)
    m['Seconds per call'] = sum(x['model_seconds'] for x in results) / calls if calls else math.nan
    stats = [x['stats'] for x in results if x.get('stats')]
    total = sum(sum(s.values()) for s in stats)
    if total:
        m['Valid at once %'] = 100 * sum(s['first'] for s in stats) / total
        m['Valid after retry %'] = 100 * sum(s['retry'] for s in stats) / total
        m['Failed %'] = 100 * sum(s['failed'] for s in stats) / total
    return m


def score(folder, judge):
    """Score the run in `folder` with `judge`; writes scores.json and report.md there and
    returns the summary."""
    folder = Path(folder)
    benchmark = json.loads((folder / 'config.json').read_text())['benchmark']
    results = sorted(finished(folder / 'results.jsonl').values(), key=lambda r: r['number'])
    rows = [_row(folder, benchmark, result, judge) for result in results]
    by_category = defaultdict(list)
    for row, result in zip(rows, results, strict=True):
        by_category[row['category']].append((row, result))
    summary = {
        'benchmark': benchmark, 'folder': str(folder), 'questions': len(rows),
        'metrics': _metrics(benchmark, rows, results) if rows else {},
        'categories': {c: _metrics(benchmark, [r for r, _ in pairs], [x for _, x in pairs])
                       for c, pairs in sorted(by_category.items())},
    }
    (folder / 'scores.json').write_text(json.dumps(rows, indent=1))
    (folder / 'report.md').write_text(report([summary]))
    return summary


def _table(names, rows):
    head = '| ' + ' | '.join(names) + ' |\n|' + ' --- |' * len(names) + '\n'
    return head + ''.join('| ' + ' | '.join(row) + ' |\n' for row in rows)


def _cell(value):
    if isinstance(value, float):
        return '-' if math.isnan(value) else f'{value:.2f}'
    return str(value)


def report(summaries):
    """Markdown for one run or several: a line each, then each run by category, then the
    deviations from the papers."""
    names = []
    for s in summaries:
        names += [n for n in s['metrics'] if n not in names]
    lines = ['# Benchmark results', '',
             _table(['Run', 'Benchmark', 'Questions', *names],
                    [[Path(s['folder']).name, s['benchmark'], str(s['questions']),
                      *[_cell(s['metrics'].get(n, math.nan)) for n in names]]
                     for s in summaries])]
    for s in summaries:
        own = list(s['metrics'])
        lines += [f'## {Path(s["folder"]).name} by category', '',
                  _table(['Category', *own],
                         [[c, *[_cell(m.get(n, math.nan)) for n in own]]
                          for c, m in s['categories'].items()])]
    lines += ['## Deviations from the papers', '', *[f'- {d}' for d in DEVIATIONS], '']
    return '\n'.join(lines)
```

- [ ] **Step 5: Run the tests**

Run: `uv run --extra sim pytest tests/test_benchmark_scoring.py tests/test_vlm_claude_code.py tests/test_benchmark.py -q && uv run ruff check src tests`
Expected: all pass; ruff clean (the `# noqa: E501` covers the verbatim prompt's long first line).

- [ ] **Step 6: Commit**

```bash
git add src/drones/sim/scoring.py src/drones/vlm/backend.py tests/test_benchmark_scoring.py tests/test_vlm_claude_code.py
git commit -m "feat(sim): score benchmark runs as FAST-EQA does, with Claude as the judge

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The `drones-benchmark` command, and the docs

**Files:**
- Modify: `src/drones/sim/render_agent.py` (extract `add_camera_args`, `camera_from`)
- Modify: `src/drones/sim/benchmark.py` (add `main`, `load_questions`, `parse_numbers`)
- Modify: `pyproject.toml` (entry point)
- Modify: `README.md`, `AGENTS.md`
- Test: `tests/test_benchmark.py` (CLI tests)

**Interfaces:**
- Consumes: `benchmark.run`, `benchmark.select`, `benchmark.Stopped`, `benchmark.RUNS_DIR` (Task 3); `scoring.score`, `scoring.report`, `scoring.claude_judge`, `scoring.JUDGE_MODEL` (Task 4); `eqa.split_scenes` (Task 3); `render_agent.parse_agent_args`.
- Produces:
  - `render_agent.add_camera_args(parser) -> None` (adds `--intrinsics --fov --width --height --no-mount --eye-height`)
  - `render_agent.camera_from(args, parser) -> (Intrinsics, Mount)`
  - `benchmark.parse_numbers(text) -> (int, int)`; `benchmark.load_questions(benchmark) -> list[eqa.Question]`; `benchmark.main(argv=None)`
  - console script `drones-benchmark = "drones.sim.benchmark:main"`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark.py`:

```python
def test_parse_numbers():
    assert benchmark.parse_numbers('3') == (3, 3)
    assert benchmark.parse_numbers('2-40') == (2, 40)
    with pytest.raises(ValueError):
        benchmark.parse_numbers('40-2')


def test_the_cli_runs_every_benchmark_in_order_and_scores_them(tmp_path, monkeypatch):
    flown = []
    monkeypatch.setattr(benchmark, 'load_questions',
                        lambda name: [question(1, name, choices=CHOICES if name in
                                               ('hm-eqa', 'mt-hm3d') else (),
                                               category='traj_1, test seen, easy',
                                               path=np.array([[0.0, 0, 1], [0.4, 0, 1]]),
                                               goal=np.array([0.4, 0, 1]))])

    def run(name, questions, agent, folder, open_view, config, eye_height, log):
        flown.append((name, folder.name, [q.number for q in questions], config['agent']))

    monkeypatch.setattr(benchmark, 'run', run)
    benchmark.main(['run', '--benchmark', 'all', '--agent', 'look-around', '--fov', '70',
                    '--out', str(tmp_path)])
    assert [f[0] for f in flown] == ['hm-eqa', 'mt-hm3d', 'express-bench', 'a-eqa',
                                     'indoor-uav']
    assert flown[0][1] == 'hm-eqa-look-around' and flown[0][3] == 'look-around'


def test_the_cli_exits_with_the_reason_a_run_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'load_questions', lambda name: [question(1)])

    def run(*args, **kwargs):
        raise benchmark.Stopped('stopped at hm-eqa question 1: usage limit reached')

    monkeypatch.setattr(benchmark, 'run', run)
    with pytest.raises(SystemExit, match='usage limit reached'):
        benchmark.main(['run', '--benchmark', 'hm-eqa', '--agent', 'look-around',
                        '--fov', '70', '--out', str(tmp_path)])


def test_the_cli_scores_folders_into_one_report(tmp_path, monkeypatch, capsys):
    from drones.sim import scoring

    folder = tmp_path / 'hm-eqa-x'
    fly(folder, Scripted(stop_after=1), [question(1)])
    monkeypatch.setattr(scoring, 'claude_judge', lambda folder, model: None)
    benchmark.main(['score', str(folder)])
    assert (folder / 'report.md').is_file() and (folder / 'scores.json').is_file()
    assert 'hm-eqa' in capsys.readouterr().out
```

And add the in-process end-to-end test that drives `main()` with a real `SceneView` on the synthetic box, gated on EGL like `tests/test_agents.py`. Add at the top of `tests/test_benchmark.py`, after the imports: `import subprocess, sys, textwrap` (as separate import lines) and copy `gl_env`, `can_render`, `needs_gl` from `tests/test_agents.py`. Then:

```python
CLI_ON_A_BOX = textwrap.dedent('''
    import json, sys
    import numpy as np
    from drones.sim import benchmark, eqa, scene_view, scenes

    def box():
        """A 4 x 4 x 2.5 m room, with a ring of vertices 1 m up for the floor area."""
        corners = [[x, y, z] for x in (-2.0, 2.0) for y in (-2.0, 2.0) for z in (0.0, 1.0, 2.5)]
        corners = np.array(corners, np.float32)
        faces = []
        for a, b, c, d in [(0, 3, 9, 6), (2, 5, 11, 8), (0, 6, 8, 2), (3, 9, 11, 5),
                           (0, 2, 5, 3), (6, 8, 11, 9)]:
            faces += [[a, b, c], [a, c, d]]
        uv = np.zeros((len(corners), 2), np.float32) + 0.5
        texture = np.full((2, 2, 3), 150, np.uint8)
        part = scenes.Part(corners, np.array(faces, np.int32), uv, texture)
        return scenes.Scene('box', [part], np.zeros(3), 2.0)

    scene = box()
    scene_view.load_scene = lambda name, dest=None: scene
    scenes.download = lambda names, dest=None: None
    benchmark.load_questions = lambda name: [eqa.Question(
        'hm-eqa', 1, 'box', 'What colour?', 'B) blue', 'colour', ('A) red', 'B) blue'),
        np.zeros(3), 0.0)]
    out = sys.argv[1]
    benchmark.main(['run', '--benchmark', 'hm-eqa', '--agent', 'look-around',
                    '--agent-arg', 'steps=3', '--fov', '70', '--width', '40', '--height', '30',
                    '--out', out])
    print(open(out + '/hm-eqa-look-around/results.jsonl').read())
''')


@needs_gl
def test_the_cli_flies_a_real_view_of_a_scan(tmp_path):
    done = subprocess.run([sys.executable, '-c', CLI_ON_A_BOX, str(tmp_path)], env=gl_env(),
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    row = json.loads(done.stdout.strip().splitlines()[-1])
    assert row['stop'] == 'done' and row['steps'] == 3 and row['budget'] == 12
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --extra sim pytest tests/test_benchmark.py -q`
Expected: the new tests fail with `AttributeError: module 'drones.sim.benchmark' has no attribute 'parse_numbers'` / `'main'`.

- [ ] **Step 3: Extract the camera options in `src/drones/sim/render_agent.py`**

Add these two functions after `parse_agent_args`:

```python
def add_camera_args(parser):
    """The agent camera's options, shared with drones-benchmark."""
    parser.add_argument('--intrinsics', type=Path,
                        help="the agent camera's calibration (default: recordings/intrinsics.json)")
    parser.add_argument('--fov', type=float,
                        help='an ideal pinhole this many degrees across instead of --intrinsics')
    parser.add_argument('--width', type=int, help='agent camera width (default: the '
                        "calibration's, or 320 with --fov)")
    parser.add_argument('--height', type=int, help='agent camera height (default: the '
                        "calibration's, or 240 with --fov)")
    parser.add_argument('--no-mount', action='store_true',
                        help="put the camera at the drone's centre, level, instead of where the "
                             'AI-deck sits')
    parser.add_argument('--eye-height', type=float, default=None,
                        help='m above a habitat start the drone begins (default: 1.0)')


def camera_from(args, parser):
    """(Intrinsics, Mount) from add_camera_args' options; a usage error if they clash."""
    from drones.sim.lens import DECK_MOUNT, DEFAULT_INTRINSICS, Intrinsics, Mount

    if (args.width is None) != (args.height is None):
        parser.error('--width and --height go together')
    if args.fov is not None and args.intrinsics is not None:
        parser.error('--fov and --intrinsics are alternatives')
    if args.fov is not None:
        width, height = (args.width, args.height) if args.width else (320, 240)
        intrinsics = Intrinsics.from_fov(width, height, math.radians(args.fov))
    else:
        path = args.intrinsics or DEFAULT_INTRINSICS
        if not path.is_file():
            parser.error(f'no calibration at {path}; pass --intrinsics or --fov')
        intrinsics = Intrinsics.load(path)
        if args.width is not None:
            intrinsics = intrinsics.resized(args.width, args.height)
    return intrinsics, Mount() if args.no_mount else DECK_MOUNT
```

In `main`, replace the six `parser.add_argument` calls for `--intrinsics` … `--eye-height` with `add_camera_args(parser)`; remove the `--width/--height` and `--fov/--intrinsics` checks from the validation block; replace the `if args.fov is not None: ... else: ...` intrinsics block with `intrinsics, mount = camera_from(args, parser)` (keep it after `import drones.sim`), and delete the later `mount = Mount() if args.no_mount else DECK_MOUNT` line. Remove `DECK_MOUNT, DEFAULT_INTRINSICS, Intrinsics, Mount` from `main`'s `drones.sim.lens` import if nothing else in `main` uses them.

Run: `uv run --extra sim pytest tests/test_agents.py -q` — expected: still passes.

- [ ] **Step 4: Add `main` to `src/drones/sim/benchmark.py`**

Add `import argparse`, `import contextlib` and `import sys` to the imports, and at the end:

```python
def parse_numbers(text):
    """'7' or '2-40': the question numbers to keep, (first, last)."""
    first, _, last = text.partition('-')
    first, last = int(first), int(last or first)
    if last < first:
        raise ValueError(f'{text}: the first number comes first')
    return first, last


def load_questions(benchmark):
    """Every question of `benchmark`. IndoorUAV's test scenes have their instructions fetched
    first: eqa.load returns only scenes that `prepare` has seen."""
    if benchmark == 'indoor-uav':
        eqa.prepare('indoor-uav', eqa.split_scenes(('test_seen.csv', 'test_unseen.csv')))
    return eqa.load(benchmark)


def main(argv=None):
    from drones.sim import render_agent

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    fly = commands.add_parser('run', help='fly an agent over one or more benchmarks')
    fly.add_argument('--benchmark', nargs='+', required=True, choices=(*eqa.BENCHMARKS, 'all'),
                     help='one or more benchmarks, or all of them in order')
    fly.add_argument('--questions', metavar='A-B', help="only these question numbers, e.g. 1-50")
    fly.add_argument('--agent', required=True,
                     help='look-around, follow-path, or package.module:factory')
    fly.add_argument('--agent-arg', action='append', default=[], metavar='KEY=VALUE',
                     help='keyword argument for the agent factory; repeat for more')
    render_agent.add_camera_args(fly)
    fly.add_argument('--out', type=Path, default=RUNS_DIR,
                     help=f'where run folders go (default: {RUNS_DIR})')
    fly.add_argument('--name', help='the run folder is BENCHMARK-NAME (default: the agent)')
    marks = commands.add_parser('score', help='score run folders and print one report')
    marks.add_argument('folders', nargs='+', type=Path)
    marks.add_argument('--judge-model', default=None,
                       help='the Claude model that marks open answers (default: sonnet)')
    args = parser.parse_args(argv)
    if args.command == 'score':
        return _score(args)

    numbers = None
    if args.questions:
        try:
            numbers = parse_numbers(args.questions)
        except ValueError as exc:
            parser.error(f'--questions: {exc}')
    agent_kwargs = render_agent.parse_agent_args(parser, args.agent_arg)
    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
    except ImportError:
        sys.exit('drones-benchmark needs the sim extra:  uv sync --extra sim')
    from drones.sim import scenes
    from drones.sim.scene_view import SceneView

    intrinsics, mount = render_agent.camera_from(args, parser)
    eye_height = eqa.EYE_HEIGHT if args.eye_height is None else args.eye_height
    try:
        agent = agents.make_agent(args.agent, **agent_kwargs)
    except (ValueError, ImportError, AttributeError) as exc:
        parser.error(f'--agent {args.agent}: {exc}')

    @contextlib.contextmanager
    def open_view(scene):
        scenes.download([scene])
        with SceneView(scene, intrinsics, mount=mount) as view:
            yield view

    names = eqa.BENCHMARKS if 'all' in args.benchmark else tuple(
        dict.fromkeys(args.benchmark))
    tag = args.name or args.agent.replace(':', '.')
    camera = {'width': intrinsics.width, 'height': intrinsics.height, 'fov': args.fov,
              'intrinsics': None if args.intrinsics is None else str(args.intrinsics),
              'mount': not args.no_mount}
    for name in names:
        config = {'benchmark': name, 'agent': args.agent, 'agent_args': agent_kwargs,
                  'camera': camera, 'eye_height': eye_height, 'questions': args.questions}
        try:
            questions = select(name, load_questions(name), numbers)
            run(name, questions, agent, args.out / f'{name}-{tag}', open_view, config,
                eye_height, print)
        except Stopped as exc:
            sys.exit(str(exc))
    print('All done. Score with:  drones-benchmark score ' +
          ' '.join(str(args.out / f'{n}-{tag}') for n in names))


def _score(args):
    from drones.sim import scoring

    model = args.judge_model or scoring.JUDGE_MODEL
    summaries = []
    for folder in args.folders:
        if not (folder / 'results.jsonl').exists():
            continue
        try:
            summaries.append(scoring.score(folder, scoring.claude_judge(folder, model)))
        except Stopped as exc:
            sys.exit(str(exc))
    if not summaries:
        sys.exit('no run folders with results.jsonl among those given')
    print(scoring.report(summaries))
```

Note: `scoring.claude_judge(folder, model)` is positional, matching the test's monkeypatched `lambda folder, model: None`. A `None` judge is fine for hm-eqa, which never judges.

- [ ] **Step 5: Register the entry point**

In `pyproject.toml`, under `[project.scripts]`, after `drones-render-agent = ...`:

```toml
drones-benchmark = "drones.sim.benchmark:main"
```

Run: `uv sync --extra sim --extra camera` (list every extra the machine had; `uv sync` drops the rest), then `uv run --extra sim drones-benchmark --help`.
Expected: the usage with `run` and `score`.

- [ ] **Step 6: Run the tests**

Run: `uv run --extra sim pytest tests/test_benchmark.py tests/test_agents.py -q && uv run ruff check src tests`
Expected: all pass (the box test skips without EGL); ruff clean.

- [ ] **Step 7: Document it**

In `README.md`, add a subsection `### Benchmarking a pilot` right after the "Claude as the pilot" subsection of the VLM section (before `### Measured here`), with this content:

````markdown
### Benchmarking a pilot

`drones-benchmark` flies an agent over every question of a benchmark and scores it the way
FAST-EQA does. It is meant to run overnight, for as many nights as a benchmark takes:

```bash
uv run --extra sim drones-benchmark run --benchmark hm-eqa \
    --agent drones.vlm.agent:make --agent-arg backend=claude-code --agent-arg action_space=discrete
uv run --extra sim drones-benchmark run --benchmark all --agent ...        # every benchmark
uv run --extra sim drones-benchmark run --benchmark a-eqa --questions 1-20 --agent ...
uv run --extra sim drones-benchmark score runs/benchmarks/*
```

A run stops at the first failure — a usage limit, a logged-out CLI, a scene that would not
download, Ctrl-C — and keeps everything finished. The same command resumes at the question it
stopped on. A run folder remembers its settings and refuses others; pass `--name` for a second
run of the same benchmark (another model, another action space). IndoorUAV runs its
`test_seen` and `test_unseen` trajectories only.

The budget is counted in model calls: Explore-EQA's `int(√floor area × 3)` for the EQA
benchmarks, twice the reference path for IndoorUAV. An EQA question whose budget runs out gets
one more call, in which the model must answer.

| Benchmark | Metrics |
| --- | --- |
| hm-eqa, mt-hm3d | SR, normalized steps (calls ÷ budget) |
| express-bench | LLM-Score C\*, E_path, d_T |
| a-eqa | LLM-Match |
| indoor-uav | SR (2 m), NE, OSR, nDTW |

Each also reports seconds per call, how many replies were valid at once, after a retry or not
at all, and how many questions ran out of budget. Open answers are marked 1 to 5 by Claude
(`--judge-model`, sonnet by default) with OpenEQA's own prompt; marks are cached in
`judged.jsonl`. `report.md` lists where this differs from the papers: Claude as the judge, no
grounding term, straight-line distances in a scan the drone can fly through, a step that flies
0.4 m where Explore-EQA's flies 3 m, and a floor area from the scan's vertices.

What a run writes, under `runs/benchmarks/<benchmark>-<name>/`:

- `config.json`, `results.jsonl` (a line per question), and after `score`, `scores.json` and
  `report.md`;
- `episodes/<number>/trajectory.npz`: `pos` and `yaw` at every step, and the `reference` path;
- `episodes/<number>/calls.jsonl` and `calls/<k>.png`: **the distillation data**. One line per
  model call, retries and the final answer included: `step`, `elapsed`, `altitude`, `pos`,
  `yaw`, the exact `prompt`, the raw `reply`, `attempt`, `valid`, `error`, the parsed `chunk`,
  `seconds`, `final`, and `image`, the frame the model saw, at the camera's own size.
````

In `AGENTS.md`:

1. In "Where new work goes", in the **Benchmark inference** bullet, after the sentence ending "`sim/` still imports no agent.", add:

```
`drones-benchmark run` (`sim/benchmark.py`) flies an agent over whole benchmarks and also reads,
by attribute, `chunk_size`, `reach`, `calls`, `stats`, `error` and `conclude(observation)`; all
are optional. `drones-benchmark score` (`sim/scoring.py`, metrics in `sim/metrics.py`) imports
`drones.vlm.backend` inside a function, for the Claude judge; nothing in `sim/` imports
`drones.vlm` at module level.
```

2. In "Commands", after the `drones-render-agent` paragraph, add:

```
`drones-benchmark run` with a Claude agent (`backend=claude-code`) and `drones-benchmark score`
on open-answer benchmarks spend the user's subscription usage: ask before running either. Its
tests fake the agent, the view and the judge. A run resumes from `results.jsonl`; a line is
written only after its episode folder, and a start deletes any folder without one.
```

- [ ] **Step 8: Run the whole suite, then commit**

Run: `uv run --extra sim pytest -q && uv run ruff check src tests`
Expected: everything passes (allow ~7 minutes); ruff clean.

```bash
git add src/drones/sim/benchmark.py src/drones/sim/render_agent.py pyproject.toml uv.lock README.md AGENTS.md tests/test_benchmark.py
git commit -m "feat(sim): drones-benchmark, an overnight benchmark of a pilot

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Add `uv.lock` only if `uv sync` changed it; check `git status --short` first.)
