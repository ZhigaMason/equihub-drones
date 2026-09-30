# A vision-language model as the pilot

Let a vision-language model (VLM) fly the drone from a question and what the camera sees. Once per
second it is shown the current frame and the question, and it replies with JSON: a chunk of 16
actions, a completion flag and, when it is finished, its answer. The JSON is validated with
pydantic before anything is executed.

This version runs in the simulator only, on the scanned scenes and EQA benchmarks that
`drones-render-agent` already films. The action schema and the pilot are kept free of the
simulator so that a real-drone adapter can be added later without changing them.

## Decisions taken with the user

- **Target:** a shared schema and pilot, with a simulator adapter first. No real-drone code in this
  version.
- **Model:** a local model, loaded in-process with transformers. No hosted API.
- **Timing:** simulated time. One model call per simulated second; its 16 actions are played back
  at 16 Hz, then a new frame is taken. The simulator waits for the model.
- **Action form:** two modes, each with its own schema, chosen when the pilot is built:
  `continuous` (forward, yaw, altitude) and `discrete` (a named move). Both convert to the same
  `control.mixer.Command`.

## Facts this design rests on

Checked in this repository and on this machine before writing:

- `control.mixer.Command` is already `forward` and `yaw` in -1..1 and `altitude` as an absolute
  target in metres, `None` to hold. `+yaw` turns left (`teleop/web/static/app.js`). The continuous
  action is this Command, so conversion needs no sign or scale.
- An agent for `drones-render-agent` is any object with `reset(question, pose)` and
  `act(observation) -> Pose | None`, loaded as `package.module:factory` with
  `--agent-arg key=value`. `agents.episode` renders one image per step and reads `agent.answer`
  afterwards. Nothing in `sim/` has to change.
- `agents.episode` stops after `MAX_STEPS = 500` steps unless `--steps` says otherwise. At 16 Hz
  that is 31 s of simulated time.
- The agent moves kinematically and nothing collides with the scan. This is true of the built-in
  agents as well.
- pydantic 2.13 is installed, but only as a dependency of FastAPI. It is not declared.
- This laptop has no GPU (AMD integrated graphics, 16 cores, 27 GB RAM). `torch` and
  `transformers` are not installed on `master`.
- `google/gemma-3n-E2B-it` is in the Hugging Face cache. The `neural-sandbox` branch loads it with
  `transformers.pipeline('image-text-to-text', ...)` and notes that it is gated: the pipeline needs
  the Hugging Face token, or the Hub returns 401.
- `AGENTS.md` keeps agent code out of `sim/` unless it is a baseline.

## Layout

A new package, `src/drones/vlm/`:

| File | What it does | May import |
| --- | --- | --- |
| `actions.py` | The pydantic schemas, and conversion of either kind of action to a `Command` | pydantic, `drones.control`, `drones.config` |
| `prompt.py` | The prompt for each mode | stdlib, `drones.vlm.actions` |
| `pilot.py` | `Pilot`: asks a backend for a chunk, validates it, hands out one `Command` per step | the two above |
| `backend.py` | The `Backend` protocol and `TransformersBackend` | numpy; torch, transformers and PIL inside functions only |
| `agent.py` | `make(...)`, the factory for `drones-render-agent`, and the kinematic adapter | `drones.sim.agents`, the rest of `drones.vlm` |

`actions`, `prompt`, `pilot` and `backend` import neither the simulator, JAX, MuJoCo, cflib nor
torch at import time. Only `agent.py` needs the `sim` extra, and only a real model run needs the
`vlm` extra.

## The schema: `vlm/actions.py`

```python
CHUNK = 16          # actions per chunk
STEP = 1 / CHUNK    # s of simulated time per action

class ContinuousAction(BaseModel):
    forward: float = Field(ge=-1, le=1)     # x config.MAX_MANUAL_SPEED, + is forward
    yaw: float = Field(ge=-1, le=1)         # x config.MAX_YAW_RATE, + turns left
    altitude: float | None = None           # absolute target in m, None holds

Move = Literal['forward', 'backward', 'turn_left', 'turn_right', 'rise', 'descend', 'hover']

class ContinuousChunk(BaseModel):
    actions: list[ContinuousAction]
    done: bool = False
    answer: str | None = None

class DiscreteChunk(BaseModel):
    actions: list[Move]
    done: bool = False
    answer: str | None = None
```

Rules, enforced by validators and the same for both chunks:

- Unknown fields are rejected (`extra='forbid'`), so a misspelt key is an error, not a silent
  default.
- `actions` has exactly `CHUNK` entries, unless `done` is true, when it may have any number from 0
  to `CHUNK`.
- `done` means stop now. The actions of a `done` chunk are not executed.
- `answer` is what the model says in reply to the question. It is kept from the last chunk that
  gave one.
- Values out of range are a validation error, not clamped. `Command` and the mixer clamp again
  downstream, which is the safety net, not the contract.
- `altitude` is not range-checked by the schema, because the envelope is operator tuning in
  `drones.config`. Whoever executes the Command clamps it, as `DroneController.set_control` does.

Conversion to a `Command`:

- `to_command(action, altitude)` takes the drone's current altitude and returns a `Command`.
- A `ContinuousAction` maps field for field. An `altitude` of `None` stays `None`: hold the
  current altitude, as the mixer reads it. A target is therefore pursued only while the model
  keeps stating it.
- A `Move` is a full-scale Command for one step: `forward` and `backward` are `forward=+1` and
  `-1`; `turn_left` and `turn_right` are `yaw=+1` and `-1`; `rise` and `descend` set the altitude
  target to `altitude ± config.MAX_CLIMB_SPEED * STEP`; `hover` is `Command()`.

`SCHEMAS = {'continuous': ContinuousChunk, 'discrete': DiscreteChunk}` names the modes.

`parse_chunk(text, space)` turns model output into a chunk. It takes the text from the first `{` to
the last `}`, so a fenced or prefaced reply still parses, and validates it with
`model_validate_json`. It raises `pydantic.ValidationError`, or `ValueError` when there is no JSON
object at all.

## The prompt: `vlm/prompt.py`

`build_prompt(space, question, choices, altitude, elapsed)` returns one string:

1. what the model is: the pilot of a small indoor drone, looking through its forward camera;
2. the question or instruction, and the choices if the benchmark is multiple choice. Without a
   question, the task is to explore;
3. the state it cannot see: altitude in metres and seconds elapsed;
4. what each action does in physical units, computed from `drones.config` (at full scale one
   action is 2.5 cm forward or 5.6 degrees of turn, a whole chunk 0.4 m or 90 degrees);
5. the reply format: JSON only, exactly 16 actions, with one short example for the mode;
6. when to set `done`, and that `answer` goes with it.

`retry_prompt(prompt, reply, error)` appends the rejected reply and the validation error and asks
again.

## The pilot: `vlm/pilot.py`

```python
class Pilot:
    def __init__(self, backend, space='continuous', max_failures=3): ...
    def reset(self, question=None, choices=()): ...
    def step(self, image, altitude) -> Command | None: ...
    answer: str | None
    error: str | None
    queries: int
```

- `step` is called once per action step, 16 times per simulated second. It returns the Command for
  that step, or `None` when the episode is over.
- When no actions are left it calls `backend.generate(prompt, image)` with the image it was just
  given, and parses the reply. Otherwise the image is ignored.
- An invalid reply is retried once with `retry_prompt`. A second invalid reply makes the chunk a
  failure: 16 `hover` Commands are played instead.
- `max_failures` failed chunks in a row end the episode: `step` returns `None` and `error` holds
  the last validation error. One valid chunk resets the count.
- A chunk with `done` ends the episode at once, and `answer` is set from it.

`Pilot` knows nothing about poses, scenes or the simulator. A real-drone adapter would call `step`
with AI-deck frames and pass the Command to `DroneController.set_control`.

## The backend: `vlm/backend.py`

```python
class Backend(Protocol):
    def generate(self, prompt: str, image: np.ndarray) -> str: ...
```

`image` is `(height, width, 3)` uint8 RGB. The return value is the model's raw text.

`TransformersBackend(model=DEFAULT_MODEL, max_new_tokens=None, device=None)`:

- `DEFAULT_MODEL = 'google/gemma-3n-E2B-it'`.
- builds `transformers.pipeline('image-text-to-text', ...)` on first use, on CUDA in bfloat16 when
  there is one and on the CPU in float32 otherwise. The token from `HF_TOKEN` or the Hugging Face
  login is passed through, because the default model is gated;
- sends one user message, the image then the prompt, and returns the generated text;
- generates greedily (`do_sample=False`), so a run is repeatable;
- `max_new_tokens` defaults to 768, enough for a continuous chunk;
- a missing `vlm` extra exits with the command that installs it, as the other optional entry
  points do.

Grammar-constrained decoding is left out. It is the fix if a small model proves unable to produce
valid chunks, and the first real run measures that (see "Measured before it is called done").

## The simulator adapter: `vlm/agent.py`

```python
def make(action_space='continuous', model=DEFAULT_MODEL, start_altitude=1.0,
         max_new_tokens=None, backend=None) -> VLMAgent
```

`VLMAgent` has `reset(question, pose)`, `act(observation)`, `answer` and `error`.

- `reset` resets the pilot with the question's text and choices. It takes the start pose to be
  `start_altitude` above the floor, which fixes the floor's height for the episode. 1.0 m is both
  `eqa.EYE_HEIGHT` and the default take-off height.
- `act` calls `pilot.step(observation.image, altitude)` and integrates the Command over `STEP`:
  - yaw changes by `yaw * MAX_YAW_RATE * STEP`;
  - the position moves `forward * MAX_MANUAL_SPEED * STEP` along the new heading;
  - a Command's altitude target is clamped to `MIN_ALTITUDE..MAX_ALTITUDE`, and the altitude moves
    towards it by at most `MAX_CLIMB_SPEED * STEP`. Without a target it stays where it is.
  It returns the new level `Pose`, or `None` when the pilot returns `None`.
- There is no smoothing, no avoidance and no collision, as with the built-in agents. The mixer is
  not used: it steps at 10 Hz, and its avoidance needs ranger readings the scan view does not make.

Run it with:

```bash
uv run --extra sim --extra vlm drones-render-agent --benchmark hm-eqa --question 1 \
    --agent drones.vlm.agent:make --agent-arg action_space=discrete --fps 16
```

`--fps 16` makes the film play in real time. No new entry point is added.

## Dependencies

- `pydantic>=2.13` joins the base dependencies.
- A new `vlm` extra: `torch`, `transformers`, `accelerate`, `timm` (Gemma 3n's vision tower) and
  `pillow`, with the versions the `vision` extra on `neural-sandbox` uses. The two extras overlap
  and will need reconciling when that branch is merged.

## Tests

All hermetic: no torch, no network, no `scenes/`.

- `tests/test_vlm_actions.py`: range and extra-field errors; the chunk length rule with and without
  `done`; each `Move`'s Command; `parse_chunk` on plain, fenced and prefaced JSON and on text with
  no JSON.
- `tests/test_vlm_pilot.py`, against a fake backend that returns scripted replies: one query per 16
  steps, each with the image of that step; an invalid reply retried once with the error in the
  prompt; hover after two invalid replies; the episode ended after three failed chunks, with
  `error` set; a valid chunk resetting the count; `done` stopping at once with `answer` set.
- `tests/test_vlm_prompt.py`: the question, choices, altitude and the mode's format are in the
  prompt.
- `tests/test_vlm_backend.py`: `TransformersBackend` with a stub pipeline injected: the message it
  sends and the text it returns.
- `tests/test_vlm_agent.py` (needs `crazyflow`, skips without it): through `agents.episode` with a
  fake view and a fake backend: a `forward` chunk moves 0.4 m along the heading; `turn_left` turns
  90 degrees to the left; `rise` climbs and stops at `MAX_ALTITUDE`; `done` ends the episode and
  leaves `answer`.
- `tests/test_architecture.py`: a fresh interpreter imports `drones.vlm.actions`, `prompt`, `pilot`
  and `backend` and finds none of torch, transformers, jax, crazyflow, mujoco or cflib loaded.

## Measured before it is called done

One real run per mode on this laptop with the default model, a few chunks each, recording:

- seconds per chunk;
- how many replies were valid at the first attempt, after the retry, and not at all.

The numbers go in `README.md`. If most chunks fail, that is reported to the user with the option of
constrained decoding, rather than worked around.

## Documentation

- `AGENTS.md`: a `vlm/` row in the layering table; where a VLM agent and a new backend go; the
  `vlm` extra under Commands; any fact that cost effort to find.
- `README.md`: what the pilot does, how to run it, both schemas and the measured numbers.

## Known limits

- **Speed.** Without a GPU a continuous chunk is expected to take tens of seconds, so an episode
  takes far longer than it lasts. A discrete chunk is about a fifth of the output.
- **Validity.** A 2B model may miscount 16 actions or break the JSON. The retry and hover fallback
  bound the damage; they do not make the model better.
- **No memory.** The model sees one frame and no history, so it can circle. Adding the previous
  chunk to the prompt is a small later change.
- **No collision.** The pilot can fly through the scan's walls.

## Out of scope

- Flying the real drone, and any code under `missions/` or `crazyflie/`.
- Wall-clock 1 Hz and asynchronous inference.
- Hosted or OpenAI-compatible backends.
- Scoring answers against a benchmark.
