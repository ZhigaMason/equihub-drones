"""The VLM pilot as an agent for scanned scenes: drones-render-agent --agent drones.vlm.agent:make

    uv run --extra sim --extra vlm drones-render-agent --benchmark hm-eqa --question 1 \\
        --agent drones.vlm.agent:make --agent-arg action_space=discrete --fps 16 --steps 48

The simulator's agent loop asks for the next pose once per step and renders a frame there. One
step is one action, 1/16 s, so `--fps 16` plays the film in real time; the pilot looks at every
sixteenth frame. `--agent-arg` takes `action_space` (continuous or discrete), `backend`
(transformers, claude-code to fly on a Claude subscription through the claude CLI, or openai for a
model a vLLM server serves at `url`), `model` (a Hugging Face id, or a Claude alias such as sonnet
or opus), `start_altitude` and `max_new_tokens`.

The drone moves kinematically, as the built-in agents do: a Command's forward and yaw are applied
at the teleop limits from `drones.config`, with no smoothing, no avoidance and nothing to collide
with. The mixer is not stepped here. It runs at 10 Hz, not 16, and its avoidance needs ranger
readings that a view of the scan alone does not produce.

Altitude needs a floor, and the agent loop gives only a start pose. The start is taken to be
`start_altitude` above the floor. The default, 1.0 m, is the take-off height and where the habitat
benchmarks' start poses are lifted to (eqa.EYE_HEIGHT). It is wrong for IndoorUAV, whose starts
are at the dataset's own flight height, and after `--eye-height`: pass the real height as
`--agent-arg start_altitude=`, or the altitude the model is told and the altitude limits are both
off by the difference.

For drones-benchmark the agent also has `chunk_size`, `reach` (m one chunk can fly at most),
`calls` (the pilot's records, with the pose of each), `stats` and `conclude`, which asks for
an answer when the benchmark's budget has run out. drones.sim reads them by name.

Each model call is reported on stderr as it returns. A call takes a minute or more without a GPU,
and drones-render-agent says nothing between loading the scene and writing the film.
"""
import itertools
import math
import sys

import numpy as np

from drones import config
from drones.control.mixer import clamp
from drones.sim.agents import Pose
from drones.vlm.actions import CHUNK, STEP, ContinuousAction, chunk_size, json_schema
from drones.vlm.backend import (CLAUDE_MODEL, DEFAULT_MODEL, MAX_NEW_TOKENS, OPENAI_URL,
                                ClaudeCodeBackend, OpenAIBackend, TransformersBackend)
from drones.vlm.pilot import Pilot

ERROR_SHOWN = 200   # characters of a validation error in a progress line
# Words a caption's action line may hold: what the film's panel was sized for (sim/render_agent
# AGENT_LINES). A longer chunk is folded into runs, and past that shown as a window.
PANEL_WORDS = CHUNK


def integrate(pose, altitude, command):
    """The level Pose one STEP after `command`, from `pose` at `altitude` m above the floor."""
    yaw = pose.yaw + math.radians(clamp(command.yaw) * config.MAX_YAW_RATE) * STEP
    ahead = clamp(command.forward) * config.MAX_MANUAL_SPEED * STEP
    climb = 0.0
    if command.altitude is not None:
        target = clamp(command.altitude, config.MIN_ALTITUDE, config.MAX_ALTITUDE)
        limit = config.MAX_CLIMB_SPEED * STEP
        climb = clamp(target - altitude, -limit, limit)
    step = np.array([ahead * math.cos(yaw), ahead * math.sin(yaw), climb])
    return Pose(np.asarray(pose.pos, float) + step, yaw)


def _word(action):
    """One action as a word of a caption: a move's name, or forward/yaw/altitude."""
    if not isinstance(action, ContinuousAction):
        return action
    altitude = '-' if action.altitude is None else f'{action.altitude:.2f}'
    return f'{action.forward:+.2f}/{action.yaw:+.2f}/{altitude}'


def describe(pilot, over=False):
    """Caption lines for what `pilot` is doing: the chunk it is flying, whole, with the action
    of this step in brackets, the completion flag and the answer. `over` says the episode has
    ended. [] before the first chunk."""
    count = sum(pilot.stats.values())
    if not count:
        return []
    lines = [] if pilot.question else ['no question: explore']
    head = f'chunk {count} at {pilot.asked_at:.0f} s   '
    chunk = pilot.chunk
    if chunk is None:
        error = ' '.join(str(pilot.error).split())[:ERROR_SHOWN]
        lines.append(f'{head}FAILED, {"stopping" if over else "hovering"}: {error}')
        return lines if over else lines + [f'hover x{pilot.size}']
    answer = '-' if pilot.answer is None else repr(pilot.answer)
    head += f'done: {str(chunk.done).lower()}   answer: {answer}'
    if chunk.done:          # its actions, if it has any, are not flown
        return lines + [head]
    if pilot.space == 'continuous':
        head += '   (forward/yaw/altitude)'
    words = [_word(action) for action in chunk.actions]
    current = pilot.played - 1
    if len(words) <= PANEL_WORDS:
        words[current] = f'[{words[current]}]'
        return lines + [head, ' '.join(words)]
    return lines + [f'{head}   action {pilot.played} of {len(words)}', _runs(words, current)]


def _runs(words, current):
    """`words` folded into runs (`forward x20`), the run holding `current` in brackets. Runs
    past PANEL_WORDS are cut to a window around that one, with `...` where they were cut."""
    runs, at = [], 0
    for word, group in itertools.groupby(words):
        n = len(list(group))
        text = word if n == 1 else f'{word} x{n}'
        runs.append(f'[{text}]' if at <= current < at + n else text)
        at += n
    mark = next(i for i, run in enumerate(runs) if run.startswith('['))
    first = max(0, min(mark - PANEL_WORDS // 4, len(runs) - PANEL_WORDS))
    shown = runs[first:first + PANEL_WORDS]
    return ' '.join((['...'] if first else []) + shown
                    + (['...'] if first + PANEL_WORDS < len(runs) else []))


class VLMAgent:
    """Flies `pilot`'s Commands in a scanned scene, from a start `start_altitude` m up."""

    def __init__(self, pilot, start_altitude=1.0):
        self.pilot, self.start_altitude = pilot, float(start_altitude)
        self._floor = 0.0
        self._over = False

    def reset(self, question, pose):
        self._floor = float(pose.pos[2]) - self.start_altitude
        self._over = False
        # An eqa.Question, or None when the scene was picked without a benchmark.
        self.pilot.reset(getattr(question, 'text', None), getattr(question, 'choices', ()))

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

    def _report(self, step, before, seconds, over):
        """One stderr line for the chunk just asked for at `step`."""
        pilot = self.pilot
        # pydantic's errors run over several lines; one is enough here.
        error = ' '.join(str(pilot.error).split())[:ERROR_SHOWN]
        if pilot.stats['failed'] > before['failed']:
            outcome = f'failed, {"stopping" if over else "hovering"}: {error}'
        else:
            outcome = 'valid after a retry' if pilot.stats['retry'] > before['retry'] else 'valid'
            if over:
                outcome += f', done, answer {pilot.answer!r}'
        print(f'vlm: chunk {sum(pilot.stats.values())} at {step * STEP:.0f} s: {outcome} '
              f'({seconds:.0f} s)', file=sys.stderr, flush=True)

    @property
    def caption(self):
        """Lines for a film to show under the frame the agent last acted on."""
        return describe(self.pilot, self._over)

    @property
    def answer(self):
        return self.pilot.answer

    @property
    def error(self):
        return self.pilot.error

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


def make(action_space='continuous', model=None, start_altitude=1.0,
         max_new_tokens=None, backend='transformers', constrain=True, chunk=CHUNK,
         url=OPENAI_URL):
    """The agent factory. `chunk` actions are flown between looks, 1 to 32, each 1/16 s.

    `backend` is where replies come from: 'transformers', a local `model` (a Hugging Face id),
    'claude-code', Claude `model` (sonnet by default, opus, haiku or a full name) through the
    claude CLI and its subscription login, or 'openai', `model` as served at `url` (vllm serve).
    Any object with `generate` replaces them all, for tests. A local or served model is held to
    the chunk's schema as it writes, unless `constrain` is 0 (or false, no, off); Claude writes
    freely, and the pilot validates every reply either way."""
    size = chunk_size(chunk)
    # --agent-arg gives 0 as a number and any other word as text.
    free = str(constrain).lower() in ('0', 'false', 'no', 'off')
    if backend == 'claude-code':
        backend = ClaudeCodeBackend(model or CLAUDE_MODEL)
    elif isinstance(backend, str) and backend not in ('transformers', 'openai'):
        raise ValueError(f'backend is transformers, claude-code or openai, not {backend!r}')
    if max_new_tokens is None:
        # MAX_NEW_TOKENS holds a chunk of the default size; a longer one needs room in
        # proportion, or every reply is cut off at the limit and rejected.
        max_new_tokens = MAX_NEW_TOKENS * max(size, CHUNK) // CHUNK
    if backend == 'openai':
        if not model:
            raise ValueError('backend openai needs the served model, as model=NAME')
        backend = OpenAIBackend(model, url, max_new_tokens,
                                schema=None if free else json_schema(action_space, size))
    if backend == 'transformers':
        backend = TransformersBackend(model or DEFAULT_MODEL, max_new_tokens,
                                      schema=None if free else json_schema(action_space, size))
    return VLMAgent(Pilot(backend, action_space, size=size), start_altitude)
