"""The pilot: one model call per chunk, one Command per step.

`step` is called once per action step, 16 times per simulated second, with the current frame. It
looks at the frame only when the last chunk has run out, so the model sees one frame a second and
its 16 actions are played back in between. The pilot waits for the model: time here is counted
in steps, not on the clock, which is what lets a model that takes a minute fly at all.

A reply that does not validate is never partly executed. It is retried once with the error, and
a second failure plays a chunk of hover, the one action that is safe whatever the model meant.
Three failed chunks in a row end the episode, because a model that cannot follow the format will
not start to.

Nothing here knows about poses or the simulator. A real-drone adapter would call `step` with
AI-deck frames and hand each Command to `DroneController.set_control`.
"""
import logging
import time
from collections import deque

from drones.vlm.actions import CHUNK, STEP, chunk_size, parse_chunk, schema_for, to_command
from drones.vlm.prompt import build_prompt, retry_prompt

logger = logging.getLogger(__name__)

ATTEMPTS = 2        # the first reply and one retry
MAX_FAILURES = 3    # failed chunks in a row before the episode ends


class Pilot:
    """Flies by asking `backend` (anything with generate(prompt, image) -> str) for chunks of
    `size` actions in action space `space`. A larger chunk is a longer stretch flown open loop:
    each action lasts STEP whatever the size, so the model looks every `size` steps. After an
    episode, `answer` is the model's reply to the question and `error` why the last chunk failed,
    if it did."""

    def __init__(self, backend, space='continuous', max_failures=MAX_FAILURES, size=CHUNK):
        schema_for(space)       # an unknown space or size fails here, not at the first frame
        self.size = chunk_size(size)
        self.backend, self.space, self.max_failures = backend, space, int(max_failures)
        self.reset()

    def reset(self, question=None, choices=()):
        """Start an episode on `question` (None to explore), with its multiple `choices`."""
        self.question, self.choices = question, tuple(choices)
        self.answer = self.error = None
        self.queries = 0
        self.seconds = 0.0      # spent in the backend
        self.stats = {'first': 0, 'retry': 0, 'failed': 0}
        self.chunk = None       # the chunk being flown; None before the first and after a failure
        self.played = 0         # how many of its actions have been handed out
        self.asked_at = 0.0     # s into the episode when it was asked for
        self._actions = deque()
        self._steps = 0
        self._failures = 0
        self._over = False

    def step(self, image, altitude):
        """The Command for this step, with the drone at `altitude` m, or None when the episode is
        over. `image` is looked at only when a new chunk is due."""
        if self._over:
            return None
        if not self._actions and not self._plan(image, altitude):
            self._over = True
            return None
        self._steps += 1
        self.played += 1
        return to_command(self._actions.popleft(), altitude)

    def _plan(self, image, altitude):
        """Queue the next chunk's actions. False when the episode ends instead."""
        prompt = build_prompt(self.space, self.question, self.choices, altitude,
                              self._steps * STEP, self.size)
        chunk = self._ask(prompt, image)
        self.chunk, self.played, self.asked_at = chunk, 0, self._steps * STEP
        if chunk is None:
            self.stats['failed'] += 1
            self._failures += 1
            if self._failures >= self.max_failures:
                logger.warning('%d chunks in a row failed; stopping', self._failures)
                return False
            self._actions.extend(['hover'] * self.size)
            return True
        self._failures = 0
        self.error = None
        if chunk.answer is not None:
            self.answer = chunk.answer
        if chunk.done:
            return False
        self._actions.extend(chunk.actions)
        return True

    def _ask(self, prompt, image):
        """A valid chunk for `image`, or None after ATTEMPTS invalid replies."""
        asked = prompt
        for attempt in range(ATTEMPTS):
            started = time.perf_counter()
            reply = self.backend.generate(asked, image)
            self.seconds += time.perf_counter() - started
            self.queries += 1
            try:
                chunk = parse_chunk(reply, self.space, self.size)
            except ValueError as exc:
                self.error = str(exc)
                logger.warning('reply %d rejected: %s', self.queries, self.error)
                asked = retry_prompt(prompt, reply, self.error)
                continue
            self.stats['retry' if attempt else 'first'] += 1
            return chunk
        return None
