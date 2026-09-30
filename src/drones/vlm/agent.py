"""The VLM pilot as an agent for scanned scenes: drones-render-agent --agent drones.vlm.agent:make

    uv run --extra sim --extra vlm drones-render-agent --benchmark hm-eqa --question 1 \\
        --agent drones.vlm.agent:make --agent-arg action_space=discrete --fps 16 --steps 48

The simulator's agent loop asks for the next pose once per step and renders a frame there. One
step is one action, 1/16 s, so `--fps 16` plays the film in real time; the pilot looks at every
sixteenth frame. `--agent-arg` takes `action_space` (continuous or discrete), `model` (a Hugging
Face id), `start_altitude` and `max_new_tokens`.

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

Each model call is reported on stderr as it returns. A call takes a minute or more without a GPU,
and drones-render-agent says nothing between loading the scene and writing the film.
"""
import math
import sys

import numpy as np

from drones import config
from drones.control.mixer import clamp
from drones.sim.agents import Pose
from drones.vlm.actions import STEP
from drones.vlm.backend import DEFAULT_MODEL, TransformersBackend
from drones.vlm.pilot import Pilot

ERROR_SHOWN = 200   # characters of a validation error in a progress line


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


class VLMAgent:
    """Flies `pilot`'s Commands in a scanned scene, from a start `start_altitude` m up."""

    def __init__(self, pilot, start_altitude=1.0):
        self.pilot, self.start_altitude = pilot, float(start_altitude)
        self._floor = 0.0

    def reset(self, question, pose):
        self._floor = float(pose.pos[2]) - self.start_altitude
        # An eqa.Question, or None when the scene was picked without a benchmark.
        self.pilot.reset(getattr(question, 'text', None), getattr(question, 'choices', ()))

    def act(self, observation):
        pose = observation.pose
        altitude = float(pose.pos[2]) - self._floor
        pilot = self.pilot
        stats, seconds = dict(pilot.stats), pilot.seconds
        command = pilot.step(observation.image, altitude)
        if pilot.stats != stats:
            self._report(observation.step, stats, pilot.seconds - seconds, command is None)
        if command is None:
            return None
        return integrate(pose, altitude, command)

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
    def answer(self):
        return self.pilot.answer

    @property
    def error(self):
        return self.pilot.error


def make(action_space='continuous', model=DEFAULT_MODEL, start_altitude=1.0,
         max_new_tokens=None, backend=None):
    """The agent factory. `backend` replaces the local model, for tests and other models."""
    if backend is None:
        backend = TransformersBackend(model, max_new_tokens)
    return VLMAgent(Pilot(backend, action_space), start_altitude)
