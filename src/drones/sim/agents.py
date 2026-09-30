"""Agents that move through a scanned scene from what their camera sees, and the loop that runs one.

An agent is anything with these two methods, and optionally an `answer`:

    class MyAgent:
        def reset(self, question, pose):       # an eqa.Question or None, and the start Pose
            ...
        def act(self, observation):            # an Observation: image, pose, step, question
            return Pose(...)                   # where to be next, or None to stop
        answer = None                          # read after the episode, for EQA benchmarks

`drones-render-agent --agent package.module:factory` loads one: `factory(**kwargs)` returns it,
with kwargs from `--agent-arg key=value`. It needs no base class, and nothing here imports it.

The agent moves kinematically: the pose it returns is where the drone is at the next step. There
is no dynamics and nothing collides with the scan, as in the benchmarks' own habitat-sim agents; a
drone-dynamics agent would step a CrazyFlow Sim and report its pose.

Poses are in the scene file's frame, as the benchmarks' are (see drones.sim.scene_view).
"""
import importlib
import math
from dataclasses import dataclass

import numpy as np

from drones.sim import eqa
from drones.sim.lens import euler_to_matrix

MAX_STEPS = 500


@dataclass(frozen=True)
class Pose:
    """The drone's position (m, scene file frame) and attitude (rad): yaw 0 faces +x and turns
    left, positive pitch puts the nose down."""
    pos: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    @property
    def rotation(self):
        return euler_to_matrix(self.yaw, self.pitch, self.roll)


@dataclass(frozen=True)
class Observation:
    image: np.ndarray       # (height, width, 3) uint8, through the camera's lens
    pose: Pose              # where the drone is
    step: int               # 0 at the start pose
    question: object        # the eqa.Question being answered, or None


class LookAround:
    """Turns once around in place, in `steps` equal turns, and stops: the panorama many EQA
    agents take before they move."""

    def __init__(self, steps=12):
        self.steps = int(steps)

    def reset(self, question, pose):
        self.start = pose

    def act(self, observation):
        if observation.step >= self.steps:
            return None
        p = observation.pose
        return Pose(p.pos, p.yaw + 2 * math.pi / self.steps, p.pitch, p.roll)


class FollowPath:
    """Flies the question's reference path (EXPRESS-Bench's walk, IndoorUAV's flight), point by
    point, facing along it, `height` above a habitat benchmark's floor. Stops at its end, or at
    once without a path."""

    def __init__(self, height=eqa.EYE_HEIGHT):
        self.height = float(height)

    def reset(self, question, pose):
        self.poses = eqa.path_poses(question, self.height) if question is not None else []

    def act(self, observation):
        step = observation.step + 1
        if step >= len(self.poses):
            return None
        pos, yaw = self.poses[step]
        return Pose(pos, yaw)


BUILTIN = {'look-around': LookAround, 'follow-path': FollowPath}


def make_agent(name, **kwargs):
    """A built-in agent by name, or `package.module:factory` called with `kwargs`."""
    if name in BUILTIN:
        return BUILTIN[name](**kwargs)
    module, sep, attr = name.partition(':')
    if not sep:
        raise ValueError(f'agent {name!r} is neither built in ({", ".join(BUILTIN)}) nor '
                         f'package.module:factory')
    return getattr(importlib.import_module(module), attr)(**kwargs)


def episode(view, agent, start, question=None, max_steps=MAX_STEPS):
    """Run `agent` in a SceneView from Pose `start`: yields each Observation it is shown, the
    start's included, until it stops or has taken `max_steps` steps. The drone in `view` is at
    the observation's pose while it is handled, for other cameras to film."""
    agent.reset(question, start)
    pose = start
    for step in range(max_steps + 1):
        observation = Observation(view.render_matrix(pose.pos, pose.rotation), pose, step,
                                  question)
        yield observation
        if step == max_steps:
            return
        pose = agent.act(observation)
        if pose is None:
            return
        pose = Pose(np.asarray(pose.pos, float), float(pose.yaw), float(pose.pitch),
                    float(pose.roll))
