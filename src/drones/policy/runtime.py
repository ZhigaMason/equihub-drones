"""Run an exported policy, hover or square, with numpy alone.

An artifact is a directory holding policy.json (what the policy observes and how its actions
scale) and actor.npz (the actor network's weights). drones.rl.export writes one after training; the
deploy script loads it on a machine that has neither JAX nor the simulator.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from drones.policy import interface, square

FORMAT_VERSION = 2
# 1: hover artifacts from before the square task, without a `task` field.
READABLE_VERSIONS = (1, 2)
TUPLE_FIELDS = ('sensors', 'target_height', 'lap_time', 'height')


def _decode(spec, action):
    """Normalised action(s) -> roll, pitch (rad), yaw rate (rad/s), thrust (N)."""
    return interface.decode_action(
        np, np.asarray(action, np.float64), max_tilt=spec.max_tilt,
        max_yaw_rate=spec.max_yaw_rate, hover_thrust=spec.hover_thrust,
        thrust_min=spec.thrust_min, thrust_max=spec.thrust_max)


@dataclass(frozen=True)
class PolicySpec:
    sensors: tuple[str, ...]
    history: int
    control_freq: int
    target_height: tuple[float, float]
    range_max: float
    flow_gain: float
    max_tilt: float
    max_yaw_rate: float
    hover_thrust: float
    thrust_min: float
    thrust_max: float
    task: str = 'hover'
    format_version: int = FORMAT_VERSION

    def __post_init__(self):
        interface.validate(self.sensors)

    @property
    def frame_size(self):
        return interface.frame_size(self.sensors)

    @property
    def observation_size(self):
        return self.history * self.frame_size

    def decode(self, action):
        return _decode(self, action)


@dataclass(frozen=True)
class SquareSpec:
    """A square-flying policy: what it observes (drones.policy.square) and how its actions scale."""
    control_freq: int
    side: float
    corner_radius: float
    lap_time: tuple[float, float]
    height: tuple[float, float]
    max_tilt: float
    max_yaw_rate: float
    hover_thrust: float
    thrust_min: float
    thrust_max: float
    task: str = 'square'
    format_version: int = FORMAT_VERSION

    @property
    def observation_size(self):
        return square.OBS_SIZE

    def decode(self, action):
        return _decode(self, action)


SPECS = {'hover': PolicySpec, 'square': SquareSpec}


class Policy:
    """The trained actor, a tanh MLP, giving the policy's deterministic action."""

    def __init__(self, spec, layers):
        if 'camera' in getattr(spec, 'sensors', ()):
            raise ValueError('camera policies need their image encoder, which this artifact '
                             'format does not carry')
        self.spec = spec
        self.layers = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in layers]
        if self.layers[0][0].shape[0] != spec.observation_size:
            raise ValueError(f'first layer takes {self.layers[0][0].shape[0]} inputs, the spec '
                             f'describes {spec.observation_size}')
        if self.layers[-1][0].shape[1] != interface.ACTION_SIZE:
            raise ValueError(f'last layer gives {self.layers[-1][0].shape[1]} outputs, expected '
                             f'{interface.ACTION_SIZE}')

    def act(self, obs):
        x = np.asarray(obs, np.float32)
        for w, b in self.layers[:-1]:
            x = np.tanh(x @ w + b)
        w, b = self.layers[-1]
        return np.clip(x @ w + b, -1.0, 1.0)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'policy.json').write_text(json.dumps(asdict(self.spec), indent=2))
        arrays = {}
        for i, (w, b) in enumerate(self.layers):
            arrays[f'w{i}'], arrays[f'b{i}'] = w, b
        np.savez(directory / 'actor.npz', **arrays)
        return directory

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        data = json.loads((directory / 'policy.json').read_text())
        version = data.get('format_version')
        if version not in READABLE_VERSIONS:
            raise ValueError(f'{directory}: artifact format {version}, this code reads '
                             f'{list(READABLE_VERSIONS)}')
        task = data.get('task', 'hover')
        if task not in SPECS:
            raise ValueError(f'{directory}: unknown task {task!r}')
        data = {k: tuple(v) if k in TUPLE_FIELDS else v for k, v in data.items()}
        spec = SPECS[task](**{**data, 'task': task, 'format_version': FORMAT_VERSION})
        with np.load(directory / 'actor.npz') as arrays:
            count = sum(1 for name in arrays.files if name.startswith('w'))
            layers = [(arrays[f'w{i}'], arrays[f'b{i}']) for i in range(count)]
        return cls(spec, layers)


class PolicyRunner:
    """Feeds live readings to a policy exactly as the simulator does.

    Keeps the stacked frame history, seeded with the first frame as at the start of an episode, and
    feeds each action back in as the next frame's previous action.
    """

    def __init__(self, policy, target_height):
        self.policy = policy
        self.target = float(target_height)
        self.history = None
        self.prev_action = np.zeros(interface.ACTION_SIZE, np.float32)

    def step(self, *, flow_rate, zrange, ranges, gyro, gravity):
        spec = self.policy.spec
        f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
        frame = interface.encode_frame(
            np, spec.sensors, flow_rate=f32(flow_rate)[None], zrange=f32([zrange]),
            ranges=f32(ranges)[None], gyro=f32(gyro)[None], gravity=f32(gravity)[None],
            target=f32([self.target]), prev_action=self.prev_action[None],
            range_max=spec.range_max)[0]
        if self.history is None:
            self.history = np.repeat(frame[None], spec.history, 0)
        else:
            self.history = np.concatenate([self.history[1:], frame[None]], 0)
        action = self.policy.act(self.history.reshape(1, -1))[0]
        self.prev_action = action.astype(np.float32)
        return action


class SquareRunner:
    """Feeds the firmware's state estimate to a square policy exactly as the simulator does.

    The reference is at `origin` (x, y and the flight height) when `t` is 0. Its first edge runs
    along `rotation`, and the policy holds heading `ref_yaw`. `side` defaults to the trained side;
    a smaller one is useful on first flights.
    """

    def __init__(self, policy, *, origin, ref_yaw, lap_time, direction=1.0, rotation=0.0,
                 side=None):
        spec = policy.spec
        side = float(side or spec.side)
        if not 2 * spec.corner_radius < side:
            raise ValueError(f'side {side} m is too small for {spec.corner_radius} m corners')
        self.policy = policy
        self.params = dict(side=side, corner_radius=spec.corner_radius, lap_time=float(lap_time),
                           direction=float(direction), rotation=float(rotation),
                           origin=np.asarray(origin, np.float64))
        self.ref_yaw = float(ref_yaw)
        self.prev_action = np.zeros(interface.ACTION_SIZE, np.float32)

    def reference(self, t):
        """Reference (position, velocity) at t seconds since the square started."""
        return square.square_reference(np, np.asarray(t, np.float64), **self.params)

    def observe(self, t, *, pos, vel, yaw, gravity):
        ref_pos, ref_vel = self.reference(t)
        ahead, _ = self.reference(t + square.LOOKAHEAD_DT * np.arange(1, square.LOOKAHEAD + 1))

        def row(x):
            return np.asarray(x, np.float32)[None]

        return square.encode_square_obs(
            np, pos_est=row(pos), vel_est=row(vel), yaw_est=row(yaw), gravity=row(gravity),
            ref_pos=row(ref_pos), ref_vel=row(ref_vel), lookahead_pos=row(ahead),
            ref_yaw=row(self.ref_yaw), prev_action=self.prev_action[None])[0]

    def step(self, t, *, pos, vel, yaw, gravity):
        action = self.policy.act(self.observe(t, pos=pos, vel=vel, yaw=yaw,
                                              gravity=gravity)[None])[0]
        self.prev_action = action.astype(np.float32)
        return action
