"""Run an exported hover policy with numpy alone.

An artifact is a directory holding policy.json (what the policy observes and how its actions
scale) and actor.npz (the actor network's weights). drones.rl.export writes one after training; the
deploy script loads it on a machine that has neither JAX nor the simulator.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from drones.policy import interface

FORMAT_VERSION = 1


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
        """Normalised action(s) -> roll, pitch (rad), yaw rate (rad/s), thrust (N)."""
        return interface.decode_action(
            np, np.asarray(action, np.float64), max_tilt=self.max_tilt,
            max_yaw_rate=self.max_yaw_rate, hover_thrust=self.hover_thrust,
            thrust_min=self.thrust_min, thrust_max=self.thrust_max)


class Policy:
    """The trained actor, a tanh MLP, giving the policy's deterministic action."""

    def __init__(self, spec, layers):
        if 'camera' in spec.sensors:
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
        if data.get('format_version') != FORMAT_VERSION:
            raise ValueError(f'{directory}: artifact format {data.get("format_version")}, '
                             f'this code reads {FORMAT_VERSION}')
        spec = PolicySpec(**{**data, 'sensors': tuple(data['sensors']),
                             'target_height': tuple(data['target_height'])})
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
