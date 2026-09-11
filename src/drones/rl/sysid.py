"""Fit CrazyFlow to real flights: a thrust gain, the action latency, and a residual wrench.

Flights come from drones-fly-square's CSV logs, from the policy or firmware phase. They are cut into
windows of `horizon` control steps. The model starts each window from its logged state, replays the
logged actions through the simulator, and is scored on how far it drifts from the logged states.
Three models are compared on held-out flights:
    uncorrected       the simulator as it is
    gain_and_latency  a fitted thrust gain, at the best-fitting latency
    full              the same plus the residual network (drones.sim.residual)

so_rpy's lift is cmd_f_coef * thrust / mass with no offset, so mass cannot be told apart from the
thrust gain; the gain carries both. The logged states are the firmware's Kalman estimate, not ground
truth, so the fit matches the simulator to what the drone believed, estimator bias included.
"""
import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import crazyflow  # noqa: F401  Must precede scipy, see drones.sim.
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from drones.missions.fly_square import LOG_FORMAT
from drones.sim.geometry import yaw_from_quat
from drones.sim.residual import init_residual
from drones.sim.square_env import SquareConfig, SquareEnv, yaw_setpoint_step

FLIGHT_PHASES = ('policy', 'firmware')
# Error scales: a window drifting this far scores 1 in each term.
POS_SCALE, VEL_SCALE, ANGLE_SCALE = 0.05, 0.1, 0.05


@dataclass(frozen=True)
class Segment:
    """A stretch of flight logged without gaps, one row per control step."""
    flight: str
    pos: np.ndarray       # (T, 3) world
    vel: np.ndarray       # (T, 3) world
    quat: np.ndarray      # (T, 4) scalar-last
    ang_vel: np.ndarray   # (T, 3) body rad/s
    action: np.ndarray    # (T, 4) the normalised action the drone acted on


def load_flight(path, control_freq=50):
    """The segments of one drones-fly-square log, split wherever a control step is missing."""
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines or lines[0] != LOG_FORMAT:
        raise ValueError(f'{path}: not a square flight log (expected {LOG_FORMAT!r} first)')
    rows = [r for r in csv.DictReader(lines[1:]) if r['phase'] in FLIGHT_PHASES]
    groups, current, last = [], [], None
    for row in rows:
        t = float(row['time'])
        if current and t - last > 1.5 / control_freq:
            groups.append(current)
            current = []
        current.append(row)
        last = t
    if current:
        groups.append(current)
    return [_segment(path.stem, group) for group in groups]


def _segment(flight, rows):
    def columns(*names):
        return np.array([[float(r[n]) for n in names] for r in rows])

    return Segment(flight, columns('x', 'y', 'z'), columns('vx', 'vy', 'vz'),
                   columns('qx', 'qy', 'qz', 'qw'), columns('gyro_x', 'gyro_y', 'gyro_z'),
                   columns('a_roll', 'a_pitch', 'a_yaw', 'a_thrust'))


def load_flights(paths, control_freq=50):
    segments = [s for path in paths for s in load_flight(path, control_freq)]
    if not segments:
        raise ValueError('no policy or firmware phase in these flights')
    return segments


def _slice(segment, start, stop):
    return dataclasses.replace(segment, **{name: getattr(segment, name)[start:stop]
                                           for name in ('pos', 'vel', 'quat', 'ang_vel', 'action')})


def split_holdout(segments, fraction):
    """Hold out whole flights, the last `fraction` of them and at least one. With a single flight,
    hold out the last `fraction` of each of its segments instead."""
    flights = sorted({s.flight for s in segments})
    if len(flights) > 1:
        held = set(flights[-max(1, round(fraction * len(flights))):])
        return ([s for s in segments if s.flight not in held],
                [s for s in segments if s.flight in held])
    train, test = [], []
    for s in segments:
        cut = int(round(len(s.pos) * (1 - fraction)))
        train.append(_slice(s, 0, cut))
        test.append(_slice(s, cut, len(s.pos)))
    return train, test


@struct.dataclass
class Windows:
    pos: np.ndarray       # (W, K + 1, 3) logged states; index 0 is where each replay starts
    vel: np.ndarray
    quat: np.ndarray
    ang_vel: np.ndarray
    action: np.ndarray    # (W, K + max latency, 4); index max_latency + j was commanded at state j
    yaw_cmd: np.ndarray   # (W, max_latency + 1) the env's own yaw setpoint when replay starts,
                          # one value per latency candidate (see _segment_yaw_cmd)

    def __len__(self):
        return self.pos.shape[0]

    def take(self, index):
        return jax.tree.map(lambda x: x[index], self)


def _segment_yaw_cmd(segment, max_latency, max_yaw_rate, control_freq):
    """The env's own integrated yaw setpoint (SquareState.yaw_cmd) at every row of `segment`, for
    each latency candidate 0..max_latency. Replays yaw_setpoint_step along the whole segment from
    its first row's heading, applying the action that was in force `latency` rows earlier (actions
    before the segment start count as zero); returns an array (T, max_latency + 1)."""
    heading = np.asarray(yaw_from_quat(segment.quat))
    trajectories = []
    for latency in range(max_latency + 1):
        yaw_cmd = np.empty(len(segment.pos), np.float32)
        yaw_cmd[0] = heading[0]
        for t in range(len(segment.pos) - 1):
            idx = t - latency
            rate_action = segment.action[idx, 2] if idx >= 0 else 0.0
            yaw_cmd[t + 1] = yaw_setpoint_step(np, yaw_cmd[t], rate_action, heading[t],
                                               max_yaw_rate, control_freq)
        trajectories.append(yaw_cmd)
    return np.stack(trajectories, -1)


def make_windows(segments, horizon, max_latency, max_yaw_rate, control_freq):
    """Every window of `horizon` steps that has `max_latency` earlier actions to replay."""
    pieces = {name: [] for name in ('pos', 'vel', 'quat', 'ang_vel', 'action', 'yaw_cmd')}
    for seg in segments:
        yaw_cmd_by_row = _segment_yaw_cmd(seg, max_latency, max_yaw_rate, control_freq)
        for s in range(max_latency, len(seg.pos) - horizon):
            for name in ('pos', 'vel', 'quat', 'ang_vel'):
                pieces[name].append(getattr(seg, name)[s:s + horizon + 1])
            pieces['action'].append(seg.action[s - max_latency:s + horizon])
            pieces['yaw_cmd'].append(yaw_cmd_by_row[s])
    if not pieces['pos']:
        raise ValueError(f'no flight segment is longer than {horizon + max_latency} control steps')
    return Windows(**{k: np.stack(v).astype(np.float32) for k, v in pieces.items()})


def window_errors(predicted, windows):
    """Scaled squared error at every step of every window, (W, K)."""
    pos, vel, quat = predicted
    dp = jnp.sum(jnp.square(pos - windows.pos[:, 1:]), -1) / POS_SCALE ** 2
    dv = jnp.sum(jnp.square(vel - windows.vel[:, 1:]), -1) / VEL_SCALE ** 2
    # 4 (1 - <q, q'>^2) is the squared angle between them near zero, and smooth at zero.
    dot = jnp.sum(quat * windows.quat[:, 1:], -1)
    da = 4.0 * (1.0 - jnp.square(dot)) / ANGLE_SCALE ** 2
    return dp + dv + da


@dataclass(frozen=True)
class SysIdConfig:
    horizon: int = 10
    latencies: tuple[int, ...] = (0, 1, 2)
    gain_steps: int = 500          # thrust gain alone, per latency candidate
    residual_steps: int = 3000     # gain and residual together, at the chosen latency
    batch: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    holdout: float = 0.2
    seed: int = 0


@dataclass
class SysIdResult:
    thrust_gain: float
    latency: int
    residual: dict
    report: dict

    @property
    def improved(self):
        """Whether the corrected simulator beats the uncorrected one on held-out flights."""
        r = self.report
        return r['full']['score_horizon'] < r['uncorrected']['score_horizon']


class Replay:
    """Replays logged actions through the simulator, corrected by a model, from logged states."""

    def __init__(self, env_config, batch):
        self.env = SquareEnv(dataclasses.replace(env_config, num_envs=batch))
        self.batch = batch
        self._rollout = jax.jit(self.rollout, static_argnums=1)

    def rollout(self, model, latency, windows):
        """Predicted (pos, vel, quat), each (batch, K, ...), for exactly `batch` windows."""
        env = self.env
        horizon = windows.pos.shape[1] - 1
        first = windows.action.shape[1] - horizon      # the max latency the windows allow for
        sim = env.with_states(env.sim.default_data, windows.pos[:, 0], windows.vel[:, 0],
                              windows.quat[:, 0], windows.ang_vel[:, 0])
        gain = jnp.exp(model['log_gain']) * jnp.ones(self.batch)

        def body(carry, j):
            sim, yaw_cmd = carry
            action = jax.lax.dynamic_index_in_dim(windows.action, first + j - latency, 1,
                                                  keepdims=False)
            sim, yaw_cmd = env.advance(sim, action, yaw_cmd, gain, model['residual'])
            s = sim.states
            return (sim, yaw_cmd), (s.pos[:, 0], s.vel[:, 0], s.quat[:, 0])

        _, predicted = jax.lax.scan(body, (sim, windows.yaw_cmd[:, latency]),
                                    jnp.arange(horizon))
        return tuple(x.swapaxes(0, 1) for x in predicted)

    def fit(self, model, latency, windows, trainable, steps, config, log):
        """Adam on the mean window error, training only the entries of `model` named in
        `trainable`."""
        frozen = {k: v for k, v in model.items() if k not in trainable}
        params = {k: v for k, v in model.items() if k in trainable}
        optimizer = optax.adam(config.learning_rate)

        def loss(params, batch):
            value = jnp.mean(window_errors(self.rollout({**frozen, **params}, latency, batch),
                                           batch))
            if 'residual' in params:
                value = value + config.weight_decay * sum(
                    jnp.sum(jnp.square(x)) for x in jax.tree.leaves(params['residual']))
            return value

        @jax.jit
        def update(params, opt_state, batch):
            value, grads = jax.value_and_grad(loss)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, value

        rng = np.random.default_rng(config.seed)
        opt_state = optimizer.init(params)
        for step in range(steps):
            batch = windows.take(rng.integers(0, len(windows), self.batch))
            params, opt_state, value = update(params, opt_state, batch)
            if step % 100 == 0 or step == steps - 1:
                log(f'  latency {latency} | fitting {" + ".join(trainable)} | step {step:4d} | '
                    f'loss {float(value):.4f}')
        return {**frozen, **params}

    def evaluate(self, model, latency, windows):
        """Errors after one step and after the whole window, averaged over every window."""
        n = len(windows)
        chunks = []
        for start in range(0, n, self.batch):
            # The last chunk wraps round to fill the batch; the extra rows are trimmed below.
            batch = windows.take(np.arange(start, start + self.batch) % n)
            chunks.append(jax.device_get(self._rollout(model, latency, batch)))
        pos, vel, quat = (np.concatenate([c[i] for c in chunks])[:n] for i in range(3))
        pos_err = np.linalg.norm(pos - windows.pos[:, 1:], axis=-1)
        vel_err = np.linalg.norm(vel - windows.vel[:, 1:], axis=-1)
        dot = np.abs(np.sum(quat * windows.quat[:, 1:], -1))
        angle = 2 * np.arccos(np.clip(dot, 0.0, 1.0))
        score = np.asarray(window_errors((pos, vel, quat), windows))

        def at(i):
            return {'pos_m': float(pos_err[:, i].mean()), 'vel_m_s': float(vel_err[:, i].mean()),
                    'angle_rad': float(angle[:, i].mean())}

        return {'one_step': at(0), 'horizon_step': at(-1),
                'score_one_step': float(score[:, 0].mean()),
                'score_horizon': float(score[:, -1].mean())}


def identify(segments, env_config=None, config=SysIdConfig(), log=print):
    """Fit the thrust gain, latency and residual to `segments`, and report held-out errors."""
    env_config = env_config or SquareConfig()
    train_segments, test_segments = split_holdout(segments, config.holdout)
    max_latency = max(config.latencies)
    train = make_windows(train_segments, config.horizon, max_latency, env_config.max_yaw_rate,
                         env_config.control_freq)
    test = make_windows(test_segments, config.horizon, max_latency, env_config.max_yaw_rate,
                        env_config.control_freq)
    log(f'System ID: {len(train)} training windows, {len(test)} held out')
    replay = Replay(env_config, config.batch)
    start = {'log_gain': jnp.zeros(()), 'residual': init_residual(jax.random.key(config.seed))}

    report = {'uncorrected': replay.evaluate(start, 0, test)}
    candidates = {}
    for latency in config.latencies:
        model = replay.fit(start, latency, train, ('log_gain',), config.gain_steps, config, log)
        candidates[latency] = model, replay.evaluate(model, latency, test)
    latency = min(candidates, key=lambda k: candidates[k][1]['score_horizon'])
    gain_model, report['gain_and_latency'] = candidates[latency]
    full = replay.fit(gain_model, latency, train, ('log_gain', 'residual'),
                      config.residual_steps, config, log)
    report['full'] = replay.evaluate(full, latency, test)

    gain = float(jnp.exp(full['log_gain']))
    report.update(thrust_gain=gain, latency=latency, horizon=config.horizon,
                  note='fitted to the firmware state estimate, not ground truth')
    return SysIdResult(thrust_gain=gain, latency=latency, residual=full['residual'],
                       report=report)
