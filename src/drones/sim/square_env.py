"""Flying a 1 x 1 m square on CrazyFlow, as a batch of differentiable JAX functions.

The policy commands attitude and collective thrust, as in the hover task. It observes the
firmware's state estimate: position, velocity and attitude as the Flow deck's Kalman filter
delivers them, modelled as the truth plus noise and a slow horizontal drift. It sees its error to a
time-parametrised reference (drones.policy.square) now and a second ahead, so it can anticipate the
corners.

Unlike HoverEnv, `step` is differentiable in the state and the action, so SHAC (drones.rl.shac) can
backpropagate through the physics. No rays are cast, every reward term is smooth, and metrics are
gradient-stopped. Worlds that finish restart inside `step`, and their new state carries no gradient
from the old.

Domain randomisation covers two sim-to-real gaps the hover task left open: thrust-to-weight and
action latency. so_rpy's lift is cmd_f_coef * thrust / mass with no offset, so a thrust gain covers
mass too. A residual wrench fitted to real flights (drones.rl.sysid) enters through CrazyFlow's
disturbance force and torque.
"""
import math
from dataclasses import dataclass

import crazyflow  # noqa: F401  Must precede scipy, see drones.sim.
import jax
import jax.numpy as jnp
from crazyflow import Control, Sim
from crazyflow.drones import load_params
from crazyflow.sim.data import SimData
from crazyflow.sim.functional import attitude_control
from crazyflow.utils import leaf_replace
from flax import struct

from drones.policy.interface import GYRO_SCALE, decode_action
from drones.policy.square import (LOOKAHEAD, LOOKAHEAD_DT, OBS_SIZE, POS_SCALE, VEL_SCALE,
                                  encode_square_obs, square_reference)
from drones.sim.calibration import calibrate_hover_thrust
from drones.sim.geometry import euler_to_quat, quat_to_matrix, wrap_angle, yaw_from_quat
from drones.sim.hover_env import YAW_BAND
from drones.sim.residual import Residual, residual_wrench

DRONE = 'cf21B_500'
CRASH_REWARD = -10.0
# Critic-only extras, on top of the policy observation: true position error (3), true velocity (3),
# angular velocity (3), reference phase as sin and cos (2), lap time (1), thrust gain (1).
PRIVILEGED_SIZE = 13


@dataclass(frozen=True)
class SquareConfig:
    num_envs: int = 64
    device: str = 'cpu'
    dynamics: str = 'so_rpy'
    sim_freq: int = 500
    control_freq: int = 50
    episode_seconds: float = 16.0
    # The square: side and corner rounding in metres; lap time and height are sampled per episode.
    side: float = 1.0
    corner_radius: float = 0.15
    lap_time: tuple[float, float] = (6.0, 10.0)
    height: tuple[float, float] = (0.8, 1.2)
    # Start error around the reference.
    start_pos_error: float = 0.1
    start_vel_error: float = 0.2
    start_tilt: float = 0.1
    # Domain randomisation: thrust gain within +/- thrust_gain_range of thrust_gain (it stands in
    # for mass too), and a latency of one of latency_steps control steps, per episode.
    thrust_gain: float = 1.0
    thrust_gain_range: float = 0.1
    latency_steps: tuple[int, ...] = (0, 1, 2)
    # State-estimate errors.
    pos_noise: float = 0.01          # m
    # m/sqrt(s), horizontal random walk; height comes from the ranger
    pos_drift: float = 0.01
    vel_noise: float = 0.03          # m/s
    attitude_noise: float = 0.01     # rad
    # Action scaling, as the hover task.
    max_tilt: float = 0.35
    max_yaw_rate: float = 1.5
    # Termination.
    min_height: float = 0.1
    max_tilt_terminate: float = 1.0
    max_error: float = 1.0

    def __post_init__(self):
        if not 0.0 < self.corner_radius < self.side / 2:
            raise ValueError(f'corner_radius must be between 0 and side / 2, got '
                             f'{self.corner_radius}')
        if not self.latency_steps or min(self.latency_steps) < 0:
            raise ValueError(f'latency_steps must be whole steps >= 0, got {self.latency_steps}')

    @property
    def episode_steps(self):
        return int(round(self.episode_seconds * self.control_freq))


@struct.dataclass
class SquareState:
    sim: SimData
    default_sim: SimData
    phase: jax.Array           # (n,) reference time at the episode start, s
    lap_time: jax.Array        # (n,)
    direction: jax.Array       # (n,) +1 counter-clockwise, -1 clockwise
    rotation: jax.Array        # (n,)
    origin: jax.Array          # (n, 3)
    ref_yaw: jax.Array         # (n,) the heading to hold
    yaw_cmd: jax.Array         # (n,) absolute yaw setpoint integrated from yaw-rate commands
    thrust_gain: jax.Array     # (n,)
    latency: jax.Array         # (n,) control steps
    actions: jax.Array         # (n, max latency + 1, 4) policy actions, newest first
    bias: jax.Array            # (n, 3) state-estimate position drift
    steps: jax.Array           # (n,)
    episode_return: jax.Array  # (n,)
    key: jax.Array


def yaw_setpoint_step(xp, yaw_cmd, yaw_rate_action, yaw, max_yaw_rate, control_freq):
    """The integrated yaw setpoint after one control step: yaw_cmd advanced by the commanded
    rate, kept within YAW_BAND of the heading so it cannot wind up."""
    yaw_cmd = yaw_cmd + yaw_rate_action * max_yaw_rate / control_freq
    error = xp.mod(yaw_cmd - yaw + math.pi, 2 * math.pi) - math.pi
    return yaw + xp.clip(error, -YAW_BAND, YAW_BAND)


class SquareEnv:
    """Vectorised square task. `reset(key)` and `step(state, action)` are jitted; `step` is
    differentiable."""

    action_size = 4
    policy_size = OBS_SIZE
    critic_size = OBS_SIZE + PRIVILEGED_SIZE
    image_shape = None

    def __init__(self, config: SquareConfig = SquareConfig(), residual=None):
        if config.sim_freq % config.control_freq:
            raise ValueError('sim_freq must be a multiple of control_freq')
        self.config = config
        self.num_envs = config.num_envs
        self.substeps = config.sim_freq // config.control_freq
        self.sim = Sim(n_worlds=config.num_envs, n_drones=1, drone=DRONE,
                       dynamics=config.dynamics, control=Control.attitude,
                       freq=config.sim_freq, device=config.device)
        self._sim_step = self.sim.build_step_fn()
        self._sim_reset = self.sim.build_reset_fn()

        params = load_params(DRONE)
        self.thrust_min = 4 * params['thrust_min']
        self.thrust_max = 4 * params['thrust_max']
        self.mass = float(self.sim.data.params.mass.ravel()[0])
        # so_rpy fits yaw as a second-order system with a steady-state gain of
        # -rpy_coef_z / cmd_rpy_coef_z != 1 (about 1.44 for cf21B_500), i.e. holding a command
        # of c settles at a yaw of about 1.44 * c, not c. `advance` uses this to command the yaw
        # that makes the model track yaw_cmd with unit gain, as the real drone's rate-mode yaw
        # setpoint does.
        p = self.sim.data.params
        self.yaw_gain = (float(-p.rpy_coef[2] / p.cmd_rpy_coef[2])
                         if config.dynamics == 'so_rpy' else 1.0)
        self.hover_thrust = calibrate_hover_thrust(self._sim_step, self.sim.default_data,
                                                   config.num_envs, self.mass, config.sim_freq)
        self.residual_model = Residual()
        self.residual = residual
        self._buffer = max(config.latency_steps) + 1
        self._reset_jit = jax.jit(self._reset)
        self.step = jax.jit(self._step)

    def policy_spec(self):
        """Fields of drones.policy.runtime.SquareSpec: what a deployed copy of the policy needs."""
        cfg = self.config
        return dict(control_freq=cfg.control_freq, side=cfg.side, corner_radius=cfg.corner_radius,
                    lap_time=cfg.lap_time, height=cfg.height, max_tilt=cfg.max_tilt,
                    max_yaw_rate=cfg.max_yaw_rate, hover_thrust=float(self.hover_thrust),
                    thrust_min=float(self.thrust_min), thrust_max=float(self.thrust_max))

    # ------------------------------------------------------------------ API
    def reset(self, key):
        """Start a fresh episode in every world. Returns (state, observation)."""
        return self._reset_jit(key, self.sim.default_data)

    def _reset(self, key, default):
        n = self.num_envs
        key, k_episode, k_obs = jax.random.split(key, 3)
        zeros = jnp.zeros(n)
        state = SquareState(
            sim=default, default_sim=default, phase=zeros, lap_time=jnp.ones(n),
            direction=jnp.ones(n), rotation=zeros, origin=jnp.zeros((n, 3)), ref_yaw=zeros,
            yaw_cmd=zeros, thrust_gain=jnp.ones(n), latency=jnp.zeros(n, jnp.int32),
            actions=jnp.zeros((n, self._buffer, 4)), bias=jnp.zeros((n, 3)),
            steps=jnp.zeros(n, jnp.int32), episode_return=zeros, key=key)
        state = self._reset_worlds(state, jnp.ones(n, bool), k_episode)
        return state, self._observe(state, k_obs)

    def _step(self, state, action):
        """Advance every world one control period. Returns (state, obs, reward, done, info).

        Worlds that finish restart in the same call; their observation is already the first of the
        next episode. `info['final_critic']` is the critic observation of the state each world
        reached, before any restart, which SHAC bootstraps from after a timeout.
        """
        cfg, n = self.config, self.num_envs
        action = jnp.clip(action, -1.0, 1.0)
        actions = jnp.concatenate([action[:, None], state.actions[:, :-1]], 1)
        applied = jnp.take_along_axis(actions, state.latency[:, None, None], 1)[:, 0]
        sim, yaw_cmd = self.advance(state.sim, applied, state.yaw_cmd, state.thrust_gain,
                                    self.residual)
        steps = state.steps + 1
        s = sim.states
        pos, vel, quat, ang_vel = s.pos[:, 0], s.vel[:, 0], s.quat[:, 0], s.ang_vel[:, 0]

        ref_pos, ref_vel = self.reference_at(state, state.phase + steps / cfg.control_freq)
        e2 = jnp.sum(jnp.square(pos - ref_pos), -1)
        ev2 = jnp.sum(jnp.square(vel - ref_vel), -1)
        up = quat_to_matrix(quat)[:, 2, 2]
        tilt = jnp.arccos(jnp.clip(jax.lax.stop_gradient(up), -1.0, 1.0))
        crashed = ((pos[:, 2] < cfg.min_height) | (tilt > cfg.max_tilt_terminate)
                   | (e2 > cfg.max_error ** 2))
        truncated = (steps >= cfg.episode_steps) & ~crashed
        done = crashed | truncated
        reward = self._reward(e2, ev2, up, ang_vel, action, state.actions[:, 0], crashed)
        episode_return = state.episode_return + reward

        key, k_reset, k_drift, k_obs = jax.random.split(state.key, 4)
        drift = (cfg.pos_drift * jnp.sqrt(1.0 / cfg.control_freq)
                 * jax.random.normal(k_drift, (n, 3)))
        state = state.replace(sim=sim, yaw_cmd=yaw_cmd, actions=actions, steps=steps,
                              episode_return=episode_return, key=key,
                              bias=state.bias + drift * jnp.array([1.0, 1.0, 0.0]))
        final = self._observe(state, k_obs)
        info = jax.lax.stop_gradient({
            'crashed': crashed,
            'truncated': truncated,
            'episode_return': jnp.where(done, episode_return, 0.0),
            'episode_length': jnp.where(done, steps, 0),
            'pos_error': jnp.sqrt(e2),
            'speed': jnp.linalg.norm(vel, axis=-1),
            'tilt': tilt,
        })
        info['final_critic'] = final['critic']
        # Most steps restart a few worlds, but skip the work entirely when none do.
        state = jax.lax.cond(done.any(), lambda st: self._reset_worlds(st, done, k_reset),
                             lambda st: st, state)
        return state, self._observe(state, k_obs), reward, done, info

    # ------------------------------------------------------------------ physics
    def advance(self, sim, action, yaw_cmd, thrust_gain, residual=None):
        """One control period of physics under `action`, the one the drone acts on now.

        Differentiable in everything. Returns (sim, yaw_cmd). drones.rl.sysid replays logged flights
        through it.
        """
        cfg = self.config
        quat = sim.states.quat[:, 0]
        yaw = yaw_from_quat(quat)
        yaw_cmd = yaw_setpoint_step(jnp, yaw_cmd, action[:, 2], yaw, cfg.max_yaw_rate,
                                    cfg.control_freq)
        roll, pitch, _, thrust = decode_action(
            jnp, action, max_tilt=cfg.max_tilt, max_yaw_rate=cfg.max_yaw_rate,
            hover_thrust=self.hover_thrust, thrust_min=self.thrust_min,
            thrust_max=self.thrust_max)
        # so_rpy's fitted yaw model is not unit-gain (see self.yaw_gain in __init__): scale the
        # commanded yaw so the model tracks yaw_cmd with unit gain at any heading, like the real
        # drone's rate-mode yaw. The error is wrapped so crossing +-pi does not jump.
        yaw_command = self.yaw_gain * (yaw + wrap_angle(yaw_cmd - yaw))
        cmd = jnp.stack([roll, pitch, yaw_command, thrust * thrust_gain], -1)[:, None]
        sim = attitude_control(sim, cmd)
        if residual is not None:
            s = sim.states
            force, torque = residual_wrench(self.residual_model, residual, s.vel[:, 0], quat,
                                            s.ang_vel[:, 0], action)
            # so_rpy adds these as disturbances every substep, and nothing clears them: they hold
            # for the whole control period.
            sim = sim.replace(states=s.replace(force=force[:, None], torque=torque[:, None]))
        return self._sim_step(sim, n_steps=self.substeps), yaw_cmd

    def with_states(self, sim, pos, vel, quat, ang_vel):
        """`sim` with every world's drone put in the given state; arrays (n, 3) or (n, 4)."""
        return sim.replace(states=sim.states.replace(
            pos=pos[:, None], vel=vel[:, None], quat=quat[:, None], ang_vel=ang_vel[:, None]))

    def reference_at(self, state, t):
        """Each world's reference at times t, shape (n,) or (n, k). Returns (pos, vel)."""
        extra = (1,) * (t.ndim - 1)

        def per_world(x):
            return x.reshape(x.shape[:1] + extra + x.shape[1:])

        cfg = self.config
        return square_reference(jnp, t, side=cfg.side, corner_radius=cfg.corner_radius,
                                lap_time=per_world(state.lap_time),
                                direction=per_world(state.direction),
                                rotation=per_world(state.rotation), origin=per_world(state.origin))

    # ------------------------------------------------------------------ episodes
    def _sample_episodes(self, key, n):
        cfg = self.config
        k = jax.random.split(key, 11)

        def uniform(k, lo, hi, shape=(n,)):
            return jax.random.uniform(k, shape, minval=lo, maxval=hi)

        lap_time = uniform(k[0], *cfg.lap_time)
        height = uniform(k[1], *cfg.height)
        direction = jnp.where(jax.random.bernoulli(k[2], 0.5, (n,)), 1.0, -1.0)
        rotation = uniform(k[3], -jnp.pi, jnp.pi)
        phase = uniform(k[4], 0.0, 1.0) * lap_time
        ref_yaw = uniform(k[5], -jnp.pi, jnp.pi)
        # Place each square so its reference passes through the world origin as the episode starts.
        start, start_vel = square_reference(jnp, phase, side=cfg.side,
                                            corner_radius=cfg.corner_radius, lap_time=lap_time,
                                            direction=direction, rotation=rotation,
                                            origin=jnp.zeros((n, 3)))
        origin = jnp.stack([-start[:, 0], -start[:, 1], height], -1)
        squash = jnp.array([1.0, 1.0, 0.5])
        pos = (jnp.zeros((n, 3)).at[:, 2].set(height)
               + uniform(k[6], -1.0, 1.0, (n, 3)) * cfg.start_pos_error * squash)
        vel = start_vel + uniform(k[7], -1.0, 1.0, (n, 3)) * cfg.start_vel_error * squash
        roll_pitch = uniform(k[8], -cfg.start_tilt, cfg.start_tilt, (n, 2))
        quat = euler_to_quat(roll_pitch[:, 0], roll_pitch[:, 1], ref_yaw)
        gain = cfg.thrust_gain * (1.0 + uniform(k[9], -cfg.thrust_gain_range,
                                                cfg.thrust_gain_range))
        latency = jax.random.choice(k[10], jnp.array(cfg.latency_steps, jnp.int32), (n,))
        return dict(lap_time=lap_time, direction=direction, rotation=rotation, phase=phase,
                    ref_yaw=ref_yaw, origin=origin, pos=pos, vel=vel, quat=quat, gain=gain,
                    latency=latency)

    def _reset_worlds(self, state, mask, key):
        """Start new episodes in the worlds selected by `mask`; leave the others untouched.

        The new episodes are sampled independently of the old state, and `jnp.where` passes no
        gradient to the side it does not pick, so restarted worlds carry no gradient across.
        """
        n = self.num_envs
        e = self._sample_episodes(key, n)
        sim = self._sim_reset(state.sim, state.default_sim, mask)
        sim = sim.replace(states=leaf_replace(sim.states, mask, pos=e['pos'][:, None],
                                              vel=e['vel'][:, None], quat=e['quat'][:, None],
                                              ang_vel=jnp.zeros((n, 1, 3))))

        def pick(new, old):
            return jnp.where(mask.reshape(mask.shape + (1,) * (old.ndim - 1)), new, old)

        return state.replace(
            sim=sim, phase=pick(e['phase'], state.phase),
            lap_time=pick(e['lap_time'], state.lap_time),
            direction=pick(e['direction'], state.direction),
            rotation=pick(e['rotation'], state.rotation), origin=pick(e['origin'], state.origin),
            ref_yaw=pick(e['ref_yaw'], state.ref_yaw), yaw_cmd=pick(e['ref_yaw'], state.yaw_cmd),
            thrust_gain=pick(e['gain'], state.thrust_gain),
            latency=pick(e['latency'], state.latency),
            actions=pick(jnp.zeros_like(state.actions), state.actions),
            bias=pick(jnp.zeros_like(state.bias), state.bias),
            steps=pick(jnp.zeros_like(state.steps), state.steps),
            episode_return=pick(jnp.zeros_like(state.episode_return), state.episode_return))

    # ------------------------------------------------------------------ observations
    def _observe(self, state, key):
        """The policy and critic observations of the current state.

        The estimate noise is drawn from `key` and does not depend on the state, so under
        differentiation it is a constant offset: the gradient of the estimate is the gradient of
        the truth.
        """
        cfg, n = self.config, self.num_envs
        s = state.sim.states
        pos, vel, quat, ang_vel = s.pos[:, 0], s.vel[:, 0], s.quat[:, 0], s.ang_vel[:, 0]
        k_pos, k_vel, k_att, k_yaw = jax.random.split(key, 4)
        pos_est = pos + state.bias + cfg.pos_noise * jax.random.normal(k_pos, (n, 3))
        vel_est = vel + cfg.vel_noise * jax.random.normal(k_vel, (n, 3))
        gravity = -quat_to_matrix(quat)[:, 2, :]    # body-frame gravity direction
        gravity = gravity + cfg.attitude_noise * jax.random.normal(k_att, (n, 3))
        gravity = gravity / jnp.linalg.norm(gravity, axis=-1, keepdims=True)
        yaw_est = yaw_from_quat(quat) + cfg.attitude_noise * jax.random.normal(k_yaw, (n,))

        t = state.phase + state.steps / cfg.control_freq
        ref_pos, ref_vel = self.reference_at(state, t)
        ahead, _ = self.reference_at(
            state, t[:, None] + LOOKAHEAD_DT * jnp.arange(1, LOOKAHEAD + 1))
        policy = encode_square_obs(jnp, pos_est=pos_est, vel_est=vel_est, yaw_est=yaw_est,
                                   gravity=gravity, ref_pos=ref_pos, ref_vel=ref_vel,
                                   lookahead_pos=ahead, ref_yaw=state.ref_yaw,
                                   prev_action=state.actions[:, 0])
        angle = 2 * jnp.pi * jnp.mod(t / state.lap_time, 1.0)
        privileged = jnp.concatenate([
            (pos - ref_pos) / POS_SCALE,
            vel / VEL_SCALE,
            ang_vel / GYRO_SCALE,
            jnp.sin(angle)[:, None],
            jnp.cos(angle)[:, None],
            state.lap_time[:, None] / 10.0,
            state.thrust_gain[:, None] - 1.0,
        ], -1)
        return {'policy': policy, 'critic': jnp.concatenate([policy, privileged], -1)}

    # ------------------------------------------------------------------ reward
    @staticmethod
    def _reward(e2, ev2, up, ang_vel, action, prev_action, crashed):
        """About 1 to 2.5 per step while flying; CRASH_REWARD for a crash.

        Smooth wherever it is differentiated. The exponentials reward tight tracking, the quadratic
        keeps a gradient far from the reference, and the constant 1 keeps the reward near 0 even at
        the 1 m crash limit, so crashing never pays. Tilt squared is 2 * (1 - R_zz), which avoids
        arccos's infinite slope when level.
        """
        tilt2 = 2.0 * (1.0 - up)
        reward = (1.0 + jnp.exp(-e2 / 0.05 ** 2) + 0.5 * jnp.exp(-ev2 / 0.25 ** 2) - e2
                  - 0.1 * tilt2 - 0.01 * jnp.sum(jnp.square(ang_vel), -1)
                  - 0.05 * jnp.sum(jnp.square(action - prev_action), -1))
        return jnp.where(crashed, CRASH_REWARD, reward)
