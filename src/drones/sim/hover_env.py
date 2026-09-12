"""Hover stabilisation on CrazyFlow, as a batch of pure JAX functions.

The policy flies the Crazyflie 2.1 Brushless the way the firmware's attitude controller is
commanded: roll, pitch, yaw rate and collective thrust. It never sees its position. It has to hold a
target height and cancel drift from the sensors it is given (`SensorConfig.enabled`): the Flow
deck's optical flow and z-ranger, the Multi-ranger deck, the IMU and a colour camera. The observation
layout and action scaling come from drones.policy.interface, shared with the deploy script.

Every episode samples a new room, start pose and target height. `reset` and `step` are pure
functions of an `EnvState`, so a whole rollout compiles into one `jax.lax.scan` and runs thousands
of worlds in parallel on a GPU, or a handful on a CPU.

Actions are normalised to [-1, 1]: roll and pitch scale to `max_tilt`, yaw rate to `max_yaw_rate`,
and thrust is piecewise linear with 0 at the calibrated hover thrust and +/-1 at the motor limits.
"""
from dataclasses import dataclass, field
from pathlib import Path

import crazyflow  # noqa: F401  Must precede scipy, see drones.sim.
import jax
import jax.numpy as jnp
from crazyflow import Control, Sim
from crazyflow.drones import load_params
from crazyflow.sim.data import SimData
from crazyflow.sim.functional import attitude_control
from crazyflow.utils import leaf_replace
from flax import struct
from mujoco import mjx

from drones.policy.interface import GYRO_SCALE, decode_action, encode_frame, frame_size
from drones.sim import sensors
from drones.sim.calibration import calibrate_hover_thrust
from drones.sim.geometry import euler_to_quat, quat_to_matrix, wrap_angle, yaw_from_quat
from drones.sim.sensors import DOWN, SensorConfig

ROOM_XML = Path(__file__).parent / 'assets' / 'room.xml'
DRONE = 'cf21B_500'
WALLS = ('wall_px', 'wall_nx', 'wall_py', 'wall_ny', 'ceiling')
WALL_HALF_THICKNESS = 0.05
# The integrated yaw setpoint stays within this band of the actual heading, so it cannot wind up.
YAW_BAND = 0.3
CRASH_REWARD = -10.0

# Critic-only extras, on top of the policy observation: position / room half-size (3),
# velocity (3), angular velocity (3), height error (1), room size (3).
PRIVILEGED_SIZE = 13
VELOCITY_SCALE = 2.0  # m/s


@dataclass(frozen=True)
class HoverConfig:
    num_envs: int = 64
    device: str = 'cpu'
    dynamics: str = 'so_rpy'       # fitted to real cf21B_500 flight data
    sim_freq: int = 500
    control_freq: int = 50
    episode_seconds: float = 10.0
    history: int = 3               # stacked frames in the policy observation
    # Room, sampled per episode: floor extent along x and y, and ceiling height, in metres.
    room_size: tuple[float, float] = (1.5, 5.0)
    room_height: tuple[float, float] = (1.6, 3.0)
    ceiling_margin: float = 0.6
    wall_margin: float = 0.4
    # Start state and target.
    target_height: tuple[float, float] = (0.5, 1.5)
    start_speed: float = 0.5
    start_tilt: float = 0.25
    start_rate: float = 1.0
    # Action scaling.
    max_tilt: float = 0.35
    max_yaw_rate: float = 1.5
    # Termination.
    crash_distance: float = 0.1    # centre-to-surface; the airframe's collision radius is 0.086
    max_tilt_terminate: float = 1.0
    sensors: SensorConfig = field(default_factory=SensorConfig)

    @property
    def episode_steps(self):
        return int(round(self.episode_seconds * self.control_freq))


@struct.dataclass
class EnvState:
    sim: SimData
    default_sim: SimData
    mjx: mjx.Data
    room: jax.Array            # (n, 3): half-width x, half-width y, ceiling height
    target: jax.Array          # (n,) target height
    yaw_cmd: jax.Array         # (n,) absolute yaw setpoint integrated from yaw-rate commands
    prev_action: jax.Array     # (n, 4)
    history: jax.Array         # (n, history, frame_size)
    ranges: jax.Array          # (n, 6) true ranger distances at the last observation
    steps: jax.Array           # (n,)
    episode_return: jax.Array  # (n,)
    key: jax.Array


class HoverEnv:
    """Vectorised hover task. Use `reset(key)` and `step(state, action)`; both are jitted."""

    def __init__(self, config: HoverConfig = HoverConfig()):
        if config.sim_freq % config.control_freq:
            raise ValueError('sim_freq must be a multiple of control_freq')
        self.config = config
        self.num_envs = config.num_envs
        self.substeps = config.sim_freq // config.control_freq
        self.frame_size = frame_size(config.sensors.enabled)

        self.sim = Sim(n_worlds=config.num_envs, n_drones=1, drone=DRONE,
                       dynamics=config.dynamics, control=Control.attitude,
                       freq=config.sim_freq, device=config.device, xml_path=ROOM_XML)
        self.mjx_model = self.sim.mjx_model
        self._sim_step = self.sim.build_step_fn()
        self._sim_reset = self.sim.build_reset_fn()
        self._wall_ids = jnp.array([self.sim.mj_model.body(n).mocapid.item() for n in WALLS])
        self.floor_geom = self.sim.mj_model.geom('floor').id
        self._geom_colours = jnp.asarray(self.sim.mj_model.geom_rgba[:, :3])

        params = load_params(DRONE)
        self.thrust_min = 4 * params['thrust_min']
        self.thrust_max = 4 * params['thrust_max']
        self.mass = float(self.sim.data.params.mass.ravel()[0])
        self.hover_thrust = calibrate_hover_thrust(self._sim_step, self.sim.default_data,
                                                   config.num_envs, self.mass, config.sim_freq)

        self._initial = (self.sim.default_data, self.sim.mjx_data)
        self._reset_jit = jax.jit(self._reset)
        self.step = jax.jit(self._step)

    # ------------------------------------------------------------------ sizes
    action_size = 4

    @property
    def policy_size(self):
        return self.config.history * self.frame_size

    @property
    def critic_size(self):
        return self.policy_size + PRIVILEGED_SIZE

    @property
    def image_shape(self):
        if not self.config.sensors.camera:
            return None
        width, height = self.config.sensors.camera_resolution
        return (height, width, 3)

    def policy_spec(self):
        """What a deployed copy of the policy needs: fields of drones.policy.runtime.PolicySpec."""
        cfg = self.config
        return dict(sensors=cfg.sensors.enabled, history=cfg.history,
                    control_freq=cfg.control_freq, target_height=cfg.target_height,
                    range_max=cfg.sensors.range_max, flow_gain=cfg.sensors.flow_gain,
                    max_tilt=cfg.max_tilt, max_yaw_rate=cfg.max_yaw_rate,
                    hover_thrust=float(self.hover_thrust), thrust_min=float(self.thrust_min),
                    thrust_max=float(self.thrust_max))

    # ------------------------------------------------------------------ API
    def reset(self, key):
        """Start a fresh episode in every world. Returns (state, observation)."""
        return self._reset_jit(key, *self._initial)

    def _reset(self, key, default, mjx_data):
        n, history = self.num_envs, self.config.history
        key, k_episode, k_sense = jax.random.split(key, 3)
        state = EnvState(
            sim=default, default_sim=default, mjx=mjx_data,
            room=jnp.ones((n, 3)), target=jnp.ones(n), yaw_cmd=jnp.zeros(n),
            prev_action=jnp.zeros((n, 4)), history=jnp.zeros((n, history, self.frame_size)),
            ranges=jnp.zeros((n, 6)), steps=jnp.zeros(n, jnp.int32),
            episode_return=jnp.zeros(n), key=key)
        state = self._reset_worlds(state, jnp.ones(n, bool), k_episode)
        return self._observe_after(state, jnp.ones(n, bool), k_sense)

    def _step(self, state, action):
        """Advance every world one control period. Returns (state, obs, reward, done, info).

        Worlds that finish are restarted in the same call: their returned observation is already
        the first of the next episode. Truncation is folded into `done` without bootstrapping.
        """
        cfg = self.config
        action = jnp.clip(action, -1.0, 1.0)
        cmd, yaw_cmd = self._command(action, state.yaw_cmd, state.sim.states.quat[:, 0])
        sim = self._sim_step(attitude_control(state.sim, cmd), n_steps=self.substeps)
        s = sim.states
        pos, vel, quat, ang_vel = s.pos[:, 0], s.vel[:, 0], s.quat[:, 0], s.ang_vel[:, 0]
        steps = state.steps + 1

        side_gap, floor_gap = self._gaps(pos, state.room)
        tilt = jnp.arccos(jnp.clip(quat_to_matrix(quat)[:, 2, 2], -1.0, 1.0))
        # The room is checked analytically; the last ranger readings also catch anything else
        # added to the scene, one control period late.
        crashed = ((jnp.minimum(side_gap, floor_gap) < cfg.crash_distance)
                   | (jnp.min(state.ranges, axis=-1) < cfg.crash_distance)
                   | (tilt > cfg.max_tilt_terminate))
        truncated = steps >= cfg.episode_steps
        done = crashed | truncated
        reward = self._reward(pos, vel, ang_vel, tilt, side_gap, state.target, action,
                              state.prev_action, crashed)
        episode_return = state.episode_return + reward
        info = {
            'crashed': crashed,
            'truncated': truncated & ~crashed,
            'episode_return': jnp.where(done, episode_return, 0.0),
            'episode_length': jnp.where(done, steps, 0),
            'height_error': jnp.abs(pos[:, 2] - state.target),
            'speed': jnp.linalg.norm(vel, axis=-1),
            'tilt': tilt,
        }

        key, k_reset, k_sense = jax.random.split(state.key, 3)
        state = state.replace(sim=sim, yaw_cmd=yaw_cmd, prev_action=action, steps=steps,
                              episode_return=episode_return, key=key)
        # Most steps restart a few worlds, but skip the work entirely when none do.
        state = jax.lax.cond(done.any(), lambda s: self._reset_worlds(s, done, k_reset),
                             lambda s: s, state)
        state, obs = self._observe_after(state, done, k_sense)
        return state, obs, reward, done, info

    # ------------------------------------------------------------------ actions
    def _command(self, action, yaw_cmd, quat):
        """Map normalised actions to CrazyFlow's attitude command [roll, pitch, yaw, thrust]."""
        cfg = self.config
        yaw = yaw_from_quat(quat)
        yaw_cmd = yaw_cmd + action[:, 2] * cfg.max_yaw_rate / cfg.control_freq
        yaw_cmd = yaw + jnp.clip(wrap_angle(yaw_cmd - yaw), -YAW_BAND, YAW_BAND)
        roll, pitch, _, thrust = decode_action(
            jnp, action, max_tilt=cfg.max_tilt, max_yaw_rate=cfg.max_yaw_rate,
            hover_thrust=self.hover_thrust, thrust_min=self.thrust_min,
            thrust_max=self.thrust_max)
        cmd = jnp.stack([roll, pitch, yaw_cmd, thrust], -1)
        return cmd[:, None, :], yaw_cmd

    # ------------------------------------------------------------------ episodes
    def _sample_episodes(self, key, n):
        cfg = self.config
        k = jax.random.split(key, 9)

        def uniform(k, lo, hi, shape=(n,)):
            return jax.random.uniform(k, shape, minval=lo, maxval=hi)

        half = uniform(k[0], *cfg.room_size, (n, 2)) / 2
        height = uniform(k[1], *cfg.room_height)
        room = jnp.concatenate([half, height[:, None]], -1)
        target = jnp.clip(uniform(k[2], *cfg.target_height), 0.3, height - cfg.ceiling_margin)
        margin = jnp.minimum(cfg.wall_margin, half - 0.15)
        xy = uniform(k[3], -1.0, 1.0, (n, 2)) * (half - margin)
        z = jnp.clip(target + uniform(k[4], -0.5, 0.5), 0.25, height - cfg.ceiling_margin)
        pos = jnp.concatenate([xy, z[:, None]], -1)
        vel = uniform(k[5], -1.0, 1.0, (n, 3)) * cfg.start_speed * jnp.array([1.0, 1.0, 0.5])
        roll_pitch = uniform(k[6], -cfg.start_tilt, cfg.start_tilt, (n, 2))
        yaw = uniform(k[7], -jnp.pi, jnp.pi)
        quat = euler_to_quat(roll_pitch[:, 0], roll_pitch[:, 1], yaw)
        ang_vel = uniform(k[8], -cfg.start_rate, cfg.start_rate, (n, 3))
        return room, target, pos, vel, quat, ang_vel, yaw

    def _reset_worlds(self, state, mask, key):
        """Start new episodes in the worlds selected by `mask`; leave the others untouched."""
        room, target, pos, vel, quat, ang_vel, yaw = self._sample_episodes(key, self.num_envs)
        sim = self._sim_reset(state.sim, state.default_sim, mask)
        sim = sim.replace(states=leaf_replace(sim.states, mask, pos=pos[:, None],
                                              vel=vel[:, None], quat=quat[:, None],
                                              ang_vel=ang_vel[:, None]))
        room = jnp.where(mask[:, None], room, state.room)
        return state.replace(
            sim=sim, mjx=self._place_walls(state.mjx, room), room=room,
            target=jnp.where(mask, target, state.target),
            yaw_cmd=jnp.where(mask, yaw, state.yaw_cmd),
            prev_action=jnp.where(mask[:, None], 0.0, state.prev_action),
            steps=jnp.where(mask, 0, state.steps),
            episode_return=jnp.where(mask, 0.0, state.episode_return))

    def _place_walls(self, mjx_data, room):
        """Move each world's walls and ceiling to its room, and update the scene geometry.

        Walls only move here, so this is the only place geometry is recomputed; the rays cast
        every step read the geom poses left by this call.
        """
        hx, hy, h = room[:, 0], room[:, 1], room[:, 2]
        t, zero = WALL_HALF_THICKNESS, jnp.zeros_like(hx)
        centres = jnp.stack([
            jnp.stack([hx + t, zero, zero], -1),
            jnp.stack([-hx - t, zero, zero], -1),
            jnp.stack([zero, hy + t, zero], -1),
            jnp.stack([zero, -hy - t, zero], -1),
            jnp.stack([zero, zero, h + t], -1),
        ], 1)
        mjx_data = mjx_data.replace(mocap_pos=mjx_data.mocap_pos.at[:, self._wall_ids].set(centres))
        return jax.vmap(mjx.kinematics, in_axes=(None, 0))(self.mjx_model, mjx_data)

    # ------------------------------------------------------------------ observations
    def _observe_after(self, state, fresh, key):
        """Sense, push a frame onto the history, and build the observation.

        Worlds flagged `fresh` have just started an episode, so their whole history is set to the
        first frame instead of mixing in frames from the previous episode.
        """
        cfg = self.config.sensors
        s = state.sim.states
        pos, vel, quat, ang_vel = s.pos[:, 0], s.vel[:, 0], s.quat[:, 0], s.ang_vel[:, 0]
        k_range, k_flow, k_imu = jax.random.split(key, 3)
        distances = sensors.ranger_distances(self.mjx_model, state.mjx, pos, quat)
        readings = sensors.ranger_readings(distances, k_range, cfg)
        flow = sensors.optical_flow(vel, ang_vel, quat, distances[:, DOWN], k_flow, cfg)
        gyro, gravity = sensors.imu(quat, ang_vel, k_imu, cfg)

        frame = encode_frame(
            jnp, cfg.enabled, flow_rate=flow / cfg.flow_gain, zrange=readings[:, DOWN],
            ranges=readings[:, :DOWN], gyro=gyro, gravity=gravity, target=state.target,
            prev_action=state.prev_action, range_max=cfg.range_max)
        history = jnp.concatenate([state.history[:, 1:], frame[:, None]], 1)
        first = jnp.repeat(frame[:, None], self.config.history, 1)
        history = jnp.where(fresh[:, None, None], first, history)
        state = state.replace(history=history, ranges=distances)

        policy = history.reshape(self.num_envs, -1)
        privileged = jnp.concatenate([
            pos / state.room,
            vel / VELOCITY_SCALE,
            ang_vel / GYRO_SCALE,
            pos[:, 2:3] - state.target[:, None],
            state.room / 5.0,
        ], -1)
        obs = {'policy': policy, 'critic': jnp.concatenate([policy, privileged], -1)}
        if cfg.camera:
            obs['image'] = sensors.render_camera(self.mjx_model, state.mjx, pos, quat,
                                                 self._geom_colours, self.floor_geom, cfg)
        return state, obs

    # ------------------------------------------------------------------ reward
    @staticmethod
    def _gaps(pos, room):
        """Distance to the nearest wall or ceiling, and to the floor."""
        hx, hy, h = room[:, 0], room[:, 1], room[:, 2]
        side = jnp.min(jnp.stack([hx - pos[:, 0], hx + pos[:, 0], hy - pos[:, 1],
                                  hy + pos[:, 1], h - pos[:, 2]], -1), -1)
        return side, pos[:, 2]

    @staticmethod
    def _reward(pos, vel, ang_vel, tilt, side_gap, target, action, prev_action, crashed):
        """Between 0 and 2 per step while flying; CRASH_REWARD for a crash.

        Staying up is worth 1 per step on its own, and the penalties are capped below that, so no
        state is worse than crashing. Without this floor an early, clumsy policy earns negative
        reward per step and learns to crash on purpose: shorter episodes, higher return.
        """
        on_height = jnp.exp(-jnp.square((pos[:, 2] - target) / 0.15))
        still = jnp.exp(-jnp.sum(jnp.square(vel), -1) / 0.1)
        penalty = (0.2 * jnp.square(tilt)
                   + 0.01 * jnp.sum(jnp.square(ang_vel), -1)
                   + 0.05 * jnp.sum(jnp.square(action - prev_action), -1)
                   + 0.5 * jnp.maximum(0.0, 0.4 - side_gap))
        reward = 1.0 + 0.5 * on_height + 0.5 * still - jnp.minimum(penalty, 1.0)
        return jnp.where(crashed, CRASH_REWARD, reward)
