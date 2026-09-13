"""Render a trained square policy to a video.

    uv run --extra sim drones-render-square runs/<name>
    uv run --extra sim drones-render-square runs/<name> --camera top --episodes 3
    uv run --extra sim drones-render-square runs/<name> --uncorrected --out uncorrected.mp4

Flies fresh episodes with the policy's deterministic actions, as drones-eval-square does, and films
them through CrazyFlow's MuJoCo renderer: the reference square in blue with the current target point
highlighted, and the flown path as an orange trail. Writes runs/<name>/renders/<camera>-seed<N>.mp4
unless --out says otherwise; its extension picks the format (.mp4, .gif, ...).

Headless Linux renders through EGL. Set MUJOCO_GL to use another backend, such as glfw on a desktop.
"""
import argparse
import os
import sys
from pathlib import Path

from drones.rl.render import END_HOLD_SECONDS, open_writer  # reused, not copied

# Room for the chase camera to orbit outside the flight path without being squeezed against an edge.
XY_MARGIN = 1.5   # m
Z_MARGIN = 1.0    # m, above the path's highest point


def lap_reference(env, state, samples=200):
    """Positions along one full lap of world 0's reference, from its current phase. (samples, 3)."""
    import jax.numpy as jnp
    import numpy as np

    lap_time = float(state.lap_time[0])
    t = jnp.asarray(float(state.phase[0]) + np.linspace(0.0, lap_time, samples))[None, :]
    pos, _ = env.reference_at(state, t)
    return np.asarray(pos[0])


def square_bounds(env, state, world):
    """A `TrajectoryRenderer` `bounds` function: a box around one lap of the reference square.

    `world` is always 0 -- drones-render-square only ever films a single-world env -- but the
    parameter is kept so this matches the `bounds(state, world)` signature it is passed as.
    """
    import numpy as np

    path = lap_reference(env, state)
    lo = np.array([path[:, 0].min() - XY_MARGIN, path[:, 1].min() - XY_MARGIN, 0.0])
    hi = np.array([path[:, 0].max() + XY_MARGIN, path[:, 1].max() + XY_MARGIN,
                   path[:, 2].max() + Z_MARGIN])
    return lo, hi


def film(env, act, key, renderer, writer, *, episodes, max_steps, stride, label):
    """Fly `episodes` fresh episodes in world 0, appending frames to `writer`.

    Returns (per-episode summaries, frames written).
    """
    import jax
    import numpy as np

    control_freq = env.config.control_freq
    hold = max(1, round(END_HOLD_SECONDS * control_freq / stride))

    def reference_now(state):
        """World 0's reference position and velocity at its current time."""
        t = state.phase[:1] + state.steps[:1] / control_freq
        pos, vel = env.reference_at(state, t)
        return np.asarray(pos[0]), np.asarray(vel[0])

    def hud(state, episode, steps, status, error, speed):
        lap_time = float(state.lap_time[0])
        elapsed = steps / control_freq
        return [
            (label, f'episode {episode}/{episodes}'),
            ('time', f'{elapsed:.1f} s'),
            ('lap', f'{elapsed % lap_time:.2f} of {lap_time:.1f} s lap'),
            ('error', f'{error:.2f} m'),
            ('speed', f'{speed:.2f} m/s'),
            ('status', status),
        ]

    summaries, frames = [], 0
    for episode, k in enumerate(jax.random.split(key, episodes), 1):
        state, obs = env.reset(k)
        renderer.reset()
        path = lap_reference(env, state)
        trail = [np.asarray(state.sim.states.pos[0, 0])]
        errors, speeds, outcome, steps = [], [], 'no crash', 0
        target, _ = reference_now(state)
        error = float(np.linalg.norm(trail[0] - target))
        speed = float(np.linalg.norm(np.asarray(state.sim.states.vel[0, 0])))
        writer.append_data(renderer.frame(
            state, trail, hud(state, episode, 0, 'flying', error, speed), path=path, target=target))
        frames += 1
        while steps < max_steps:
            next_state, obs, _, done, info = env.step(state, act(obs))
            steps += 1
            if bool(done[0]):
                # The env has already restarted this world, so the pre-step state is the last view.
                outcome = 'crashed' if bool(info['crashed'][0]) else 'no crash'
                break
            state = next_state
            trail.append(np.asarray(state.sim.states.pos[0, 0]))
            errors.append(float(info['pos_error'][0]))
            speeds.append(float(info['speed'][0]))
            if steps % stride == 0:
                error, speed = float(info['pos_error'][0]), float(info['speed'][0])
                target, _ = reference_now(state)
                writer.append_data(renderer.frame(
                    state, trail, hud(state, episode, steps, 'flying', error, speed),
                    path=path, target=target))
                frames += 1
        last = renderer.frame(state, trail, hud(state, episode, steps, outcome, error, speed),
                              path=path, target=target)
        for _ in range(hold):
            writer.append_data(last)
        frames += hold
        summaries.append({
            'episode': episode,
            'outcome': outcome,
            'seconds': steps / control_freq,
            'laps': (steps / control_freq) / float(state.lap_time[0]),
            'pos_error_m': float(np.mean(errors)) if errors else float('nan'),
            'speed_m_s': float(np.mean(speeds)) if speeds else float('nan'),
        })
    return summaries, frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path, help='run directory written by drones-train-square')
    parser.add_argument('--out', type=Path,
                        help='video file, format from its extension '
                             '(default: RUN/renders/CAMERA-seedN.mp4)')
    parser.add_argument('--camera', choices=('chase', 'top'), default='chase',
                        help='chase: follows the drone; top: the whole square from above')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seconds', type=float,
                        help='stop each episode after this long (default: the full episode)')
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--font-scale', type=int, choices=(100, 150, 200), default=100,
                        help='size of the text in the corner, in percent (default: 100, the '
                             'smallest; for relatively smaller text, raise --width/--height)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--open-loop', action='store_true',
                        help='fly zero action (calibrated hover thrust, no feedback) for '
                             'comparison')
    parser.add_argument('--uncorrected', action='store_true',
                        help='ignore a fitted residual (runs from drones-finetune-square)')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error('--episodes must be at least 1')
    if args.fps <= 0 or (args.seconds is not None and args.seconds <= 0):
        parser.error('--fps and --seconds must be positive')
    if min(args.width, args.height) < 16:
        parser.error('--width and --height must be at least 16')

    # The GL backend is fixed when mujoco is imported, so this goes before anything imports it.
    if sys.platform.startswith('linux'):
        os.environ.setdefault('MUJOCO_GL', 'egl')
    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
        import imageio  # noqa: F401
    except ImportError:
        sys.exit('Rendering needs the sim extra:  uv sync --extra sim')
    import warnings

    import jax
    import jax.numpy as jnp

    # imageio-ffmpeg starts ffmpeg with fork + exec, which is safe; Python warns only because JAX
    # has threads running.
    warnings.filterwarnings('ignore', message=r'os\.fork\(\) was called', category=RuntimeWarning)

    from drones.rl.evaluate_square import load_square_run
    from drones.sim.render import TrajectoryRenderer

    with jax.default_device(jax.devices(args.device)[0]):
        env, agent, params = load_square_run(args.run, 1, args.device,
                                             corrected=not args.uncorrected)
        if args.open_loop:
            label, act = 'open-loop', lambda o: jnp.zeros((1, env.action_size))
        else:
            label = 'policy'
            act = jax.jit(lambda o: agent.act(params['actor'], o))
        control_freq = env.config.control_freq
        stride = max(1, round(control_freq / args.fps))
        fps = control_freq / stride
        max_steps = env.config.episode_steps
        if args.seconds is not None:
            max_steps = min(max_steps, max(1, round(args.seconds * control_freq)))
        suffix = '-open-loop' if args.open_loop else ''
        out = args.out or args.run / 'renders' / f'{args.camera}-seed{args.seed}{suffix}.mp4'
        out.parent.mkdir(parents=True, exist_ok=True)
        with (TrajectoryRenderer(env, args.camera, args.width, args.height,
                                 font_scale=args.font_scale,
                                 bounds=lambda s, w: square_bounds(env, s, w)) as renderer,
              open_writer(out, fps) as writer):
            summaries, frames = film(env, act, jax.random.key(args.seed), renderer, writer,
                                     episodes=args.episodes, max_steps=max_steps, stride=stride,
                                     label=label)

    print(f'Wrote {out}: {frames} frames at {fps:g} fps, {args.camera} camera')
    for s in summaries:
        print(f'  episode {s["episode"]}: {s["outcome"]} after {s["seconds"]:.1f} s '
              f'({s["laps"]:.2f} laps), error {s["pos_error_m"]:.2f} m, '
              f'speed {s["speed_m_s"]:.2f} m/s')


if __name__ == '__main__':
    main()
