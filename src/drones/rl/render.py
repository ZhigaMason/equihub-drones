"""Render a trained hover policy to a video.

    uv run --extra sim drones-render-hover runs/<name>
    uv run --extra sim drones-render-hover runs/<name> --camera top --episodes 3
    uv run --extra sim drones-render-hover runs/<name> --open-loop --out open-loop.gif
    uv run --extra sim drones-render-hover runs/<name> --camera deck

Flies fresh episodes with the policy's deterministic actions, as drones-eval-hover does, and films
them through CrazyFlow's MuJoCo renderer with the flight path drawn as a trail. Each episode samples
its own room and start from the seed. Writes runs/<name>/renders/<camera>-seed<N>.mp4 unless --out
says otherwise; its extension picks the format (.mp4, .gif, ...).

--camera deck films what the AI-deck would see instead, through the intrinsics calibrated on the
real deck (--intrinsics, recordings/intrinsics.json by default) and at their resolution unless
--width/--height rescale it. See drones.sim.deck_camera.

Headless Linux renders through EGL. Set MUJOCO_GL to use another backend, such as glfw on a desktop.
"""
import argparse
import os
import sys
from pathlib import Path

END_HOLD_SECONDS = 0.6   # each episode's last frame is held, so a crash is visible
CAMERAS = ('chase', 'top', 'deck')
TRAJECTORY_SIZE = (640, 480)   # default --width/--height of the chase and top cameras


def add_camera_arguments(parser, top_help):
    """--camera, --intrinsics, --width, --height and --font-scale, shared by the render CLIs."""
    parser.add_argument('--camera', choices=CAMERAS, default='chase',
                        help=f'chase: follows the drone; top: {top_help}; deck: what the '
                             f"AI-deck sees, through the deck's calibrated lens")
    parser.add_argument('--intrinsics', type=Path,
                        help='deck camera calibration (default: recordings/intrinsics.json)')
    parser.add_argument('--width', type=int,
                        help=f"default: {TRAJECTORY_SIZE[0]}, or the calibration's for deck")
    parser.add_argument('--height', type=int,
                        help=f"default: {TRAJECTORY_SIZE[1]}, or the calibration's for deck")
    parser.add_argument('--font-scale', type=int, choices=(100, 150, 200), default=100,
                        help='size of the text in the corner, in percent (default: 100, the '
                             'smallest; for relatively smaller text, raise --width/--height)')


def check_camera_arguments(parser, args):
    if args.camera != 'deck':
        args.width = args.width or TRAJECTORY_SIZE[0]
        args.height = args.height or TRAJECTORY_SIZE[1]
    elif (args.width is None) != (args.height is None):
        parser.error('--camera deck takes both --width and --height, or neither')
    if args.width is not None and min(args.width, args.height) < 16:
        parser.error('--width and --height must be at least 16')


def make_renderer(parser, env, args, **kwargs):
    """The renderer --camera names. `kwargs` go to a TrajectoryRenderer; the deck camera draws no
    markers or text. Call after mujoco is imported, and after anything that replaces the model."""
    if args.camera != 'deck':
        from drones.sim.render import TrajectoryRenderer
        return TrajectoryRenderer(env, args.camera, args.width, args.height,
                                  font_scale=args.font_scale, **kwargs)
    from drones.sim.deck_camera import DeckCamera
    from drones.sim.lens import DECK_MOUNT, DEFAULT_INTRINSICS, Intrinsics, Mount
    path = args.intrinsics or DEFAULT_INTRINSICS
    if not path.is_file():
        parser.error(f'no calibration at {path}; calibrate the deck (drones.vision.calibrate) or '
                     f'pass --intrinsics')
    intrinsics = Intrinsics.load(path)
    if args.width is not None:
        intrinsics = intrinsics.resized(args.width, args.height)
    # Mount the camera as the trained policy's own raycast camera was, where the task has one.
    sensors = getattr(env.config, 'sensors', None)
    mount = Mount(DECK_MOUNT.offset, sensors.camera_pitch if sensors else 0.0)
    return DeckCamera(env, intrinsics, mount=mount)


def film(env, act, key, renderer, writer, *, episodes, max_steps, stride, label):
    """Fly `episodes` fresh episodes in world 0, appending frames to `writer`.

    Returns (per-episode summaries, frames written).
    """
    import jax
    import numpy as np

    control_freq = env.config.control_freq
    hold = max(1, round(END_HOLD_SECONDS * control_freq / stride))

    def hud(state, episode, steps, status):
        s = state.sim.states
        return [
            (label, f'episode {episode}/{episodes}'),
            ('time', f'{steps / control_freq:.1f} s'),
            ('height', f'{float(s.pos[0, 0, 2]):.2f} m  (target {float(state.target[0]):.2f})'),
            ('speed', f'{float(np.linalg.norm(s.vel[0, 0])):.2f} m/s'),
            ('status', status),
        ]

    summaries, frames = [], 0
    for episode, k in enumerate(jax.random.split(key, episodes), 1):
        state, obs = env.reset(k)
        renderer.reset()
        trail = [np.asarray(state.sim.states.pos[0, 0])]
        errors, speeds, outcome, steps = [], [], 'no crash', 0
        writer.append_data(renderer.frame(state, trail, hud(state, episode, 0, 'flying')))
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
            errors.append(float(info['height_error'][0]))
            speeds.append(float(info['speed'][0]))
            if steps % stride == 0:
                writer.append_data(
                    renderer.frame(state, trail, hud(state, episode, steps, 'flying')))
                frames += 1
        last = renderer.frame(state, trail, hud(state, episode, steps, outcome))
        for _ in range(hold):
            writer.append_data(last)
        frames += hold
        summaries.append({
            'episode': episode,
            'outcome': outcome,
            'seconds': steps / control_freq,
            'height_error_m': float(np.mean(errors)) if errors else float('nan'),
            'speed_m_s': float(np.mean(speeds)) if speeds else float('nan'),
        })
    return summaries, frames


def open_writer(path, fps):
    import imageio.v2 as imageio

    if path.suffix.lower() == '.gif':
        return imageio.get_writer(path, mode='I', duration=1000 / fps, loop=0)
    # imageio otherwise rescales every frame to a multiple of 16 px, which would move every pixel
    # of a deck render off the calibration it was drawn through. yuv420p needs only even sizes.
    return imageio.get_writer(path, fps=fps, macro_block_size=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path, help='run directory written by drones-train-hover')
    parser.add_argument('--out', type=Path,
                        help='video file, format from its extension '
                             '(default: RUN/renders/CAMERA-seedN.mp4)')
    add_camera_arguments(parser, 'the whole room from above')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seconds', type=float,
                        help='stop each episode after this long (default: the full episode)')
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--open-loop', action='store_true',
                        help='fly zero action (calibrated hover thrust, no feedback) for '
                             'comparison')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error('--episodes must be at least 1')
    if args.fps <= 0 or (args.seconds is not None and args.seconds <= 0):
        parser.error('--fps and --seconds must be positive')
    check_camera_arguments(parser, args)

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

    from drones.rl.evaluate import load_run
    from drones.rl.networks import ActorCritic

    with jax.default_device(jax.devices(args.device)[0]):
        env, model, params = load_run(args.run, 1, args.device)
        if args.open_loop:
            label, act = 'open-loop hover', lambda o: jnp.zeros((1, env.action_size))
        else:
            label = 'policy'
            act = jax.jit(lambda o: model.apply(params, o, method=ActorCritic.act))
        control_freq = env.config.control_freq
        stride = max(1, round(control_freq / args.fps))
        fps = control_freq / stride
        max_steps = env.config.episode_steps
        if args.seconds is not None:
            max_steps = min(max_steps, max(1, round(args.seconds * control_freq)))
        suffix = '-open-loop' if args.open_loop else ''
        out = args.out or args.run / 'renders' / f'{args.camera}-seed{args.seed}{suffix}.mp4'
        out.parent.mkdir(parents=True, exist_ok=True)
        with (make_renderer(parser, env, args) as renderer,
              open_writer(out, fps) as writer):
            summaries, frames = film(env, act, jax.random.key(args.seed), renderer, writer,
                                     episodes=args.episodes, max_steps=max_steps, stride=stride,
                                     label=label)

    print(f'Wrote {out}: {frames} frames at {fps:g} fps, {args.camera} camera, '
          f'sensors {", ".join(env.config.sensors.enabled)}')
    for s in summaries:
        print(f'  episode {s["episode"]}: {s["outcome"]} after {s["seconds"]:.1f} s, '
              f'height error {s["height_error_m"]:.2f} m, speed {s["speed_m_s"]:.2f} m/s')


if __name__ == '__main__':
    main()
