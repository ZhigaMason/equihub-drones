"""Film an agent in a scanned scene: a chase camera and its own camera, side by side.

    uv run --extra sim drones-render-agent --benchmark indoor-uav --question 3 --agent follow-path
    uv run --extra sim drones-render-agent --benchmark hm-eqa --question 1          # look-around
    uv run --extra sim drones-render-agent --scene Bowlus --agent mypkg.agents:make --agent-arg k=v
    uv run --extra sim drones-render-agent --scene Bowlus --ask 'Find the sofa.' --agent ...

Left, the drone from behind (CrazyFlow's model of it, in the scan). Right, what the agent sees: the
AI-deck's view through the calibrated lens (--intrinsics, recordings/intrinsics.json by default),
or an ideal pinhole (--fov). That image is exactly the one the agent is given, at its own size; the
chase view matches its height. Under both, the step, the pose and the question or instruction,
and then whatever the agent has to say about that frame: the lines of its `caption`, if it has
one (the VLM pilot's is the chunk of actions it is flying, with its completion flag). A frame is
captioned once the agent has acted on it, so the last frame shows why it stopped.

The agent is a built-in (look-around, follow-path) or any `package.module:factory` that returns an
object with reset() and act(); see drones.sim.agents. With --benchmark, it starts from the
question's start pose (eqa.start_pose) and its scene is downloaded if needed. With --scene, it
starts over the scan's most open floor, and --ask gives it a question of your own.

Writes runs/agent-renders/<name>.mp4 unless --out says otherwise; the extension picks the format.
Without a display (a cluster node, over ssh) it renders through EGL, with one through GLFW; set
MUJOCO_GL to choose (egl, osmesa, glfw). The choice is made in drones/sim/__init__.py, because
mujoco settles it on first import, which is before main() runs.
"""
import argparse
import math
import sys
import textwrap
from pathlib import Path

BENCHMARKS = ('hm-eqa', 'mt-hm3d', 'express-bench', 'a-eqa', 'indoor-uav')
OUT_DIR = Path('runs/agent-renders')
CAPTION_LINES = 3     # the step and pose, then two for the question
AGENT_LINES = 6       # under them, for an agent with a `caption`
LABEL_RGB = (255, 255, 255)
CAPTION_BG = (24, 24, 24)


def parse_agent_args(parser, pairs):
    """--agent-arg key=value pairs as factory kwargs; values that parse as numbers become them."""
    kwargs = {}
    for pair in pairs:
        key, sep, value = pair.partition('=')
        if not sep:
            parser.error(f'--agent-arg takes key=value, not {pair!r}')
        for kind in (int, float):
            try:
                value = kind(value)
                break
            except ValueError:
                pass
        kwargs[key] = value
    return kwargs


def add_camera_args(parser):
    """The agent camera's options, shared with drones-benchmark."""
    parser.add_argument('--intrinsics', type=Path,
                        help="the agent camera's calibration (default: recordings/intrinsics.json)")
    parser.add_argument('--fov', type=float,
                        help='an ideal pinhole this many degrees across instead of --intrinsics')
    parser.add_argument('--width', type=int, help='agent camera width (default: the '
                        "calibration's, or 320 with --fov)")
    parser.add_argument('--height', type=int, help='agent camera height (default: the '
                        "calibration's, or 240 with --fov)")
    parser.add_argument('--no-mount', action='store_true',
                        help="put the camera at the drone's centre, level, instead of where the "
                             'AI-deck sits')
    parser.add_argument('--eye-height', type=float, default=None,
                        help='m above a habitat start the drone begins (default: 1.0)')


def camera_from(args, parser):
    """(Intrinsics, Mount) from add_camera_args' options; a usage error if they clash."""
    from drones.sim.lens import DECK_MOUNT, DEFAULT_INTRINSICS, Intrinsics, Mount

    if (args.width is None) != (args.height is None):
        parser.error('--width and --height go together')
    if args.fov is not None and args.intrinsics is not None:
        parser.error('--fov and --intrinsics are alternatives')
    if args.fov is not None:
        width, height = (args.width, args.height) if args.width else (320, 240)
        intrinsics = Intrinsics.from_fov(width, height, math.radians(args.fov))
    else:
        path = args.intrinsics or DEFAULT_INTRINSICS
        if not path.is_file():
            parser.error(f'no calibration at {path}; pass --intrinsics or --fov')
        intrinsics = Intrinsics.load(path)
        if args.width is not None:
            intrinsics = intrinsics.resized(args.width, args.height)
    return intrinsics, Mount() if args.no_mount else DECK_MOUNT


def pick_question(benchmark, number):
    """(scene name, question) for `benchmark`'s question `number`, its scene downloaded."""
    from drones.sim import eqa, scenes

    scene = eqa.locate(benchmark, number)
    if scene is None:
        sys.exit(f'{benchmark} has no question {number}')
    scenes.download([scene])
    eqa.prepare(benchmark, [scene])
    question = next((q for q in eqa.load(benchmark) if q.number == number), None)
    if question is None:
        sys.exit(f'{benchmark} question {number} did not load')
    return scene, question


def compose(chase, fpv, lines, scale, font, rows=CAPTION_LINES):
    """[chase | fpv] over a caption of `rows` lines, each pixel `scale` times, padded to even
    sizes for mp4."""
    import numpy as np
    from PIL import Image, ImageDraw

    top = np.concatenate([chase, fpv], axis=1)
    top = top.repeat(scale, axis=0).repeat(scale, axis=1)
    line_height = font.size + 4 * scale
    height = top.shape[0] + rows * line_height + 4 * scale
    width = top.shape[1]
    canvas = Image.new('RGB', (width + width % 2, height + height % 2), CAPTION_BG)
    canvas.paste(Image.fromarray(top), (0, 0))
    draw = ImageDraw.Draw(canvas)
    pad = 3 * scale
    draw.text((pad, pad), 'chase', fill=LABEL_RGB, font=font)
    draw.text((chase.shape[1] * scale + pad, pad), 'agent camera', fill=LABEL_RGB, font=font)
    for i, line in enumerate(lines[:rows]):
        draw.text((pad, top.shape[0] + 2 * scale + i * line_height), line, fill=LABEL_RGB,
                  font=font)
    return np.asarray(canvas)


def caption(observation, text, width_chars, agent_lines=None):
    """The caption's lines: the step and pose, the question `text` in two (blank if shorter, so
    what follows never moves), then `agent_lines` wrapped into AGENT_LINES, if there are any."""
    p = observation.pose
    x, y, z = p.pos
    head = (f'step {observation.step}   pos ({x:.2f}, {y:.2f}, {z:.2f}) m   '
            f'yaw {math.degrees(p.yaw):.0f}  pitch {math.degrees(p.pitch):.0f}  '
            f'roll {math.degrees(p.roll):.0f} deg')
    lines = [head] + (textwrap.wrap(text, width_chars) + ['', ''])[:CAPTION_LINES - 1]
    if agent_lines is None:
        return lines
    wrapped = [part for line in agent_lines for part in textwrap.wrap(str(line), width_chars)]
    return lines + (wrapped + [''] * AGENT_LINES)[:AGENT_LINES]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument('--benchmark', choices=BENCHMARKS,
                       help="start from one of this benchmark's questions")
    where.add_argument('--scene', help='a scene name, HM3D id or .glb, without a question')
    parser.add_argument('--question', type=int, default=1,
                        help="with --benchmark, the question's number in the benchmark's own "
                             'order, from 1 (default: 1)')
    parser.add_argument('--ask', metavar='TEXT',
                        help='with --scene, a question or instruction of your own for the agent')
    parser.add_argument('--agent', default='look-around',
                        help='look-around, follow-path, or package.module:factory '
                             '(default: look-around)')
    parser.add_argument('--agent-arg', action='append', default=[], metavar='KEY=VALUE',
                        help='keyword argument for the agent factory; repeat for more')
    add_camera_args(parser)
    parser.add_argument('--chase-distance', type=float, default=None,
                        help='m behind the drone (default: 1.0), less where the scan is closer')
    parser.add_argument('--steps', type=int, default=None,
                        help='stop the agent after this many steps (default: 500)')
    parser.add_argument('--fps', type=float, default=5.0)
    parser.add_argument('--scale', type=int, default=2,
                        help='each rendered pixel is this many in the video (default: 2)')
    parser.add_argument('--out', type=Path,
                        help='video file, format from its extension '
                             f'(default: {OUT_DIR}/NAME.mp4)')
    args = parser.parse_args(argv)
    if args.ask is not None and args.benchmark is not None:
        parser.error('--ask goes with --scene; a benchmark question brings its own text')
    if args.fps <= 0 or args.scale < 1 or (args.steps is not None and args.steps < 0):
        parser.error('--fps and --scale must be positive, --steps not negative')
    agent_kwargs = parse_agent_args(parser, args.agent_arg)

    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
        import imageio.v2 as imageio
    except ImportError:
        sys.exit('drones-render-agent needs the sim extra:  uv sync --extra sim')
    import warnings

    import numpy as np
    from PIL import ImageFont

    from drones.sim import agents, eqa
    from drones.sim.scene_view import CHASE_DISTANCE, DRONE, ChaseCamera, SceneView

    # imageio-ffmpeg starts ffmpeg with fork + exec, which is safe; Python warns only because JAX
    # has threads running.
    warnings.filterwarnings('ignore', message=r'os\.fork\(\) was called', category=RuntimeWarning)

    intrinsics, mount = camera_from(args, parser)
    try:
        agent = agents.make_agent(args.agent, **agent_kwargs)
    except (ValueError, ImportError, AttributeError) as exc:
        parser.error(f'--agent {args.agent}: {exc}')

    eye_height = eqa.EYE_HEIGHT if args.eye_height is None else args.eye_height
    question = None
    if args.benchmark is not None:
        scene, question = pick_question(args.benchmark, args.question)
        name = f'{args.benchmark}-{args.question}'
        text = question.text
    else:
        scene, name, text = args.scene, Path(args.scene).stem, args.ask or ''
        if args.ask:
            # No start, path or answer: the agent is given only the words.
            question = eqa.Question('ask', 0, name, args.ask, '', 'ask')
    name += f'-{args.agent.replace(":", ".")}'
    out = args.out or OUT_DIR / f'{name}.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading {scene} ...', flush=True)
    try:
        view = SceneView(scene, intrinsics, mount=mount, drone=DRONE)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    with view:
        if args.benchmark is not None:
            pos, yaw = eqa.start_pose(question, view.scene.origin, eye_height)
        else:
            pos, yaw = view.scene.origin + [0.0, 0.0, eye_height], 0.0
        start = agents.Pose(np.asarray(pos, float), yaw)
        chase_width = 2 * round(intrinsics.height * 4 / 3 / 2)
        chase = ChaseCamera(view, chase_width, intrinsics.height,
                            distance=args.chase_distance or CHASE_DISTANCE)
        font = ImageFont.load_default(size=11 * args.scale)
        width_chars = int((chase_width + intrinsics.width) * args.scale / (0.6 * font.size))
        max_steps = agents.MAX_STEPS if args.steps is None else args.steps
        # Decided before the first frame: every frame of a film is the same size.
        captioned = getattr(agent, 'caption', None) is not None
        rows = CAPTION_LINES + (AGENT_LINES if captioned else 0)

        frames = 0
        if out.suffix.lower() == '.gif':
            writer = imageio.get_writer(out, mode='I', duration=1000 / args.fps, loop=0)
        else:
            # Not rescaled to a multiple of 16: the agent's pixels stay the lens's.
            writer = imageio.get_writer(out, fps=args.fps, macro_block_size=2)
        with writer:
            episode = agents.episode(view, agent, start, question, max_steps)
            # Each frame once the agent has acted on it, so its caption is about that frame.
            for observation in agents.decided(episode):
                side = chase.render(observation.pose.pos, observation.pose.rotation)
                said = agent.caption if captioned else None
                writer.append_data(compose(side, observation.image,
                                           caption(observation, text, width_chars, said),
                                           args.scale, font, rows))
                frames += 1

    print(f'Wrote {out}: {frames} frames at {args.fps:g} fps, agent {args.agent}')
    if question is not None:
        answer = getattr(agent, 'answer', None)
        print(f'  question: {question.text}')
        if answer is not None:
            print(f'  agent answered: {answer}')
        if question.answer:
            print(f'  {question.reveal}: {question.answer}')


if __name__ == '__main__':
    main()
