"""Fly an agent over every question of a benchmark, overnight, and keep what it did.

    uv run --extra sim drones-benchmark run --benchmark hm-eqa \\
        --agent drones.vlm.agent:make --agent-arg backend=claude-code
    uv run --extra sim drones-benchmark score runs/benchmarks/*

A run is meant to take nights: one benchmark in full is hundreds of questions and thousands of
model calls. So it resumes. Each finished question is one line of results.jsonl, written after
its episode folder, and a start skips those lines' questions and deletes anything else it
finds, so a run stopped anywhere -- a usage limit, Ctrl-C, a power cut -- flies the
interrupted question again from its start next time. Every line is fsynced, and results.jsonl
and config.json are only ever replaced whole (written aside, fsynced, renamed), so a power cut
loses at most the question being flown. One run at a time per folder: a lock file enforces
it. It stops at the first failure rather than skipping the question: a question that cannot be
flown tonight (a usage limit, a scene that would not download) is flown tomorrow, not dropped
from the score.

The budget is counted in decisions, one per model call for the VLM pilot. EQA questions get
Explore-EQA's int(sqrt(floor area) * 3); IndoorUAV's flights twice their reference length in
`reach`es. An EQA agent that has `conclude` is asked for its answer when the budget runs out,
as Explore-EQA always answers at the end.

Every model call an agent records (`calls`, see drones.vlm.agent) is saved with the frame it
saw, as data to distil a smaller pilot from. Agents are read by attribute, so this imports
none: `chunk_size` (steps per decision, 1 without it), `reach` (m one decision flies at most),
`calls`, `stats`, `answer`, `error` and `conclude(observation)` are all optional.

Scoring is drones.sim.scoring; the metrics themselves are drones.sim.metrics.
"""
import argparse
import contextlib
import fcntl
import functools
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from drones.sim import agents, eqa, metrics

RUNS_DIR = Path('runs/benchmarks')
EQA = ('hm-eqa', 'mt-hm3d', 'express-bench', 'a-eqa')
TEST_SPLITS = ('test seen', 'test unseen')      # how eqa writes IndoorUAV's split names
# Settings that change what a result means; a resume must keep them. --questions may change.
COMPARED = ('benchmark', 'agent', 'agent_args', 'camera', 'eye_height', 'chunk_size', 'reach')


class Stopped(Exception):
    """The run stopped; the message says where and why. Everything finished is kept."""


def select(benchmark, questions, numbers=None):
    """`questions` numbered within `numbers` (first, last), and for indoor-uav only its test
    splits: its other trajectories are training data."""
    keep = [q for q in questions
            if numbers is None or numbers[0] <= q.number <= numbers[1]]
    if benchmark == 'indoor-uav':
        # eqa writes the category as 'traj_3, test seen, easy', or 'traj_3, unsplit'.
        keep = [q for q in keep if q.category.split(', ')[1:2] in ([s] for s in TEST_SPLITS)]
    return keep


def finished(results):
    """{number: results line} of the questions in `results`. A line cut off by a kill is not
    a finished question."""
    done = {}
    if not Path(results).exists():
        return done
    for line in Path(results).read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[row['number']] = row
    return done


def _plain(value):
    """`value` as JSON can hold it."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _atomic_write(path, text):
    """Replace `path` with `text` whole: a kill leaves the old file or the new, never a torn
    one. write_text truncates first, and a kill then loses what the file held."""
    path = Path(path)
    aside = path.with_suffix(path.suffix + '.tmp')
    with open(aside, 'w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(aside, path)


def _check_config(folder, config):
    path = folder / 'config.json'
    config = _plain(config)
    if not path.exists():
        _atomic_write(path, json.dumps(config, indent=2))
        return
    old = json.loads(path.read_text())
    changed = [f'{key}: {old.get(key)!r} then, {config.get(key)!r} now'
               for key in COMPARED if old.get(key) != config.get(key)]
    if changed:
        raise Stopped(f'{folder} was run with other settings ({"; ".join(changed)}). Pass '
                      '--name for a new run, or delete the folder to start again.')


def _clean(folder, done):
    """Delete the episode folders of unfinished questions, and fix a torn last line."""
    episodes = folder / 'episodes'
    for path in episodes.iterdir():
        # A stray file is none of ours to delete, and rmtree on it would stop every start.
        if path.is_dir() and (not path.name.isdigit() or int(path.name) not in done):
            shutil.rmtree(path)
    results = folder / 'results.jsonl'
    if results.exists():
        canonical = ''.join(json.dumps(row) + '\n' for row in done.values())
        # Only when a torn line is to go: a rewrite on every start is a window to be killed in.
        if results.read_text() != canonical:
            _atomic_write(results, canonical)


def _lift(question, eye_height):
    """IndoorUAV's paths are at flight height; habitat's are on the floor."""
    return 0.0 if question.benchmark == 'indoor-uav' else eye_height


def _reference(question, eye_height):
    """(M, 3) the question's reference path at the height the drone flies it; (0, 3) without."""
    poses = eqa.path_poses(question, eye_height)
    return np.array([p for p, _ in poses], float).reshape(-1, 3)


def _budget(question, view, reference, reach):
    if question.benchmark == 'indoor-uav':
        return metrics.path_budget(metrics.path_length(reference), reach)
    scene = view.scene
    floor = (question.start if question.start is not None else scene.origin)[2]
    # The scene's vertices are moved so its origin is the world's; benchmark poses are not.
    return metrics.eqa_budget(metrics.floor_area(scene.vertices + scene.origin, floor))


def _fly(view, agent, question, start, budget):
    """Fly one episode; (stop reason, decisions, chunk size, poses, last observation)."""
    chunk = int(getattr(agent, 'chunk_size', 1))
    poses, last = [], None
    # Only poses are kept: the frames of a long episode would fill the memory.
    for observation in agents.episode(view, agent, start, question, budget * chunk):
        poses.append(observation.pose)
        last = observation
    if last.step == budget * chunk:
        stop = 'budget'
        if question.benchmark in EQA and hasattr(agent, 'conclude'):
            agent.conclude(last)
    else:
        stop = 'failed' if getattr(agent, 'error', None) else 'done'
    decisions = budget if stop == 'budget' else last.step // chunk + 1
    return stop, decisions, chunk, poses, last


def _write_episode(folder, number, pos, yaw, reference, calls):
    """The episode's folder, written aside and moved into place whole."""
    from PIL import Image

    part = folder / 'episodes' / f'{number}.part'
    shutil.rmtree(part, ignore_errors=True)
    (part / 'calls').mkdir(parents=True)
    np.savez_compressed(part / 'trajectory.npz', pos=pos, yaw=yaw, reference=reference)
    with open(part / 'calls.jsonl', 'w') as f:
        for k, call in enumerate(calls):
            row = {'k': k, **{key: _plain(v) for key, v in call.items() if key != 'image'}}
            image = call.get('image')
            if isinstance(image, np.ndarray) and image.ndim == 3:
                row['image'] = f'calls/{k:03d}.png'
                Image.fromarray(np.ascontiguousarray(image, np.uint8)).save(part / row['image'])
            f.write(json.dumps(row) + '\n')
    final = folder / 'episodes' / str(number)
    shutil.rmtree(final, ignore_errors=True)
    part.rename(final)


def _start_height(view, question, start, eye_height):
    """m the start is above the floor: the eye height on habitat benchmarks, but IndoorUAV
    starts at the dataset's own flight height, so there it is measured down to the scan."""
    if question.benchmark != 'indoor-uav' or not hasattr(view, 'distance'):
        return float(eye_height)
    down = view.distance(start.pos, np.array([0.0, 0.0, -1.0]))
    return float(down) if np.isfinite(down) else float(eye_height)


def _question(view, agent, question, eye_height):
    """Fly `question` in `view`: (results line, flown positions, yaws, reference, calls)."""
    started = time.perf_counter()
    pos, yaw = eqa.start_pose(question, view.scene.origin, eye_height)
    start = agents.Pose(np.asarray(pos, float), float(yaw))
    height = _start_height(view, question, start, eye_height)
    if hasattr(agent, 'start_altitude'):
        agent.start_altitude = height     # reset() reads it: the agent's floor is below it
    reference = _reference(question, eye_height)
    reach = float(getattr(agent, 'reach', metrics.DEFAULT_REACH))
    budget = _budget(question, view, reference, reach)
    stop, decisions, chunk, poses, last = _fly(view, agent, question, start, budget)
    flown = np.array([p.pos for p in poses], float)
    yaws = np.array([p.yaw for p in poses], float)
    calls = list(getattr(agent, 'calls', None) or [])
    goal = None
    if question.goal is not None:
        goal = np.asarray(question.goal, float) + [0.0, 0.0, _lift(question, eye_height)]
    row = {
        'benchmark': question.benchmark, 'number': question.number, 'scene': question.scene,
        'category': question.category, 'question': question.text,
        'choices': list(question.choices), 'truth': question.answer,
        'answer': getattr(agent, 'answer', None), 'stop': stop,
        'start': 'benchmark' if question.start is not None else 'origin',
        'start_altitude': height,
        'decisions': decisions, 'budget': budget, 'steps': last.step, 'chunk_size': chunk,
        'path_length': metrics.path_length(flown), 'final_pos': flown[-1].tolist(),
        'final_yaw': float(yaws[-1]), 'goal': None if goal is None else goal.tolist(),
        'reference_length': metrics.path_length(reference) if len(reference) else None,
        'seconds': time.perf_counter() - started, 'model_calls': len(calls),
        'model_seconds': float(sum(c.get('seconds', 0.0) for c in calls)),
        'stats': _plain(getattr(agent, 'stats', None)),
    }
    return row, flown, yaws, reference, calls


def run(benchmark, questions, agent, folder, open_view, config, eye_height=eqa.EYE_HEIGHT,
        log=print):
    """Fly `agent` over `questions` of `benchmark` not yet in `folder`/results.jsonl, a scene at
    a time through `open_view(scene)` (a context manager giving a SceneView). Raises Stopped
    at the first failure, with the interrupted question's folder removed."""
    folder = Path(folder)
    (folder / 'episodes').mkdir(parents=True, exist_ok=True)
    with open(folder / '.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Stopped(f'another drones-benchmark is already running in {folder}') from None
        # Held until the file closes, which is when this returns or raises.
        _run(benchmark, questions, agent, folder, open_view, config, eye_height, log)


def _run(benchmark, questions, agent, folder, open_view, config, eye_height, log):
    _check_config(folder, config)
    done = finished(folder / 'results.jsonl')
    _clean(folder, done)
    todo = [q for q in questions if q.number not in done]
    log(f'{benchmark}: {len(done)} done, {len(todo)} to fly, into {folder}')
    current = None
    try:
        for scene, group in eqa.by_scene(todo).items():
            current = group[0]
            with open_view(scene) as view:
                for question in group:
                    current = question
                    row, flown, yaws, reference, calls = _question(view, agent, question,
                                                                   eye_height)
                    # The folder first, then the line: a line means its folder is whole.
                    _write_episode(folder, question.number, flown, yaws, reference, calls)
                    with open(folder / 'results.jsonl', 'a') as f:
                        f.write(json.dumps(_plain(row)) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                    log(f'{benchmark} {question.number}: {row["stop"]}, '
                        f'{row["decisions"]}/{row["budget"]} decisions, answer '
                        f'{row["answer"]!r}, truth {row["truth"]!r} ({row["seconds"]:.0f} s)')
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        if isinstance(exc, Stopped):
            raise
        if current is not None:
            shutil.rmtree(folder / 'episodes' / f'{current.number}.part', ignore_errors=True)
            if current.number not in finished(folder / 'results.jsonl'):
                # An interrupt after the folder was moved into place, before its line.
                shutil.rmtree(folder / 'episodes' / str(current.number), ignore_errors=True)
        where = f'{benchmark} question {current.number}' if current else benchmark
        reason = 'interrupted' if isinstance(exc, KeyboardInterrupt) else str(exc) or repr(exc)
        raise Stopped(f'stopped at {where}: {reason}. Run the same command again to resume '
                      'there.') from exc


def parse_numbers(text):
    """'7' or '2-40': the question numbers to keep, (first, last)."""
    first, _, last = text.partition('-')
    first, last = int(first), int(last or first)
    if last < first:
        raise ValueError(f'{text}: the first number comes first')
    return first, last


def load_questions(benchmark):
    """Every question of `benchmark`. IndoorUAV's test scenes have their instructions fetched
    first: eqa.load returns only scenes that `prepare` has seen."""
    if benchmark == 'indoor-uav':
        eqa.prepare('indoor-uav', eqa.split_scenes(('test_seen.csv', 'test_unseen.csv')))
    return eqa.load(benchmark)


def main(argv=None):
    from drones.sim import render_agent

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    fly = commands.add_parser('run', help='fly an agent over one or more benchmarks')
    fly.add_argument('--benchmark', nargs='+', required=True, choices=(*eqa.BENCHMARKS, 'all'),
                     help='one or more benchmarks, or all of them in order')
    fly.add_argument('--questions', metavar='A-B', help='only these question numbers, e.g. 1-50')
    fly.add_argument('--agent', required=True,
                     help='look-around, follow-path, or package.module:factory')
    fly.add_argument('--agent-arg', action='append', default=[], metavar='KEY=VALUE',
                     help='keyword argument for the agent factory; repeat for more')
    render_agent.add_camera_args(fly)
    fly.add_argument('--out', type=Path, default=RUNS_DIR,
                     help=f'where run folders go (default: {RUNS_DIR})')
    fly.add_argument('--name', help='the run folder is BENCHMARK-NAME (default: the agent)')
    marks = commands.add_parser('score', help='score run folders and print one report')
    marks.add_argument('folders', nargs='+', type=Path)
    marks.add_argument('--judge-model', default=None,
                       help='the Claude model that marks open answers (default: sonnet)')
    args = parser.parse_args(argv)
    if args.command == 'score':
        return _score(args)

    numbers = None
    if args.questions:
        try:
            numbers = parse_numbers(args.questions)
        except ValueError as exc:
            parser.error(f'--questions: {exc}')
    agent_kwargs = render_agent.parse_agent_args(parser, args.agent_arg)
    from drones.sim import scenes
    from drones.sim.scene_view import SceneView

    intrinsics, mount = render_agent.camera_from(args, parser)
    eye_height = eqa.EYE_HEIGHT if args.eye_height is None else args.eye_height
    try:
        agent = agents.make_agent(args.agent, **agent_kwargs)
    except (ValueError, ImportError, AttributeError) as exc:
        parser.error(f'--agent {args.agent}: {exc}')

    @contextlib.contextmanager
    def open_view(scene):
        scenes.download([scene])
        with SceneView(scene, intrinsics, mount=mount) as view:
            yield view

    names = eqa.BENCHMARKS if 'all' in args.benchmark else tuple(
        dict.fromkeys(args.benchmark))
    tag = args.name or args.agent.replace(':', '.')
    camera = {'width': intrinsics.width, 'height': intrinsics.height, 'fov': args.fov,
              'intrinsics': None if args.intrinsics is None else str(args.intrinsics),
              'mount': not args.no_mount}
    for name in names:
        config = {'benchmark': name, 'agent': args.agent, 'agent_args': agent_kwargs,
                  'chunk_size': getattr(agent, 'chunk_size', 1),
                  'reach': getattr(agent, 'reach', None),
                  'camera': camera, 'eye_height': eye_height, 'questions': args.questions}
        try:
            try:
                questions = select(name, load_questions(name), numbers)
            except Stopped:
                raise
            except Exception as exc:
                raise Stopped(f'stopped at {name}: {exc}. Run the same command again to '
                              'resume there.') from exc
            run(name, questions, agent, args.out / f'{name}-{tag}', open_view, config,
                eye_height, functools.partial(print, flush=True))
        except Stopped as exc:
            sys.exit(str(exc))
    print('All done. Score with:  drones-benchmark score ' +
          ' '.join(str(args.out / f'{n}-{tag}') for n in names), flush=True)


def _score(args):
    from drones.sim import scoring

    model = args.judge_model or scoring.JUDGE_MODEL
    summaries = []
    for folder in args.folders:
        if not (folder / 'results.jsonl').exists():
            print(f'skipping {folder}: no results.jsonl', file=sys.stderr, flush=True)
            continue
        try:
            summaries.append(scoring.score(folder, scoring.claude_judge(folder, model)))
        except Stopped as exc:
            sys.exit(str(exc))
    if not summaries:
        sys.exit('no run folders with results.jsonl among those given')
    print(scoring.report(summaries), flush=True)
