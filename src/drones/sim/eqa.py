"""Embodied question answering and navigation benchmarks: their prompts, placed in our scenes.

    uv run --extra sim drones-download-scenes --benchmark hm-eqa --count 5
    uv run --extra sim drones-explore-scene --benchmark hm-eqa

| Benchmark     | Questions             | Scenes | Start pose                        |
| ------------- | --------------------- | ------ | --------------------------------- |
| hm-eqa        | 500, multiple choice  |    266 | per scene and floor               |
| mt-hm3d       | 1587, multiple choice |    828 | per scene and floor, for most     |
| express-bench | 2044, open answer     |    174 | per question, with its whole path |
| a-eqa         | 184, open answer      |     57 | none: the scene's open floor      |
| indoor-uav    | 7200 instructions     |    982 | per trajectory, with its path     |

A-EQA is OpenEQA's active subset: the 184 question ids in open-eqa-v0-184-questions.json, all on
HM3D, looked up in open-eqa-v0.json. Every scene is in IndoorUAV's archive (drones.sim.scenes).

indoor-uav is IndoorUAV's own vision-language navigation set: a drone's flight through a scene with
an instruction for it (the "question"; Space reveals the detailed one). Its 8020 trajectories span
Gibson, HM3D, MP3D and Replica; the 7200 on Gibson and HM3D, whose scenes load here, are used. They
live in without_screenshot.zip (500 MB, 1.4 M members), so `indoor_uav_index` reads its central
directory once (~35 s) and keeps where each trajectory's files are, and `prepare` fetches a scene's
prompts in one range request -- a scene's files are contiguous, 0.25 MB typically, 1.3 MB at most.
Its JSON is GBK-encoded (curly quotes as 0xA1 0xAF), not UTF-8.

Sources are pinned to a commit and cached under scenes/benchmarks/.

Poses are habitat-sim's: y up, and an agent at rest facing -z. HM3D meshes are z-up (habitat loads
them with up = +z, front = +y), so a habitat point (x, y, z) is (x, -z, y) in the mesh and in our
world frame, and a heading of theta about habitat's +y is a yaw of theta + pi/2 here. Checked
against the data: of 72 EXPRESS-Bench path points in scene 00006, none passes through geometry
0.3-1.2 m above it with this mapping, and 40 do with y mirrored.

IndoorUAV's posture.json rows are [x, y, height, yaw in degrees]: habitat's (x, z, y) and the
negated heading, so (x, -y, height) and a yaw of 90 degrees - yaw here. Checked the same way: 2 of
429 path points in Nemacolin touch geometry this way, 67 with y mirrored; and the drone's motion
between frames runs along the yaw.
"""
import ast
import csv
import io
import json
import math
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drones.sim.scenes import SCENES_DIR

_GITHUB = 'https://raw.githubusercontent.com'
SOURCES = {
    'hm-eqa': {
        'questions.csv': f'{_GITHUB}/Stanford-ILIAD/explore-eqa/'
                         '18381da3370c5f7729594f10ebc49b5644bbb88c/data/questions.csv',
        'poses.csv': f'{_GITHUB}/Stanford-ILIAD/explore-eqa/'
                     '18381da3370c5f7729594f10ebc49b5644bbb88c/data/scene_init_poses.csv',
    },
    'mt-hm3d': {
        'questions.csv': 'https://huggingface.co/datasets/zmling/MT-HM3D/resolve/'
                         '177cb2528bc0872e82494c3092c0982d171c1a8e/MT-HM3D/MT-HM3D-contextual.csv',
        'poses.csv': 'https://huggingface.co/datasets/zmling/MT-HM3D/resolve/'
                     '177cb2528bc0872e82494c3092c0982d171c1a8e/scene_init_poses.csv',
        'poses_all.csv': 'https://huggingface.co/datasets/zmling/MT-HM3D/resolve/'
                         '177cb2528bc0872e82494c3092c0982d171c1a8e/scene_init_poses_all.csv',
    },
    'express-bench': {
        'express-bench.json': f'{_GITHUB}/HCPLab-SYSU/EXPRESS-Bench/'
                              'e8789dadfbd51af6850dc4d2b0526ca0ee78386a/data/express-bench.json',
    },
    'a-eqa': {
        'open-eqa-v0.json': f'{_GITHUB}/facebookresearch/open-eqa/'
                            'cfa3fce4595c1622bb2f8a38ae2ca9aae9eb685b/data/open-eqa-v0.json',
        'a-eqa-184.json': f'{_GITHUB}/facebookresearch/open-eqa/'
                          'cfa3fce4595c1622bb2f8a38ae2ca9aae9eb685b/assets/'
                          'open-eqa-v0-184-questions.json',
    },
}
BENCHMARKS = (*SOURCES, 'indoor-uav')
INDOOR_UAV_ARCHIVE = 'without_screenshot.zip'
INDOOR_UAV_SPLITS = ('train.csv', 'test_seen.csv', 'test_unseen.csv')
INDOOR_UAV_FILES = ('instruction.json', 'instruction_pro.json', 'posture.json')
# HM-EQA pads its four choices with this where a question has fewer; it is not a real option.
NOT_AN_OPTION = '(Do not choose this option)'


@dataclass
class Question:
    benchmark: str
    number: int              # 1-based, in the benchmark's own order
    scene: str               # HM3D id, 00006-HkseAnWCgqk (A-EQA: the hash, until resolved)
    text: str
    answer: str
    category: str
    choices: tuple = ()      # 'A) ...' strings, multiple-choice benchmarks only
    start: np.ndarray | None = None   # (3,) in the scene file's frame, on the floor
    yaw: float | None = None          # rad, our convention: 0 faces +x, + turns left
    path: np.ndarray | None = None    # (N, 3) a reference walk or flight, file frame
    goal: np.ndarray | None = None    # (3,) where it ends
    reveal: str = 'answer'            # what Space shows: the answer, or the detailed instruction


def habitat_point(p):
    """A habitat-sim position (y up) in the frame of a z-up HM3D mesh."""
    return np.array([p[0], -p[2], p[1]], float)


def _wrap(angle):
    """`angle` in [-pi, pi): a yaw setpoint 270 degrees off would turn the drone the long way."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def habitat_yaw(theta):
    """Our yaw for a habitat heading of `theta` about +y (at theta = 0 the agent faces -z)."""
    return _wrap(theta + math.pi / 2)


def fetch(benchmark, dest=SCENES_DIR):
    """The benchmark's source files, downloaded once into dest/benchmarks/<benchmark>/."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f'unknown benchmark {benchmark!r}; one of {", ".join(BENCHMARKS)}')
    folder = Path(dest) / 'benchmarks' / benchmark
    folder.mkdir(parents=True, exist_ok=True)
    sources = SOURCES.get(benchmark)
    if sources is None:   # indoor-uav: its split files, from IndoorUAV's own repository
        from drones.sim.scenes import archive_url

        sources = {name: None for name in INDOOR_UAV_SPLITS}
        missing = [n for n in sources if not (folder / n).exists()]
        sources = {n: archive_url(n) for n in missing}
    for name, url in sources.items():
        path = folder / name
        if not path.exists():
            print(f'Fetching {benchmark} {name} ...', flush=True)
            partial = path.with_suffix(path.suffix + '.part')
            partial.write_bytes(urllib.request.urlopen(url).read())
            partial.rename(path)
    return folder


def _poses(*paths):
    """{scene_floor: (start, yaw)} from explore-eqa style scene_init_poses.csv files."""
    poses = {}
    for path in paths:
        for row in csv.DictReader(open(path, newline='')):
            poses.setdefault(row['scene_floor'], (
                habitat_point([float(row['init_x']), float(row['init_y']), float(row['init_z'])]),
                habitat_yaw(float(row['init_angle']))))
    return poses


def _multiple_choice(benchmark, folder, pose_files):
    poses = _poses(*[folder / p for p in pose_files])
    questions = []
    for number, row in enumerate(csv.DictReader(open(folder / 'questions.csv', newline='')), 1):
        choices = ast.literal_eval(row['choices'])
        letters = 'ABCDEFGH'
        labelled = tuple(f'{letters[i]}) {c}' for i, c in enumerate(choices) if c != NOT_AN_OPTION)
        answer = row['answer'].strip()
        if answer in letters[:len(choices)]:
            answer = f'{answer}) {choices[letters.index(answer)]}'
        start, yaw = poses.get(f'{row["scene"]}_{row["floor"]}', (None, None))
        questions.append(Question(benchmark, number, row['scene'], row['question'], answer,
                                  row['label'], labelled, start, yaw))
    return questions


def _express(folder):
    questions = []
    for number, e in enumerate(json.loads((folder / 'express-bench.json').read_text()), 1):
        steps = [s['position'] for s in e.get('actions', {}).values()]
        path = np.array([habitat_point(p) for p in [e['start_position'], *steps]])
        w, _, y, _ = e['start_rotation']   # [w, x, y, z]: a turn about habitat's +y only
        questions.append(Question(
            'express-bench', number, e['scene_id'].split('/')[-1], e['question'], e['answer'],
            e['type'], (), habitat_point(e['start_position']), habitat_yaw(2 * math.atan2(y, w)),
            path, habitat_point(e['goal_position'])))
    return questions


def _a_eqa(folder):
    subset = set(json.loads((folder / 'a-eqa-184.json').read_text()))
    questions = []
    for q in json.loads((folder / 'open-eqa-v0.json').read_text()):
        if q['question_id'] in subset:
            scene_hash = q['episode_history'].rsplit('-', 1)[1]   # hm3d-v0/000-hm3d-<hash>
            questions.append(Question('a-eqa', len(questions) + 1, scene_hash, q['question'],
                                      q['answer'], q['category']))
    return questions


# ------------------------------------------------------------------ IndoorUAV
def _trajectory_order(name):
    """traj_3 before traj_-3 before traj_4: each trajectory beside its reverse."""
    n = int(name.split('_')[1])
    return abs(n), n < 0


def indoor_uav_index(dest=SCENES_DIR):
    """{group/scene: {'span': [start, end], 'trajectories': {traj: {file: [offset, size,
    method]}}}} for IndoorUAV's Gibson and HM3D trajectories, built once from the archive's
    central directory and cached as index.json."""
    path = fetch('indoor-uav', dest) / 'index.json'
    if path.exists():
        return json.loads(path.read_text())
    import zipfile

    from drones.sim.scenes import _HttpFile, archive_url

    print(f'Indexing IndoorUAV {INDOOR_UAV_ARCHIVE} (1.4 M members, once, ~35 s) ...', flush=True)
    raw = _HttpFile(archive_url(INDOOR_UAV_ARCHIVE))
    index = {}
    with zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20)) as archive:
        for info in archive.infolist():
            parts = info.filename.split('/')   # without_screenshot/<group>/<scene>/<traj>/...
            if len(parts) < 4 or not parts[2] or not parts[1].startswith(('gibson', 'hm3d')):
                continue
            scene = index.setdefault(f'{parts[1]}/{parts[2]}',
                                     {'span': [info.header_offset, 0], 'trajectories': {}})
            end = info.header_offset + 30 + len(info.orig_filename.encode()) + len(info.extra)
            end += info.compress_size + 1024   # room for a local extra field unlike the central
            scene['span'] = [min(scene['span'][0], info.header_offset), max(scene['span'][1], end)]
            if len(parts) == 5 and parts[4] in INDOOR_UAV_FILES:
                scene['trajectories'].setdefault(parts[3], {})[parts[4]] = [
                    info.header_offset, info.compress_size, info.compress_type]
    for scene in index.values():
        scene['span'][1] = min(scene['span'][1], raw.size)
        complete = {t: f for t, f in scene['trajectories'].items()
                    if all(name in f for name in INDOOR_UAV_FILES)}
        scene['trajectories'] = {t: complete[t] for t in sorted(complete, key=_trajectory_order)}
    index = dict(sorted(index.items()))
    path.write_text(json.dumps(index))
    return index


def _scene_id(key, hm3d_ids):
    """Our name for an IndoorUAV scene: the Gibson name, or the HM3D id."""
    group, name = key.split('/')
    return name if group.startswith('gibson') else hm3d_ids.get(name, name)


def _decode_text(data):
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('gbk')


def _prompts_path(dest, key):
    return fetch('indoor-uav', dest) / 'scenes' / (key.replace('/', '__') + '.json')


def _fetch_prompts(dest, key, entry):
    """A scene's instructions and postures, from one range request over its span of the zip."""
    from drones.sim.scenes import archive_url

    start, end = entry['span']
    url = urllib.request.urlopen(urllib.request.Request(
        archive_url(INDOOR_UAV_ARCHIVE), method='HEAD')).url
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end - 1}'})
    blob = urllib.request.urlopen(request).read()
    prompts = {}
    for trajectory, files in entry['trajectories'].items():
        prompts[trajectory] = {}
        for name, (offset, size, method) in files.items():
            local = offset - start
            if blob[local:local + 4] != b'PK\x03\x04':
                raise RuntimeError(f'{key}/{trajectory}/{name}: no local header at {offset}')
            skip = 30 + sum(struct.unpack('<HH', blob[local + 26:local + 30]))
            data = blob[local + skip:local + skip + size]
            data = zlib.decompress(data, -15) if method == 8 else data
            prompts[trajectory][name] = json.loads(_decode_text(data))
    path = _prompts_path(dest, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompts))


def _indoor_uav(dest):
    """Questions for every IndoorUAV trajectory whose scene's prompts are cached (`prepare`)."""
    from drones.sim.scenes import hm3d_index

    folder = fetch('indoor-uav', dest)
    difficulty = {}
    for split in INDOOR_UAV_SPLITS:
        for row in csv.DictReader(open(folder / split, newline='')):
            difficulty[row['traj_path'].strip('/')] = f'{split[:-4].replace("_", " ")}, ' \
                                                      f'{row["difficulty"]}'
    ids = hm3d_index(dest)
    questions, number = [], 0
    for key, entry in indoor_uav_index(dest).items():
        cached = _prompts_path(dest, key)
        prompts = json.loads(cached.read_text()) if cached.exists() else None
        for trajectory in entry['trajectories']:
            number += 1
            if prompts is None:
                continue
            files = prompts[trajectory]
            posture = np.asarray(files['posture.json'], float)
            path = np.column_stack([posture[:, 0], -posture[:, 1], posture[:, 2]])
            questions.append(Question(
                'indoor-uav', number, _scene_id(key, ids), files['instruction.json']['instruction'],
                files['instruction_pro.json']['instruction'],
                f'{trajectory}, {difficulty.get(f"{key}/{trajectory}", "unsplit")}',
                start=path[0], yaw=_wrap(math.radians(90.0 - posture[0, 3])), path=path,
                goal=path[-1],
                reveal='detailed instruction'))
    return questions


# ------------------------------------------------------------------ all benchmarks
def busiest(benchmark, count, dest=SCENES_DIR):
    """The `count` scenes `benchmark` asks the most about."""
    if benchmark != 'indoor-uav':
        return list(by_scene(load(benchmark, dest)))[:count]
    from drones.sim.scenes import hm3d_index

    index, ids = indoor_uav_index(dest), hm3d_index(dest)
    ranked = sorted(index, key=lambda key: (-len(index[key]['trajectories']), key))
    return [_scene_id(key, ids) for key in ranked[:count]]


def prepare(benchmark, scenes, dest=SCENES_DIR):
    """Make sure `load` has the prompts for `scenes`. Only indoor-uav fetches per scene."""
    if benchmark != 'indoor-uav':
        return
    from drones.sim.scenes import hm3d_index

    ids, wanted = hm3d_index(dest), set(scenes)
    for key, entry in indoor_uav_index(dest).items():
        if _scene_id(key, ids) in wanted and not _prompts_path(dest, key).exists():
            print(f'Fetching IndoorUAV instructions for {key} ...', flush=True)
            _fetch_prompts(dest, key, entry)


def locate(benchmark, number, dest=SCENES_DIR):
    """The scene of question `number` of `benchmark`, or None."""
    if benchmark != 'indoor-uav':
        return next((q.scene for q in load(benchmark, dest) if q.number == number), None)
    from drones.sim.scenes import hm3d_index

    seen = 0
    for key, entry in indoor_uav_index(dest).items():
        seen += len(entry['trajectories'])
        if number <= seen:
            return _scene_id(key, hm3d_index(dest))
    return None


def load(benchmark, dest=SCENES_DIR):
    """Every question of `benchmark`, with scenes named by HM3D id (or Gibson name). For
    indoor-uav, only those of scenes whose prompts `prepare` has fetched."""
    if benchmark == 'indoor-uav':
        return _indoor_uav(dest)
    folder = fetch(benchmark, dest)
    if benchmark == 'hm-eqa':
        return _multiple_choice(benchmark, folder, ['poses.csv'])
    if benchmark == 'mt-hm3d':
        return _multiple_choice(benchmark, folder, ['poses.csv', 'poses_all.csv'])
    if benchmark == 'express-bench':
        return _express(folder)
    from drones.sim.scenes import hm3d_index

    questions = _a_eqa(folder)
    ids = hm3d_index(dest)
    for q in questions:
        q.scene = ids.get(q.scene, q.scene)
    return questions


def by_scene(questions):
    """{scene id: [questions]}, scenes ordered by how many questions they carry, most first."""
    groups = {}
    for q in questions:
        groups.setdefault(q.scene, []).append(q)
    return dict(sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])))
