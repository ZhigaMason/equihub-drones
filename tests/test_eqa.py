"""drones.sim.eqa: benchmark questions parsed and placed in our frame.

Hermetic: each benchmark's files are written where fetch() caches them, so nothing is downloaded.
"""
import io
import json
import math
import zipfile

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones.sim import eqa


def cache(dest, benchmark, files):
    folder = dest / 'benchmarks' / benchmark
    folder.mkdir(parents=True)
    for name, text in files.items():
        (folder / name).write_text(text)


@pytest.mark.parametrize('theta', [0.0, 0.7, -2.0, math.pi])
def test_habitat_heading_becomes_our_yaw(theta):
    # A habitat agent turned theta about +y faces (-sin theta, 0, -cos theta) (-z at rest).
    facing = eqa.habitat_point([-math.sin(theta), 0.0, -math.cos(theta)])
    yaw = eqa.habitat_yaw(theta)
    np.testing.assert_allclose(facing, [math.cos(yaw), math.sin(yaw), 0.0], atol=1e-12)


def test_habitat_up_is_our_up():
    np.testing.assert_allclose(eqa.habitat_point([1.0, 2.0, 3.0]), [1.0, -3.0, 2.0])


def test_hm_eqa_drops_padding_choices_and_takes_the_floor_pose(tmp_path):
    cache(tmp_path, 'hm-eqa', {
        'questions.csv': (
            'scene,floor,question,choices,question_formatted,answer,label\n'
            '00004-VqCaAuuoeWk,1,Is the lamp on?,"[\'(Do not choose this option)\', \'Yes\', '
            '\'No\', \'(Do not choose this option)\']",x,B,existence\n'
            '00005-yPKGKBCyYx8,0,Is it covered?,"[\'Yes\', \'No\', \'Maybe\', \'Never\']",x,D,'
            'count\n'),
        'poses.csv': ('scene_floor,init_x,init_y,init_z,init_angle\n'
                      '00004-VqCaAuuoeWk_1,1.0,2.5,-3.0,0.0\n'),
    })
    first, second = eqa.load('hm-eqa', tmp_path)
    assert first.choices == ('B) Yes', 'C) No')          # letters kept, padding gone
    assert first.answer == 'B) Yes' and first.category == 'existence'
    np.testing.assert_allclose(first.start, [1.0, 3.0, 2.5])
    assert first.yaw == pytest.approx(math.pi / 2)
    assert second.answer == 'D) Never'
    assert second.start is None and second.yaw is None     # floor 0 has no pose: open floor


def test_express_bench_carries_its_path_and_heading(tmp_path):
    turn = math.radians(150)   # EXPRESS-Bench stores [w, x, y, z], turning about habitat +y
    episode = {
        'scene_id': 'hm3d/train/00006-HkseAnWCgqk', 'type': 'state',
        'question': 'Did I leave the faucet running?', 'answer': 'No.',
        'start_position': [0.0, 3.0, 4.0],
        'start_rotation': [math.cos(turn / 2), 0.0, math.sin(turn / 2), 0.0],
        'goal_position': [2.0, 3.0, -1.0],
        'actions': {'step_0': {'position': [0.0, 3.0, 3.75]},
                    'step_1': {'position': [0.0, 3.0, 3.5]}},
    }
    cache(tmp_path, 'express-bench', {'express-bench.json': json.dumps([episode])})
    [q] = eqa.load('express-bench', tmp_path)
    assert q.scene == '00006-HkseAnWCgqk' and q.choices == ()
    np.testing.assert_allclose(q.path, [[0, -4, 3], [0, -3.75, 3], [0, -3.5, 3]])
    np.testing.assert_allclose(q.goal, [2, 1, 3])
    assert q.yaw == pytest.approx(eqa.habitat_yaw(turn))
    # Walking forward in habitat (-z) is +y here; its path heads that way from the start.
    assert q.path[-1][1] > q.path[0][1]


def test_a_eqa_is_the_184_subset_with_scenes_resolved_by_hash(tmp_path):
    questions = [{'question_id': 'keep', 'question': 'What is on the bed?', 'answer': 'A pillow',
                  'category': 'object recognition',
                  'episode_history': 'hm3d-v0/000-hm3d-BFRyYbPCCPE'},
                 {'question_id': 'drop', 'question': 'Not in A-EQA', 'answer': '-',
                  'category': 'x', 'episode_history': 'scannet-v0/002-scannet-scene0709_00'}]
    cache(tmp_path, 'a-eqa', {'open-eqa-v0.json': json.dumps(questions),
                              'a-eqa-184.json': json.dumps(['keep'])})
    (tmp_path / 'hm3d').mkdir()
    (tmp_path / 'hm3d' / 'index.json').write_text(json.dumps({'BFRyYbPCCPE': '00826-BFRyYbPCCPE'}))
    [q] = eqa.load('a-eqa', tmp_path)
    assert (q.scene, q.number, q.answer) == ('00826-BFRyYbPCCPE', 1, 'A pillow')
    assert q.start is None


def test_by_scene_puts_the_busiest_scene_first():
    qs = [eqa.Question('x', i, scene, '', '', '') for i, scene in enumerate('abbcbc', 1)]
    groups = eqa.by_scene(qs)
    assert list(groups) == ['b', 'c', 'a']
    assert [q.number for q in groups['b']] == [2, 3, 5]


def test_unknown_benchmark_is_an_error(tmp_path):
    with pytest.raises(ValueError, match='unknown benchmark'):
        eqa.load('eqa-9000', tmp_path)


class _Ranged(io.BytesIO):
    """An in-memory archive served as scenes._HttpFile and urlopen serve the real one."""

    def __init__(self, data):
        super().__init__(data)
        self.size = len(data)


def indoor_uav_archive():
    """A without_screenshot.zip in miniature: two Gibson trajectories (one a reverse), an HM3D
    one, and an MP3D one that must be ignored. Instructions GBK-encoded, as IndoorUAV's are."""
    def trajectory(zf, folder, instruction, posture):
        zf.writestr(f'{folder}/instruction.json', json.dumps(
            {'instruction': instruction}, ensure_ascii=False).encode('gbk'))
        zf.writestr(f'{folder}/instruction_pro.json',
                    json.dumps({'instruction': f'In detail: {instruction}'}))
        zf.writestr(f'{folder}/posture.json', json.dumps(posture))
        zf.writestr(f'{folder}/real_action.json', '{}')
        zf.writestr(f'{folder}/saved_transformations/1.txt', '1 0 0 0')

    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
        root = 'without_screenshot'
        trajectory(zf, f'{root}/gibson_1/Adrian/traj_-1', 'Fly back to the hallway’s end',
                   [[1.0, 2.0, 1.3, 180.0], [1.0, 3.0, 1.3, 180.0]])
        trajectory(zf, f'{root}/gibson_1/Adrian/traj_1', 'Fly to the sofa',
                   [[1.0, 2.0, 1.3, 0.0], [2.0, 2.0, 1.5, 0.0]])
        trajectory(zf, f'{root}/hm3d_3/HkseAnWCgqk/traj_2', 'Ascend to the stairs',
                   [[0.0, 0.0, 1.2, 90.0]])
        trajectory(zf, f'{root}/mp3d_1/abc/traj_1', 'Not loadable here', [[0, 0, 1, 0]])
    return out.getvalue()


@pytest.fixture
def indoor_uav(tmp_path, monkeypatch):
    from drones.sim import scenes

    data = indoor_uav_archive()
    requests = []

    class Response(io.BytesIO):
        url = 'https://cdn/without_screenshot.zip'

    def urlopen(request):
        requests.append(request.get_method())
        if request.get_method() == 'HEAD':
            return Response(b'')
        start, end = map(int, request.headers['Range'].split('=')[1].split('-'))
        return Response(data[start:end + 1])

    monkeypatch.setattr(scenes, 'archive_url', lambda file=None: 'https://modelscope/x')
    monkeypatch.setattr(scenes, '_HttpFile', lambda url: _Ranged(data))
    monkeypatch.setattr(eqa.urllib.request, 'urlopen', urlopen)
    cache(tmp_path, 'indoor-uav', {
        'train.csv': 'traj_path,difficulty\n/gibson_1/Adrian/traj_1,easy\n',
        'test_seen.csv': 'traj_path,difficulty\n/hm3d_3/HkseAnWCgqk/traj_2,hard\n',
        'test_unseen.csv': 'traj_path,difficulty\n'})
    (tmp_path / 'hm3d').mkdir()
    (tmp_path / 'hm3d' / 'index.json').write_text(
        json.dumps({'HkseAnWCgqk': '00006-HkseAnWCgqk'}))
    return tmp_path, requests


def test_indoor_uav_index_keeps_gibson_and_hm3d_trajectories(indoor_uav):
    dest, _ = indoor_uav
    index = eqa.indoor_uav_index(dest)
    assert list(index) == ['gibson_1/Adrian', 'hm3d_3/HkseAnWCgqk']      # MP3D left out
    assert list(index['gibson_1/Adrian']['trajectories']) == ['traj_1', 'traj_-1']
    assert eqa.busiest('indoor-uav', 5, dest) == ['Adrian', '00006-HkseAnWCgqk']
    assert eqa.locate('indoor-uav', 3, dest) == '00006-HkseAnWCgqk'
    assert eqa.locate('indoor-uav', 4, dest) is None


def test_indoor_uav_prompts_come_per_scene_in_one_request(indoor_uav):
    dest, requests = indoor_uav
    assert eqa.load('indoor-uav', dest) == []       # nothing prepared yet
    requests.clear()
    eqa.prepare('indoor-uav', ['Adrian'], dest)
    assert requests.count('GET') == 1
    to_sofa, back = eqa.load('indoor-uav', dest)
    assert (to_sofa.number, back.number) == (1, 2)   # numbered over the whole index
    assert back.text == 'Fly back to the hallway’s end'   # GBK decoded
    assert to_sofa.answer == 'In detail: Fly to the sofa' and to_sofa.reveal != 'answer'
    assert to_sofa.category == 'traj_1, train, easy' and back.category.endswith('unsplit')
    # posture rows are habitat's (x, z, y) and the negated heading: (x, -y, height) here.
    np.testing.assert_allclose(to_sofa.path, [[1, -2, 1.3], [2, -2, 1.5]])
    np.testing.assert_allclose(to_sofa.goal, [2, -2, 1.5])
    assert to_sofa.yaw == pytest.approx(math.pi / 2)
    assert back.yaw == pytest.approx(-math.pi / 2)   # 90 - 180 degrees, wrapped
    # back's posture moves +y in the file (habitat +z), so -y here: along a yaw of -90 degrees.
    assert back.path[1][1] < back.path[0][1]
