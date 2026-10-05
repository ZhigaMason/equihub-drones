"""drones-benchmark's run loop on a fake view: what it writes, where it stops, how it resumes.
No scene, model or renderer: `open_view` hands back a stand-in that numbers its frames."""
import json
from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones.sim import agents, benchmark, eqa

CHOICES = ('A) red', 'B) blue')


class Room:
    """A scenes.Scene stand-in: a 4 x 4 m floor, walls 1 m up, its origin at the world's."""
    origin = np.zeros(3)
    vertices = np.array([[x, y, 1.0] for x in (-2.0, 2.0) for y in (-2.0, 2.0)])


class FakeView:
    def __init__(self, scene):
        self.scene = Room()
        self.opened = scene

    def render_matrix(self, pos, rotation):
        return np.full((4, 6, 3), 7, np.uint8)


@contextmanager
def fake_open(scene):
    yield FakeView(scene)


class Scripted:
    """Asks every `chunk_size` steps (recording a call), flies 0.1 m ahead a step, and says
    'B' and stops at `stop_after`, if given. Raises at question `fail_at`."""
    chunk_size, reach = 4, 0.4

    def __init__(self, stop_after=None, fail_at=None):
        self.stop_after, self.fail_at = stop_after, fail_at
        self.questions = []

    def reset(self, question, pose):
        self.questions.append(question.number)
        self.calls, self.answer, self.error, self.concluded = [], None, None, False

    def act(self, observation):
        if observation.question.number == self.fail_at:
            raise RuntimeError('claude failed: usage limit reached')
        if observation.step % self.chunk_size == 0:
            self.calls.append({'step': observation.step, 'prompt': 'fly', 'reply': '{}',
                               'seconds': 0.5, 'image': observation.image,
                               'pos': [float(x) for x in observation.pose.pos]})
        if self.stop_after is not None and observation.step >= self.stop_after:
            self.answer = 'B'
            return None
        return agents.Pose(observation.pose.pos + [0.1, 0.0, 0.0], observation.pose.yaw)

    def conclude(self, observation):
        self.concluded = True
        self.answer = 'A'
        return 'A'


def question(number, bench='hm-eqa', **kwargs):
    fields = dict(benchmark=bench, number=number, scene='room', text='What colour?',
                  answer='B) blue', category='colour', choices=CHOICES,
                  start=np.zeros(3), yaw=0.0)
    fields.update(kwargs)
    return eqa.Question(**fields)


def fly(tmp_path, agent, questions, bench='hm-eqa', config=None):
    benchmark.run(bench, questions, agent, tmp_path, fake_open,
                  config or {'benchmark': bench, 'agent': 'scripted'}, log=lambda *a: None)
    return [json.loads(line) for line in (tmp_path / 'results.jsonl').read_text().splitlines()]


def test_a_finished_question_writes_its_line_trajectory_and_calls(tmp_path):
    (row,) = fly(tmp_path, Scripted(stop_after=5), [question(1)])
    assert row['number'] == 1 and row['answer'] == 'B' and row['truth'] == 'B) blue'
    assert row['stop'] == 'done' and row['start'] == 'benchmark'
    assert row['steps'] == 5 and row['decisions'] == 2 and row['chunk_size'] == 4
    assert row['budget'] == 12                            # int(sqrt(16) * 3)
    assert row['path_length'] == pytest.approx(0.5)
    assert row['final_pos'] == pytest.approx([0.5, 0.0, 1.0])
    assert row['model_calls'] == 2 and row['model_seconds'] == pytest.approx(1.0)
    episode = tmp_path / 'episodes' / '1'
    trajectory = np.load(episode / 'trajectory.npz')
    assert trajectory['pos'].shape == (6, 3) and trajectory['yaw'].shape == (6,)
    calls = [json.loads(line) for line in (episode / 'calls.jsonl').read_text().splitlines()]
    assert [c['k'] for c in calls] == [0, 1] and calls[1]['step'] == 4
    assert calls[0]['image'] == 'calls/000.png' and (episode / 'calls/000.png').is_file()
    assert json.loads((tmp_path / 'config.json').read_text())['agent'] == 'scripted'


def test_the_budget_stops_the_agent_and_asks_it_to_conclude(tmp_path):
    agent = Scripted()
    (row,) = fly(tmp_path, agent, [question(1)])
    assert agent.concluded
    assert row['stop'] == 'budget' and row['decisions'] == 12 and row['steps'] == 48
    assert row['answer'] == 'A'


def test_a_navigation_flight_is_not_asked_for_an_answer(tmp_path):
    path = np.array([[0.0, 0, 1], [0.4, 0, 1]])
    q = question(1, 'indoor-uav', start=path[0], path=path, goal=path[-1],
                 category='traj_1, test seen, easy', choices=())
    agent = Scripted()
    (row,) = fly(tmp_path, agent, [q], bench='indoor-uav')
    assert not agent.concluded
    assert row['budget'] == 2                             # ceil(2 * 0.4 / 0.4)
    assert row['reference_length'] == pytest.approx(0.4)
    assert row['goal'] == pytest.approx([0.4, 0.0, 1.0])  # IndoorUAV is at flight height already
    reference = np.load(tmp_path / 'episodes' / '1' / 'trajectory.npz')['reference']
    assert reference.shape == (2, 3)


def test_a_failure_stops_the_run_keeps_what_finished_and_resumes_there(tmp_path):
    questions = [question(n) for n in (1, 2, 3)]
    with pytest.raises(benchmark.Stopped, match='question 2.*usage limit'):
        fly(tmp_path, Scripted(stop_after=1, fail_at=2), questions)
    assert [r['number'] for r in benchmark.finished(tmp_path / 'results.jsonl').values()] == [1]
    assert sorted(p.name for p in (tmp_path / 'episodes').iterdir()) == ['1']
    again = Scripted(stop_after=1)
    rows = fly(tmp_path, again, questions)
    assert again.questions == [2, 3]
    assert [r['number'] for r in rows] == [1, 2, 3]


def test_a_torn_results_line_is_flown_again(tmp_path):
    fly(tmp_path, Scripted(stop_after=1), [question(1)])
    with open(tmp_path / 'results.jsonl', 'a') as f:
        f.write('{"number": 2, "benchm')                  # killed mid-write
    (tmp_path / 'episodes' / '2.part').mkdir()
    again = Scripted(stop_after=1)
    rows = fly(tmp_path, again, [question(1), question(2)])
    assert again.questions == [2]
    assert [r['number'] for r in rows] == [1, 2]
    assert not (tmp_path / 'episodes' / '2.part').exists()


def test_other_settings_in_the_same_folder_are_refused(tmp_path):
    fly(tmp_path, Scripted(stop_after=1), [question(1)])
    with pytest.raises(benchmark.Stopped, match='agent'):
        fly(tmp_path, Scripted(stop_after=1), [question(2)],
            config={'benchmark': 'hm-eqa', 'agent': 'another'})


def test_a_builtin_agent_runs_without_any_optional_attribute(tmp_path):
    (row,) = fly(tmp_path, agents.LookAround(steps=3), [question(1)])
    assert row['stop'] == 'done' and row['answer'] is None and row['chunk_size'] == 1
    assert row['decisions'] == 4 and row['model_calls'] == 0 and row['stats'] is None
    assert (tmp_path / 'episodes' / '1' / 'calls.jsonl').read_text() == ''


def test_a_question_without_a_start_begins_at_the_open_floor(tmp_path):
    (row,) = fly(tmp_path, Scripted(stop_after=0), [question(1, 'a-eqa', start=None, yaw=None,
                                                            choices=())], bench='a-eqa')
    assert row['start'] == 'origin'
    assert row['final_pos'] == pytest.approx([0.0, 0.0, 1.0])


def test_select_keeps_the_numbers_and_indoor_uavs_test_splits():
    qs = [question(1), question(2), question(3)]
    assert [q.number for q in benchmark.select('hm-eqa', qs, (2, 3))] == [2, 3]
    nav = [question(1, 'indoor-uav', category='traj_1, test seen, easy'),
           question(2, 'indoor-uav', category='traj_2, train, hard'),
           question(3, 'indoor-uav', category='traj_3, test unseen, easy'),
           question(4, 'indoor-uav', category='traj_4, unsplit')]
    assert [q.number for q in benchmark.select('indoor-uav', nav)] == [1, 3]
