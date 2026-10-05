"""drones-benchmark score on hand-made run folders, with a scripted judge."""
import json

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones.sim import benchmark, scoring


class FakeJudge:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, image):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def make_run(folder, bench, rows, trajectories=None):
    (folder / 'episodes').mkdir(parents=True)
    (folder / 'config.json').write_text(json.dumps({'benchmark': bench}))
    base = {'benchmark': bench, 'scene': 's', 'question': 'Q?', 'choices': [], 'truth': 't',
            'answer': 'a', 'stop': 'done', 'start': 'benchmark', 'decisions': 3, 'budget': 12,
            'steps': 40, 'chunk_size': 16, 'path_length': 1.0, 'final_pos': [0, 0, 1],
            'final_yaw': 0.0, 'goal': None, 'reference_length': None, 'seconds': 9.0,
            'model_calls': 3, 'model_seconds': 6.0,
            'stats': {'first': 2, 'retry': 1, 'failed': 0}}
    with open(folder / 'results.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps({**base, **row}) + '\n')
    for number, (pos, reference) in (trajectories or {}).items():
        (folder / 'episodes' / str(number)).mkdir()
        np.savez(folder / 'episodes' / str(number) / 'trajectory.npz', pos=pos,
                 yaw=np.zeros(len(pos)), reference=reference)
    return folder


def judge(folder, *replies, model='sonnet'):
    backend = FakeJudge(*replies)
    return scoring.Judge(backend, model, folder / 'judged.jsonl'), backend


def test_multiple_choice_success_rate_and_normalized_steps(tmp_path):
    choices = ['A) red', 'B) blue']
    folder = make_run(tmp_path / 'run', 'hm-eqa', [
        {'number': 1, 'choices': choices, 'truth': 'B) blue', 'answer': 'B', 'category': 'c1',
         'decisions': 6, 'budget': 12},
        {'number': 2, 'choices': choices, 'truth': 'A) red', 'answer': 'blue', 'category': 'c2',
         'decisions': 12, 'budget': 12, 'stop': 'budget'},
        {'number': 3, 'choices': choices, 'truth': 'A) red', 'answer': None, 'category': 'c2',
         'decisions': 3, 'budget': 12, 'stop': 'failed'}])
    summary = scoring.score(folder, judge(folder)[0])
    m = summary['metrics']
    assert m['SR'] == pytest.approx(100 / 3)
    assert m['Normalized steps'] == pytest.approx((0.5 + 1 + 0.25) / 3)
    assert m['Out of budget %'] == pytest.approx(100 / 3)
    assert m['Seconds per call'] == pytest.approx(2.0)
    assert m['Valid at once %'] == pytest.approx(100 * 6 / 9)
    assert summary['categories']['c2']['SR'] == 0.0
    rows = json.loads((folder / 'scores.json').read_text())
    assert [r['correct'] for r in rows] == [True, False, False]
    assert 'SR' in (folder / 'report.md').read_text()


def test_a_eqa_is_judged_with_openeqas_prompt_and_cached(tmp_path):
    folder = make_run(tmp_path / 'run', 'a-eqa', [
        {'number': 1, 'question': 'What is on the bed?', 'truth': 'a cat', 'answer': 'a dog',
         'category': 'objects'},
        {'number': 2, 'answer': None, 'category': 'objects'}])
    first, backend = judge(folder, 'Your mark: 3')
    summary = scoring.score(folder, first)
    (prompt,) = backend.prompts                       # the unanswered one is not asked
    assert prompt.startswith('You are an AI assistant who will help me to evaluate')
    assert prompt.endswith('Question: What is on the bed?\nAnswer: a cat\nResponse: a dog')
    assert summary['metrics']['LLM-Match'] == pytest.approx(100 * (0.5 + 0) / 2)
    again, backend = judge(folder)                    # no replies: it must not ask
    assert scoring.score(folder, again)['metrics']['LLM-Match'] == pytest.approx(25.0)
    other, backend = judge(folder, '5', model='opus')
    assert scoring.score(folder, other)['metrics']['LLM-Match'] == pytest.approx(50.0)


def test_express_bench_llm_score_e_path_and_distance_to_target(tmp_path):
    folder = make_run(tmp_path / 'run', 'express-bench', [
        {'number': 1, 'answer': 'x', 'reference_length': 2.0, 'path_length': 4.0,
         'goal': [3.0, 4.0, 1.0], 'final_pos': [0.0, 0.0, 1.0], 'category': 'c'}])
    summary = scoring.score(folder, judge(folder, '5')[0])
    m = summary['metrics']
    assert m['LLM-Score C*'] == pytest.approx(100.0)
    assert m['E_path'] == pytest.approx(50.0)
    assert m['d_T (m)'] == pytest.approx(5.0)


def test_indoor_uav_navigation_metrics_from_the_trajectories(tmp_path):
    reference = np.array([[0.0, 0, 1], [4, 0, 1]])
    folder = make_run(tmp_path / 'run', 'indoor-uav', [
        {'number': 1, 'goal': [4.0, 0, 1], 'category': 'traj_1, test seen, easy',
         'answer': None},
        {'number': 2, 'goal': [4.0, 0, 1], 'category': 'traj_2, test unseen, hard',
         'answer': None}], trajectories={
        1: (np.array([[0.0, 0, 1], [3, 0, 1]]), reference),
        2: (np.array([[0.0, 0, 1], [4, 0, 1], [0, 0, 1]]), reference)})
    summary = scoring.score(folder, judge(folder)[0])
    m = summary['metrics']
    assert m['SR'] == pytest.approx(50.0)
    assert m['OSR'] == pytest.approx(100.0)
    assert m['NE (m)'] == pytest.approx((1 + 4) / 2)
    assert 0 < m['nDTW'] <= 100
    assert set(summary['categories']) == {'test seen, easy', 'test unseen, hard'}


def test_an_unreadable_judge_reply_stops_scoring_and_keeps_the_cache(tmp_path):
    folder = make_run(tmp_path / 'run', 'a-eqa', [{'number': 1}, {'number': 2}])
    first, _ = judge(folder, '4', 'no idea', 'still no idea')
    with pytest.raises(benchmark.Stopped, match='question 2'):
        scoring.score(folder, first)
    cached = [json.loads(line) for line in (folder / 'judged.jsonl').read_text().splitlines()]
    assert [c['number'] for c in cached] == [1]


def test_the_report_lists_every_run_and_the_deviations(tmp_path):
    a = make_run(tmp_path / 'a', 'hm-eqa', [{'number': 1, 'choices': ['A) x'], 'truth': 'A) x',
                                             'answer': 'A', 'category': 'c'}])
    b = make_run(tmp_path / 'b', 'a-eqa', [{'number': 1, 'category': 'c'}])
    summaries = [scoring.score(a, judge(a)[0]), scoring.score(b, judge(b, '2')[0])]
    text = scoring.report(summaries)
    assert 'hm-eqa' in text and 'a-eqa' in text
    for deviation in scoring.DEVIATIONS:
        assert deviation in text


def test_a_missing_trajectory_stops_scoring_with_the_question_named(tmp_path):
    folder = make_run(tmp_path / 'run', 'indoor-uav', [
        {'number': 3, 'goal': [4.0, 0, 1], 'category': 'traj_1, test seen, easy',
         'answer': None}])
    with pytest.raises(benchmark.Stopped, match='question 3: no episodes/3/trajectory.npz'):
        scoring.score(folder, judge(folder)[0])


def test_d_t_of_questions_with_no_target_is_nan_without_a_warning(tmp_path, recwarn):
    folder = make_run(tmp_path / 'run', 'express-bench', [{'number': 1}])
    (folder / 'judged.jsonl').write_text('')
    summary = scoring.score(folder, judge(folder, '4')[0])
    assert summary['metrics']['d_T (m)'] != summary['metrics']['d_T (m)']
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
