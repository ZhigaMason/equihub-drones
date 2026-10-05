"""The benchmark metrics on hand-worked numbers: what FAST-EQA, EXPRESS-Bench, OpenEQA and
IndoorUAV define, as drones.sim.metrics computes them."""
import math

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones.sim import metrics

CHOICES = ('A) red', 'B) blue', 'C) green')


def test_floor_area_is_the_box_of_the_vertices_in_the_band_above_the_floor():
    v = np.array([[0.0, 0, 1.0], [4, 0, 1.0], [0, 3, 1.5], [9, 9, 0.05], [9, 9, 2.5]])
    # 0.05 m and 2.5 m above the floor are outside 0.1 to 2.0, so only the first three count.
    assert metrics.floor_area(v, floor_z=0.0) == pytest.approx(12.0)
    assert metrics.floor_area(v + [0, 0, 5], floor_z=5.0) == pytest.approx(12.0)


def test_the_eqa_budget_is_explore_eqas():
    assert metrics.eqa_budget(16.0) == 12            # int(4 * 3)
    assert metrics.eqa_budget(50.0) == int(math.sqrt(50) * 3)


def test_the_path_budget_is_twice_the_reference_in_reaches():
    assert metrics.path_budget(2.0, reach=0.4) == 10
    assert metrics.path_budget(2.1, reach=0.4) == 11


def test_budgets_are_never_below_one():
    assert metrics.floor_area(np.zeros((0, 3)), 0.0) == 0.0
    assert metrics.eqa_budget(0.0) == 1
    assert metrics.path_budget(0.0) == 1


def test_path_length_sums_the_straight_segments():
    assert metrics.path_length([[0, 0, 0], [3, 4, 0], [3, 4, 2]]) == pytest.approx(7.0)
    assert metrics.path_length([[1, 1, 1]]) == 0.0
    assert metrics.path_length(np.zeros((0, 3))) == 0.0


@pytest.mark.parametrize('answer, letter', [
    ('B', 'B'), ('b', 'B'), ('B)', 'B'), ('(B)', 'B'), ('B.', 'B'), ('B) blue', 'B'),
    ('blue', 'B'), ('Blue.', 'B'), ('  C  ', 'C'),
    ('Blue sofa', None), ('D', None), ('', None), (None, None), ('I think B', None),
])
def test_choice_letter_reads_the_shapes_a_model_answers_in(answer, letter):
    assert metrics.choice_letter(answer, CHOICES) == letter


def test_the_truth_reads_as_its_own_letter():
    assert metrics.choice_letter('B) blue', CHOICES) == 'B'


@pytest.mark.parametrize('text, mark', [('5', 5), (' 3\n', 3), ('Your mark: 4', 4),
                                        ('Thinking.\nYour mark: 2\nBecause.', 2)])
def test_parse_mark_reads_openeqa_replies(text, mark):
    assert metrics.parse_mark(text) == mark


@pytest.mark.parametrize('text', ['six', 'Your mark: 7', '0', ''])
def test_parse_mark_refuses_anything_else(text):
    with pytest.raises(ValueError):
        metrics.parse_mark(text)


def test_the_llm_scores():
    marks = [5, 3, 1]
    assert metrics.llm_score(marks) == pytest.approx(100 * (1 + 0.6 + 0.2) / 3)     # C*
    assert metrics.llm_match(marks) == pytest.approx(100 * (1 + 0.5 + 0) / 3)       # OpenEQA
    assert math.isnan(metrics.llm_score([]))


def test_e_path_weights_each_mark_by_path_efficiency():
    # 5 over a path twice the reference: 1 * 0.5; 3 over a shorter path: 0.6 * 1.
    value = metrics.e_path([5, 3], reference_lengths=[2.0, 4.0], flown_lengths=[4.0, 1.0])
    assert value == pytest.approx(100 * (0.5 + 0.6) / 2)
    assert metrics.e_path([5], [0.0], [0.0]) == pytest.approx(100.0)   # stayed, needed to


def test_dtw_on_a_hand_worked_pair():
    a = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    b = np.array([[0.0, 0, 0], [2, 0, 0]])
    # (a0,b0)=0, (a1,b0)=1 or (a1,b1)=1, (a2,b1)=0: the cheapest warping costs 1.
    assert metrics.dtw(a, b) == pytest.approx(1.0)
    assert metrics.dtw(a, a) == 0.0


def test_ndtw_is_one_on_the_reference_and_falls_with_distance():
    ref = np.array([[0.0, 0, 1], [1, 0, 1], [2, 0, 1]])
    assert metrics.ndtw(ref, ref) == pytest.approx(1.0)
    off = ref + [0, 3, 0]
    # Three matched points 3 m off: DTW 9, over 3 points x d_th 10.
    assert metrics.ndtw(ref, off) == pytest.approx(math.exp(-9 / 30))


def test_navigation_success_error_and_oracle():
    path = np.array([[0.0, 0, 1], [5, 0, 1], [10, 0, 1]])
    goal = np.array([5.5, 0, 1])
    assert metrics.navigation(path, goal) == {'success': False, 'ne': pytest.approx(4.5),
                                              'oracle': True}
    assert metrics.navigation(path, np.array([9.0, 0, 1]))['success'] is True
