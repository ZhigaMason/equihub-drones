import pytest

from drones import config
from drones.control.avoidance import ceiling_detected, repulsion

A = config.AVOID_DISTANCE
H = min(config.AVOID_HARD_DISTANCE, A)
M = config.MAX_AVOID_SPEED


def test_nothing_in_range_means_no_push():
    assert repulsion(None, None, None, None) == (0.0, 0.0)


def test_walls_at_or_beyond_avoid_distance_are_ignored():
    assert repulsion(A, A + 0.5, A + 0.01, 5.0) == (0.0, 0.0)


@pytest.mark.parametrize('distance', [H, H / 2, 0.01])
def test_full_push_at_and_inside_hard_distance(distance):
    vx, _ = repulsion(distance, None, None, None)
    assert vx == pytest.approx(-M)


def test_ramp_is_linear_between_hard_and_avoid_distance():
    vx, _ = repulsion((A + H) / 2, None, None, None)
    assert vx == pytest.approx(-M / 2)


@pytest.mark.parametrize('side, axis, sign', [
    ('front', 0, -1),   # wall ahead -> back off
    ('back', 0, +1),    # wall behind -> move forward
    ('left', 1, -1),    # wall on the left -> move right (-y)
    ('right', 1, +1),   # wall on the right -> move left (+y)
])
def test_push_is_away_from_the_wall(side, axis, sign):
    ranges = dict(front=None, back=None, left=None, right=None)
    ranges[side] = H / 2
    velocity = repulsion(**ranges)
    assert velocity[axis] == pytest.approx(sign * M)
    assert velocity[1 - axis] == 0.0


def test_symmetric_corridor_cancels():
    assert repulsion(None, None, 0.5, 0.5) == pytest.approx((0.0, 0.0))


def test_closer_wall_wins_in_an_asymmetric_corridor():
    _, vy = repulsion(None, None, left=H, right=A - 0.05)
    assert vy < 0


def test_ceiling_detected():
    assert ceiling_detected(config.CEILING_DISTANCE - 0.01)
    assert not ceiling_detected(config.CEILING_DISTANCE + 0.01)
    assert not ceiling_detected(None)
