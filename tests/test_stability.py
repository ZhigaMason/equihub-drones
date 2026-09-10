"""Closed-loop checks of the control law against a minimal drone model.

The model is a first-order velocity lag standing in for how quickly a
Crazyflie reaches a commanded velocity. It is a regression guard for the
avoidance tuning, and the seed of a proper simulator backend.
"""
import pytest

from drones import config
from drones.control.avoidance import repulsion
from drones.control.mixer import UPDATE_PERIOD, Command, Mixer, Ranges


def fly_corridor(width, start, *, seconds=30.0, v0=0.0, lag=0.35,
                 auto=True, avoid=True):
    """Lateral position between two walls, measured from the left wall."""
    mixer = Mixer(1.0)
    x, v = start, v0
    trace = []
    for _ in range(int(seconds / UPDATE_PERIOD)):
        sp = mixer.step(Command(), Ranges(left=x, right=width - x),
                        auto=auto, avoid=avoid)
        v += (sp.vy - v) * (UPDATE_PERIOD / lag)
        # +vy is a move to the LEFT, which shrinks the distance to the left wall.
        x -= v * UPDATE_PERIOD
        trace.append(x)
    return trace


@pytest.mark.parametrize('lag', [0.35, 0.6])
@pytest.mark.parametrize('width, start', [
    (2.0, 0.3), (1.2, 0.35), (0.8, 0.25), (0.6, 0.15)])
def test_corridor_settles_without_oscillating_or_touching(width, start, lag):
    trace = fly_corridor(width, start, lag=lag)
    settled = trace[-60:]
    assert max(settled) - min(settled) < 0.01, 'limit cycle'
    assert min(min(trace), width - max(trace)) > 0.05, 'touched a wall'
    final = trace[-1]
    assert abs(repulsion(None, None, final, width - final)[1]) < 0.02, \
        'came to rest while still being pushed'


@pytest.mark.parametrize('lag', [0.35, 0.6])
def test_avoidance_stops_a_drift_that_would_otherwise_get_close(lag):
    # Drifting at 1.5 m/s towards the left wall from 0.9 m out.
    coasting = fly_corridor(6.0, 0.9, v0=1.5, lag=lag, auto=False, avoid=False)
    avoiding = fly_corridor(6.0, 0.9, v0=1.5, lag=lag)
    assert min(coasting) < config.AVOID_DISTANCE, \
        'scenario no longer reaches the avoidance zone; retune this test'
    assert min(avoiding) > 0.05, 'touched the wall'
    assert min(avoiding) > min(coasting) + 0.05
