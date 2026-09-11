"""drones-fly-square against a fake Crazyflie: conversions, sequencing, safety and the log."""
import csv
import math

import numpy as np
import pytest

from drones.missions.fly_policy import DEFAULT_SIGNS, to_setpoint
from drones.missions.fly_square import (COLUMNS, LOG_FORMAT, SquareLimits, SquareOptions,
                                        firmware_action, fly_square, read_state,
                                        square_abort_reason, square_recorder)
from drones.policy.runtime import Policy, SquareSpec
from drones.policy.square import OBS_SIZE

SPEC = SquareSpec(control_freq=50, side=1.0, corner_radius=0.15, lap_time=(6.0, 10.0),
                  height=(0.8, 1.2), max_tilt=0.35, max_yaw_rate=1.5, hover_thrust=0.44,
                  thrust_min=0.085, thrust_max=0.8)
HOVERING = {
    'stateEstimate.x': 0.0, 'stateEstimate.y': 0.0, 'stateEstimate.z': 1.0,
    'stateEstimate.vx': 0.0, 'stateEstimate.vy': 0.0, 'stateEstimate.vz': 0.0,
    'stateEstimate.qx': 0.0, 'stateEstimate.qy': 0.0, 'stateEstimate.qz': 0.0,
    'stateEstimate.qw': 1.0, 'gyro.x': 0.0, 'gyro.y': 0.0, 'gyro.z': 0.0,
    'controller.cmd_thrust': 38000.0, 'controller.roll': 0.0, 'controller.pitch': 0.0,
    'controller.yawRate': 0.0, 'range.zrange': 1000, 'range.front': 2000, 'range.back': 2000,
    'range.left': 2000, 'range.right': 2000, 'range.up': 8190,
}


def constant_policy(action=(0.0, 0.0, 0.0, 0.0)):
    return Policy(SPEC, [(np.zeros((OBS_SIZE, 4)), np.asarray(action, float))])


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, args))


class FakeLink:
    def __init__(self):
        self.cf = type('CF', (), {})()
        self.cf.commander = Recorder()
        self.cf.supervisor = Recorder()


class FakeLog:
    def __init__(self, **changes):
        self.latest = {**HOVERING, **changes}

    def age(self):
        return 0.0


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def run(policy=None, log=None, record=None, **options):
    """Fly 0.1 of a 6 s lap: short enough that a drone standing still stays on the square."""
    link, clock = FakeLink(), Clock()
    options = {'lap_time': 6.0, 'laps': 0.1, **options}
    reason = fly_square(link, policy or constant_policy(), log or FakeLog(),
                        SquareOptions(**options), record=record, now=clock.now, sleep=clock.sleep)
    return reason, link.cf.commander.calls, link.cf.supervisor.calls


def names(calls):
    return [name for name, _ in calls]


# ---------------------------------------------------------------------- conversions
def test_read_state_converts_units_and_heading():
    half = math.sqrt(0.5)
    state, tilt = read_state({**HOVERING, 'stateEstimate.qz': half, 'stateEstimate.qw': half,
                              'gyro.z': 90.0, 'stateEstimate.vx': 0.3})
    assert state['yaw'] == pytest.approx(math.pi / 2)
    np.testing.assert_allclose(state['gyro'], [0.0, 0.0, math.pi / 2])
    np.testing.assert_allclose(state['vel'], [0.3, 0.0, 0.0])
    np.testing.assert_allclose(state['gravity'], [0.0, 0.0, -1.0], atol=1e-12)
    assert state['zrange'] == pytest.approx(1.0) and tilt == pytest.approx(0.0)


@pytest.mark.parametrize('action', [[0.2, -0.4, 0.5, 0.3], [-0.6, 0.1, -0.2, -0.5]])
def test_firmware_action_inverts_to_setpoint(action):
    roll, pitch, yaw_rate, thrust = to_setpoint(np.array(action), SPEC, 38000.0, DEFAULT_SIGNS)
    latest = {'controller.roll': roll, 'controller.pitch': pitch, 'controller.yawRate': yaw_rate,
              'controller.cmd_thrust': float(thrust)}
    np.testing.assert_allclose(firmware_action(latest, SPEC, 38000.0, DEFAULT_SIGNS), action,
                               atol=1e-3)


def test_abort_when_off_the_square():
    state, tilt = read_state({**HOVERING, 'stateEstimate.x': 0.6})
    reason = square_abort_reason(state, tilt, 0.0, np.array([0.0, 0.0, 1.0]), SquareLimits())
    assert 'off the square' in reason
    state, tilt = read_state(HOVERING)
    assert square_abort_reason(state, tilt, 0.0, np.array([0.1, 0.0, 1.0]), SquareLimits()) is None


# ---------------------------------------------------------------------- flights
def test_policy_flight_unlocks_takes_off_flies_and_lands():
    reason, commander, supervisor = run()
    assert reason == 'square done'
    assert commander[0] == ('send_setpoint', (0, 0, 0, 0))
    assert names(supervisor) == ['send_arming_request', 'send_arming_request']
    flown = [args for name, args in commander[1:] if name == 'send_setpoint']
    assert 29 <= len(flown) <= 31   # 0.6 s at 50 Hz, give or take the float clock's last tick
    assert names(commander)[-2:] == ['send_stop_setpoint', 'send_notify_setpoint_stop']


def test_firmware_mode_flies_position_setpoints_along_the_square():
    reason, commander, _ = run(firmware=True)
    assert reason == 'square done'
    points = [args for name, args in commander if name == 'send_position_setpoint']
    assert points and points[0][:3] == pytest.approx((0.0, 0.0, 1.0))
    assert points[-1][0] > 0.2 and abs(points[-1][1]) < 1e-9   # first edge: straight ahead
    assert not [args for name, args in commander[1:] if name == 'send_setpoint']


def test_falling_behind_the_square_hands_back_and_lands():
    # The square starts where the drone is; a drone that never moves is left 0.5 m behind in
    # under a second.
    reason, commander, _ = run(laps=0.5)
    assert 'off the square' in reason
    assert 'send_stop_setpoint' in names(commander)


def test_the_log_holds_what_was_applied(tmp_path):
    path = tmp_path / 'flight.csv'
    record, close = square_recorder(path)
    run(policy=constant_policy([0.4, 0.0, 0.0, 0.2]), record=record, authority=0.5)
    close()
    lines = path.read_text().splitlines()
    assert lines[0] == LOG_FORMAT
    rows = list(csv.DictReader(lines[1:]))
    assert tuple(rows[0]) == COLUMNS and 29 <= len(rows) <= 31
    assert float(rows[0]['a_roll']) == pytest.approx(0.2)   # the policy's 0.4 at half authority
    times = [float(r['time']) for r in rows]
    assert times == sorted(times)
