"""The deploy script against a fake Crazyflie: conversions, sequencing and safety."""
import dataclasses
import math

import numpy as np
import pytest

from drones.missions import fly_policy
from drones.missions.fly_policy import (DEFAULT_SIGNS, FLOW_RESOLUTION, FlightOptions, Limits,
                                        abort_reason, check_task, fly, read_inputs, to_setpoint)
from drones.policy.interface import BASELINE
from drones.policy.runtime import Policy, PolicySpec

SPEC = PolicySpec(sensors=BASELINE, history=3, control_freq=50, target_height=(0.5, 1.5),
                  range_max=4.0, flow_gain=0.488, max_tilt=0.35, max_yaw_rate=1.5,
                  hover_thrust=0.44, thrust_min=0.085, thrust_max=0.8)
HOVERING = {'motion.deltaX': 0, 'motion.deltaY': 0, 'range.zrange': 1000,
            'range.front': 2000, 'range.back': 2000, 'range.left': 2000, 'range.right': 2000,
            'range.up': 8190, 'gyro.x': 0.0, 'gyro.y': 0.0, 'gyro.z': 0.0,
            'controller.cmd_thrust': 38000.0, 'stateEstimate.qx': 0.0, 'stateEstimate.qy': 0.0,
            'stateEstimate.qz': 0.0, 'stateEstimate.qw': 1.0}


def hover_policy():
    """A policy that always outputs zero action, i.e. hover."""
    return Policy(SPEC, [(np.zeros((SPEC.observation_size, 8)), np.zeros(8)),
                         (np.zeros((8, 4)), np.zeros(4))])


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


def run(policy=None, log=None, **options):
    link, clock = FakeLink(), Clock()
    reason = fly(link, policy or hover_policy(), log or FakeLog(),
                 FlightOptions(duration=1.0, **options), now=clock.now, sleep=clock.sleep)
    return reason, link.cf.commander.calls, link.cf.supervisor.calls


# ---------------------------------------------------------------------- conversions
def test_flow_counts_become_the_simulators_apparent_motion():
    # Moving forward at v over height h, the simulator reports gain * v / h pixels along +x. The
    # firmware reads -deltaY at a tenth of a pixel as its x, so the raw counts are the inverse.
    v, h = 0.5, 1.0
    raw_delta_y = -(SPEC.flow_gain * v / h) / FLOW_RESOLUTION
    inputs, _ = read_inputs({**HOVERING, 'motion.deltaY': raw_delta_y}, SPEC)
    np.testing.assert_allclose(inputs['flow_rate'], [v / h, 0.0], atol=1e-9)
    inputs, _ = read_inputs({**HOVERING, 'motion.deltaX': -10}, SPEC)
    np.testing.assert_allclose(inputs['flow_rate'], [0.0, FLOW_RESOLUTION * 10 / SPEC.flow_gain])


def test_check_task_refuses_a_non_hover_policy():
    assert check_task(SPEC, 'runs/x/policy') is None
    square_spec = dataclasses.replace(SPEC, task='square')
    reason = check_task(square_spec, 'runs/x/policy')
    assert reason is not None and 'square' in reason and 'drones-fly-square' in reason


def test_ranges_are_metres_clipped_to_the_sensor_range():
    inputs, _ = read_inputs(HOVERING, SPEC)
    assert inputs['zrange'] == 1.0
    np.testing.assert_allclose(inputs['ranges'], [2, 2, 2, 2, 4])  # 8190 mm is out of range


def test_gyro_is_converted_to_radians_and_attitude_to_gravity():
    q = [math.sin(0.15), 0.0, 0.0, math.cos(0.15)]  # 0.3 rad of roll
    inputs, tilt = read_inputs({**HOVERING, 'gyro.x': 90.0, 'stateEstimate.qx': q[0],
                                'stateEstimate.qw': q[3]}, SPEC)
    np.testing.assert_allclose(inputs['gyro'], [math.pi / 2, 0, 0])
    np.testing.assert_allclose(inputs['gravity'], [0, -math.sin(0.3), -math.cos(0.3)], atol=1e-9)
    assert tilt == pytest.approx(math.degrees(0.3))


def test_setpoint_signs_follow_the_legacy_commander():
    deg = math.degrees(SPEC.max_tilt)
    assert to_setpoint([1, 0, 0, 0], SPEC, 38000)[0] == pytest.approx(deg)       # roll as is
    assert to_setpoint([0, 1, 0, 0], SPEC, 38000)[1] == pytest.approx(-deg)      # pitch inverted
    assert to_setpoint([0, 0, 1, 0], SPEC, 38000)[2] == pytest.approx(
        -math.degrees(SPEC.max_yaw_rate))                                         # yaw rate inverted
    assert DEFAULT_SIGNS == {'roll': 1.0, 'pitch': -1.0, 'yaw_rate': -1.0}


def test_thrust_maps_through_the_measured_hover_command():
    assert to_setpoint([0, 0, 0, 0], SPEC, 38000)[3] == 38000
    boosted = 38000 * (SPEC.hover_thrust + 0.5 * (SPEC.thrust_max - SPEC.hover_thrust)) / 0.44
    assert to_setpoint([0, 0, 0, 0.5], SPEC, 38000)[3] == round(min(boosted, 60000))
    assert to_setpoint([0, 0, 0, 1], SPEC, 38000)[3] == fly_policy.THRUST_COMMAND_MAX


def test_zero_authority_is_plain_hover():
    roll, pitch, yaw, thrust = to_setpoint([1, 1, 1, 1], SPEC, 38000, authority=0.0)
    assert (abs(roll), abs(pitch), abs(yaw), thrust) == (0, 0, 0, 38000)


@pytest.mark.parametrize('changes, tilt, age, expected', [
    ({}, 0.0, 0.0, None),
    ({}, 35.0, 0.0, 'tilt'),
    ({}, 0.0, 1.0, 'sensor data'),
    ({'range.left': 150}, 0.0, 0.0, 'obstacle'),
    ({'range.zrange': 100}, 0.0, 0.0, 'height'),
    ({'range.zrange': 2500}, 0.0, 0.0, 'height'),
])
def test_abort_reasons(changes, tilt, age, expected):
    inputs, _ = read_inputs({**HOVERING, **changes}, SPEC)
    reason = abort_reason(inputs, tilt, age, target=1.0, limits=Limits())
    assert (reason is None) if expected is None else expected in reason


# ---------------------------------------------------------------------- sequencing
def test_dry_run_never_arms_or_commands_the_motors():
    reason, commands, supervisor = run(dry_run=True)
    assert reason == 'dry run'
    assert commands == [] and supervisor == []


def test_flight_sequence_unlock_takeoff_policy_land():
    reason, commands, supervisor = run()
    assert reason == 'time up'
    assert commands[0] == ('send_setpoint', (0, 0, 0, 0)), 'thrust lock released on the ground'
    assert supervisor == [('send_arming_request', (True,)), ('send_arming_request', (False,))]
    names = [name for name, _ in commands[1:]]
    first_policy = names.index('send_setpoint')
    last_policy = len(names) - 1 - names[::-1].index('send_setpoint')
    assert set(names[:first_policy]) == {'send_hover_setpoint'}, 'firmware takes off'
    assert set(names[first_policy:last_policy + 1]) == {'send_setpoint'}, 'policy in the air'
    assert names[-2:] == ['send_stop_setpoint', 'send_notify_setpoint_stop']
    assert set(names[last_policy + 1:-2]) == {'send_hover_setpoint'}, 'firmware lands'
    # A zero-action policy commands exactly the measured hover thrust.
    assert commands[1 + first_policy] == ('send_setpoint', (0.0, -0.0, -0.0, 38000))


def test_a_tripped_limit_hands_back_and_lands():
    reason, commands, _ = run(log=FakeLog(**{'range.front': 100}))
    assert 'obstacle' in reason
    names = [name for name, _ in commands]
    assert names.count('send_setpoint') == 1, 'only the ground unlock; the policy never flew'
    assert names[-1] == 'send_notify_setpoint_stop'


def test_an_implausible_hover_thrust_lands_without_handing_over():
    reason, commands, _ = run(log=FakeLog(**{'controller.cmd_thrust': 0.0}))
    assert 'implausible' in reason
    assert [n for n, _ in commands].count('send_setpoint') == 1


def test_an_exception_mid_flight_still_lands():
    policy = hover_policy()
    policy.act = lambda obs: (_ for _ in ()).throw(RuntimeError('boom'))
    with pytest.raises(RuntimeError, match='boom'):
        run(policy=policy)
