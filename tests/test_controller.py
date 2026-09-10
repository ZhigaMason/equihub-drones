"""The flight state machine against fake cflib objects.

The control law is covered by test_mixer.py; this checks the orchestration
around it: watchdog, ceiling, emergency stop and command priority.
"""
import json
import threading
import time

import pytest

from drones import config
from drones.crazyflie.controller import DroneController


class FakeRanger:
    def __init__(self, **readings):
        self.front = self.back = self.left = self.right = self.up = None
        self.down = 1.0
        for name, value in readings.items():
            setattr(self, name, value)


class FakeMotionCommander:
    def __init__(self):
        self.setpoints = []

    def start_linear_motion(self, vx, vy, vz, yaw=0.0):
        self.setpoints.append((vx, vy, vz, yaw))


class FakeSupervisor:
    def __init__(self):
        self.emergency_stopped = False

    def send_emergency_stop(self):
        self.emergency_stopped = True


class FakeLink:
    def __init__(self):
        self.cf = type('CF', (), {})()
        self.cf.supervisor = FakeSupervisor()


class FakeBattery:
    voltage = 3.9


def fly(controller, ranger=None, *, seconds, client=None):
    """Run the fly loop for up to `seconds`. `client(controller)` is called
    at 20 Hz on a side thread, the way the phone page talks."""
    mc, scf = FakeMotionCommander(), FakeLink()
    stop = threading.Event()
    if client:
        def talk():
            while not stop.is_set():
                client(controller)
                time.sleep(1 / 20)
        threading.Thread(target=talk, daemon=True).start()
    timer = threading.Timer(seconds, controller._shutdown.set)
    timer.start()
    try:
        reason = controller._fly_loop(scf, ranger or FakeRanger(),
                                      FakeBattery(), mc)
    finally:
        stop.set()
        timer.cancel()
    return reason, mc, scf


@pytest.fixture
def airborne():
    controller = DroneController()
    controller._mixer.reset(1.0)
    controller._set(target_altitude=1.0, desired_altitude=1.0)
    controller.heartbeat()
    return controller


def test_sticks_reach_the_drone(airborne):
    _, mc, _ = fly(airborne, seconds=1.5,
                   client=lambda c: c.set_control(1.0, -1.0))
    vx, vy, _, yaw = mc.setpoints[-1]
    assert vx == pytest.approx(config.MAX_MANUAL_SPEED, rel=1e-3)
    assert vy == 0.0
    assert yaw == pytest.approx(-config.MAX_YAW_RATE)


def test_ceiling_ends_the_flight(airborne):
    ranger = FakeRanger(up=config.CEILING_DISTANCE / 2)
    reason, _, _ = fly(airborne, ranger, seconds=3.0,
                       client=lambda c: c.heartbeat())
    assert reason.startswith('ceiling')


def test_a_briefly_quiet_client_stops_motion_but_height_is_held(airborne):
    airborne.set_control(1.0, 1.0, 1.5)
    airborne._set(last_client=time.time() - config.STICK_TIMEOUT - 0.1)
    reason, mc, _ = fly(airborne, seconds=1.0)
    assert reason == 'shutting down'
    vx, _, vz, yaw = mc.setpoints[-1]
    assert (vx, yaw) == (0.0, 0.0)
    assert vz > 0, 'should still be climbing towards the last requested height'
    assert airborne.snapshot()['desired_altitude'] == 1.5


def test_a_silent_client_lands_the_drone(airborne):
    airborne._set(last_client=time.time() - config.LINK_TIMEOUT - 1)
    reason, _, _ = fly(airborne, seconds=2.0)
    assert reason.startswith('no client')


def test_auto_mode_does_not_need_a_client(airborne):
    airborne._set(auto_mode=True,
                  last_client=time.time() - config.LINK_TIMEOUT - 1)
    reason, _, _ = fly(airborne, seconds=0.5)
    assert reason == 'shutting down'


def test_emergency_stop_beats_queued_toggles_within_one_cycle(airborne):
    for i in range(40):
        airborne.submit('avoid', i % 2 == 0)
    airborne.submit('estop')
    for _ in range(10):
        airborne.submit('avoid', True)
    reason, mc, scf = fly(airborne, seconds=2.0)
    assert reason == 'emergency stop'
    assert scf.cf.supervisor.emergency_stopped
    assert airborne._estopped
    assert mc.setpoints == [], 'no setpoint may be sent after an e-stop'


def test_land_command_ends_the_flight(airborne):
    airborne.submit('land')
    reason, _, _ = fly(airborne, seconds=2.0)
    assert reason == 'commanded'


def test_set_control_clamps_every_input():
    controller = DroneController()
    controller.set_control(9, -9, 99)
    assert controller._forward == 1.0
    assert controller._yaw == -1.0
    assert controller._desired_altitude == config.MAX_ALTITUDE


def test_snapshot_is_json_serialisable():
    snapshot = DroneController().snapshot()
    assert json.loads(json.dumps(snapshot))['limits'] == {
        'min': config.MIN_ALTITUDE, 'max': config.MAX_ALTITUDE}
