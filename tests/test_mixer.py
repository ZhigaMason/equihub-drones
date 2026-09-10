"""The control law in isolation: no threads, no clock, no hardware."""
import pytest

from drones import config
from drones.control.avoidance import repulsion
from drones.control.mixer import Command, Mixer, Ranges, clamp

CLEAR = Ranges()
WALL_AHEAD = Ranges(front=config.AVOID_HARD_DISTANCE / 2)


def settle(mixer, command, ranges=CLEAR, steps=60, **flags):
    """Step long enough for the smoothing to converge; return the last setpoint."""
    for _ in range(steps):
        setpoint = mixer.step(command, ranges, **flags)
    return setpoint


def test_hover_with_no_input_is_all_zero():
    sp = settle(Mixer(1.0), Command())
    assert (sp.vx, sp.vy, sp.vz, sp.yaw_rate) == pytest.approx((0, 0, 0, 0))


@pytest.mark.parametrize('forward', [1.0, -1.0, 0.5])
def test_forward_stick_scales_to_max_manual_speed(forward):
    sp = settle(Mixer(1.0), Command(forward=forward))
    assert sp.vx == pytest.approx(forward * config.MAX_MANUAL_SPEED)
    assert sp.vy == 0.0


@pytest.mark.parametrize('yaw', [1.0, -1.0, 0.5])
def test_turn_stick_scales_to_max_yaw_rate_without_moving(yaw):
    sp = settle(Mixer(1.0), Command(yaw=yaw))
    assert sp.yaw_rate == pytest.approx(yaw * config.MAX_YAW_RATE)
    assert (sp.vx, sp.vy) == pytest.approx((0, 0))


def test_stick_input_is_clamped():
    sp = settle(Mixer(1.0), Command(forward=9, yaw=-9))
    assert sp.vx == pytest.approx(config.MAX_MANUAL_SPEED)
    assert sp.yaw_rate == pytest.approx(-config.MAX_YAW_RATE)


@pytest.mark.parametrize('junk', [None, 'x', object()])
def test_non_numeric_input_is_treated_as_zero(junk):
    assert clamp(junk) == 0.0


def test_avoidance_is_added_under_a_full_forward_stick():
    sp = settle(Mixer(1.0), Command(forward=1.0), WALL_AHEAD)
    assert sp.vx == pytest.approx(config.MAX_MANUAL_SPEED - config.MAX_AVOID_SPEED)


def test_avoidance_off_gives_the_stick_full_authority():
    sp = settle(Mixer(1.0), Command(forward=1.0), WALL_AHEAD, avoid=False)
    assert sp.vx == pytest.approx(config.MAX_MANUAL_SPEED)


def test_lateral_motion_comes_only_from_avoidance():
    wall_left = Ranges(left=0.25)
    sp = settle(Mixer(1.0), Command(forward=0.0, yaw=1.0), wall_left)
    assert sp.vy == pytest.approx(repulsion(None, None, 0.25, None)[1])


def test_auto_ignores_stick_and_height_but_keeps_avoiding():
    mixer = Mixer(1.0)
    wall_right = Ranges(right=0.2)
    sp = settle(mixer, Command(forward=1, yaw=1, altitude=2.0), wall_right,
                auto=True)
    assert (sp.vx, sp.vz, sp.yaw_rate) == pytest.approx((0, 0, 0))
    assert sp.vy == pytest.approx(repulsion(None, None, None, 0.2)[1])
    assert mixer.altitude == pytest.approx(1.0)


def test_auto_avoids_even_with_the_avoidance_toggle_off():
    sp = settle(Mixer(1.0), Command(), Ranges(left=0.2), auto=True, avoid=False)
    assert sp.vy < 0


@pytest.mark.parametrize('start, target', [(1.0, 1.8), (1.0, 0.3), (0.5, 1.5)])
def test_altitude_converges_to_the_target(start, target):
    mixer = Mixer(start)
    settle(mixer, Command(altitude=target), steps=300)
    assert mixer.altitude == pytest.approx(target, abs=0.01)


@pytest.mark.parametrize('request_m, limit', [
    (99.0, config.MAX_ALTITUDE),
    (-5.0, config.MIN_ALTITUDE),
])
def test_altitude_target_is_clamped_to_the_envelope(request_m, limit):
    mixer = Mixer(1.0)
    settle(mixer, Command(altitude=request_m), steps=400)
    assert mixer.altitude == pytest.approx(limit)


def test_climb_rate_never_exceeds_the_cap():
    mixer = Mixer(config.MIN_ALTITUDE)
    rates = [mixer.step(Command(altitude=config.MAX_ALTITUDE), CLEAR).vz
             for _ in range(150)]
    assert max(abs(r) for r in rates) <= config.MAX_CLIMB_SPEED + 1e-9


def test_no_altitude_request_holds_height():
    mixer = Mixer(1.2)
    settle(mixer, Command(altitude=None), steps=50)
    assert mixer.altitude == pytest.approx(1.2)


def test_smoothing_ramps_instead_of_jumping():
    first = Mixer(1.0).step(Command(forward=1.0), CLEAR)
    assert 0 < first.vx < config.MAX_MANUAL_SPEED


def test_reset_starts_from_rest_at_the_new_height():
    mixer = Mixer(1.0)
    settle(mixer, Command(forward=1.0))
    mixer.reset(0.8)
    sp = mixer.step(Command(), CLEAR)
    assert sp.vx == 0.0
    assert mixer.altitude == pytest.approx(0.8)
