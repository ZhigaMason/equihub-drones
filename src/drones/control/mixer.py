"""The flight control law: operator input + ranger readings -> velocity setpoint.

Pure and hardware-free, so the real Crazyflie, a simulator and an RL
environment all fly with identical behaviour. Nothing here imports cflib,
starts a thread or reads the clock: time advances one `UPDATE_PERIOD` per
`Mixer.step()` call.
"""
from dataclasses import dataclass

from drones import config
from drones.control.avoidance import repulsion

# Control loop period (s). Every consumer steps the mixer at this rate.
UPDATE_PERIOD = 0.1
# Fraction of the new horizontal command blended in each step. Lower is
# gentler but slower to react; a sweep showed no oscillation even at 1.0, so
# this is tuned for response rather than stability margin.
SMOOTHING = 0.5
# Proportional gain turning an altitude error (m) into a climb rate (m/s).
ALTITUDE_GAIN = 1.5


@dataclass(frozen=True)
class Ranges:
    """One Multi-ranger reading in metres. None means nothing in range."""
    front: float | None = None
    back: float | None = None
    left: float | None = None
    right: float | None = None
    up: float | None = None
    down: float | None = None


@dataclass(frozen=True)
class Command:
    """What the operator, or a policy, asks for this step.

    `forward` and `yaw` are normalised to -1..1. `altitude` is an absolute
    target in metres; None holds the current one.
    """
    forward: float = 0.0
    yaw: float = 0.0
    altitude: float | None = None


@dataclass(frozen=True)
class Setpoint:
    """Body-frame velocity: +x forward, +y left, +z up; yaw rate in deg/s."""
    vx: float
    vy: float
    vz: float
    yaw_rate: float


def clamp(value, low=-1.0, high=1.0):
    """Clamp to [low, high]; anything non-numeric becomes 0."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(low, min(high, value))


class Mixer:
    """Turns commands and ranger readings into setpoints, one step at a time.

    Holds the only state the control law needs: the smoothed horizontal
    velocity and the commanded altitude. MotionCommander integrates
    velocity_z into an absolute height setpoint, so `altitude` mirrors that
    integration and is what the altitude envelope is enforced against.
    """

    def __init__(self, altitude=0.0):
        self.reset(altitude)

    def reset(self, altitude):
        """Start from rest at `altitude`, e.g. right after take-off."""
        self.altitude = altitude
        self._vx = 0.0
        self._vy = 0.0

    def step(self, command, ranges, *, auto=False, avoid=True):
        """Advance one UPDATE_PERIOD and return the setpoint to send.

        Manual input drives forward/back, turn and height. There is no
        sideways control: lateral motion only ever comes from the avoidance
        vector. In `auto` mode manual input is ignored, height is held and
        avoidance is on regardless of `avoid`.
        """
        if auto:
            manual_x = yaw = 0.0
        else:
            manual_x = clamp(command.forward) * config.MAX_MANUAL_SPEED
            yaw = clamp(command.yaw) * config.MAX_YAW_RATE

        if auto or avoid:
            push_x, push_y = repulsion(ranges.front, ranges.back,
                                       ranges.left, ranges.right)
        else:
            push_x = push_y = 0.0

        vz = 0.0 if auto else self._climb_rate(command.altitude)
        self.altitude = clamp(self.altitude + vz * UPDATE_PERIOD,
                              config.MIN_ALTITUDE, config.MAX_ALTITUDE)

        self._vx += (manual_x + push_x - self._vx) * SMOOTHING
        self._vy += (push_y - self._vy) * SMOOTHING
        return Setpoint(self._vx, self._vy, vz, yaw)

    def _climb_rate(self, desired):
        """Proportional climb towards `desired`, so the height control reads
        as an absolute altitude rather than a climb command."""
        if desired is None:
            return 0.0
        desired = clamp(desired, config.MIN_ALTITUDE, config.MAX_ALTITUDE)
        vz = clamp(ALTITUDE_GAIN * (desired - self.altitude),
                   -config.MAX_CLIMB_SPEED, config.MAX_CLIMB_SPEED)
        if vz > 0 and self.altitude >= config.MAX_ALTITUDE:
            return 0.0
        if vz < 0 and self.altitude <= config.MIN_ALTITUDE:
            return 0.0
        return vz
