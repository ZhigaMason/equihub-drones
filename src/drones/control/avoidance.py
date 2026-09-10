"""Turns Multi-ranger distances into a body-frame velocity that avoids walls."""
from drones.config import (AVOID_DISTANCE, AVOID_HARD_DISTANCE,
                          CEILING_DISTANCE, MAX_AVOID_SPEED)


def _ramp(distance, avoid_distance, hard_distance, max_speed):
    """How hard one ranger pushes, given what it reads.

    Zero at `avoid_distance`, ramping to the full `max_speed` by
    `hard_distance` and staying there closer in. The saturation point matters:
    a ramp that only reaches full speed at the wall itself is still barely
    pushing at the distance where it actually needs to act.

    A reading of None means "nothing in range" and contributes nothing.
    """
    if distance is None or distance >= avoid_distance:
        return 0.0
    if distance <= hard_distance:
        return max_speed
    span = avoid_distance - hard_distance
    return max_speed * (avoid_distance - distance) / span


def _push(near, far, avoid_distance, max_speed, hard_distance):
    """Speed (m/s) pushing away from `near`, opposed by the `far` side.

    Opposing sensors cancel, so a narrow corridor centres the drone instead of
    making it oscillate.
    """
    speed = (_ramp(near, avoid_distance, hard_distance, max_speed)
             - _ramp(far, avoid_distance, hard_distance, max_speed))
    return max(-max_speed, min(max_speed, speed))


def repulsion(front, back, left, right,
              avoid_distance=AVOID_DISTANCE, max_speed=MAX_AVOID_SPEED,
              hard_distance=AVOID_HARD_DISTANCE):
    """Body-frame (vx, vy) in m/s that moves the drone away from nearby walls.

    Body frame is +x forward, +y left, so moving away from the front ranger
    means going backwards and moving away from the right ranger means going
    left.
    """
    hard_distance = min(hard_distance, avoid_distance)
    vx = -_push(front, back, avoid_distance, max_speed, hard_distance)
    vy = _push(right, left, avoid_distance, max_speed, hard_distance)
    return vx, vy


def ceiling_detected(up, ceiling_distance=CEILING_DISTANCE):
    """True if the upward ranger sees something close enough to be a ceiling."""
    return up is not None and up < ceiling_distance
