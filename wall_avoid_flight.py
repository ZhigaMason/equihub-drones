"""Autonomous hover-and-avoid flight for a Crazyflie 2.1 Brushless.

Requires a Flow deck v2 (height + position hold) and a Multi-ranger deck
(obstacle distances). The Crazyflie takes off to a fixed height, holds it
while gently pushing itself away from any wall the horizontal rangers see,
and lands as soon as the upward ranger reports a ceiling above it.

Settings come from `.env` via `drone.config`.
"""
import logging
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils.multiranger import Multiranger

from drone.avoidance import ceiling_detected, repulsion
from drone.config import (AVOID_DISTANCE, CEILING_DISTANCE, MAX_AVOID_SPEED,
                          MAX_FLIGHT_TIME, TAKEOFF_HEIGHT, TAKEOFF_VELOCITY,
                          URI)

# Control loop period (s).
UPDATE_PERIOD = 0.1
# Fraction of the new command blended in each cycle. Lower is gentler.
SMOOTHING = 0.3
# Consecutive ceiling readings needed before landing, so that a single
# spurious measurement cannot end the flight.
CEILING_SAMPLES = 3

logging.basicConfig(level=logging.ERROR)


def _wait_for_ranger_data(ranger, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        # `up` is None both when out of range and before the first packet, so
        # check a value that is always present once logging has started.
        if ranger.down is not None:
            return True
        time.sleep(0.1)
    return False


def _check_decks(scf):
    """Abort early if a deck the flight depends on is missing."""
    missing = []
    for param, name in (('deck.bcFlow2', 'Flow deck v2'),
                        ('deck.bcMultiranger', 'Multi-ranger deck')):
        if int(scf.cf.param.get_value(param, timeout=5)) != 1:
            missing.append(name)
    if missing:
        raise RuntimeError('Deck(s) not detected: ' + ', '.join(missing))


def fly(scf):
    _check_decks(scf)

    scf.cf.supervisor.send_arming_request(True)
    time.sleep(1.0)

    with Multiranger(scf) as ranger:
        if not _wait_for_ranger_data(ranger):
            raise RuntimeError('No data from the Multi-ranger deck')

        if ceiling_detected(ranger.up, CEILING_DISTANCE):
            raise RuntimeError(
                f'Ceiling already {ranger.up:.2f} m above the drone, not taking off')

        mc = MotionCommander(scf, default_height=TAKEOFF_HEIGHT)
        vx = vy = 0.0
        ceiling_hits = 0
        reason = 'time limit reached'

        try:
            print(f'Taking off to {TAKEOFF_HEIGHT:.2f} m...')
            mc.take_off(TAKEOFF_HEIGHT, TAKEOFF_VELOCITY)
            time.sleep(1.0)

            print('Hovering. Ctrl-C to land.')
            started = time.time()
            while True:
                if time.time() - started > MAX_FLIGHT_TIME:
                    break

                if ceiling_detected(ranger.up, CEILING_DISTANCE):
                    ceiling_hits += 1
                    if ceiling_hits >= CEILING_SAMPLES:
                        reason = f'ceiling at {ranger.up:.2f} m'
                        break
                else:
                    ceiling_hits = 0

                target_vx, target_vy = repulsion(
                    ranger.front, ranger.back, ranger.left, ranger.right,
                    AVOID_DISTANCE, MAX_AVOID_SPEED)

                vx += (target_vx - vx) * SMOOTHING
                vy += (target_vy - vy) * SMOOTHING

                mc.start_linear_motion(vx, vy, 0.0)
                time.sleep(UPDATE_PERIOD)
        except KeyboardInterrupt:
            reason = 'interrupted'
        finally:
            print(f'Landing ({reason})...')
            # land() zeroes horizontal motion and is a no-op if take-off
            # never got the drone airborne.
            mc.land(TAKEOFF_VELOCITY)
            print('Landed.')


def main():
    cflib.crtp.init_drivers()
    print(f'Connecting to {URI}')
    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            fly(scf)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
