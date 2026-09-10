"""Autonomous hover-and-avoid flight for a Crazyflie 2.1 Brushless.

Requires a Flow deck v2 (height + position hold) and a Multi-ranger deck
(obstacle distances). The Crazyflie takes off to a fixed height, holds it
while pushing itself away from any wall the horizontal rangers see, and lands
as soon as the upward ranger reports a ceiling above it.

It flies the same control law as the phone page's auto mode.

Run with:  uv run drones-wall-avoid
"""
import logging
import sys
import time

import cflib.crtp
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils.multiranger import Multiranger

from drones import config
from drones.control.avoidance import ceiling_detected
from drones.control.mixer import UPDATE_PERIOD, Command, Mixer
from drones.control.safety import CeilingMonitor
from drones.crazyflie.link import (check_decks, open_link, read_ranges,
                                   resolve_uri, wait_for_ranger_data)


def fly(scf):
    check_decks(scf)

    scf.cf.supervisor.send_arming_request(True)
    time.sleep(1.0)

    with Multiranger(scf) as ranger:
        if not wait_for_ranger_data(ranger):
            raise RuntimeError('No data from the Multi-ranger deck')

        if ceiling_detected(ranger.up):
            raise RuntimeError(
                f'Ceiling already {ranger.up:.2f} m above the drone, not taking off')

        height = config.TAKEOFF_HEIGHT
        mc = MotionCommander(scf, default_height=height)
        mixer = Mixer(height)
        ceiling = CeilingMonitor()
        reason = 'time limit reached'

        try:
            print(f'Taking off to {height:.2f} m...')
            mc.take_off(height, config.TAKEOFF_VELOCITY)
            time.sleep(1.0)

            print('Hovering. Ctrl-C to land.')
            started = time.time()
            while time.time() - started <= config.MAX_FLIGHT_TIME:
                ranges = read_ranges(ranger)
                if ceiling.update(ranges.up):
                    reason = f'ceiling at {ranges.up:.2f} m'
                    break

                # Auto mode: no operator input, height held, avoidance on.
                setpoint = mixer.step(Command(), ranges, auto=True)
                mc.start_linear_motion(setpoint.vx, setpoint.vy, setpoint.vz)
                time.sleep(UPDATE_PERIOD)
        except KeyboardInterrupt:
            reason = 'interrupted'
        finally:
            print(f'Landing ({reason})...')
            # land() zeroes horizontal motion and is a no-op if take-off
            # never got the drone airborne.
            mc.land(config.TAKEOFF_VELOCITY)
            print('Landed.')


def main():
    logging.basicConfig(level=logging.ERROR)
    cflib.crtp.init_drivers()
    try:
        uri = resolve_uri(config.URI)
        print(f'Connecting to {uri}')
        with open_link(uri) as scf:
            fly(scf)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
