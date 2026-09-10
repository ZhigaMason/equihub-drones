"""Helpers for talking to a real Crazyflie over cflib.

Everything cflib-specific that is not flight logic lives here, so the
controller and the missions share one implementation of connecting,
pre-flight checks and sensor reads.
"""
import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from drones.control.mixer import Ranges

logger = logging.getLogger(__name__)

# cflib caches each drone's parameter and log tables here, which makes
# reconnecting much faster.
CACHE_DIR = './cache'

REQUIRED_DECKS = (('deck.bcFlow2', 'Flow deck v2'),
                  ('deck.bcMultiranger', 'Multi-ranger deck'))


def resolve_uri(configured):
    """Pick the interface to connect to, and explain clearly when we cannot.

    A bare 'Cannot find a Crazyradio Dongle' is useless when the drone is
    sitting right there on a USB cable, so every failure names what a scan
    did find.
    """
    available = [uri for uri, _ in cflib.crtp.scan_interfaces()]

    if configured and configured.lower() != 'auto':
        if not available or configured in available:
            # Radio URIs do not always show up in a scan (the drone may be
            # off), so try the configured one rather than second-guessing it.
            return configured
        raise RuntimeError(
            f'{configured} is not available. Found: {", ".join(available)}. '
            f'Set CFLIB_URI in .env to one of those, or to "auto".')

    if not available:
        raise RuntimeError(
            'No Crazyflie found. Plug in the Crazyradio dongle (or the drone '
            'over USB) and switch the drone on.')
    if len(available) > 1:
        raise RuntimeError(
            f'Several interfaces found: {", ".join(available)}. '
            f'Set CFLIB_URI in .env to the one you want.')
    logger.info('Auto-selected %s', available[0])
    return available[0]


def open_link(uri):
    """A SyncCrazyflie for `uri`, to be used as a context manager."""
    return SyncCrazyflie(uri, cf=Crazyflie(rw_cache=CACHE_DIR))


def check_decks(scf):
    """Abort early if a deck the flight depends on is missing."""
    missing = [name for param, name in REQUIRED_DECKS
               if int(scf.cf.param.get_value(param, timeout=5)) != 1]
    if missing:
        raise RuntimeError('Deck(s) not detected: ' + ', '.join(missing))


def wait_for_ranger_data(ranger, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        # `up` is None both when out of range and before the first packet, so
        # check a value that is always present once logging has started.
        if ranger.down is not None:
            return True
        time.sleep(0.1)
    return False


def read_ranges(ranger):
    """Snapshot a cflib Multiranger as hardware-free `Ranges`."""
    return Ranges(front=ranger.front, back=ranger.back, left=ranger.left,
                  right=ranger.right, up=ranger.up, down=ranger.down)


def abort_motion(mc):
    """Tear a MotionCommander down without the descent that land() performs.

    Mirrors MotionCommander.land() minus the `down()` call. It reaches into
    the private setpoint thread because there is no public way to stop
    streaming setpoints without first flying the drone to the ground.
    """
    if not mc._is_flying:
        return
    mc._thread.stop()
    mc._thread = None
    mc._cf.commander.send_stop_setpoint()
    mc._cf.commander.send_notify_setpoint_stop()
    mc._is_flying = False


class BatteryLog:
    """Context manager logging the battery voltage at 1 Hz."""

    def __init__(self, scf):
        self._cf = scf.cf
        self.voltage = None
        self._config = LogConfig('battery', 1000)
        self._config.add_variable('pm.vbat', 'float')
        self._config.data_received_cb.add_callback(self._received)

    def _received(self, timestamp, data, logconf):
        self.voltage = round(data['pm.vbat'], 2)

    def __enter__(self):
        self._cf.log.add_config(self._config)
        self._config.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._config.delete()
