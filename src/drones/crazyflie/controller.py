"""Owns the Crazyflie link and runs the flight state machine on its own thread.

Everything that talks to cflib happens on the single `_thread_main` thread.
The web layer only pushes commands into a queue and reads an immutable
telemetry snapshot, so no cflib object is ever touched from an async handler.

The control law itself lives in `drones.control`; this module decides *when*
to fly, and what to do about operators, links and emergencies.
"""
import logging
import queue
import threading
import time
from dataclasses import asdict

import cflib.crtp
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils.multiranger import Multiranger

from drones import config
from drones.control.avoidance import ceiling_detected
from drones.control.mixer import UPDATE_PERIOD, Command, Mixer, clamp
from drones.control.safety import CeilingMonitor
from drones.crazyflie.link import (BatteryLog, abort_motion, check_decks,
                                   open_link, read_ranges, resolve_uri,
                                   wait_for_ranger_data)

logger = logging.getLogger(__name__)

# Lifecycle states reported to the UI.
DISCONNECTED = 'disconnected'
CONNECTING = 'connecting'
IDLE = 'idle'
TAKING_OFF = 'taking_off'
FLYING = 'flying'
LANDING = 'landing'
ERROR = 'error'


class DroneController:
    def __init__(self):
        self._commands = queue.Queue()
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        # Flight thread only; the web layer never touches it.
        self._mixer = Mixer()

        # --- shared state, guarded by _lock ---
        self._state = DISCONNECTED
        self._message = 'Not connected'
        self._telemetry = {}
        self._battery = None
        self._avoid_enabled = True
        self._auto_mode = False
        # Manual input is forward/back, turn and height. Sideways motion
        # comes from the avoidance vector alone.
        self._forward = 0.0                  # normalised -1..1
        self._yaw = 0.0                      # normalised -1..1
        self._desired_altitude = None        # metres, None = hold current
        self._last_client = 0.0
        self._target_altitude = 0.0
        self._estopped = False

        self._thread = threading.Thread(target=self._thread_main, daemon=True)

    # ------------------------------------------------------------------
    # Public API, called from the web layer
    # ------------------------------------------------------------------
    def start(self):
        cflib.crtp.init_drivers()
        self._thread.start()

    def stop(self):
        self._shutdown.set()
        self._commands.put(('disconnect', None))
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def submit(self, name, payload=None):
        """Queue a command for the flight thread. Never blocks."""
        self._commands.put((name, payload))

    def set_control(self, forward, yaw, altitude=None):
        """Set forward/back and turn (both normalised -1..1) and the desired
        altitude in metres, and feed the watchdog."""
        with self._lock:
            self._forward = clamp(forward)
            self._yaw = clamp(yaw)
            if altitude is not None:
                self._desired_altitude = clamp(
                    altitude, config.MIN_ALTITUDE, config.MAX_ALTITUDE)
            self._last_client = time.time()

    def heartbeat(self):
        with self._lock:
            self._last_client = time.time()

    def snapshot(self):
        """Immutable view of the current state, safe to serialise to JSON."""
        with self._lock:
            return {
                'state': self._state,
                'message': self._message,
                'battery': self._battery,
                'avoid': self._avoid_enabled,
                'auto': self._auto_mode,
                'altitude': round(self._target_altitude, 2),
                'desired_altitude': (None if self._desired_altitude is None
                                     else round(self._desired_altitude, 2)),
                'ranges': dict(self._telemetry),
                # The page draws the height slider from these.
                'limits': {'min': config.MIN_ALTITUDE,
                           'max': config.MAX_ALTITUDE},
            }

    # ------------------------------------------------------------------
    # Internal helpers, flight thread only
    # ------------------------------------------------------------------
    def _set(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, '_' + key, value)

    @staticmethod
    def _brief(exc):
        """First line of an exception. cflib embeds whole tracebacks in its
        link-error messages, which are unreadable on a phone."""
        text = str(exc).strip().splitlines()
        return (text[0] if text else exc.__class__.__name__)[:140]

    def _apply_setting(self, name, payload):
        """Toggles that are valid in every state, including on the ground
        and before the link is up."""
        if name == 'avoid':
            self._set(avoid_enabled=bool(payload))
        elif name == 'auto':
            self._set(auto_mode=bool(payload))
        else:
            return False
        return True

    def _publish(self, state=None, message=None):
        if message:
            logger.info(message)
        with self._lock:
            if state is not None:
                self._state = state
            if message is not None:
                self._message = message

    def _client_silent_for(self):
        with self._lock:
            return time.time() - self._last_client

    # ------------------------------------------------------------------
    # Flight thread
    # ------------------------------------------------------------------
    def _thread_main(self):
        while not self._shutdown.is_set():
            try:
                name, payload = self._commands.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._apply_setting(name, payload):
                continue
            if name == 'connect':
                try:
                    self._session()
                except Exception as exc:               # noqa: BLE001
                    logger.exception('Session ended with an error')
                    self._publish(ERROR, f'Error: {self._brief(exc)}')
                finally:
                    self._set(telemetry={}, battery=None, target_altitude=0.0)
                    if self._state != ERROR:
                        self._publish(DISCONNECTED, 'Disconnected')

    def _session(self):
        """Connect, hold the link open, and run flights until disconnected."""
        self._publish(CONNECTING, 'Looking for a Crazyflie')
        self._drain_commands()

        uri = resolve_uri(config.URI)
        self._publish(CONNECTING, f'Connecting to {uri}')

        with open_link(uri) as scf:
            check_decks(scf)
            with Multiranger(scf) as ranger, BatteryLog(scf) as battery:
                if not wait_for_ranger_data(ranger):
                    raise RuntimeError('No data from the Multi-ranger deck')
                self._publish(IDLE, 'Connected and ready')
                self._ground_loop(scf, ranger, battery)

    def _drain_commands(self):
        while True:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                return

    def _ground_loop(self, scf, ranger, battery):
        """On the ground and connected: wait for a take-off or disconnect."""
        while not self._shutdown.is_set():
            self._update_telemetry(read_ranges(ranger), battery)
            try:
                name, payload = self._commands.get(timeout=UPDATE_PERIOD)
            except queue.Empty:
                continue

            if name == 'disconnect':
                return
            if name == 'estop':
                scf.cf.supervisor.send_emergency_stop()
                self._publish(IDLE, 'Emergency stop (was on the ground)')
            elif name == 'recover':
                # Clears the supervisor lock left by an emergency stop or a
                # tumble, so the drone can be armed again without a reboot.
                scf.cf.supervisor.send_crash_recovery_request()
                self._publish(IDLE, 'Crash recovery sent')
            elif self._apply_setting(name, payload):
                pass
            elif name == 'takeoff':
                if ceiling_detected(ranger.up):
                    self._publish(IDLE, f'Ceiling {ranger.up:.2f} m above, '
                                        'refusing to take off')
                    continue
                self._set(auto_mode=bool(payload), forward=0.0, yaw=0.0,
                          desired_altitude=None)
                self._flight(scf, ranger, battery)
                if not scf.is_link_open():
                    return
                # A flight error leaves us connected and on the ground, so
                # the message stays visible and Recover is still reachable.
                self._publish(IDLE, 'On the ground')

    def _flight(self, scf, ranger, battery):
        """Take off, fly until told to land, then land. Always lands."""
        height = min(config.TAKEOFF_HEIGHT, config.MAX_ALTITUDE)
        mc = MotionCommander(scf, default_height=height)
        self._estopped = False
        reason = 'landing'
        try:
            self._publish(TAKING_OFF, f'Taking off to {height:.2f} m')
            self.heartbeat()
            scf.cf.supervisor.send_arming_request(True)
            time.sleep(1.0)
            mc.take_off(height, config.TAKEOFF_VELOCITY)
            self._mixer.reset(height)
            self._set(target_altitude=height, desired_altitude=height)
            time.sleep(1.0)
            self._publish(FLYING, 'Flying')
            reason = self._fly_loop(scf, ranger, battery, mc)
        except Exception as exc:                        # noqa: BLE001
            logger.exception('Flight failed')
            self._publish(ERROR, f'Flight error: {self._brief(exc)}')
            reason = 'error'
        finally:
            try:
                if self._estopped:
                    # The motors are already cut; descending would just send
                    # three seconds of setpoints to a locked drone.
                    abort_motion(mc)
                else:
                    self._publish(LANDING, f'Landing ({reason})')
                    # land() zeroes horizontal motion and is a no-op if
                    # take-off never got the drone airborne.
                    mc.land(config.TAKEOFF_VELOCITY)
            except Exception:                           # noqa: BLE001
                logger.exception('Landing failed, sending emergency stop')
                scf.cf.supervisor.send_emergency_stop()
            self._set(target_altitude=0.0, desired_altitude=None)

    def _fly_loop(self, scf, ranger, battery, mc):
        ceiling = CeilingMonitor()
        started = time.time()

        while not self._shutdown.is_set():
            # One read per cycle, so telemetry and control see the same data.
            ranges = read_ranges(ranger)
            self._update_telemetry(ranges, battery)

            action = self._drain_pending()
            if action == 'estop':
                scf.cf.supervisor.send_emergency_stop()
                self._publish(ERROR, 'EMERGENCY STOP - motors cut')
                self._estopped = True
                return 'emergency stop'
            if action == 'land':
                return 'commanded'
            if action == 'disconnect':
                return 'client disconnected'

            # A ceiling always ends the flight, in every mode.
            if ceiling.update(ranges.up):
                return f'ceiling at {ranges.up:.2f} m'

            with self._lock:
                auto = self._auto_mode
                avoid_on = self._avoid_enabled

            if auto and time.time() - started > config.MAX_FLIGHT_TIME:
                return 'auto time limit'

            # Watchdog: a locked phone screen or dropped Wi-Fi must not leave
            # the drone flying on its last stick input.
            silent = self._client_silent_for()
            if silent > config.LINK_TIMEOUT and not auto:
                return f'no client for {silent:.1f} s'
            # Stop moving when the client goes quiet, but keep holding the
            # last commanded height: hovering is the safe default.
            if silent > config.STICK_TIMEOUT:
                self._set(forward=0.0, yaw=0.0)

            with self._lock:
                command = Command(self._forward, self._yaw,
                                  self._desired_altitude)
            setpoint = self._mixer.step(command, ranges, auto=auto,
                                        avoid=avoid_on)
            self._set(target_altitude=self._mixer.altitude)

            mc.start_linear_motion(setpoint.vx, setpoint.vy, setpoint.vz,
                                   setpoint.yaw_rate)
            time.sleep(UPDATE_PERIOD)

        return 'shutting down'

    # Terminal actions, most urgent first. Settings are applied as they are
    # seen; only the winning terminal action is returned.
    _PRIORITY = ('estop', 'land', 'disconnect')

    def _drain_pending(self):
        """Consume every queued command so an emergency stop is never stuck
        behind a burst of toggles. Returns the most urgent terminal action."""
        action = None
        while True:
            try:
                name, payload = self._commands.get_nowait()
            except queue.Empty:
                return action
            if name == 'auto':
                self._set(forward=0.0, yaw=0.0)
            if self._apply_setting(name, payload):
                continue
            if name in self._PRIORITY and (
                    action is None
                    or self._PRIORITY.index(name) < self._PRIORITY.index(action)):
                action = name

    def _update_telemetry(self, ranges, battery):
        self._set(telemetry=asdict(ranges), battery=battery.voltage)
