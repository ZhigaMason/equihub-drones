"""Owns the Crazyflie link and runs the flight control loop on its own thread.

Everything that talks to cflib happens on the single `_thread_main` thread.
The web layer only pushes commands into a queue and reads an immutable
telemetry snapshot, so no cflib object is ever touched from an async handler.
"""
import logging
import queue
import threading
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils.multiranger import Multiranger

from drone import config
from drone.avoidance import ceiling_detected, repulsion

logger = logging.getLogger(__name__)

# Control loop period (s).
UPDATE_PERIOD = 0.1
# Fraction of the new command blended in each cycle. Lower is gentler, but
# also slower to react; a sweep showed no oscillation even at 1.0, so this is
# tuned for response rather than stability margin.
SMOOTHING = 0.5
# Proportional gain turning an altitude error (m) into a climb rate (m/s).
ALTITUDE_GAIN = 1.5
# Consecutive ceiling readings needed before landing, so that a single
# spurious measurement cannot end the flight.
CEILING_SAMPLES = 3

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
            self._forward = _clamp(forward)
            self._yaw = _clamp(yaw)
            if altitude is not None:
                self._desired_altitude = _clamp(
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

        uri = _resolve_uri(config.URI)
        self._publish(CONNECTING, f'Connecting to {uri}')

        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            self._check_decks(scf)
            with Multiranger(scf) as ranger, _battery_log(scf) as battery:
                if not _wait_for_data(ranger):
                    raise RuntimeError('No data from the Multi-ranger deck')
                self._publish(IDLE, 'Connected and ready')
                self._ground_loop(scf, ranger, battery)

    def _drain_commands(self):
        while True:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                return

    def _check_decks(self, scf):
        missing = [name for param, name in
                   (('deck.bcFlow2', 'Flow deck v2'),
                    ('deck.bcMultiranger', 'Multi-ranger deck'))
                   if int(scf.cf.param.get_value(param, timeout=5)) != 1]
        if missing:
            raise RuntimeError('Deck(s) not detected: ' + ', '.join(missing))

    def _ground_loop(self, scf, ranger, battery):
        """On the ground and connected: wait for a take-off or disconnect."""
        while not self._shutdown.is_set():
            self._update_telemetry(ranger, battery)
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
                    _abort_motion(mc)
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
        vx = vy = 0.0
        ceiling_hits = 0
        started = time.time()

        while not self._shutdown.is_set():
            self._update_telemetry(ranger, battery)

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

            with self._lock:
                auto = self._auto_mode
                avoid_on = self._avoid_enabled

            # A ceiling always ends the flight, in every mode.
            if ceiling_detected(ranger.up):
                ceiling_hits += 1
                if ceiling_hits >= CEILING_SAMPLES:
                    return f'ceiling at {ranger.up:.2f} m'
            else:
                ceiling_hits = 0

            if auto and time.time() - started > config.MAX_FLIGHT_TIME:
                return 'auto time limit'

            # Watchdog: a locked phone screen or dropped Wi-Fi must not leave
            # the drone flying on its last stick input.
            silent = self._client_silent_for()
            if silent > config.LINK_TIMEOUT and not auto:
                return f'no client for {silent:.1f} s'
            # Stop turning when the client goes quiet, but keep holding the
            # last commanded height: hovering is the safe default.
            if silent > config.STICK_TIMEOUT:
                self._set(forward=0.0, yaw=0.0)

            target_vx, target_vy, vz, yaw = self._mix(ranger, auto, avoid_on)
            vx += (target_vx - vx) * SMOOTHING
            vy += (target_vy - vy) * SMOOTHING

            mc.start_linear_motion(vx, vy, vz, yaw)
            time.sleep(UPDATE_PERIOD)

        return 'shutting down'

    def _mix(self, ranger, auto, avoid_on):
        """Combine stick input, height tracking, wall repulsion and the limits.

        The stick drives forward/back and turn. There is no sideways control:
        lateral motion comes only from the avoidance vector.
        """
        with self._lock:
            altitude = self._target_altitude
            desired = self._desired_altitude
            if auto:
                manual_x = yaw = 0.0
            else:
                manual_x = self._forward * config.MAX_MANUAL_SPEED
                yaw = self._yaw * config.MAX_YAW_RATE

        if auto or avoid_on:
            push_x, push_y = repulsion(ranger.front, ranger.back,
                                       ranger.left, ranger.right)
        else:
            push_x = push_y = 0.0

        # Track the requested height with a proportional climb rate, so the
        # slider reads as an absolute altitude rather than a climb command.
        if auto or desired is None:
            vz = 0.0
        else:
            vz = _clamp(ALTITUDE_GAIN * (desired - altitude),
                        -config.MAX_CLIMB_SPEED, config.MAX_CLIMB_SPEED)

        # MotionCommander integrates velocity_z into an absolute setpoint, so
        # the commanded altitude is tracked and clamped here too.
        if vz > 0 and altitude >= config.MAX_ALTITUDE:
            vz = 0.0
        elif vz < 0 and altitude <= config.MIN_ALTITUDE:
            vz = 0.0
        self._set(target_altitude=_clamp(altitude + vz * UPDATE_PERIOD,
                                         config.MIN_ALTITUDE,
                                         config.MAX_ALTITUDE))

        return manual_x + push_x, push_y, vz, yaw

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

    def _update_telemetry(self, ranger, battery):
        self._set(
            telemetry={
                'front': ranger.front, 'back': ranger.back,
                'left': ranger.left, 'right': ranger.right,
                'up': ranger.up, 'down': ranger.down,
            },
            battery=battery.voltage,
        )


def _abort_motion(mc):
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


def _resolve_uri(configured):
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


def _clamp(value, low=-1.0, high=1.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(low, min(high, value))


def _wait_for_data(ranger, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        # `up` is None both when out of range and before the first packet, so
        # check a value that is always present once logging has started.
        if ranger.down is not None:
            return True
        time.sleep(0.1)
    return False


class _battery_log:
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
