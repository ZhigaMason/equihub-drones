"""Fly an exported hover policy on the real Crazyflie.

    uv run drones-fly-policy runs/<name>/policy --dry-run         # sensors and policy live, motors off
    uv run drones-fly-policy runs/<name>/policy --authority 0.3   # first flights: 30% of the policy
    uv run drones-fly-policy runs/<name>/policy

The firmware takes off and hovers on its own Flow-deck controller. In the air the policy takes over
through the attitude commander, and the firmware takes back control to land when time is up, on
Ctrl-C, or the moment a safety limit trips. While the firmware hovers, the script measures the
thrust command that holds this particular drone up; that maps the policy's thrust (newtons in the
simulator) onto the real command.

Needs only numpy and cflib: the policy runs from its numpy artifact.
"""
import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drones.policy.runtime import Policy, PolicyRunner

logger = logging.getLogger(__name__)

# The Flow deck driver swaps and negates the PMW3901 axes before the firmware uses them
# (flowdeck_v1v2.c: accpx = -deltaY, accpy = -deltaX), and the firmware's flow model reads them at
# a tenth of a pixel (mm_flow.c: FLOW_RESOLUTION 0.10, "10x the motion pixels, experimentally
# measured"). Applying both gives the pixels the simulator's flow model reports.
FLOW_RESOLUTION = 0.10

# Signs taking the simulator's right-handed attitude command to cflib's send_setpoint. From the
# firmware: the legacy RPYT commander inverts yaw rate (crtp_commander_rpyt.c, "legacy rate input is
# inverted") and uses the legacy CF2 pitch, which is inverted (stateEstimate.pitch). Roll is taken as
# is. Verify on a tethered drone before the first free flight; each is a command-line flag.
DEFAULT_SIGNS = {'roll': 1.0, 'pitch': -1.0, 'yaw_rate': -1.0}

THRUST_COMMAND_MIN = 10000   # of 65535: near idle
THRUST_COMMAND_MAX = 60000   # leaves the attitude loop some headroom
LOG_PERIOD_MS = 10
RANGERS = ('front', 'back', 'left', 'right', 'up')

# Log blocks, each within cflib's 26-byte packet.
LOG_BLOCKS = {
    'policy_flow_ranges': [('motion.deltaX', 'int16_t'), ('motion.deltaY', 'int16_t'),
                           ('range.zrange', 'uint16_t')]
                          + [(f'range.{r}', 'uint16_t') for r in RANGERS],
    'policy_gyro_thrust': [('gyro.x', 'float'), ('gyro.y', 'float'), ('gyro.z', 'float'),
                           ('controller.cmd_thrust', 'float')],
    'policy_attitude': [(f'stateEstimate.q{a}', 'float') for a in 'xyzw'],
}


@dataclass(frozen=True)
class Limits:
    max_tilt_deg: float = 30.0
    min_range: float = 0.2              # any Multi-ranger reading closer than this aborts
    min_height: float = 0.15
    max_height_above_target: float = 1.0
    max_log_age: float = 0.25           # seconds without fresh sensor data aborts


@dataclass(frozen=True)
class FlightOptions:
    height: float = 1.0
    duration: float = 10.0
    authority: float = 1.0
    signs: dict = field(default_factory=lambda: dict(DEFAULT_SIGNS))
    limits: Limits = field(default_factory=Limits)
    takeoff_seconds: float = 2.0
    calibrate_seconds: float = 3.0
    dry_run: bool = False


# ---------------------------------------------------------------------- pure conversions
def gravity_and_tilt(quat):
    """Unit gravity direction in the body frame, and tilt in degrees, from [x, y, z, w]."""
    x, y, z, w = quat
    r22 = 1 - 2 * (x * x + y * y)
    gravity = -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), r22])
    return gravity / np.linalg.norm(gravity), math.degrees(math.acos(max(-1.0, min(1.0, r22))))


def read_inputs(latest, spec):
    """Latest cflib log values -> (policy inputs in the simulator's units and axes, tilt in deg)."""
    flow_px = FLOW_RESOLUTION * np.array([-latest['motion.deltaY'], -latest['motion.deltaX']],
                                         float)

    def metres(mm):
        return min(max(mm / 1000.0, 0.0), spec.range_max)

    gravity, tilt = gravity_and_tilt([latest[f'stateEstimate.q{a}'] for a in 'xyzw'])
    inputs = {
        'flow_rate': flow_px / spec.flow_gain,
        'zrange': metres(latest['range.zrange']),
        'ranges': np.array([metres(latest[f'range.{r}']) for r in RANGERS]),
        'gyro': np.radians([latest['gyro.x'], latest['gyro.y'], latest['gyro.z']]),
        'gravity': gravity,
    }
    return inputs, tilt


def to_setpoint(action, spec, hover_command, signs=DEFAULT_SIGNS, authority=1.0):
    """Normalised action -> send_setpoint(roll deg, pitch deg, yaw rate deg/s, thrust command).

    `authority` scales the whole action towards hover: 0 is pure hover, 1 the full policy. Thrust
    maps through its ratio to the simulator's hover thrust, anchored on `hover_command`, the command
    measured to hold this drone up.
    """
    roll, pitch, yaw_rate, thrust = spec.decode(np.clip(action, -1, 1) * authority)
    command = hover_command * float(thrust) / spec.hover_thrust
    command = int(round(min(max(command, THRUST_COMMAND_MIN), THRUST_COMMAND_MAX)))
    return (signs['roll'] * math.degrees(float(roll)),
            signs['pitch'] * math.degrees(float(pitch)),
            signs['yaw_rate'] * math.degrees(float(yaw_rate)),
            command)


def abort_reason(inputs, tilt, log_age, target, limits):
    """Why the policy must hand back control right now, or None."""
    if log_age > limits.max_log_age:
        return f'sensor data {log_age:.2f} s old'
    if tilt > limits.max_tilt_deg:
        return f'tilt {tilt:.0f} deg'
    nearest = float(np.min(inputs['ranges']))
    if nearest < limits.min_range:
        return f'obstacle {nearest:.2f} m away'
    height = inputs['zrange']
    if height < limits.min_height or height > target + limits.max_height_above_target:
        return f'height {height:.2f} m'
    return None


def ticks(duration, freq, now, sleep):
    """Yield loop indices at `freq` Hz for `duration` seconds, without drifting."""
    start, i = now(), 0
    while now() - start < duration:
        yield i
        i += 1
        delay = start + i / freq - now()
        if delay > 0:
            sleep(delay)


# ---------------------------------------------------------------------- the drone
class SensorLog:
    """Streams every variable the policy and the safety checks need, at 100 Hz."""

    def __init__(self, scf, now=time.monotonic):
        from cflib.crazyflie.log import LogConfig
        self._cf = scf.cf
        self._now = now
        self.latest = {}
        self._arrived = {}
        self._configs = []
        for name, variables in LOG_BLOCKS.items():
            conf = LogConfig(name, LOG_PERIOD_MS)
            for variable, kind in variables:
                conf.add_variable(variable, kind)
            conf.data_received_cb.add_callback(self._received)
            self._configs.append(conf)

    def _received(self, timestamp, data, logconf):
        self.latest.update(data)
        self._arrived[logconf.name] = self._now()

    def age(self):
        """Seconds since the stalest block arrived; inf until every block has arrived once."""
        if len(self._arrived) < len(self._configs):
            return math.inf
        return self._now() - min(self._arrived.values())

    def wait(self, timeout=3.0):
        deadline = self._now() + timeout
        while self.age() > 0.1:
            if self._now() > deadline:
                raise RuntimeError('no sensor data: check that both decks are attached')
            time.sleep(0.05)

    def __enter__(self):
        for conf in self._configs:
            self._cf.log.add_config(conf)
            conf.start()
        return self

    def __exit__(self, *exc):
        for conf in self._configs:
            conf.delete()


def fly(scf, policy, log, options, record=None, now=time.monotonic, sleep=time.sleep):
    """Take off on the firmware, fly the policy, land on the firmware.

    Returns why the policy phase ended. Landing runs in a `finally`, so it happens whatever goes
    wrong after arming, exceptions included.
    """
    if options.dry_run:
        return _dry_run(policy, log, options, record, now, sleep)
    cf, freq = scf.cf, policy.spec.control_freq
    commander = cf.commander
    commander.send_setpoint(0, 0, 0, 0)  # zero thrust releases the legacy commander's thrust lock
    cf.supervisor.send_arming_request(True)
    sleep(1.0)
    try:
        steps = options.takeoff_seconds * freq
        for i in ticks(options.takeoff_seconds, freq, now, sleep):
            commander.send_hover_setpoint(0, 0, 0, options.height * min(1.0, (i + 1) / steps))

        samples = []
        for _ in ticks(options.calibrate_seconds, freq, now, sleep):
            commander.send_hover_setpoint(0, 0, 0, options.height)
            samples.append(float(log.latest.get('controller.cmd_thrust', 0.0)))
        hover_command = float(np.median(samples[len(samples) // 2:]))  # second half: settled
        if not THRUST_COMMAND_MIN < hover_command < THRUST_COMMAND_MAX:
            return f'implausible hover thrust command {hover_command:.0f}'
        logger.info('Hover thrust command %.0f', hover_command)

        runner = PolicyRunner(policy, options.height)
        reason = None
        try:
            for _ in ticks(options.duration, freq, now, sleep):
                inputs, tilt = read_inputs(log.latest, policy.spec)
                reason = abort_reason(inputs, tilt, log.age(), options.height, options.limits)
                if reason:
                    break
                action = runner.step(**inputs)
                setpoint = to_setpoint(action, policy.spec, hover_command, options.signs,
                                       options.authority)
                commander.send_setpoint(*setpoint)
                if record:
                    record('policy', inputs, tilt, action, setpoint)
        except KeyboardInterrupt:
            reason = 'interrupted'
        return reason or 'time up'
    finally:
        _land(commander, log, options, freq, now, sleep)
        cf.supervisor.send_arming_request(False)


def _land(commander, log, options, freq, now, sleep):
    """Hand control back to the firmware's hover controller, steady up, then descend."""
    z = float(np.clip(log.latest.get('range.zrange', 1000 * options.height) / 1000.0,
                      0.2, options.height))
    for _ in ticks(1.0, freq, now, sleep):
        commander.send_hover_setpoint(0, 0, 0, z)
    steps = 2.0 * freq
    for i in ticks(2.0, freq, now, sleep):
        commander.send_hover_setpoint(0, 0, 0, max(0.05, z * (1 - (i + 1) / steps)))
    commander.send_stop_setpoint()
    commander.send_notify_setpoint_stop()


def _dry_run(policy, log, options, record, now, sleep):
    """Everything but the motors: read the sensors, run the policy, print what it would command."""
    spec, freq = policy.spec, policy.spec.control_freq
    runner = PolicyRunner(policy, options.height)
    for i in ticks(options.duration, freq, now, sleep):
        inputs, tilt = read_inputs(log.latest, spec)
        action = runner.step(**inputs)
        roll, pitch, yaw_rate, thrust = spec.decode(np.clip(action, -1, 1) * options.authority)
        if record:
            record('dry-run', inputs, tilt, action, None)
        if i % max(1, freq // 2) == 0:
            reason = abort_reason(inputs, tilt, log.age(), options.height, options.limits)
            print(f'flow {inputs["flow_rate"][0]:+5.2f} {inputs["flow_rate"][1]:+5.2f} rad/s | '
                  f'z {inputs["zrange"]:4.2f} m | nearest {inputs["ranges"].min():4.2f} m | '
                  f'tilt {tilt:4.1f} | -> roll {math.degrees(roll):+5.1f} '
                  f'pitch {math.degrees(pitch):+5.1f} yaw {math.degrees(yaw_rate):+6.1f}/s '
                  f'thrust {thrust / spec.hover_thrust:4.2f}x hover'
                  + (f' | would abort: {reason}' if reason else ''), flush=True)
    return 'dry run'


def flight_recorder(path):
    """A CSV writer for flight records; returns (record function, close function)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['time', 'phase', 'flow_x', 'flow_y', 'zrange', *RANGERS, 'gyro_x', 'gyro_y',
                     'gyro_z', 'tilt_deg', 'a_roll', 'a_pitch', 'a_yaw', 'a_thrust',
                     'roll_deg', 'pitch_deg', 'yaw_rate_deg', 'thrust_cmd'])
    start = time.monotonic()

    def record(phase, inputs, tilt, action, setpoint):
        writer.writerow([round(time.monotonic() - start, 4), phase, *inputs['flow_rate'],
                         inputs['zrange'], *inputs['ranges'], *inputs['gyro'], tilt, *action,
                         *(setpoint or ('', '', '', ''))])

    return record, f.close


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('artifact', type=Path, help='policy directory, e.g. runs/<name>/policy')
    parser.add_argument('--dry-run', action='store_true',
                        help='read sensors and run the policy with the motors off')
    parser.add_argument('--height', type=float, default=1.0, help='hover height, m')
    parser.add_argument('--duration', type=float, default=10.0, help='seconds of policy control')
    parser.add_argument('--authority', type=float, default=1.0,
                        help='0..1: scale the policy towards plain hover for first flights')
    for axis in DEFAULT_SIGNS:
        parser.add_argument(f'--{axis.replace("_", "-")}-sign', type=float, choices=(-1.0, 1.0),
                            default=DEFAULT_SIGNS[axis])
    parser.add_argument('--yes', action='store_true', help='skip the confirmation before arming')
    args = parser.parse_args(argv)

    try:
        policy = Policy.load(args.artifact)
    except (OSError, ValueError) as exc:
        sys.exit(f'Cannot load {args.artifact}: {exc}')
    low, high = policy.spec.target_height
    if not low <= args.height <= high:
        sys.exit(f'--height {args.height} is outside the {low}-{high} m the policy trained on')
    if not 0.0 <= args.authority <= 1.0:
        sys.exit('--authority must be between 0 and 1')
    signs = {axis: getattr(args, f'{axis}_sign') for axis in DEFAULT_SIGNS}
    options = FlightOptions(height=args.height, duration=args.duration, authority=args.authority,
                            signs=signs, dry_run=args.dry_run)

    import cflib.crtp

    from drones import config
    from drones.crazyflie.link import check_decks, open_link, resolve_uri

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('cflib').setLevel(logging.ERROR)
    cflib.crtp.init_drivers()
    stamp = time.strftime('%Y%m%d-%H%M%S') + ('-dry' if args.dry_run else '')
    log_path = args.artifact.parent / 'flights' / f'{stamp}.csv'

    uri = resolve_uri(config.URI)
    print(f'Connecting to {uri} | sensors: {", ".join(policy.spec.sensors)}')
    with open_link(uri) as scf:
        check_decks(scf)
        with SensorLog(scf) as log:
            log.wait()
            if not args.dry_run and not args.yes:
                answer = input(f'Motors will spin: take off to {args.height} m and hand over to '
                               f'the policy at {args.authority:.0%} authority. Type "fly": ')
                if answer.strip() != 'fly':
                    sys.exit('Not flying.')
            record, close = flight_recorder(log_path)
            try:
                reason = fly(scf, policy, log, options, record)
            finally:
                close()
    print(f'Policy phase ended: {reason}. Flight log: {log_path}')


if __name__ == '__main__':
    main()
