"""Fly an exported square policy on the real Crazyflie, and log flights for system identification.

    uv run drones-fly-square runs/<name>/policy --dry-run        # motors off: estimate, policy live
    uv run drones-fly-square runs/<name>/policy --firmware       # firmware flies the square; log it
    uv run drones-fly-square runs/<name>/policy --side 0.5 --authority 0.3   # first policy flights
    uv run drones-fly-square runs/<name>/policy

Take-off, hover calibration and landing are drones-fly-policy's: the firmware climbs and hovers on
its Flow-deck controller, and the thrust command that holds this drone up is measured. Then the
square starts where the drone hovers: the first edge straight ahead, turning left (--clockwise
turns right). The policy flies it through the attitude commander, from the firmware's state
estimate. With --firmware, the firmware's own position controller flies the same reference
instead, so system-ID data can be logged before any policy is trusted with the motors.

Every flight is logged to runs/<name>/flights/<stamp>-square*.csv, the format drones-finetune-square
reads. Rows hold the action the drone acted on: the policy's scaled by --authority, or the firmware
controller's command converted to the policy's units.

Needs only numpy and cflib.
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

from drones.missions.fly_policy import (DEFAULT_SIGNS, RANGERS, THRUST_COMMAND_MAX,
                                        THRUST_COMMAND_MIN, SensorLog, gravity_and_tilt, land,
                                        take_off, ticks, to_setpoint)
from drones.policy.runtime import Policy, SquareRunner
from drones.policy.square import heading

logger = logging.getLogger(__name__)

LOG_FORMAT = '# format: square-log v1'
COLUMNS = ('time', 'phase', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'qx', 'qy', 'qz', 'qw',
           'gyro_x', 'gyro_y', 'gyro_z', 'ref_x', 'ref_y', 'ref_z', 'ref_vx', 'ref_vy', 'ref_vz',
           'a_roll', 'a_pitch', 'a_yaw', 'a_thrust', 'roll_deg', 'pitch_deg', 'yaw_rate_deg',
           'thrust_cmd', 'hover_command')

# Log blocks, each within cflib's 26-byte packet.
LOG_BLOCKS = {
    'square_pos_vel': [(f'stateEstimate.{a}', 'float') for a in ('x', 'y', 'z', 'vx', 'vy', 'vz')],
    'square_attitude': [(f'stateEstimate.q{a}', 'float') for a in 'xyzw'],
    'square_gyro_thrust': [('gyro.x', 'float'), ('gyro.y', 'float'), ('gyro.z', 'float'),
                           ('controller.cmd_thrust', 'float')],
    # The firmware controller's attitude and yaw-rate command, logged for --firmware flights.
    'square_controller': [('controller.roll', 'float'), ('controller.pitch', 'float'),
                          ('controller.yawRate', 'float')],
    'square_ranges': [('range.zrange', 'uint16_t')] + [(f'range.{r}', 'uint16_t')
                                                       for r in RANGERS],
}


@dataclass(frozen=True)
class SquareLimits:
    max_tilt_deg: float = 30.0
    min_range: float = 0.2               # any Multi-ranger reading closer than this aborts
    min_height: float = 0.15
    max_height_above_target: float = 0.5
    max_error: float = 0.5               # m from the reference
    max_log_age: float = 0.25            # seconds without fresh sensor data aborts


@dataclass(frozen=True)
class SquareOptions:
    height: float = 1.0
    side: float | None = None            # default: the side the policy trained on
    lap_time: float = 8.0
    laps: float = 2.0
    clockwise: bool = False
    authority: float = 1.0
    firmware: bool = False
    dry_run: bool = False
    signs: dict = field(default_factory=lambda: dict(DEFAULT_SIGNS))
    limits: SquareLimits = field(default_factory=SquareLimits)
    takeoff_seconds: float = 2.0
    calibrate_seconds: float = 3.0

    @property
    def duration(self):
        return self.laps * self.lap_time


# ---------------------------------------------------------------------- pure conversions
def read_state(latest):
    """Latest cflib log values -> (state estimate in the simulator's frame and units, tilt in deg).

    The firmware's world frame is the simulator's: +x forward at take-off, +y left, +z up. The gyro
    is in the body frame, as CrazyFlow keeps angular velocity.
    """
    quat = np.array([latest[f'stateEstimate.q{a}'] for a in 'xyzw'], float)
    gravity, tilt = gravity_and_tilt(quat)
    return {
        'pos': np.array([latest[f'stateEstimate.{a}'] for a in 'xyz'], float),
        'vel': np.array([latest[f'stateEstimate.v{a}'] for a in 'xyz'], float),
        'quat': quat,
        'yaw': float(heading(np, quat)),
        'gravity': gravity,
        'gyro': np.radians([latest['gyro.x'], latest['gyro.y'], latest['gyro.z']]),
        'zrange': latest['range.zrange'] / 1000.0,
        'ranges': np.array([latest[f'range.{r}'] / 1000.0 for r in RANGERS]),
    }, tilt


def firmware_action(latest, spec, hover_command, signs=DEFAULT_SIGNS):
    """The firmware controller's command as the policy's normalised action: to_setpoint inverted.

    This assumes the logged controller.roll/pitch follow the setpoint conventions (attitudeDesired,
    read the same way to_setpoint writes them). Check it on the first --firmware flight: the logged
    a_pitch should be positive while the drone accelerates forward.

    controller.yawRate is not a commanded rate: controller_pid.c logs rateDesired.yaw, the *output*
    of the yaw attitude PID, so it cannot be inverted back into a setpoint. The yaw action is logged
    as 0 instead: the firmware holds a constant heading through --firmware flights, and sysid
    recovers yaw_cmd from the segment-start heading, which models that exactly.
    """
    roll = math.radians(signs['roll'] * latest['controller.roll'])
    pitch = math.radians(signs['pitch'] * latest['controller.pitch'])
    thrust = spec.hover_thrust * latest['controller.cmd_thrust'] / hover_command
    if thrust >= spec.hover_thrust:
        a_thrust = (thrust - spec.hover_thrust) / (spec.thrust_max - spec.hover_thrust)
    else:
        a_thrust = (thrust - spec.hover_thrust) / (spec.hover_thrust - spec.thrust_min)
    return np.clip([roll / spec.max_tilt, pitch / spec.max_tilt, 0.0, a_thrust], -1.0, 1.0)


def square_abort_reason(state, tilt, log_age, ref_pos, limits):
    """Why the square must be handed back to the firmware right now, or None."""
    if log_age > limits.max_log_age:
        return f'sensor data {log_age:.2f} s old'
    if tilt > limits.max_tilt_deg:
        return f'tilt {tilt:.0f} deg'
    nearest = float(np.min(state['ranges']))
    if nearest < limits.min_range:
        return f'obstacle {nearest:.2f} m away'
    height = state['zrange']
    if height < limits.min_height or height > ref_pos[2] + limits.max_height_above_target:
        return f'height {height:.2f} m'
    error = float(np.linalg.norm(state['pos'] - ref_pos))
    if error > limits.max_error:
        return f'{error:.2f} m off the square'
    return None


def square_recorder(path):
    """A CSV writer for square flights; returns (record function, close function)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'w', newline='')
    f.write(LOG_FORMAT + '\n')
    writer = csv.writer(f)
    writer.writerow(COLUMNS)

    def record(time, phase, state, ref_pos, ref_vel, action, setpoint, hover_command):
        writer.writerow([round(time, 4), phase, *state['pos'], *state['vel'], *state['quat'],
                         *state['gyro'], *ref_pos, *ref_vel, *action,
                         *(setpoint or ('', '', '', '')), hover_command])

    return record, f.close


def start_square(policy, latest, options):
    """A runner whose square starts where the drone hovers now, first edge straight ahead."""
    state, _ = read_state(latest)
    origin = np.array([state['pos'][0], state['pos'][1], options.height])
    return SquareRunner(policy, origin=origin, ref_yaw=state['yaw'], lap_time=options.lap_time,
                        direction=-1.0 if options.clockwise else 1.0, rotation=state['yaw'],
                        side=options.side)


# ---------------------------------------------------------------------- the drone
def fly_square(scf, policy, log, options, record=None, now=time.monotonic, sleep=time.sleep):
    """Take off on the firmware, fly the square, land on the firmware.

    Returns why the square ended. Landing runs in a `finally`, so it happens whatever goes wrong
    after arming, exceptions included.
    """
    if options.dry_run:
        return _dry_run(policy, log, options, record, now, sleep)
    spec, freq = policy.spec, policy.spec.control_freq
    cf = scf.cf
    commander = cf.commander
    commander.send_setpoint(0, 0, 0, 0)  # zero thrust releases the legacy commander's thrust lock
    cf.supervisor.send_arming_request(True)
    sleep(1.0)
    try:
        hover_command = take_off(commander, log, options, freq, now, sleep)
        if not THRUST_COMMAND_MIN < hover_command < THRUST_COMMAND_MAX:
            return f'implausible hover thrust command {hover_command:.0f}'
        logger.info('Hover thrust command %.0f', hover_command)

        runner = start_square(policy, log.latest, options)
        phase = 'firmware' if options.firmware else 'policy'
        start, reason = now(), None
        try:
            for i in ticks(options.duration, freq, now, sleep):
                t = i / freq
                state, tilt = read_state(log.latest)
                ref_pos, ref_vel = runner.reference(t)
                reason = square_abort_reason(state, tilt, log.age(), ref_pos, options.limits)
                if reason:
                    break
                if options.firmware:
                    commander.send_position_setpoint(*ref_pos, math.degrees(runner.ref_yaw))
                    applied = firmware_action(log.latest, spec, hover_command, options.signs)
                    setpoint = None
                else:
                    action = runner.step(t, pos=state['pos'], vel=state['vel'], yaw=state['yaw'],
                                         gravity=state['gravity'])
                    applied = np.clip(action, -1.0, 1.0) * options.authority
                    setpoint = to_setpoint(action, spec, hover_command, options.signs,
                                           options.authority)
                    commander.send_setpoint(*setpoint)
                if record:
                    record(now() - start, phase, state, ref_pos, ref_vel, applied, setpoint,
                           hover_command)
        except KeyboardInterrupt:
            reason = 'interrupted'
        return reason or 'square done'
    finally:
        land(commander, log, options, freq, now, sleep)
        cf.supervisor.send_arming_request(False)


def _dry_run(policy, log, options, record, now, sleep):
    """Everything but the motors: read the estimate, run the policy, print what it would command."""
    spec, freq = policy.spec, policy.spec.control_freq
    runner = start_square(policy, log.latest, options)
    start = now()
    for i in ticks(options.duration, freq, now, sleep):
        t = i / freq
        state, tilt = read_state(log.latest)
        ref_pos, ref_vel = runner.reference(t)
        action = runner.step(t, pos=state['pos'], vel=state['vel'], yaw=state['yaw'],
                             gravity=state['gravity'])
        applied = np.clip(action, -1.0, 1.0) * options.authority
        if record:
            record(now() - start, 'dry-run', state, ref_pos, ref_vel, applied, None, float('nan'))
        if i % max(1, freq // 2) == 0:
            reason = square_abort_reason(state, tilt, log.age(), ref_pos, options.limits)
            roll, pitch, yaw_rate, thrust = spec.decode(applied)
            pos = state['pos']
            print(f'pos {pos[0]:+5.2f} {pos[1]:+5.2f} {pos[2]:4.2f} m | '
                  f'ref {ref_pos[0]:+5.2f} {ref_pos[1]:+5.2f} | tilt {tilt:4.1f} | '
                  f'-> roll {math.degrees(roll):+5.1f} pitch {math.degrees(pitch):+5.1f} '
                  f'yaw {math.degrees(yaw_rate):+6.1f}/s '
                  f'thrust {thrust / spec.hover_thrust:4.2f}x hover'
                  + (f' | would abort: {reason}' if reason else ''), flush=True)
    return 'dry run'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('artifact', type=Path, help='policy directory, e.g. runs/<name>/policy')
    parser.add_argument('--dry-run', action='store_true',
                        help='read the estimate and run the policy with the motors off')
    parser.add_argument('--firmware', action='store_true',
                        help='the firmware position controller flies the square; log it')
    parser.add_argument('--height', type=float, default=1.0, help='flight height, m')
    parser.add_argument('--side', type=float, help='side of the square, m (default: as trained)')
    parser.add_argument('--lap-time', type=float, default=8.0, help='seconds per lap')
    parser.add_argument('--laps', type=float, default=2.0)
    parser.add_argument('--clockwise', action='store_true')
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
    spec = policy.spec
    if spec.task != 'square':
        sys.exit(f'{args.artifact} is a {spec.task} policy; drones-fly-square flies square ones')
    for name, value, (low, high) in (('--height', args.height, spec.height),
                                     ('--lap-time', args.lap_time, spec.lap_time)):
        if not low <= value <= high:
            sys.exit(f'{name} {value} is outside the {low}-{high} the policy trained on')
    if args.side is not None and not 2 * spec.corner_radius < args.side <= spec.side:
        sys.exit(f'--side must be over {2 * spec.corner_radius} m and at most {spec.side} m')
    if not 0.0 <= args.authority <= 1.0:
        sys.exit('--authority must be between 0 and 1')
    signs = {axis: getattr(args, f'{axis}_sign') for axis in DEFAULT_SIGNS}
    options = SquareOptions(height=args.height, side=args.side, lap_time=args.lap_time,
                            laps=args.laps, clockwise=args.clockwise, authority=args.authority,
                            firmware=args.firmware, dry_run=args.dry_run, signs=signs)

    import cflib.crtp

    from drones import config
    from drones.crazyflie.link import check_decks, open_link, resolve_uri

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('cflib').setLevel(logging.ERROR)
    cflib.crtp.init_drivers()
    stamp = (time.strftime('%Y%m%d-%H%M%S') + '-square' + ('-firmware' if args.firmware else '')
             + ('-dry' if args.dry_run else ''))
    log_path = args.artifact.parent / 'flights' / f'{stamp}.csv'

    uri = resolve_uri(config.URI)
    who = 'the firmware position controller' if args.firmware else \
        f'the policy at {args.authority:.0%} authority'
    print(f'Connecting to {uri} | {options.laps:g} laps of {options.lap_time:g} s')
    with open_link(uri) as scf:
        check_decks(scf)
        with SensorLog(scf, blocks=LOG_BLOCKS) as log:
            log.wait()
            if not args.dry_run and not args.yes:
                answer = input(f'Motors will spin: take off to {args.height} m and fly the square '
                               f'with {who}. Type "fly": ')
                if answer.strip() != 'fly':
                    sys.exit('Not flying.')
            record, close = square_recorder(log_path)
            try:
                reason = fly_square(scf, policy, log, options, record)
            finally:
                close()
    print(f'Square ended: {reason}. Flight log: {log_path}')


if __name__ == '__main__':
    main()
