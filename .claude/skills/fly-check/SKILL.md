---
name: fly-check
description: Use before a trained policy is flown on the real Crazyflie - the pre-flight checklist, sign verification, authority ramp and abort limits. Read this when the user asks about flying, sim-to-real, or what to check before the first flight.
---

# Before a policy touches the real drone

**The agent never runs these commands.** `drones-fly-policy`, `drones-fly-square`,
`drones-wall-avoid` and `drones-web` open a radio link and spin motors next to a person; `--dry-run`
still connects. A hook blocks them. Your job is to get the operator ready, read the logs afterwards,
and say plainly what is not yet verified.

## The one thing that is not verified

`DEFAULT_SIGNS` in `missions/fly_policy.py` inverts pitch and yaw rate:

```python
{'roll': 1.0, 'pitch': -1.0, 'yaw_rate': -1.0}
```

These come from **reading the Crazyflie firmware source, not from flying**:

| | Simulator | On the drone |
| --- | --- | --- |
| Roll setpoint | +roll moves right | sent as is |
| Pitch setpoint | +pitch moves forward | negated — legacy CF2 pitch is inverted |
| Yaw-rate setpoint | +yaw rate turns left | negated — "legacy rate input is inverted" |
| Optical flow | px per 10 ms frame, +x forward | axis swap in `flowdeck_v1v2.c` |

A wrong sign means the policy pushes the drone the way it is already falling. `--roll-sign`,
`--pitch-sign` and `--yaw-rate-sign` flip them. **Verify every one on a tethered drone before a free
flight.** Say so whenever a first flight comes up; do not let it pass silently because the README
mentions it.

## The checklist

1. **Environment.** Open space, textured non-reflective floor (the Flow deck needs surface texture),
   nothing overhead within `CEILING_DISTANCE`. Battery above ~3.2 V.
2. **Link.** `CFLIB_URI` is `auto` by default and scans for a single interface. On Linux the
   Crazyradio needs udev rules or nothing can open the dongle.
3. **Dry run, motors off.** The operator runs it, moving the drone by hand over the floor:
   ```bash
   uv run drones-fly-policy runs/<name>/policy --dry-run
   ```
   It streams the sensors, runs the policy and prints what it *would* command and when it *would*
   abort. This is where a wrong sensor mapping shows up harmlessly.
4. **Tethered, low authority.** `--authority` scales the whole action towards pure hover: 0 is
   hover, 1 the full policy.
   ```bash
   uv run drones-fly-policy runs/<name>/policy --authority 0.3 --duration 10
   uv run drones-fly-square  runs/<name>/policy --side 0.5 --authority 0.3
   ```
5. **Ramp** authority only after each step flies clean.

## What the flight does

1. One zero-thrust setpoint on the ground releases the legacy commander's thrust lock. Arm, take off
   on the firmware's own Flow-deck hover controller.
2. Hover 3 s reading `controller.cmd_thrust` — the command that holds *this* drone up. The policy's
   thrust maps through its ratio to the simulator's calibrated hover thrust, anchored on that value.
3. Attitude control passes to the policy at 50 Hz via `commander.send_setpoint`.
4. Control returns to the firmware, and the drone lands, on time-up, Ctrl-C, or an abort. Landing is
   in a `finally`, so an exception lands too; if landing itself fails, an emergency stop is sent.

## Abort limits

`drones-fly-policy` (`Limits`):

| Limit | Value |
| --- | --- |
| tilt | 30° |
| nearest Multi-ranger reading | 0.2 m |
| height | below 0.15 m, or more than 1.0 m above target |
| sensor data age | 0.25 s |

`drones-fly-square` uses the same, tightened: **0.5 m** above the reference height, plus **0.5 m off
the reference square**.

## Square flights, and the data they produce

```bash
uv run drones-fly-square runs/<name>/policy --side 0.5 --authority 0.3   # first flights
uv run drones-fly-square runs/<name>/policy --lap-time 8 --laps 2 --clockwise
uv run drones-fly-square runs/<name>/policy --firmware --laps 3          # firmware flies it
```

The square starts where the drone hovers, first edge straight ahead.

`--firmware` hands the same square to the firmware's own position controller and logs the roll,
pitch and thrust it commands, in the policy's action units. This collects system-ID data **before
any policy has flown** — the safest way to get real data. On the first `--firmware` flight, check
that `a_pitch` is positive while the drone accelerates forward; that sign also comes from reading
the source. `a_yaw` logs as 0 by design: the firmware holds a constant heading rather than
commanding a yaw rate.

Every flight, dry or not, is logged to `runs/<name>/flights/<timestamp>-square*.csv`. Reading and
analysing those logs is agent work — only the flying is not.

## Feeding real flights back into the simulator

```bash
uv run --extra sim drones-finetune-square runs/<name> --flights runs/<name>/flights/*-square*.csv
```

Replays the logged actions from the logged states and fits a thrust gain, the action latency, and a
residual wrench network, then reports held-out error for the uncorrected simulator, gain-and-latency
alone, and the full correction in `runs/<name>-ft/sysid.json`. **If the correction does not beat the
uncorrected simulator on held-out flights, it stops there** — that is the check working, not a
failure to route around. Otherwise it continues SHAC in the corrected simulator.

The logged states are the firmware's Kalman estimate, not ground truth, so the fit matches the
simulator to what the drone believed, estimator drift included. Say so when reporting results.

Note: the numbers currently in `runs/ft-check-ft/sysid.json` come from **synthetic** flights
(residuals ~1e-10). They are a self-consistency check, not evidence about hardware.

## Known sim-to-real gaps

The square task randomises thrust gain (±10%) and action latency (0–2 control steps). The hover task
randomises neither. Neither task models sensor latency or quantised flow, and the thrust mapping
assumes lift is proportional to the thrust command near hover.
