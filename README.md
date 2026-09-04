# drones

Control software for a **Crazyflie 2.1 Brushless** with a **Flow deck v2** and a
**Multi-ranger deck**.

Two ways to fly:

| | |
| --- | --- |
| `uv run python -m drone.server` | Web page you open on your phone: joystick, take-off/land, live ranger telemetry, emergency stop |
| `uv run python wall_avoid_flight.py` | Headless script: take off to 1 m, hold, avoid walls, land on a detected ceiling |

Both share the avoidance logic in `drone/avoidance.py` and the settings in
`drone/config.py`, which reads `.env`.

## Setup

```bash
uv sync
cp .env.example .env
```

`CFLIB_URI` defaults to `auto`, which scans and uses whichever single
interface is present — a Crazyradio dongle (`radio://…`) or the drone plugged
straight in over USB (`usb://0`). Set it explicitly if you have more than one
drone or radio in range. If the configured URI is not available, the error
names what the scan *did* find.

`uv sync` pulls in `websockets`. uvicorn has no WebSocket implementation of its
own, and without one the control page loads but every control is dead — the
server logs `Unsupported upgrade request` and the page shows "link lost". The
server refuses to start if it is missing.

`.env` is loaded automatically and is git-ignored.

On Linux the Crazyradio needs udev rules, otherwise nothing can open the
dongle — see the [Bitcraze USB permissions guide](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_permissions/).

## Phone control page

```bash
uv run python -m drone.server
```

It prints the LAN URL to open, e.g. `http://10.105.255.23:8000/`. Your phone
must be on the same network as this machine. Add the page to your home screen
for a full-screen, chrome-free control panel.

The UI is served with `Cache-Control: no-store`, so a restarted server always
means a fresh page — phones cache aggressively, and a stale `app.js` shows
controls that silently do nothing.

The page has:

- **Height slider** (left) — an absolute target height in metres, not a climb
  command: drag it and the drone flies there and holds. The dashed "now" line
  is the measured height, so you can see it tracking. It follows the drone's
  own target during take-off and landing rather than fighting it.
- **Fly / turn joystick** (right) — up and down is forward/back, left and
  right turns on the spot. Springs back to centre. Both are proportional to
  deflection, up to `MAX_MANUAL_SPEED` and `MAX_YAW_RATE`.
- **No sideways control.** Point the nose where you want to go. Lateral motion
  comes only from the avoidance vector.
- **Connect / Take off / Land** and a live status line.
- **Auto wall-avoid** — hands the flight over to the autonomous behaviour;
  joysticks are disabled and greyed out while it runs.
- **Avoidance on/off** — the override toggle. On by default, so the repulsion
  vector is blended underneath your stick input and the drone resists being
  flown into a wall. Turn it off for full manual authority.
- **Emergency stop** — cuts the motors immediately. It is drained ahead of any
  other queued command, so it lands within one 100 ms control cycle.
- **Recover after stop** — clears the supervisor lock left by an emergency stop
  or a tumble, so you can arm again without rebooting the drone.

### Access control

`WEB_TOKEN` in `.env` is empty by default, which means **anyone on your network
can fly the drone**. Set it to any string and the page and WebSocket both
require `?token=<value>`; the printed URL includes it.

## How avoidance works

Each of the four horizontal rangers, when it reads below `AVOID_DISTANCE`,
contributes a push away from that side. The push ramps from 0 at
`AVOID_DISTANCE` to the full `MAX_AVOID_SPEED` by `AVOID_HARD_DISTANCE`, and
stays at full strength closer in. Opposing sensors cancel, so in a narrow
corridor the drone centres itself instead of oscillating. The result is
low-pass filtered each 100 ms cycle so motion stays smooth.

**Where the saturation point sits is what makes this firm.** A ramp that only
reaches full speed at the wall itself is still barely pushing at the distance
where it needs to act. With the defaults the response at 0.4 m is 0.55 m/s:

| Wall distance | Push |
| --- | --- |
| 0.9 m and beyond | 0 (ignored) |
| 0.7 m | 0.22 m/s |
| 0.5 m | 0.44 m/s |
| 0.4 m | 0.55 m/s |
| 0.35 m and closer | 0.60 m/s (full) |

To tune: `AVOID_DISTANCE` sets how early it notices, `AVOID_HARD_DISTANCE` how
early it commits fully, `MAX_AVOID_SPEED` how hard it shoves. Raising
`AVOID_HARD_DISTANCE` towards `AVOID_DISTANCE` makes the response nearly
on/off; lowering it lets the drone approach further before reacting hard.

A closed-loop simulation (real repulsion maths, 0.35 s velocity lag) settles
without oscillation in corridors from 0.6 m to 2.0 m, and shows no oscillation
even at 4× the gain — it is a saturating proportional controller with no
integral term, so there is a wide stability margin. A drone drifting at
1.0 m/s straight at a wall is stopped with 0.7 m to spare from a 0.9 m start,
or 0.56 m if the drone tracks velocity sluggishly.

Under manual control the push is *added* to your stick input, and
`MAX_AVOID_SPEED` (0.6) deliberately exceeds `MAX_MANUAL_SPEED` (0.4) so
avoidance wins: a full-forward stick into a wall 0.3 m ahead nets -0.2 m/s, so
the drone still backs off. Turn **Avoidance off** for full manual authority.
There is no sideways stick, so lateral motion is always the avoidance vector.

The ceiling check needs three consecutive readings under `CEILING_DISTANCE`
before landing, so a single spurious measurement cannot end the flight. It is
active in every mode, including manual.

## Safety behaviour

- **Watchdog.** The page sends control state 20×/s. After `STICK_TIMEOUT`
  (0.7 s) of silence the drone stops turning but *keeps holding its height*,
  since hovering is the safe default; after `LINK_TIMEOUT` (3 s) it lands. A
  locked screen, a backgrounded tab, or dropped Wi-Fi therefore cannot leave
  it turning on a stale input.
- **Altitude envelope.** The slider spans `MIN_ALTITUDE`…`MAX_ALTITUDE` and the
  commanded height is clamped to it. The drone tracks the target with a
  proportional climb rate capped at `MAX_CLIMB_SPEED`, so a large drag is a
  smooth ramp rather than a lurch.
- **Landing is in a `finally`.** Any exception in the control loop still brings
  the drone down; if landing itself fails, an emergency stop is sent.
- **Pre-flight checks.** Refuses to fly if either deck is missing, if the
  Multi-ranger produces no data, or if a ceiling is already detected.

## Configuration

Everything below lives in `.env`; the defaults are in `drone/config.py`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CFLIB_URI` | `auto` | `auto` scans for whatever is plugged in; or set `radio://<radio>/<channel>/<rate>/<address>` or `usb://0` |
| `TAKEOFF_HEIGHT` | `1.0` | Hover height in metres |
| `TAKEOFF_VELOCITY` | `0.3` | Take-off / landing speed, m/s |
| `MIN_ALTITUDE` / `MAX_ALTITUDE` | `0.2` / `2.0` | Altitude envelope in metres |
| `AVOID_DISTANCE` | `0.9` | A wall closer than this (m) starts pushing the drone away |
| `AVOID_HARD_DISTANCE` | `0.35` | At or below this (m) the push is at full strength |
| `MAX_AVOID_SPEED` | `0.6` | Cap on the avoidance speed, m/s |
| `CEILING_DISTANCE` | `0.5` | Something above the drone within this (m) triggers landing |
| `MAX_MANUAL_SPEED` | `0.4` | Forward/back speed at full stick, m/s |
| `MAX_CLIMB_SPEED` | `0.3` | Cap on the climb/descent rate, m/s |
| `MAX_YAW_RATE` | `90.0` | Turn rate at full joystick deflection, deg/s |
| `MAX_FLIGHT_TIME` | `60.0` | Auto mode lands after this many seconds |
| `WEB_HOST` / `WEB_PORT` | `0.0.0.0` / `8000` | Server bind address |
| `WEB_TOKEN` | *(empty)* | If set, required as `?token=…` |
| `STICK_TIMEOUT` / `LINK_TIMEOUT` | `0.7` / `3.0` | Watchdog thresholds, seconds |

## Layout

```
drone/
  config.py       .env-backed settings
  avoidance.py    ranger distances -> body-frame velocity
  controller.py   owns the link; runs the 10 Hz control loop on one thread
  server.py       FastAPI: serves the page, WebSocket for commands + telemetry
  static/         the phone UI
wall_avoid_flight.py   standalone autonomous flight
```

All cflib calls happen on the controller's single flight thread. The web layer
only pushes commands into a queue and reads an immutable telemetry snapshot.

## Before the first flight

- Fly in open space over a textured, non-reflective floor — the Flow deck needs
  surface texture to hold position.
- The Crazyflie is armed automatically on take-off. Keep clear of the props.
- Check the battery reading on the page; below ~3.2 V it turns red and you
  should land.
