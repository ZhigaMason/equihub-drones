"""Fly the simulated drone around a scanned house from the keyboard, optionally with the questions
an embodied question answering benchmark asks about it.

    uv run --extra sim drones-explore-scene                          # every scene in scenes/
    uv run --extra sim drones-explore-scene scenes/Bowlus.glb        # one of them first
    uv run --extra sim drones-explore-scene --benchmark hm-eqa       # its downloaded scenes
    uv run --extra sim drones-explore-scene --benchmark express-bench --question 12
    uv run --extra sim drones-explore-scene --benchmark indoor-uav        # its instructions

    W / S           forward / back        A / D           left / right
    Up / Down       climb / descend       Left / Right    turn (Q / E too)
    Shift           twice as fast         C               chase or first-person camera
    N / P           next / previous scene R               back to the start
    ] / [           next / previous question in this scene
    Space           show or hide the answer               I   hide the question
    PgDn / PgUp     scroll a question longer than the window
    H               hide the key help     Esc             quit

Simulator only: nothing here opens a radio link. The drone is CrazyFlow's cf21B_500 under its own
state controller (position, velocity and yaw setpoints), the same airframe the square task trains
on. The keys move the setpoint, so letting go of everything leaves the drone hovering where it is.

With --benchmark (hm-eqa, mt-hm3d, express-bench, a-eqa, indoor-uav; see drones.sim.eqa) each scene
comes with its questions -- for indoor-uav, IndoorUAV's navigation instructions. A question with a
start pose starts the drone there, facing the way the benchmark's agent faces: the scan is moved
so the floor under that pose is the world origin, which keeps the simulator's own reset untouched,
and the drone takes off to the pose's height (IndoorUAV's) or START_HEIGHT (habitat's, on the
floor). EXPRESS-Bench's reference walk and IndoorUAV's flight are drawn, to their goal.

The scan stays scenery, as `drones.sim.scenes` requires: it is never in the MJX model the drone
flies in. Walls stop the *setpoint* instead -- `Pilot` casts rays against the scan with
`mj_rayMesh` and slides along whatever it would pass through -- so the drone cannot be steered
through a wall, though a fast approach can still overshoot into one a little.
"""
import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drones.sim.scenes import SCENES_DIR

DRONE = 'cf21B_500'
SIM_FREQ = 500            # Hz
CONTROL_FREQ = 50         # Hz; CrazyFlow's state controller wants at least 20
SPEED = 0.8               # m/s horizontal, doubled with Shift
CLIMB = 0.5               # m/s
TURN = math.radians(90)   # rad/s
START_HEIGHT = 1.0        # m, the setpoint the drone takes off to
MIN_HEIGHT = 0.1          # m, lowest setpoint: the drone does not land
RADIUS = 0.15             # m kept between the setpoint and the scan
LEASH = 0.3               # m the setpoint may lead the drone, so it cannot run off ahead of it
CHASE_DISTANCE = 1.0      # m behind the drone
CHASE_ELEVATION = -20.0   # degrees
FPV_AHEAD = 0.08          # m in front of the drone's centre, clear of its own propellers
CAMERA_MARGIN = 0.1       # m kept between the chase camera and the scan
MAX_TICKS = 5             # control ticks run per frame before the sim is allowed to fall behind
PATH_POINTS = 150         # markers drawn for a benchmark's reference path, at most
PATH_LIFT = 0.05          # m above the floor, so the path is not buried in it
PATH_RGBA = (0.2, 0.45, 1.0, 1.0)
START_RGBA = (0.2, 0.8, 0.3, 1.0)
GOAL_RGBA = (1.0, 0.35, 0.1, 1.0)
PANEL_WIDTH = 0.42        # of the window, for the question text
PANEL_PAD = 8             # px around the question panel's text
PANEL_RGBA = (0.1, 0.1, 0.1, 0.75)
FLOOR_PROBE = 0.3         # m above a start pose to look down for the floor from ...
FLOOR_SEARCH = 3.0        # m ... and how far down; habitat starts sit on it, IndoorUAV's ~1.3 m up


@dataclass
class Intent:
    """What the keys ask for, each in -1..1 (fast: 1 or 2)."""
    forward: float = 0.0
    left: float = 0.0
    up: float = 0.0
    turn: float = 0.0     # + turns left, as yaw does
    fast: float = 1.0


class Pilot:
    """Turns key intents into CrazyFlow state commands, keeping the setpoint out of the scan.

    `distance(origin, direction)` is the distance to the scan along a unit ray, or inf. Pure numpy,
    so the flying logic is testable without a window or a simulator.
    """

    def __init__(self, distance, start=(0.0, 0.0, START_HEIGHT), yaw=0.0):
        self.distance = distance
        self.reset(start, yaw)

    def reset(self, start=(0.0, 0.0, START_HEIGHT), yaw=0.0):
        self.setpoint = np.array(start, float)
        self.yaw = float(yaw)
        self.velocity = np.zeros(3)

    def _clear(self, origin, direction, length):
        """Whether a sphere of RADIUS can move `length` along the axis `direction` from `origin`.

        Five parallel rays, from the centre and from four points RADIUS / 2 off it, so a table edge
        a little to one side of the centre still stops it.
        """
        side = np.cross(direction, [0.0, 0.0, 1.0])
        if np.linalg.norm(side) < 1e-6:
            side = np.array([1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        other = np.cross(direction, side)
        offsets = [np.zeros(3)] + [s * RADIUS / 2 for s in (side, -side, other, -other)]
        return all(self.distance(origin + o, direction) > length + RADIUS for o in offsets)

    def update(self, intent, dt, drone_pos):
        """Advance the setpoint by `dt` seconds of `intent`. Returns the (13,) state command."""
        self.yaw = (self.yaw + intent.turn * TURN * dt + math.pi) % (2 * math.pi) - math.pi
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        speed = SPEED * intent.fast
        velocity = np.array([speed * (c * intent.forward - s * intent.left),
                             speed * (s * intent.forward + c * intent.left),
                             CLIMB * intent.fast * intent.up])
        step = velocity * dt
        drone_pos = np.asarray(drone_pos, float)
        # One axis at a time, so a blocked axis stops and the others slide along the wall.
        for axis in range(3):
            if step[axis] == 0.0:
                continue
            # The leash only stops the keys pushing the setpoint further ahead of the drone; it
            # never pulls back a setpoint already there, such as the take-off height.
            ahead = self.setpoint[axis] - drone_pos[axis]
            if abs(ahead + step[axis]) > LEASH and abs(ahead + step[axis]) > abs(ahead):
                velocity[axis] = 0.0
                continue
            direction = np.zeros(3)
            direction[axis] = math.copysign(1.0, step[axis])
            if self._clear(self.setpoint, direction, abs(step[axis])):
                self.setpoint[axis] += step[axis]
            else:
                velocity[axis] = 0.0
        self.setpoint[2] = max(self.setpoint[2], MIN_HEIGHT)
        self.velocity = velocity
        command = np.zeros(13)
        command[0:3], command[3:6], command[9] = self.setpoint, velocity, self.yaw
        return command


def scene_distance(model, data, geoms):
    """A `Pilot.distance` over the scan's mesh geoms, ignoring the drone and everything else."""
    import mujoco

    def distance(origin, direction):
        hits = [mujoco.mj_rayMesh(model, data, g, np.asarray(origin, float),
                                  np.asarray(direction, float)) for g in geoms]
        hits = [h for h in hits if h >= 0.0]
        return min(hits) if hits else math.inf
    return distance


@dataclass
class Stop:
    """One scene to visit, with the questions asked about it (none outside --benchmark)."""
    path: Path
    questions: list


def list_scenes(first):
    """Every .glb in scenes/ and scenes/hm3d/, with `first` (a path, or None) at the front."""
    found = sorted(SCENES_DIR.glob('*.glb')) + sorted((SCENES_DIR / 'hm3d').glob('*.glb'))
    if first is not None:
        first = Path(first)
        found = [first] + [p for p in found if p.resolve() != first.resolve()]
    return found


def ascii_text(text):
    """`text` as MuJoCo's fonts can draw it: they have printable ASCII only."""
    import unicodedata

    for fancy, plain in (('\u2018', "'"), ('\u2019', "'"), ('\u201c', '"'), ('\u201d', '"'),
                         ('\u2013', '-'), ('\u2014', '-'), ('\u2026', '...')):
        text = text.replace(fancy, plain)
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()


def wrap(text, width):
    """`text` broken into lines of at most `width` characters, at spaces."""
    import textwrap

    return '\n'.join(textwrap.fill(line, width) if line else '' for line in text.split('\n'))


class Explorer:
    """The window, the simulator and the scene currently loaded in both."""

    def __init__(self, stops, width, height, visible=True, question=0):
        import glfw
        import jax.numpy as jnp
        from crazyflow.control import Control
        from crazyflow.sim import Sim

        self.glfw, self.jnp = glfw, jnp
        self.stops, self.index = stops, 0
        self.sim = Sim(n_worlds=1, n_drones=1, drone=DRONE, control=Control.state, freq=SIM_FREQ)
        self.substeps = SIM_FREQ // CONTROL_FREQ
        self.fpv, self.help, self.show_answer, self.show_question = False, True, False, True
        self.scroll = 0
        if not glfw.init():
            sys.exit('Could not initialise GLFW: drones-explore-scene needs a desktop session.')
        glfw.window_hint(glfw.VISIBLE, visible)   # hidden: tests render without a window
        self.window = glfw.create_window(width, height, 'drones-explore-scene', None, None)
        if not self.window:
            glfw.terminate()
            sys.exit('Could not open a window.')
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_key_callback(self.window, self._on_key)
        self._pressed = []
        self.load(0, question)

    # -------------------------------------------------------------- scenes
    @property
    def questions(self):
        return self.stops[self.index].questions

    @property
    def question(self):
        return self.questions[self.qi] if self.questions else None

    def load(self, index, question=0):
        import mujoco

        from drones.sim import scenes

        self.index = index % len(self.stops)
        path = self.stops[self.index].path
        print(f'Loading {path.stem} ...', flush=True)
        self.scene = scenes.load(path)
        lower, upper = scenes.attach(self.sim, self.scene)
        self.geoms = lower + upper
        self.model, self.data = self.sim.mj_model, self.sim.mj_data
        self.base = self.model.geom_pos[self.geoms].copy()
        self.body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'drone:0')
        self.mocap = self.model.body_mocapid[self.body]
        self.mjv_scene = mujoco.MjvScene(self.model, maxgeom=2000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.pilot = Pilot(scene_distance(self.model, self.data, self.geoms))
        self.select(question)

    def select(self, qi):
        """Make question `qi` of this scene the current one, and start the drone at its pose."""
        self.qi = qi % len(self.questions) if self.questions else 0
        self.show_answer, self.scroll = False, 0
        q = self.question
        # World = file frame - anchor. Without a start pose the anchor is the open floor load()
        # found, where the scene already sits. With one, it is the floor under the start: habitat
        # benchmarks start on the floor, IndoorUAV's at the drone's flight height above it.
        self.anchor = self.scene.origin
        self.takeoff = START_HEIGHT
        if q is not None and q.start is not None:
            self._place(q.start)
            # A few probes, not one: a ray through a vertex shared by triangles can slip through.
            below = min(self.pilot.distance(np.array([dx, dy, FLOOR_PROBE]),
                                            np.array([0.0, 0.0, -1.0]))
                        for dx, dy in ((0, 0), (0.03, 0.01), (-0.01, 0.03), (-0.03, -0.02)))
            floor = FLOOR_PROBE - below if below < FLOOR_SEARCH else 0.0
            self.anchor = q.start + np.array([0.0, 0.0, floor])
            self.takeoff = max(START_HEIGHT, -floor)
        self._place(self.anchor)
        self.restart()

    def _place(self, anchor):
        """Move the scan so `anchor` (file frame) is the world origin."""
        import mujoco

        self.model.geom_pos[self.geoms] = self.base + (self.scene.origin - anchor)
        mujoco.mj_kinematics(self.model, self.data)   # the scan's pose, for the rays

    def restart(self):
        self.sim.reset()
        q = self.question
        self.pilot.reset(start=(0.0, 0.0, self.takeoff),
                         yaw=q.yaw if q is not None and q.yaw is not None else 0.0)
        self.sim_time, self.wall_start = 0.0, time.perf_counter()

    # -------------------------------------------------------------- input
    def _on_key(self, window, key, scancode, action, mods):
        if action == self.glfw.PRESS:
            self._pressed.append(key)

    def intent(self):
        glfw, w = self.glfw, self.window

        def held(*keys):
            return float(any(glfw.get_key(w, k) == glfw.PRESS for k in keys))

        return Intent(forward=held(glfw.KEY_W) - held(glfw.KEY_S),
                      left=held(glfw.KEY_A) - held(glfw.KEY_D),
                      up=held(glfw.KEY_UP) - held(glfw.KEY_DOWN),
                      turn=held(glfw.KEY_LEFT, glfw.KEY_Q) - held(glfw.KEY_RIGHT, glfw.KEY_E),
                      fast=1.0 + held(glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT))

    def handle_presses(self):
        glfw = self.glfw
        for key in self._pressed:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window, True)
            elif key == glfw.KEY_C:
                self.fpv = not self.fpv
            elif key == glfw.KEY_H:
                self.help = not self.help
            elif key == glfw.KEY_I:
                self.show_question = not self.show_question
            elif key == glfw.KEY_SPACE:
                self.show_answer = not self.show_answer
            elif key in (glfw.KEY_PAGE_DOWN, glfw.KEY_PAGE_UP):
                self.scroll = max(0, self.scroll + (5 if key == glfw.KEY_PAGE_DOWN else -5))
            elif key == glfw.KEY_R:
                self.restart()
            elif key in (glfw.KEY_RIGHT_BRACKET, glfw.KEY_LEFT_BRACKET) and self.questions:
                self.select(self.qi + (1 if key == glfw.KEY_RIGHT_BRACKET else -1))
            elif key in (glfw.KEY_N, glfw.KEY_P) and len(self.stops) > 1:
                self.load(self.index + (1 if key == glfw.KEY_N else -1))
        self._pressed.clear()

    # -------------------------------------------------------------- simulation
    def drone(self):
        """(position, velocity, quaternion xyzw) of the simulated drone."""
        states = self.sim.data.states
        return (np.asarray(states.pos[0, 0]), np.asarray(states.vel[0, 0]),
                np.asarray(states.quat[0, 0]))

    def advance(self):
        """Run the control ticks that wall-clock time has caught up with."""
        dt = 1.0 / CONTROL_FREQ
        behind = time.perf_counter() - self.wall_start - self.sim_time
        ticks = min(int(behind / dt), MAX_TICKS)
        if ticks == MAX_TICKS:   # a slow frame (a scene load, the first jit): skip, do not rush
            self.wall_start = time.perf_counter() - self.sim_time
        intent = self.intent()
        for _ in range(ticks):
            pos, _, _ = self.drone()
            command = self.pilot.update(intent, dt, pos)
            self.sim.state_control(self.jnp.asarray(command[None, None]))
            self.sim.step(self.substeps)
            self.sim_time += dt

    # -------------------------------------------------------------- drawing
    def _place_camera(self, pos):
        cam, yaw = self.camera, self.pilot.yaw
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        if self.fpv:
            cam.lookat[:] = pos + forward * (FPV_AHEAD + 1.0)
            cam.distance, cam.azimuth, cam.elevation = 1.0, math.degrees(yaw), 0.0
            return
        e = math.radians(CHASE_ELEVATION)
        view = np.array([math.cos(e) * forward[0], math.cos(e) * forward[1], math.sin(e)])
        # Pull the camera in when the scan is between it and the drone.
        hit = self.pilot.distance(pos, -view)
        cam.lookat[:] = pos
        cam.distance = max(0.2, min(CHASE_DISTANCE, hit - CAMERA_MARGIN))
        cam.azimuth, cam.elevation = math.degrees(yaw), CHASE_ELEVATION

    def _add_marker(self, pos, radius, rgba):
        import mujoco

        scene = self.mjv_scene
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.full(3, radius), np.asarray(pos, float), np.eye(3).ravel(),
                            np.asarray(rgba, np.float32))
        scene.ngeom += 1

    def _draw_path(self):
        """The question's reference walk and goal, in world coordinates (file frame - anchor)."""
        q = self.question
        if q is None or q.path is None:
            return
        lift = np.array([0.0, 0.0, PATH_LIFT])
        path = q.path - self.anchor + lift
        for p in path[::max(1, math.ceil(len(path) / PATH_POINTS))]:
            self._add_marker(p, 0.03, PATH_RGBA)
        self._add_marker(path[0], 0.07, START_RGBA)
        if q.goal is not None:
            self._add_marker(q.goal - self.anchor + lift, 0.09, GOAL_RGBA)

    def _question_text(self, width):
        q = self.question
        chars = max(20, int(width * PANEL_WIDTH / max(1, self.context.charWidth[ord('n')])))
        lines = [f'{q.benchmark}  #{q.number}  ({self.qi + 1}/{len(self.questions)} here)',
                 f'{q.category}', '', wrap(ascii_text(q.text), chars)]
        if q.choices:
            lines += ['', *[wrap(c, chars) for c in q.choices]]
        lines += ['', wrap(ascii_text(f'{q.reveal.capitalize()}: {q.answer}'), chars)
                  if self.show_answer
                  else f'Space: show the {q.reveal}']
        if q.start is None:
            lines += ['', wrap('No start pose: starting on the open floor.', chars)]
        return '\n'.join(lines)

    def _draw_panel(self, viewport, text):
        """`text` in a box at the top right. Not mjr_overlay: that stops at 500 characters
        (mjMAXOVERLAY), and IndoorUAV's detailed instructions run past a thousand."""
        import mujoco

        con = self.context
        lines = text.split('\n')
        step = con.charHeight + 2
        fits = max(1, (viewport.height - 2 * PANEL_PAD) // step)
        self.scroll = min(self.scroll, max(0, len(lines) - fits))
        shown = lines[self.scroll:self.scroll + fits]
        if self.scroll + fits < len(lines):
            shown[-1] = f'... PgDn for more ({len(lines) - self.scroll - fits} lines)'
        if self.scroll:
            shown[0] = '... PgUp for the start'
        width = max(sum(con.charWidth[ord(c)] for c in line) for line in shown)
        box = mujoco.MjrRect(viewport.width - width - 2 * PANEL_PAD - 4,
                             viewport.height - len(shown) * step - 2 * PANEL_PAD - 4,
                             width + 2 * PANEL_PAD, len(shown) * step + 2 * PANEL_PAD)
        # mjr_text's (x, y) are relative to the viewport of the last mjr_ call -- this rectangle.
        mujoco.mjr_rectangle(box, *PANEL_RGBA)
        for i, line in enumerate(shown):
            y = box.height - PANEL_PAD - (i + 1) * step + 3
            mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL, line, con, PANEL_PAD / box.width,
                            y / box.height, 1.0, 1.0, 1.0)

    def draw(self):
        import mujoco

        pos, vel, quat = self.drone()
        self.data.mocap_pos[self.mocap] = pos
        self.data.mocap_quat[self.mocap] = [quat[3], quat[0], quat[1], quat[2]]   # wxyz
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self._place_camera(pos)
        width, height = self.glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.camera,
                               mujoco.mjtCatBit.mjCAT_ALL, self.mjv_scene)
        self._draw_path()
        mujoco.mjr_render(viewport, self.mjv_scene, self.context)
        status = (f'{self.scene.name} ({self.index + 1}/{len(self.stops)})\n{pos[2]:.2f} m\n'
                  f'{np.linalg.norm(vel):.2f} m/s\n{"first-person" if self.fpv else "chase"}')
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, 'scene\nheight\nspeed\ncamera', status, self.context)
        if self.question is not None and self.show_question:
            self._draw_panel(viewport, self._question_text(width))
        if self.help:
            keys, what = ['WASD', 'Up / Down', 'Left / Right, Q / E', 'Shift', 'C', 'N / P',
                          'R', 'H', 'Esc'], ['move', 'climb / descend', 'turn', 'fast',
                                             'chase / first-person', 'next / previous scene',
                                             'restart', 'hide this', 'quit']
            if self.questions:
                keys[6:6] = ['] / [', 'Space', 'PgDn / PgUp', 'I']
                what[6:6] = ['next / previous question', f'show the {self.question.reveal}',
                             'scroll the question', 'hide the question']
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                               viewport, '\n'.join(keys), '\n'.join(what), self.context)
        self.glfw.swap_buffers(self.window)

    def run(self):
        while not self.glfw.window_should_close(self.window):
            self.glfw.poll_events()
            self.handle_presses()
            self.advance()
            self.draw()
        self.glfw.terminate()


def benchmark_stops(benchmark, scene=None, number=None, dest=SCENES_DIR):
    """(stops, question index) for --benchmark: every downloaded scene with its questions, the
    chosen scene or question's first. A chosen scene not yet downloaded is fetched."""
    from drones.sim import eqa, scenes

    first = None
    if number is not None:
        first = eqa.locate(benchmark, number, dest)
        if first is None:
            sys.exit(f'{benchmark} has no question {number}')
    elif scene is not None:
        first = scene if not scenes.is_hm3d(scene) or '-' in scene else \
            scenes.hm3d_index(dest).get(scene, scene)
    if first is not None:
        scenes.download([first], dest)
    # IndoorUAV's prompts come per scene: fetch them for every scene already on disk.
    present = [s for s in eqa.busiest(benchmark, None, dest) if scenes.scene_path(s, dest).exists()]
    eqa.prepare(benchmark, present, dest)
    groups = eqa.by_scene(eqa.load(benchmark, dest))
    if first is not None and first not in groups:
        sys.exit(f'{benchmark} asks nothing about scene {scene}')
    qi = 0
    if number is not None:
        qi = [q.number for q in groups[first]].index(number)
    order = ([first] if first else []) + [s for s in groups if s != first]
    stops = [Stop(scenes.scene_path(s, dest), groups[s]) for s in order
             if scenes.scene_path(s, dest).exists()]
    if not stops:
        sys.exit(f'None of {benchmark}\'s scenes is downloaded yet:\n'
                 f'  uv run --extra sim drones-download-scenes --benchmark {benchmark} --count 3')
    return stops, qi


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('scene', nargs='?',
                        help='a .glb to start in (default: the first in scenes/); with '
                             '--benchmark, an HM3D scene id. N and P cycle through the rest')
    parser.add_argument('--benchmark',
                        choices=('hm-eqa', 'mt-hm3d', 'express-bench', 'a-eqa', 'indoor-uav'),
                        help="show this benchmark's questions (IndoorUAV: its instructions), in "
                             'the scenes it asks them about')
    parser.add_argument('--question', type=int,
                        help="with --benchmark, start at this question (its number in the "
                             "benchmark's own order, from 1), downloading its scene if needed")
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args(argv)
    if args.question is not None and args.benchmark is None:
        parser.error('--question needs --benchmark')

    try:
        import drones.sim  # noqa: F401  CrazyFlow first, before anything imports scipy.
        import glfw  # noqa: F401
    except ImportError:
        sys.exit('drones-explore-scene needs the sim extra:  uv sync --extra sim')
    question = 0
    if args.benchmark is not None:
        stops, question = benchmark_stops(args.benchmark, args.scene, args.question)
    else:
        if args.scene is not None and not Path(args.scene).is_file():
            parser.error(f'no scene at {args.scene}')
        stops = [Stop(p, []) for p in list_scenes(args.scene)]
        if not stops:
            parser.error(f'no scenes in {SCENES_DIR}/; fetch some with drones-download-scenes')
    Explorer(stops, args.width, args.height, question=question).run()


if __name__ == '__main__':
    main()
