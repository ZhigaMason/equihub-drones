"""Film HoverEnv or SquareEnv through CrazyFlow's MuJoCo renderer.

`TrajectoryRenderer` draws one world offscreen, with the flight path as a trail and a few lines of
numbers in the corner. Two cameras:

- `chase` follows the drone from the side of a bounding box's centre, pulled in when a surface would
  come between them. MuJoCo's default free camera starts outside a hover room, where the wall slabs
  hide everything.
- `top` looks straight down on the whole box, +x to the right and +y up, with a body named 'ceiling'
  hidden if the model has one.

The box the cameras frame is `bounds(state, world)`, a caller-supplied function; by default it is
the hover room from `state.room`. A square task instead frames the reference path (see
drones.rl.render_square), which has no room to read.

Offscreen rendering needs an OpenGL backend picked before mujoco is imported: `MUJOCO_GL=egl` on a
headless Linux node. osmesa does not work with the sim extra's PyOpenGL.
"""
import math

import mujoco
import numpy as np

CAMERAS = ('chase', 'top')
# Percent. MuJoCo also names a 50% size, but it renders exactly like 100%.
FONT_SCALES = (100, 150, 200, 250, 300)
# MuJoCo draws geom groups 0-2 by default, so a geom moved to group 3 disappears from the picture.
HIDDEN_GROUP = 3
# Markers share the scene's 1000 geoms with the drone and the room: up to MAX_TRAIL for the flown
# path and MAX_PATH for a reference path, plus a couple of fixed start/target markers.
MAX_TRAIL = 400
MAX_PATH = 200
TRAIL_RGBA = np.array([1.0, 0.35, 0.1, 1.0])
START_RGBA = np.array([0.2, 0.8, 0.3, 1.0])
PATH_RGBA = np.array([0.2, 0.45, 1.0, 1.0])
MARKER_SCALE = 0.006     # marker radius per metre of camera distance: the same size on screen
CHASE_DISTANCE = 1.2     # m, when nothing is in the way
CHASE_ELEVATION = -25.0  # degrees
CHASE_TURN = 0.15        # fraction of its azimuth error the chase camera turns each frame
CAMERA_MARGIN = 0.05     # m kept between the chase camera and any surface
MIN_DISTANCE = 0.1       # m: closer than this the camera is inside the airframe


def forward_vector(azimuth, elevation):
    """MuJoCo's free-camera viewing direction; the camera sits at lookat - distance * forward."""
    a, e = math.radians(azimuth), math.radians(elevation)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def chase_distance(lookat, forward, lo, hi, distance, margin=CAMERA_MARGIN):
    """How far back from `lookat` along -forward the camera can sit and stay inside the box.

    `lo` and `hi` are the box's low and high corners, three numbers each.
    """
    back = -np.asarray(forward, float)
    lo = np.asarray(lo, float) + margin
    hi = np.asarray(hi, float) - margin
    limit = distance
    for axis in range(3):
        if back[axis] > 1e-9:
            limit = min(limit, (hi[axis] - lookat[axis]) / back[axis])
        elif back[axis] < -1e-9:
            limit = min(limit, (lo[axis] - lookat[axis]) / back[axis])
    return max(limit, MIN_DISTANCE)


def top_distance(lo, hi, aspect, fovy):
    """Height above the box from which a straight-down camera frames it in x and y."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    half_x, half_y = (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2
    half_height = max(half_y, half_x / aspect)   # +y is up in the image, +x across it
    return 1.1 * half_height / math.tan(math.radians(fovy) / 2)


def _room_bounds(state, world):
    """The default `bounds`: HoverEnv's room, centred on the origin, floor at z = 0."""
    room = np.asarray(state.room[world], float)
    return np.array([-room[0], -room[1], 0.0]), np.array([room[0], room[1], room[2]])


class TrajectoryRenderer:
    """Offscreen frames of one world of a HoverEnv or SquareEnv. Use as a context manager, or call
    close()."""

    def __init__(self, env, camera='chase', width=640, height=480, world=0,
                 distance=CHASE_DISTANCE, font_scale=100, bounds=None):
        """`bounds(state, world)` returns the (lo, hi) box the cameras frame; defaults to the hover
        room read from `state.room`."""
        if camera not in CAMERAS:
            raise ValueError(f'camera must be one of {CAMERAS}, not {camera!r}')
        if font_scale not in FONT_SCALES:
            raise ValueError(f'font_scale must be one of {FONT_SCALES}, not {font_scale!r}')
        self.env, self.camera, self.world = env, camera, world
        self.width, self.height, self.distance = width, height, distance
        self.font_scale = font_scale
        self._bounds = bounds or _room_bounds
        self._viewer = None
        self._azimuth = None
        if camera == 'top':
            model = env.sim.mj_model
            # Only the renderer reads mj_model's geom groups; the sensors cast rays in
            # env.mjx_model.
            ceiling = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ceiling')
            if ceiling != -1:
                model.geom_group[model.geom_bodyid == ceiling] = HIDDEN_GROUP

    def reset(self):
        """Forget the chase camera's heading, for a new episode."""
        self._azimuth = None

    def frame(self, state, trail=(), hud=(), path=(), target=None):
        """An RGB image (height, width, 3) of `state`, an EnvState.

        `trail` is a sequence of flown positions to mark, oldest first, drawn as an orange trail
        from a green start marker. `path` is a sequence of reference positions drawn as small blue
        markers, and `target` a single reference position -- typically the current one -- drawn as
        a larger blue marker. `hud` is (label, value) pairs written in the top-left corner.
        """
        sim, w = self.env.sim, self.world
        # Sim.render draws from the Sim object, and the env keeps its state outside it. HoverEnv
        # also carries its own `mjx` (walls moved per episode); SquareEnv's plain scene has none,
        # and Sim.render resyncs sim.mjx_data from sim.data itself (crazyflow's
        # requires_mujoco_sync).
        sim.data = state.sim
        if hasattr(state, 'mjx'):
            sim.mjx_data = state.mjx
        if self._viewer is None:
            sim.render(mode='rgb_array', world=w, width=self.width, height=self.height)
            self._viewer = sim.viewer.viewer
            self._set_font_scale()
        pos = np.asarray(state.sim.states.pos[w, 0], float)
        lo, hi = self._bounds(state, w)
        self._place_camera(pos, lo, hi)

        radius = MARKER_SCALE * self._viewer.cam.distance
        trail = np.asarray(trail, float).reshape(-1, 3)
        for p in trail[::max(1, math.ceil(len(trail) / MAX_TRAIL))]:
            self._viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.full(3, radius),
                                    pos=p, rgba=TRAIL_RGBA)
        if len(trail):
            self._viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=np.full(3, 2.5 * radius), pos=trail[0], rgba=START_RGBA)
        path = np.asarray(path, float).reshape(-1, 3)
        for p in path[::max(1, math.ceil(len(path) / MAX_PATH))]:
            self._viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.full(3, radius),
                                    pos=p, rgba=PATH_RGBA)
        if target is not None:
            self._viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=np.full(3, 2.5 * radius), pos=np.asarray(target, float),
                                    rgba=PATH_RGBA)
        # add_overlay appends to what is already there, and nothing clears it between frames.
        self._viewer._overlays.clear()
        for label, value in hud:
            self._viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, label, value)
        return sim.render(mode='rgb_array', world=w, width=self.width, height=self.height)

    def _set_font_scale(self):
        # gymnasium builds its rendering context with 150% text and takes no option, so rebuild it.
        v = self._viewer
        v.make_context_current()
        v.con.free()
        v.con = mujoco.MjrContext(self.env.sim.mj_model,
                                  getattr(mujoco.mjtFontScale, f'mjFONTSCALE_{self.font_scale}'))
        v._set_mujoco_buffer()

    def _place_camera(self, pos, lo, hi):
        cam = self._viewer.cam
        if self.camera == 'top':
            fovy = self.env.sim.mj_model.vis.global_.fovy
            # x and y at the box's centre, but z at its floor: looking down from mid-height would
            # change the camera's absolute height above the floor for the same top_distance.
            cam.lookat[:] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
            cam.distance = top_distance(lo, hi, self.width / self.height, fovy)
            cam.azimuth, cam.elevation = 90.0, -90.0
            return
        # Look outwards from the box's centre through the drone, so the camera stays in the box
        # and the nearest edge is the backdrop. Near the centre that direction is noise: keep it.
        centre = (lo[:2] + hi[:2]) / 2
        dx, dy = pos[0] - centre[0], pos[1] - centre[1]
        if math.hypot(dx, dy) > 0.15:
            target = math.degrees(math.atan2(dy, dx))
            self._azimuth = target if self._azimuth is None else (
                self._azimuth + CHASE_TURN * wrap_degrees(target - self._azimuth))
        elif self._azimuth is None:
            self._azimuth = 45.0
        cam.lookat[:] = pos
        cam.azimuth, cam.elevation = self._azimuth, CHASE_ELEVATION
        cam.distance = chase_distance(pos, forward_vector(self._azimuth, CHASE_ELEVATION), lo, hi,
                                      self.distance)

    def close(self):
        self.env.sim.close()
        self._viewer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
