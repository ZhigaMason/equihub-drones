"""Film the hover task through CrazyFlow's MuJoCo renderer.

`TrajectoryRenderer` draws one world of a `HoverEnv` offscreen, with the flight path as a trail and
a few lines of numbers in the corner. Two cameras:

- `chase` follows the drone from the room-centre side, pulled in when a surface would come between
  them. MuJoCo's default free camera starts outside the room, where the wall slabs hide everything.
- `top` looks straight down on the whole room, +x to the right and +y up, with the ceiling hidden.

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
# Trail markers share the scene's 1000 geoms with the drone and the room.
MAX_TRAIL = 400
TRAIL_RGBA = np.array([1.0, 0.35, 0.1, 1.0])
START_RGBA = np.array([0.2, 0.8, 0.3, 1.0])
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


def chase_distance(lookat, forward, room, distance, margin=CAMERA_MARGIN):
    """How far back from `lookat` along -forward the camera can sit and stay inside the room.

    `room` is (half-width x, half-width y, ceiling height), centred on the origin, floor at z = 0.
    """
    back = -np.asarray(forward, float)
    lo = np.array([-room[0] + margin, -room[1] + margin, margin])
    hi = np.array([room[0] - margin, room[1] - margin, room[2] - margin])
    limit = distance
    for axis in range(3):
        if back[axis] > 1e-9:
            limit = min(limit, (hi[axis] - lookat[axis]) / back[axis])
        elif back[axis] < -1e-9:
            limit = min(limit, (lo[axis] - lookat[axis]) / back[axis])
    return max(limit, MIN_DISTANCE)


def top_distance(room, aspect, fovy):
    """Height above the floor from which a straight-down camera frames the whole room."""
    half_height = max(room[1], room[0] / aspect)   # +y is up in the image, +x across it
    return 1.1 * half_height / math.tan(math.radians(fovy) / 2)


class TrajectoryRenderer:
    """Offscreen frames of one world of a HoverEnv. Use as a context manager, or call close()."""

    def __init__(self, env, camera='chase', width=640, height=480, world=0,
                 distance=CHASE_DISTANCE, font_scale=100):
        if camera not in CAMERAS:
            raise ValueError(f'camera must be one of {CAMERAS}, not {camera!r}')
        if font_scale not in FONT_SCALES:
            raise ValueError(f'font_scale must be one of {FONT_SCALES}, not {font_scale!r}')
        self.env, self.camera, self.world = env, camera, world
        self.width, self.height, self.distance = width, height, distance
        self.font_scale = font_scale
        self._viewer = None
        self._azimuth = None
        if camera == 'top':
            model = env.sim.mj_model
            # Only the renderer reads mj_model's geom groups; the sensors cast rays in env.mjx_model.
            model.geom_group[model.geom_bodyid == model.body('ceiling').id] = HIDDEN_GROUP

    def reset(self):
        """Forget the chase camera's heading, for a new episode."""
        self._azimuth = None

    def frame(self, state, trail=(), hud=()):
        """An RGB image (height, width, 3) of `state`, an EnvState.

        `trail` is a sequence of positions to mark, oldest first; `hud` is (label, value) pairs
        written in the top-left corner.
        """
        sim, w = self.env.sim, self.world
        # Sim.render draws from the Sim object, and the env keeps its state outside it.
        sim.data, sim.mjx_data = state.sim, state.mjx
        if self._viewer is None:
            sim.render(mode='rgb_array', world=w, width=self.width, height=self.height)
            self._viewer = sim.viewer.viewer
            self._set_font_scale()
        pos =np.asarray(state.sim.states.pos[w, 0], float)
        self._place_camera(pos, np.asarray(state.room[w], float))

        radius = MARKER_SCALE * self._viewer.cam.distance
        trail = np.asarray(trail, float).reshape(-1, 3)
        for p in trail[::max(1, math.ceil(len(trail) / MAX_TRAIL))]:
            self._viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.full(3, radius),
                                    pos=p, rgba=TRAIL_RGBA)
        if len(trail):
            self._viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=np.full(3, 2.5 * radius), pos=trail[0], rgba=START_RGBA)
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

    def _place_camera(self, pos, room):
        cam = self._viewer.cam
        if self.camera == 'top':
            fovy = self.env.sim.mj_model.vis.global_.fovy
            cam.lookat[:] = 0.0
            cam.distance = top_distance(room, self.width / self.height, fovy)
            cam.azimuth, cam.elevation = 90.0, -90.0
            return
        # Look outwards from the room centre through the drone, so the camera stays in the room
        # and the nearest wall is the backdrop. Near the centre that direction is noise: keep it.
        if math.hypot(pos[0], pos[1]) > 0.15:
            target = math.degrees(math.atan2(pos[1], pos[0]))
            self._azimuth = target if self._azimuth is None else (
                self._azimuth + CHASE_TURN * wrap_degrees(target - self._azimuth))
        elif self._azimuth is None:
            self._azimuth = 45.0
        cam.lookat[:] = pos
        cam.azimuth, cam.elevation = self._azimuth, CHASE_ELEVATION
        cam.distance = chase_distance(pos, forward_vector(self._azimuth, CHASE_ELEVATION), room,
                                      self.distance)

    def close(self):
        self.env.sim.close()
        self._viewer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
