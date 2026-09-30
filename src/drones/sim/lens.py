"""A calibrated camera for any MuJoCo model: OpenCV intrinsics and distortion, any pose, roll too.

The pieces every image source in the simulator shares, whatever it looks at: `DeckCamera` (the
AI-deck on a drone in a HoverEnv or SquareEnv) and `SceneView` (a dataset's scanned scene alone, for
benchmark inference) are thin wrappers that only decide the model and the pose.

- `Intrinsics` is the lens: the JSON `drones.vision.calibrate` writes (neural-sandbox branch) for
  the real deck, `recordings/intrinsics.json`, or an ideal pinhole from a field of view.
- `Mount` is where a camera sits on the drone: a body-frame offset and a downward pitch.
- `LensCamera` renders one model through a lens from a camera pose.

Why `LensCamera` is built this way:

- A MuJoCo camera draws a symmetric frustum from one fovy, with square pixels. The deck's principal
  point is ~24 px below the image centre and fx != fy, so after mjv_updateScene the frustum of the
  scene's GL cameras is overwritten directly. That needs no camera in the model, so it works on a
  model compiled at any point, `scenes.attach`'s included.
- OpenGL cannot draw lens distortion. The scene is rendered through an ideal pinhole covering
  everything the distorted image sees, then every output pixel is looked up through OpenCV's
  distortion model. The lookup is computed once, in numpy: the sim extra has no OpenCV.
- The GL camera's pose is overwritten too, because a free mjvCamera cannot roll. The free camera is
  still aimed along the optical axis first: mjv_updateScene places the headlight from it.
- MuJoCo's near plane is a fraction of the model's extent. With a 30 m house in the model that is
  ~0.3 m, so a wall closer than that would vanish; the near plane is capped at NEAR_CLIP.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_INTRINSICS = Path('recordings/intrinsics.json')
FORMAT_VERSION = 1       # of intrinsics.json
SUPERSAMPLE = 2          # pinhole samples per output pixel, along each axis
UNDISTORT_ITERATIONS = 20
PAD = 1                  # pinhole pixels beyond the outermost sample, for the bilinear taps
NEAR_CLIP = 0.01         # m, at most
# The default mjvOption draws geom groups 0-2; `hidden_geoms` move here while a view is built.
HIDDEN_GROUP = 5


@dataclass(frozen=True)
class Intrinsics:
    """OpenCV's pinhole camera: pixel centres at integer (u, v), +u right, +v down, and distortion
    (k1, k2, p1, p2) with k3 = 0, as the calibration fixes it."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def load(cls, path=DEFAULT_INTRINSICS):
        c = json.loads(Path(path).read_text())
        if c.get('version') != FORMAT_VERSION:
            raise ValueError(f'{path} is intrinsics version {c.get("version")}, this reads '
                             f'version {FORMAT_VERSION}')
        return cls(int(c['width']), int(c['height']), float(c['fx']), float(c['fy']),
                   float(c['cx']), float(c['cy']), tuple(float(d) for d in c['distortion']))

    @classmethod
    def from_fov(cls, width, height, hfov):
        """An ideal pinhole with square pixels, centred, `hfov` radians across: how habitat-sim
        and most EQA agents describe their camera (hfov 90 degrees, typically)."""
        f = width / 2 / math.tan(hfov / 2)
        return cls(width, height, f, f, (width - 1) / 2, (height - 1) / 2)

    def resized(self, width, height):
        """The same lens at another resolution: a calibration made on the raw 324x244 stream
        describes the 162x122 colour stream once halved."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy,
                          (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5, self.distortion)

    def distort(self, x, y):
        """Normalised pinhole coordinates (x right, y down, at unit depth) to distorted ones."""
        k1, k2, p1, p2 = self.distortion
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    def undistort(self, xd, yd):
        """The inverse of `distort`, by fixed-point iteration as cv2.undistortPoints does. It
        converges for a lens as mild as the deck's (|k1| ~ 0.01); check `distort` of the result
        before trusting it on a fisheye."""
        x, y = xd, yd
        for _ in range(UNDISTORT_ITERATIONS):
            dx, dy = self.distort(x, y)
            x, y = x + (xd - dx), y + (yd - dy)
        return x, y

    def pixel(self, x, y):
        """Where normalised pinhole coordinates land in the image, distortion included."""
        xd, yd = self.distort(x, y)
        return self.fx * xd + self.cx, self.fy * yd + self.cy


def quat_to_matrix(quat):
    """Rotation matrix of a scalar-last quaternion [x, y, z, w]."""
    x, y, z, w = np.asarray(quat, float) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def euler_to_matrix(yaw=0.0, pitch=0.0, roll=0.0):
    """Body-to-world rotation, yaw about z, then pitch about y, then roll about x. Positive pitch
    puts the nose down, and positive yaw turns left, as the drone's attitude does."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


@dataclass(frozen=True)
class Mount:
    """Where a camera sits on the drone: `offset` is the optical centre in the body frame (m), and
    `pitch` (rad) tilts the optical axis down, like SensorConfig.camera_pitch."""
    offset: tuple = (0.0, 0.0, 0.0)
    pitch: float = 0.0

    def camera_pose(self, pos, rotation):
        """(eye, camera rotation) for a drone at `pos` with body-to-world `rotation`."""
        eye = np.asarray(pos, float) + rotation @ np.asarray(self.offset, float)
        return eye, rotation @ euler_to_matrix(pitch=self.pitch)


# Estimated from the AI-deck's layout (camera at the deck's front edge, deck on top of the frame),
# not measured on this drone.
DECK_MOUNT = Mount(offset=(0.02, 0.0, 0.015))


class LensCamera:
    """Offscreen renders of `model` through `intrinsics`.

    A camera rotation is body-style: its columns are the optical axis, image left and image up, in
    the world frame, so a drone's own attitude is a camera looking straight ahead. `hidden_geoms`
    are left out of every view (the camera's own drone). Use as a context manager, or call
    close().
    """

    def __init__(self, model, intrinsics, supersample=SUPERSAMPLE, hidden_geoms=()):
        if supersample < 1:
            raise ValueError(f'supersample must be at least 1, not {supersample}')
        self.model, self.intrinsics = model, intrinsics
        self.width, self.height = intrinsics.width, intrinsics.height
        self._supersample = supersample
        self._hidden = np.asarray(hidden_geoms, np.intp)
        self._build_lookup()
        width, height = self._render_size
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self._renderer = mujoco.Renderer(model, height, width)
        self._camera = mujoco.MjvCamera()

    def _build_lookup(self):
        k, s = self.intrinsics, self._supersample
        sub = (np.arange(s) + 0.5) / s - 0.5
        u = (np.arange(k.width)[:, None] + sub).ravel()
        v = (np.arange(k.height)[:, None] + sub).ravel()
        uu, vv = np.meshgrid(u, v)
        x, y = k.undistort((uu - k.cx) / k.fx, (vv - k.cy) / k.fy)

        # Pinhole pixels s times finer than the lens's own, placed so that without distortion
        # every sample falls on a pixel centre and the lookup is exact.
        px, py = 1 / (s * k.fx), 1 / (s * k.fy)
        x0, y0 = x.min() - (0.5 + PAD) * px, y.min() - (0.5 + PAD) * py
        width = math.ceil((x.max() - x0) / px + 0.5 + PAD)
        height = math.ceil((y.max() - y0) / py + 0.5 + PAD)
        self._render_size = width, height
        # The window on the unit-depth plane the pinhole render covers; y down, as OpenCV's.
        self._window = x0, x0 + width * px, y0, y0 + height * py

        i, j = (x - x0) / px - 0.5, (y - y0) / py - 0.5
        i0, j0 = np.floor(i).astype(np.intp), np.floor(j).astype(np.intp)
        fi, fj = (i - i0)[..., None], (j - j0)[..., None]
        self._taps = [(j0, i0, (1 - fj) * (1 - fi)), (j0, i0 + 1, (1 - fj) * fi),
                      (j0 + 1, i0, fj * (1 - fi)), (j0 + 1, i0 + 1, fj * fi)]

    def render(self, data, eye, rotation):
        """An RGB image (height, width, 3) uint8 from a camera at `eye` with body-style
        `rotation`. `data` is an MjData of this camera's model, put through mj_kinematics."""
        eye = np.asarray(eye, float)
        forward, up = rotation[:, 0], rotation[:, 2]

        cam = self._camera
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 1.0
        cam.lookat[:] = eye + forward
        cam.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
        cam.elevation = math.degrees(math.asin(np.clip(forward[2], -1.0, 1.0)))
        # The model may be shared with other renderers, so its groups change only for this call.
        groups = self.model.geom_group[self._hidden].copy()
        self.model.geom_group[self._hidden] = HIDDEN_GROUP
        try:
            self._renderer.update_scene(data, camera=cam)
        finally:
            self.model.geom_group[self._hidden] = groups

        x0, x1, y0, y1 = self._window
        for gl in self._renderer.scene.camera:   # both eyes: mjr_render draws their average
            near = gl.frustum_near = min(gl.frustum_near, NEAR_CLIP)
            gl.pos[:], gl.forward[:], gl.up[:] = eye, forward, up
            gl.frustum_center = (x0 + x1) / 2 * near
            gl.frustum_width = (x1 - x0) / 2 * near
            gl.frustum_top, gl.frustum_bottom = -y0 * near, -y1 * near
        pinhole = self._renderer.render().astype(np.float32)

        image = sum(pinhole[j, i] * weight for j, i, weight in self._taps)
        s = self._supersample
        image = image.reshape(self.height, s, self.width, s, 3).mean(axis=(1, 3))
        return np.clip(np.rint(image), 0, 255).astype(np.uint8)

    def close(self):
        self._renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
