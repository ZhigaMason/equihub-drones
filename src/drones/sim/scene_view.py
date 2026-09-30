"""Camera images of a dataset's scanned scene alone, for benchmark inference.

    from drones.sim import eqa
    from drones.sim.lens import Intrinsics
    from drones.sim.scene_view import SceneView

    q = eqa.load('hm-eqa')[0]
    with SceneView(q.scene, Intrinsics.load()) as view:        # the deck's lens, or from_fov(...)
        pos, yaw = eqa.start_pose(q, view.scene.origin)
        image = view.render(pos, yaw)                          # (244, 324, 3) uint8
        pos = pos + [0.5, 0.0, 0.0]                            # wherever the agent goes next
        image = view.render(pos, yaw, pitch=0.2)

An agent answering an EQA question or following an IndoorUAV instruction needs only an image for a
pose. It does not need the dynamics, so this builds no CrazyFlow Sim: the model holds the scan and
nothing else, and a render costs one offscreen draw.

Poses are in the scene file's frame, the frame benchmark poses are in (after eqa.habitat_point),
z up, so a question's start and path go in as they are. `scenes.load` shifts the scan's vertices so
its open floor is at the origin; here the geoms are shifted back by `scene.origin`. With a `mount`,
a pose is the drone's and the camera sits on it (lens.DECK_MOUNT for the AI-deck); without one it
is the camera's own.

The scan is lit as `scenes.attach` lights it, by the headlight only.

`drone=` also puts CrazyFlow's model of the drone in the scene, posed at every render, for a
`ChaseCamera` to film from behind. The view through `intrinsics` leaves it out, as a real deck's
frames do.
"""
import math
from pathlib import Path

import mujoco
import numpy as np

from drones.sim import scenes
from drones.sim.lens import (SUPERSAMPLE, Intrinsics, LensCamera, Mount, euler_to_matrix,
                             quat_to_matrix)

DRONE = 'cf21B_500'       # CrazyFlow's Crazyflie 2.1 Brushless, the one the simulator flies
CHASE_DISTANCE = 1.0      # m behind the drone, when the scan leaves room
CHASE_ELEVATION = math.radians(20)   # looking down on it
CHASE_HFOV = math.radians(70)
CAMERA_MARGIN = 0.1       # m kept between the chase camera and the scan
MIN_CHASE_DISTANCE = 0.2  # m; closer than this the drone fills the picture


def load_scene(scene, dest=scenes.SCENES_DIR):
    """A scenes.Scene from a Scene, a .glb path, or a scene's name (Gibson) or id (HM3D)."""
    if isinstance(scene, scenes.Scene):
        return scene
    path = Path(scene)
    if path.suffix != '.glb':
        path = scenes.scene_path(str(scene), dest)
    if not path.is_file():
        raise FileNotFoundError(f'no scene at {path}; fetch it with drones-download-scenes '
                                f'{Path(path).stem}')
    return scenes.load(path)


def add_drone(spec, drone=DRONE):
    """Add CrazyFlow's model `drone` to `spec` as a mocap body named 'drone'."""
    import crazyflow

    path = Path(crazyflow.__file__).parent / 'drones' / f'{drone}.xml'
    drone_spec = mujoco.MjSpec.from_file(str(path))
    # The drone's meshes are files relative to its own XML; the scan's are vertex data.
    spec.meshdir = str((path.parent / drone_spec.meshdir).resolve())
    body = drone_spec.body('drone')
    body.mocap = True
    spec.worldbody.add_frame().attach_body(body, '', '')


def scene_model(scene, drone=None):
    """A model of `scene`, in the scene file's frame, with CrazyFlow's model `drone` if given."""
    spec = mujoco.MjSpec()
    scenes.add_to_spec(spec, scene)
    if drone is not None:
        add_drone(spec, drone)
    model = spec.compile()
    # Compiling re-centres each mesh on its own frame and puts that offset in geom_pos. The scan's
    # geoms are the world body's; the drone's sit on their own body.
    model.geom_pos[model.geom_bodyid == 0] += scene.origin
    model.vis.quality.offsamples = 4
    model.vis.headlight.ambient[:] = scenes.HEADLIGHT_AMBIENT
    model.vis.headlight.diffuse[:] = scenes.HEADLIGHT_DIFFUSE
    return model


class SceneView:
    """Renders one scanned scene through `intrinsics`. `scene` is anything `load_scene` takes;
    `drone` names a CrazyFlow drone model to show to other cameras (`camera`, `ChaseCamera`).
    Use as a context manager, or call close()."""

    def __init__(self, scene, intrinsics, mount=None, supersample=SUPERSAMPLE,
                 dest=scenes.SCENES_DIR, drone=None):
        self.scene = load_scene(scene, dest)
        self.mount = mount or Mount()
        self.model = model = scene_model(self.scene, drone)
        self.data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, self.data)
        self._drone_body, drone_geoms = -1, ()
        if drone is not None:
            self._drone_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'drone')
            drone_geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] == self._drone_body)
        self.lens = LensCamera(model, intrinsics, supersample, hidden_geoms=drone_geoms)
        self.width, self.height = self.lens.width, self.lens.height
        self._cameras = [self.lens]

    def render(self, pos, yaw=0.0, pitch=0.0, roll=0.0):
        """An RGB image (height, width, 3) uint8 from `pos` (file frame) at this attitude, in rad:
        yaw 0 faces +x and turns left, positive pitch looks down."""
        return self.render_matrix(pos, euler_to_matrix(yaw, pitch, roll))

    def render_quat(self, pos, quat):
        """As `render`, with the attitude as a scalar-last quaternion."""
        return self.render_matrix(pos, quat_to_matrix(quat))

    def render_matrix(self, pos, rotation):
        """As `render`, with the attitude as a body-to-world rotation matrix. Also moves the drone
        there, for the other cameras."""
        pos = np.asarray(pos, float)
        self.place(pos, rotation)
        return self.lens.render(self.data, *self.mount.camera_pose(pos, rotation))

    def place(self, pos, rotation):
        """Pose the drone (if the scene has one) at `pos` with body-to-world `rotation`."""
        if self._drone_body < 0:
            return
        mocap = self.model.body_mocapid[self._drone_body]
        self.data.mocap_pos[mocap] = pos
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation, float).ravel())
        self.data.mocap_quat[mocap] = quat   # MuJoCo's scalar-first, straight from the matrix
        mujoco.mj_kinematics(self.model, self.data)

    def camera(self, intrinsics, supersample=SUPERSAMPLE):
        """Another camera on this scene, which sees the drone. Closed with the view."""
        lens = LensCamera(self.model, intrinsics, supersample)
        self._cameras.append(lens)
        return lens

    def distance(self, origin, direction):
        """How far the scan is from `origin` along unit `direction` (inf if it is not there)."""
        groups = np.zeros(mujoco.mjNGROUP, np.uint8)
        groups[[scenes.LOWER_GROUP, scenes.UPPER_GROUP]] = 1
        geom = np.zeros(1, np.int32)
        # By group, and with the drone's body excluded: mj_ray would hit its collision sphere.
        hit = mujoco.mj_ray(self.model, self.data, np.asarray(origin, float),
                            np.asarray(direction, float), groups, 1, self._drone_body, geom)
        return hit if hit >= 0 else math.inf

    def close(self):
        for lens in self._cameras:
            lens.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ChaseCamera:
    """Films the drone of a SceneView from behind and above, through an ideal pinhole, pulled in
    when the scan is between them. The view must have been built with `drone=`."""

    def __init__(self, view, width, height, hfov=CHASE_HFOV, distance=CHASE_DISTANCE,
                 elevation=CHASE_ELEVATION):
        self.view, self.distance, self.elevation = view, distance, elevation
        self.lens = view.camera(Intrinsics.from_fov(width, height, hfov))

    def render(self, pos, rotation):
        """An RGB image of the drone at `pos` with body-to-world `rotation`, from behind its
        heading: the camera does not roll or pitch with it."""
        pos = np.asarray(pos, float)
        self.view.place(pos, rotation)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        camera = euler_to_matrix(yaw, self.elevation)
        back = -camera[:, 0]
        room = self.view.distance(pos, back) - CAMERA_MARGIN
        eye = pos + back * max(MIN_CHASE_DISTANCE, min(self.distance, room))
        return self.lens.render(self.view.data, eye, camera)
