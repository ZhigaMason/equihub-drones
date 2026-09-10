"""Simulated Crazyflie sensors, as pure batched JAX functions.

Each model reports what the real hardware reports, so a policy trained on these readings reads the
same quantities off the drone:

* Multi-ranger deck: VL53L1x time-of-flight rangers looking front, back, left, right and up.
* Flow deck v2: a VL53L1x looking down (the z-ranger) and a PMW3901 optical flow sensor.
* IMU: gyro rates, and the gravity direction implied by the onboard attitude estimate.
* A forward-looking colour camera, the geometry of an AI deck.

Rays are cast with ``mjx.ray`` against the scene: floor, walls, ceiling and anything else added to
the scene XML. The drone's own geoms live in groups 2 and 3 and are excluded, so a sensor never sees
its own airframe. All arrays are batched over worlds: positions are (n, 3), quaternions (n, 4).
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from mujoco import mjx

from drones.policy.interface import BASELINE, validate
from drones.sim.geometry import quat_to_matrix

# Geom groups the rays can hit: everything except the drones' visual (2) and collision (3) geoms.
SCENE_GEOMGROUP = (1, 1, 0, 0, 1, 1, 1, 1)

# Ranger axes in the body frame (+x forward, +y left, +z up). The first five are the Multi-ranger
# deck, the last is the Flow deck's downward z-ranger.
RANGER_NAMES = ('front', 'back', 'left', 'right', 'up', 'down')
_RANGER_AXES = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
DOWN = RANGER_NAMES.index('down')

SKY = (0.55, 0.7, 0.85)
_FLOOR_LIGHT = (0.35, 0.45, 0.55)
_FLOOR_DARK = (0.15, 0.2, 0.28)


@dataclass(frozen=True)
class SensorConfig:
    # What the policy observes: any of drones.policy.interface.SENSORS. Every ray is still cast each
    # step, because crash detection and the flow model need them regardless.
    enabled: tuple[str, ...] = BASELINE
    # VL53L1x in long-distance mode reads out to about 4 m. Noise grows with distance.
    range_max: float = 4.0
    range_noise_abs: float = 0.005
    range_noise_rel: float = 0.01
    # PMW3901, with the constants the Crazyflie firmware's flow model uses (mm_flow.c): 35 pixels
    # across 0.71674 rad, one frame every 10 ms. The firmware also clamps height at 0.1 m.
    flow_npix: float = 35.0
    flow_thetapix: float = 0.71674
    flow_dt: float = 0.01
    flow_noise: float = 0.05
    flow_min_height: float = 0.1
    gyro_noise: float = 0.01
    gravity_noise: float = 0.01
    # Forward camera, used when 'camera' is enabled. 70 degrees vertical matches CrazyFlow's
    # fpv_cam; the resolution is kept small because every pixel is a ray in every world.
    camera_resolution: tuple[int, int] = (32, 24)  # (width, height)
    camera_fov_y: float = 1.2217
    camera_pitch: float = 0.0  # rad, positive tilts the optical axis down
    floor_tile: float = 0.5

    def __post_init__(self):
        validate(self.enabled)

    @property
    def camera(self):
        return 'camera' in self.enabled

    @property
    def flow_gain(self):
        """Counts per frame per rad/s of apparent motion: the firmware's dt * Npix / thetapix."""
        return self.flow_dt * self.flow_npix / self.flow_thetapix


def cast_rays(mjx_model, mjx_data, origins, directions):
    """Cast rays from `origins` (n, 3) along unit `directions` (n, k, 3), in the world frame.

    Returns distances (n, k), inf where nothing is hit, and the geom ids hit (n, k), -1 for a miss.
    """
    def one_world(data, origin, dirs):
        return jax.vmap(lambda v: mjx.ray(mjx_model, data, origin, v,
                                          geomgroup=SCENE_GEOMGROUP))(dirs)

    dist, geom = jax.vmap(one_world)(mjx_data, origins, directions)
    return jnp.where(dist < 0, jnp.inf, dist), geom


def ranger_distances(mjx_model, mjx_data, pos, quat):
    """True distance along each ranger's axis (n, 6), in RANGER_NAMES order; inf for no hit."""
    axes = jnp.einsum('nij,kj->nki', quat_to_matrix(quat), jnp.asarray(_RANGER_AXES))
    return cast_rays(mjx_model, mjx_data, pos, axes)[0]


def ranger_readings(distances, key, config):
    """What the VL53L1x chips report (n, 6): noisy, and clipped to the sensor's range.

    Out of range reads as `range_max`. cflib reports it as None; the deployed policy should map
    None to `range_max` too.
    """
    clipped = jnp.minimum(distances, config.range_max)
    std = config.range_noise_abs + config.range_noise_rel * clipped
    noisy = clipped + std * jax.random.normal(key, clipped.shape)
    return jnp.clip(noisy, 0.0, config.range_max)


def optical_flow(vel, ang_vel, quat, down_distance, key, config):
    """PMW3901 counts per 10 ms frame (n, 2), from the Crazyflie firmware's flow model.

    Flow is the apparent motion of the ground: body-frame velocity over the distance to it, less
    the drone's own rotation. Forward motion gives +x, leftward motion +y. The firmware divides by
    the estimated height and multiplies by the tilt term R[2][2]; over flat ground that equals
    dividing by the down-ranger's slant distance, which is used here so the model also holds over
    obstacles.

    Args:
        vel: World-frame velocity (n, 3).
        ang_vel: Body-frame angular velocity (n, 3), rad/s.
        quat: Orientation (n, 4).
        down_distance: True down-ranger distance (n,).
    """
    rot = quat_to_matrix(quat)
    v_body = jnp.einsum('nji,nj->ni', rot, vel)  # R^T v
    distance = jnp.clip(down_distance, config.flow_min_height, config.range_max)
    rate = jnp.stack([v_body[:, 0] / distance - ang_vel[:, 1],
                      v_body[:, 1] / distance + ang_vel[:, 0]], -1)
    flow = config.flow_gain * rate
    return flow + config.flow_noise * jax.random.normal(key, flow.shape)


def imu(quat, ang_vel, key, config):
    """Gyro rates (n, 3) in rad/s, and the unit gravity direction in the body frame (n, 3).

    The gravity direction carries the roll and pitch the onboard estimator reports, without the
    wrap-around of Euler angles: level flight reads (0, 0, -1).
    """
    gravity = -quat_to_matrix(quat)[:, 2, :]  # R^T @ (0, 0, -1)
    k_gyro, k_gravity = jax.random.split(key)
    gyro = ang_vel + config.gyro_noise * jax.random.normal(k_gyro, ang_vel.shape)
    gravity = gravity + config.gravity_noise * jax.random.normal(k_gravity, gravity.shape)
    return gyro, gravity / jnp.linalg.norm(gravity, axis=-1, keepdims=True)


def camera_directions(config):
    """Unit pixel rays in the body frame, (height, width, 3); row 0 is the top of the image.

    The optical axis looks along +x (forward), image right is -y and image up is +z, like the
    forward-facing camera of an AI deck. `camera_pitch` tilts the whole view down.
    """
    width, height = config.camera_resolution
    tan_y = jnp.tan(config.camera_fov_y / 2)
    tan_x = tan_y * width / height
    u = ((jnp.arange(width) + 0.5) / width * 2 - 1) * tan_x      # left to right
    v = (1 - (jnp.arange(height) + 0.5) / height * 2) * tan_y    # top to bottom
    uu, vv = jnp.meshgrid(u, v)
    dirs = jnp.stack([jnp.ones_like(uu), -uu, vv], -1)
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    c, s = jnp.cos(config.camera_pitch), jnp.sin(config.camera_pitch)
    return jnp.stack([c * dirs[..., 0] + s * dirs[..., 2],
                      dirs[..., 1],
                      -s * dirs[..., 0] + c * dirs[..., 2]], -1)


def render_camera(mjx_model, mjx_data, pos, quat, geom_colours, floor_geom, config):
    """Flat-shaded colour images (n, height, width, 3) in [0, 1], by raycasting the scene.

    Each pixel takes the colour of the geom its ray hits, the floor gets a checkerboard so the image
    carries texture, and everything darkens with distance. It has the right geometry and field of
    view for an AI deck, but it is not photoreal: CrazyFlow's gaussian-splat camera
    (crazyflow.sim.sensors.splat, GPU only) is the drop-in upgrade.

    Args:
        geom_colours: RGB per geom (ngeom, 3), from the MuJoCo model.
        floor_geom: Geom id of the floor plane.
    """
    width, height = config.camera_resolution
    dirs = jnp.einsum('nij,pj->npi', quat_to_matrix(quat),
                      camera_directions(config).reshape(-1, 3))
    dist, geom = cast_rays(mjx_model, mjx_data, pos, dirs)
    hit = jnp.isfinite(dist)
    reach = jnp.where(hit, dist, 0.0)

    colour = geom_colours[jnp.maximum(geom, 0)]
    points = pos[:, None, :] + dirs * reach[..., None]
    tile = jnp.floor(points[..., 0] / config.floor_tile) + jnp.floor(points[..., 1] / config.floor_tile)
    floor = jnp.where((tile % 2 > 0)[..., None], jnp.asarray(_FLOOR_LIGHT), jnp.asarray(_FLOOR_DARK))
    colour = jnp.where((geom == floor_geom)[..., None], floor, colour)
    shade = 0.35 + 0.65 * jnp.exp(-reach / 6.0)
    image = jnp.where(hit[..., None], colour * shade[..., None], jnp.asarray(SKY))
    return image.reshape(-1, height, width, 3)
