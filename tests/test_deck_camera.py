"""The simulated AI-deck camera: its lens model, and that a rendered point lands on the pixel the
intrinsics put it on.

Rendering runs in a fresh interpreter, because the OpenGL backend is fixed when mujoco is first
imported and other tests have imported it already. The render tests skip where no EGL is available.
"""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones.sim.lens import Intrinsics, quat_to_matrix

# subprocess forks this JAX-threaded interpreter only to exec another one, which is safe.
pytestmark = pytest.mark.filterwarnings(r'ignore:os\.fork\(\) was called:RuntimeWarning')

# The deck's calibration from recordings/intrinsics.json, rounded.
DECK = Intrinsics(324, 244, 183.9, 187.1, 165.4, 145.1, (-0.0059, -0.0017, -0.0398, 0.0030))
# Distortion strong enough that a render ignoring it misses by pixels, not fractions of one.
STRONG = Intrinsics(160, 120, 110.0, 95.0, 88.0, 52.0, (-0.25, 0.06, 0.004, -0.003))


def gl_env():
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'egl')
    return env


def can_render():
    code = 'import mujoco; c = mujoco.GLContext(16, 16); c.make_current(); c.free()'
    return subprocess.run([sys.executable, '-c', code], env=gl_env(),
                          capture_output=True).returncode == 0


needs_gl = pytest.mark.skipif(not can_render(), reason='no offscreen OpenGL (MUJOCO_GL=egl)')


def test_loads_what_the_calibration_writes(tmp_path):
    path = tmp_path / 'intrinsics.json'
    path.write_text(json.dumps({
        'version': 1, 'width': 324, 'height': 244, 'fx': 183.9, 'fy': 187.1, 'cx': 165.4,
        'cy': 145.1, 'distortion': list(DECK.distortion), 'rms_px': 0.26, 'views': 60}))
    assert Intrinsics.load(path) == DECK
    path.write_text(json.dumps({'version': 2}))
    with pytest.raises(ValueError, match='version'):
        Intrinsics.load(path)


@pytest.mark.parametrize('k', [DECK, STRONG])
def test_undistort_inverts_distort(k):
    u, v = np.meshgrid(np.linspace(-0.5, k.width - 0.5, 17), np.linspace(-0.5, k.height - 0.5, 13))
    xd, yd = (u - k.cx) / k.fx, (v - k.cy) / k.fy
    x, y = k.undistort(xd, yd)
    # 1e-6 at unit depth is a ten-thousandth of a pixel; STRONG's corners converge only to 1e-7.
    np.testing.assert_allclose(k.distort(x, y), (xd, yd), atol=1e-6)


def test_halving_the_image_keeps_every_ray():
    # Pixel edges, not centres, scale with the image: (0.5, 0.5) of the half image is the corner
    # shared with (0, 0) ... (1, 1) of the full one.
    half = DECK.resized(162, 122)
    for u, v in [(-0.5, -0.5), (323.5, 243.5), (100.0, 200.0)]:
        full = ((u - DECK.cx) / DECK.fx, (v - DECK.cy) / DECK.fy)
        uh, vh = (u + 0.5) / 2 - 0.5, (v + 0.5) / 2 - 0.5
        np.testing.assert_allclose(((uh - half.cx) / half.fx, (vh - half.cy) / half.fy), full)


def test_quaternions_are_scalar_last():
    np.testing.assert_allclose(quat_to_matrix([0, 0, 0, 1]), np.eye(3))
    s = np.sqrt(0.5)
    np.testing.assert_allclose(quat_to_matrix([0, 0, s, s]) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


RENDER_POINTS = textwrap.dedent('''
    import json, math, sys, types
    import mujoco, numpy as np
    from drones.sim.deck_camera import DeckCamera
    from drones.sim.lens import Intrinsics, Mount, quat_to_matrix

    k = Intrinsics(*json.loads(sys.argv[1]))
    pitch, supersample = float(sys.argv[2]), int(sys.argv[3])
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <visual><headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
      <worldbody>
        <body name="drone" mocap="true">
          <geom type="box" size=".05 .05 .01" rgba="1 0 0 1"/>
          <geom type="sphere" size=".01" pos=".12 .05 -.03" rgba="1 0 0 1"/>
        </body>
        <body name="target" mocap="true">
          <geom type="sphere" size=".02" rgba="1 1 1 1" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>""")
    ns = types.SimpleNamespace
    env = ns(sim=ns(mj_model=model, data=ns(core=ns(drone_mocap_ids=[0]))))
    # The red sphere on the drone sits in plain view of this mount, away from every target.
    offset = np.array([0.02, 0.0, 0.015])
    cam = DeckCamera(env, k, mount=Mount(tuple(offset), pitch), supersample=supersample)
    data = cam.data

    pos = np.array([0.3, -0.2, 1.0])
    # Yawed, pitched and rolled at once, so a swapped axis or sign cannot cancel out.
    half = np.array([0.2, -0.15, 0.6]) / 2
    cr, cp, cy = np.cos(half)
    sr, sp, sy = np.sin(half)
    quat = [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]
    rot = quat_to_matrix(quat)
    axis, up = np.array([math.cos(pitch), 0, -math.sin(pitch)]), np.array(
        [math.sin(pitch), 0, math.cos(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    data.mocap_pos[0] = pos
    data.mocap_quat[0] = [quat[3], *quat[:3]]

    found = []
    for u, v in json.loads(sys.argv[4]):
        x, y = k.undistort((u - k.cx) / k.fx, (v - k.cy) / k.fy)
        depth = 1.5
        body = offset + depth * (axis + x * right - y * up)
        data.mocap_pos[1] = pos + rot @ body
        mujoco.mj_kinematics(model, data)
        image = cam.render(data, pos, quat).astype(float)
        white = image.min(axis=2)
        red = image[..., 0] - image[..., 1:].max(axis=2)
        vv, uu = np.nonzero(white > 30)
        w = white[vv, uu]
        found.append([float((uu * w).sum() / w.sum()), float((vv * w).sum() / w.sum()),
                      int((red > 100).sum())])
    cam.close()
    print(json.dumps({'found': found, 'shape': list(image.shape)}))
''')


@needs_gl
@pytest.mark.parametrize('k, pitch, supersample', [
    (STRONG, 0.0, 2),
    (STRONG, 0.35, 1),
    (DECK, 0.0, 2),
])
def test_a_point_lands_where_the_intrinsics_put_it(k, pitch, supersample):
    w, h = k.width, k.height
    pixels = [(k.cx, k.cy), (8.0, 8.0), (w - 9.0, 10.0), (12.0, h - 9.0), (w - 10.0, h - 8.0),
              (w / 2, h / 4)]
    args = [json.dumps([w, h, k.fx, k.fy, k.cx, k.cy, list(k.distortion)]), str(pitch),
            str(supersample), json.dumps(pixels)]
    done = subprocess.run([sys.executable, '-c', RENDER_POINTS, *args], env=gl_env(),
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result['shape'] == [h, w, 3]
    for (u, v), (fu, fv, red) in zip(pixels, result['found'], strict=True):
        # A 2 cm sphere 1.5 m away is a few pixels across; its centroid is its centre's
        # projection to well under a pixel, while ignoring the distortion misses by several.
        assert abs(fu - u) < 0.4 and abs(fv - v) < 0.4, ((u, v), (fu, fv))
        assert red == 0, 'the camera sees its own drone'
