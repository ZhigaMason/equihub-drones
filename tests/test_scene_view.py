"""Scene views for benchmark inference: poses in the scene file's frame, and the question poses
eqa hands them.

Rendering runs in a fresh interpreter, because the OpenGL backend is fixed when mujoco is first
imported and other tests have imported it already. The render tests skip where no EGL is available.
"""
import json
import math
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones.sim import eqa
from drones.sim.lens import Mount, euler_to_matrix

pytestmark = pytest.mark.filterwarnings(r'ignore:os\.fork\(\) was called:RuntimeWarning')


def gl_env():
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'egl')
    return env


def can_render():
    code = 'import mujoco; c = mujoco.GLContext(16, 16); c.make_current(); c.free()'
    return subprocess.run([sys.executable, '-c', code], env=gl_env(),
                          capture_output=True).returncode == 0


needs_gl = pytest.mark.skipif(not can_render(), reason='no offscreen OpenGL (MUJOCO_GL=egl)')


def question(benchmark, start=None, yaw=None, path=None):
    return eqa.Question(benchmark, 1, 'scene', 'text', 'answer', 'category', start=start, yaw=yaw,
                        path=path)


def test_euler_angles_follow_the_drone_attitude_signs():
    np.testing.assert_allclose(euler_to_matrix(yaw=math.pi / 2) @ [1, 0, 0], [0, 1, 0],
                               atol=1e-12)                       # + yaw turns left
    assert (euler_to_matrix(pitch=0.3) @ [1, 0, 0])[2] < 0       # + pitch puts the nose down
    assert (euler_to_matrix(roll=0.3) @ [0, 1, 0])[2] > 0        # + roll drops the right side


def test_a_mount_moves_the_eye_with_the_body():
    eye, rot = Mount((0.1, 0.0, 0.02), pitch=0.2).camera_pose(
        np.array([1.0, 2.0, 3.0]), euler_to_matrix(yaw=math.pi / 2))
    np.testing.assert_allclose(eye, [1.0, 2.1, 3.02], atol=1e-12)
    np.testing.assert_allclose(rot, euler_to_matrix(yaw=math.pi / 2, pitch=0.2), atol=1e-12)


def test_habitat_starts_are_lifted_off_the_floor_and_indoor_uav_flies_already():
    start = np.array([1.0, 2.0, 0.1])
    pos, yaw = eqa.start_pose(question('hm-eqa', start, 0.5), origin=np.zeros(3), height=1.2)
    np.testing.assert_allclose(pos, [1.0, 2.0, 1.3])
    assert yaw == 0.5
    pos, _ = eqa.start_pose(question('indoor-uav', start, 0.5), origin=np.zeros(3))
    np.testing.assert_allclose(pos, start)
    # A-EQA gives no start: the scene's open floor stands in for it.
    pos, yaw = eqa.start_pose(question('a-eqa'), origin=np.array([4.0, 5.0, 0.2]), height=1.0)
    np.testing.assert_allclose(pos, [4.0, 5.0, 1.2])
    assert yaw == 0.0


def test_path_poses_face_along_the_path():
    path = np.array([[0.0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1.5]])
    poses = eqa.path_poses(question('indoor-uav', path=path, yaw=0.0))
    np.testing.assert_allclose([p for p, _ in poses], path)
    yaws = [y for _, y in poses]
    assert yaws[0] == pytest.approx(0.0)
    assert yaws[1] == pytest.approx(math.pi / 4)   # the turn, seen from both neighbours
    assert yaws[2] == pytest.approx(math.pi / 2)
    assert yaws[3] == pytest.approx(math.pi / 2)   # climbing in place keeps the heading
    assert eqa.path_poses(question('hm-eqa')) == []


RENDER_CUBE = textwrap.dedent('''
    import itertools, json, sys
    import numpy as np
    from drones.sim import scenes
    from drones.sim.lens import Intrinsics, euler_to_matrix
    from drones.sim.scene_view import SceneView

    target, origin = np.array([2.0, -1.0, 1.2]), np.array([5.0, -3.0, 0.5])
    corners = np.array(list(itertools.product((-0.02, 0.02), repeat=3)))
    faces = []
    for axis in range(3):
        for side in (0, 1):
            quad = [i for i, c in enumerate(corners) if (c[axis] > 0) == side]
            faces += [[quad[0], quad[1], quad[3]], [quad[0], quad[3], quad[2]]]
    part = scenes.Part((corners + target - origin).astype(np.float32),
                       np.array(faces, np.int32), np.zeros((8, 2), np.float32),
                       np.full((2, 2, 3), 255, np.uint8))
    scene = scenes.Scene('cube', [part], origin, 1.0)
    k = Intrinsics(160, 120, 110.0, 95.0, 84.0, 52.0, (-0.1, 0.02, 0.003, -0.002))

    found = []
    with SceneView(scene, k) as view:
        # Flat light, so a pixel's brightness is how much of it the cube covers and the weighted
        # centroid is sub-pixel; the scene's own headlight lights each face differently.
        view.model.vis.headlight.ambient[:], view.model.vis.headlight.diffuse[:] = 1.0, 0.0
        for eye, yaw, pitch, roll in json.loads(sys.argv[1]):
            image = view.render(eye, yaw, pitch, roll).astype(float).min(axis=2)
            v, u = np.nonzero(image > 10)
            w = image[v, u]
            found.append([float((u * w).sum() / w.sum()), float((v * w).sum() / w.sum())])
    print(json.dumps(found))
''')


@needs_gl
def test_a_point_in_the_file_frame_lands_where_the_lens_puts_it():
    from drones.sim.lens import Intrinsics

    k = Intrinsics(160, 120, 110.0, 95.0, 84.0, 52.0, (-0.1, 0.02, 0.003, -0.002))
    target = np.array([2.0, -1.0, 1.2])   # the scene's file frame, not the shifted one
    poses = [([0.5, -1.0, 1.2], 0.0, 0.0, 0.0),
             ([0.8, -1.6, 1.5], 0.3, 0.1, 0.0),
             ([2.0, -2.5, 0.9], math.pi / 2 + 0.2, -0.15, 0.25),
             ([3.2, 0.1, 1.0], -2.5, 0.0, -0.3)]
    done = subprocess.run([sys.executable, '-c', RENDER_CUBE, json.dumps(poses)], env=gl_env(),
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    found = json.loads(done.stdout.strip().splitlines()[-1])
    for (eye, yaw, pitch, roll), (fu, fv) in zip(poses, found, strict=True):
        p = euler_to_matrix(yaw, pitch, roll).T @ (target - eye)   # forward, left, up
        u, v = k.pixel(-p[1] / p[0], -p[2] / p[0])
        assert abs(fu - u) < 0.4 and abs(fv - v) < 0.4, ((u, v), (fu, fv))
