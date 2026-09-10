"""drones-render-hover: camera placement, and real renders of a saved run.

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

from drones.sim.render import (CAMERA_MARGIN, MIN_DISTANCE, chase_distance, forward_vector,
                               top_distance)

# subprocess forks this JAX-threaded interpreter only to exec another one, which is safe.
pytestmark = pytest.mark.filterwarnings(r'ignore:os\.fork\(\) was called:RuntimeWarning')

SMALLEST_ROOM = (0.75, 0.75, 1.6)   # the smallest room HoverConfig samples
LARGEST_ROOM = (2.5, 2.5, 3.0)


def gl_env():
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'egl')
    return env


def can_render():
    code = 'import mujoco; c = mujoco.GLContext(16, 16); c.make_current(); c.free()'
    return subprocess.run([sys.executable, '-c', code], env=gl_env(),
                          capture_output=True).returncode == 0


needs_gl = pytest.mark.skipif(not can_render(), reason='no offscreen OpenGL (MUJOCO_GL=egl)')


def test_forward_vector_follows_mujoco_free_camera_angles():
    np.testing.assert_allclose(forward_vector(0, 0), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(forward_vector(90, 0), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(forward_vector(0, -90), [0, 0, -1], atol=1e-12)


@pytest.mark.parametrize('room', [SMALLEST_ROOM, LARGEST_ROOM])
def test_chase_camera_never_leaves_the_room(room):
    # Outside the room the opaque wall slabs hide the drone: the bug this camera exists to avoid.
    rng = np.random.default_rng(0)
    inner = np.array(room) - 0.15
    for _ in range(2000):
        pos = rng.uniform([-inner[0], -inner[1], 0.15], [inner[0], inner[1], inner[2]])
        forward = forward_vector(rng.uniform(-180, 180), rng.uniform(-60, 0))
        camera = pos - chase_distance(pos, forward, room, 1.2) * forward
        assert abs(camera[0]) <= room[0] - CAMERA_MARGIN + 1e-9
        assert abs(camera[1]) <= room[1] - CAMERA_MARGIN + 1e-9
        assert CAMERA_MARGIN - 1e-9 <= camera[2] <= room[2] - CAMERA_MARGIN + 1e-9


def test_chase_camera_keeps_its_distance_when_nothing_is_in_the_way():
    forward = forward_vector(0, -25)
    assert chase_distance(np.array([1.0, 0, 1.0]), forward, LARGEST_ROOM, 1.2) == 1.2
    assert chase_distance(np.array([0.7, 0, 1.0]), forward, SMALLEST_ROOM, 1.2) >= MIN_DISTANCE


@pytest.mark.parametrize('room', [SMALLEST_ROOM, LARGEST_ROOM, (2.5, 0.75, 2.0)])
def test_top_camera_frames_the_whole_room(room):
    aspect, fovy = 640 / 480, 45.0
    half_height = top_distance(room, aspect, fovy) * math.tan(math.radians(fovy) / 2)
    assert half_height >= room[1] and half_height * aspect >= room[0]


@pytest.fixture(scope='module')
def saved_run(tmp_path_factory):
    """A run directory as drones-train-hover writes it, with untrained parameters."""
    import jax

    from drones.rl.networks import ActorCritic
    from drones.rl.ppo import PPOConfig, config_to_json, save_params
    from drones.sim.hover_env import HoverConfig, HoverEnv

    run = tmp_path_factory.mktemp('run')
    env_config, ppo_config = HoverConfig(num_envs=1), PPOConfig()
    env = HoverEnv(env_config)
    _, obs = env.reset(jax.random.key(0))
    model = ActorCritic(env.action_size, ppo_config.hidden, use_image=False)
    save_params(run / 'params.msgpack', model.init(jax.random.key(0), obs))
    (run / 'config.json').write_text(json.dumps({
        'env': json.loads(config_to_json(env_config)),
        'ppo': json.loads(config_to_json(ppo_config)),
    }))
    return run


@needs_gl
def test_both_cameras_see_the_trail_and_the_hud():
    # A camera behind a wall renders fine and shows nothing: the trail must change the picture.
    code = textwrap.dedent('''
        import json, jax, numpy as np
        from drones.sim.hover_env import HoverConfig, HoverEnv
        from drones.sim.render import TrajectoryRenderer
        env = HoverEnv(HoverConfig(num_envs=1))
        state, _ = env.reset(jax.random.key(4))
        pos = np.asarray(state.sim.states.pos[0, 0])
        trail = pos + np.linspace([-0.3, -0.3, 0.0], [0.3, 0.3, 0.0], 40)
        result = {}
        for camera in ('chase', 'top'):
            with TrajectoryRenderer(env, camera, 160, 120) as r:
                bare = r.frame(state).astype(int)
                result[camera] = {
                    'shape': list(bare.shape),
                    'trail': int((np.abs(r.frame(state, trail).astype(int) - bare).sum(-1) > 30).sum()),
                    'hud': int((np.abs(r.frame(state, hud=[('time', '1.0 s')]).astype(int) - bare).sum(-1) > 30).sum()),
                }
        print(json.dumps(result))
    ''')
    done = subprocess.run([sys.executable, '-c', code], env=gl_env(), capture_output=True,
                          text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    for camera, seen in result.items():
        assert seen['shape'] == [120, 160, 3], camera
        assert seen['trail'] > 20, f'{camera} camera cannot see the trail: {seen}'
        assert seen['hud'] > 20, f'{camera} camera shows no HUD text: {seen}'


@needs_gl
@pytest.mark.parametrize('camera, suffix', [('chase', '.mp4'), ('top', '.gif')])
def test_cli_writes_a_playable_video(saved_run, tmp_path, camera, suffix):
    import imageio.v2 as imageio

    out = tmp_path / f'flight{suffix}'
    done = subprocess.run(
        [sys.executable, '-m', 'drones.rl.render', str(saved_run), '--out', str(out),
         '--camera', camera, '--seconds', '0.4', '--width', '160', '--height', '128'],
        env=gl_env(), capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert f'Wrote {out}' in done.stdout
    frames = imageio.mimread(out, memtest=False)
    assert len(frames) >= 2
    assert frames[0].shape[:2] == (128, 160)
