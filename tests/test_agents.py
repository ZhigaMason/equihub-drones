"""Agents in scanned scenes: the built-ins, the loop that runs one, and drones-render-agent.

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

from drones.sim import agents, eqa

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


class FakeView:
    """Records where it was asked to look; an image is its call number."""

    def __init__(self):
        self.poses = []

    def render_matrix(self, pos, rotation):
        self.poses.append((np.array(pos), rotation))
        return np.full((2, 2, 3), len(self.poses), np.uint8)


def test_look_around_turns_once_in_place_and_stops():
    start = agents.Pose(np.array([1.0, 2.0, 1.0]), yaw=0.5)
    seen = list(agents.episode(FakeView(), agents.LookAround(steps=4), start))
    assert [o.step for o in seen] == [0, 1, 2, 3, 4]
    assert [o.pose.yaw for o in seen] == pytest.approx([0.5 + i * math.pi / 2 for i in range(5)])
    assert all(np.array_equal(o.pose.pos, start.pos) for o in seen)


def test_follow_path_flies_the_reference_path():
    path = np.array([[0.0, 0, 1], [1, 0, 1], [1, 1, 1]])
    q = eqa.Question('indoor-uav', 1, 'scene', 'fly', 'detail', 'nav', start=path[0], yaw=0.0,
                     path=path)
    view = FakeView()
    seen = list(agents.episode(view, agents.FollowPath(), agents.Pose(path[0]), q))
    np.testing.assert_allclose([o.pose.pos for o in seen], path)
    assert seen[1].question is q
    # The image an agent gets is the view's render at the pose it is shown with.
    assert [int(o.image[0, 0, 0]) for o in seen] == [1, 2, 3]
    assert [int(o.image[0, 0, 0]) for o in agents.episode(
        FakeView(), agents.FollowPath(), agents.Pose(path[0]))] == [1]   # no question, no path


def test_episode_stops_after_max_steps():
    seen = list(agents.episode(FakeView(), agents.LookAround(steps=100),
                               agents.Pose(np.zeros(3)), max_steps=3))
    assert len(seen) == 4


def test_agents_load_by_name_or_import_path():
    assert isinstance(agents.make_agent('look-around', steps=3), agents.LookAround)
    agent = agents.make_agent('drones.sim.agents:LookAround', steps=2)
    assert isinstance(agent, agents.LookAround) and agent.steps == 2
    with pytest.raises(ValueError, match='package.module:factory'):
        agents.make_agent('nonsense')


def test_decided_yields_each_frame_once_the_agent_has_acted_on_it():
    # A film captions a frame with what the agent made of it, the last frame included.
    class Counting:
        def reset(self, question, pose):
            self.acts = 0

        def act(self, observation):
            self.acts += 1
            return observation.pose if self.acts < 3 else None

    agent = Counting()
    start = agents.Pose(np.zeros(3))
    seen = [(o.step, agent.acts)
            for o in agents.decided(agents.episode(FakeView(), agent, start))]
    assert seen == [(0, 1), (1, 2), (2, 3)]
    assert list(agents.decided(iter(()))) == []


def test_caption_keeps_the_question_to_two_lines_and_puts_the_agents_lines_under_it():
    from drones.sim import render_agent

    observation = agents.Observation(None, agents.Pose(np.zeros(3)), 4, None)
    lines = render_agent.caption(observation, 'Find the sofa.', 40,
                                 ['chunk 1', 'forward ' * 40])
    assert lines[0].startswith('step 4')
    assert lines[1:4] == ['Find the sofa.', '', 'chunk 1']
    assert len(lines) == render_agent.CAPTION_LINES + render_agent.AGENT_LINES
    assert all(len(line) <= 40 for line in lines[1:])
    assert lines[4] == ('forward ' * 5).strip()
    # Without agent lines the caption is the three it always was.
    assert len(render_agent.caption(observation, '', 40)) == render_agent.CAPTION_LINES


def test_ask_goes_with_a_scene_not_a_benchmark(capsys):
    from drones.sim import render_agent

    with pytest.raises(SystemExit):
        render_agent.main(['--benchmark', 'hm-eqa', '--ask', 'Find the sofa.'])
    assert '--ask goes with --scene' in capsys.readouterr().err


CLI_ON_A_BOX = textwrap.dedent('''
    import json, sys
    import numpy as np
    from drones.sim import scene_view, scenes
    from drones.sim.lens import Intrinsics
    from drones.sim.agents import Pose

    def box(origin):
        """A 4 x 4 x 2.5 m room around the origin's floor, faces inward, one colour per wall."""
        lo, hi = np.array([-2.0, -2.0, 0.0]), np.array([2.0, 2.0, 2.5])
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                 [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
        texture = np.array([[[200, 60, 60], [60, 200, 60]], [[60, 60, 200], [200, 200, 60]]],
                           np.uint8)
        uv = np.array([[0.25 + 0.5 * (i % 2), 0.25 + 0.5 * (i // 4 % 2)] for i in range(8)],
                      np.float32)
        part = scenes.Part(corners.astype(np.float32), np.array(faces, np.int32), uv, texture)
        return scenes.Scene('box', [part], np.asarray(origin, float), 2.0)

    scene = box([3.0, -1.0, 0.2])
    scene_view.load_scene = lambda name, dest=None: scene

    # The drone shows in a chase view of it, and never in the view through its own lens.
    k = Intrinsics.from_fov(64, 48, 1.2)
    pose = Pose(scene.origin + [0.0, 0.0, 1.0], yaw=0.3)
    views = {}
    for drone in (scene_view.DRONE, None):   # the same shots, with and without the drone
        with scene_view.SceneView(scene, k, drone=drone) as view:
            chase = scene_view.ChaseCamera(view, 128, 96, distance=0.6)
            views[drone] = (view.render_matrix(pose.pos, pose.rotation).astype(int),
                            chase.render(pose.pos, pose.rotation).astype(int))
    (own, behind), (own_empty, behind_empty) = views[scene_view.DRONE], views[None]
    result = {'own_diff': int(np.abs(own - own_empty).max()),
              'chase_changed': int((np.abs(behind - behind_empty).max(axis=2) > 30).sum())}

    from drones.sim import render_agent
    out = sys.argv[1]
    render_agent.main(['--scene', 'box', '--fov', '70', '--width', '80', '--height', '60',
                       '--agent', 'look-around', '--agent-arg', 'steps=3', '--scale', '1',
                       '--out', out])
    print(json.dumps(result))
''')


@needs_gl
def test_cli_films_chase_and_agent_camera_side_by_side(tmp_path):
    import imageio.v2 as imageio

    out = tmp_path / 'agent.mp4'
    done = subprocess.run([sys.executable, '-c', CLI_ON_A_BOX, str(out)], env=gl_env(),
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result['own_diff'] <= 2, 'the agent camera sees its own drone'
    assert result['chase_changed'] > 20, 'the chase camera does not show the drone'
    assert f'Wrote {out}: 4 frames' in done.stdout
    frames = imageio.mimread(out, memtest=False)
    assert len(frames) == 4
    # 80 wide chase (4:3 of 60) beside the 80 x 60 agent camera, a caption under them.
    assert frames[0].shape[1] == 160 and frames[0].shape[0] > 60
