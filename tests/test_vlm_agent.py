"""The VLM pilot in the simulator's agent loop, with a scripted backend and a fake view: how far
a chunk moves the drone, and that the question reaches the model."""
import json
import math
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones import config
from drones.sim import agents, eqa
from drones.vlm import agent as vlm_agent
from drones.vlm.actions import CHUNK

DONE = json.dumps({'done': True, 'answer': 'B'})


def moves(move, n=CHUNK):
    return json.dumps({'actions': [move] * n})


class FakeBackend:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, prompt, image):
        self.calls.append((prompt, image))
        return self.replies.pop(0)


class FakeView:
    """An image is the number of the render that made it, from 1."""

    def __init__(self):
        self.count = 0

    def render_matrix(self, pos, rotation):
        self.count += 1
        return np.full((2, 2, 3), self.count, np.uint8)


def run(*replies, space='discrete', question=None, yaw=math.pi / 2, **kwargs):
    backend = FakeBackend(*replies)
    agent = vlm_agent.make(action_space=space, backend=backend, **kwargs)
    start = agents.Pose(np.array([1.0, 2.0, 1.5]), yaw=yaw)
    seen = list(agents.episode(FakeView(), agent, start, question))
    return agent, backend, seen


def test_a_chunk_of_forward_moves_a_second_of_full_speed_along_the_heading():
    agent, backend, seen = run(moves('forward'), DONE)
    assert [o.step for o in seen] == list(range(CHUNK + 1))
    # Facing +y (yaw 90 degrees): 0.4 m/s for one second.
    assert seen[-1].pose.pos == pytest.approx([1.0, 2.0 + config.MAX_MANUAL_SPEED, 1.5])
    assert seen[1].pose.pos == pytest.approx([1.0, 2.0 + config.MAX_MANUAL_SPEED / CHUNK, 1.5])
    assert seen[-1].pose.yaw == pytest.approx(math.pi / 2)
    assert (seen[-1].pose.pitch, seen[-1].pose.roll) == (0.0, 0.0)
    assert agent.answer == 'B'
    assert agent.error is None


def test_the_model_is_shown_the_frame_of_each_chunks_first_step():
    _, backend, _ = run(moves('hover'), moves('hover'), DONE)
    assert [int(image[0, 0, 0]) for _, image in backend.calls] == [1, CHUNK + 1, 2 * CHUNK + 1]


def test_turn_left_turns_left_and_backward_goes_back():
    _, _, seen = run(moves('turn_left'), moves('backward'), DONE, yaw=0.0)
    turned = seen[CHUNK].pose
    assert turned.yaw == pytest.approx(math.radians(config.MAX_YAW_RATE))    # +yaw is left
    assert turned.pos == pytest.approx([1.0, 2.0, 1.5])
    # Now facing +y, so backward is -y.
    assert seen[-1].pose.pos == pytest.approx([1.0, 2.0 - config.MAX_MANUAL_SPEED, 1.5])


def test_rise_climbs_at_the_climb_limit_and_stops_at_the_ceiling():
    # The start is taken to be 1.0 m up; MAX_ALTITUDE is 2.0, four chunks of rise would be 1.2.
    _, _, seen = run(*[moves('rise')] * 4, DONE)
    heights = [o.pose.pos[2] - 1.5 for o in seen]
    assert heights[CHUNK] == pytest.approx(config.MAX_CLIMB_SPEED)
    assert heights[-1] == pytest.approx(config.MAX_ALTITUDE - 1.0)
    assert max(heights) <= config.MAX_ALTITUDE - 1.0 + 1e-9


def test_descend_stops_at_the_floor_limit():
    _, _, seen = run(*[moves('descend')] * 4, DONE)
    assert seen[-1].pose.pos[2] - 1.5 == pytest.approx(config.MIN_ALTITUDE - 1.0)


def test_start_altitude_sets_where_the_floor_is():
    _, backend, seen = run(*[moves('rise')] * 2, DONE, start_altitude=1.9)
    assert '1.90 m' in backend.calls[0][0]
    assert seen[-1].pose.pos[2] - 1.5 == pytest.approx(config.MAX_ALTITUDE - 1.9)


def test_a_continuous_altitude_is_approached_at_the_climb_limit_and_null_holds():
    up = json.dumps({'actions': [{'forward': 0.0, 'yaw': 0.0, 'altitude': 1.8}] * CHUNK})
    level = json.dumps({'actions': [{'forward': 0.5, 'yaw': 0.0, 'altitude': None}] * CHUNK})
    _, _, seen = run(up, level, DONE, space='continuous', yaw=0.0)
    assert seen[CHUNK].pose.pos[2] == pytest.approx(1.5 + config.MAX_CLIMB_SPEED)
    # Null does not go on climbing to 1.8: it stays where it is, as the mixer holds.
    assert seen[-1].pose.pos == pytest.approx(
        [1.0 + 0.5 * config.MAX_MANUAL_SPEED, 2.0, 1.5 + config.MAX_CLIMB_SPEED])


def test_the_question_and_its_choices_reach_the_model():
    q = eqa.Question('hm-eqa', 1, 'scene', 'What colour is the sofa?', 'B', 'object',
                     choices=('A) red', 'B) blue'))
    agent, backend, _ = run(DONE, question=q)
    prompt = backend.calls[0][0]
    assert 'What colour is the sofa?' in prompt and 'B) blue' in prompt
    assert agent.answer == 'B'


def test_without_a_question_it_explores():
    _, backend, _ = run(DONE)
    assert 'explore' in backend.calls[0][0]


def test_replies_that_never_validate_end_the_episode_with_the_error():
    agent, _, seen = run(*['nonsense'] * 6)
    assert len(seen) == 2 * CHUNK + 1            # two chunks of hover, then it gives up
    assert all(o.pose.pos == pytest.approx([1.0, 2.0, 1.5]) for o in seen)
    assert 'no JSON object' in agent.error
    assert agent.answer is None


def test_an_agent_can_fly_a_second_episode():
    backend = FakeBackend(moves('forward'), DONE, json.dumps({'done': True, 'answer': 'C'}))
    agent = vlm_agent.make(action_space='discrete', backend=backend)
    for z in (1.5, 4.0):
        start = agents.Pose(np.array([0.0, 0.0, z]))
        seen = list(agents.episode(FakeView(), agent, start))
    assert len(seen) == 1
    assert agent.answer == 'C'
    assert '1.00 m' in backend.calls[-1][0]       # the floor moved with the new start


@pytest.mark.parametrize('kwargs', [
    {'action_space': 'categorical'},
    {'start_altitude': 'high'},
    {'max_new_tokens': 'many'},
])
def test_a_wrong_argument_is_a_value_error(kwargs):
    # drones-render-agent turns a ValueError from the factory into a usage error.
    with pytest.raises(ValueError):
        vlm_agent.make(**kwargs)


def test_make_loads_no_model():
    agent = vlm_agent.make()
    assert agent.pilot.backend._pipe is None
    assert agent.pilot.space == 'continuous'


def test_each_model_call_is_reported_as_it_happens(capsys):
    # A call takes a minute or more on a CPU, and drones-render-agent prints nothing until the
    # film is written.
    run(moves('forward'), 'nonsense', 'nonsense', DONE)
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith('vlm:')]
    assert len(lines) == 3
    assert 'chunk 1 at 0 s' in lines[0] and 'valid' in lines[0]
    assert 'chunk 2 at 1 s' in lines[1] and 'failed' in lines[1]
    assert 'no JSON object' in lines[1]
    assert 'chunk 3 at 2 s' in lines[2] and 'done' in lines[2] and "'B'" in lines[2]


def test_giving_up_is_reported_with_the_error(capsys):
    run(*['nonsense'] * 6)
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith('vlm:')]
    assert len(lines) == 3
    assert 'stopping' in lines[-1] and 'no JSON object' in lines[-1]


def captions(*replies, space='discrete', question=None):
    """The agent's caption once it has acted on each frame, as a film shows it."""
    agent = vlm_agent.make(action_space=space, backend=FakeBackend(*replies))
    start = agents.Pose(np.array([1.0, 2.0, 1.5]))
    assert agent.caption == []
    return [agent.caption for _ in agents.decided(agents.episode(FakeView(), agent, start,
                                                                 question))]


def test_the_caption_shows_the_whole_chunk_and_marks_the_action_being_flown():
    shown = captions(moves('forward', 12)[:-2] + ', "turn_left", "turn_left", "turn_left", '
                     '"turn_left"]}', DONE)
    assert len(shown) == CHUNK + 1
    assert shown[0] == ['no question: explore',
                        'chunk 1 at 0 s   done: false   answer: -',
                        '[forward]' + ' forward' * 11 + ' turn_left' * 4]
    assert shown[5][2] == 'forward ' * 5 + '[forward]' + ' forward' * 6 + ' turn_left' * 4
    assert shown[15][2].endswith('turn_left [turn_left]')
    # The chunk that ended the episode is shown on the last frame, the one it was asked on.
    assert shown[16] == ['no question: explore', "chunk 2 at 1 s   done: true   answer: 'B'"]


def test_a_question_is_left_to_the_film_to_show():
    q = eqa.Question('hm-eqa', 1, 'scene', 'What colour is the sofa?', 'B', 'object')
    shown = captions(moves('hover'), DONE, question=q)
    assert shown[0][0] == 'chunk 1 at 0 s   done: false   answer: -'
    assert not any('no question' in line for lines in shown for line in lines)


def test_continuous_actions_are_captioned_as_forward_yaw_altitude():
    actions = ([{'forward': 0.5, 'yaw': -0.25, 'altitude': 1.4}]
               + [{'forward': 0.0, 'yaw': 1.0, 'altitude': None}] * (CHUNK - 1))
    shown = captions(json.dumps({'actions': actions}), DONE, space='continuous')
    assert shown[0][1] == 'chunk 1 at 0 s   done: false   answer: -   (forward/yaw/altitude)'
    assert shown[0][2] == '[+0.50/-0.25/1.40]' + ' +0.00/+1.00/-' * (CHUNK - 1)
    assert shown[1][2].startswith('+0.50/-0.25/1.40 [+0.00/+1.00/-] +0.00')


def test_a_failed_chunk_is_captioned_with_its_error():
    shown = captions(*['nonsense'] * 6)
    assert shown[0][1:] == ['chunk 1 at 0 s   FAILED, hovering: the reply holds no JSON object',
                            'hover x16']
    assert shown[CHUNK][1].startswith('chunk 2 at 1 s   FAILED, hovering')
    assert shown[-1][1:] == ['chunk 3 at 2 s   FAILED, stopping: the reply holds no JSON object']


FILM_ON_A_BOX = textwrap.dedent('''
    import json, sys
    import numpy as np
    from drones.sim import render_agent, scene_view, scenes
    from drones.vlm import agent as vlm_agent

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
    scene = scenes.Scene('box', [part], np.array([3.0, -1.0, 0.2]), 2.0)
    scene_view.load_scene = lambda name, dest=None: scene

    class Scripted:
        replies = [json.dumps({'actions': ['forward'] * 16}),
                   json.dumps({'done': True, 'answer': 'B'})]

        def generate(self, prompt, image):
            assert 'Find the sofa.' in prompt, prompt
            return self.replies.pop(0)

    def make():
        return vlm_agent.make(action_space='discrete', backend=Scripted())

    render_agent.main(['--scene', 'box', '--fov', '70', '--width', '80', '--height', '60',
                       '--agent', '__main__:make', '--ask', 'Find the sofa.', '--scale', '1',
                       '--out', sys.argv[1]])
''')


def gl_env():
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'egl')
    return env


def can_render():
    code = 'import mujoco; c = mujoco.GLContext(16, 16); c.make_current(); c.free()'
    return subprocess.run([sys.executable, '-c', code], env=gl_env(),
                          capture_output=True).returncode == 0


# Starting the film's interpreter forks, which Python warns about once JAX has threads.
@pytest.mark.filterwarnings(r'ignore:os\.fork\(\) was called:RuntimeWarning')
@pytest.mark.skipif(not can_render(), reason='no offscreen OpenGL (MUJOCO_GL=egl)')
def test_the_film_asks_the_question_and_captions_every_frame_with_the_chunk(tmp_path):
    import imageio.v2 as imageio

    from drones.sim import render_agent

    out = tmp_path / 'vlm.mp4'
    done = subprocess.run([sys.executable, '-c', FILM_ON_A_BOX, str(out)], env=gl_env(),
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert f'Wrote {out}: {CHUNK + 1} frames' in done.stdout
    assert 'question: Find the sofa.' in done.stdout
    assert 'agent answered: B' in done.stdout
    frames = imageio.mimread(out, memtest=False)
    assert len(frames) == CHUNK + 1
    # 60 rows of picture, then 15-row lines: the step, two for the question, the agent's panel.
    rows = render_agent.CAPTION_LINES + render_agent.AGENT_LINES
    height = 60 + rows * 15 + 4
    assert frames[0].shape[:2] == (height + height % 2, 160)     # mp4 wants even sizes
    panel = slice(60 + render_agent.CAPTION_LINES * 15, None)
    assert frames[0][panel].max() > 150, 'nothing is written in the agent panel'
    # The last frame shows the chunk that ended the episode, not the one before it.
    change = np.abs(frames[-1][panel].astype(int) - frames[0][panel].astype(int))
    assert change.max() > 100


def test_the_longest_caption_fits_the_films_panel():
    # The deck's lens at the default scale gives the film 98 characters a line. Every action at
    # its longest, and no question, is the most the panel has to hold.
    from drones.sim import render_agent

    actions = [{'forward': -0.25, 'yaw': -0.75, 'altitude': 1.25}] * CHUNK
    shown = captions(json.dumps({'actions': actions}), DONE, space='continuous')[0]
    observation = agents.Observation(None, agents.Pose(np.zeros(3)), 0, None)
    lines = render_agent.caption(observation, '', 98, shown)
    assert sum(line.count('1.25') for line in lines) == CHUNK, 'actions are cut off'
