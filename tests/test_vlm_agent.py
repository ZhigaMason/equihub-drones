"""The VLM pilot in the simulator's agent loop, with a scripted backend and a fake view: how far
a chunk moves the drone, and that the question reaches the model."""
import json
import math

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
