"""The VLM pilot against a scripted backend: when it asks, what it does with a bad reply, and
when it stops."""
import json

import pytest

from drones.control.mixer import Command
from drones.vlm.actions import CHUNK
from drones.vlm.pilot import Pilot

FORWARD = json.dumps({'actions': ['forward'] * CHUNK})
LEFT = json.dumps({'actions': ['turn_left'] * CHUNK})
DONE = json.dumps({'actions': [], 'done': True, 'answer': 'B'})
BAD = json.dumps({'actions': ['forward'] * 3})


class FakeBackend:
    """Replies from a script, and records each (prompt, image) it was asked with."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, prompt, image):
        self.calls.append((prompt, image))
        return self.replies.pop(0)


def fly(pilot, steps, altitude=1.0):
    """Commands for `steps` steps; the image of step i is the number i."""
    return [pilot.step(i, altitude) for i in range(steps)]


def started(backend, space='discrete', **kwargs):
    pilot = Pilot(backend, space, **kwargs)
    pilot.reset('Find the sofa.')
    return pilot


def test_one_query_per_chunk_with_the_frame_of_that_step():
    backend = FakeBackend(FORWARD, LEFT, DONE)
    pilot = started(backend)
    commands = fly(pilot, 2 * CHUNK + 1)
    assert commands[:CHUNK] == [Command(forward=1.0)] * CHUNK
    assert commands[CHUNK:2 * CHUNK] == [Command(yaw=1.0)] * CHUNK
    assert commands[-1] is None
    assert [image for _, image in backend.calls] == [0, CHUNK, 2 * CHUNK]
    assert pilot.queries == 3
    assert pilot.stats == {'first': 3, 'retry': 0, 'failed': 0}


def test_the_prompt_carries_the_question_altitude_and_elapsed_time():
    backend = FakeBackend(FORWARD, DONE)
    pilot = Pilot(backend, 'discrete')
    pilot.reset('What colour is the sofa?', ('A) red', 'B) blue'))
    fly(pilot, CHUNK + 1, altitude=1.5)
    first, second = (prompt for prompt, _ in backend.calls)
    assert 'What colour is the sofa?' in first and 'B) blue' in first
    assert '1.50 m' in first
    assert 'flown for 0 s' in first
    assert 'flown for 1 s' in second


def test_done_stops_at_once_and_keeps_the_answer():
    backend = FakeBackend(json.dumps({'actions': ['forward'] * 5, 'done': True, 'answer': 'B'}))
    pilot = started(backend)
    assert pilot.step(0, 1.0) is None       # the five actions of a done chunk are not flown
    assert pilot.answer == 'B'
    assert pilot.error is None
    assert pilot.step(1, 1.0) is None       # and it stays stopped, without asking again
    assert pilot.queries == 1


def test_an_answer_given_before_done_is_kept():
    early = json.dumps({'actions': ['hover'] * CHUNK, 'answer': 'a sofa'})
    pilot = started(FakeBackend(early, json.dumps({'done': True})))
    fly(pilot, CHUNK + 1)
    assert pilot.answer == 'a sofa'


def test_an_invalid_reply_is_retried_once_with_the_error():
    backend = FakeBackend(BAD, FORWARD, DONE)
    pilot = started(backend)
    assert fly(pilot, CHUNK) == [Command(forward=1.0)] * CHUNK
    retry, image = backend.calls[1]
    assert image == 0                        # the same frame
    assert BAD in retry and 'exactly 16' in retry
    assert pilot.stats == {'first': 0, 'retry': 1, 'failed': 0}
    assert pilot.error is None


def test_two_invalid_replies_hover_for_a_chunk():
    backend = FakeBackend(BAD, 'no json here', FORWARD)
    pilot = started(backend)
    assert fly(pilot, CHUNK) == [Command()] * CHUNK
    assert 'no JSON object' in pilot.error
    assert pilot.step(CHUNK, 1.0) == Command(forward=1.0)      # and then it asks again
    assert pilot.stats == {'first': 1, 'retry': 0, 'failed': 1}
    assert pilot.error is None


def test_three_failed_chunks_in_a_row_end_the_episode():
    backend = FakeBackend(*[BAD] * 6)
    pilot = started(backend)
    commands = fly(pilot, 2 * CHUNK + 1)
    assert commands[:2 * CHUNK] == [Command()] * (2 * CHUNK)
    assert commands[-1] is None
    assert 'exactly 16' in pilot.error
    assert pilot.queries == 6
    assert pilot.step(99, 1.0) is None
    assert pilot.queries == 6


def test_a_valid_chunk_resets_the_failure_count():
    backend = FakeBackend(BAD, BAD, BAD, BAD, FORWARD, BAD, BAD, BAD, BAD, DONE)
    pilot = started(backend)
    commands = fly(pilot, 5 * CHUNK + 1)
    assert None not in commands[:5 * CHUNK]
    assert commands[-1] is None
    assert pilot.answer == 'B'
    assert pilot.stats == {'first': 2, 'retry': 0, 'failed': 4}


def test_continuous_chunks_become_their_commands():
    action = {'forward': 0.5, 'yaw': -0.25, 'altitude': 1.4}
    backend = FakeBackend(json.dumps({'actions': [action] * CHUNK}))
    pilot = started(backend, 'continuous')
    assert fly(pilot, CHUNK) == [Command(0.5, -0.25, 1.4)] * CHUNK


def test_rise_is_relative_to_the_altitude_at_each_step():
    pilot = started(FakeBackend(json.dumps({'actions': ['rise'] * CHUNK})))
    first = pilot.step(0, 1.0)
    second = pilot.step(1, 1.5)
    assert second.altitude - 1.5 == pytest.approx(first.altitude - 1.0)
    assert first.altitude > 1.0


def test_reset_starts_a_clean_episode():
    backend = FakeBackend(FORWARD, BAD, BAD, LEFT)
    pilot = started(backend)
    fly(pilot, 3)                            # 13 actions of the first chunk are left over
    pilot.reset('Find the door.')
    fly(pilot, CHUNK)                        # a failed chunk: hover, error set
    assert pilot.error is not None
    pilot.reset('Find the window.')
    assert (pilot.answer, pilot.error, pilot.queries) == (None, None, 0)
    assert pilot.stats == {'first': 0, 'retry': 0, 'failed': 0}
    assert pilot.step(0, 1.0) == Command(yaw=1.0)
    assert 'Find the window.' in backend.calls[-1][0]
    assert 'flown for 0 s' in backend.calls[-1][0]


def test_an_unknown_action_space_fails_when_the_pilot_is_built():
    with pytest.raises(ValueError, match='continuous, discrete'):
        Pilot(FakeBackend(), 'categorical')


def test_step_before_reset_explores():
    backend = FakeBackend(FORWARD)
    assert Pilot(backend, 'discrete').step(0, 1.0) == Command(forward=1.0)
    assert 'explore' in backend.calls[0][0]
