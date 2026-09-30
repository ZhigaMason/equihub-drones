"""The VLM pilot's prompt: the question, the state the camera cannot show, and the format."""
import pytest

from drones.vlm.actions import MOVES, SPACES, parse_chunk
from drones.vlm.prompt import build_prompt, example, retry_prompt


@pytest.mark.parametrize('space', SPACES)
def test_the_example_in_the_prompt_is_a_valid_chunk(space):
    assert example(space) in build_prompt(space, 'Find the sofa.')
    assert len(parse_chunk(example(space), space).actions) == 16


def test_the_question_choices_and_state_are_in_the_prompt():
    prompt = build_prompt('discrete', 'What colour is the sofa?', ('A) red', 'B) blue'),
                          altitude=1.25, elapsed=7.0)
    assert 'What colour is the sofa?' in prompt
    assert 'A) red' in prompt and 'B) blue' in prompt
    assert 'letter' in prompt
    assert '1.25 m' in prompt
    assert '7 s' in prompt


@pytest.mark.parametrize('question', [None, ''])
def test_without_a_question_the_task_is_to_explore(question):
    prompt = build_prompt('continuous', question)
    assert 'explore' in prompt
    assert 'None' not in prompt
    assert 'letter' not in prompt


def test_each_mode_describes_its_own_actions_in_physical_units():
    # At the shipped tuning: 0.4 m/s and 90 deg/s over 1/16 s.
    discrete = build_prompt('discrete', 'Find the sofa.')
    assert all(f'"{move}"' in discrete for move in MOVES)
    assert '2.5 cm' in discrete and '5.6 degrees' in discrete
    assert '"altitude"' not in discrete

    continuous = build_prompt('continuous', 'Find the sofa.')
    assert '"forward": F' in continuous
    assert '2.5 cm' in continuous and '5.6 degrees' in continuous
    assert '"turn_left"' not in continuous


def test_an_unknown_action_space_is_a_value_error():
    with pytest.raises(ValueError, match='continuous, discrete'):
        build_prompt('categorical')


def test_a_retry_shows_the_rejected_reply_and_the_error_at_bounded_length():
    prompt = build_prompt('discrete', 'Find the sofa.')
    again = retry_prompt(prompt, '{"actions": ["forward"]}', 'a chunk has exactly 16 actions')
    assert again.startswith(prompt)
    assert '{"actions": ["forward"]}' in again
    assert 'a chunk has exactly 16 actions' in again
    # A runaway reply or a 16-line pydantic error must not double the next prompt.
    long = retry_prompt(prompt, 'x' * 100_000, 'e' * 100_000)
    assert len(long) < len(prompt) + 2_500
