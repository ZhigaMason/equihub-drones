"""The VLM pilot's prompt: the question, the state the camera cannot show, and the format."""
import pytest

from drones.vlm.actions import CHUNK, MOVES, SPACES, parse_chunk
from drones.vlm.prompt import build_prompt, retry_prompt


@pytest.mark.parametrize('space', SPACES)
@pytest.mark.parametrize('choices', [(), ('A) red', 'B) blue')])
def test_nothing_in_the_prompt_would_pass_as_a_reply(space, choices):
    # A small model given a complete example returned that example for every frame. The format
    # is shown as a template instead, which is not itself a chunk, done or not.
    for line in build_prompt(space, 'Find the sofa.', choices).splitlines():
        with pytest.raises(ValueError):
            parse_chunk(line, space)


@pytest.mark.parametrize('space, slot', [('discrete', 'm'), ('continuous', 'a')])
def test_the_template_has_a_slot_for_each_of_the_sixteen_actions(space, slot):
    prompt = build_prompt(space, 'Find the sofa.')
    slots = ', '.join(f'{slot}{i}' for i in range(1, CHUNK + 1))
    assert '{"actions": [' + slots + '], "done": false, "answer": null}' in prompt
    assert '"done": true' in prompt


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


def test_the_discrete_action_space_is_described_move_by_move():
    # At the shipped tuning: 0.4 m/s, 90 deg/s and 0.3 m/s over 1/16 s.
    prompt = build_prompt('discrete', 'Find the sofa.')
    assert 'ACTION SPACE' in prompt
    assert f'{len(MOVES)} moves' in prompt
    described = [line for line in prompt.splitlines() if line.startswith('- "')]
    assert [line.split('"')[1] for line in described] == list(MOVES)     # one line each
    assert '2.5 cm' in prompt and '5.6 degrees' in prompt and '1.9 cm' in prompt
    # What a move does to the picture, and what a whole chunk of it adds up to.
    assert 'left of the image' in prompt and 'right of the image' in prompt
    assert '40 cm' in prompt and '90 degrees' in prompt
    assert '"altitude"' not in prompt


def test_the_continuous_action_space_is_described_field_by_field():
    prompt = build_prompt('continuous', 'Find the sofa.')
    assert 'ACTION SPACE' in prompt
    assert '{"forward": F, "yaw": Y, "altitude": H}' in prompt
    assert '2.5 cm' in prompt and '5.6 degrees' in prompt and '1.9 cm' in prompt
    assert 'from -1 to 1' in prompt
    assert 'from 0.2 to 2 m' in prompt and 'null' in prompt
    assert '40 cm' in prompt and '90 degrees' in prompt
    assert '"turn_left"' not in prompt


@pytest.mark.parametrize('space', SPACES)
def test_the_prompt_says_to_choose_from_the_image(space):
    prompt = build_prompt(space, 'Find the sofa.')
    assert 'HOW TO CHOOSE' in prompt
    assert 'Look at the image' in prompt


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
