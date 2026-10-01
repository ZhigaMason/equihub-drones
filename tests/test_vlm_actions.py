"""The VLM pilot's action schema: what a chunk may hold, and the Command each action becomes."""
import json

import pytest
from pydantic import ValidationError

from drones import config
from drones.control.mixer import Command
from drones.vlm.actions import (CHUNK, MOVES, STEP, ContinuousAction, ContinuousChunk,
                                DiscreteChunk, parse_chunk, schema_for, to_command)


def continuous(n=CHUNK, action=None, **extra):
    action = action or {'forward': 0.5, 'yaw': -0.25, 'altitude': None}
    return json.dumps({'actions': [action] * n, **extra})


def discrete(n=CHUNK, move='forward', **extra):
    return json.dumps({'actions': [move] * n, **extra})


def test_a_full_chunk_of_either_kind_validates():
    chunk = parse_chunk(continuous(), 'continuous')
    assert isinstance(chunk, ContinuousChunk)
    assert len(chunk.actions) == CHUNK == 16
    assert chunk.actions[0] == ContinuousAction(forward=0.5, yaw=-0.25, altitude=None)
    assert (chunk.done, chunk.answer) == (False, None)

    chunk = parse_chunk(discrete(), 'discrete')
    assert isinstance(chunk, DiscreteChunk)
    assert chunk.actions == ['forward'] * 16


@pytest.mark.parametrize('n', [0, 15, 17])
def test_a_chunk_that_is_not_done_needs_exactly_sixteen_actions(n):
    with pytest.raises(ValidationError, match='exactly 16'):
        parse_chunk(discrete(n), 'discrete')
    with pytest.raises(ValidationError, match='exactly 16'):
        parse_chunk(continuous(n), 'continuous')


def test_a_done_chunk_may_be_short_or_empty_but_not_long():
    assert parse_chunk('{"done": true, "answer": "B"}', 'discrete').actions == []
    assert len(parse_chunk(discrete(3, done=True), 'discrete').actions) == 3
    assert parse_chunk(continuous(0, done=True, answer='a sofa'), 'continuous').answer == 'a sofa'
    with pytest.raises(ValidationError, match='at most 16'):
        parse_chunk(discrete(17, done=True), 'discrete')


@pytest.mark.parametrize('action', [
    {'forward': 1.2, 'yaw': 0.0, 'altitude': None},
    {'forward': 0.0, 'yaw': -1.01, 'altitude': None},
])
def test_out_of_range_is_an_error_not_clamped(action):
    with pytest.raises(ValidationError):
        parse_chunk(continuous(action=action), 'continuous')


@pytest.mark.parametrize('action', [
    '{"forward": NaN, "yaw": 0.0, "altitude": null}',
    '{"forward": 0.0, "yaw": 0.0, "altitude": NaN}',
    '{"forward": 0.0, "yaw": 0.0, "altitude": Infinity}',
    '{"forward": 0.0, "yaw": 0.0, "altitude": 1e999}',
])
def test_nan_and_infinity_are_rejected(action):
    text = '{"actions": [' + ', '.join([action] * CHUNK) + ']}'
    with pytest.raises(ValueError):
        parse_chunk(text, 'continuous')


def test_unknown_fields_and_moves_are_rejected():
    with pytest.raises(ValidationError):
        parse_chunk(continuous(action={'forward': 0, 'yaw': 0, 'speed': 1}), 'continuous')
    with pytest.raises(ValidationError):
        parse_chunk(continuous(thoughts='I see a door'), 'continuous')
    with pytest.raises(ValidationError):
        parse_chunk(discrete(move='go_down'), 'discrete')
    with pytest.raises(ValidationError):
        parse_chunk(continuous(action={'yaw': 0.0}), 'continuous')     # forward is required


def test_the_two_kinds_are_not_interchangeable():
    with pytest.raises(ValidationError):
        parse_chunk(discrete(), 'continuous')
    with pytest.raises(ValidationError):
        parse_chunk(continuous(), 'discrete')


def test_a_continuous_action_maps_field_for_field():
    action = ContinuousAction(forward=0.5, yaw=-0.25, altitude=1.4)
    assert to_command(action, altitude=1.0) == Command(0.5, -0.25, 1.4)
    # None holds the current altitude, as the mixer reads it.
    held = ContinuousAction(forward=0.0, yaw=0.0)
    assert to_command(held, altitude=1.0) == Command(0.0, 0.0, None)


def test_each_move_is_a_full_scale_command_for_one_step():
    climb = config.MAX_CLIMB_SPEED * STEP
    expected = {
        'forward': Command(forward=1.0),
        'backward': Command(forward=-1.0),
        'turn_left': Command(yaw=1.0),       # +yaw turns left
        'turn_right': Command(yaw=-1.0),
        'rise': Command(altitude=1.0 + climb),
        'descend': Command(altitude=1.0 - climb),
        'hover': Command(),
    }
    assert set(expected) == set(MOVES)
    for move, command in expected.items():
        assert to_command(move, altitude=1.0) == command, move


@pytest.mark.parametrize('wrap', [
    '{}',
    '```json\n{}\n```',
    'Here is my plan:\n{}\nI will look for the sofa.',
])
def test_parse_chunk_finds_the_json_in_a_fenced_or_prefaced_reply(wrap):
    chunk = parse_chunk(wrap.replace('{}', discrete()), 'discrete')
    assert chunk.actions == ['forward'] * CHUNK


def test_parse_chunk_rejects_a_reply_that_is_cut_off():
    # What max_new_tokens leaves behind: the object never closes.
    cut = continuous()[:-40]
    with pytest.raises(ValueError):
        parse_chunk(cut, 'continuous')


@pytest.mark.parametrize('text', ['', 'I cannot see anything.', '} {'])
def test_parse_chunk_rejects_a_reply_without_json(text):
    with pytest.raises(ValueError, match='no JSON object'):
        parse_chunk(text, 'discrete')


def test_an_unknown_action_space_names_the_valid_ones():
    with pytest.raises(ValueError, match='continuous, discrete'):
        schema_for('categorical')
    with pytest.raises(ValueError, match='continuous, discrete'):
        parse_chunk(discrete(), 'categorical')


@pytest.mark.parametrize('action', [
    {'forward': True, 'yaw': 0.0},
    {'forward': '0.5', 'yaw': 0.0},
    {'forward': 0.0, 'yaw': 0.0, 'altitude': True},
    {'forward': 0.0, 'yaw': 0.0, 'altitude': '1.2'},
])
def test_a_value_of_the_wrong_type_is_rejected_not_coerced(action):
    # pydantic's default would fly `true` as full speed ahead.
    with pytest.raises(ValidationError):
        parse_chunk(continuous(action=action), 'continuous')


@pytest.mark.parametrize('done', ['"true"', '1'])
def test_done_must_be_a_boolean(done):
    with pytest.raises(ValidationError):
        parse_chunk('{"done": %s, "answer": "B"}' % done, 'discrete')


def test_whole_numbers_are_still_numbers():
    chunk = parse_chunk(continuous(action={'forward': 1, 'yaw': -1, 'altitude': 1}), 'continuous')
    assert chunk.actions[0] == ContinuousAction(forward=1.0, yaw=-1.0, altitude=1.0)


@pytest.mark.parametrize('space', ['continuous', 'discrete'])
def test_the_grammar_is_the_pydantic_schema_with_the_count_pinned(space):
    from drones.vlm.actions import ANSWER_MAX, SCHEMAS, json_schema

    own = SCHEMAS[space].model_json_schema()
    grammar = json_schema(space)
    actions = grammar['properties']['actions']
    assert (actions['type'], actions['minItems'], actions['maxItems']) == ('array', CHUNK, CHUNK)
    assert actions['items'] == own['properties']['actions']['items']
    assert grammar['required'] == ['actions', 'done', 'answer']
    assert grammar['additionalProperties'] is False
    assert {'type': 'string', 'maxLength': ANSWER_MAX} in grammar['properties']['answer']['anyOf']
    assert grammar.get('$defs') == own.get('$defs')
    # The model's own schema is not what got edited.
    assert 'minItems' not in SCHEMAS[space].model_json_schema()['properties']['actions']


def test_the_grammar_of_an_unknown_action_space_names_the_valid_ones():
    from drones.vlm.actions import json_schema

    with pytest.raises(ValueError, match='continuous, discrete'):
        json_schema('categorical')


def test_a_chunk_of_another_size_validates_against_that_size():
    assert len(parse_chunk(discrete(8), 'discrete', size=8).actions) == 8
    with pytest.raises(ValidationError, match='exactly 8'):
        parse_chunk(discrete(), 'discrete', size=8)
    assert len(parse_chunk(discrete(32), 'discrete', size=32).actions) == 32
    assert parse_chunk(discrete(3, done=True), 'discrete', size=8).done
    with pytest.raises(ValidationError, match='at most 8'):
        parse_chunk(discrete(9, done=True), 'discrete', size=8)


@pytest.mark.parametrize('size', [1, 8, 32])
def test_the_grammar_pins_the_count_to_the_chunk_size(size):
    from drones.vlm.actions import json_schema

    actions = json_schema('continuous', size)['properties']['actions']
    assert (actions['minItems'], actions['maxItems']) == (size, size)


@pytest.mark.parametrize('given, size', [(1, 1), (16, 16), (32, 32), ('8', 8), (8.0, 8)])
def test_a_chunk_size_is_a_whole_number_from_one_to_thirty_two(given, size):
    from drones.vlm.actions import chunk_size

    assert chunk_size(given) == size


@pytest.mark.parametrize('given', [0, 33, -4, 8.5, 'many', None, True, 'inf', '1e400', 'nan'])
def test_any_other_chunk_size_is_a_value_error_naming_the_range(given):
    from drones.vlm.actions import chunk_size

    with pytest.raises(ValueError, match='1 to 32'):
        chunk_size(given)
