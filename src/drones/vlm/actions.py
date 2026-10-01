"""What a VLM pilot may say: a chunk of 16 actions, a completion flag and an answer.

The model writes JSON and nothing it writes is executed before pydantic has accepted all of it. A
chunk is rejected whole, never repaired: a `forward` of 1.2 or a fifteenth action missing is a
model that did not follow the format, and the pilot asks again rather than guess what was meant.

There are two kinds of chunk, because it is an open question which a model flies better with:
`continuous` actions are a `control.mixer.Command` field for field, `discrete` ones are named
moves. Both become a Command, so everything after this module is shared, and a Command is what
the teleop safety layer already takes on the real drone.
"""
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from drones import config
from drones.control.mixer import Command

CHUNK = 16          # actions per chunk: one model call covers one second
STEP = 1 / CHUNK    # s each action lasts
ANSWER_MAX = 300    # characters of an answer a constrained model may write

Move = Literal['forward', 'backward', 'turn_left', 'turn_right', 'rise', 'descend', 'hover']
MOVES = get_args(Move)


class _Strict(BaseModel):
    # extra='forbid': a misspelt key must be an error, not a silently defaulted field.
    # allow_inf_nan=False: JSON parsers accept NaN and Infinity, and a NaN altitude would reach
    # the pose.
    # strict=True: pydantic otherwise coerces, and would fly `"forward": true` as full speed
    # ahead and end the episode on `"done": "true"`. A whole number is still a float.
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, strict=True)


class ContinuousAction(_Strict):
    forward: float = Field(ge=-1, le=1)    # x config.MAX_MANUAL_SPEED, + is ahead
    yaw: float = Field(ge=-1, le=1)        # x config.MAX_YAW_RATE, + turns left
    # An absolute target in metres; None holds the current altitude. Not range-checked here: the
    # envelope is operator tuning, and whoever executes the Command clamps to it.
    altitude: float | None = None


class _Chunk(_Strict):
    done: bool = False          # stop now; this chunk's actions are not executed
    answer: str | None = None   # the reply to the question, usually given with `done`

    @model_validator(mode='after')
    def _count(self):
        n = len(self.actions)
        if self.done and n > CHUNK:
            raise ValueError(f'a done chunk has at most {CHUNK} actions, not {n}')
        if not self.done and n != CHUNK:
            raise ValueError(f'a chunk has exactly {CHUNK} actions, not {n}')
        return self


class ContinuousChunk(_Chunk):
    actions: list[ContinuousAction] = []


class DiscreteChunk(_Chunk):
    actions: list[Move] = []


SCHEMAS = {'continuous': ContinuousChunk, 'discrete': DiscreteChunk}
SPACES = tuple(SCHEMAS)


def schema_for(space):
    """The chunk model for action space `space`."""
    if space not in SCHEMAS:
        raise ValueError(f'action space {space!r} is not one of {", ".join(SPACES)}')
    return SCHEMAS[space]


def json_schema(space):
    """The JSON schema of a chunk in action space `space`, for a backend that can hold a model
    to one while it writes (constrained decoding).

    It is the pydantic model's own schema, narrowed to what lm-format-enforcer can hold a model
    to exactly. `actions` has exactly CHUNK items, so a model cannot stop at fifteen; that
    includes a done chunk, whose actions are then not flown. (The schema also allows a done
    chunk fewer, but "CHUNK, or none" cannot be stated: the enforcer's `maxItems` of 0 admits
    one item, and a `const` of false for `done` crashes it.) Every key is required, so none is
    left to a default, and an answer has a length, or a model that never closes the string
    writes to the token limit.

    One rule stays with `parse_chunk` alone: forward and yaw lie between -1 and 1. The enforcer
    ignores minimum and maximum.
    """
    schema = schema_for(space).model_json_schema()      # a fresh dict each call
    items = schema['properties']['actions']['items']
    schema['properties']['actions'] = {'type': 'array', 'items': items,
                                       'minItems': CHUNK, 'maxItems': CHUNK}
    schema['properties']['answer'] = {'anyOf': [{'type': 'string', 'maxLength': ANSWER_MAX},
                                                {'type': 'null'}]}
    # Also the order the keys are written in, which the backend enforces.
    schema['required'] = ['actions', 'done', 'answer']
    return schema


def parse_chunk(text, space):
    """The validated chunk in a model's reply `text`. Raises ValueError (pydantic's
    ValidationError is one) when there is no JSON object or it does not fit the schema."""
    schema = schema_for(space)
    # From the first brace to the last, so a fenced or prefaced reply still parses. A reply cut
    # off at the token limit then fails as invalid JSON, which is what it is.
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise ValueError('the reply holds no JSON object')
    return schema.model_validate_json(text[start:end + 1])


def to_command(action, altitude):
    """The Command for one action, a ContinuousAction or a move's name, with the drone at
    `altitude` (m)."""
    if isinstance(action, ContinuousAction):
        return Command(action.forward, action.yaw, action.altitude)
    climb = config.MAX_CLIMB_SPEED * STEP
    return {
        'forward': Command(forward=1.0),
        'backward': Command(forward=-1.0),
        'turn_left': Command(yaw=1.0),
        'turn_right': Command(yaw=-1.0),
        'rise': Command(altitude=altitude + climb),
        'descend': Command(altitude=altitude - climb),
        'hover': Command(),
    }[action]
