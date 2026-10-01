"""What constrained decoding lets a model write: the grammar the backend hands to
lm-format-enforcer, walked a character at a time. No model and no tokenizer are loaded."""
import json

import pytest

pytest.importorskip('lmformatenforcer')

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig

from drones.vlm.actions import ANSWER_MAX, CHUNK, json_schema, parse_chunk
from drones.vlm.backend import PARSER_CONFIG


def stops(space, text):
    """None if the grammar lets a model write all of `text` and end there, else what it had
    written when the next character was refused."""
    parser = JsonSchemaParser(json_schema(space), CharacterLevelParserConfig(**PARSER_CONFIG))
    for i, char in enumerate(text):
        if char not in parser.get_allowed_characters():
            return text[:i]
        parser = parser.add_character(char)
    return None if parser.can_end() else text


def discrete(n=CHUNK, move='forward', done=False, answer=None):
    return json.dumps({'actions': [move] * n, 'done': done, 'answer': answer})


def continuous(n=CHUNK, **action):
    action = action or {'forward': 0.5, 'yaw': -0.25, 'altitude': 1.4}
    return json.dumps({'actions': [action] * n, 'done': False, 'answer': None})


@pytest.mark.parametrize('space, text', [
    ('discrete', discrete()),
    ('discrete', discrete(move='hover', done=True, answer='B')),
    ('continuous', continuous()),
    ('continuous', continuous(forward=1.0, yaw=0.0, altitude=None)),
    ('continuous', continuous(forward=1, yaw=0)),              # altitude left out: hold
])
def test_a_valid_chunk_can_be_written_and_then_validates(space, text):
    assert stops(space, text) is None
    parse_chunk(text, space)


@pytest.mark.parametrize('n', [1, 15])
def test_a_chunk_cannot_stop_short_of_sixteen_actions(n):
    written = stops('discrete', discrete(n))
    assert written.endswith('"forward"') and written.count('forward') == n     # no `]` here
    assert stops('continuous', continuous(n)) is not None


def test_a_done_chunk_has_sixteen_actions_too():
    # lm-format-enforcer cannot say "sixteen, or none if done": its maxItems of 0 admits one
    # item. So the count is always sixteen, and a done chunk's actions are simply not flown.
    assert stops('discrete', discrete(0, done=True, answer='B')) == '{"actions": ['
    assert stops('discrete', discrete(1, done=True, answer='B')) is not None


def test_a_chunk_cannot_run_past_sixteen_actions():
    assert stops('discrete', discrete(17)).count('forward') == CHUNK
    assert stops('continuous', continuous(17)) is not None


@pytest.mark.parametrize('text, written', [
    ('{"actions": [m1, m2', '{"actions": ['),                  # the prompt's own placeholders
    ('{"actions": ["go_down"', '{"actions": ["'),              # not one of the moves
    ('```json\n' + discrete(), ''),                            # a code fence
    ('Sure! ' + discrete(), ''),                               # prose
    ('{"done": true', '{"'),                                   # keys out of order
    (json.dumps(json.loads(discrete()), indent=2), '{\n'),     # pretty-printing: wasted tokens
    (discrete()[:-len(', "answer": null}')] + '}', discrete()[:-len(', "answer": null}')]),
])
def test_what_a_discrete_reply_cannot_contain(text, written):
    assert stops('discrete', text) == written


def test_a_continuous_action_has_its_own_keys_and_no_others():
    assert stops('continuous', continuous(forward=1, yaw=0, speed=2)) is not None
    assert stops('continuous', continuous(forward=1)) is not None          # yaw is required
    assert stops('continuous', continuous(forward='fast', yaw=0)) is not None


def test_an_answer_cannot_run_on():
    assert stops('discrete', discrete(done=True, answer='x' * ANSWER_MAX)) is None
    assert stops('discrete', discrete(done=True, answer='x' * (ANSWER_MAX + 1))) is not None


def test_what_the_grammar_leaves_to_pydantic():
    # lm-format-enforcer ignores minimum and maximum, so a number out of range can still be
    # written. The schema rejects it, and the pilot asks again.
    fast = continuous(forward=7.5, yaw=0.0)
    assert stops('continuous', fast) is None
    with pytest.raises(ValueError):
        parse_chunk(fast, 'continuous')


class Tokenizer:
    """A toy tokenizer: a token is a short string, a reply is their concatenation. Enough of
    transformers' tokenizer interface for the backend to index it."""

    EOS = 0

    def __init__(self):
        words = ['"forward"', '"turn_left"', '"actions"', '"done"', '"answer"', 'false', 'true',
                 'null', ', ', ': ']
        self.tokens = ['<eos>'] + words + sorted(set(json.dumps(MOVE_TEXT) + 'm0123456789'))
        self.all_special_ids = [self.EOS]

    def __len__(self):
        return len(self.tokens)

    def encode(self, text):
        return [self.tokens.index(text)]

    def decode(self, ids):
        return ''.join('' if i == self.EOS else self.tokens[i] for i in ids)

    def ids(self, text):
        """`text` as tokens, longest first."""
        out = []
        while text:
            token = max((t for t in self.tokens[1:] if text.startswith(t)), key=len)
            out.append(self.tokens.index(token))
            text = text[len(token):]
        return out


MOVE_TEXT = {'actions': ['forward', 'backward', 'turn_left', 'turn_right', 'rise', 'descend',
                         'hover'], 'done': False, 'answer': None}


class Tokens(list):
    """What transformers hands the filter: a tensor, of which it uses tolist()."""

    def tolist(self):
        return list(self)


def constrained_filter(schema):
    from drones.vlm.backend import TransformersBackend

    class Pipe:
        tokenizer = Tokenizer()

        class model:
            class generation_config:
                eos_token_id = [Tokenizer.EOS]

        def __call__(self, text, **kwargs):
            self.kwargs = kwargs
            return [{'generated_text': [{'content': ''}]}]

    pipe = Pipe()
    TransformersBackend(pipe=pipe, schema=schema).generate('fly', None)
    return pipe.tokenizer, pipe.kwargs['generate_kwargs']['prefix_allowed_tokens_fn']


@pytest.fixture(autouse=False)
def no_pil(monkeypatch):
    # generate() turns the image into a PIL image; these tests have none to give.
    import types
    fake = types.ModuleType('PIL')
    fake.Image = types.SimpleNamespace(fromarray=lambda array: array)
    monkeypatch.setitem(__import__('sys').modules, 'PIL', fake)
    monkeypatch.setattr('numpy.ascontiguousarray', lambda image: image)


def test_the_backend_holds_a_real_tokenizer_to_the_schema_token_by_token(no_pil):
    pytest.importorskip('torch')
    tokenizer, allowed = constrained_filter(json_schema('discrete'))
    prompt = [5, 3]                            # whatever came before the reply
    reply = discrete()
    written = Tokens(prompt)
    for token in tokenizer.ids(reply):
        assert token in allowed(0, written), f'{tokenizer.decode(written[2:])!r} + ' \
                                             f'{tokenizer.tokens[token]!r}'
        written.append(token)
    assert Tokenizer.EOS in allowed(0, written)               # and it may stop there
    # Not before: fifteen actions cannot be closed, and no placeholder can be written.
    fifteen = Tokens(prompt + tokenizer.ids(discrete(15)[:discrete(15).index(']')]))
    assert tokenizer.ids(']')[0] not in allowed(0, fifteen)
    start = Tokens(prompt + tokenizer.ids('{"actions": ['))
    assert tokenizer.ids('m')[0] not in allowed(0, start)
    assert tokenizer.ids('"forward"')[0] in allowed(0, start)
