"""The Claude Code backend against a fake `claude` executable: the flags it runs with, the message
it sends and what it makes of the reply. The real CLI is never started here; a real run is a
manual step, and it spends the subscription's usage."""
import base64
import json
import struct
import sys
import textwrap
import zlib
from pathlib import Path

import numpy as np
import pytest

from drones.vlm.backend import CLAUDE_MODEL, ClaudeCodeBackend, png

RESULT = {'type': 'result', 'subtype': 'success', 'is_error': False,
          'result': ' {"done": true} '}


def fake_claude(tmp_path, events=(RESULT,), status=0, stderr=''):
    """A `claude` that records its argv, stdin and cwd in tmp_path/call.json and prints
    `events` as stream-json lines."""
    script = tmp_path / 'claude'
    script.write_text(textwrap.dedent(f'''\
        #!{sys.executable}
        import json, os, sys
        record = {{'argv': sys.argv[1:], 'stdin': sys.stdin.read(), 'cwd': os.getcwd()}}
        with open({str(tmp_path / 'call.json')!r}, 'w') as f:
            json.dump(record, f)
        for event in {list(events)!r}:
            print(json.dumps(event))
        print({stderr!r}, file=sys.stderr)
        sys.exit({status})
    '''))
    script.chmod(0o755)
    return str(script)


def call(tmp_path):
    return json.loads((tmp_path / 'call.json').read_text())


def frame():
    image = np.zeros((4, 6, 3), np.uint8)
    image[..., 0] = 200
    image[1, 2] = (1, 2, 3)
    return image


def decode_png(data):
    """(height, width, 3) uint8 from an 8-bit RGB PNG with no filtering, as `png` writes it."""
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    at, chunks = 8, {}
    while at < len(data):
        (length,), kind = struct.unpack('>I', data[at:at + 4]), data[at + 4:at + 8]
        body = data[at + 8:at + 8 + length]
        assert struct.unpack('>I', data[at + 8 + length:at + 12 + length])[0] \
            == zlib.crc32(kind + body)
        chunks[kind] = chunks.get(kind, b'') + body
        at += 12 + length
    width, height, depth, colour = struct.unpack('>IIBB', chunks[b'IHDR'][:10])
    assert (depth, colour) == (8, 2)
    rows = zlib.decompress(chunks[b'IDAT'])
    stride = 1 + 3 * width
    assert all(rows[i * stride] == 0 for i in range(height))      # filter type None
    pixels = [rows[i * stride + 1:(i + 1) * stride] for i in range(height)]
    return np.frombuffer(b''.join(pixels), np.uint8).reshape(height, width, 3)


def test_png_round_trips_a_frame_and_a_view_into_one():
    image = frame()
    assert (decode_png(png(image)) == image).all()
    wide = np.zeros((4, 12, 3), np.uint8)
    wide[:, ::2] = image
    assert (decode_png(png(wide[:, ::2])) == image).all()     # not contiguous


def test_it_sends_the_image_then_the_prompt_and_returns_the_reply_untouched(tmp_path):
    backend = ClaudeCodeBackend(executable=fake_claude(tmp_path))
    assert backend.generate('fly', frame()) == ' {"done": true} '
    (line,) = call(tmp_path)['stdin'].splitlines()
    message = json.loads(line)
    assert message['type'] == 'user'
    assert message['message']['role'] == 'user'
    image, text = message['message']['content']
    assert text == {'type': 'text', 'text': 'fly'}
    assert image['type'] == 'image'
    assert image['source']['type'] == 'base64'
    assert image['source']['media_type'] == 'image/png'
    assert (decode_png(base64.b64decode(image['source']['data'])) == frame()).all()


def test_it_runs_one_stateless_turn_with_nothing_but_the_model(tmp_path):
    ClaudeCodeBackend(model='opus', executable=fake_claude(tmp_path)).generate('fly', frame())
    argv = call(tmp_path)['argv']
    flag = {name: argv[argv.index(name) + 1] for name in
            ('--model', '--tools', '--setting-sources', '--input-format', '--output-format',
             '--system-prompt')}
    assert flag['--model'] == 'opus'
    # No tools, no settings (so none of this repo's hooks), no MCP servers, not even the
    # claude.ai connectors, which --tools '' leaves in: the model can only look and reply.
    assert flag['--tools'] == '' and flag['--setting-sources'] == ''
    assert {'-p', '--strict-mcp-config', '--disable-slash-commands',
            '--no-session-persistence'} <= set(argv)
    # Images go in only as stream-json, and the CLI then insists on stream-json out.
    assert flag['--input-format'] == flag['--output-format'] == 'stream-json'
    assert '--verbose' in argv              # which stream-json output requires with -p
    # Its own short system prompt, in place of Claude Code's, which describes a coding agent.
    assert 'drone' in flag['--system-prompt']


def test_it_runs_outside_the_repository(tmp_path):
    # In the repository the CLI would read CLAUDE.md, which is about the code, not the flight.
    ClaudeCodeBackend(executable=fake_claude(tmp_path)).generate('fly', frame())
    repository = Path(__file__).resolve().parents[1]
    cwd = Path(call(tmp_path)['cwd']).resolve()
    assert repository != cwd and repository not in cwd.parents


def test_the_defaults():
    backend = ClaudeCodeBackend()
    assert backend.model == CLAUDE_MODEL == 'sonnet'
    assert backend.executable == 'claude'


def test_an_error_result_raises_with_the_cli_message(tmp_path):
    error = dict(RESULT, subtype='error_during_execution', is_error=True,
                 result='Claude AI usage limit reached')
    backend = ClaudeCodeBackend(executable=fake_claude(tmp_path, [error], status=1))
    with pytest.raises(RuntimeError, match='usage limit reached'):
        backend.generate('fly', frame())


def test_a_failed_run_without_a_result_raises_with_stderr(tmp_path):
    backend = ClaudeCodeBackend(executable=fake_claude(tmp_path, [], status=1,
                                                       stderr='Invalid API key'))
    with pytest.raises(RuntimeError, match='Invalid API key'):
        backend.generate('fly', frame())


def test_without_the_cli_it_says_how_to_install_it(tmp_path):
    backend = ClaudeCodeBackend(executable=str(tmp_path / 'missing'))
    with pytest.raises(SystemExit, match='Claude Code'):
        backend.generate('fly', frame())


def test_a_call_that_hangs_raises(tmp_path):
    script = tmp_path / 'claude'
    script.write_text(f'#!{sys.executable}\nimport time\ntime.sleep(30)\n')
    script.chmod(0o755)
    with pytest.raises(RuntimeError, match='timed out'):
        ClaudeCodeBackend(executable=str(script), timeout=0.5).generate('fly', frame())


def test_a_text_only_call_sends_no_image_and_its_own_system_prompt(tmp_path):
    backend = ClaudeCodeBackend(executable=fake_claude(tmp_path), system='You grade answers.')
    assert backend.generate('Your mark?', None) == ' {"done": true} '
    message = json.loads(call(tmp_path)['stdin'])
    assert message['message']['content'] == [{'type': 'text', 'text': 'Your mark?'}]
    argv = call(tmp_path)['argv']
    assert argv[argv.index('--system-prompt') + 1] == 'You grade answers.'


def test_it_records_the_model_the_cli_reports_in_its_init_event(tmp_path):
    init = {'type': 'system', 'subtype': 'init', 'model': 'claude-sonnet-5-5'}
    backend = ClaudeCodeBackend(model='sonnet', executable=fake_claude(tmp_path, [init, RESULT]))
    assert backend.resolved_model is None
    backend.generate('fly', frame())
    assert backend.resolved_model == 'claude-sonnet-5-5'
