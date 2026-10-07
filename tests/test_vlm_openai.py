"""The OpenAI-API backend against a stand-in server on localhost: what it sends, what it returns
and how it fails. No model is served here; a real vLLM run is a manual step."""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from drones.vlm import agent
from drones.vlm.actions import json_schema
from drones.vlm.backend import MAX_NEW_TOKENS, OpenAIBackend, png


@pytest.fixture
def server():
    """A server that records each request body and answers with `server.reply`."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.path, body))
            status, reply = self.server.reply
            out = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(('127.0.0.1', 0), Handler)
    httpd.seen = seen
    httpd.url = f'http://127.0.0.1:{httpd.server_port}/v1'
    httpd.reply = (200, {'model': 'Qwen/served', 'choices': [
        {'message': {'role': 'assistant', 'content': '{"done": true}'}}]})
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def frame():
    image = np.zeros((4, 6, 3), np.uint8)
    image[..., 0] = 200
    return image


def test_it_sends_the_image_then_the_prompt_greedily_without_thinking(server):
    backend = OpenAIBackend('Qwen/asked', server.url + '/', max_new_tokens=32)
    assert backend.generate('fly', frame()) == '{"done": true}'
    ((path, body),) = server.seen
    assert path == '/v1/chat/completions'
    assert body['model'] == 'Qwen/asked'
    (message,) = body['messages']
    image, text = message['content']
    assert text == {'type': 'text', 'text': 'fly'}
    prefix = 'data:image/png;base64,'
    assert image['image_url']['url'] == prefix + base64.b64encode(png(frame())).decode()
    assert (body['temperature'], body['max_tokens']) == (0.0, 32)
    assert body['chat_template_kwargs'] == {'enable_thinking': False}
    assert 'response_format' not in body
    # The model the server ran, for the saved calls; the asked name may be an alias.
    assert backend.resolved_model == 'Qwen/served'


def test_a_schema_goes_to_the_server_as_structured_output(server):
    schema = json_schema('discrete', 4)
    OpenAIBackend('m', server.url, schema=schema).generate('fly', frame())
    fmt = server.seen[0][1]['response_format']
    assert fmt == {'type': 'json_schema',
                   'json_schema': {'name': 'chunk', 'schema': schema, 'strict': True}}


def test_a_judge_asks_in_words_alone_with_its_own_system_prompt(server):
    OpenAIBackend('m', server.url, system='mark it').generate('is it right?')
    messages = server.seen[0][1]['messages']
    assert messages == [{'role': 'system', 'content': 'mark it'},
                        {'role': 'user', 'content': [{'type': 'text', 'text': 'is it right?'}]}]


def test_a_server_error_stops_the_run_with_its_message(server):
    server.reply = (400, {'error': 'image too large'})
    with pytest.raises(RuntimeError, match='400.*image too large'):
        OpenAIBackend('m', server.url).generate('fly', frame())


def test_no_server_stops_the_run():
    with pytest.raises(RuntimeError, match='did not answer'):
        OpenAIBackend('m', 'http://127.0.0.1:9/v1', timeout=5).generate('fly', frame())


def test_the_agent_factory_builds_it_constrained_to_the_chunk(server):
    pilot = agent.make('discrete', backend='openai', model='Qwen/x', url=server.url,
                       chunk=8).pilot
    backend = pilot.backend
    assert isinstance(backend, OpenAIBackend)
    assert (backend.model, backend.url) == ('Qwen/x', server.url)
    assert backend.schema == json_schema('discrete', 8)
    assert backend.max_new_tokens == MAX_NEW_TOKENS
    free = agent.make('discrete', backend='openai', model='Qwen/x', constrain=0).pilot.backend
    assert free.schema is None


def test_the_factory_needs_the_served_model_name():
    with pytest.raises(ValueError, match='model=NAME'):
        agent.make('discrete', backend='openai')
