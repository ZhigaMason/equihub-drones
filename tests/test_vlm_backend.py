"""The transformers backend against a stub pipeline: what it sends and what it returns. No model
is loaded here; a real run is a manual step."""
import sys

import numpy as np
import pytest

pytest.importorskip('PIL')

from drones.vlm.backend import DEFAULT_MODEL, MAX_NEW_TOKENS, TransformersBackend


class StubPipe:
    def __init__(self, reply=' {"done": true} '):
        self.reply = reply

    def __call__(self, text, **kwargs):
        self.text, self.kwargs = text, kwargs
        return [{'generated_text': text + [{'role': 'assistant', 'content': self.reply}]}]


def frame():
    image = np.zeros((4, 6, 3), np.uint8)
    image[..., 0] = 200
    return image


def test_it_sends_the_image_then_the_prompt_and_returns_the_reply_untouched():
    pipe = StubPipe()
    backend = TransformersBackend(pipe=pipe, max_new_tokens=32)
    assert backend.generate('fly', frame()) == ' {"done": true} '
    (message,) = pipe.text
    assert message['role'] == 'user'
    image, text = message['content']
    assert (image['type'], text) == ('image', {'type': 'text', 'text': 'fly'})
    assert image['image'].size == (6, 4)             # PIL's (width, height)
    assert image['image'].getpixel((0, 0)) == (200, 0, 0)
    # Greedy, so a run is repeatable.
    assert pipe.kwargs == {'max_new_tokens': 32, 'do_sample': False}


def test_a_view_into_a_larger_frame_is_accepted():
    # A cropped or flipped array is not contiguous, which Image.fromarray refuses.
    wide = np.zeros((4, 12, 3), np.uint8)
    pipe = StubPipe()
    TransformersBackend(pipe=pipe).generate('fly', wide[:, ::2])
    assert pipe.text[0]['content'][0]['image'].size == (6, 4)


def test_the_defaults():
    backend = TransformersBackend(pipe=StubPipe())
    assert backend.model == DEFAULT_MODEL == 'google/gemma-3n-E2B-it'
    assert backend.max_new_tokens == MAX_NEW_TOKENS == 768
    assert TransformersBackend(max_new_tokens='64', pipe=StubPipe()).max_new_tokens == 64


def test_nothing_is_loaded_until_the_first_frame():
    # drones-render-agent builds the agent before the scene; a bad scene name must not cost a
    # model load.
    assert TransformersBackend()._pipe is None


def test_without_the_vlm_extra_it_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)      # makes `import torch` raise ImportError
    with pytest.raises(SystemExit, match='--extra vlm'):
        TransformersBackend().generate('fly', frame())
