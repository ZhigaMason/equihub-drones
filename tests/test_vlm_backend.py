"""The transformers backend against a stub pipeline: what it sends and what it returns. No model
is loaded here; a real run is a manual step."""
import sys
import types

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
    # Greedy, so a run is repeatable. It has to go in generate_kwargs: the pipeline hands any
    # other keyword to the processor, which ignores it, and Gemma's own config samples.
    assert pipe.kwargs == {'max_new_tokens': 32, 'generate_kwargs': {'do_sample': False}}


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


def test_loading_keeps_triton_out_and_asks_for_the_model_on_the_cpu(monkeypatch):
    # torch imports triton if it is installed, and triton's own LLVM segfaults when it loads
    # after Mesa has made an EGL context, which the simulator has by the first frame.
    calls = []

    def pipeline(task, **kwargs):
        calls.append((task, kwargs, sys.modules.get('triton', 'absent')))
        return StubPipe()

    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.float32, torch.bfloat16 = 'float32', 'bfloat16'
    transformers = types.ModuleType('transformers')
    transformers.pipeline = pipeline
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    monkeypatch.delitem(sys.modules, 'triton', raising=False)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    try:
        backend = TransformersBackend(model='some/model')
        assert backend.generate('fly', frame()) == ' {"done": true} '
    finally:
        if sys.modules.get('triton', 'absent') is None:
            del sys.modules['triton']
    (task, kwargs, triton), = calls
    assert triton is None                      # `import triton` now raises ImportError
    assert task == 'image-text-to-text'
    assert kwargs == {'model': 'some/model', 'device': 'cpu', 'dtype': 'float32'}


def test_without_lm_format_enforcer_a_schema_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, 'lmformatenforcer', None)
    pipe = StubPipe()
    pipe.tokenizer = object()
    with pytest.raises(SystemExit, match='--extra vlm') as stopped:
        TransformersBackend(pipe=pipe, schema={'type': 'object'}).generate('fly', frame())
    # The cause too: an import can fail for reasons other than a missing extra.
    assert 'lmformatenforcer' in str(stopped.value)
