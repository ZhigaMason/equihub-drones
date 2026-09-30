"""Where a VLM pilot's replies come from: one method, and a local model behind it.

The pilot asks for text and validates it itself, so a backend is only `generate(prompt, image)`.
That keeps the model out of the tests, which script the replies, and lets another model or a
server stand in without touching the pilot.

`TransformersBackend` runs a Hugging Face image-text-to-text checkpoint in this process. It loads
nothing until the first frame: torch takes seconds to import and the model longer, and neither is
needed to build an agent or to import this module.
"""
import logging
import os
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

# The smallest Gemma that accepts images. It is gated: accept the licence on Hugging Face and
# `hf auth login` once, or set HF_TOKEN.
DEFAULT_MODEL = 'google/gemma-3n-E2B-it'
# A continuous chunk is 16 objects of about 25 tokens each; this leaves room for the rest.
MAX_NEW_TOKENS = 768


class Backend(Protocol):
    def generate(self, prompt: str, image: np.ndarray) -> str:
        """The model's raw reply to `prompt` and an (height, width, 3) uint8 RGB `image`."""


class TransformersBackend:
    """A local `model` through transformers' image-text-to-text pipeline, on `device` ('cuda'
    where there is one, by default). `pipe` is a ready pipeline to use instead of loading one."""

    def __init__(self, model=DEFAULT_MODEL, max_new_tokens=None, device=None, pipe=None):
        self.model, self.device = model, device
        self.max_new_tokens = MAX_NEW_TOKENS if max_new_tokens is None else int(max_new_tokens)
        self._pipe = pipe

    def _load(self):
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            raise SystemExit('The VLM pilot needs the vlm extra:  '
                             'uv sync --extra sim --extra camera --extra vlm') from None
        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('Loading %s on %s', self.model, device)
        # Without HF_TOKEN, transformers falls back to the token `hf auth login` saved.
        token = os.getenv('HF_TOKEN')
        return pipeline('image-text-to-text', model=self.model, device=device,
                        dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                        **({'token': token} if token else {}))

    def generate(self, prompt, image):
        if self._pipe is None:
            self._pipe = self._load()
        from PIL import Image

        picture = Image.fromarray(np.ascontiguousarray(image))
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': picture},
                                                 {'type': 'text', 'text': prompt}]}]
        # Greedy, so the same frame and question give the same chunk.
        out = self._pipe(text=messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        return out[0]['generated_text'][-1]['content']
