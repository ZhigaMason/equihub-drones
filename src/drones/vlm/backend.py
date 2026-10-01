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
import sys
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

# The smallest Gemma that accepts images. It is gated: accept the licence on Hugging Face and
# `hf auth login` once, or set HF_TOKEN.
DEFAULT_MODEL = 'google/gemma-3n-E2B-it'
# A continuous chunk is 16 objects of about 25 tokens each; this leaves room for the rest.
MAX_NEW_TOKENS = 768
# How lm-format-enforcer reads a schema here. One space at most between tokens: it otherwise lets
# a model indent its JSON, which is a few hundred tokens of nothing. Keys in the schema's
# `required` order, so every reply is laid out alike.
PARSER_CONFIG = {'max_consecutive_whitespaces': 1, 'force_json_field_order': True}
EXTRA = 'The VLM pilot needs the vlm extra:  uv sync --extra sim --extra camera --extra vlm'


def _vocabulary(pipe):
    """lm-format-enforcer's index of `pipe`'s tokenizer: every token's text, as a prefix tree.

    This is what lmformatenforcer.integrations.transformers builds, done here because that
    module cannot be imported under transformers 5 (0.11.3 imports PreTrainedTokenizerBase from
    transformers.tokenization_utils, which transformers 5 removed). Decoding each token after a
    "0" tells a word-start token by the space it gains. It takes seconds for a 262k vocabulary,
    so a backend does it once.
    """
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

    tokenizer = pipe.tokenizer
    zero = tokenizer.encode('0')[-1]
    special = set(tokenizer.all_special_ids)
    regular = []
    for token in range(len(tokenizer)):
        if token in special:
            continue
        after_zero = tokenizer.decode([zero, token])[1:]
        regular.append((token, after_zero, len(after_zero) > len(tokenizer.decode([token]))))

    def decode(tokens):
        return tokenizer.decode(tokens).rstrip('\ufffd')    # a half-decoded character

    # The tokens generation stops at: a chat model ends its turn with one the tokenizer does not
    # call eos (Gemma's <end_of_turn>), so the model's own list, where it has one.
    config = getattr(getattr(pipe, 'model', None), 'generation_config', None)
    eos = getattr(config, 'eos_token_id', None)
    eos = tokenizer.eos_token_id if eos is None else eos
    return TokenEnforcerTokenizerData(regular, decode, eos, False, len(tokenizer))


class Backend(Protocol):
    def generate(self, prompt: str, image: np.ndarray) -> str:
        """The model's raw reply to `prompt` and an (height, width, 3) uint8 RGB `image`."""


class TransformersBackend:
    """A local `model` through transformers' image-text-to-text pipeline, on `device` ('cuda'
    where there is one, by default). `pipe` is a ready pipeline to use instead of loading one.

    With `schema`, a JSON schema, decoding is constrained: at every token the model may choose
    only among those that keep its reply a prefix of something the schema allows. It then
    cannot write prose, a code fence, an unknown key or the wrong number of items, whatever it
    would have preferred. Without one, it writes freely and the pilot validates afterwards."""

    def __init__(self, model=DEFAULT_MODEL, max_new_tokens=None, device=None, pipe=None,
                 schema=None):
        self.model, self.device, self.schema = model, device, schema
        self.max_new_tokens = MAX_NEW_TOKENS if max_new_tokens is None else int(max_new_tokens)
        self._pipe = pipe
        self._vocabulary = None     # lm-format-enforcer's index of the tokenizer

    def _load(self):
        # torch imports triton when it is installed, and on Linux it always is. triton brings its
        # own LLVM, which segfaults on import once Mesa has made an EGL context (Mesa loads the
        # system's LLVM) - and the simulator has one by the first frame. None makes `import
        # triton` raise ImportError, which torch takes as triton not being installed. Only
        # torch.compile needs it, and nothing here compiles. A triton already imported is left.
        sys.modules.setdefault('triton', None)
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            # The cause as well: it is not always a missing extra.
            raise SystemExit(f'{EXTRA}  ({exc})') from None
        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('Loading %s on %s', self.model, device)
        # Without HF_TOKEN, transformers falls back to the token `hf auth login` saved.
        token = os.getenv('HF_TOKEN')
        return pipeline('image-text-to-text', model=self.model, device=device,
                        dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                        **({'token': token} if token else {}))

    def _allowed_tokens(self):
        """transformers' `prefix_allowed_tokens_fn` for one reply held to `schema`."""
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
            from lmformatenforcer.tokenenforcer import TokenEnforcer
        except ImportError as exc:
            raise SystemExit(f'{EXTRA}  ({exc})') from None
        if self._vocabulary is None:
            self._vocabulary = _vocabulary(self._pipe)
        parser = JsonSchemaParser(self.schema, CharacterLevelParserConfig(**PARSER_CONFIG))
        # One per reply: it follows the tokens of the reply it was made for.
        enforcer = TokenEnforcer(self._vocabulary, parser)

        def allowed(batch, tokens):
            return enforcer.get_allowed_tokens(tokens.tolist()).allowed_tokens

        return allowed

    def generate(self, prompt, image):
        if self._pipe is None:
            self._pipe = self._load()
        from PIL import Image

        picture = Image.fromarray(np.ascontiguousarray(image))
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': picture},
                                                 {'type': 'text', 'text': prompt}]}]
        # Greedy, so the same frame and question give the same chunk. In generate_kwargs, because
        # the pipeline passes any other keyword to the processor, which drops it with a warning,
        # and the model's own generation config samples.
        options = {'do_sample': False}
        if self.schema is not None:
            options['prefix_allowed_tokens_fn'] = self._allowed_tokens()
        out = self._pipe(text=messages, max_new_tokens=self.max_new_tokens,
                         generate_kwargs=options)
        return out[0]['generated_text'][-1]['content']
