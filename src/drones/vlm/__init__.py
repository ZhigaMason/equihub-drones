"""A vision-language model as the pilot: a question and a camera frame in, action chunks out.

`actions`, `prompt`, `pilot` and `backend` import neither the simulator nor a model, so the same
pilot can later sit on the real drone. Only `agent`, the simulator adapter, needs the ``sim``
extra, and only running a real model needs ``vlm``: ``uv sync --extra sim --extra vlm``.
"""
