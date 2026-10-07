# FAST-EQA as an agent, with swappable models: `drones.fasteqa`

Date: 2026-10-07. Status: approved in conversation, awaiting review of this document.

## Goal

An agent that answers the EQA benchmarks (`hm-eqa`, `mt-hm3d`, `express-bench`, `a-eqa`) the way
FAST-EQA (arXiv 2602.15813) does, as closely as the paper lets us, run as
`--agent drones.fasteqa:make` by `drones-render-agent` and `drones-benchmark` without changing
either. Every learned part of it can be swapped from the command line.

What the user decided:

- **As faithful as possible**, leaving room for CLIP-style models better than CLIP, SigLIP above
  all.
- **Two VLM backends**, as in the paper: a per-step `scorer` (the paper's Prismatic-7B) and a
  `reasoner` for parsing, the stop check and the answer (the paper's GPT-4o).
- **Depth from either an oracle (MuJoCo) or a separate model fed the frames.**

What the paper does not give, so is chosen here and recorded as a deviation: its prompts (in a
supplementary that was not available), λ, k, DBSCAN and voxel parameters, how often the stop check
runs, and whether a panorama costs one step. The code is not released.

## The paper's pipeline, in brief

1. An LLM parses the question into visual targets T = {T_m} and the rooms R they are likely in.
2. RGB-D is fused into a TSDF; a 2D slice of its occupancy gives frontiers. Doorways (free cells
   pinched between occupied ones, Eq. 1) are clustered with DBSCAN into frontier centroids,
   prioritised by cluster size and closeness to unexplored space: **global relevance (GR)**.
3. A VLM tags the current room each step; inside a room from R, the agent turns a full panorama
   looking for targets: **local relevance (LR)**.
4. Each observation is scored per target, rel = λ·rel_CLIP(o, T_m) + (1−λ)·rel_VLM(o, Q), where
   rel_VLM is the VLM's probability of "yes" to "does this image hold evidence to answer Q?". The
   top k observations per target are kept: memory is M×k images, whatever the episode's length.
5. GPT-4o, given the stop prompt and the memory, decides whether it can answer; the answer is a
   chain-of-thought over the M×k images.

Ablations (100 HM-EQA questions, 76% SR): without doorway frontiers 67%, without CLIP 69%, without
VLM relevance 72%, without CoT 72%.

## Where it lives

`src/drones/fasteqa/`, a package of its own. CLAUDE.md keeps agent code out of `sim/` unless it
is a baseline, and `sim/` may not import `drones.vlm` at module level; this agent needs both.

| Module | Holds |
| --- | --- |
| `fasteqa/__init__.py` | `make(**agent_args)`, the factory, re-exported |
| `fasteqa/agent.py` | `FastEQAAgent`: the agent protocol, the GR/LR state machine, motion |
| `fasteqa/mapping.py` | `OccupancyMap`: depth fusion, doorway and frontier cells, DBSCAN, A* (numpy only) |
| `fasteqa/memory.py` | `Memory`: per-target top-k of scored observations |
| `fasteqa/prompts.py` | parse, relevance, stop and answer prompts, and parsing their replies |
| `fasteqa/embedder.py` | the `Embedder` protocol, `SiglipEmbedder`, `ClipEmbedder` |
| `fasteqa/depth.py` | the `DepthModel` protocol, `DepthAnything` |

Layering: `fasteqa/` may import numpy, `drones.sim` (for `Pose`, intrinsics) and
`drones.vlm.backend`; torch, transformers and PIL only inside functions. AGENTS.md's layering
table gets a row, and `tests/test_architecture.py` a test that importing `drones.fasteqa` pulls in
none of torch, transformers or PIL.

## The swappable parts

All chosen by `--agent-arg key=value`. A value of the form `package.module:factory` loads any
object with the right methods, as `make_agent` does for agents.

| Arg | Role | Built in | Contract |
| --- | --- | --- | --- |
| `reasoner=` | parse, stop check, CoT answer | `claude-code` (default), `openai`, `transformers` | `Backend.generate(prompt, images)` |
| `scorer=` | rel_VLM, every view | the same three; default `transformers` with `google/gemma-3n-E2B-it` | `generate`, and `yes_probability` if it has one |
| `embedder=` | rel_embed, room tags, frontier scores | `siglip` (default, `google/siglip2-base-patch16-224`), `clip` (`openai/clip-vit-base-patch32`) | `image(img) -> (d,)`, `text(str) -> (d,)`, unit vectors |
| `depth=` | depth for the map | `oracle` (default), `depth-anything` | `estimate(image) -> (H, W)` float32 m |

`reasoner_model=`, `scorer_model=`, `embedder_model=`, `depth_model=` name the checkpoint or model;
`reasoner_url=`, `scorer_url=` the server for `openai`. `lambda_=` (default 0.5), `k=` (default 3),
`start_altitude=` (set by the benchmark) complete the arguments.

### Backends: several images, and P("yes")

Two additions to `drones.vlm.backend`, both backward compatible:

- `generate(prompt, image)` also accepts a list of images, sent as that many image blocks
  (Claude stream-json, OpenAI `image_url` parts, transformers chat messages). One array, or None,
  behaves as today. The stop check and the answer send all M×k memory images.
- An optional `yes_probability(prompt, image) -> float`, P(first token is "yes") renormalised over
  yes and no. `OpenAIBackend` reads it from `logprobs`/`top_logprobs` (vLLM serves them);
  `TransformersBackend` from the first generated token's logits. The scorer uses it when the
  backend has it, and otherwise asks for a one-word reply and reads yes as 1.0 and anything else
  as 0.0. `ClaudeCodeBackend` has no logprobs, so Claude as a scorer is the fallback, and costs a
  call per view.

## Depth

Both sources give the same thing: **z-depth along the optical axis per pixel, in m, OpenCV's
convention** (the one `recordings/intrinsics.json` uses). The mapper back-projects it through
`Intrinsics.undistort` and the camera pose (`Mount.camera_pose`) and never knows which source it
had. Pixels with no depth (nothing hit, or past `MAX_DEPTH` = 5 m) are left out.

- `depth=oracle`: MuJoCo's depth buffer, rendered with the frame. `LensCamera.render(data, eye,
  rotation, depth=True)` returns `(image, depth)`: the buffer is linearised from the near and far
  planes, converted from distance along the pinhole axis to the lens's z-depth (the same axis, so
  only the lookup changes), and sampled through the lens lookup with the **nearest** tap, because
  blending across an edge invents points in mid-air. `SceneView.render_matrix(..., depth=True)`
  passes it through.
- `depth=depth-anything`: `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` through
  transformers' depth-estimation pipeline, its output resized to the frame. It is metric, so it
  needs no scale fit; a relative-depth model would, and is out of scope.

### What `sim/agents.py` gains

Two attributes, in the style the benchmark already reads agents by:

- An agent with `depth = True` is given `Observation.depth`, a (H, W) float32 array; otherwise it
  is None and no depth is rendered. `FastEQAAgent.depth` is True for `oracle` and False for an
  estimator, which works from `observation.image` itself.
- An agent with a `camera` attribute has it **set** by `agents.episode` to a
  `CameraInfo(intrinsics, mount)` of the view, before `reset`. Back-projection needs it whatever
  the depth source.

`Observation` gains `depth: np.ndarray | None = None`, a default, so code that builds one is
unchanged.

## The agent, one decision at a time

A decision is one Explore-EQA step: up to `reach` = **3.0 m** of travel (the paper's step), or one
panorama. `chunk_size` = 16 frames per decision: the motion is interpolated over them, so films
are smooth and the benchmark's budget, `int(√area × 3)` decisions, means what it does in the
paper. The agent decides on the first frame of a decision and plays the other 15 out, as the VLM
pilot plays a chunk; every frame is fused into the map and scored only if it is the first or a
panorama view (eight per panorama, 45° apart), which bounds scorer calls at nine per decision.

On `reset(question, pose)`: clear map and memory; **parse** the question (reasoner, once) into
JSON `{"targets": [...], "rooms": [...]}`, with at most `MAX_TARGETS` = 4 targets; embed the
targets, the rooms and the fixed list of room names.

On each decision:

1. **Fuse** the frame's depth into `OccupancyMap`: a 2D grid of 0.1 m cells over the band 0.1 to
   2.0 m above the floor. A hit marks its cell occupied, the cells the ray crossed free (2D
   Bresenham). It is the paper's TSDF projected to the slice it uses, not a TSDF.
2. **Score** the view: for each target, rel = λ·s(embedder) + (1−λ)·P(yes) from the scorer, where
   s is the embedder's `match`, in [0, 1]: SigLIP's own sigmoid of its scaled cosine (its trained
   `logit_scale` and `logit_bias`, a probability by construction), or for CLIP, which has no
   bias, the cosine mapped linearly from its typical range (0.15 to 0.35). Offer the view to
   `Memory`.
3. **Room**: the room name whose text embedding is closest to the frame's. If it is in R and the
   room has not been panned, the decision is a **panorama** (LR). The room stays panned until
   the agent has left it (another room tag for two decisions running).
4. Otherwise **GR**: cells that are free with occupied cells on both sides along x or along y
   (doorways, Eq. 1), and free cells next to unknown ones (frontiers). Cluster each set with
   DBSCAN (eps 0.3 m, 4 points; a grid-neighbour implementation in numpy). Score each cluster
   centroid by `w_size·size + w_unknown·unknown within 1 m + w_room·max cos(view toward it, R)`,
   doorways first at equal score, and pick the best reachable one by A* over free cells,
   inflated by 0.2 m. Fly up to 3 m along the path, facing along it. With no reachable frontier,
   turn 90° in place.
5. **Stop check**: when the memory changed this decision, ask the reasoner with the stop prompt
   and the memory images. If it says it can answer, it answers in the same reply, chain of
   thought first, ending with `Answer: X`, and the agent returns None.
6. On `conclude(observation)` (budget run out): fuse and score the last frame, then the CoT
   answer from memory.

The answer for a multiple-choice question is read with `metrics.choice_letter`; an open answer is
the text after `Answer:`.

Also exposed, read by the benchmark by attribute: `chunk_size` (16), `reach` (3.0), `calls` (every
reasoner and scorer call: role, prompt, reply, images saved as for the VLM pilot, seconds,
`pos`, `yaw`), `stats`, `error`, `answer`, and `caption` (the mode, the chosen frontier, the
memory's best score per target, the room tag) for films.

Failures: a backend that raises stops the run, as the pilot's do. A parse reply that does not
validate is retried once, then the question itself becomes the only target and R is empty
(the agent explores by frontiers alone).

## Deviations from the paper

Added to `drones.sim.scoring.DEVIATIONS` for runs of this agent, and listed in its docstring:

- A 2D occupancy grid from depth, not a TSDF.
- Prompts written here; λ = 0.5 and k = 3 chosen here, not tuned.
- Rooms tagged by the embedder zero-shot, not by Prismatic.
- The relevance VLM is whatever `scorer` is (Gemma 3n E2B by default), not Prismatic-7B; with
  no logprobs, a yes/no read as 1/0.
- A panorama costs one decision.
- The stop check runs only when the memory changed.
- Movement is kinematic with nothing to collide with, but every path is planned on the map, so
  the drone does not fly through what it has seen.

## Testing

Nothing loads a model or needs `scenes/`, as for the rest of the suite.

- `test_fasteqa_mapping.py`: fusion of a synthetic depth image into known free and occupied cells;
  doorway detection on a drawn wall with a gap; DBSCAN on hand-placed points; A* around a wall,
  and none through it.
- `test_fasteqa_memory.py`: top-k per target, ties, `changed`.
- `test_fasteqa_prompts.py`: parse replies (valid, fenced, invalid), stop and answer replies,
  `Answer:` extraction.
- `test_vlm_backend.py` / `test_vlm_openai.py` / `test_vlm_claude_code.py`: several images in one
  request; `yes_probability` from a fake vLLM response with `top_logprobs`.
- `test_deck_camera.py` or `test_scene_view.py` (sim): oracle depth on the synthetic box reads a
  wall at its known distance, at the centre and at distorted off-centre pixels; nearest taps leave
  no depth between a near and a far surface.
- `test_fasteqa_agent.py` (sim): the agent on the synthetic box with fake reasoner, scorer,
  embedder and depth model; both depth sources give (H, W) float32 in m; a full
  `drones-benchmark run` on one question ends with an answer, a `calls.jsonl` and a trajectory
  that never leaves the box.

Real model calls (Claude usage, SigLIP, Depth Anything downloads) happen only when the user asks.

## Documentation

AGENTS.md: the layering row; `depth` and `camera` in the agent protocol paragraph ("Where new work
goes"); backends taking several images and `yes_probability`; the command line. README: a section
under benchmarking with the command and the swappable parts. pyproject: no new extra (SigLIP, CLIP
and Depth Anything run on the `vlm` extra's transformers and torch).
