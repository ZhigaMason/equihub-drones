"""drones.sim.scenes: reading a Gibson-style .glb, finding open floor, adding it to the renderer.

Hermetic: the scene is a synthetic room written as a .glb here, never a download.
"""
import io
import json
import struct
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones.sim import scenes
from test_render import gl_env, needs_gl, saved_square_run  # noqa: F401  saved_square_run: fixture

# subprocess forks this JAX-threaded interpreter only to exec another one, which is safe.
pytestmark = pytest.mark.filterwarnings(r'ignore:os\.fork\(\) was called:RuntimeWarning')

FLOOR_Z = 0.3                 # the scan's floor is not at 0; load() must move it there
ROOM = (-1.0, 5.0, -2.0, 2.0)  # x0, x1, y0, y1
PILLAR = (3.0, 5.0)           # x range of a block filling the room's far end, floor to ceiling
HEIGHT = 2.5


def rectangle(origin, u, v, step=0.2):
    """Triangles tiling the rectangle origin + [0, 1] u + [0, 1] v, finely enough to scan."""
    n, m = max(1, round(np.linalg.norm(u) / step)), max(1, round(np.linalg.norm(v) / step))
    a, b = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, m + 1), indexing='ij')
    points = origin + a[..., None] * u + b[..., None] * v
    index = np.arange((n + 1) * (m + 1)).reshape(n + 1, m + 1)
    quads = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], -1)
    faces = np.concatenate([quads[..., [0, 1, 2]], quads[..., [0, 2, 3]]]).reshape(-1, 3)
    return points.reshape(-1, 3), faces


def room_mesh():
    x0, x1, y0, y1 = ROOM
    parts = [
        rectangle([x0, y0, FLOOR_Z], [x1 - x0, 0, 0], [0, y1 - y0, 0]),           # floor, up
        rectangle([x0, y0, FLOOR_Z + HEIGHT], [0, y1 - y0, 0], [x1 - x0, 0, 0]),  # ceiling
        rectangle([x0, y0, FLOOR_Z], [x1 - x0, 0, 0], [0, 0, HEIGHT]),
        rectangle([x0, y1, FLOOR_Z], [x1 - x0, 0, 0], [0, 0, HEIGHT]),
        rectangle([x0, y0, FLOOR_Z], [0, y1 - y0, 0], [0, 0, HEIGHT]),
        rectangle([x1, y0, FLOOR_Z], [0, y1 - y0, 0], [0, 0, HEIGHT]),
        rectangle([PILLAR[0], y0, FLOOR_Z], [0, y1 - y0, 0], [0, 0, HEIGHT]),     # pillar face
    ]
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.uint32)


def jpeg(image):
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(image).save(out, 'JPEG')
    return out.getvalue()


def glb_bytes(primitives, images, basis=False):
    """A .glb as Gibson and HM3D lay them out: `primitives` are (vertices, faces, uv, image
    index), `images` encoded bytes. basis=True marks the images as HM3D's Basis textures."""
    chunks, accessors, meshes = [], [], []
    for vertices, faces, uv, image in primitives:
        first = len(chunks)
        chunks += [vertices.astype(np.float32).tobytes(), uv.astype(np.float32).tobytes(),
                   faces.astype(np.uint32).tobytes()]
        accessors += [
            {'bufferView': first, 'componentType': 5126, 'count': len(vertices), 'type': 'VEC3'},
            {'bufferView': first + 1, 'componentType': 5126, 'count': len(uv), 'type': 'VEC2'},
            {'bufferView': first + 2, 'componentType': 5125, 'count': faces.size,
             'type': 'SCALAR'}]
        meshes.append({'primitives': [{'attributes': {'POSITION': first, 'TEXCOORD_0': first + 1},
                                       'indices': first + 2, 'material': image}]})
    image_views = list(range(len(chunks), len(chunks) + len(images)))
    chunks += list(images)
    views, binary = [], b''
    for chunk in chunks:
        views.append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(chunk)})
        binary += chunk + b'\0' * (-len(chunk) % 4)
    mime = 'image/x-basis' if basis else 'image/jpeg'
    textures = [{'extensions': {'GOOGLE_texture_basis': {'source': i}}} if basis else {'source': i}
                for i in range(len(images))]
    gltf = {
        'asset': {'version': '2.0'},
        'buffers': [{'byteLength': len(binary)}],
        'bufferViews': views,
        'accessors': accessors,
        'images': [{'bufferView': v, 'mimeType': mime} for v in image_views],
        'textures': textures,
        'materials': [{'pbrMetallicRoughness': {'baseColorTexture': {'index': i}}}
                      for i in range(len(images))],
        'meshes': meshes,
        'nodes': [{'mesh': i} for i in range(len(meshes))],
    }
    if basis:
        gltf['extensionsUsed'] = gltf['extensionsRequired'] = ['GOOGLE_texture_basis']
    text = json.dumps(gltf).encode()
    text += b' ' * (-len(text) % 4)
    body = (struct.pack('<I4s', len(text), b'JSON') + text
            + struct.pack('<I4s', len(binary), b'BIN\0') + binary)
    return struct.pack('<4sII', b'glTF', 2, 12 + len(body)) + body


def write_glb(path, vertices, faces, uv, image):
    """A single-mesh, single-texture .glb laid out as Gibson's are."""
    path.write_bytes(glb_bytes([(vertices, faces, uv, 0)], [jpeg(image)]))


def room_uv(vertices):
    return (vertices[:, :2] - vertices[:, :2].min(0)) / np.ptp(vertices[:, :2], 0)


@pytest.fixture(scope='module')
def room_glb(tmp_path_factory):
    vertices, faces = room_mesh()
    uv = room_uv(vertices)
    image = np.zeros((64, 64, 3), np.uint8)
    image[:32] = (200, 60, 40)   # top half red: where glTF's v = 0 points
    path = tmp_path_factory.mktemp('scenes') / 'Room.glb'
    write_glb(path, vertices, faces.reshape(-1), uv, image)
    return path


def test_read_glb_keeps_gltf_texture_coordinates(room_glb):
    [part] = scenes.read_glb(room_glb)
    expected_vertices, expected_faces = room_mesh()
    np.testing.assert_allclose(part.vertices, expected_vertices)
    np.testing.assert_array_equal(part.faces, expected_faces)
    # MuJoCo's usertexcoord has v down the image, as glTF does, so they pass through unflipped.
    np.testing.assert_allclose(part.uv, room_uv(expected_vertices))
    assert part.texture[0, 0, 0] > 150 and part.texture[-1, -1, 0] < 50


def square(offset):
    """Two triangles, a 1 m square at x = offset, and its texture coordinates."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    vertices[:, 0] += offset
    return vertices, np.array([[0, 1, 2], [0, 2, 3]]), vertices[:, :2] - [offset, 0]


def test_read_glb_groups_primitives_by_texture(tmp_path):
    red, blue = np.full((8, 8, 3), (220, 30, 30), np.uint8), np.full((8, 8, 3), (30, 30, 220),
                                                                      np.uint8)
    primitives = [(*square(0), 0), (*square(2), 1), (*square(4), 0)]
    path = tmp_path / 'two.glb'
    path.write_bytes(glb_bytes(primitives, [jpeg(red), jpeg(blue)]))
    parts = scenes.read_glb(path)
    assert [len(p.faces) for p in parts] == [4, 2]           # red twice, blue once
    assert parts[0].texture[0, 0, 0] > 150 and parts[1].texture[0, 0, 2] > 150
    # The second red square's faces index its own vertices, after the first square's four.
    np.testing.assert_allclose(parts[0].vertices[parts[0].faces[2:]].min((0, 1)), [4, 0, 0])


def test_debasis_glb_decodes_and_flips_hm3d_textures(tmp_path, monkeypatch):
    from drones.sim import basis

    # What basisu hands back for HM3D: bottom row first. Red rows first here means the texture
    # the glTF way up has red at the bottom.
    decoded = np.zeros((16, 16, 3), np.uint8)
    decoded[:8] = (220, 30, 30)
    decoded[8:] = (30, 30, 220)
    seen = []
    monkeypatch.setattr(basis, 'decode', lambda blobs: seen.extend(blobs) or [decoded])
    blob = glb_bytes([(*square(0), 0)], [b'not really basis'], basis=True)
    with pytest.raises(ValueError, match='x-basis'):
        (tmp_path / 'raw.glb').write_bytes(blob)
        scenes.read_glb(tmp_path / 'raw.glb')
    (tmp_path / 'plain.glb').write_bytes(scenes.debasis_glb(blob))
    assert [bytes(b) for b in seen] == [b'not really basis']
    [part] = scenes.read_glb(tmp_path / 'plain.glb')
    assert part.texture[0, 0, 2] > 150 and part.texture[-1, -1, 0] > 150   # blue on top now
    np.testing.assert_allclose(part.vertices, square(0)[0])   # geometry untouched
    gltf, _ = scenes._split_glb((tmp_path / 'plain.glb').read_bytes())
    assert 'extensionsRequired' not in gltf and gltf['images'][0]['mimeType'] == 'image/jpeg'


def test_load_puts_the_open_floor_at_the_origin(room_glb):
    scene = scenes.load(room_glb)
    # The floor ends up at z = 0 ...
    assert abs(scene.vertices[:, 2].min()) < 0.06
    # ... and the origin is in the open part, clear of the walls at y = +-2 and the pillar at x = 3.
    x0, _, y0, y1 = ROOM
    centre = -scene.vertices[:, :2].min(0) + [x0, y0]   # where the origin was in the scan
    assert x0 + 1.5 < centre[0] < PILLAR[0] - 1.5 and abs(centre[1]) < 0.3
    np.testing.assert_allclose(scene.origin[:2], centre, atol=1e-5)
    assert abs(scene.origin[2] - FLOOR_Z) < 0.06
    assert 1.5 <= scene.clearance <= (y1 - y0) / 2


def test_attach_changes_only_the_rendered_model(room_glb):
    import mujoco

    from drones.sim.square_env import SquareConfig, SquareEnv

    env = SquareEnv(SquareConfig(num_envs=1))
    before = env.sim.mj_model
    mjx_model = env.sim.mjx_model
    lower, upper = scenes.attach(env.sim, scenes.load(room_glb))
    model = env.sim.mj_model
    assert env.sim.mjx_model is mjx_model
    assert (model.nq, model.nmocap) == (before.nq, before.nmocap)
    assert len(lower) == len(upper) == 1 and model.ngeom == before.ngeom + 2
    assert (model.geom_group[lower] == scenes.LOWER_GROUP).all()
    assert (model.geom_group[upper] == scenes.UPPER_GROUP).all()
    assert not model.geom_contype[lower + upper].any()
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    assert model.geom_group[floor] == 3   # hidden: the scan brings its own floor


@needs_gl
def test_render_square_films_inside_a_scene(request, room_glb, tmp_path):
    import imageio.v2 as imageio

    run = request.getfixturevalue('saved_square_run')

    out = tmp_path / 'flight.gif'
    done = subprocess.run(
        [sys.executable, '-m', 'drones.rl.render_square', str(run), '--out', str(out),
         '--scene', str(room_glb), '--camera', 'top', '--seconds', '0.4', '--width', '64',
         '--height', '48'],
        env=gl_env(), capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert 'Scene Room:' in done.stdout and f'Wrote {out}' in done.stdout
    frames = imageio.mimread(out, memtest=False)
    # Looking down with the ceiling hidden, the red-textured floor fills the frame.
    red = frames[0][..., 0].astype(int) - frames[0][..., 2]
    assert (red > 60).mean() > 0.3
