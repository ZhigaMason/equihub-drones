"""Scanned indoor scenes from the IndoorUAV dataset, as a backdrop for rendering and exploring.

    uv run --extra sim drones-download-scenes                   # the default ten Gibson scenes
    uv run --extra sim drones-download-scenes Adrian Bowlus     # any Gibson scene by name
    uv run --extra sim drones-download-scenes 00006-HkseAnWCgqk # any HM3D scene, by id or hash
    uv run --extra sim drones-download-scenes --benchmark hm-eqa --count 5   # see drones.sim.eqa
    uv run --extra sim drones-download-scenes --benchmark indoor-uav        # IndoorUAV's own

IndoorUAV (valyentine/Indoor_UAV on ModelScope, AAAI 2026) ships its habitat-sim scenes as one
47 GB `scene_datasets.zip`. The CDN honours HTTP range requests, so `download` reads the zip's
central directory and pulls out single members (~10-50 MB each) without fetching the rest.

Two of its datasets are used, both z-up in metres, the same frame as ours:

- Gibson: one self-contained .glb per scene, a single mesh with one embedded 16k JPEG texture.
  Stored as downloaded, scenes/<Name>.glb.
- HM3D (all 900 train and val scenes): a .basis.glb of ~66 meshes over ~12 textures, which are
  Basis Universal files no image library reads. `download` decodes them once (drones.sim.basis)
  and stores a plain .glb with JPEG textures, scenes/hm3d/<00006-HkseAnWCgqk>.glb. The EQA
  benchmarks in drones.sim.eqa are all built on these scenes.

Replica's meshes are 100-200 MB .ply files and are not used.

A scene is visual only. It is added to the renderer's MjModel (`attach`) and never to the MJX
model the dynamics run on: nothing collides with it and no sensor sees it. The square task observes
the state estimate, so the policy's behaviour is unchanged by the scene around it.
"""
import argparse
import io
import json
import struct
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

DATASET = ('valyentine', 'Indoor_UAV')
ARCHIVE = 'scene_datasets.zip'
MEMBER = 'scene_datasets/gibson/{}.glb'
HM3D_MEMBER = 'scene_datasets/hm3d/{id}/{hash}.basis.glb'
SCENES_DIR = Path('scenes')
HM3D_JPEG_QUALITY = 92
# The ten Gibson scenes with the most trajectories in IndoorUAV's train.csv.
DEFAULT_SCENES = ('Nemacolin', 'Reyno', 'Capistrano', 'Roeville', 'Bowlus', 'Mosquito', 'Ballou',
                  'Goffs', 'Mesic', 'Soldier')

# Gibson textures are 16384 x 16384 JPEGs, 768 MB decoded. JPEG draft mode decodes at 1/4 scale
# directly, in a quarter of a second, and 4096 is plenty for a camera a metre or two away.
TEXTURE_SIZE = 4096
# Geometry between these heights above the floor counts as an obstacle: below is rug and skirting,
# above is out of the flight envelope (the square flies at 0.8-1.2 m).
OBSTACLE_LOW = 0.15    # m
CUT_HEIGHT = 2.0       # m; the top camera hides everything above this, walls and ceiling included
CELL = 0.1             # m, the occupancy grid the free spot is searched on
# Scans carry their lighting in the texture, and CrazyFlow's headlight (0.5 ambient + 0.6 diffuse)
# on top of it washes them out. These show a scan roughly as captured.
HEADLIGHT_AMBIENT, HEADLIGHT_DIFFUSE = 0.35, 0.4
LOWER_GROUP, UPPER_GROUP = 1, 2   # the top camera moves UPPER_GROUP to its hidden group


# ------------------------------------------------------------------ download
class _HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, enough for zipfile."""

    def __init__(self, url):
        head = urllib.request.urlopen(urllib.request.Request(url, method='HEAD'))
        self.url, self.size, self.pos = head.url, int(head.headers['Content-Length']), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def readinto(self, buffer):
        end = min(self.pos + len(buffer), self.size)
        if end <= self.pos:
            return 0
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end - 1}'})
        data = urllib.request.urlopen(request).read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def archive_url(file=ARCHIVE):
    """The download URL of `file` in IndoorUAV's ModelScope repository."""
    from modelscope.hub.api import HubApi

    namespace, name = DATASET
    return HubApi().get_dataset_file_url(file, name, namespace)


def is_hm3d(name):
    """HM3D scenes are named by their id, 00006-HkseAnWCgqk, or its hash alone, HkseAnWCgqk;
    Gibson scenes by a capitalised word, Adrian."""
    return '-' in name or not name[:1].isupper()


def scene_path(name, dest=SCENES_DIR):
    """Where `download` puts scene `name` (an HM3D hash alone resolves through hm3d_index)."""
    dest = Path(dest)
    if not is_hm3d(name):
        return dest / f'{name}.glb'
    if '-' not in name:
        name = hm3d_index(dest).get(name, name)
    return dest / 'hm3d' / f'{name}.glb'


def _open_archive():
    return zipfile.ZipFile(io.BufferedReader(_HttpFile(archive_url()), buffer_size=1 << 20))


def _hm3d_ids(archive):
    """{hash: id} of every HM3D scene in the archive."""
    ids = {}
    for member in archive.namelist():
        parts = member.split('/')
        if len(parts) == 4 and parts[1] == 'hm3d' and parts[3].endswith('.basis.glb'):
            ids[parts[2].split('-', 1)[1]] = parts[2]
    return ids


def hm3d_index(dest=SCENES_DIR):
    """{hash: id} of the HM3D scenes in IndoorUAV, cached in dest/hm3d/index.json.

    OpenEQA names scenes by hash alone; everything else, and the archive, by id.
    """
    path = Path(dest) / 'hm3d' / 'index.json'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with _open_archive() as archive:
            path.write_text(json.dumps(_hm3d_ids(archive), indent=0, sort_keys=True))
    return json.loads(path.read_text())


def download(names, dest=SCENES_DIR):
    """Fetch scenes `names` (Gibson names, HM3D ids or hashes), skipping any already there."""
    todo = [n for n in names if not scene_path(n, dest).exists()]
    if not todo:
        return
    with _open_archive() as archive:
        ids = _hm3d_ids(archive)
        for name in todo:
            if is_hm3d(name):
                hm3d_id = name if '-' in name else ids.get(name)
                if hm3d_id is None or hm3d_id.split('-', 1)[1] not in ids:
                    sys.exit(f'No HM3D scene {name!r} in {ARCHIVE}')
                member = HM3D_MEMBER.format(id=hm3d_id, hash=hm3d_id.split('-', 1)[1])
            else:
                member = MEMBER.format(name)
                if member not in archive.NameToInfo:
                    sys.exit(f'No Gibson scene {name!r} in {ARCHIVE}')
            info = archive.getinfo(member)
            print(f'{name}: {info.file_size / 1e6:.0f} MB', flush=True)
            target = scene_path(name if not is_hm3d(name) else hm3d_id, dest)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src:
                blob = src.read()
            if is_hm3d(name):
                print(f'{name}: decoding Basis textures', flush=True)
                blob = debasis_glb(blob)
            part = target.with_suffix('.glb.part')
            part.write_bytes(blob)
            part.rename(target)


def debasis_glb(blob):
    """A .basis.glb rewritten as a plain .glb: every Basis image decoded and stored as JPEG.

    Only the images' buffer views change; every accessor keeps its buffer view index, so the
    geometry is byte for byte what IndoorUAV ships.
    """
    from PIL import Image

    from drones.sim import basis

    gltf, binary = _split_glb(blob)
    views = gltf['bufferViews']

    def view(i):
        v = views[i]
        return binary[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']]

    images = gltf.get('images', [])
    targets = [i for i, image in enumerate(images) if image.get('mimeType') == 'image/x-basis']
    decoded = basis.decode([view(images[i]['bufferView']) for i in targets])
    replaced = {}
    for i, rgb in zip(targets, decoded, strict=True):
        out = io.BytesIO()
        # HM3D's .basis images are stored bottom row first, the other way up from glTF's own
        # images: unflipped, every triangle samples the wrong patch of the atlas and the scan
        # renders as confetti. Checked by eye on scene 00006, which is right only flipped.
        Image.fromarray(rgb[::-1]).save(out, 'JPEG', quality=HM3D_JPEG_QUALITY)
        replaced[images[i]['bufferView']] = out.getvalue()
        images[i]['mimeType'] = 'image/jpeg'
    for texture in gltf.get('textures', []):
        source = texture.pop('extensions', {}).get('GOOGLE_texture_basis', {}).get('source')
        if source is not None:
            texture['source'] = source
    for key in ('extensionsUsed', 'extensionsRequired'):
        if key in gltf:
            gltf[key] = [e for e in gltf[key] if e != 'GOOGLE_texture_basis']
            if not gltf[key]:
                del gltf[key]
    chunks, offset = [], 0
    for i, v in enumerate(views):
        data = replaced.get(i, view(i))
        v['byteOffset'], v['byteLength'] = offset, len(data)
        padding = -len(data) % 4
        chunks.append(bytes(data) + b'\0' * padding)
        offset += len(data) + padding
    gltf['buffers'] = [{'byteLength': offset}]
    return _join_glb(gltf, b''.join(chunks))


def _split_glb(blob):
    """(glTF JSON, BIN chunk) of a .glb."""
    length = struct.unpack('<I', blob[12:16])[0]
    gltf = json.loads(blob[20:20 + length])
    start = 20 + length
    binary_length = struct.unpack('<I', blob[start:start + 4])[0]
    return gltf, memoryview(blob)[start + 8:start + 8 + binary_length]


def _join_glb(gltf, binary):
    text = json.dumps(gltf, separators=(',', ':')).encode()
    text += b' ' * (-len(text) % 4)
    binary += b'\0' * (-len(binary) % 4)
    body = (struct.pack('<I4s', len(text), b'JSON') + text
            + struct.pack('<I4s', len(binary), b'BIN\0') + binary)
    return struct.pack('<4sII', b'glTF', 2, 12 + len(body)) + body


# ------------------------------------------------------------------ loading
_COMPONENT = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3}


@dataclass
class Part:
    """The triangles that share one texture."""
    vertices: np.ndarray   # (V, 3) float32
    faces: np.ndarray      # (F, 3) int32
    uv: np.ndarray         # (V, 2) float32, v down the image: glTF's convention and, for
                           # usertexcoord, MuJoCo's (its OBJ loader flips v to get there)
    texture: np.ndarray    # (H, W, 3) uint8


@dataclass
class Scene:
    """A scan, moved so `origin` (in the file's frame) is at the world origin: its floor at z = 0.

    load() puts the most open patch of floor there; drones.sim.explore re-anchors a scene on a
    benchmark question's start pose by moving the geoms instead of reloading.
    """
    name: str
    parts: list            # of Part, in world coordinates
    origin: np.ndarray     # (3,) the world origin, in the file's frame
    clearance: float       # m: the square box of this half-width around the origin is clear

    @property
    def vertices(self):
        return np.concatenate([p.vertices for p in self.parts])

    @property
    def faces(self):
        offsets = np.cumsum([0] + [len(p.vertices) for p in self.parts[:-1]])
        return np.concatenate([p.faces + o for p, o in zip(self.parts, offsets, strict=True)])


def _decode_image(data, mime):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None   # a known 16k Gibson texture, not a decompression bomb
    image = Image.open(io.BytesIO(bytes(data)))
    if mime == 'image/jpeg':
        image.draft('RGB', (TEXTURE_SIZE, TEXTURE_SIZE))
    image = image.convert('RGB')
    if max(image.size) > TEXTURE_SIZE:
        image = image.resize((TEXTURE_SIZE, TEXTURE_SIZE))
    return np.asarray(image)


def read_glb(path):
    """The Parts of a .glb: its triangles grouped by texture, in the file's frame.

    Handles what Gibson and (debasis'd) HM3D files contain -- indexed triangles with one texture
    coordinate set, JPEG or PNG images, and nodes without transforms. Anything else is refused
    rather than drawn wrong.
    """
    gltf, binary = _split_glb(Path(path).read_bytes())

    def view(i):
        v = gltf['bufferViews'][i]
        offset = v.get('byteOffset', 0)
        return binary[offset:offset + v['byteLength']]

    def accessor(i):
        a = gltf['accessors'][i]
        width = _WIDTH[a['type']]
        array = np.frombuffer(view(a['bufferView']), _COMPONENT[a['componentType']],
                              a['count'] * width, a.get('byteOffset', 0))
        return array.reshape(a['count'], width) if width > 1 else array

    for node in gltf.get('nodes', []):
        if any(k in node for k in ('matrix', 'translation', 'rotation', 'scale')):
            raise ValueError(f'{path}: node transforms are not supported')
    images, groups = {}, {}
    for mesh in gltf['meshes']:
        for primitive in mesh['primitives']:
            if primitive.get('mode', 4) != 4 or 'TEXCOORD_0' not in primitive['attributes']:
                continue   # only textured triangles
            material = gltf['materials'][primitive['material']]
            texture = material['pbrMetallicRoughness']['baseColorTexture']['index']
            texture = gltf['textures'][texture]
            source = texture.get('source', texture.get('extensions', {}).get(
                'GOOGLE_texture_basis', {}).get('source'))   # an un-debasis'd HM3D file
            if source not in images:
                image = gltf['images'][source]
                if image.get('mimeType') not in ('image/jpeg', 'image/png'):
                    raise ValueError(f'{path}: {image.get("mimeType")} image; '
                                     f'HM3D scenes must come through drones-download-scenes')
                images[source] = _decode_image(view(image['bufferView']), image['mimeType'])
            groups.setdefault(source, []).append(primitive)
    parts = []
    for source, primitives in groups.items():
        vertices, faces, uv, offset = [], [], [], 0
        for primitive in primitives:
            v = accessor(primitive['attributes']['POSITION']).astype(np.float32)
            vertices.append(v)
            uv.append(accessor(primitive['attributes']['TEXCOORD_0']).astype(np.float32))
            faces.append(accessor(primitive['indices']).astype(np.int32).reshape(-1, 3) + offset)
            offset += len(v)
        parts.append(Part(np.concatenate(vertices), np.concatenate(faces), np.concatenate(uv),
                          images[source]))
    return parts


# Barycentric weights of the points sampled on every triangle: corners, centre, edge midpoints and
# three inner points. Scanned walls and floors can be triangles half a metre across, whose corners
# alone leave holes in a 0.1 m grid.
_SAMPLES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1],
                     [4, 1, 1], [1, 4, 1], [1, 1, 4]], np.float32)
_SAMPLES /= _SAMPLES.sum(1, keepdims=True)


def _floor_levels(tri, up, area, count=4):
    """Heights of up to `count` storeys: the peaks of upward-facing area, at least 1 m apart."""
    z = tri[up, :, 2].mean(1)
    hist, edges = np.histogram(z, np.arange(z.min(), z.max() + 0.1, 0.05), weights=area[up])
    levels = []
    for i in np.argsort(hist)[::-1]:
        level = float(edges[i] + 0.025)
        if hist[i] < 0.1 * hist.max() or len(levels) == count:
            break
        if all(abs(level - other) > 1.0 for other in levels):
            levels.append(level)
    return levels


def free_spot(vertices, faces):
    """(floor z, centre xy, clearance) of the most open patch of floor on any storey.

    Clearance is a Chebyshev distance, so the whole square box of that half-width is floor with
    nothing between OBSTACLE_LOW and CUT_HEIGHT above it -- room for a 1 m square at any rotation.
    """
    import scipy.ndimage   # after crazyflow: drones.sim imports it first

    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(normal, axis=1)
    up = normal[:, 2] > 0.9 * np.maximum(area, 1e-12)
    samples = np.einsum('sk,tkd->tsd', _SAMPLES, tri)
    lo = vertices[:, :2].min(0)
    shape = tuple(np.ceil((vertices[:, :2].max(0) - lo) / CELL).astype(int) + 1)

    def mask(points):
        grid = np.zeros(shape, bool)
        grid[tuple(((points[:, :2] - lo) / CELL).astype(int).T)] = True
        return grid

    best = (-1, 0.0, None)
    for floor in _floor_levels(tri, up, area):
        flat = samples[up]
        flat = flat[np.abs(flat[:, :, 2] - floor).max(1) < 0.1].reshape(-1, 3)
        # Closing fills the few-cell holes a scan leaves in a floor; it does not bridge rooms.
        has_floor = scipy.ndimage.binary_closing(mask(flat), iterations=2)
        points = samples.reshape(-1, 3)
        height = points[:, 2] - floor
        blocked = mask(points[(height > OBSTACLE_LOW) & (height < CUT_HEIGHT)])
        clearance = scipy.ndimage.distance_transform_cdt(np.pad(has_floor & ~blocked, 1),
                                                         metric='chessboard')[1:-1, 1:-1]
        cell = np.unravel_index(np.argmax(clearance), shape)
        if clearance[cell] > best[0]:
            best = (clearance[cell], floor, lo + (np.array(cell) + 0.5) * CELL)
    clearance, floor, centre = best
    return floor, centre, (clearance - 0.5) * CELL


def load(path):
    """A Scene from a .glb, moved so its most open patch of floor is at the origin."""
    path = Path(path)
    parts = read_glb(path)
    scene = Scene(path.stem, parts, np.zeros(3), 0.0)
    floor, centre, clearance = free_spot(scene.vertices, scene.faces)
    origin = np.array([centre[0], centre[1], floor], np.float32)
    for part in parts:
        part.vertices = part.vertices - origin
    scene.origin, scene.clearance = origin.astype(float), float(clearance)
    return scene


# ------------------------------------------------------------------ rendering
def add_to_spec(spec, scene):
    """Add `scene` to `spec`'s world body as visual-only mesh geoms: per texture, one below
    CUT_HEIGHT (LOWER_GROUP) and one above it (UPPER_GROUP)."""
    for i, part in enumerate(scene.parts):
        name = f'scene_{scene.name}_{i}'
        tex = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                               width=part.texture.shape[1], height=part.texture.shape[0],
                               nchannel=3)
        tex.data = part.texture.tobytes()
        material = spec.add_material(name=name, specular=0.0, shininess=0.0)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        upper = part.vertices[part.faces, 2].max(1) > CUT_HEIGHT
        for half, faces, group in (('lower', part.faces[~upper], LOWER_GROUP),
                                   ('upper', part.faces[upper], UPPER_GROUP)):
            if not len(faces):
                continue
            # Each mesh gets its own compacted vertex list: MuJoCo rejects unreferenced vertices.
            used, faces = np.unique(faces, return_inverse=True)
            if len(used) < 4:
                continue   # MuJoCo refuses a mesh this small; a lone triangle is not missed
            mesh = spec.add_mesh(name=f'{name}_{half}', uservert=part.vertices[used].ravel(),
                                 usertexcoord=part.uv[used].ravel(),
                                 userface=faces.reshape(-1).astype(np.int32))
            # Visual only; 'shell' inertia stops MuJoCo computing a volume for an open scan.
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            spec.worldbody.add_geom(name=mesh.name, type=mujoco.mjtGeom.mjGEOM_MESH,
                                    meshname=mesh.name, material=material.name, group=group,
                                    contype=0, conaffinity=0, mass=0.0)


def attach(sim, scene):
    """Give `sim`'s renderer a model with `scene` in it. Returns (lower, upper): the ids of the
    scene's geoms below and above CUT_HEIGHT.

    Only sim.mj_model and sim.mj_data change. The scene is static world geometry, so qpos and the
    mocap bodies line up with the MJX model's and Sim.render copies them across as before.
    CrazyFlow's own floor plane is hidden: the scan has a floor of its own.
    """
    spec = sim.spec.copy()
    add_to_spec(spec, scene)
    model = spec.compile()
    sim._unweld_drones(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if floor != -1:
        model.geom_group[floor] = 3
    model.vis.quality.offsamples = 4
    model.vis.headlight.ambient[:] = HEADLIGHT_AMBIENT
    model.vis.headlight.diffuse[:] = HEADLIGHT_DIFFUSE
    sim.close()
    sim.mj_model, sim.mj_data = model, mujoco.MjData(model)
    ids = {half: [] for half in ('lower', 'upper')}
    for i in range(len(scene.parts)):
        for half in ids:
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                     f'scene_{scene.name}_{i}_{half}')
            if geom != -1:
                ids[half].append(geom)
    return ids['lower'], ids['upper']


def main(argv=None):
    parser = argparse.ArgumentParser(description='Download IndoorUAV scenes (ModelScope) for '
                                                 'drones-render-square and drones-explore-scene.')
    parser.add_argument('names', nargs='*',
                        help='Gibson names or HM3D ids (default: the ten Gibson scenes '
                             f'{" ".join(DEFAULT_SCENES)}, unless --benchmark is given)')
    parser.add_argument('--benchmark',
                        choices=('hm-eqa', 'mt-hm3d', 'express-bench', 'a-eqa', 'indoor-uav'),
                        help="also fetch this benchmark's questions and its scenes, those with "
                             'the most questions first')
    parser.add_argument('--count', type=int, default=3,
                        help='how many of the benchmark\'s scenes (default: 3, ~30 MB each)')
    parser.add_argument('--dest', type=Path, default=SCENES_DIR)
    args = parser.parse_args(argv)
    names = list(args.names)
    if args.benchmark:
        from drones.sim import eqa

        busiest = eqa.busiest(args.benchmark, args.count, args.dest)
        eqa.prepare(args.benchmark, busiest, args.dest)
        names += busiest
    elif not names:
        names = list(DEFAULT_SCENES)
    download(names, args.dest)
    for name in names:
        print(scene_path(name, args.dest))


if __name__ == '__main__':
    main()
