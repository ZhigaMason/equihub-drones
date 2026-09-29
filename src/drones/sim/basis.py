"""Decode Basis Universal textures, as HM3D's .basis.glb scenes carry them, to RGB arrays.

No Python package decodes a raw .basis file: pybasis has no wheels, and texture2ddecoder decodes
GPU blocks, not Basis's supercompressed container. Binomial's reference encoder, basisu, does it
with `-unpack`, and its repository ships the tool compiled to WebAssembly (bin/basisu_st.wasm, a
WASI program). That runs under the `wasmtime` wheel, so nobody needs a C++ toolchain.
The .wasm is fetched once from a pinned commit, checked against its SHA-256, and cached.

HM3D's textures are ETC1S (under 1 bit per texel). `-format_only 0` unpacks them through ETC1, of
which ETC1S is a subset, so the PNG is exact. It is decoded identically to a natively built basisu,
checked pixel for pixel on HM3D scene 00446.
"""
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

COMMIT = '99f52d63aa6799cbdaecfe977111dc5ec3b31d47'   # BinomialLLC/basis_universal, 2026-09-01
WASM_URL = f'https://raw.githubusercontent.com/BinomialLLC/basis_universal/{COMMIT}/bin/basisu_st.wasm'
WASM_SHA256 = 'b42d951b1bf146133578e8c7927ad4a4a857552846a46a3ee33b541b2a06bc7d'


def cache_dir():
    return Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'drones'


def wasm_path():
    """The pinned basisu_st.wasm, downloaded and verified on first use."""
    path = cache_dir() / f'basisu_st-{COMMIT[:12]}.wasm'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print('Fetching the Basis Universal decoder (basisu_st.wasm, 8 MB) ...', flush=True)
        data = urllib.request.urlopen(WASM_URL).read()
        if hashlib.sha256(data).hexdigest() != WASM_SHA256:
            raise RuntimeError(f'{WASM_URL} does not match its pinned SHA-256; not using it')
        partial = path.with_suffix('.part')
        partial.write_bytes(data)
        partial.rename(path)
    return path


def decode(blobs):
    """RGB arrays (H, W, 3) uint8 of the full-resolution level of each .basis file in `blobs`.

    One run of basisu for all of them: instantiating the module costs more than a texture.
    """
    try:
        import wasmtime
    except ImportError as error:
        raise ImportError('Decoding HM3D textures needs wasmtime:  uv sync --extra sim') from error
    from PIL import Image

    engine = wasmtime.Engine()
    module = wasmtime.Module.from_file(engine, str(wasm_path()))
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        names = [f'texture{i:03d}' for i in range(len(blobs))]
        for name, blob in zip(names, blobs, strict=True):
            (work / f'{name}.basis').write_bytes(bytes(blob))
        config = wasmtime.WasiConfig()
        config.argv = ['basisu', '-unpack', '-no_ktx', '-format_only', '0',
                       '-output_path', '/work', *[f'/work/{n}.basis' for n in names]]
        config.preopen_dir(str(work), '/work')
        log = work / 'basisu.log'
        config.stdout_file = str(log)
        config.stderr_file = str(log)
        store = wasmtime.Store(engine)
        store.set_wasi(config)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        try:
            linker.instantiate(store, module).exports(store)['_start'](store)
            code = 0
        except wasmtime.ExitTrap as exit_:
            code = exit_.code
        images = []
        for name in names:
            level0 = work / f'{name}_unpacked_rgb_ETC1_RGB_0_0000.png'
            if code != 0 or not level0.exists():
                raise RuntimeError(f'basisu failed on {name} (exit {code}):\n'
                                   + log.read_text(errors='replace')[-2000:])
            images.append(np.asarray(Image.open(level0).convert('RGB')))
    return images
