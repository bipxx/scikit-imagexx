"""Capture frozen original 0.26 isotropic endpoint and mutation observations."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import scipy
from scipy.ndimage import _morphology, _nd_image
import skimage
from skimage import morphology

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = 'f2675a3a426973aee06fbaa79e9d45e2ab7ee80677df00525dba0de789278085'
OPERATIONS = ('erosion', 'dilation', 'opening', 'closing')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def cases():
    mask = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0],
                     [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=bool)
    for kind in OPERATIONS:
        for name, image, radius, spacing in (
            ('threshold', mask, 1., None),
            ('sqrt-before', mask, np.nextafter(np.sqrt(2.), 0.).item(), None),
            ('sqrt-exact', mask, np.sqrt(2.).item(), None),
            ('all-true', np.ones((4, 5), dtype=bool), 2., None),
            ('all-false', np.zeros((4, 5), dtype=bool), 2., None),
            ('all-true-anisotropic', np.ones((4, 5), dtype=bool), 1.7, (0.75, 1.5)),
            ('all-false-volume', np.zeros((3, 3, 4), dtype=bool), 1.7, (0.75, 0.5, 1.25)),
            ('large-radius', mask, 20., (0.625, 1.75)),
            ('anisotropic', mask, 1.1, [0.75, 1.5]),
            ('volume', np.stack((mask, mask[::-1], mask.T)), 1.6, (0.7, 1.3, 0.9)),
            ('line', mask[2].copy(), 0.5, 0.5),
            ('negative', mask, -1, None),
            ('negative-inf', mask, -float('inf'), None),
            ('positive-inf', mask, float('inf'), None),
            ('nan', mask, float('nan'), None),
            ('negative-zero', mask, -0., None),
        ):
            yield f'{kind}:{name}', kind, image.copy(), dict(radius=radius, spacing=spacing), None, True
        image = mask.copy()
        yield f'{kind}:out-alias', kind, image, dict(radius=1.), image, True
        storage = np.zeros(mask.size + 1, dtype=bool)
        image, out = storage[:-1].reshape(mask.shape), storage[1:].reshape(mask.shape)
        image[...] = mask
        yield f'{kind}:out-overlap', kind, image, dict(radius=1.), out, True
        yield f'{kind}:out-disjoint', kind, mask.copy(), dict(radius=1.25), np.ones_like(mask), True
        yield f'{kind}:strided', kind, mask[:, ::-1], dict(radius=1.), None, False
        yield f'{kind}:fortran', kind, np.asfortranarray(mask), dict(radius=1.), None, False
        yield f'{kind}:numeric', kind, np.where(mask, -2, 0).astype(np.int16), dict(radius=1.), None, False
        yield f'{kind}:dtype-out', kind, mask.copy(), dict(radius=1.), np.zeros(mask.shape, dtype=np.int32), False
        yield f'{kind}:out-stride', kind, mask.copy(), dict(radius=1.), np.zeros_like(mask)[:, ::-1], False
        yield f'{kind}:out-bad-shape', kind, mask.copy(), dict(radius=1.), np.zeros((4, 3), dtype=bool), False
        readonly = np.zeros_like(mask)
        readonly.flags.writeable = False
        yield f'{kind}:out-readonly', kind, mask.copy(), dict(radius=1.), readonly, False
        yield f'{kind}:spacing-length', kind, mask.copy(), dict(radius=1., spacing=(1.,)), None, False
        yield f'{kind}:zero-spacing', kind, mask.copy(), dict(radius=1., spacing=(0., 1.)), None, False
        yield f'{kind}:radius-array', kind, mask.copy(), dict(radius=np.ones(mask.shape) * 1.1), None, False
        yield f'{kind}:scalar', kind, np.array(True), dict(radius=1.), None, False
        yield f'{kind}:empty', kind, np.empty((0, 4), dtype=bool), dict(radius=1.), None, False
        yield f'{kind}:rank-four', kind, mask[None, None].copy(), dict(radius=1.), None, False


def snapshot(array):
    return dict(shape=list(array.shape), strides=list(array.strides), dtype=array.dtype.str,
                data=array.tobytes().hex(), writeable=bool(array.flags.writeable))


def observe(target, image, options, out):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            value = target(image, out=out, **options)
        except Exception as error:
            row = dict(error=[type(error).__module__, type(error).__name__, str(error)])
        else:
            row = dict(result=snapshot(value), returns_out=out is not None and value is out,
                       shares_input=np.shares_memory(value, image), owns=bool(value.flags.owndata))
    row.update(image_after=snapshot(image), out_after=None if out is None else snapshot(out),
               warnings=[[w.category.__module__, w.category.__name__, str(w.message)] for w in notices])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    version = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, original=skimage.__version__)
    assert tuple(version.values()) == ('3.13.14', '2.5.4', '1.18.1', '0.26.0')
    paths = dict(original=Path(inspect.getsourcefile(morphology.isotropic_erosion)),
                 scipy_wrapper=Path(_morphology.__file__), scipy_binary=Path(_nd_image.__file__),
                 observer=Path(__file__), benchmark=ROOT/'benchmarks/isotropic_morphology.py')
    sources = {name: sha(path.read_bytes()) for name, path in paths.items()}
    assert sources['original'] == ORIGINAL_SHA256
    bench = importlib.util.spec_from_file_location('isotropic_cases', paths['benchmark'])
    module = importlib.util.module_from_spec(bench)
    bench.loader.exec_module(module)
    rows = []
    for name, kind, image, options, out, native in cases():
        rows.append(dict(id=name, native=native, input=snapshot(image),
                         result=observe(getattr(morphology, 'isotropic_' + kind), image, options, out)))
    endpoint = []
    for spec in module.case_specs():
        (image,), options = module.make_case(spec)
        result = getattr(morphology, spec['api'].rsplit('.', 1)[1])(image, **options)
        endpoint.append(dict(id=spec['id'], input_sha256=sha(image.tobytes()), output_sha256=sha(result.tobytes()), strides=list(result.strides)))
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(dict(version=version, sources=sources, rows=rows, benchmark=endpoint), indent=2) + '\n')
    print(f'Frozen {len(rows)} original observations and {len(endpoint)} benchmark endpoints')


if __name__ == '__main__':
    main()
