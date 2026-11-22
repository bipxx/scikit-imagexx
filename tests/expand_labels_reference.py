"""Frozen original expand_labels observations for the family's initial domain."""

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
from scipy.ndimage import _nd_image, _morphology
import skimage
from skimage.segmentation import expand_labels


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA256 = 'f517d235d8cf95aec3fcc6b1b0ef9f6eae36cfffeb149c477f1ed95beabc4044'
MORPHOLOGY_SHA256 = 'acc30fc9bf11de6fef46353bb26dd0a86a3917945fd185a9a50534bde779d8ba'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def make_cases():
    line = [0, 10, 0, 0, 0, 20, 0]
    grid = [[0, 4, 0, 0], [8, 0, 0, 0], [0, 3, 0, 0]]
    diagonal = [[0, 11, 0], [99, 0, 0], [0, 0, 5]]
    volume = np.zeros((3, 3, 3), dtype=np.int16)
    volume[0, 0, 1] = 4
    volume[0, 1, 0] = 8
    volume[1, 0, 0] = 16
    volume[2, 2, 2] = -7
    yield 'default-line', np.array(line, dtype=np.int8), {}, True
    yield 'line-tie', np.array(line, dtype=np.uint16), dict(distance=3), True
    yield 'diagonal-reverse-lex-tie', np.array(diagonal), dict(distance=1), True
    yield 'competing-default', np.array(grid, dtype=np.uint8), {}, True
    yield 'competing-anisotropic', np.array(grid, dtype=np.int32), dict(spacing=[1, 0.5]), True
    yield 'decimal-spacing', np.array(grid, dtype=np.int32), dict(distance=1.4, spacing=(0.7, 1.3)), True
    yield 'diagonal-sqrt-edge', np.array(diagonal), dict(distance=np.sqrt(2).item()), True
    yield 'diagonal-sqrt-before', np.array(diagonal), dict(distance=np.nextafter(np.sqrt(2), 0).item()), True
    yield 'volume-tie', volume, dict(distance=2), True
    yield 'volume-anisotropic', volume, dict(distance=1.5, spacing=(0.75, 1.5, 0.75)), True
    yield 'singleton-axis', volume[:, :1, :].copy(), dict(distance=2), True
    yield 'dense-no-background', np.arange(1, 15, dtype=np.int16).reshape(2, 7), dict(distance=7), True
    yield 'no-labels', np.zeros((3, 5), dtype=np.uint64), dict(distance=3), True
    yield 'no-labels-infinite', np.zeros((3, 5), dtype=np.int64), dict(distance=float('inf')), True
    yield 'no-labels-volume', np.zeros((2, 3, 4), dtype=np.uint16), dict(distance=3), True
    for title, distance in (('zero', 0), ('signed-zero', -0.), ('negative', -1),
                            ('negative-inf', -float('inf')), ('nan', float('nan')), ('inf', float('inf'))):
        yield 'radius-' + title, np.array(grid, dtype=np.int64), dict(distance=distance), True
    for dtype in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'):
        info = np.iinfo(dtype)
        image = np.array([[0, info.max, 0], [info.min or 1, 0, 0]], dtype=dtype)
        yield 'words-' + dtype, image, dict(distance=3), True
    yield 'fortran-layout', np.array(grid, dtype=np.int16, order='F'), dict(distance=2), False
    yield 'reversed-layout', np.array(grid, dtype=np.uint32)[:, ::-1], dict(distance=2), False
    yield 'transposed-layout', volume.transpose(2, 0, 1), dict(distance=2), False
    yield 'foreign-owner', np.frombuffer(bytearray(np.array(line, dtype=np.int16).tobytes()), dtype=np.int16), {}, False
    yield 'opposite-endian', np.array(grid, dtype='>u2'), dict(distance=2), False
    yield 'dtype-metadata', np.array(grid, dtype=np.dtype('int16', metadata={'field': 'labels'})), {}, False
    yield 'floating-labels', np.array(grid, dtype=np.float32) / 2, dict(distance=2), False
    yield 'boolean-labels', np.array(grid, dtype=bool), dict(distance=2), False
    yield 'empty-axis', np.zeros((2, 0, 3), dtype=np.int32), dict(distance=2), False
    yield 'scalar-label', np.asarray(1, dtype=np.int64), {}, False
    yield 'rank-four', volume[None, ...], dict(distance=2), False
    yield 'wrong-spacing-length', np.array(grid), dict(spacing=(1,)), False
    yield 'zero-spacing', np.array(grid), dict(spacing=(1., 0.)), False
    yield 'negative-spacing', np.array(grid), dict(spacing=(-1., 1.)), False
    yield 'none-spacing', np.array(grid), dict(spacing=None), False
    yield 'wrong-distance-type', np.array(grid), dict(distance='near'), False


def snapshot(array):
    return dict(shape=list(array.shape), strides=list(array.strides), dtype=array.dtype.str,
                dtype_metadata=None if array.dtype.metadata is None else dict(array.dtype.metadata),
                bytes_hex=array.tobytes().hex(), writes=bool(array.flags.writeable))


def observe(image, options):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            result = expand_labels(image, **options)
        except Exception as exc:
            output = dict(error=[type(exc).__module__, type(exc).__name__, str(exc)])
        else:
            output = dict(result=snapshot(result), aliases=np.shares_memory(result, image))
    output['warnings'] = [[item.category.__module__, item.category.__name__, str(item.message)] for item in notices]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert (platform.python_version(), np.__version__, skimage.__version__, scipy.__version__) == ('3.13.14', '2.5.4', '0.26.0', '1.18.1')
    sources = dict(upstream=sha(Path(inspect.getsourcefile(expand_labels)).read_bytes()),
                   scipy_morphology=sha(Path(_morphology.__file__).read_bytes()),
                   scipy_native=sha(Path(_nd_image.__file__).read_bytes()),
                   observer=sha(Path(__file__).read_bytes()))
    assert sources['upstream'] == UPSTREAM_SHA256 and sources['scipy_morphology'] == MORPHOLOGY_SHA256
    bench = ROOT / 'benchmarks' / 'expand_labels.py'
    spec = importlib.util.spec_from_file_location('expand_labels_cases', bench)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    sources['benchmark_recipes'] = sha(bench.read_bytes())
    rows = []
    for name, image, options, native in make_cases():
        before = snapshot(image)
        rows.append(dict(id=name, native=native, input=before, output=observe(image, options)))
        assert snapshot(image) == before
    timing_rows = []
    for row in benchmark.case_specs():
        (image,), options = benchmark.make_case(row)
        result = expand_labels(image, **options)
        timing_rows.append(dict(id=row['id'], input_sha256=sha(image.tobytes()),
                                output_sha256=sha(result.tobytes()), shape=list(image.shape),
                                dtype=image.dtype.str, output_strides=list(result.strides)))
    data = dict(original=skimage.__version__, numpy=np.__version__, scipy=scipy.__version__,
                python=platform.python_version(), sources=sources, rows=rows, benchmark=timing_rows)
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(data, indent=2) + '\n')
    print(f'Captured {len(rows)} original contract rows and {len(timing_rows)} benchmark outputs')


if __name__ == '__main__':
    main()
