"""Capture pinned original public binary_blobs outputs and provider state."""
import argparse
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
from pathlib import Path
import platform
from unittest.mock import patch
import warnings

import numpy as np
import scipy
from scipy.ndimage import _filters, _nd_image
import skimage
from skimage.data import _binary_blobs
from skimage._shared import filters as upstream_filters

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = 'c8f82559d5b6972670b0f566b1ef356909f453cc58dd76b42da3a4938d3a9416'


def cases():
    def generator(name):
        return np.random.Generator(getattr(np.random, name)(92825))

    rows = (
        ('default', dict(rng=1982), True),
        ('line', dict(length=17, n_dim=1, rng=12, volume_fraction=0.15, blob_size_fraction=0.13), True),
        ('line-wrap', dict(length=25, n_dim=1, rng=878, volume_fraction=0.42, boundary_mode='wrap'), True),
        ('plane-endpoint-zero', dict(length=20, rng=33, volume_fraction=0.), True),
        ('plane-endpoint-full', dict(length=20, rng=33, volume_fraction=1.), True),
        ('volume-endpoint-zero', dict(length=11, n_dim=3, rng=45, volume_fraction=0., boundary_mode='wrap'), True),
        ('volume-endpoint-full', dict(length=11, n_dim=3, rng=45, volume_fraction=1.), True),
        ('pcg-generator', dict(length=12, n_dim=1, rng=generator('PCG64')), True),
        ('mt-generator', dict(length=10, n_dim=3, rng=generator('MT19937'), blob_size_fraction=0.23), True),
        ('philox-generator', dict(length=13, n_dim=3, rng=generator('Philox'), volume_fraction=0.98,
                                  boundary_mode='wrap'), True),
        ('radius-zero-ties', dict(length=4, n_dim=2, rng=12, blob_size_fraction=0.11, volume_fraction=0.), True),
        ('many-tied-zeros', dict(length=16, n_dim=3, rng=927, blob_size_fraction=0.13, volume_fraction=0.95), True),
        ('one-point', dict(length=13, rng=9, blob_size_fraction=1., boundary_mode='wrap'), True),
        ('singleton', dict(length=1, n_dim=1, rng=6, blob_size_fraction=0.1), True),
        ('nearest-double-cutoff', dict(length=21, rng=34999, blob_size_fraction=0.22,
                                      volume_fraction=float(np.nextafter(0.5, 1.))), True),
        ('seed-sequence-list', dict(length=12, rng=[15, 13]), False),
        ('numpy-integer', dict(length=np.int64(12), rng=np.int64(197)), False),
        ('invalid-boundary-before-draw', dict(length=7, rng=generator('PCG64'), boundary_mode='reflect'), False),
        ('unhashable-boundary', dict(length=7, rng=3, boundary_mode=['wrap']), False),
        ('invalid-seed', dict(length=7, rng='seed'), False),
        ('negative-seed', dict(length=7, rng=-8), False),
        ('zero-volume-error', dict(length=7, rng=generator('PCG64'), volume_fraction=-0.01), False),
        ('full-volume-error', dict(length=7, rng=generator('PCG64'), volume_fraction=1.01), False),
        ('nan-volume-error', dict(length=7, rng=generator('PCG64'), volume_fraction=float('nan')), False),
        ('vector-volume', dict(length=2, rng=12, volume_fraction=np.array([0.2, 0.3])), False),
        ('clamped-fraction', dict(length=4, n_dim=3, rng=generator('PCG64'), blob_size_fraction=0.01), False),
        ('negative-fraction', dict(length=4, rng=9, blob_size_fraction=-0.03), False),
        ('large-fraction', dict(length=4, rng=9, blob_size_fraction=2.), False),
        ('nan-fraction', dict(length=4, rng=9, blob_size_fraction=float('nan')), False),
        ('zero-length', dict(length=0, rng=7), False),
        ('negative-length', dict(length=-3, rng=7), False),
        ('float-length', dict(length=3.0, rng=7), False),
        ('rank-zero', dict(length=7, n_dim=0, rng=generator('PCG64')), False),
        ('rank-negative', dict(length=7, n_dim=-1, rng=7), False),
        ('rank-four', dict(length=4, n_dim=4, rng=7, blob_size_fraction=0.4), False),
    )
    yield from rows


def sha(data):
    return hashlib.sha256(data).hexdigest()


def generator_state(rng):
    if not isinstance(rng, np.random.Generator):
        return None
    state = json.dumps(rng.bit_generator.state, sort_keys=True, default=lambda item: item.tolist())
    return sha(state.encode())


def snapshot(image):
    array = np.asarray(image)
    return dict(type=type(image).__name__, dtype=array.dtype.str, shape=list(array.shape),
                strides=list(array.strides), count=int(np.count_nonzero(array)),
                data_sha256=sha(array.tobytes()), owns=bool(array.flags.owndata))


def observe(function, options):
    status = dict(state_before=generator_state(options.get('rng')))
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            status['output'] = snapshot(function(**options))
        except Exception as error:
            status['error'] = [type(error).__module__, type(error).__name__, str(error)]
    status['warnings'] = [[w.category.__module__, w.category.__name__, str(w.message)] for w in notices]
    status['state_after'] = generator_state(options.get('rng'))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    original = _binary_blobs.binary_blobs
    version = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, original=skimage.__version__)
    assert tuple(version.values()) == ('3.13.14', '2.5.4', '1.18.1', '0.26.0')
    paths = dict(original=Path(inspect.getsourcefile(original)), scipy_filters=Path(_filters.__file__),
                 scipy_binary=Path(_nd_image.__file__), skimage_gaussian=Path(upstream_filters.__file__),
                 observer=Path(__file__), benchmark=ROOT/'benchmarks/binary_blobs.py')
    sources = {name: sha(path.read_bytes()) for name, path in paths.items()}
    assert sources['original'] == SOURCE_SHA
    spec = spec_from_file_location('blobs_benchmark', paths['benchmark'])
    benchmark = module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    captured = []

    def record_gaussian(mask, *args, **kwargs):
        result = upstream_filters.gaussian(mask, *args, **kwargs)
        captured.append(dict(input=sha(mask.tobytes()), output=sha(result.tobytes())))
        return result

    rows, endpoints = [], []
    with patch.object(_binary_blobs, 'gaussian', record_gaussian):
        for name, options, native in cases():
            captured.clear()
            observed = observe(original, options)
            rows.append(dict(id=name, native=native, observed=observed, gaussian=list(captured)))
        for spec in benchmark.case_specs():
            captured.clear()
            _, options = benchmark.make_case(spec)
            endpoints.append(dict(id=spec['id'], observed=observe(original, options), gaussian=list(captured)))
    assert all('output' in row['observed'] and len(row['gaussian']) == 1 for row in endpoints)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(dict(version=version, sources=sources, rows=rows, benchmark=endpoints), indent=2) + '\n')
    print(f'Frozen {len(rows)} original observations and {len(endpoints)} benchmark endpoints')


if __name__ == '__main__':
    main()
