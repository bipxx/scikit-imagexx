"""Paired one-thread timings of the owned estimate_transform factory routes."""

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import partial
import os
from pathlib import Path
import sys

if __package__:
    from .run import measure
else:
    from run import measure

import numpy as np
import skimage
from skimage import transform as original

import skimagexx as xx
from skimagexx import _estimate_transform, _geometric


@dataclass(frozen=True)
class Case:
    model: str
    count: int
    dimensions: int
    weighted: bool = False


CASES = (
    Case('euclidean', 8, 2),
    Case('euclidean', 32, 3),
    Case('similarity', 8, 2),
    Case('similarity', 32, 3),
    Case('affine', 8, 2),
    Case('affine', 32, 3, weighted=True),
    Case('projective', 8, 2),
    Case('projective', 32, 3, weighted=True),
)


def workload(case):
    rng = np.random.default_rng(2026)
    src = rng.uniform(-2, 2, (case.count, case.dimensions))
    dst = src + np.arange(case.dimensions) + 0.25
    dst += rng.normal(0, 0.02, dst.shape)
    weights = {'weights': rng.uniform(0.5, 1.5, case.count)} if case.weighted else {}
    src.flags.writeable = dst.flags.writeable = False
    for array in weights.values():
        array.flags.writeable = False
    return (case.model, src, dst), weights


def main():
    candidate = xx.transform.estimate_transform
    assert candidate is _estimate_transform.estimate_transform
    assert skimage.__version__ == '0.26.0'
    assert xx.get_num_threads() == 1
    xx.set_fallback_enabled(False)

    repeat, target_ms = 11, 25
    report = {
        'scope': 'Warm factory calls, including returned model construction and destruction',
        'worker_count': xx.get_num_threads(),
        'numerical_provider': 'NumPy normalization, matrix operations, and LAPACK for fitting',
        'python': sys.version,
        'executable': sys.executable,
        'scikit_image': skimage.__version__,
        'numpy': np.__version__,
        'avx2': xx.has_avx2(),
        'thread_environment': {
            variable: os.environ[variable]
            for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')
        },
        'repeat': repeat,
        'target_ms': target_ms,
        'files': {},
        'cases': [],
    }
    sources = {
        'benchmark': __file__,
        'candidate_factory': _estimate_transform.__file__,
        'candidate_classes': _geometric.__file__,
        'candidate_extension': xx._core.__file__,
        'original_factory_and_classes': original.estimate_transform.__code__.co_filename,
    }
    for label, source in sources.items():
        path = Path(source).resolve()
        report['files'][label] = {
            'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for case in CASES:
        operands, weights = workload(case)
        baseline = partial(original.estimate_transform, *operands, **weights)
        current = partial(candidate, *operands, **weights)
        expected, actual = baseline(), current()
        assert type(actual).__name__ == type(expected).__name__
        assert type(actual).__module__ == _geometric.__name__
        np.testing.assert_allclose(actual.params, expected.params, rtol=1e-12, atol=1e-12)
        timing = measure(baseline, current, repeat, target_ms)
        timing['candidate_ms'] = timing.pop('cpp_ms')
        report['cases'].append({**asdict(case), **timing})

    print(json.dumps(report, indent=2))
    return int(any(case['speedup'] < 1.0 for case in report['cases']))


if __name__ == '__main__':
    sys.exit(main())
