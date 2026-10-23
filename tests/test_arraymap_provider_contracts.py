"""Cold-process, bounded returned-ArrayMap provider lookup contracts.

Reference observations establish a separate compatibility scope. They do not
weaken the sealed692/0486 dense-array contract or claim new native coverage.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
MATRIX = ROOT / 'coverage/contracts/arraymap-providers.json'
CASES = [
    'plain-len', 'plain-array', 'max-return-len', 'max-warn-len', 'max-error-len',
    'zeros-fill-array', 'zeros-warn-array', 'zeros-error-array',
    'allocator-before-max-twice', 'max-rebind-array', 'zeros-rebind-array',
    'zeros-warning-error-array', 'both-warning-array', 'zeros-fill-typed-array',
    'max-return-float-keys-len',
]


def profile():
    return 'minimum' if np.__version__.startswith('1.') else 'windows' if os.name == 'nt' else 'modern'


def scalar(value):
    if isinstance(value, np.generic):
        return dict(type=type(value).__name__, dtype=value.dtype.str, bytes=value.tobytes().hex())
    return dict(type=type(value).__name__, value=value)


def array(value):
    return dict(dtype=value.dtype.str, shape=list(value.shape), strides=list(value.strides),
                owned=bool(value.flags.owndata), writeable=bool(value.flags.writeable),
                bytes=value.tobytes().hex())


def probe(backend, case):
    import skimage
    from skimage.segmentation import relabel_sequential as reference
    from skimage.util._map_array import ArrayMap
    assert skimage.__version__ == '0.26.0'
    if backend == 'reference':
        factory = reference
    else:
        import skimagexx as xx
        xx.set_fallback_enabled(False)
        assert xx.get_num_threads() == 1
        factory = xx.segmentation.relabel_sequential
    labels = np.array([0, 2, 4], np.uint16)
    _, mapping, _ = factory(labels)
    if case == 'max-return-float-keys-len':
        mapping.in_values = mapping.in_values.astype(np.float64)
    original_keys, original_values = mapping.in_values, mapping.out_values
    before = dict(keys=array(original_keys), values=array(original_values), cache=mapping._array is None)
    trace = []
    original_max, original_zeros = np.max, np.zeros

    def zeros(length, *, dtype):
        trace.append(dict(stage='zeros', length=scalar(length), dtype=np.dtype(dtype).str))
        if case in ('zeros-warn-array', 'zeros-warning-error-array', 'both-warning-array'):
            warnings.warn('ArrayMap zeros provider warning', RuntimeWarning)
        if case == 'zeros-error-array':
            raise LookupError('ArrayMap zeros provider failed')
        if case == 'zeros-rebind-array':
            mapping.in_values = np.array([0, 1, 3], np.uint16)
            mapping.out_values = np.array([0, 8, 9], np.int16)
        output = original_zeros(length, dtype=dtype)
        if case in ('zeros-fill-array', 'zeros-fill-typed-array'):
            output.fill(7)
        return output

    def second_zeros(length, *, dtype):
        trace.append(dict(stage='zeros-second', length=scalar(length), dtype=np.dtype(dtype).str))
        output = original_zeros(length, dtype=dtype)
        output.fill(9)
        return output

    def first_zeros(length, *, dtype):
        trace.append(dict(stage='zeros-first', length=scalar(length), dtype=np.dtype(dtype).str))
        output = original_zeros(length, dtype=dtype)
        output.fill(7)
        return output

    def maximum(keys):
        trace.append(dict(stage='max', keys=array(keys)))
        if case in ('max-warn-len', 'both-warning-array'):
            warnings.warn('ArrayMap max provider warning', UserWarning)
        if case == 'max-error-len':
            raise RuntimeError('ArrayMap max provider failed')
        if case == 'allocator-before-max-twice':
            np.zeros = second_zeros
        if case == 'max-rebind-array':
            mapping.in_values = np.array([0, 1, 3], np.uint16)
            mapping.out_values = np.array([0, 8, 9], np.int16)
        return np.int64(6) if case.startswith('max-return') else original_max(keys)

    if not case.startswith('plain-'):
        np.max = maximum
        np.zeros = first_zeros if case == 'allocator-before-max-twice' else zeros
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('error' if case == 'zeros-warning-error-array' else 'always')
        try:
            if case.endswith('-len'):
                result = len(mapping)
            elif case == 'allocator-before-max-twice':
                result = [mapping.__array__(), mapping.__array__()]
            elif case == 'zeros-fill-typed-array':
                result = mapping.__array__(dtype=np.dtype('f4'), copy=False)
            else:
                result = mapping.__array__()
            error = None
        except Exception as caught:
            result, error = None, dict(type=type(caught).__name__, message=str(caught))
        finally:
            np.max, np.zeros = original_max, original_zeros
    describe = lambda value: array(value) if isinstance(value, np.ndarray) else scalar(value)
    observation = dict(result=[describe(value) for value in result] if isinstance(result, list) else describe(result),
        error=error, trace=trace, warnings=[dict(type=type(w.message).__name__, message=str(w.message)) for w in notices],
        before=before, after=dict(keys=array(mapping.in_values), values=array(mapping.out_values), cache=mapping._array is None,
                                 same_keys=mapping.in_values is original_keys, same_values=mapping.out_values is original_values),
        old_keys=array(original_keys), old_values=array(original_values), labels=array(labels))
    if backend != 'reference':
        assert mapping.__class__.__module__ == 'skimagexx._component_consumers'
    return dict(backend=backend, case=case, profile=profile(), observation=observation,
                source_hashes={name:hashlib.sha256(inspect.getsource(getattr(ArrayMap, name)).encode()).hexdigest() for name in ('__len__', '__array__')},
                versions=dict(python=sys.version, numpy=np.__version__, skimage=skimage.__version__))


def cold(backend, case, pythonpath=None):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    if pythonpath is not None:
        env['PYTHONPATH'] = str(pythonpath)
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '1'
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe', backend, case],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    return json.loads(process.stdout)


@pytest.mark.parametrize('case', CASES)
def test_frozen_original_provider_contract(case):
    frozen = json.loads(MATRIX.read_text())
    actual = cold('reference', case)
    expected = frozen['reference'][profile()][case]
    assert actual['observation'] == expected
    assert actual['source_hashes'] == frozen['source_hashes']


def test_reference_scope_stays_separate_from_native_acceptance():
    frozen = json.loads(MATRIX.read_text())
    assert frozen['native_acceptance_checks'] == 0
    assert len(frozen['case_ids']) == len(CASES) == 15
    assert frozen['original_dense_contract_unchanged'] == ['6920dff', '0486cbb']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', nargs=2, choices=None, metavar=('BACKEND', 'CASE'))
    arguments = parser.parse_args()
    backend, case = arguments.probe
    assert backend in ('reference', 'candidate') and case in CASES
    print(json.dumps(probe(backend, case), sort_keys=True))
