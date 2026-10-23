"""Private native dense-pipeline parity; never registers/forwards public APIs."""
import importlib
import importlib.util
import copy
import inspect
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import skimagexx as xx
from skimage import morphology as reference


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
contract = load(ROOT / 'tests/test_footprint_contracts.py', 'footprint_contract_helpers')
if os.environ.get('SKIMAGEXX_FOOTPRINT_BACKEND'):
    native = load(ROOT / 'python/skimagexx/_footprint_dense.py', 'skimagexx._footprint_dense_candidate')
    native._core = importlib.import_module(os.environ['SKIMAGEXX_FOOTPRINT_BACKEND'])
else:
    native = importlib.import_module('skimagexx._footprint_dense')
INITIAL_WORKERS = native._core.get_num_threads()
CASES = [case for case in contract.CASES
         if case[1] in contract.NAMES[:4] and case[3].get('decomposition') is None]


@pytest.fixture(params=[1, 4], autouse=True)
def workers(request):
    core = native._core
    previous = core.get_num_threads()
    if os.environ.get('SKIMAGEXX_FOOTPRINT_PORTABLE') and request.param != 1:
        pytest.skip('portable no-OpenMP build supports one worker')
    core.set_num_threads(request.param)
    assert core.get_num_threads() == request.param
    yield
    core.set_num_threads(previous)


@pytest.mark.parametrize('case', CASES, ids=[case[0] for case in CASES])
def test_frozen_dense_cases(case):
    key, name, args, kwargs = case
    expected = contract.observe(getattr(reference, name), args, kwargs)
    actual = contract.observe(getattr(native, name), args, kwargs)
    assert actual == expected, key


@pytest.mark.parametrize('name', contract.NAMES[:4])
def test_signatures_and_no_forwarding(name, monkeypatch):
    assert inspect.signature(getattr(native, name)) == inspect.signature(getattr(reference, name))
    def reject(*args, **kwargs):
        raise AssertionError('original footprint implementation was called')
    monkeypatch.setattr(reference, name, reject)
    result = getattr(native, name)((5, 7) if name == 'footprint_rectangle' else 5)
    assert result.dtype == np.uint8 and result.flags.owndata
    with pytest.raises(NotImplementedError, match='decomposition=None'):
        getattr(native, name)((5, 7) if name == 'footprint_rectangle' else 5,
                              decomposition='sequence')


@pytest.mark.parametrize('name', contract.NAMES[:4])
@pytest.mark.parametrize('dtype', ['uint8', 'float32', 'float64', '>i2', 'object', 'U3'])
def test_parallel_size_and_fractional_edges(name, dtype):
    # Both 2-D and 3-D exceed the global parallel threshold.
    radius = 258 if name != 'ball' else 35
    args = ((513, 517),) if name == 'footprint_rectangle' else (radius,)
    options = dict(dtype=dtype)
    assert contract.observe(getattr(native, name), args, options) == contract.observe(
        getattr(reference, name), args, options)


@pytest.mark.parametrize('name', ['diamond', 'disk', 'ball'])
@pytest.mark.parametrize('radius', [np.float32(.5), np.float32(1.3), np.float64(-1.3),
                                  np.int16(4), np.array(2.), np.array([2.]),
                                  -.1, -1.1, .1, 2.7])
@pytest.mark.parametrize('strict', [False, True])
def test_scalar_promotions(name, radius, strict):
    kwargs = {} if name == 'diamond' else dict(strict_radius=strict)
    def run(function):
        # Upstream intentionally mutates a 0-D radius under strict_radius=False
        # through radius += .5. Freeze the mutation too, using disjoint inputs.
        value = copy.deepcopy(radius)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:
                result = contract.describe(function(value, **kwargs), [])
                error = None
            except Exception as caught:
                result, error = None, (type(caught), str(caught))
        return result, error, contract.describe(value, []), [(type(w.message), str(w.message)) for w in notices]
    assert run(getattr(native, name)) == run(getattr(reference, name))


@pytest.mark.parametrize('name', contract.NAMES[:4])
@pytest.mark.parametrize('dtype', [None, 'U', 'S', 'V0', 'V1', 'V3', 'datetime64[D]', 'timedelta64[s]',
                                  'datetime64', 'timedelta64', np.longdouble,
                                  np.dtype([('o', object), ('n', 'i2')])])
def test_dtype_providers(name, dtype):
    args = ((2, 3),) if name == 'footprint_rectangle' else (2,)
    def run(function):
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:
                array, error = function(*args, dtype=dtype), None
            except Exception as caught:
                array, error = None, (type(caught), str(caught))
        if array is None:
            result = None
        elif array.dtype.kind in 'mM':
            # Generic-unit datetime/timedelta .tolist() can itself fail. That
            # must not turn a successful footprint call into an expected error.
            result = (array.dtype.str, array.shape, array.strides,
                      array.flags.owndata, array.flags.writeable,
                      contract.describe(array.view('i8'), []))
        else:
            result = contract.describe(array, [])
        return result, error, [(type(w.message), str(w.message)) for w in notices]
    assert run(getattr(native, name)) == run(getattr(reference, name))


def test_binding_guards():
    core = native._core
    axis = np.array([1., 0., 1.])
    zero, one = np.array(0, dtype=np.uint8), np.array(1, dtype=np.uint8)
    def call(output, kind=2, coordinates=axis, z=zero, o=one):
        return core.footprint_dense_into(output, coordinates, kind, 1., z, o)
    for output in [np.empty((2, 3), dtype=np.uint8), np.empty(9, dtype=np.uint8),
                   np.empty((3, 3), dtype=np.uint8).T]:
        with pytest.raises(ValueError): call(output)
    readonly = np.empty((3, 3), dtype=np.uint8); readonly.flags.writeable = False
    with pytest.raises(ValueError): call(readonly)
    with pytest.raises(ValueError): call(np.empty((3, 3), dtype=np.uint8), kind=4)
    with pytest.raises(TypeError): call(np.empty((3, 3), dtype=object))
    with pytest.raises(ValueError): call(np.empty((3, 3), dtype=np.uint8), z=np.array(0.))
    data = np.ones((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match='disjoint'):
        core.footprint_dense_into(data, data[0], 2, 1., data[0, :1], data[0, :1])


def test_shared_global_default():
    assert INITIAL_WORKERS == 1
    assert native._core.get_num_threads() in (1, 4)
    assert native._core.footprint_dense_into.__module__ == native._core.__name__
    # An integrated run must use the one main-core state. Standalone runs are
    # identified explicitly and never stand in for public-registration proof.
    if not os.environ.get('SKIMAGEXX_FOOTPRINT_BACKEND'):
        assert native._core is xx._core
        assert native._core.get_num_threads is xx.get_num_threads
    if os.environ.get('SKIMAGEXX_DISABLE_SIMD'):
        assert not native._core.has_avx2()
