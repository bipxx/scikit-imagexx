"""Exact private decomposition parity, including tuple order and view sharing."""
import importlib
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from skimage import morphology as reference


ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load('footprint_decomposition_contract', ROOT / 'tests/test_footprint_contracts.py')
if os.environ.get('SKIMAGEXX_FOOTPRINT_BACKEND'):
    dense = load('skimagexx._footprint_dense', ROOT / 'python/skimagexx/_footprint_dense.py')
    native = load('skimagexx._footprint_decomposition', ROOT / 'python/skimagexx/_footprint_decomposition.py')
    native._core = dense._core = importlib.import_module(os.environ['SKIMAGEXX_FOOTPRINT_BACKEND'])
else:
    dense = importlib.import_module('skimagexx._footprint_dense')
    native = importlib.import_module('skimagexx._footprint_decomposition')
INITIAL_WORKERS = native._core.get_num_threads()
CASES = [case for case in contract.CASES if case[1] in contract.NAMES[:4]]


@pytest.fixture(params=[1, 4], autouse=True)
def workers(request):
    if os.environ.get('SKIMAGEXX_FOOTPRINT_PORTABLE') and request.param != 1:
        pytest.skip('portable no-OpenMP build supports one worker')
    previous = native._core.get_num_threads()
    native._core.set_num_threads(request.param)
    yield
    native._core.set_num_threads(previous)


def check(name, args, kwargs):
    assert contract.observe(getattr(native, name), args, kwargs) == contract.observe(
        getattr(reference, name), args, kwargs)


@pytest.mark.parametrize('case', CASES, ids=[case[0] for case in CASES])
def test_frozen_cases(case):
    _, name, args, kwargs = case
    check(name, args, kwargs)


@pytest.mark.parametrize('name,maximum', [('disk', 251), ('ball', 101)])
def test_complete_sphere_tables(name, maximum):
    for radius in range(-maximum, maximum):
        check(name, (radius,), dict(decomposition='sequence'))


@pytest.mark.parametrize('dtype', [None, 'uint8', 'float64', 'object', 'U3', 'complex64'])
def test_ordered_rotation_aliases(dtype):
    # Radius 5 uses T elements; t0 is owned, rotations alias the same base.
    actual = native.disk(5, dtype=dtype, decomposition='sequence')
    expected = reference.disk(5, dtype=dtype, decomposition='sequence')
    for sequence in [actual, expected]:
        leaves = [a for a, _ in sequence]
        if len(leaves) >= 4:
            assert leaves[0].flags.owndata
            for leaf in leaves[1:4]:
                assert not leaf.flags.owndata
                assert np.shares_memory(leaves[0], leaf)
    check('disk', (5,), dict(dtype=dtype, decomposition='sequence'))


@pytest.mark.parametrize('radius', [0, .1, .5, 1.3, 2.7, 5, 20, 80, -2, -.1])
@pytest.mark.parametrize('strict', [False, True])
def test_cross_plans(radius, strict):
    check('disk', (radius,), dict(decomposition='crosses', strict_radius=strict))


@pytest.mark.parametrize('radius', [np.float32(2.5), np.int16(3), np.array(2.), np.array([2.])])
@pytest.mark.parametrize('dtype', ['uint8', 'object', 'complex64'])
def test_cross_scalar_warning_order(radius, dtype):
    check('disk', (radius,), dict(dtype=dtype, decomposition='crosses'))


@pytest.mark.parametrize('dtype', [None, 'bool', 'float16', '>i2', 'object', 'U', 'S', 'V1',
                                  np.dtype([('a', 'i2'), ('b', '?')])])
@pytest.mark.parametrize('name', ['disk', 'ball'])
def test_dtype_and_index_edges(dtype, name):
    for radius in [0, 1, 3, 5, False, np.int16(2), np.array(0), np.array([0]), 2.5]:
        check(name, (radius,), dict(dtype=dtype, decomposition='sequence'))


def test_no_upstream_forwarding_and_default(monkeypatch):
    assert INITIAL_WORKERS == 1
    def reject(*args, **kwargs):
        raise AssertionError('original generator called')
    for name in contract.NAMES[:4]:
        monkeypatch.setattr(reference, name, reject)
        args = ((3, 5),) if name == 'footprint_rectangle' else (5,)
        assert isinstance(getattr(native, name)(*args, decomposition='sequence'), tuple)
    assert isinstance(native.disk(20, decomposition='crosses'), tuple)


def test_raw_guards_and_worker_cross_counts():
    with pytest.raises((ValueError, RuntimeError)):
        native._core.footprint_sphere_counts(4, 0)
    with pytest.raises(IndexError): native._core.footprint_sphere_counts(2, 251)
    with pytest.raises(ValueError): native._core.footprint_cross_sums(np.array([2, 3], dtype=np.intp), 3)
    with pytest.raises(ValueError): native._core.footprint_cross_costs(np.array([1., 0., 2., 1.]), 4.)
    # Large 1-D costs activate shared global workers without allocating a disk.
    costs = np.arange(-150000, 150001, dtype=float) ** 2
    plans = native._core.footprint_cross_costs(costs, 150000. ** 2)
    assert plans and all(v >= 0 and h >= 0 and n > 0 for v, h, n in plans)
    previous = native._core.get_num_threads()
    try:
        native._core.set_num_threads(1)
        assert native._core.footprint_cross_costs(costs, 150000. ** 2) == plans
    finally:
        native._core.set_num_threads(previous)
    output = np.empty((3, 3), dtype=np.uint8)
    zero, one = np.array(0, dtype=np.uint8), np.array(1, dtype=np.uint8)
    for kind, dimensions, axis, direction in [(3, 2, 0, 0), (0, 4, 0, 0), (0, 2, 2, 0), (0, 2, 0, 1)]:
        with pytest.raises(ValueError):
            native._core.footprint_element_into(output, zero, one, kind, dimensions, axis, direction, 0, 0)
    with pytest.raises(TypeError):
        native._core.footprint_element_into(output.astype(object), zero, one, 0, 2, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        native._core.footprint_element_into(output, output[:1, :1], one, 0, 2, 0, 0, 0, 0)
