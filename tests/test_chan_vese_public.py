import inspect
from pathlib import Path
import runpy
import warnings

import numpy as np
import pytest
from skimage.segmentation import chan_vese as original
import skimagexx as xx
from skimagexx import _core
from skimagexx._chan_vese import chan_vese
from skimagexx._compat import FallbackWarning


_RECIPES = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'benchmarks/chan_vese.py'))
_BENCHMARK_CASES = _RECIPES['case_specs']()


@pytest.fixture
def strict():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)


def _compare(actual, expected):
    if isinstance(expected, np.ndarray):
        assert type(actual) is type(expected) and actual.dtype == np.dtype(np.bool_)
        np.testing.assert_array_equal(actual, expected)
        return
    assert type(actual) is tuple and len(actual) == 3
    mask, level_set, energies = actual
    wanted_mask, wanted_level_set, wanted_energies = expected
    assert type(mask) is type(wanted_mask) and mask.dtype == wanted_mask.dtype
    assert type(level_set) is type(wanted_level_set) and level_set.dtype == wanted_level_set.dtype
    assert type(energies) is list and len(energies) == len(wanted_energies)
    assert all(type(a) is type(b) for a, b in zip(energies, wanted_energies))
    np.testing.assert_array_equal(mask, wanted_mask)
    tolerance = 2e-6 if level_set.dtype == np.float32 else 2e-13
    np.testing.assert_allclose(level_set, wanted_level_set, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(energies, wanted_energies, rtol=tolerance, atol=tolerance)


def test_chan_vese_public_signature_and_ownership(strict, monkeypatch):
    assert xx.segmentation.chan_vese is chan_vese
    assert inspect.signature(chan_vese) == inspect.signature(original)
    native = _core.chan_vese_evolve
    assert inspect.isbuiltin(native) and native.__module__ == 'skimagexx._core'
    image = np.arange(30, dtype=np.float64).reshape(5, 6)
    phi = image - 12
    calls = []

    def trace(*args):
        calls.append((args[0] is image, args[1] is phi))
        return native(*args)

    monkeypatch.setattr(_core, 'chan_vese_evolve', trace)
    _compare(xx.segmentation.chan_vese(image, 0.25, 1.0, 1.0, 0.0, 2, 0.5, phi, True),
             original(image, 0.25, 1.0, 1.0, 0.0, 2, 0.5, phi, True))
    assert calls == [(True, True)]


@pytest.mark.parametrize('spec', _BENCHMARK_CASES, ids=lambda spec: spec['id'])
def test_chan_vese_benchmark_parity(spec, strict):
    args, kwargs = _RECIPES['make_case'](spec)
    before = args[0].copy()
    phi = kwargs['init_level_set']
    old_phi = phi.copy() if type(phi) is np.ndarray else None
    _compare(chan_vese(*args, **kwargs), original(*args, **kwargs))
    np.testing.assert_array_equal(args[0], before)
    if old_phi is not None:
        np.testing.assert_array_equal(phi, old_phi)


@pytest.mark.parametrize('dtype', [np.bool_, np.int16, np.uint64, np.float16, np.float32, np.float64])
@pytest.mark.parametrize('sign', [-1, 0, 1])
def test_chan_vese_zero_iterations_and_empty_regions(dtype, sign, strict):
    image = (np.arange(35).reshape(5, 7) - 5).astype(dtype)
    phi_dtype = np.float32 if dtype in (np.float16, np.float32) else np.float64
    phi = np.full(image.shape, sign, dtype=phi_dtype)
    options = dict(init_level_set=phi, extended_output=True)
    untouched = chan_vese(image, max_num_iter=0, **options)
    assert untouched[1] is phi
    assert not np.shares_memory(untouched[0], phi)
    _compare(untouched, original(image, max_num_iter=0, **options))
    converged = chan_vese(image, max_num_iter=50, dt=0., **options)
    assert len(converged[2]) == 1 and not np.shares_memory(converged[1], phi)
    _compare(converged, original(image, max_num_iter=50, dt=0., **options))


@pytest.mark.parametrize('shape', [(1, 11), (13, 1), (1, 1), (7, 9)])
def test_chan_vese_boundary_and_constant_image(shape, strict):
    image = np.full(shape, 2., dtype=np.float32)
    phi = (np.arange(image.size, dtype=np.int32).reshape(shape) - image.size // 2) / 10
    kwargs = dict(init_level_set=phi, extended_output=True, tol=0., max_num_iter=7)
    _compare(chan_vese(image, **kwargs), original(image, **kwargs))


def test_chan_vese_parallel_stage_matches_one_worker_and_stock(strict):
    # Count exceeds execution.hpp's partition threshold of 256 * 1024.
    spec = dict(_BENCHMARK_CASES[-1], shape=(521, 529), iterations=3)
    args, kwargs = _RECIPES['make_case'](spec)
    previous = xx.get_num_threads()
    try:
        xx.set_num_threads(1)
        serial = chan_vese(*args, **kwargs)
        _compare(serial, original(*args, **kwargs))
        try:
            xx.set_num_threads(4)
        except ValueError as error:
            if str(error) != 'this build has no OpenMP support':
                raise
            assert xx.get_num_threads() == 1
            return
        parallel = chan_vese(*args, **kwargs)
    finally:
        xx.set_num_threads(previous)
    for lhs, rhs in zip(serial, parallel):
        np.testing.assert_array_equal(lhs, rhs)


class _Array(np.ndarray):
    pass


def _array_variants():
    dense = np.arange(36, dtype=np.float64).reshape(6, 6)
    read_only = dense.copy()
    read_only.flags.writeable = False
    return (
        dense.view(_Array),
        dense.astype(np.dtype('f8', metadata={'source': 'tagged'})),
        dense.astype(np.longdouble),
        dense.astype(dense.dtype.newbyteorder('S')),
        dense[:, ::-1],
        np.asfortranarray(dense),
        read_only,
        np.lib.stride_tricks.as_strided(dense, shape=dense.shape, strides=dense.strides, writeable=True),
    )


@pytest.mark.parametrize('source', _array_variants(),
                         ids=['subclass', 'metadata', 'longdouble', 'byte-order', 'strided', 'fortran',
                              'read-only', 'unbounded-owner'])
@pytest.mark.parametrize('argument', ['image', 'initial'])
def test_chan_vese_restricted_array_fallback(source, argument, strict, monkeypatch):
    image = np.arange(36, dtype=np.float64).reshape(6, 6)
    direct_args = (source, image) if argument == 'image' else (image, source)
    with pytest.raises(ValueError, match='chan_vese requires|requires ndarray-backed storage'):
        _core.chan_vese_evolve(*direct_args, .25, 1., 1., .001, 3, .5, True, 0)

    def unexpected(*args):
        pytest.fail('restricted input reached native evolution')

    monkeypatch.setattr(_core, 'chan_vese_evolve', unexpected)
    kwargs = dict(extended_output=True, max_num_iter=3)
    if argument == 'image':
        image = source
    else:
        kwargs['init_level_set'] = source
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        chan_vese(image, **kwargs)
    xx.set_fallback_enabled(True)
    if argument == 'initial' and type(source) is _Array:
        with pytest.warns(FallbackWarning), pytest.raises(ValueError, match='dimensions of initial level set'):
            chan_vese(image, **kwargs)
    else:
        with pytest.warns(FallbackWarning):
            actual = chan_vese(image, **kwargs)
        _compare(actual, original(image, **kwargs))


@pytest.mark.parametrize('options', [
    {'init_level_set': 'not a preset'}, {'init_level_set': np.zeros((1, 1))},
    {'mu': np.float64(.25)}, {'lambda1': -1}, {'dt': float('inf')},
])
def test_chan_vese_unsupported_controls_obey_fallback(options, strict):
    image = np.arange(35, dtype=np.float64).reshape(5, 7)
    kwargs = dict(options, max_num_iter=3)
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        chan_vese(image, **kwargs)
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings(record=True) as items:
        warnings.simplefilter('always')
        try:
            expected = original(image, **kwargs)
        except ValueError as error:
            with pytest.raises(ValueError) as raised:
                chan_vese(image, **kwargs)
            assert str(raised.value) == str(error)
        else:
            _compare(chan_vese(image, **kwargs), expected)
    assert sum(item.category is FallbackWarning for item in items) == 1


def test_chan_vese_nonfinite_fallback(strict):
    image = np.arange(30, dtype=np.float32).reshape(5, 6)
    image[0, 0] = np.nan
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        chan_vese(image, max_num_iter=3)
    xx.set_fallback_enabled(True)
    with pytest.warns(FallbackWarning):
        actual = chan_vese(image, max_num_iter=3)
    _compare(actual, original(image, max_num_iter=3))
