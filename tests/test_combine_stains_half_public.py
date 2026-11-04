"""Saved half stain compatibility through real cores; no original calls."""
from contextlib import contextmanager
from copy import deepcopy
import base64
import importlib
import inspect
import json
from pathlib import Path
import sys
import zlib
import numpy as np
import pytest
import skimagexx as xx
from skimagexx import color as owned
from skimagexx.coverage import native_dtypes, parity_tolerance
if __package__:
    from .combine_stains_half_contract import cases, fixture, NUMERICAL_POLICY
    from .combine_stains_half_observer import observe, SIGNATURE
else:
    from combine_stains_half_contract import cases, fixture, NUMERICAL_POLICY
    from combine_stains_half_observer import observe, SIGNATURE
ROOT = Path(__file__).parents[1]
CASES = cases()


@pytest.fixture(scope='module')
def reference():
    return json.loads((ROOT / 'coverage/contracts/combine-stains-half.json').read_bytes())


@contextmanager
def strict_owned(monkeypatch, workers):
    fallback = xx.set_fallback_enabled(False)
    previous = xx.get_num_threads()
    def denied(*args, **kwargs):
        raise AssertionError('half combine_stains forwarded to a provider')
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(owned, 'fallback', denied)
            patch.setattr(importlib.import_module('skimagexx._compat'), 'fallback', denied)
            yield
            assert xx.get_num_threads() == workers
    finally:
        xx.set_num_threads(previous)
        xx.set_fallback_enabled(fallback)
        assert xx.get_num_threads() == previous


def decoded(array):
    raw = zlib.decompress(base64.b64decode(array['raw_zlib_base64']))
    return np.frombuffer(raw, dtype=array['dtype']).reshape(array['shape'])


def test_combine_stains_half_signature_alias_and_policy():
    assert xx.combine_stains is owned.combine_stains
    assert str(inspect.signature(owned.combine_stains)) == SIGNATURE
    assert native_dtypes('color.combine_stains') == ('uint8', 'float16', 'float32', 'float64')
    assert parity_tolerance('color.combine_stains', np.dtype('float16')) == (NUMERICAL_POLICY['rtol'], NUMERICAL_POLICY['atol'])
    assert parity_tolerance('color.combine_stains', np.dtype('float16')) == parity_tolerance('color.combine_stains', np.dtype('float32'))
    assert parity_tolerance('color.combine_stains', np.dtype('float64')) == (2e-13, 2e-13)
    for name in ('separate_stains', 'rgb2hed', 'hed2rgb'):
        assert 'float16' not in native_dtypes('color.' + name)
        assert parity_tolerance('color.' + name, np.dtype('float16')) == (2e-13, 2e-13)


@pytest.mark.parametrize('workers', (1, 4))
@pytest.mark.parametrize('spec', CASES, ids=[spec['id'] for spec in CASES])
def test_combine_stains_half_saved_public_pipeline(spec, workers, reference, monkeypatch):
    expected = next(row for row in reference['original_observations'][np.__version__] if row['id'] == spec['id'])
    entry = owned._core.stain_transform
    assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
    returns, normalizations, admission = [], [], []
    def profile(frame, event, value):
        if event == 'call' and frame.f_globals.get('__name__') == 'skimage.color.colorconv':
            raise AssertionError('original color algorithm executed')
        if event == 'c_return' and value is entry:
            returns.append('stain_transform')
        if event == 'c_call' and frame.f_code is owned._stain.__code__ and getattr(value, '__name__', '') == 'astype':
            receiver = getattr(value, '__self__', None)
            if type(receiver) is np.ndarray and receiver.dtype.type is np.float16:
                normalizations.append(receiver.dtype.str)
        if event == 'return' and frame.f_code is owned._stain.__code__:
            admission.append(frame.f_locals['moved'].dtype.type is np.float32 and type(frame.f_locals.get('result')) is np.ndarray)
    with strict_owned(monkeypatch, workers):
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            actual = observe(owned.combine_stains, spec, np)
        finally:
            sys.setprofile(previous)
    assert returns == ['stain_transform'] and normalizations == ['<f2'] and admission == [True]
    structural_actual, structural_expected = deepcopy(actual), deepcopy(expected)
    for row in (structural_actual, structural_expected):
        row['result'].pop('raw_sha256')
        row['result'].pop('raw_zlib_base64')
    assert structural_actual == structural_expected
    args, _, _ = fixture(spec, np)
    matrix = args[1]
    scale = max(1., float(np.max(np.sum(np.abs(matrix), axis=0)))) if matrix.size else 1.
    np.testing.assert_allclose(decoded(actual['result']), decoded(expected['result']),
        rtol=NUMERICAL_POLICY['rtol'], atol=NUMERICAL_POLICY['atol'] * scale)


@pytest.mark.parametrize('workers', (1, 4))
@pytest.mark.parametrize('dtype', ('float32', 'float64'))
def test_existing_float32_combine_path_reuses_saved_reference(dtype, workers, reference, monkeypatch):
    spec = next(spec for spec in CASES if spec['id'] == 'combine-half/standard-' + ('f32' if dtype == 'float32' else 'f64'))
    expected = next(row for row in reference['original_observations'][np.__version__] if row['id'] == spec['id'])
    args, options, _ = fixture(spec, np)
    image = args[0].astype(np.float32)
    with strict_owned(monkeypatch, workers):
        actual = owned.combine_stains(image, args[1], **options)
    assert actual.dtype.str == expected['result']['dtype'] and actual.dtype.metadata is None
    assert actual.shape == tuple(expected['result']['shape'])
    scale = max(1., float(np.max(np.sum(np.abs(args[1]), axis=0))))
    np.testing.assert_allclose(actual, decoded(expected['result']), rtol=2e-6, atol=2e-6 * scale)


@pytest.mark.parametrize('workers', (1, 4))
@pytest.mark.parametrize('name', ('separate_stains', 'rgb2hed', 'hed2rgb'))
def test_other_stain_half_provider_partitions_unchanged(name, workers, monkeypatch):
    image = np.zeros((2, 3), dtype=np.float16)
    matrix = np.eye(3)
    args = (image, matrix) if name == 'separate_stains' else (image,)
    old_budget, fallback = xx.get_num_threads(), xx.set_fallback_enabled(False)
    calls = []
    sentinel = object()
    def provider(module, api, *received, **options):
        calls.append((module, api, received, options))
        return sentinel
    try:
        xx.set_num_threads(workers)
        with pytest.raises(NotImplementedError):
            getattr(owned, name)(*args)
        xx.set_fallback_enabled(True)
        with monkeypatch.context() as patch:
            patch.setattr(owned, 'fallback', provider)
            assert getattr(owned, name)(*args) is sentinel
        assert len(calls) == 1 and calls[0][0:2] == ('color', name)
        assert all(a is b for a, b in zip(calls[0][2], args))
        assert calls[0][3] == dict(channel_axis=-1)
    finally:
        xx.set_num_threads(old_budget)
        xx.set_fallback_enabled(fallback)


@pytest.mark.parametrize('kind', ('image-subclass', 'matrix-subclass', 'non-native-endian', 'half-list'))
def test_half_type_provider_limits_preserve_original_objects(kind, monkeypatch):
    class Array(np.ndarray):
        pass
    image = np.zeros((2, 3), dtype=np.float16)
    matrix = np.eye(3)
    if kind == 'image-subclass':
        image = image.view(Array)
    elif kind == 'matrix-subclass':
        matrix = matrix.view(Array)
    elif kind == 'non-native-endian':
        image = image.astype(image.dtype.newbyteorder('S'))
    else:
        image = list(image)
    calls = []
    sentinel = object()
    def provider(module, name, *args, **options):
        calls.append((module, name, args, options))
        return sentinel
    with monkeypatch.context() as patch:
        patch.setattr(owned, 'fallback', provider)
        assert owned.combine_stains(image, matrix) is sentinel
    assert len(calls) == 1 and calls[0][0:2] == ('color', 'combine_stains')
    assert calls[0][2][0] is image and calls[0][2][1] is matrix
    assert calls[0][3] == dict(channel_axis=-1)


@pytest.mark.parametrize('workers', (1, 4))
def test_half_large_independent_pixel_partition(reference, monkeypatch, workers):
    # execution.hpp::parallel_min is256*1024; include one pixel past threshold.
    image = np.zeros((256 * 1024 + 1, 3), dtype=np.float16)
    matrix = np.eye(3, dtype=np.float32)
    before = image.copy()
    with strict_owned(monkeypatch, workers):
        actual = owned.combine_stains(image, matrix)
    row = next(r for r in reference['original_observations'][np.__version__] if r['id'] == 'combine-half/standard-f32')
    assert actual.dtype.str == row['result']['dtype']
    assert actual.shape == image.shape and actual.flags.owndata and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, np.ones_like(actual))
    np.testing.assert_array_equal(image, before)
    assert not np.shares_memory(actual, image)


@pytest.mark.parametrize('workers', (1, 4))
@pytest.mark.parametrize('name', ('separate_stains', 'combine_stains', 'rgb2hed', 'hed2rgb'))
def test_existing_float32_stain_consumers_stay_native(name, workers, monkeypatch):
    separate = name in ('separate_stains', 'rgb2hed')
    image = np.ones((3, 5, 3), dtype=np.float32) if separate else np.zeros((3, 5, 3), dtype=np.float32)
    args = (image, np.eye(3)) if name in ('separate_stains', 'combine_stains') else (image,)
    entry = owned._core.stain_transform
    returns, casts = [], []
    def profile(frame, event, value):
        if event == 'c_return' and value is entry:
            returns.append(name)
        if event == 'c_call' and frame.f_code is owned._stain.__code__ and getattr(value, '__name__', '') == 'astype':
            casts.append('astype')
    with strict_owned(monkeypatch, workers):
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            actual = getattr(owned, name)(*args)
        finally:
            sys.setprofile(previous)
    assert returns == [name] and not casts
    assert actual.dtype == np.float64 and actual.shape == image.shape
    np.testing.assert_array_equal(actual, np.zeros_like(actual) if separate else np.ones_like(actual))
