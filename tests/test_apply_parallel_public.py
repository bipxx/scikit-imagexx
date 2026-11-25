"""Public Dask overlap, callback and optional-dependency behavior."""

import builtins
import inspect
from importlib import import_module
from collections import Counter

import numpy as np
import pytest
from skimage import util as original

import skimagexx as xx
from skimagexx import _apply_parallel as owned


@pytest.fixture
def strict():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)


@pytest.fixture
def provider():
    try:
        import dask
        import dask.array as da
    except ImportError:
        errors = []
        for target in (original, xx.util):
            with pytest.raises(RuntimeError, match="Could not import 'dask'") as caught:
                target.apply_parallel(np.negative, np.arange(4))
            errors.append(str(caught.value))
        assert errors[0] == errors[1]
        yield None
        return
    with dask.config.set(scheduler='synchronous'):
        yield da


def test_public_apply_parallel_signature():
    from skimagexx.util import apply_parallel

    assert apply_parallel is xx.util.apply_parallel is owned.apply_parallel
    assert inspect.signature(apply_parallel) == inspect.signature(original.apply_parallel)


@pytest.mark.parametrize('shape', ((), (0,), (1,), (5,), (4, 4), (0, 3), (2, 4), (7, 9, 2)))
@pytest.mark.parametrize('workers', (1, 2, 4, 20))
def test_default_chunks_match_original(shape, workers):
    reference = import_module('skimage.util.apply_parallel')._get_chunks
    if not shape:
        for target in (reference, owned._default_chunks):
            with pytest.raises(ZeroDivisionError):
                target(shape, workers)
    else:
        assert owned._default_chunks(shape, workers) == reference(shape, workers)


@pytest.mark.parametrize(
    'options',
    [
        {},
        dict(chunks=4, dtype='int16'),
        dict(chunks=((3, 2, 2), (4, 5)), depth=(2, 1), mode='wrap'),
        dict(chunks=(3, 4), depth=1, mode='edge'),
        dict(chunks=(3, 4), depth=(1, 0), mode='symmetric', channel_axis=-1),
        dict(chunks=(3, 4), depth=1, mode='reflect', channel_axis=2, dtype='int32'),
        dict(chunks=(3, 4, 2), depth=(1, 0, 0), mode='none', channel_axis=-1),
    ],
)
def test_apply_parallel_eager_and_callback_parity(options, provider, strict):
    if provider is None:
        # The fixture already compared the unavailable-provider contract.
        return
    image = np.arange(7 * 9, dtype=np.int16).reshape(7, 9)[:, ::-1]
    if 'channel_axis' in options:
        image = np.stack((image, -image), axis=-1)

    def pipeline(backend):
        calls = []

        def callback(block, offset, *, scale):
            calls.append((block.shape, block.dtype, block.tobytes()))
            return (np.roll(block, 1, axis=0) + offset) * scale

        result = backend.apply_parallel(
            callback, image, extra_arguments=(1,), extra_keywords={'scale': 2}, **options,
        )
        return result, Counter(calls)

    expected, expected_calls = pipeline(original)
    actual, calls = pipeline(xx.util)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    assert calls == expected_calls


@pytest.mark.parametrize('depth', (0, 1))
def test_eager_ufunc_preserves_chunk_output_and_warnings(depth, provider, strict):
    if provider is None:
        return
    image = np.broadcast_to(np.asarray([1, 1e30, 2, 1e30], dtype=np.float32)[:, None], (4, 3))

    def pipeline(backend):
        with pytest.warns(RuntimeWarning, match='overflow encountered in square') as caught:
            actual = backend.apply_parallel(
                np.square, image, chunks=(2,), depth=depth, mode='symmetric',
                channel_axis=-1, dtype=image.dtype,
            )
        return actual, [str(warning.message) for warning in caught]

    expected, warnings = pipeline(original)
    actual, actual_warnings = pipeline(xx.util)
    np.testing.assert_array_equal(actual, expected)
    assert actual_warnings == warnings


@pytest.mark.parametrize('dask_input', (False, True))
def test_apply_parallel_lazy_provider_and_hot_swap(provider, dask_input, strict, monkeypatch):
    if provider is None:
        return
    image = np.arange(48, dtype=np.uint16).reshape(6, 8)
    if dask_input:
        image = provider.from_array(image, chunks=(2, 3))

    def pipeline(backend):
        return backend.apply_parallel(
            np.negative, image, chunks=(3, 4), depth=1, mode='periodic',
            dtype=image.dtype, compute=None if dask_input else False,
        )

    expected = pipeline(original)

    def forbidden(*args, **kwargs):
        pytest.fail('unexpected upstream apply_parallel dispatch')

    monkeypatch.setattr(original, 'apply_parallel', forbidden)
    actual = pipeline(xx.util)
    assert type(actual) is type(expected) is provider.Array
    assert (actual.shape, actual.dtype, actual.chunks) == (expected.shape, expected.dtype, expected.chunks)
    np.testing.assert_array_equal(actual[1:5, 1:7].compute(), expected[1:5, 1:7].compute())


@pytest.mark.parametrize(
    'image',
    (
        [1, 2, 3, 4],
        np.arange(4).view(type('ArraySubclass', (np.ndarray,), {})),
        np.arange(4, dtype=np.dtype('int64', metadata={'tag': 1})),
    ),
)
def test_apply_parallel_arraylike_fallback(provider, image):
    if provider is None:
        return
    args = (np.negative, image, 2)
    previous = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            xx.util.apply_parallel(*args)
        xx.set_fallback_enabled(True)
        with pytest.warns(xx.FallbackWarning, match='skimage.util.apply_parallel'):
            actual = xx.util.apply_parallel(*args)
        np.testing.assert_array_equal(actual, original.apply_parallel(*args))
    finally:
        xx.set_fallback_enabled(previous)


def test_missing_dask_error_is_preserved(monkeypatch, strict):
    imported = builtins.__import__

    def without_dask(name, *args, **kwargs):
        if name == 'dask.array':
            raise ImportError('no dask array')
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_dask)
    errors = []
    for target in (original, xx.util):
        with pytest.raises(RuntimeError, match="Could not import 'dask'") as caught:
            target.apply_parallel(np.negative, np.arange(4))
        errors.append(str(caught.value))
    assert errors[0] == errors[1]
