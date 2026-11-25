"""Public hot-swap parity for the bounded native rotate and swirl routes."""

import inspect
import sys
import warnings

import numpy as np
import pytest


CASES = (
    ('rotate', np.float32, (19, 23), dict(angle=31.5)),
    ('rotate', np.uint8, (24, 17, 3), dict(angle=90, resize=True, preserve_range=True)),
    ('rotate', np.float64, (8, 13), dict(angle=-53, resize=True, center=(6.25, 3.5),
                                        order=3, mode='wrap', clip=False)),
    ('rotate', np.bool_, (9, 14), dict(angle=90, order=0, mode='edge')),
    ('rotate', np.float64, (7, 9), dict(angle=17, order=2, mode='symmetric')),
    ('rotate', np.float32, (1, 7), dict(angle=24, order=1, mode='reflect')),
    ('swirl', np.float64, (17, 21), dict()),
    ('swirl', np.float32, (12, 14, 3), dict(center=(3.25, 7.5), strength=-2, radius=34,
                                          rotation=0.4, output_shape=(19, 11, 3), mode='wrap')),
    ('swirl', np.bool_, (16, 9), dict(center=(2, 3), radius=17, order=0, mode='edge')),
    ('swirl', np.uint8, (19, 13), dict(strength=1.75, radius=26, rotation=1.4,
                                     order=2, mode='constant', cval=27, preserve_range=True)),
    ('swirl', np.float64, (11, 17), dict(strength=-1.7, radius=3.25, order=3, mode='symmetric')),
)


def _observe(fn, image, options):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            output, error = fn(image, **options), None
        except Exception as caught:
            output, error = None, (type(caught), str(caught))
    return output, error, [(item.category, str(item.message)) for item in notices]


def _sample(dtype, shape):
    rng = np.random.default_rng(47213)
    if dtype is np.bool_:
        return rng.random(shape) > 0.5
    if dtype is np.uint8:
        return rng.integers(0, 256, size=shape, dtype=dtype)
    return rng.uniform(-0.8, 1.1, size=shape).astype(dtype)


def _assert_result(actual, expected, image, order):
    assert type(actual) is type(expected) is np.ndarray
    assert actual.dtype == expected.dtype
    assert actual.dtype.metadata == expected.dtype.metadata
    assert actual.shape == expected.shape and actual.strides == expected.strides
    assert actual.flags.writeable == expected.flags.writeable
    assert not np.shares_memory(actual, image)
    if order == 0 or actual.dtype.kind in 'biu':
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = 3e-6 if actual.dtype == np.float32 else 3e-13
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('name,dtype,shape,options', CASES)
def test_native_public_parity(name, dtype, shape, options):
    from skimage import transform as original
    import skimagexx as xx
    from skimagexx import _core, _rotate_swirl

    image = _sample(dtype, shape)
    before = image.copy()
    public, stock = getattr(xx.transform, name), getattr(original, name)
    assert public is getattr(_rotate_swirl, name)
    assert inspect.signature(public) == inspect.signature(stock)
    expected, error, notices = _observe(stock, image, options)
    assert error is None
    builtin = _core.warp_sample
    assert inspect.isbuiltin(builtin)
    visited = []

    def trace(frame, event, result):
        if event == 'c_return' and result is builtin:
            visited.append(result)

    initial = xx.get_num_threads()
    fallback = xx.set_fallback_enabled(False)
    profile = sys.getprofile()
    try:
        outputs = []
        worker_counts = (1, 4) if _core.has_openmp else (1,)
        for workers in worker_counts:
            xx.set_num_threads(workers)
            try:
                sys.setprofile(trace)
                output, error, emitted = _observe(public, image, options)
            finally:
                sys.setprofile(profile)
            assert error is None
            assert emitted == notices
            _assert_result(output, expected, image, options.get('order'))
            outputs.append(output)
        if len(outputs) == 2:
            np.testing.assert_array_equal(*outputs)
    finally:
        xx.set_num_threads(initial)
        xx.set_fallback_enabled(fallback)
    assert visited == [builtin] * len(worker_counts)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('name,layout,options', (
    ('rotate', 'readonly', dict(angle=25)),
    ('rotate', 'tagged', dict(angle=17)),
    ('rotate', 'unsupported-dtype', dict(angle=18, resize=True)),
    ('rotate', 'contiguous', dict(angle=27, order=5)),
    ('swirl', 'strided', dict(radius=8)),
    ('swirl', 'contiguous', dict(output_shape=(5, 4.5))),
))
def test_compatibility_routes(name, layout, options):
    from skimage import transform as original
    import skimagexx as xx

    image = _sample(np.float64, (9, 11))
    if layout == 'readonly':
        image.flags.writeable = False
    elif layout == 'tagged':
        image = image.view(np.dtype(np.float64, metadata={'test': 'warp'}))
    elif layout == 'strided':
        image = image[:, ::2]
    elif layout == 'unsupported-dtype':
        image = image.astype(np.float16)
    public, stock = getattr(xx.transform, name), getattr(original, name)
    expected, error, notices = _observe(stock, image, options)
    fallback = xx.set_fallback_enabled(True)
    try:
        actual, actual_error, emitted = _observe(public, image, options)
        assert actual_error == error
        assert [(category, message) for category, message in emitted
                if category is not xx.FallbackWarning] == notices
        assert [message for category, message in emitted if category is xx.FallbackWarning] == [
            f'skimage.transform.{name} executed by scikit-image fallback']
        if error is None:
            _assert_result(actual, expected, image, options.get('order'))
        xx.set_fallback_enabled(False)
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            public(image, **options)
    finally:
        xx.set_fallback_enabled(fallback)
