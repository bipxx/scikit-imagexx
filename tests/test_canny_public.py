import inspect
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from skimage.feature import canny as original

import skimagexx as xx
from skimagexx import _core
from skimagexx._canny import canny
from skimagexx._compat import FallbackWarning


@pytest.fixture
def fallback_disabled():
    prior = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(prior)


def assert_parity(image, **kwargs):
    snapshot = image.copy()
    mask = kwargs.get('mask')
    mask_snapshot = mask.copy() if mask is not None else None
    expected = original(image, **kwargs)
    got = canny(image, **kwargs)
    assert type(got) is np.ndarray
    assert got.dtype == np.bool_ and got.flags.c_contiguous and got.flags.writeable
    assert not np.shares_memory(got, image)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(image, snapshot)
    if mask is not None:
        assert not np.shares_memory(got, mask)
        np.testing.assert_array_equal(mask, mask_snapshot)


def test_native_export_and_hot_swap(fallback_disabled, monkeypatch):
    from skimage import feature as stock_feature

    assert xx.feature.canny is canny
    assert inspect.signature(canny) == inspect.signature(original)
    assert inspect.isbuiltin(_core.canny_native) and _core.canny_native.__module__ == 'skimagexx._core'
    exports = {}
    exec('from skimagexx.feature import *', exports)
    assert exports['canny'] is canny

    image = np.zeros((9, 11), dtype=np.uint8)
    image[2:7, 4:8] = 255
    mask = np.ones_like(image, dtype=bool)
    calls = []
    native = _core.canny_native

    def probe(*args):
        calls.append(args[:2])
        return native(*args)

    monkeypatch.setattr(_core, 'canny_native', probe)

    def pipeline(backend):
        return backend.feature.canny(image, 1.2, 40, 80, mask, False, mode='mirror')

    expected = pipeline(SimpleNamespace(feature=stock_feature))
    assert np.any(expected)
    np.testing.assert_array_equal(pipeline(xx), expected)
    assert len(calls) == 1 and calls[0][0] is image and calls[0][1] is mask


@pytest.mark.parametrize('dtype', [np.bool_, np.int8, np.uint8, np.int16, np.uint16,
                                  np.int32, np.uint32, np.float32, np.float64])
def test_dtype_threshold_and_mask_matrix(dtype, fallback_disabled):
    rng = np.random.default_rng(927366)
    shape = (19, 31)
    if dtype is np.bool_:
        image = rng.random(shape) > .65
        low, high = .12, .42
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        image = rng.integers(info.min, info.max, size=shape, dtype=dtype)
        low, high = info.max * .12, info.max * .42
    else:
        image = (rng.random(shape) * 2 - 1).astype(dtype)
        low, high = .12, .42
    mask = np.ones(shape, dtype=bool)
    mask[3:5, :] = False
    mask[:, 12:16] = False
    mask[8:12, 19:22] = False
    for mode, cval in [('constant', 0), ('constant', .25), ('grid-constant', .125),
                       ('wrap', 0), ('mirror', 0), ('grid-mirror', 0), ('nearest', 0)]:
        for selected in [None, mask]:
            assert_parity(image, sigma=.8, low_threshold=low, high_threshold=high,
                          mode=mode, cval=cval, mask=selected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.uint8])
def test_quantiles_gradients_and_edge_plateaus(dtype, fallback_disabled):
    rng = np.random.default_rng(55947)
    for shape in [(1, 7), (2, 9), (5, 3), (9, 8), (17, 28)]:
        step = np.zeros(shape, dtype=dtype)
        step[:, shape[1] // 2:] = 1 if dtype != np.uint8 else 255
        random = (rng.random(shape) * (1 if dtype != np.uint8 else 255)).astype(dtype)
        mask = rng.random(shape) > .08
        for image in [step, random]:
            for low, high in [(0, 1), (.2, .2), (.43, .78), (None, None)]:
                for sigma in [0, .09, 1, 1.7]:
                    assert_parity(image, sigma=sigma, mask=mask, low_threshold=low,
                                  high_threshold=high, use_quantiles=True)
    for sigma in [0, .55, .8, 1, 1.75, 3]:
        for diagonal in [-1, 0, 1]:
            y, x = np.indices((33, 49))
            image = ((x + diagonal * y) > 23).astype(dtype)
            if dtype is np.uint8:
                image *= 255
            assert_parity(image, sigma=sigma)
            assert_parity(image, sigma=sigma, mode='reflect', low_threshold=0, high_threshold=0)


def test_workers_preserve_result(fallback_disabled):
    image = np.random.default_rng(1126).random((528, 508), dtype=np.float32)
    mask = np.ones(image.shape, dtype=bool)
    mask[55:130, 75:175] = False
    expected = original(image, mask=mask, sigma=1.3)
    prior = xx.get_num_threads()
    assert prior >= 1
    try:
        for workers in (1, 4):
            xx.set_num_threads(workers)
            np.testing.assert_array_equal(canny(image, mask=mask, sigma=1.3), expected)
    finally:
        xx.set_num_threads(prior)


class ArraySubclass(np.ndarray):
    pass


def fallback_cases():
    image = np.arange(90, dtype=np.float64).reshape(9, 10) / 90
    readonly = image.copy()
    readonly.flags.writeable = False
    readonly_mask = np.ones(image.shape, dtype=bool)
    readonly_mask.flags.writeable = False
    swapped = image.astype(image.dtype.newbyteorder('S'))
    metadata = image.astype(np.dtype('float64', metadata={'unit': 'intensity'}))
    nonfinite = image.copy()
    nonfinite[3, 3] = np.nan
    foreign_owner = np.frombuffer(bytearray(image.tobytes()), dtype=image.dtype).reshape(image.shape)
    return [
        (readonly, {}), (image[:, ::2], {}), (np.asfortranarray(image), {}), (swapped, {}),
        (image.view(ArraySubclass), {}), (image.astype(np.float16), {}), (metadata, {}),
        (image.astype(np.longdouble), {}), (image.astype(np.int64), {}), (nonfinite, {}),
        (foreign_owner, {}), (image, {'mask': readonly_mask}),
        (image, {'mask': np.ones((9, 20), dtype=bool)[:, ::2]}),
        (image, {'mask': image > .2, 'sigma': (1., 2.)}),
        (image, {'sigma': -1}), (image, {'low_threshold': .4, 'high_threshold': .2}),
        (image, {'low_threshold': -1, 'use_quantiles': True}), (image, {'mode': 'invalid'}),
        (image, {'cval': -.25}), (image, {'mask': np.zeros((1, 1), dtype=bool)}),
        (np.empty((0, 0), dtype=np.float32), {}), (np.zeros(8, dtype=np.uint8), {}),
    ]


@pytest.mark.parametrize('image,options', fallback_cases(), ids=[
    'readonly', 'strided', 'fortran', 'swapped', 'subclass', 'float16', 'metadata',
    'longdouble', 'int64', 'nonfinite', 'foreign-owner', 'readonly-mask', 'strided-mask',
    'anisotropic-sigma', 'negative-sigma', 'decreasing-threshold', 'invalid-quantile',
    'unknown-mode', 'negative-constant', 'mask-shape', 'empty', 'wrong-rank',
])
def test_fallback_contract(image, options, fallback_disabled):
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        canny(image, **options)
    xx.set_fallback_enabled(True)

    def observe(function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result, error = function(image, **options), None
            except Exception as exc:
                result, error = None, (type(exc), str(exc))
        return result, error, [(w.category, str(w.message)) for w in caught]

    result, error, notices = observe(canny)
    expected, expected_error, expected_notices = observe(original)
    assert error == expected_error
    assert sum(category is FallbackWarning for category, _ in notices) == 1
    assert [notice for notice in notices if notice[0] is not FallbackWarning] == expected_notices
    if error is None:
        np.testing.assert_array_equal(result, expected)


def test_private_binding_rejects_unowned_and_unsupported_layouts(fallback_disabled):
    image = np.ones((5, 7), dtype=np.float64)
    dangling = np.lib.stride_tricks.as_strided(image, shape=(7, 7), strides=image.strides)
    readonly = image.copy()
    readonly.flags.writeable = False
    empty_weights = np.empty(0, dtype=np.float64)
    for invalid in [dangling, readonly, image.view(ArraySubclass), image.astype(np.longdouble), image[:, ::2]]:
        assert _core.canny_native(invalid, None, empty_weights, 1, 0, .1, .2, False, False, True) is None
    assert _core.canny_native(image, dangling, empty_weights, 1, 0, .1, .2, False, False, True) is None
