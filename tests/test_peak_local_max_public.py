import inspect
import warnings

import numpy as np
import pytest
import skimage
from skimage import feature as stock

import skimagexx as xx
from skimagexx import _core
from skimagexx import _peak_local_max as owned
from skimagexx._compat import FallbackWarning


@pytest.fixture
def strict():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)


def assert_same(actual, expected):
    assert type(actual) is type(expected) is np.ndarray
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if actual.dtype.kind == 'f':
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        tolerance = 2e-6 if actual.dtype == np.float32 else 1e-12
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance, equal_nan=True)
    else:
        np.testing.assert_array_equal(actual, expected)


def call_native(name, *args, **kwargs):
    expected = getattr(stock, name)(*args, **kwargs)
    actual = getattr(xx.feature, name)(*args, **kwargs)
    assert_same(actual, expected)
    assert all(not np.shares_memory(actual, arr) for arr in (*args, *kwargs.values())
               if isinstance(arr, np.ndarray))
    return actual


@pytest.mark.parametrize('name,entry', [('peak_local_max', 'peak_local_max_2d'),
                                       ('corner_peaks', 'peak_local_max_2d'),
                                       ('corner_subpix', 'corner_subpix_2d')])
def test_installed_signature_and_binding(name, entry, monkeypatch, strict):
    assert skimage.__version__ == '0.26.0'
    public = getattr(xx.feature, name)
    assert public is getattr(owned, name)
    assert inspect.signature(public) == inspect.signature(getattr(stock, name))
    binding = getattr(_core, entry)
    assert inspect.isbuiltin(binding) and binding.__module__ == 'skimagexx._core'
    calls = []

    def trace(*args):
        calls.append(args)
        return binding(*args)

    monkeypatch.setattr(_core, entry, trace)
    image = np.zeros((9, 11))
    image[4, 4] = 1
    corners = np.array([[4, 4]], dtype=np.intp)
    args = (image, corners) if name == 'corner_subpix' else (image,)
    call_native(name, *args)
    assert len(calls) == 1 and calls[0][0] is image
    if name == 'corner_subpix':
        assert calls[0][1] is corners


@pytest.mark.parametrize('name', ['peak_local_max', 'corner_peaks'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.uint8])
@pytest.mark.parametrize('norm', [1, 1.5, 2, 3, np.inf])
def test_peaks_and_connected_suppression(name, dtype, norm, strict):
    image = np.zeros((19, 23), dtype=dtype)
    image[2:7, 3:10] = 7
    image[14:17, 15:18] = 7
    image[0, 1] = image[18, 22] = 9
    image[10, 12] = 2
    before = image.copy()
    for kwargs in ({'exclude_border': (1, 0), 'min_distance': 3, 'p_norm': norm},
                   {'exclude_border': False, 'min_distance': 2, 'p_norm': norm,
                    'num_peaks': 5, 'threshold_abs': 2, 'threshold_rel': 0.2}):
        call_native(name, image, **kwargs)
        if name == 'corner_peaks':
            call_native(name, image, indices=False, **kwargs)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('labels_dtype', [np.int32, np.int64])
def test_label_order_caps_border_and_isolated_opening(labels_dtype, strict):
    image = np.zeros((20, 20))
    labels = np.zeros(image.shape, dtype=labels_dtype)
    # Label 1 contains a cross and isolated members. Only the isolated members
    # survive the original trivial-ROI binary-opening rule.
    for row, col in [(3, 3), (8, 8), (7, 8), (9, 8), (8, 7), (8, 9), (12, 12), (15, 16)]:
        labels[row, col] = 1
        image[row, col] = 4
    labels[15:18, 2:5] = 13
    image[16, 3] = 8
    labels[0, 17] = 2
    image[0, 17] = 9
    prior_labels = labels.copy()
    result = call_native('peak_local_max', image, labels=labels, min_distance=2, num_peaks_per_label=2)
    np.testing.assert_array_equal(result, [[3, 3], [12, 12], [16, 3]])
    for name in ('peak_local_max', 'corner_peaks'):
        call_native(name, image, labels=labels, min_distance=2, exclude_border=False,
                    num_peaks=2, num_peaks_per_label=2, p_norm=2)
        call_native(name, image, labels=labels, num_peaks=0, num_peaks_per_label=0)
        call_native(name, image, labels=np.zeros_like(labels))
    np.testing.assert_array_equal(labels, prior_labels)


def test_unsigned_negation_and_float32_relative_rounding(strict):
    integer = np.zeros((9, 11), dtype=np.uint8)
    integer[4, 7] = 255
    result = call_native('peak_local_max', integer, threshold_abs=-1, num_peaks=1)
    np.testing.assert_array_equal(result, [[1, 1]])
    float_image = np.zeros((12, 12), dtype=np.float32)
    high = np.float32(2.031)
    relative = 0.71
    product = np.float32(relative * float(high))
    float_image[3, 3] = high
    float_image[6, 3] = np.nextafter(product, np.float32(-np.inf))
    float_image[6, 6] = product
    float_image[9, 9] = np.nextafter(product, np.float32(np.inf))
    call_native('peak_local_max', float_image, threshold_rel=relative)
    call_native('corner_peaks', float_image, threshold_rel=relative)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.uint8])
@pytest.mark.parametrize('window', [3, 6, 11, 15])
def test_subpixel_edges_dots_and_singular(dtype, window, strict):
    image = np.zeros((57, 61), dtype=dtype)
    image[1:19, 1:19] = 40
    image[19:37, 19:37] = 40
    image[36, 10] = 50
    corners = np.array([[19, 19], [36, 10], [48, 48], [1, 1], [19, 19]], dtype=np.int32)
    before_image, before_corners = image.copy(), corners.copy()
    result = call_native('corner_subpix', image, corners, window_size=window)
    assert np.isnan(result[2]).all()
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(corners, before_corners)
    call_native('corner_subpix', image, np.empty((0, 2), dtype=np.intp), window_size=window)


@pytest.mark.parametrize('name', ['peak_local_max', 'corner_peaks', 'corner_subpix'])
@pytest.mark.parametrize('reason', ['read-only', 'view', 'subclass', 'metadata', 'byte-order', 'longdouble'])
def test_compatibility_routing(name, reason, strict):
    image = np.zeros((15, 17))
    image[5, 6] = 1
    if reason == 'read-only':
        image.setflags(write=False)
    elif reason == 'view':
        image = image.T
    elif reason == 'subclass':
        image = image.view(type('Image', (np.ndarray,), {}))
    elif reason == 'metadata':
        image = image.view(np.dtype('float64', metadata={'peak': 'measure'}))
    elif reason == 'byte-order':
        image = image.astype(np.dtype('float64').newbyteorder('S'))
    else:
        image = image.astype(np.longdouble)
    corners = np.array([[5, 6]], dtype=np.intp)
    args = (image, corners) if name == 'corner_subpix' else (image,)
    public, original = getattr(xx.feature, name), getattr(stock, name)
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        public(*args)
    xx.set_fallback_enabled(True)
    # SciPy may reject genuine extended precision and reports that error
    # through fallback just as it does on a direct original call.
    try:
        expected = original(*args)
    except (TypeError, ValueError, RuntimeError) as error:
        with pytest.warns(FallbackWarning), pytest.raises(type(error)) as observed:
            public(*args)
        assert str(observed.value) == str(error)
    else:
        with pytest.warns(FallbackWarning):
            actual = public(*args)
        assert_same(actual, expected)


def test_stock_warning_and_custom_footprint_fallback(strict):
    image = np.eye(5)
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        xx.feature.peak_local_max(image, footprint=np.eye(3, dtype=bool))
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual = xx.feature.peak_local_max(image, min_distance=0)
    assert [warning.category for warning in caught] == [FallbackWarning, RuntimeWarning]
    with pytest.warns(RuntimeWarning):
        expected = stock.peak_local_max(image, min_distance=0)
    assert_same(actual, expected)


def test_direct_binding_checked_storage():
    from numpy.lib.stride_tricks import as_strided

    invalid = as_strided(np.zeros(1), shape=(5, 5), strides=(40, 8))
    with pytest.raises(ValueError, match='ndarray-backed storage'):
        _core.peak_local_max_2d(invalid, None, 1, None, None, (0, 0), 50, 50, 0, False, True)
