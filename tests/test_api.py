"""Public interchangeability, native dispatch, and per-function reference tests."""
import inspect
import numpy as np
import pytest
import skimage
from skimage import filters as reference_filters, color as reference_color, util as reference_util
import skimagexx as xx
from skimagexx.coverage import native_functions, native_dtypes, needs_color_input, default_call, parity_tolerance
from skimage import exposure as reference_exposure

MODULES = [(xx.filters, reference_filters), (xx.color, reference_color), (xx.util, reference_util),
           (xx.exposure, reference_exposure)]
FUNCTIONS = [(native, original, name) for native, original in MODULES for name in native.__all__]


def test_registry_matches_per_function_tests():
    tested = {module.__name__.split('.', 1)[1]+'.'+name for module, _, name in FUNCTIONS}
    assert tested == set(native_functions())


def test_unported_namespace_interchangeability():
    a = np.zeros((9, 11), dtype=bool)
    a[3:6, 4:7] = True
    with pytest.raises(NotImplementedError):
        xx.morphology.dilation(a)
    xx.set_fallback_enabled(True)
    from skimage import morphology
    with pytest.warns(xx.FallbackWarning):
        actual = xx.morphology.dilation(a)
    np.testing.assert_array_equal(actual, morphology.dilation(a))


@pytest.fixture(autouse=True, params=[1, 4])
def configuration(request):
    previous = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


@pytest.mark.parametrize('native,original,name', FUNCTIONS, ids=[x[2] for x in FUNCTIONS])
def test_signatures(native, original, name):
    assert inspect.signature(getattr(native, name)) == inspect.signature(getattr(original, name))


@pytest.mark.parametrize('native,original,name', FUNCTIONS, ids=[x[2] for x in FUNCTIONS])
@pytest.mark.parametrize('dtype', [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize('shape', [(1, 1), (3, 19), (519, 521)])
def test_every_native_function(native, original, name, dtype, shape):
    rng = np.random.default_rng(15)
    api = native.__name__.split('.', 1)[1]+'.'+name
    a = rng.integers(0, 256, (*shape, 3) if needs_color_input(api) else shape, dtype=np.uint8)
    if dtype != np.uint8:
        a = (a/255).astype(dtype)
    if np.dtype(dtype).name not in native_dtypes(api):
        with pytest.raises(NotImplementedError):
            getattr(native, name)(a)
        return
    args, options = default_call(api,a)
    actual = getattr(native, name)(*args,**options)
    expected = getattr(original, name)(*args,**options)
    assert np.asarray(actual).dtype == np.asarray(expected).dtype
    rtol,atol = parity_tolerance(api,np.dtype(dtype))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize('name', ['sobel', 'prewitt', 'scharr'])
@pytest.mark.parametrize('axis', [0, 1])
def test_gradient_axes(name, axis):
    a = np.random.default_rng(0).random((17, 23), dtype=np.float32)
    np.testing.assert_allclose(getattr(xx.filters, name)(a, axis=axis),
                               getattr(reference_filters, name)(a, axis=axis), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('axis', [-3, -2, -1, 0, 1, 2])
def test_channel_axes(axis):
    a = np.random.default_rng(0).random((13, 17))
    rgb = xx.color.gray2rgb(a, channel_axis=axis)
    np.testing.assert_array_equal(rgb, reference_color.gray2rgb(a, channel_axis=axis))
    np.testing.assert_allclose(xx.color.rgb2gray(rgb, channel_axis=axis),
                               reference_color.rgb2gray(rgb, channel_axis=axis), atol=2e-13)


@pytest.mark.parametrize('signed', [False, True])
def test_invert_signed_and_nonfinite(signed):
    a = np.array([[0.0, -0.0, np.inf, -np.inf, np.nan, -1, 2]])
    expected = reference_util.invert(a, signed)
    actual = xx.util.invert(a, signed)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_pipeline_can_swap_backend():
    a = np.random.default_rng(0).integers(0, 256, (37, 41, 3), dtype=np.uint8)
    def pipeline(backend):
        gray = backend.color.rgb2gray(a)
        smooth = backend.filters.gaussian(gray, 1.2, mode='reflect')
        edges = backend.filters.scharr(smooth, axis=0)
        return edges > backend.filters.threshold_mean(edges)
    np.testing.assert_array_equal(pipeline(xx), pipeline(skimage))


def test_explicit_fallback_for_unsupported_options():
    a = np.arange(81, dtype=np.uint16).reshape(9, 9)
    with pytest.raises(NotImplementedError):
        xx.filters.gaussian(a, mode='wrap')
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.gaussian(a, mode='wrap')
    np.testing.assert_array_equal(actual, reference_filters.gaussian(a, mode='wrap'))
    out = np.empty(a.shape)
    with pytest.warns(xx.FallbackWarning):
        result = xx.filters.gaussian(a, out=out)
    assert result is out
    np.testing.assert_array_equal(out, reference_filters.gaussian(a))
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.threshold_otsu(a, nbins=32)
    assert actual == reference_filters.threshold_otsu(a, nbins=32)


def test_unported_function_is_never_native():
    a = np.arange(9.0).reshape(3, 3)
    with pytest.raises(NotImplementedError):
        xx.filters.laplace(a)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.laplace(a)
    np.testing.assert_array_equal(actual, reference_filters.laplace(a))


def test_mask_sequence_axis_and_histogram_fallback():
    xx.set_fallback_enabled(True)
    a = np.arange(81.0).reshape(9, 9)
    mask = a > 10
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.sobel(a, mask, axis=[0, 1], mode='nearest')
    np.testing.assert_array_equal(actual, reference_filters.sobel(a, mask, axis=[0, 1], mode='nearest'))
    hist = np.histogram(a, bins=16)
    hist = (hist[0], (hist[1][:-1]+hist[1][1:])/2)
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.threshold_otsu(hist=hist)
    assert actual == reference_filters.threshold_otsu(hist=hist)
    # Preserve 0.26's handling of negative axes instead of normalizing it.
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.sobel(a, axis=-1)
    np.testing.assert_array_equal(actual, reference_filters.sobel(a, axis=-1))
