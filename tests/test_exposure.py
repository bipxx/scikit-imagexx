"""Native exposure correction contracts against scikit-image 0.26."""
import numpy as np
import pytest
from skimage import exposure as original
import skimagexx as xx


@pytest.fixture(autouse=True, params=[1, 4])
def configuration(request):
    previous = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1)
    xx.set_fallback_enabled(previous)


def compare(name, image, **kwargs):
    actual = getattr(xx.exposure, name)(image, **kwargs)
    expected = getattr(original, name)(image, **kwargs)
    assert actual.dtype == expected.dtype
    if image.dtype == np.uint8:
        np.testing.assert_array_equal(actual, expected)
    else:
        tol = 2e-6 if image.dtype == np.float32 else 2e-13
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize('gamma', [0, 0.1, 0.5, 1, 1.3, 2, 3.7, 10])
@pytest.mark.parametrize('gain', [0, 0.3, 1, 2.5])
def test_all_gamma_uint8_values(gamma, gain):
    compare('adjust_gamma', np.arange(256, dtype=np.uint8).reshape(16, 16), gamma=gamma, gain=gain)


@pytest.mark.parametrize('gain', [0, 0.1, 0.5, 1])
@pytest.mark.parametrize('inv', [False, True])
def test_all_log_uint8_values(gain, inv):
    compare('adjust_log', np.arange(256, dtype=np.uint8).reshape(16, 16), gain=gain, inv=inv)


@pytest.mark.parametrize('gain', [0, 1, 10, 40])
@pytest.mark.parametrize('cutoff', [-0.5, 0, 0.3, 0.5, 1, 1.5])
@pytest.mark.parametrize('inv', [False, True])
def test_all_sigmoid_uint8_values(gain, cutoff, inv):
    compare('adjust_sigmoid', np.arange(256, dtype=np.uint8).reshape(16, 16), gain=gain, cutoff=cutoff, inv=inv)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('name,options', [
    ('adjust_gamma', {'gamma': 0.3, 'gain': 1.7}),
    ('adjust_gamma', {'gamma': 2, 'gain': 0}),
    ('adjust_log', {'gain': 1.9}), ('adjust_log', {'gain': 0.7, 'inv': True}),
    ('adjust_sigmoid', {'gain': 7, 'cutoff': 0.3}),
    ('adjust_sigmoid', {'gain': 7, 'cutoff': 0.7, 'inv': True}),
])
def test_float_ranges_strides_and_no_mutation(dtype, name, options):
    a = np.random.default_rng(43).uniform(0, 3, (51, 67)).astype(dtype).T[::-1, ::2]
    before = a.copy()
    compare(name, a, **options)
    np.testing.assert_array_equal(a, before)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('name', ['adjust_gamma', 'adjust_log', 'adjust_sigmoid'])
def test_negative_inputs(dtype, name):
    a = np.array([[0, 1, -1]], dtype=dtype)
    with pytest.raises(ValueError, match='non-negative'):
        getattr(xx.exposure, name)(a)
    with pytest.raises(ValueError, match='non-negative'):
        getattr(original, name)(a)


def test_unsupported_gain_explicit_fallback():
    a = np.arange(256, dtype=np.uint8).reshape(16, 16)
    with pytest.raises(NotImplementedError):
        xx.exposure.adjust_log(a, gain=2)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        actual = xx.exposure.adjust_log(a, gain=2)
    np.testing.assert_array_equal(actual, original.adjust_log(a, gain=2))


def test_exposure_pipeline_hot_swap():
    a = np.random.default_rng(0).integers(0, 256, (519, 521), dtype=np.uint8)
    def pipeline(module):
        return module.adjust_log(module.adjust_gamma(a, 0.7), inv=True)
    np.testing.assert_array_equal(pipeline(xx.exposure), pipeline(original))


@pytest.mark.parametrize('dtype', [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize('shape', [(), (17,), (3, 5, 7), (2, 3, 5, 7)])
@pytest.mark.parametrize('name', ['adjust_gamma', 'adjust_log', 'adjust_sigmoid', 'rescale_intensity'])
def test_pointwise_dimensions(dtype, shape, name):
    a = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8)
    if dtype != np.uint8:
        a = (a/255).astype(dtype)
    a = np.asarray(a)
    compare(name, a)


@pytest.mark.parametrize('dtype', [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize('in_range', ['image', 'dtype', (0, 0.7), (0.7, 0), (12, 190)])
@pytest.mark.parametrize('out_range', ['image', 'dtype', (0, 0.7), (0.7, 0), (-2, 4)])
def test_rescale_ranges(dtype, in_range, out_range):
    a = np.arange(256, dtype=np.uint8).reshape(16, 16)
    if dtype != np.uint8:
        a = (a/255).astype(dtype)
        a -= dtype(0.3)
    compare('rescale_intensity', a, in_range=in_range, out_range=out_range)


@pytest.mark.parametrize('value', [0, 0.3, 1, 130])
@pytest.mark.parametrize('dtype', [np.uint8, np.float32, np.float64])
def test_rescale_constants(value, dtype):
    a = np.full((17, 19), value, dtype=dtype)
    compare('rescale_intensity', a)
    compare('rescale_intensity', a, out_range=(0, 0.7))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_rescale_nan_behavior_and_warning(dtype):
    a = np.ones((7, 13), dtype=dtype)
    a[2, 3] = np.nan
    with pytest.warns(UserWarning, match='intensity levels are NaN'):
        expected = original.rescale_intensity(a)
    with pytest.warns(UserWarning, match='intensity levels are NaN'):
        actual = xx.exposure.rescale_intensity(a)
    np.testing.assert_array_equal(actual, expected)
    compare('rescale_intensity', a, in_range=(0, 2), out_range=(0, 1))


@pytest.mark.parametrize('source_range', ['image', 'dtype'])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('shape', [(1,), (17, 19), (519, 521)])
def test_histogram_exact_counts_centers_dtype(source_range, normalize, shape):
    a = np.random.default_rng(77).integers(11, 177, shape, dtype=np.uint8)
    actual = xx.exposure.histogram(a, nbins=3, source_range=source_range, normalize=normalize)
    expected = original.histogram(a, nbins=3, source_range=source_range, normalize=normalize)
    for left, right in zip(actual, expected):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('pattern', ['constant', 'binary', 'ramp', 'random'])
def test_equalization_and_cdf(pattern):
    rng = np.random.default_rng(11)
    a = {'constant': np.full((519, 521), 17, np.uint8),
         'binary': rng.integers(0, 2, (519, 521), dtype=np.uint8)*255,
         'ramp': np.arange(256, dtype=np.uint8).reshape(16, 16),
         'random': rng.integers(11, 177, (519, 521), dtype=np.uint8)}[pattern]
    for view in [a, a.T[::-1, ::2]]:
        actual, centers = xx.exposure.cumulative_distribution(view, nbins=3)
        expected, expected_centers = original.cumulative_distribution(view, nbins=3)
        assert actual.dtype == expected.dtype
        assert centers.dtype == expected_centers.dtype
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(centers, expected_centers)
        compare('equalize_hist', view, nbins=3)


@pytest.mark.parametrize('name', ['histogram', 'cumulative_distribution', 'equalize_hist'])
def test_flattened_rgb_warning(name):
    a = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
    with pytest.warns(UserWarning, match='color image'):
        expected = getattr(original, name)(a)
    with pytest.warns(UserWarning, match='color image'):
        actual = getattr(xx.exposure, name)(a)
    np.testing.assert_array_equal(actual, expected)


def test_histogram_options_fallback():
    a = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
    mask = a > 20
    for name, options in [('histogram', {'channel_axis': -1}), ('equalize_hist', {'mask': mask})]:
        with pytest.raises(NotImplementedError):
            getattr(xx.exposure, name)(a, **options)
        xx.set_fallback_enabled(True)
        with pytest.warns(xx.FallbackWarning):
            actual = getattr(xx.exposure, name)(a, **options)
        expected = getattr(original, name)(a, **options)
        if isinstance(expected, tuple):
            for left, right in zip(actual, expected):
                np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_array_equal(actual, expected)
        xx.set_fallback_enabled(False)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('gamma', [0, 0.01, 0.3, 0.7, 1.3, 2.7, 5])
def test_vector_math_dynamic_range(dtype, gamma):
    values = np.logspace(-20, 20, 1025).astype(dtype)
    # Keep the expected power representable in float32 as well as float64.
    if dtype == np.float32:
        values = np.minimum(values, dtype(1e6))
    a = values.reshape(25, 41)
    compare('adjust_gamma', a, gamma=gamma, gain=0.7)
    compare('adjust_log', a, gain=1.3)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('gain', [0, 0.7, 10, 40, 1000])
def test_vector_sigmoid_range_and_fallback_lanes(dtype, gain):
    a = np.linspace(0, 10, 1025, dtype=dtype).reshape(25, 41)
    with np.errstate(over='ignore'):
        compare('adjust_sigmoid', a, gain=gain, cutoff=0.7)
        compare('adjust_sigmoid', a, gain=gain, cutoff=0.7, inv=True)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('name', ['adjust_gamma', 'adjust_log', 'adjust_sigmoid'])
def test_nonfinite_exposure_values(dtype, name):
    a = np.array([[0, -0.0, 1, np.nan, np.inf, np.nextafter(dtype(0), dtype(1)), 0.3]], dtype=dtype)
    with np.errstate(all='ignore'):
        compare(name, a)
