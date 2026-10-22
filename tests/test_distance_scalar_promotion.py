"""Scalar promotion must preserve early float32 rounding, not cast the answer."""
import numpy as np
import pytest
from skimage import color as reference
import skimagexx as xx


@pytest.fixture(autouse=True, params=[1, 4])
def configuration(request):
    previous = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


@pytest.mark.parametrize('name', ['deltaE_cie76', 'deltaE_ciede94'])
@pytest.mark.parametrize('layout', ['scalar', 'reversed', 'unaligned', 'array', 'broadcast'])
def test_scalar_and_array_promotion(name, layout):
    rng = np.random.default_rng(864)
    for _ in range(12):
        a, b = (rng.uniform(-100, 100, (2, 3))).astype(np.float32)
        if layout == 'reversed':
            a, b = a[::-1], b[::-1]
        elif layout == 'unaligned':
            a = np.ndarray((3,), np.float32, bytearray(13), offset=1)
            a[:] = [20, 10, 5]
        elif layout == 'array':
            a, b = a[None, :], b[None, :]
        elif layout == 'broadcast':
            a = np.broadcast_to(a, (2, 3))
        expected = getattr(reference, name)(a, b)
        actual = getattr(xx.color, name)(a, b)
        assert type(actual) is type(expected)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('a,b', [([20, 10, 5], [25, 15, 10]),
                              ([16777216, 2, 3], [-1, 0, 0])])
def test_cie76_legacy_precision(a, b):
    a, b = np.array(a, np.float32), np.array(b, np.float32)
    actual = xx.color.deltaE_cie76(a, b)
    expected = reference.deltaE_cie76(a, b)
    assert actual.dtype == expected.dtype
    if np.asarray(np.float32(1) ** 2).dtype == np.float64:
        difference = (b - a).astype(np.float64)
        golden = np.sqrt(sum(float(v) ** 2 for v in difference))
        assert actual == golden
        assert actual != np.float64(np.float32(golden))
    else:
        assert actual == expected


@pytest.mark.parametrize('parameter', ['kL', 'kC', 'kH', 'k1', 'k2'])
@pytest.mark.parametrize('kind', [int, float, np.float32])
def test_cie94_scalar_parameter_types(parameter, kind):
    a = np.array([20, 10, 5], np.float32)
    b = np.array([25, 15, 10], np.float32)
    options = {parameter: kind(2 if kind is int else 0.731)}
    expected = reference.deltaE_ciede94(a, b, **options)
    actual = xx.color.deltaE_ciede94(a, b, **options)
    assert actual.dtype == expected.dtype
    # Fixed inputs have identical initial hypot results on supported platforms.
    tolerance = 2e-14 if expected.dtype == np.float64 else 2e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
