"""Large integer constants promote NumPy 1.x float32 chroma stages."""
from pathlib import Path
import warnings
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


@pytest.mark.parametrize('length', list(range(18)) + [31, 32, 33, 65, 262149])
@pytest.mark.parametrize('options', [{}, {'kC': 0}, {'kH': 0}, {'kL': 2, 'kC': .5}])
def test_mixed_precision_packets_and_nonfinite_masks(length, options):
    data = np.loadtxt(Path(__file__).parent / 'data/ciede2000-sharma.txt')
    first = np.resize(data[:, :3].astype(np.float32), (length, 3))
    second = np.resize(data[:, 3:6].astype(np.float32), (length, 3))
    first.flags.writeable = False
    second.flags.writeable = False
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = reference.deltaE_ciede2000(first, second, **options)
        actual = xx.color.deltaE_ciede2000(first, second, **options)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2.5e-4)


@pytest.mark.parametrize('layout', ['scalar', 'array', 'channel-first', 'reversed', 'broadcast'])
def test_discontinuous_pair_layouts(layout):
    data = np.loadtxt(Path(__file__).parent / 'data/ciede2000-sharma.txt')
    a = np.tile(data[9, :3].astype(np.float32), (9, 11, 1))
    b = np.tile(data[9, 3:6].astype(np.float32), (9, 11, 1))
    axis = -1
    if layout == 'scalar': a, b = a[0, 0], b[0, 0]
    elif layout == 'channel-first': a, b, axis = np.moveaxis(a, -1, 0), np.moveaxis(b, -1, 0), 0
    elif layout == 'reversed': a, b = a[::-1, ::-1], b[::-1, ::-1]
    elif layout == 'broadcast': b = b[:1, :1]
    expected = reference.deltaE_ciede2000(a, b, channel_axis=axis)
    actual = xx.color.deltaE_ciede2000(a, b, channel_axis=axis)
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2.5e-4)


def test_broadcast_channel_axis_error_parity():
    # Upstream normalizes against the first input before moving either axis.
    a, b = np.zeros((2, 3, 3), np.float32), np.zeros(3, np.float32)
    with pytest.raises(getattr(np, 'exceptions', np).AxisError) as expected:
        reference.deltaE_ciede2000(a, b)
    with pytest.raises(type(expected.value)) as actual:
        xx.color.deltaE_ciede2000(a, b)
    assert str(actual.value) == str(expected.value)
