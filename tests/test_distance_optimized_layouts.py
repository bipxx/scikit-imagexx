"""Exact native graph reuse for constant CMC RHS and bounded CIE76 loads."""
import numpy as np
import pytest
from skimage import color as reference
import skimagexx as xx


@pytest.fixture(autouse=True, params=[1, 4])
def native_workers(request):
    old_workers = xx.get_num_threads()
    old_fallback = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    try:
        yield
    finally:
        xx.set_num_threads(old_workers)
        xx.set_fallback_enabled(old_fallback)


@pytest.mark.parametrize('count', [3, 4, 5, 8, 9, 17, 262149])
@pytest.mark.parametrize('reverse_channels', [False, True])
def test_cmc_constant_rhs_matches_repeated_rhs_words(count, reverse_channels):
    a = (np.arange(count * 3, dtype=np.float32).reshape(count, 3) - 23) / np.float32(7)
    color = np.array([31.5, -12.25, 7.75], dtype=np.float32)
    if reverse_channels:
        color = color[::-1]
    b = np.broadcast_to(color, a.shape)
    a.flags.writeable = False
    before_a, before_b = a.tobytes(), b.tobytes()
    repeated = xx.color.deltaE_cmc(a, b.copy())
    actual = xx.color.deltaE_cmc(a, b)
    np.testing.assert_array_equal(actual.view(np.uint32), repeated.view(np.uint32))
    expected = reference.deltaE_cmc(a, b)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert type(actual) is type(expected) and actual.dtype == expected.dtype
    assert actual.shape == expected.shape and not np.shares_memory(actual, a)
    assert a.tobytes() == before_a and b.tobytes() == before_b


@pytest.mark.parametrize('word', [0x80000000, 0x00000001, 0x7f000000, 0x7f800000, 0x7fc00001, 0x7f800001])
def test_cmc_constant_rhs_domain_keeps_repeated_rhs_graph(word):
    # Signed zero remains eligible; subnormal, extreme, infinity, qNaN and
    # signaling NaN inputs retain the old graph.
    a = np.arange(27, dtype=np.float32).reshape(9, 3)
    color = np.array([0x41fc0000, word, 0x40f80000], dtype=np.uint32).view(np.float32)
    b = np.broadcast_to(color, a.shape)
    with np.errstate(all='ignore'):
        repeated = xx.color.deltaE_cmc(a, b.copy())
        actual = xx.color.deltaE_cmc(a, b)
    np.testing.assert_array_equal(actual.view(np.uint32), repeated.view(np.uint32))


@pytest.mark.parametrize('count', [7, 8, 9, 15, 16, 17])
def test_cie76_reverse_and_four_stride_exact_three_channel_span(count):
    # The final pixel has exactly three accessible floats: no padding lane.
    storage = np.arange((count - 1) * 4 + 3, dtype=np.float32)
    a = np.ndarray((count, 3), dtype=np.float32, buffer=storage, strides=(16, 4))
    b = (np.arange(count * 3, dtype=np.float32).reshape(count, 3) / np.float32(3))[::-1]
    a.flags.writeable = False
    b.flags.writeable = False
    before_a, before_b = a.tobytes(), b.tobytes()
    dense = xx.color.deltaE_cie76(a.copy(), b.copy())
    actual = xx.color.deltaE_cie76(a, b)
    np.testing.assert_array_equal(actual.view(np.uint32), dense.view(np.uint32))
    np.testing.assert_array_equal(actual, reference.deltaE_cie76(a, b))
    assert a.tobytes() == before_a and b.tobytes() == before_b


def test_cie76_row_strides_and_reversed_pixels_keep_dense_words():
    first = np.arange(5 * 34 * 3, dtype=np.float32).reshape(5, 34, 3)[:, ::2]
    second = np.arange(5 * 17 * 3, dtype=np.float32).reshape(5, 17, 3)[:, ::-1]
    actual = xx.color.deltaE_cie76(first, second)
    dense = xx.color.deltaE_cie76(first.copy(), second.copy())
    np.testing.assert_array_equal(actual.view(np.uint32), dense.view(np.uint32))
    np.testing.assert_array_equal(actual, reference.deltaE_cie76(first, second))
