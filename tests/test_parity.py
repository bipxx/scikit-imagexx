"""The supported compatibility contract, checked against scikit-image 0.26."""
import concurrent.futures
import subprocess
import sys
import numpy as np
import pytest
from skimage import color, filters
from skimagexx import _core as xx


@pytest.fixture(autouse=True, params=[1, 4])
def workers(request):
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1)


def image(shape, dtype, seed=42):
    rng = np.random.default_rng(seed)
    if dtype == np.uint8:
        return rng.integers(0, 256, shape, dtype=dtype)
    return rng.uniform(-1, 2, shape).astype(dtype)


def close(actual, expected):
    assert actual.dtype == expected.dtype
    tol = 2e-6 if expected.dtype == np.float32 else 2e-13
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(1, 1), (1, 19), (17, 1), (23, 31), (513, 515)])
def test_rgb(dtype, shape):
    a = image((*shape, 3), dtype)
    close(xx.rgb2gray(a), color.rgb2gray(a))


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(1, 1), (1, 19), (17, 1), (23, 31), (513, 515)])
def test_sobel(dtype, shape):
    a = image(shape, dtype)
    close(xx.sobel(a), filters.sobel(a))


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(1, 1), (1, 19), (17, 1), (23, 31), (513, 515)])
@pytest.mark.parametrize("mode", ["nearest", "reflect"])
@pytest.mark.parametrize("sigma,truncate", [(0, 4), (0.2, 4), (1.3, 4), (7, 2), (1, 0)])
def test_gaussian(dtype, shape, mode, sigma, truncate):
    a = image(shape, dtype)
    close(xx.gaussian(a, sigma, mode=mode, truncate=truncate),
          filters.gaussian(a, sigma, mode=mode, truncate=truncate))


def test_preserve_range():
    a = image((97, 101), np.uint8)
    close(xx.gaussian(a, preserve_range=True), filters.gaussian(a, preserve_range=True))


@pytest.mark.parametrize("shape", [(1, 1), (1, 19), (17, 1), (23, 31), (513, 515)])
def test_otsu(shape):
    for seed in range(15):
        a = image(shape, np.uint8, seed)
        assert xx.threshold_otsu(a) == filters.threshold_otsu(a)


def test_otsu_ties_and_constant():
    for value in [0, 42, 255]:
        a = np.full((520, 520), value, np.uint8)
        assert xx.threshold_otsu(a) == filters.threshold_otsu(a)
    for values in [[0, 255], [20, 200], [0, 1, 2], [0, 0, 1, 2, 2], list(range(256))]:
        a = np.tile(np.array(values, dtype=np.uint8), (1025, 3))
        assert xx.threshold_otsu(a) == filters.threshold_otsu(a)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
def test_strides_and_input_unchanged(dtype):
    a = image((41, 51), dtype).T[::-1, ::2]
    original = a.copy()
    for native, reference in [(xx.gaussian, filters.gaussian), (xx.sobel, filters.sobel)]:
        close(native(a), reference(a))
    if dtype == np.uint8:
        assert xx.threshold_otsu(a) == filters.threshold_otsu(a)
    rgb = image((41, 51, 3), dtype)[::-1, ::2, ::-1]
    close(xx.rgb2gray(rgb), color.rgb2gray(rgb))
    np.testing.assert_array_equal(a, original)


def test_patterns():
    a = np.zeros((33, 35), np.float64)
    a[16, 17] = 1
    for a in [a, np.ones_like(a), np.indices(a.shape).sum(axis=0) % 2.0]:
        close(xx.gaussian(a), filters.gaussian(a))
        close(xx.sobel(a), filters.sobel(a))


def test_nonfinite_pixels():
    a = np.zeros((9, 11), np.float64)
    a[4, 4] = np.nan
    a[6, 6] = np.inf
    with np.errstate(invalid="ignore"):
        close(xx.gaussian(a), filters.gaussian(a))
        close(xx.sobel(a), filters.sobel(a))


@pytest.mark.parametrize("fn", [xx.gaussian, xx.sobel, xx.threshold_otsu])
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (3,), (2, 3, 4)])
def test_bad_shape(fn, shape):
    with pytest.raises(ValueError):
        fn(np.zeros(shape, np.uint8))


@pytest.mark.parametrize("fn", [xx.gaussian, xx.sobel, xx.rgb2gray])
@pytest.mark.parametrize("dtype", [np.int16, np.float16, np.bool_, np.complex64])
def test_bad_dtype(fn, dtype):
    shape = (3, 4, 3) if fn == xx.rgb2gray else (3, 4)
    with pytest.raises(TypeError):
        fn(np.zeros(shape, dtype))


@pytest.mark.parametrize("sigma", [-1, np.nan, np.inf, 1e100])
def test_bad_sigma(sigma):
    with pytest.raises(ValueError):
        xx.gaussian(np.zeros((3, 4)), sigma)


def test_bad_options():
    a = np.zeros((3, 4))
    for truncate in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            xx.gaussian(a, truncate=truncate)
    with pytest.raises(ValueError):
        xx.gaussian(a, mode="wrap")
    with pytest.raises(TypeError):
        xx.threshold_otsu(a)
    with pytest.raises(ValueError):
        xx.rgb2gray(np.zeros((3, 4, 4)))
    with pytest.raises(ValueError):
        xx.set_num_threads(0)


def test_concurrent_calls():
    a = image((519, 521), np.float32)
    expected = filters.gaussian(a)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        for result in pool.map(xx.gaussian, [a]*6):
            close(result, expected)


def test_default_is_serial():
    # A fresh interpreter must not inherit the fixture's explicit worker setting.
    subprocess.run([sys.executable, "-c",
                    "import skimagexx as xx; assert xx.get_num_threads() == 1"], check=True)


def test_worker_limit_is_global_and_can_disable_parallelism():
    a = image((519, 521), np.float32)
    a8 = image(a.shape, np.uint8)
    rgb = image((*a.shape, 3), np.uint8)
    functions = [lambda: xx.rgb2gray(rgb), lambda: xx.gaussian(a),
                 lambda: xx.sobel(a), lambda: xx.threshold_otsu(a8)]
    xx.set_num_threads(1)
    serial = [fn() for fn in functions]
    xx.set_num_threads(4)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        assert pool.submit(xx.get_num_threads).result() == 4
        for fn, expected in zip(functions, serial):
            np.testing.assert_array_equal(pool.submit(fn).result(), expected)
        # Updating from a different caller thread updates the same global limit.
        pool.submit(xx.set_num_threads, 1).result()
    assert xx.get_num_threads() == 1
    for fn, expected in zip(functions, serial):
        np.testing.assert_array_equal(fn(), expected)


@pytest.mark.parametrize("width", list(range(1, 34)))
def test_simd_tails_and_border_overlap(width):
    a = image((3, width), np.uint8, width)
    close(xx.sobel(a), filters.sobel(a))
    close(xx.gaussian(a, 3), filters.gaussian(a, 3))
    rgb = image((3, width, 3), np.float32, width)
    close(xx.rgb2gray(rgb), color.rgb2gray(rgb))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unaligned_input(dtype):
    shape = (5, 17)
    raw = np.zeros(np.prod(shape)*np.dtype(dtype).itemsize+1, np.uint8)
    a = np.ndarray(shape, dtype=dtype, buffer=raw, offset=1)
    a[:] = image(shape, dtype)
    assert not a.flags.aligned
    close(xx.sobel(a), filters.sobel(a))
    close(xx.gaussian(a), filters.gaussian(a))


def test_otsu_late_outlier_and_skew():
    a = np.full((519, 521), 99, np.uint8)
    a[-1, -1] = 255
    assert xx.threshold_otsu(a) == filters.threshold_otsu(a)
    a.flat[::997] = 0
    assert xx.threshold_otsu(a) == filters.threshold_otsu(a)
