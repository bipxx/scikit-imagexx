import numpy as np
import pytest
import skimage
from skimage.metrics import structural_similarity as original

import skimagexx as xx
from skimagexx import metrics

from structural_similarity_reference import ATOL, cases


@pytest.fixture
def strict_native():
    saved_fallback = xx.set_fallback_enabled(False)
    saved_workers = xx.get_num_threads()
    xx.set_num_threads(1)
    try:
        yield
    finally:
        xx.set_num_threads(saved_workers)
        xx.set_fallback_enabled(saved_fallback)


@pytest.mark.parametrize(("left", "right", "options"),
                         [pytest.param(left, right, options, id=label)
                          for label, left, right, options in cases()])
def test_uniform_score(left, right, options, strict_native):
    assert skimage.__version__ == "0.26.0"
    left_bytes, right_bytes = left.tobytes(), right.tobytes()
    expected = original(left, right, **options)
    actual = metrics.structural_similarity(left, right, **options)
    assert type(actual) is type(expected) is np.float64
    assert actual == pytest.approx(expected, abs=ATOL[left.dtype.name], rel=0)
    assert left.tobytes() == left_bytes and right.tobytes() == right_bytes


def test_default_options_aliasing_and_immutable_view(strict_native):
    values = np.arange(63, dtype=np.float32).reshape(7, 9) / 64
    view = np.ndarray(values.shape, dtype=values.dtype, buffer=values.tobytes())
    assert not view.flags.writeable
    actual = metrics.structural_similarity(view, view, data_range=1)
    assert type(actual) is np.float64
    assert actual == original(view, view, data_range=1) == 1.0


def test_workers_preserve_large_score(strict_native):
    if not xx._core.has_openmp:
        pytest.skip("OpenMP workers unavailable in this build")
    _, left, right, options = next(case for case in cases() if case[0] == "float32-cutover")
    expected = metrics.structural_similarity(left, right, **options)
    xx.set_num_threads(2)
    actual = metrics.structural_similarity(left, right, **options)
    assert type(actual) is np.float64 and actual == expected


@pytest.mark.parametrize("mode", ("full", "gaussian", "channels", "layout", "integer", "mixed",
                                  "owner", "exceptional", "wide", "range", "option", "short",
                                  "missing-range"))
def test_other_calls_follow_compatibility_policy(mode):
    a = np.zeros((15, 15))
    b = np.full_like(a, 0.25)
    options = {"data_range": 1.0}
    if mode == "full":
        options["full"] = True
    elif mode == "gaussian":
        options["gaussian_weights"] = True
    elif mode == "channels":
        a, b = a[..., None], b[..., None]
        options["channel_axis"] = -1
    elif mode == "layout":
        a, b = a.T, b.T
    elif mode == "integer":
        a, b = a.astype("uint8"), b.astype("uint8")
    elif mode == "mixed":
        a = a.astype("float32")
    elif mode == "owner":
        a = np.frombuffer(bytearray(a.tobytes()), dtype=a.dtype).reshape(a.shape)
    elif mode == "exceptional":
        a[0, 0] = np.nan
    elif mode == "wide":
        a[0, 0] = 1.5
    elif mode == "range":
        options["data_range"] = 5.0
    elif mode == "option":
        options["K1"] = 0.025
    elif mode == "short":
        a, b = a[:2], b[:2]
    else:
        options.clear()
    saved = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            metrics.structural_similarity(a, b, **options)
        xx.set_fallback_enabled(True)
        with pytest.warns(xx.FallbackWarning, match=r"skimage\.metrics\.structural_similarity"):
            if mode in ("short", "missing-range"):
                with pytest.raises(ValueError) as expected:
                    original(a, b, **options)
                with pytest.raises(ValueError) as actual:
                    metrics.structural_similarity(a, b, **options)
                assert str(actual.value) == str(expected.value)
            else:
                actual = metrics.structural_similarity(a, b, **options)
                expected = original(a, b, **options)
                if mode == "full":
                    assert actual[0] == expected[0]
                    np.testing.assert_array_equal(actual[1], expected[1])
                else:
                    np.testing.assert_equal(actual, expected)
    finally:
        xx.set_fallback_enabled(saved)
