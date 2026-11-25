"""Compare the strict active contour wrapper and installed stock implementation."""

import inspect
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = runpy.run_path(str(ROOT / 'benchmarks' / 'active_contour.py'))
CASES = list(BENCHMARK['case_specs']())


@pytest.fixture
def api():
    import skimage
    from skimage import segmentation as stock
    from skimagexx import _compat, _core, segmentation
    from skimagexx._active_contour import active_contour

    assert skimage.__version__ == '0.26.0'
    assert segmentation.active_contour is active_contour
    assert inspect.signature(active_contour) == inspect.signature(stock.active_contour)
    entry = _core.active_contour_finite
    assert inspect.isbuiltin(entry)
    previous = _compat.set_fallback_enabled(False)
    try:
        yield active_contour, stock.active_contour, entry, _compat
    finally:
        _compat.set_fallback_enabled(previous)


def compare(api, image, snake, options):
    public, original, entry, _ = api
    expected = original(image, snake, **options)
    snapshots = image.tobytes(), snake.tobytes()
    calls = []

    def observe(frame, event, value):
        if event == 'c_return' and value is entry:
            calls.append(value)
        if event == 'call' and frame.f_code is original.__code__:
            raise AssertionError('strict candidate called the original')

    prior = sys.getprofile()
    try:
        sys.setprofile(observe)
        observed = public(image, snake, **options)
    finally:
        sys.setprofile(prior)

    assert calls == [entry]
    assert type(observed) is np.ndarray and observed.dtype == expected.dtype
    assert observed.shape == expected.shape == snake.shape
    assert observed.flags.owndata and observed.flags.c_contiguous and observed.flags.writeable
    assert observed.base is None and not np.shares_memory(observed, snake)
    assert snapshots == (image.tobytes(), snake.tobytes())
    tolerance = 2e-3 if observed.dtype == np.float32 else 1e-8
    np.testing.assert_allclose(observed, expected, atol=tolerance, rtol=0)
    boundary = options.get('boundary_condition', 'periodic')
    if boundary.startswith('fixed'):
        np.testing.assert_array_equal(observed[0], snake[0].astype(observed.dtype))
    if boundary.endswith('fixed'):
        np.testing.assert_array_equal(observed[-1], snake[-1].astype(observed.dtype))
    return observed


@pytest.mark.parametrize('spec', CASES, ids=[spec['id'] for spec in CASES])
def test_fixed_benchmark_cases_match_stock(api, spec):
    args, options = BENCHMARK['make_case'](spec)
    compare(api, *args, options)


@pytest.mark.parametrize('boundary', [
    'periodic', 'fixed', 'free', 'fixed-free', 'free-fixed', 'fixed-fixed', 'free-free',
])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.uint8])
def test_fractional_and_outside_coordinates(api, boundary, dtype):
    y, x = np.ogrid[:7, :10]
    image = (10 + 2 * x + 3 * y + x * y + y**2).astype(dtype)
    if dtype != np.uint8:
        image /= 255
    snake = np.array([[-0.9, 0.2], [0.7, 2.3], [1.5, 1.5], [3.2, 4.9],
                      [4.7, 6.1], [6.0, 9.0], [5.9, 9.8]])
    compare(api, image, snake, dict(boundary_condition=boundary, w_line=0.4,
                                    w_edge=-0.2, max_num_iter=15))


def test_convergence_and_process_worker_limit(api):
    from skimagexx import get_num_threads, set_num_threads

    assert get_num_threads() == 1
    spec = next(spec for spec in CASES if spec['id'] == 'rgb-byte-fixed-free')
    # Sobel enables the optional worker pool at 256 Ki pixels per plane.
    args, options = BENCHMARK['make_case']({**spec, 'shape': (512, 520)})
    options.update(convergence=0.1, max_px_move=0.001, max_num_iter=45)
    serial = compare(api, *args, options)
    # A full history window lets the result converge on iteration 11.
    np.testing.assert_array_equal(serial, api[0](*args, **{**options, 'max_num_iter': 11}))
    try:
        set_num_threads(4)
    except ValueError as error:
        assert str(error) == 'this build has no OpenMP support'
        assert get_num_threads() == 1
        return
    try:
        assert get_num_threads() == 4
        np.testing.assert_array_equal(api[0](*args, **options), serial)
    finally:
        set_num_threads(1)


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize('target', ['image', 'snake'])
@pytest.mark.parametrize('convert', [
    lambda a: a.view(ArraySubclass),
    np.asfortranarray,
    lambda a: a[::-1],
    lambda a: a.astype(a.dtype.newbyteorder('S')),
    lambda a: a.view(np.dtype(a.dtype, metadata={'unit': 'pixel'})),
    lambda a: a.astype(np.longdouble),
    lambda a: np.broadcast_to(a, a.shape),
    lambda a: np.lib.stride_tricks.as_strided(a, shape=a.shape, strides=a.strides),
])
def test_exotic_layouts_and_dtypes_use_live_fallback(api, monkeypatch, target, convert):
    from skimage import segmentation as stock

    public, _, _, compat = api
    (image, snake), options = BENCHMARK['make_case'](CASES[0])
    if target == 'image':
        image = convert(image)
    else:
        snake = convert(snake)
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        public(image, snake, **options)

    marker = object()
    seen = []

    def provider(*args, **kwargs):
        seen.append((args, kwargs))
        return marker

    monkeypatch.setattr(stock, 'active_contour', provider)
    prior_enable = compat.set_fallback_enabled(True)
    try:
        with pytest.warns(compat.FallbackWarning, match='skimage.segmentation.active_contour'):
            assert public(image, snake, **options) is marker
        assert len(seen) == 1
        args, kwargs = seen[0]
        assert args[0] is image and args[1] is snake
        assert kwargs == {'boundary_condition': options['boundary_condition']}
        assert isinstance(public(*BENCHMARK['make_case'](CASES[0])[0], **options), np.ndarray)
        assert len(seen) == 1
    finally:
        compat.set_fallback_enabled(prior_enable)


def test_outside_numerical_domain_delegates(api):
    public, original, _, compat = api
    (image, snake), options = BENCHMARK['make_case'](CASES[0])
    for invalid in ({'boundary_condition': 'invalid'}, {'max_num_iter': 0}):
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            public(image, snake, **{**options, **invalid})
        compat.set_fallback_enabled(True)
        try:
            with pytest.raises(ValueError) as expected:
                original(image, snake, **{**options, **invalid})
            with pytest.warns(compat.FallbackWarning), pytest.raises(ValueError) as observed:
                public(image, snake, **{**options, **invalid})
            assert str(observed.value) == str(expected.value)
        finally:
            compat.set_fallback_enabled(False)

    image[0, 0] = np.nan
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        public(image, snake, **options)
