"""Parity and ingress checks for the native morphological snakes family."""

import inspect
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
from skimage.segmentation import morphsnakes as stock


BENCHMARK = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'benchmarks/morphological_snakes.py'))
CASES = tuple(BENCHMARK['case_specs']())


@pytest.fixture
def strict():
    import skimagexx

    previous = skimagexx.set_fallback_enabled(False)
    try:
        yield skimagexx
    finally:
        skimagexx.set_fallback_enabled(previous)


@pytest.fixture
def reset_phase(monkeypatch):
    # Take one full turn of the actual stock iterator, then preserve this
    # process's original phase on test exit.
    first, second = next(stock._curvop.funcs), next(stock._curvop.funcs)
    ordered = (first, second) if first.__code__.co_names[0] == 'sup_inf' else (second, first)

    def reset(reverse=False):
        stages = stock._fcycle(ordered[::-1] if reverse else ordered)
        monkeypatch.setattr(stock, '_curvop', stages)

    return reset


def assert_result(actual, expected, source):
    assert type(actual) is np.ndarray and actual.shape == expected.shape
    assert actual.dtype == expected.dtype and actual.strides == expected.strides
    assert actual.flags.writeable and actual.flags.owndata
    assert not np.shares_memory(actual, source)
    if actual.dtype.kind == 'f':
        rtol = 2e-6 if actual.dtype.itemsize == 4 else 5e-13
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0)
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('spec', CASES, ids=[case['id'] for case in CASES])
def test_fixed_public_cases(strict, reset_phase, spec):
    name = spec['api'].rsplit('.', 1)[-1]
    args, kwargs = BENCHMARK['make_case'](spec)
    original = getattr(stock, name)
    native = getattr(strict.segmentation, name)
    compiled = getattr(strict._core, name)
    assert inspect.signature(native) == inspect.signature(original)
    assert inspect.isbuiltin(compiled)
    reset_phase()
    expected = original(*args, **kwargs)
    before = args[0].copy()
    calls = []

    def profile(frame, event, target):
        if event == 'c_return' and target is compiled:
            calls.append(target)
        if event == 'call' and frame.f_code is original.__code__:
            raise AssertionError('unexpected original algorithm call in strict mode')

    reset_phase()
    prior = sys.getprofile()
    try:
        sys.setprofile(profile)
        actual = native(*args, **kwargs)
    finally:
        sys.setprofile(prior)
    assert calls == [compiled]
    assert_result(actual, expected, args[0])
    np.testing.assert_array_equal(args[0], before)


@pytest.mark.parametrize('shape', [(2, 2), (7, 14), (3, 8, 13)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('reverse', [False, True])
def test_seeded_contour_evolution(strict, reset_phase, shape, dtype, reverse):
    rng = np.random.default_rng(731)
    image = rng.uniform(-1.2, 1.0, size=shape).astype(dtype)
    initial = rng.integers(-2, 6, size=shape, dtype=np.int16)
    saved = initial.copy()
    calls = (
        ('morphological_chan_vese', dict(num_iter=3, smoothing=1, lambda1=0.7, lambda2=1.25)),
        ('morphological_geodesic_active_contour', dict(num_iter=3, smoothing=1, balloon=1)),
        ('morphological_geodesic_active_contour', dict(num_iter=2, smoothing=2, threshold=-0.2, balloon=-0.7)),
        ('morphological_chan_vese', dict(num_iter=1, smoothing=0)),
        ('morphological_chan_vese', dict(num_iter=0, smoothing=2)),
    )

    reset_phase(reverse)
    expected = [getattr(stock, name)(image, init_level_set=initial, **options) for name, options in calls]
    reset_phase(reverse)
    for (name, options), wanted in zip(calls, expected):
        actual = getattr(strict.segmentation, name)(image, init_level_set=initial, **options)
        assert_result(actual, wanted, image)
    np.testing.assert_array_equal(initial, saved)


@pytest.mark.parametrize('shape', [(1, 9), (6, 11), (2, 5, 9)])
@pytest.mark.parametrize('sigma', [0.5, 3.3, 16])
def test_inverse_gradient_boundaries(strict, shape, sigma):
    image = np.random.default_rng(179).uniform(0.02, 1.75, size=shape)
    expected = stock.inverse_gaussian_gradient(image, alpha=181, sigma=sigma)
    actual = strict.segmentation.inverse_gaussian_gradient(image, alpha=181, sigma=sigma)
    assert_result(actual, expected, image)


def test_worker_limit_parity(strict, reset_phase):
    image = np.random.default_rng(514).uniform(0.02, 1.75, size=(512, 513)).astype(np.float32)
    public = strict.segmentation
    before = strict.get_num_threads()

    def run(workers):
        strict.set_num_threads(workers)
        assert strict.get_num_threads() == workers
        reset_phase(True)
        return (
            public.inverse_gaussian_gradient(image, sigma=1),
            public.morphological_chan_vese(image, 1),
            public.morphological_geodesic_active_contour(image, 1, threshold=0.8, balloon=1),
        )

    try:
        serial, parallel = run(1), run(4)
    finally:
        strict.set_num_threads(before)
    for actual, expected in zip(parallel, serial):
        np.testing.assert_array_equal(actual, expected)


def test_disabled_fallback_and_descriptor_contract(strict):
    class Subclass(np.ndarray):
        pass

    native = strict.segmentation
    image = np.zeros((8, 9))
    readonly = image.view()
    readonly.flags.writeable = False
    for bad in (
        image.view(Subclass), image.view(np.dtype('float64', metadata={'tag': 1})),
        image.astype(image.dtype.newbyteorder('S')), image[:, ::-1], readonly,
        image.astype(np.longdouble), image.astype(np.uint8), np.full_like(image, np.nan),
        np.full_like(image, 1e200),
    ):
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            native.inverse_gaussian_gradient(bad)
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            native.morphological_chan_vese(bad, 1)
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            native.morphological_geodesic_active_contour(bad, 1)
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        native.morphological_chan_vese(image, 1, iter_callback=lambda u: u.fill(0))
    with pytest.raises(ValueError, match='ordinary writable'):
        strict._core.inverse_gaussian_gradient(readonly, 100, 1)
    with pytest.raises(TypeError, match='float32 or float64'):
        strict._core.inverse_gaussian_gradient(image.astype(np.longdouble), 100, 1)


def test_mixed_native_and_callback_fallback_share_curvature(strict, reset_phase):
    image = np.random.default_rng(188).uniform(0.02, 1.75, size=(22, 23))
    native = strict.segmentation
    expected_observations = []
    actual_observations = []

    def run(surface, history, expect_fallback=False):
        first = surface.morphological_chan_vese(image, 1)

        def observe(u):
            history.append(u.copy())
            u[3, 4] = 1

        if expect_fallback:
            with pytest.warns(strict.FallbackWarning, match='morphological_geodesic_active_contour'):
                second = surface.morphological_geodesic_active_contour(image, 3, iter_callback=observe)
        else:
            second = surface.morphological_geodesic_active_contour(image, 3, iter_callback=observe)
        third = surface.morphological_chan_vese(image, 2, lambda2=0.7)
        return first, second, third

    reset_phase(True)
    expected = run(stock, expected_observations)
    reset_phase(True)
    strict.set_fallback_enabled(True)
    actual = run(native, actual_observations, True)
    for out, ref in zip(actual + tuple(actual_observations), expected + tuple(expected_observations)):
        assert_result(out, ref, image)
