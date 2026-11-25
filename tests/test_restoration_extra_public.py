"""Live scikit-image 0.26 public contracts for the restoration leaf."""

import inspect

import numpy as np
import pytest
from skimage import restoration as original
from skimage import __version__ as reference_version
import skimagexx as xx
from skimagexx import restoration as native
from skimagexx._compat import FallbackWarning


@pytest.fixture(autouse=True)
def strict_native():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)


def compare(actual, expected, **tolerance):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.dtype.metadata == expected.dtype.metadata
    assert actual.shape == expected.shape
    assert actual.strides == expected.strides
    assert actual.flags.owndata == expected.flags.owndata
    assert actual.flags.writeable == expected.flags.writeable
    np.testing.assert_allclose(actual, expected, **tolerance)


@pytest.mark.parametrize('name', (
    'wiener', 'unsupervised_wiener', 'denoise_invariant', 'calibrate_denoiser', 'cycle_spin'
))
def test_signature(name):
    assert reference_version == '0.26.0'
    assert inspect.signature(getattr(native, name)) == inspect.signature(getattr(original, name))


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int16, 'longlong', '>f4'])
@pytest.mark.parametrize('shape', [(9, 13), (6, 9, 10)])
def test_wiener_array_and_dtype(dtype, shape):
    source = np.random.default_rng(3).normal(size=shape)
    image = (source * 12).astype(np.dtype(dtype, metadata={'source': 'tag'}))[..., ::-1]
    psf = np.ones((3,) * len(shape)) / 3**len(shape)
    compare(native.wiener(image, psf, 0.16), original.wiener(image, psf, 0.16),
            rtol=5e-5, atol=3e-6)


@pytest.mark.parametrize('transfer', [False, True])
def test_wiener_supplied_regularizer(transfer):
    from skimage.restoration.uft import ir2tf

    image = np.arange(13 * 14, dtype=np.float64).reshape(13, 14) / 128
    psf = np.array([[0.3, 0.1], [0.1, 0.5]])
    reg = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    if transfer:
        psf, reg = (ir2tf(value, image.shape) for value in (psf, reg))
    compare(native.wiener(image, psf, 0.5, reg=reg, clip=False),
            original.wiener(image, psf, 0.5, reg=reg, clip=False), atol=1e-7)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(17, 19), (11, 20)])
def test_gibbs_chains_and_generator_state(dtype, shape):
    image = np.random.default_rng(4).random(shape).astype(dtype)
    kernel = np.ones((3, 3), dtype=dtype) / 9
    params = {'max_num_iter': 12, 'burnin': 3, 'min_num_iter': 5, 'threshold': 0}
    rng = np.random.default_rng(25)
    reference_rng = np.random.default_rng(25)
    actual, chains = native.unsupervised_wiener(image, kernel, user_params=params, rng=rng)
    expected, reference = original.unsupervised_wiener(
        image, kernel, user_params=params, rng=reference_rng
    )
    compare(actual, expected, rtol=4e-5, atol=2e-6)
    assert tuple(chains) == tuple(reference)
    for key in chains:
        assert len(chains[key]) == params['max_num_iter'] + 1
        np.testing.assert_allclose(chains[key], reference[key], rtol=4e-6)
    assert rng.bit_generator.state == reference_rng.bit_generator.state


def _scale(image, *, weight=1, channel_axis=None):
    return image * weight


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.uint8])
@pytest.mark.parametrize('channel_axis,shape', [(None, (3, 1, 11)), (-1, (7, 11, 3))])
def test_j_invariant_masks_dtype_and_callback(dtype, channel_axis, shape):
    image = np.asfortranarray(np.arange(np.prod(shape)).reshape(shape)).astype(
        np.dtype(dtype, metadata={'payload': 9})
    )
    kwargs = dict(weight=0.7, channel_axis=channel_axis)
    snapshots = []

    def callback(value, **options):
        snapshots.append(value.copy())
        return _scale(value, **options)

    spatial_shape = shape if channel_axis is None else shape[:-1]
    masks = [np.indices(spatial_shape).sum(axis=0) % 3 == 0,
             (slice(1, None, 2),) * len(spatial_shape)]
    actual = native.denoise_invariant(image, callback, masks=iter(masks), denoiser_kwargs=kwargs)
    native_snapshots = snapshots[:]
    snapshots.clear()
    expected = original.denoise_invariant(image, callback, masks=masks, denoiser_kwargs=kwargs)
    compare(actual, expected, atol=0, rtol=0)
    assert len(native_snapshots) == len(snapshots) == len(masks)
    for a, b in zip(native_snapshots, snapshots):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('approximate', [True, False])
@pytest.mark.parametrize('dtype', [np.uint8, np.float16, np.float32])
def test_calibration_search_and_returned_partial(dtype, approximate):
    image = np.arange(147, dtype=np.float64).reshape(7, 21).astype(dtype)
    params = {'weight': [0.3, 0.5, 0.7, 0.9], 'channel_axis': [None]}
    args = dict(stride=2, approximate_loss=approximate, extra_output=True)
    actual, (settings, losses) = native.calibrate_denoiser(image, _scale, params, **args)
    expected, (ref_settings, ref_losses) = original.calibrate_denoiser(image, _scale, params, **args)
    assert settings == ref_settings
    assert actual.keywords == expected.keywords
    np.testing.assert_allclose(losses, ref_losses, atol=0, rtol=0)
    compare(actual(image), expected(image), atol=0, rtol=0)


@pytest.mark.parametrize('channel_axis,shape', [(None, (13, 17)), (-1, (9, 14, 3)), (0, (3, 9, 14))])
@pytest.mark.parametrize('tagged', [False, True])
def test_cycle_spin_order_shift_steps_and_layout(channel_axis, shape, tagged):
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if tagged:
        image = image.view(np.dtype('f4', metadata={'source': 'test'}))
    calls = []

    def callback(x, offset):
        calls.append(x[0, 0].copy())
        x[0] += offset
        return x

    options = dict(max_shifts=(4, 2), shift_steps=(2, 1), workers=1, channel_axis=channel_axis,
                   func_kw={'offset': 3})
    actual = native.cycle_spin(image, callback, **options)
    native_calls = calls[:]
    calls.clear()
    expected = original.cycle_spin(image, callback, **options)
    compare(actual, expected, atol=0, rtol=0)
    np.testing.assert_array_equal(native_calls, calls)
    assert len(calls) == 9


def test_cycle_numpy_integer_error_and_validation():
    image = np.arange(15).reshape(3, 5)
    for function in (original.cycle_spin, native.cycle_spin):
        with pytest.raises(TypeError, match='Cannot cast'):
            function(image, lambda value: value, 1, workers=1)
        with pytest.raises(ValueError, match='shift_steps must all be'):
            function(image, lambda value: value, 1, shift_steps=0, workers=1)


def test_cycle_deprecated_worker_alias():
    image = np.arange(15, dtype=np.float64).reshape(3, 5)
    results = []
    messages = []
    for function in (original.cycle_spin, native.cycle_spin):
        with pytest.warns(FutureWarning) as recorded:
            results.append(function(image, _scale, 1, 1, 1, {'weight': 0.5}))
        messages.append(str(recorded[0].message))
        with pytest.warns(FutureWarning), pytest.raises(ValueError, match='conflicting values'):
            function(image, _scale, 1, num_workers=1, workers=None)
    compare(*results, atol=0, rtol=0)
    assert messages[0] == messages[1]


def test_global_fft_worker_snapshot():
    image = np.random.default_rng(8).normal(size=(128, 155))
    psf = np.full((5, 5), 0.04)
    original_workers = xx.get_num_threads()
    assert original_workers == 1
    try:
        try:
            xx.set_num_threads(2)
        except ValueError as error:
            # Minimum serial builds verify their existing worker rejection.
            assert 'no OpenMP support' in str(error)
        np.testing.assert_allclose(native.wiener(image, psf, 0.3),
                                   original.wiener(image, psf, 0.3), atol=2e-12)
    finally:
        xx.set_num_threads(original_workers)


def test_fallback_policy_and_complex_wiener_output():
    image = np.arange(80, dtype=np.float64).reshape(8, 10) / 80
    psf = np.ones((3, 3)) / 9
    with pytest.raises(NotImplementedError, match='restoration.wiener'):
        native.wiener(image, psf, 0.2, is_real=False)
    xx.set_fallback_enabled(True)
    with pytest.warns(FallbackWarning, match='restoration.wiener'):
        actual = native.wiener(image, psf, 0.2, is_real=False)
    compare(actual, original.wiener(image, psf, 0.2, is_real=False), atol=0, rtol=0)
