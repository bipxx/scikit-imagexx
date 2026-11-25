import inspect
import warnings

import numpy as np
import pytest
from skimage import transform as original
import skimagexx as xx
from skimagexx import _core, _warp
from skimagexx._compat import FallbackWarning
from skimagexx._geometric import AffineTransform


MODES = ('constant', 'edge', 'symmetric', 'reflect', 'wrap')


@pytest.fixture(autouse=True)
def strict():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)


def test_public_signatures():
    for name in ('warp', 'warp_coords'):
        current = getattr(xx.transform, name)
        assert current is getattr(_warp, name)
        assert inspect.signature(current) == inspect.signature(getattr(original, name))


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('order', range(4))
@pytest.mark.parametrize('dtype', (np.float32, np.float64, np.uint8))
def test_warp_projective_matches_stock(mode, order, dtype):
    source = np.random.default_rng(20).integers(6, 250, size=(15, 22), dtype=np.uint8).astype(dtype)
    matrix = np.array([[.95, -.08, -.7], [.15, 1.1, -.8], [.006, -.008, 1.]])
    kwargs = dict(output_shape=(19, 25), mode=mode, order=order, cval=3.4, clip=True,
                  preserve_range=False)
    with warnings.catch_warnings(record=True) as expected:
        warnings.simplefilter('always')
        reference = original.warp(source, matrix, **kwargs)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        result = xx.transform.warp(source, matrix, **kwargs)
    assert [(w.category, str(w.message)) for w in observed] == [
        (w.category, str(w.message)) for w in expected
    ]
    assert result.dtype == reference.dtype
    np.testing.assert_allclose(result, reference, rtol=2e-6, atol=3e-5)


@pytest.mark.parametrize('dtype', (np.float32, np.float64, 'f8'))
@pytest.mark.parametrize('shape', ((19, 25), (19, 25, 3)))
def test_warp_coords_color_and_inverse(dtype, shape):
    source_map = original.ProjectiveTransform(matrix=np.array([
        [1.02, .12, 8.], [.06, 1.02, -3.], [.0003, -.0006, 1.],
    ]))
    expected = original.warp_coords(source_map.inverse, shape, dtype)
    result = xx.transform.warp_coords(source_map.inverse, shape, dtype)
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-12)


def test_owned_transform_and_multichannel():
    image = np.random.default_rng(6).random((31, 39, 3)).astype(np.float32)
    matrix = np.array([[.95, -.08, -.7], [.15, 1.1, -.8], [0., 0., 1.]])
    model = AffineTransform(matrix=matrix)
    expected = original.warp(image, original.AffineTransform(matrix=matrix), order=3,
                             output_shape=(27, 42), mode='reflect')
    np.testing.assert_allclose(xx.transform.warp(image, model, order=3,
                              output_shape=(27, 42), mode='reflect'), expected, atol=8e-7)


def test_nearest_affine_preserves_stock_half_pixel_ties():
    image = np.arange(53 * 576, dtype=np.float64).reshape(53, 576)
    matrix = np.array([[1.01, -.04, 3.2], [.06, .95, -7.3], [0., 0., 1.]])
    expected = original.warp(image, matrix, order=0, mode='edge')
    np.testing.assert_array_equal(xx.transform.warp(image, matrix, order=0, mode='edge'),
                                  expected)


@pytest.mark.parametrize('constant', (3.4, np.float64(3.4)))
def test_float32_clip_respects_cval_scalar_type(constant):
    image = np.arange(6, 22, dtype=np.float32).reshape(4, 4)
    matrix = np.array([[1., 0., -2.], [0., 1., 0.], [0., 0., 1.]])
    kwargs = dict(cval=constant, output_shape=(4, 6))
    np.testing.assert_array_equal(xx.transform.warp(image, matrix, **kwargs),
                                  original.warp(image, matrix, **kwargs))


def test_native_dtype_alias_view_tag_and_subclass():
    matrix = np.eye(3)
    image = np.arange(60., dtype=np.dtype('=f8')).reshape(6, 10)
    np.testing.assert_array_equal(xx.transform.warp(image, matrix), image)
    tagged = image.view(np.dtype('f8', metadata={'units': 'arb'}))
    long = image.astype(np.longdouble)
    subclass = image.view(type('Subarray', (np.ndarray,), {}))
    params = tuple(matrix.flat)
    for outside in (tagged, long, subclass):
        assert _core.warp_sample(outside, (6, 10), 0, params, 1, 1, 0., True, False, False) is None
        with pytest.raises(NotImplementedError, match='outside native coverage'):
            xx.transform.warp(outside, matrix)


def test_read_only_noncontiguous_and_arbitrary_maps_fall_back():
    image = np.arange(100., dtype=np.float64).reshape(10, 10)
    readonly = image.copy()
    readonly.flags.writeable = False

    def shift(coords):
        return coords + [.25, 2.25]

    for array, transform in ((readonly, np.eye(3)), (image[:, ::2], np.eye(3)), (image, shift)):
        with pytest.raises(NotImplementedError, match='outside native coverage'):
            xx.transform.warp(array, transform)
    with pytest.raises(NotImplementedError, match='outside native coverage'):
        xx.transform.warp_coords(shift, (6, 5))
    xx.set_fallback_enabled(True)
    with pytest.warns(FallbackWarning):
        result = xx.transform.warp(image, shift)
    np.testing.assert_array_equal(result, original.warp(image, shift))


def test_owned_homography_fallback_uses_stock_matrix_branch():
    image = np.arange(100., dtype=np.float64).reshape(10, 10)[:, ::2]
    params = np.array([[1., 0., .25], [0., 1., .4], [0., 0., 1.]])
    model = AffineTransform(matrix=params)
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        xx.transform.warp(image, model, order=3)
    xx.set_fallback_enabled(True)
    with pytest.warns(FallbackWarning):
        result = xx.transform.warp(image, model, order=3)
    np.testing.assert_array_equal(
        result, original.warp(image, original.AffineTransform(matrix=params), order=3))


def test_workers_have_identical_results():
    image = np.random.default_rng(3).random((513, 531))
    matrix = np.array([[1., -.04, -.7], [.03, 1., -.8], [.0002, .0001, 1.]])
    previous = xx.get_num_threads()
    try:
        xx.set_num_threads(1)
        serial = xx.transform.warp(image, matrix, order=3, mode='reflect')
        if _core.has_openmp:
            xx.set_num_threads(4)
        np.testing.assert_array_equal(
            xx.transform.warp(image, matrix, order=3, mode='reflect'), serial)
    finally:
        xx.set_num_threads(previous)
