import inspect
import pickle
from unittest.mock import patch

import numpy as np
import pytest
from skimage import transform as stock

import skimagexx as xx
from skimagexx._compat import FallbackWarning


def _controls(dtype=np.float64):
    src = np.array([[0, 0], [4, 0], [0, 4], [4, 4], [2, 2]], dtype=dtype)
    dst = np.array([[2, 1], [6, 1], [2, 5], [6, 5], [4.5, 3.2]], dtype=dtype)
    return src, dst


def test_thin_plate_spline_static_members_and_private_math():
    cls = xx.transform.ThinPlateSplineTransform
    original = stock.ThinPlateSplineTransform
    assert inspect.signature(cls) == inspect.signature(original)
    for name in dir(original):
        actual = inspect.getattr_static(cls, name)
        expected = inspect.getattr_static(original, name)
        assert type(actual) is type(expected), name
        if isinstance(expected, property):
            actual, expected = actual.fget, expected.fget
        elif isinstance(expected, (classmethod, staticmethod)):
            actual, expected = actual.__func__, expected.__func__
        if inspect.isroutine(expected):
            try:
                signature = str(inspect.signature(expected))
            except (TypeError, ValueError) as error:
                with pytest.raises(type(error)):
                    inspect.signature(actual)
            else:
                assert str(inspect.signature(actual)) == signature, name

    src, dst = _controls()
    a, b = cls(), original()
    assert a.src is b.src is None and a._spline_mappings is b._spline_mappings is None
    assert a._estimate(src, dst) is b._estimate(src, dst) is None
    distances = np.array([0, 1e-8, 1, 3])
    np.testing.assert_array_equal(a._radial_basis_kernel(distances), b._radial_basis_kernel(distances))
    actual, expected = a._radial_distance(src), b._radial_distance(src)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(a._spline_function(src, actual), b._spline_function(src, expected), atol=1e-12)


@pytest.mark.parametrize('dtype', [np.int32, np.float32, np.float64])
@pytest.mark.parametrize('workers', [1, 2])
def test_thin_plate_spline_public_hot_swap(dtype, workers):
    cls = xx.transform.ThinPlateSplineTransform
    original = stock.ThinPlateSplineTransform
    assert inspect.isbuiltin(xx._core.thin_plate_spline_fit)
    assert inspect.isbuiltin(xx._core.thin_plate_spline_apply)
    src, dst = _controls(dtype)
    src = np.asarray(src[::-1], order='F')
    dst = dst[::-1]
    query = np.vstack([src, [[-1, -2], [5, 6], [4, 2], [1.5, 1], [2, 3]]])
    if workers == 2:
        query = np.tile(query, (8192, 1))
    ref = original.from_estimate(src, dst)
    before = xx.set_fallback_enabled(False)
    previous_workers = xx.get_num_threads()
    try:
        xx.set_num_threads(workers)
        with patch.object(stock, 'ThinPlateSplineTransform', cls):
            model = stock.ThinPlateSplineTransform.from_estimate(src, dst)
            assert model.src is src
            np.testing.assert_allclose(model._spline_mappings, ref._spline_mappings, atol=1e-12)
            for coords in (query, np.asfortranarray(query[::-1]), query[:0], query[:1].tolist()):
                got, expected = model(coords), ref(coords)
                np.testing.assert_allclose(got, expected, atol=1e-12)
                assert got.dtype == expected.dtype and got.strides == expected.strides
            src[0, 0] += 1
            np.testing.assert_allclose(model(query[:8]), ref(query[:8]), atol=1e-12)
            np.testing.assert_allclose(pickle.loads(pickle.dumps(model))(query[:8]), ref(query[:8]), atol=1e-12)
            with pytest.warns(FutureWarning, match='ThinPlateSplineTransform.from_estimate'):
                assert model.estimate(src, dst)
    finally:
        xx.set_num_threads(previous_workers)
        xx.set_fallback_enabled(before)


def test_thin_plate_spline_validation_and_singular_fallback():
    cls = xx.transform.ThinPlateSplineTransform
    src, dst = _controls()
    with pytest.raises(ValueError, match='Transformation is undefined'):
        cls()(src)
    with pytest.raises(NotImplementedError, match='Not supported'):
        cls().inverse
    with pytest.raises(ValueError, match='Need at least 3'):
        cls.from_estimate(src[:2], dst[:2])
    with pytest.raises(ValueError, match='didn.t match'):
        cls.from_estimate(src, dst[:3])
    with pytest.raises(ValueError, match='Input `coords` must have shape'):
        cls.from_estimate(src, dst)([1, 2])
    with pytest.warns(FallbackWarning):
        failure = cls.from_estimate(np.zeros_like(src), dst)
    original = stock.ThinPlateSplineTransform.from_estimate(np.zeros_like(src), dst)
    assert not failure and str(failure) == str(original)
