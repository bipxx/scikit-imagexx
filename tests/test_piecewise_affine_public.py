import inspect
import pickle
from unittest.mock import patch

import numpy as np
import pytest
from skimage import transform as stock

import skimagexx as xx
from skimagexx._compat import FallbackWarning


def _controls(dtype=np.float64):
    src = np.array([[0, 0], [4, 0], [0, 3], [4, 4], [1, 1]], dtype=dtype)
    dst = np.array([[1, 2], [5, 2], [1, 5], [5, 6], [1.5, 3.5]], dtype=dtype)
    return src, dst


def test_piecewise_affine_static_members_and_private_estimation():
    cls = xx.transform.PiecewiseAffineTransform
    original = stock.PiecewiseAffineTransform
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
    a, src_a, dst_a = cls._prepare_estimation(src, dst.tolist())
    b, src_b, dst_b = original._prepare_estimation(src, dst.tolist())
    assert type(a) is cls and type(b) is original
    assert a.affines is b.affines is None and a.inverse_affines is b.inverse_affines is None
    assert src_a is src_b is src
    np.testing.assert_array_equal(dst_a, dst_b)
    assert a._estimate(src_a, dst_a) is b._estimate(src_b, dst_b) is None
    np.testing.assert_allclose(a(src), b(src), atol=1e-12)


@pytest.mark.parametrize('dtype', [np.int32, np.float32, np.float64])
@pytest.mark.parametrize('workers', [1, 2])
def test_piecewise_affine_public_hot_swap(dtype, workers):
    cls = xx.transform.PiecewiseAffineTransform
    original = stock.PiecewiseAffineTransform
    assert inspect.isbuiltin(xx._core.piecewise_affine_fit)
    assert inspect.isbuiltin(xx._core.piecewise_affine_apply)
    src, dst = _controls(dtype)
    query = np.vstack([src, [[-1, -2], [5, 6], [4, 2], [1.5, 1], [2, 3]]])
    if workers == 2:
        query = np.tile(query, (8192, 1))
    ref = original.from_estimate(src, dst)
    before = xx.set_fallback_enabled(False)
    previous_workers = xx.get_num_threads()
    try:
        xx.set_num_threads(workers)
        with patch.object(stock, 'PiecewiseAffineTransform', cls):
            model = stock.PiecewiseAffineTransform.from_estimate(src, dst)
            for actual, expected in ((model.affines, ref.affines),
                                     (model.inverse_affines, ref.inverse_affines)):
                assert len(actual) == len(expected)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a.params, b.params, atol=1e-12)
            column_major = np.asfortranarray(query[::-1])
            for coords in (query, column_major, column_major[::2], query[:0], query[:1].tolist()):
                got, expected = model(coords), ref(coords)
                np.testing.assert_allclose(got, expected, atol=1e-12)
                assert got.dtype == expected.dtype and got.strides == expected.strides
            np.testing.assert_allclose(model.inverse(dst), ref.inverse(dst), atol=1e-12)
            np.testing.assert_allclose(model.residuals(src, dst), ref.residuals(src, dst), atol=1e-12)
            assert model.inverse.affines is not model.inverse_affines
            assert model.inverse.affines[0] is model.inverse_affines[0]
            np.testing.assert_allclose(pickle.loads(pickle.dumps(model))(src), dst, atol=1e-12)
            with pytest.warns(FutureWarning, match='PiecewiseAffineTransform.from_estimate'):
                assert model.estimate(src, dst)
    finally:
        xx.set_num_threads(previous_workers)
        xx.set_fallback_enabled(before)


def test_piecewise_affine_identity_and_explicit_fallback():
    cls = xx.transform.PiecewiseAffineTransform
    src, dst = _controls()
    with pytest.raises(AttributeError, match='find_simplex'):
        cls.identity(dimensionality=3)(src)
    model = cls.from_estimate(src, dst)
    original = stock.PiecewiseAffineTransform.from_estimate(src, dst)
    model.affines[0].params[0, 2] += 3
    original.affines[0].params[0, 2] += 3
    with pytest.warns(FallbackWarning):
        np.testing.assert_allclose(model(src), original(src), atol=1e-12)

    src3 = np.vstack([np.eye(3), np.zeros(3)])
    with pytest.warns(FallbackWarning):
        model3 = cls.from_estimate(src3, src3 + 1)
    with pytest.warns(FallbackWarning):
        np.testing.assert_allclose(model3(src3), src3 + 1, atol=1e-12)
