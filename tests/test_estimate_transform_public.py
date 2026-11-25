"""Factory parity against the original scikit-image public call."""

import inspect
import warnings

import numpy as np
import pytest
from skimage import transform as original

import skimagexx as xx
from skimagexx import _estimate_transform, _geometric


MODELS = {
    'affine': 'AffineTransform',
    'euclidean': 'EuclideanTransform',
    'projective': 'ProjectiveTransform',
    'similarity': 'SimilarityTransform',
}
DELEGATED = ('essential', 'fundamental', 'piecewise-affine', 'polynomial')


@pytest.fixture(autouse=True)
def strict_fallback():
    previous = xx.set_fallback_enabled(False)
    yield
    xx.set_fallback_enabled(previous)


@pytest.fixture
def factory():
    candidate = xx.transform.estimate_transform
    assert candidate is _estimate_transform.estimate_transform
    return candidate


def points(ndim=2):
    src = np.random.default_rng(831).uniform(-3, 3, (12, ndim))
    dst = src + np.arange(ndim) + 0.5
    return src, dst


def capture(call, *args, **kwargs):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        try:
            result, error = call(*args, **kwargs), None
        except Exception as caught:
            result, error = None, caught
    return result, error, [(item.category, str(item.message)) for item in recorded]


def test_signature(factory):
    assert inspect.signature(factory) == inspect.signature(original.estimate_transform)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('ndim', (2, 3, 4))
@pytest.mark.parametrize('layout', ('c', 'fortran', 'strided', 'lists'))
def test_owned_fit(factory, model, ndim, layout):
    src, dst = points(ndim)
    query = src.copy()
    if layout == 'fortran':
        src, dst = np.asfortranarray(src), np.asfortranarray(dst)
    elif layout == 'strided':
        src = src.astype(np.float32)[::-2, ::-1]
        dst = dst.astype(np.float32)[::-2, ::-1]
    elif layout == 'lists':
        src, dst = src.tolist(), dst.tolist()
    actual, error, notices = capture(factory, model.upper(), src, dst)
    expected, original_error, original_notices = capture(original.estimate_transform, model.upper(), src, dst)
    assert notices == original_notices == []
    assert error is original_error is None
    assert type(actual) is getattr(xx.transform, MODELS[model])
    assert type(expected) is getattr(original, MODELS[model])
    assert bool(actual) and bool(expected)
    assert actual.dimensionality == expected.dimensionality == ndim
    assert actual.params.dtype == expected.params.dtype
    np.testing.assert_allclose(actual.params, expected.params, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual(query), expected(query), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual.inverse(query), expected.inverse(query), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('model', ('affine', 'projective'))
@pytest.mark.parametrize('as_keyword', (False, True))
def test_weight_forwarding(factory, model, as_keyword):
    src, dst = points()
    dst[0] += 0.375
    weights = np.arange(1, len(src) + 1, dtype=np.float64)
    args = () if as_keyword else (weights,)
    kwargs = {'weights': weights} if as_keyword else {}
    actual = factory(model, src, dst, *args, **kwargs)
    expected = original.estimate_transform(model, src, dst, *args, **kwargs)
    np.testing.assert_allclose(actual.params, expected.params, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('model', ('euclidean', 'similarity'))
def test_reflected_rigid_fit(factory, model):
    src, dst = points(3)
    dst[:, 0] *= -1
    actual = factory(model, src, dst)
    expected = original.estimate_transform(model, src, dst)
    np.testing.assert_array_equal(actual.params, expected.params)


def test_underconstrained_projective_fit(factory):
    src, dst = points()
    actual = factory('projective', src[:4], dst[:4])
    expected = original.estimate_transform('projective', src[:4], dst[:4])
    np.testing.assert_array_equal(actual.params, expected.params)


@pytest.mark.parametrize('model', MODELS)
def test_failed_estimation(factory, model):
    src, dst = np.zeros((5, 2)), np.ones((5, 2))
    actual, error, notices = capture(factory, model, src, dst)
    expected, original_error, original_notices = capture(original.estimate_transform, model, src, dst)
    assert (error, notices) == (original_error, original_notices)
    assert type(actual) is _geometric.FailedEstimation
    assert not actual and not expected
    assert (str(actual), repr(actual)) == (str(expected), repr(expected))
    for operation in (lambda model: model(src), lambda model: model.params):
        _, actual_error, actual_notices = capture(operation, actual)
        _, expected_error, expected_notices = capture(operation, expected)
        assert actual_notices == expected_notices
        assert type(actual_error) is type(expected_error)
        assert str(actual_error) == str(expected_error)


@pytest.mark.parametrize('model,src,dst,args,kwargs', (
    ('bogus', None, None, (), {}),
    ('AFFINE ', None, None, (), {}),
    (None, None, None, (), {}),
    (b'euclidean', None, None, (), {}),
    ('affine', [[0], [1]], [[1], [2]], (), {}),
    ('euclidean', [0, 1], [1, 2], (), {}),
    ('projective', np.empty((0, 2)), np.empty((0, 2)), (), {}),
    ('projective', [[0, 0], [1, 1]], [[0, 0], [1, 1], [2, 2]], (), {}),
    ('similarity', [[0, 0], [1, 1]], [[0, 0], [1, 1]], (1,), {}),
    ('affine', [[0, 0], [1, 1]], [[0, 0], [1, 1]], (), {'order': 2}),
    ('projective', [[0, 0], [1, 1]], [[0, 0], [1, 1]], (), {'weights': [-1, 1]}),
))
def test_invalid_inputs(factory, model, src, dst, args, kwargs):
    _, error, notices = capture(factory, model, src, dst, *args, **kwargs)
    _, original_error, original_notices = capture(original.estimate_transform, model, src, dst, *args, **kwargs)
    assert type(error) is type(original_error)
    assert str(error) == str(original_error)
    assert notices == original_notices


@pytest.mark.parametrize('model', DELEGATED)
def test_explicit_fallback(factory, model):
    src, dst = points()
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        factory(model, src, dst)

    xx.set_fallback_enabled(True)
    args = (2,) if model == 'polynomial' else ()
    actual, error, notices = capture(factory, model.upper(), src, dst, *args)
    expected, original_error, original_notices = capture(original.estimate_transform, model, src, dst, *args)
    assert notices == [
        (xx.FallbackWarning, 'skimage.transform.estimate_transform executed by scikit-image fallback'),
        *original_notices,
    ]
    assert type(error) is type(original_error)
    assert str(error) == str(original_error)
    assert type(actual) is type(expected)
    if expected:
        np.testing.assert_allclose(actual(src), expected(src), atol=1e-12, rtol=1e-12)


def test_owned_factory_does_not_call_original_factory(monkeypatch, factory):
    def reject(*args, **kwargs):
        raise AssertionError('owned factory called scikit-image')

    monkeypatch.setattr(original, 'estimate_transform', reject)
    src, dst = points()
    for model in MODELS:
        assert type(factory(model, src, dst)) is getattr(xx.transform, MODELS[model])


def test_tagged_points_use_compatibility(factory):
    src, dst = points(3)
    src = src.view(np.dtype('f8', metadata={'units': 'pixels'}))
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        factory('projective', src, dst)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        actual = factory('projective', src, dst)
    assert type(actual) is original.ProjectiveTransform
    np.testing.assert_allclose(actual.params, original.estimate_transform('projective', src, dst).params)
