"""Public class contracts, model kernels and RANSAC protocol behavior."""

import inspect
from warnings import catch_warnings, simplefilter

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from skimage import measure as original
from skimagexx import FallbackWarning, set_fallback_enabled
from skimagexx import measure


@pytest.fixture
def strict():
    previous = set_fallback_enabled(False)
    try:
        yield
    finally:
        set_fallback_enabled(previous)


@pytest.mark.parametrize('name', ['CircleModel', 'EllipseModel', 'RansacModelProtocol'])
def test_class_contract(name):
    expected = getattr(original, name)
    actual = getattr(measure, name)
    assert inspect.signature(actual) == inspect.signature(expected)
    assert set(dir(expected)) <= set(dir(actual))
    assert issubclass(actual, expected.__bases__[0])
    for name in dir(expected):
        a, b = inspect.getattr_static(actual, name), inspect.getattr_static(expected, name)
        if isinstance(b, property):
            assert isinstance(a, property)
            assert inspect.signature(a.fget) == inspect.signature(b.fget)
        if isinstance(b, (classmethod, staticmethod)):
            assert isinstance(a, type(b))
        try:
            signature = inspect.signature(getattr(expected, name))
        except (TypeError, ValueError, AttributeError):
            continue
        assert inspect.signature(getattr(actual, name)) == signature


@pytest.mark.parametrize('name, parameters', [
    ('CircleModel', ((3., -2.), 7.)),
    ('EllipseModel', ((3., -2.), (7., 4.), 0.42)),
])
def test_model_geometry(name, parameters, strict):
    reference = getattr(original, name)
    family = getattr(measure, name)
    expected = reference(*parameters)
    actual = family(*parameters)
    t = np.linspace(-0.55, 5.8, 180, dtype=np.dtype('double'))
    data = np.ascontiguousarray(expected.predict_xy(t) + 0.025 * np.random.default_rng(18).normal(size=(180, 2)))
    assert_allclose(actual.predict_xy(t.reshape(5, 36)), expected.predict_xy(t.reshape(5, 36)))
    assert_allclose(actual.predict_xy(0.75), expected.predict_xy(0.75))
    assert_allclose(actual.residuals(data), expected.residuals(data), rtol=2e-5, atol=4e-7)
    try:
        fitted_reference = reference.from_estimate(data)
    except TypeError as error:
        # NumPy builds returning complex eigenvectors expose a reference
        # TypeError. The C++ consumer covers successful ellipse fitting.
        assert name == 'EllipseModel'
        with pytest.raises(TypeError) as fit_error:
            family.from_estimate(data)
        assert str(fit_error.value) == str(error)
        with pytest.warns(FutureWarning), pytest.raises(TypeError) as legacy_error:
            actual.estimate(data)
        assert str(legacy_error.value) == str(error)
        return
    fitted = family.from_estimate(data)
    assert type(fitted) is family
    assert isinstance(fitted, measure.RansacModelProtocol)
    assert_allclose(fitted.center, fitted_reference.center, atol=1e-9)
    assert_allclose(fitted.predict_xy(t), fitted_reference.predict_xy(t), atol=1e-9)
    with pytest.warns(FutureWarning):
        success = actual.estimate(data)
    assert success
    assert_allclose(actual.center, fitted.center)


@pytest.mark.parametrize('name, parameters', [
    ('CircleModel', ((0., 1.), 2.)),
    ('EllipseModel', ((0., 1.), (5., 2.), 0.42)),
])
def test_dtype_metadata_and_legacy(name, parameters):
    reference = getattr(original, name)
    family = getattr(measure, name)
    data = reference(*parameters).predict_xy(np.arange(10, dtype=np.float32))
    marked = data.view(np.dtype(data.dtype, metadata={'unit': 'xy'}))
    try:
        upstream = reference.from_estimate(marked)
    except TypeError as error:
        assert name == 'EllipseModel'
        with pytest.warns(FallbackWarning), pytest.raises(TypeError) as native_error:
            family.from_estimate(marked)
        assert str(native_error.value) == str(error)
    else:
        with pytest.warns(FallbackWarning):
            fitted = family.from_estimate(marked)
        assert_allclose(fitted.center, upstream.center)
    with pytest.warns(FallbackWarning):
        failed = family.from_estimate(np.zeros((6, 2)))
    assert not failed and type(failed) is type(reference.from_estimate(np.zeros((6, 2))))
    with catch_warnings(record=True) as warnings:
        simplefilter('always')
        legacy = family()
        with pytest.raises(ValueError, match='params'):
            legacy.predict_xy(0.)
    assert any(issubclass(item.category, FutureWarning) for item in warnings)


class OffsetModel:
    def __init__(self, value):
        self.value = value

    @classmethod
    def from_estimate(cls, x, y):
        return cls(np.mean(y - x))

    def residuals(self, x, y):
        return y - x - self.value


@pytest.mark.parametrize('initial', [None, np.arange(20) < 10])
def test_ransac_protocol_callbacks_and_rng(initial, strict):
    assert inspect.signature(measure.ransac) == inspect.signature(original.ransac)
    x = np.arange(20.)
    y = x + 2
    y[::5] += 6

    def run(function):
        generator = np.random.default_rng(41)
        events = []

        def valid_data(a, b):
            events.append(('data', a.tolist(), b.tolist()))
            return True

        def valid_model(model, a, b):
            events.append(('model', model.value, len(a), len(b)))
            return model.value < 7

        model, inliers = function(
            (x, y), OffsetModel, 4, .15, valid_data, valid_model, max_trials=60,
            stop_probability=.999, rng=generator, initial_inliers=initial,
        )
        return model.value, inliers, events, generator.bit_generator.state

    expected = run(original.ransac)
    actual = run(measure.ransac)
    assert actual[0] == expected[0]
    assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_ransac_native_circle(strict):
    rng = np.random.default_rng(331)
    points = original.CircleModel((1., 2.), 4.).predict_xy(np.linspace(0., 6., 160))
    points += rng.normal(0, .01, points.shape)
    points[::7] += 2
    low, expected = original.ransac(points, original.CircleModel, 4, .08, rng=73)
    model, actual = measure.ransac(points, measure.CircleModel, 4, .08, rng=73)
    assert_array_equal(actual, expected)
    assert type(model) is measure.CircleModel
    assert_allclose(model.center, low.center)
    assert_allclose(model.radius, low.radius)


def test_live_list_and_hot_swap(monkeypatch, strict):
    from skimagexx import _core
    from skimagexx import _geometric_models as wrappers

    assert all(
        getattr(measure, name) is getattr(wrappers, name)
        for name in ('CircleModel', 'EllipseModel', 'RansacModelProtocol', 'ransac')
    )
    assert inspect.isbuiltin(_core.measure_ransac_loop)
    x = np.arange(10.)
    y = x + 3

    def run(function):
        arrays = [x, y + 1]

        def update(*args):
            arrays[1] = y
            return True

        model, inliers = function(arrays, OffsetModel, 3, 0.5, is_data_valid=update, rng=813)
        return model.value, inliers

    before = run(original.ransac)
    monkeypatch.setattr(original, 'ransac', measure.ransac)
    after = run(original.ransac)
    assert after[0] == before[0]
    assert_array_equal(after[1], before[1])
