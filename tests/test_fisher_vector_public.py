import importlib.util
import inspect
import sys
from types import ModuleType

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import logsumexp
from skimage import feature as upstream

import skimagexx as skx
from skimagexx import _core, _fisher_vector, feature
from skimagexx._compat import FallbackWarning


HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def _descriptors(dimensions=12):
    generator = np.random.default_rng(581)
    return np.vstack([generator.normal(loc, .75, (70, dimensions)) for loc in (-3., 0., 3.)])


def _expectation(data, weights, means, covariances):
    residuals = data[:, None, :] - means
    log_joint = np.log(weights) - .5 * np.sum(
        np.log(2 * np.pi * covariances) + residuals**2 / covariances, axis=2
    )
    log_density = logsumexp(log_joint, axis=1)
    return log_density, log_joint - log_density[:, None]


def _maximization(data, log_resp, reg_covar):
    resp = np.exp(log_resp)
    mass = resp.sum(axis=0) + 10 * np.finfo(data.dtype).eps
    means = (resp.T @ data) / mass[:, None]
    residuals = data[:, None, :] - means
    covariances = (resp[:, :, None] * residuals**2).sum(axis=0) / mass[:, None] + reg_covar
    return mass / mass.sum(), means, covariances, 1 / np.sqrt(covariances)


class FakeGaussianMixture:
    def __init__(self, n_components, covariance_type, *, max_iter=3, reg_covar=1e-6):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.reg_covar = reg_covar

    def _initialize(self, data):
        self.n_features_in_ = data.shape[1]
        groups = np.array_split(data, self.n_components)
        self.weights_ = np.array([len(group) / len(data) for group in groups])
        self.means_ = np.array([group.mean(axis=0) for group in groups])
        self.covariances_ = np.array([group.var(axis=0) for group in groups]) + self.reg_covar
        self.precisions_cholesky_ = 1 / np.sqrt(self.covariances_)

    def _estimate_log_prob_resp(self, data):
        return _expectation(data, self.weights_, self.means_, self.covariances_)

    def _m_step(self, data, log_resp):
        self.weights_, self.means_, self.covariances_, self.precisions_cholesky_ = (
            _maximization(data, log_resp, self.reg_covar)
        )

    def fit(self, data):
        self._initialize(data)
        for _ in range(self.max_iter):
            _, log_resp = self._estimate_log_prob_resp(data)
            self._m_step(data, log_resp)
        return self


@pytest.fixture
def gaussian_mixture(monkeypatch):
    if HAS_SKLEARN:
        from sklearn.mixture import GaussianMixture

        yield GaussianMixture
        return

    def check_is_fitted(gmm):
        assert hasattr(gmm, "n_features_in_")

    def validate_data(gmm, data, *, reset):
        assert reset is False
        if data.shape[1] != gmm.n_features_in_:
            raise ValueError(f"Expected {gmm.n_features_in_} features, got {data.shape[1]}")
        return data

    try:
        with monkeypatch.context() as modules:
            for name in ("sklearn", "sklearn.mixture", "sklearn.utils", "sklearn.utils.validation"):
                modules.setitem(sys.modules, name, ModuleType(name))
            sys.modules["sklearn.mixture"].GaussianMixture = FakeGaussianMixture
            validation = sys.modules["sklearn.utils.validation"]
            validation.check_is_fitted = check_is_fitted
            validation.validate_data = validate_data
            _fisher_vector._provider.cache_clear()
            yield FakeGaussianMixture
    finally:
        _fisher_vector._provider.cache_clear()


def _fisher_reference(data, gmm, *, improved=False, alpha=0.5):
    weights, means, covariances = gmm.weights_, gmm.means_, gmm.covariances_
    _, log_resp = _expectation(data, weights, means, covariances)
    resp = np.exp(log_resp)
    standardized = (data[:, None, :] - means) / np.sqrt(covariances)
    weighted = resp[:, :, None] / np.sqrt(weights[:, None])
    vector = np.concatenate((
        (resp.mean(axis=0) - weights) / np.sqrt(weights),
        (weighted * standardized).mean(axis=0).ravel(),
        (weighted * (1 - standardized**2)).mean(axis=0).ravel() / np.sqrt(2),
    ))
    if improved:
        vector = np.sign(vector) * np.abs(vector)**alpha
        vector /= np.linalg.norm(vector)
    return vector


def _record_native_calls(monkeypatch, *names):
    calls = {}
    for name in names:
        native = getattr(_core, name)
        assert inspect.isbuiltin(native)
        calls[name] = 0

        def record(*args, call=native, key=name):
            result = call(*args)
            if result is not None:
                calls[key] += 1
            return result

        monkeypatch.setattr(_core, name, record)
    return calls


def test_public_signatures():
    for name in ("learn_gmm", "fisher_vector"):
        assert inspect.signature(getattr(feature, name)) == inspect.signature(getattr(upstream, name))


@pytest.mark.parametrize("improved,alpha", [(False, 0.0), (True, 0.5), (True, 0.75)])
@pytest.mark.parametrize("components", [1, 3])
def test_fisher_gradients(improved, alpha, components, gaussian_mixture, monkeypatch):
    train = _descriptors()
    options = {"random_state": 13} if HAS_SKLEARN else {}
    gmm = gaussian_mixture(n_components=components, covariance_type="diag", **options).fit(train)
    test = np.ascontiguousarray(train[::3, ::-1])
    reference = upstream.fisher_vector if HAS_SKLEARN else _fisher_reference
    expected = reference(test, gmm, improved=improved, alpha=alpha)
    calls = _record_native_calls(monkeypatch, "_diagonal_fisher_vector")
    previous = skx.set_fallback_enabled(False)
    try:
        result = feature.fisher_vector(test, gmm, improved=improved, alpha=alpha)
    finally:
        skx.set_fallback_enabled(previous)
    assert result.dtype == expected.dtype
    assert result.shape == (components * (1 + 2 * train.shape[1]),)
    assert_allclose(result, expected, rtol=2e-9, atol=2e-12)
    assert calls["_diagonal_fisher_vector"] > 0


@pytest.mark.parametrize("init_params", [
    "kmeans", "random", "k-means++", "random_from_data",
] if HAS_SKLEARN else [None])
@pytest.mark.parametrize("split", [False, True])
def test_diagonal_gmm_estimator(init_params, split, gaussian_mixture, monkeypatch):
    train = _descriptors(8)
    descriptors = np.split(train, 3) if split else train
    if HAS_SKLEARN:
        options = {"covariance_type": "diag", "init_params": init_params, "random_state": 27,
                   "n_init": 2, "max_iter": 400}
        expected = upstream.learn_gmm(descriptors, n_modes=3, gm_args=options)
    else:
        options = {"covariance_type": "diag", "reg_covar": 0.02}
        expected = gaussian_mixture(n_components=3, **options).fit(train)
        if not split:
            _assert_native_em_steps(train, expected)
    calls = _record_native_calls(
        monkeypatch, "_diagonal_gmm_expectation", "_diagonal_gmm_maximization",
    )
    previous = skx.set_fallback_enabled(False)
    try:
        result = feature.learn_gmm(descriptors, n_modes=3, gm_args=options)
    finally:
        skx.set_fallback_enabled(previous)
    assert type(result) is gaussian_mixture
    if HAS_SKLEARN:
        assert result.get_params() == expected.get_params()
        assert result.n_iter_ == expected.n_iter_
        assert result.converged_ == expected.converged_
        assert_allclose(result.predict_proba(train), expected.predict_proba(train), atol=2e-12)
    assert result.n_features_in_ == expected.n_features_in_
    for attr in ("weights_", "means_", "covariances_", "precisions_", "precisions_cholesky_",
                 "lower_bound_", "lower_bounds_"):
        if hasattr(expected, attr):
            assert_allclose(getattr(result, attr), getattr(expected, attr), rtol=1e-9, atol=2e-12)
    assert "_m_step" not in vars(result)
    assert "_estimate_log_prob_resp" not in vars(result)
    assert all(count > 0 for count in calls.values())


def _assert_native_em_steps(train, gmm):
    log_density, log_resp = _expectation(train, gmm.weights_, gmm.means_, gmm.covariances_)
    actual_density, actual_resp = _core._diagonal_gmm_expectation(
        train, gmm.weights_, gmm.means_, gmm.precisions_cholesky_,
    )
    assert_allclose(actual_density, log_density, rtol=1e-12, atol=2e-12)
    assert_allclose(actual_resp, log_resp, rtol=1e-12, atol=2e-12)
    actual = _core._diagonal_gmm_maximization(train, log_resp, gmm.reg_covar)
    expected = _maximization(train, log_resp, gmm.reg_covar)
    for result, reference in zip(actual, expected, strict=True):
        assert_allclose(result, reference, rtol=1e-12, atol=2e-12)


def test_dtype_alias_and_metadata(gaussian_mixture, monkeypatch):
    train = _descriptors()
    options = {"random_state": 13} if HAS_SKLEARN else {}
    gmm = gaussian_mixture(n_components=3, covariance_type="diag", **options).fit(train)
    dtype = np.dtype("d", metadata={"origin": "test"})
    alias = train.view(dtype)
    reference = upstream.fisher_vector if HAS_SKLEARN else _fisher_reference
    expected = reference(alias, gmm)
    calls = _record_native_calls(monkeypatch, "_diagonal_fisher_vector")
    previous = skx.set_fallback_enabled(False)
    try:
        result = feature.fisher_vector(alias, gmm)
        if not HAS_SKLEARN:
            with pytest.raises(ValueError, match="features"):
                feature.fisher_vector(alias[:, :-1].copy(), gmm)
    finally:
        skx.set_fallback_enabled(previous)
    assert result.dtype.metadata == expected.dtype.metadata
    assert_allclose(result, expected, atol=2e-12)
    assert calls["_diagonal_fisher_vector"] > 0


def test_provider_fallback_and_stock_validation():
    train = _descriptors()
    if not HAS_SKLEARN:
        _assert_missing_provider_errors(train)
        return
    options = {"covariance_type": "diag", "random_state": 13}
    float32 = train.astype(np.float32)
    gmm = upstream.learn_gmm(float32, n_modes=3, gm_args=options)
    with pytest.warns(FallbackWarning):
        result = feature.learn_gmm(float32, n_modes=3, gm_args=options)
    assert_allclose(result.weights_, gmm.weights_)
    with pytest.warns(FallbackWarning):
        descriptor = feature.fisher_vector(float32, gmm, improved=True, alpha=0.25)
    assert_allclose(descriptor, upstream.fisher_vector(float32, gmm, improved=True, alpha=0.25))
    with pytest.warns(FallbackWarning), pytest.raises(KeyError, match="covariance_type"):
        feature.learn_gmm(train, n_modes=3, gm_args={"random_state": 13})
    with pytest.raises(ValueError, match="features"):
        feature.fisher_vector(train[:, :-1].copy(), gmm)


def _assert_missing_provider_errors(train):
    _fisher_vector._provider.cache_clear()
    previous = skx.set_fallback_enabled(True)
    try:
        assert _fisher_vector._provider() is None
        for name, args, kwargs in (
            ("fisher_vector", (train, None), {}),
            ("learn_gmm", (train,), {"n_modes": 3}),
        ):
            public = getattr(feature, name)
            original = getattr(upstream, name)
            skx.set_fallback_enabled(False)
            with pytest.raises(NotImplementedError, match="fallback is disabled"):
                public(*args, **kwargs)
            skx.set_fallback_enabled(True)
            with pytest.raises(ImportError) as expected:
                original(*args, **kwargs)
            with pytest.warns(FallbackWarning), pytest.raises(ImportError) as result:
                public(*args, **kwargs)
            assert type(result.value) is type(expected.value)
            assert str(result.value) == str(expected.value)
    finally:
        skx.set_fallback_enabled(previous)
        _fisher_vector._provider.cache_clear()
