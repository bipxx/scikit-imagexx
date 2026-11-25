import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.mixture import GaussianMixture
from skimage import feature as upstream

import skimagexx as skx
from skimagexx import _core, feature
from skimagexx._compat import FallbackWarning


def _descriptors(dimensions=12):
    generator = np.random.default_rng(581)
    return np.vstack([generator.normal(loc, .75, (70, dimensions)) for loc in (-3., 0., 3.)])


def test_public_signatures():
    for name in ("learn_gmm", "fisher_vector"):
        assert inspect.signature(getattr(feature, name)) == inspect.signature(getattr(upstream, name))


@pytest.mark.parametrize("improved,alpha", [(False, 0.0), (True, 0.5), (True, 0.75)])
@pytest.mark.parametrize("components", [1, 3])
def test_fisher_gradients(improved, alpha, components):
    train = _descriptors()
    gmm = GaussianMixture(n_components=components, covariance_type="diag", random_state=13).fit(train)
    test = np.ascontiguousarray(train[::3, ::-1])
    expected = upstream.fisher_vector(test, gmm, improved=improved, alpha=alpha)
    previous = skx.set_fallback_enabled(False)
    try:
        result = feature.fisher_vector(test, gmm, improved=improved, alpha=alpha)
    finally:
        skx.set_fallback_enabled(previous)
    assert result.dtype == expected.dtype
    assert result.shape == (components * (1 + 2 * train.shape[1]),)
    assert_allclose(result, expected, rtol=2e-9, atol=2e-12)


@pytest.mark.parametrize("init_params", ["kmeans", "random", "k-means++", "random_from_data"])
@pytest.mark.parametrize("split", [False, True])
def test_diagonal_gmm_estimator(init_params, split, monkeypatch):
    train = _descriptors(8)
    descriptors = np.split(train, 3) if split else train
    options = {"covariance_type": "diag", "init_params": init_params, "random_state": 27,
               "n_init": 2, "max_iter": 400}
    expected = upstream.learn_gmm(descriptors, n_modes=3, gm_args=options)
    calls = {}
    for name in ("_diagonal_gmm_expectation", "_diagonal_gmm_maximization"):
        native = getattr(_core, name)
        assert inspect.isbuiltin(native)
        calls[name] = 0

        def record(*args, call=native, key=name):
            result = call(*args)
            if result is not None:
                calls[key] += 1
            return result

        monkeypatch.setattr(_core, name, record)
    previous = skx.set_fallback_enabled(False)
    try:
        result = feature.learn_gmm(descriptors, n_modes=3, gm_args=options)
    finally:
        skx.set_fallback_enabled(previous)
    assert type(result) is GaussianMixture
    assert result.get_params() == expected.get_params()
    assert result.n_iter_ == expected.n_iter_
    assert result.converged_ == expected.converged_
    assert result.n_features_in_ == expected.n_features_in_
    for attr in ("weights_", "means_", "covariances_", "precisions_", "precisions_cholesky_",
                 "lower_bound_", "lower_bounds_"):
        if hasattr(expected, attr):
            assert_allclose(getattr(result, attr), getattr(expected, attr), rtol=1e-9, atol=2e-12)
    assert_allclose(result.predict_proba(train), expected.predict_proba(train), atol=2e-12)
    assert "_m_step" not in vars(result)
    assert "_estimate_log_prob_resp" not in vars(result)
    assert all(count > 0 for count in calls.values())


def test_dtype_alias_and_metadata():
    train = _descriptors()
    gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=13).fit(train)
    dtype = np.dtype("d", metadata={"origin": "test"})
    alias = train.view(dtype)
    expected = upstream.fisher_vector(alias, gmm)
    previous = skx.set_fallback_enabled(False)
    try:
        result = feature.fisher_vector(alias, gmm)
    finally:
        skx.set_fallback_enabled(previous)
    assert result.dtype.metadata == expected.dtype.metadata
    assert_allclose(result, expected, atol=2e-12)


def test_provider_fallback_and_stock_validation():
    train = _descriptors()
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
