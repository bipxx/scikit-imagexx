"""Diagonal GMM expectation, maximization, and Fisher gradients."""

from functools import lru_cache

import numpy as np

from . import _core
from ._compat import fallback


@lru_cache(maxsize=1)
def _provider():
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.utils.validation import check_is_fitted, validate_data
    except ImportError:
        return None
    return GaussianMixture, check_is_fitted, validate_data


def _supported(data):
    return (type(data) is np.ndarray and data.dtype == np.float64 and data.ndim == 2
            and data.shape[0] > 0 and 0 < data.shape[1] <= 2048
            and data.flags.c_contiguous and data.flags.aligned)


def fisher_vector(descriptors, gmm, *, improved=False, alpha=0.5):
    """Compute the gradients of a fitted diagonal Gaussian mixture."""
    provider = _provider()
    if (provider is not None and type(gmm) is provider[0] and _supported(descriptors)
            and gmm.covariance_type == "diag" and type(improved) is bool
            and (not improved or type(alpha) is float and 0.5 <= alpha <= 1.0)
            and "predict_proba" not in vars(gmm)
            and "_estimate_log_prob_resp" not in vars(gmm)
            and "_estimate_log_prob" not in vars(gmm)
            and "_estimate_log_weights" not in vars(gmm)
            and "_estimate_weighted_log_prob" not in vars(gmm)
            and all(hasattr(gmm, attr) for attr in
                    ("weights_", "means_", "covariances_", "precisions_cholesky_"))):
        _, check_is_fitted, validate_data = provider
        check_is_fitted(gmm)
        data = validate_data(gmm, descriptors, reset=False)
        result = _core._diagonal_fisher_vector(
            data, gmm.weights_, gmm.means_, gmm.precisions_cholesky_,
            gmm.covariances_, improved, alpha if improved else 0.5,
        )
        if result is not None:
            return result
    return fallback("feature", "fisher_vector", descriptors, gmm, improved=improved, alpha=alpha)


def _fit(gmm, descriptors):
    # Let sklearn own validation, initialization, convergence, warnings, and
    # estimator state. Substitute the numerical loops for this fit only.
    def expectation(data, **kwargs):
        result = _core._diagonal_gmm_expectation(
            data, gmm.weights_, gmm.means_, gmm.precisions_cholesky_,
        )
        if result is None:
            return type(gmm)._estimate_log_prob_resp(gmm, data, **kwargs)
        return result

    def maximization(data, log_resp, **kwargs):
        result = _core._diagonal_gmm_maximization(data, log_resp, gmm.reg_covar)
        if result is None:
            type(gmm)._m_step(gmm, data, log_resp, **kwargs)
        else:
            gmm.weights_, gmm.means_, gmm.covariances_, gmm.precisions_cholesky_ = result

    gmm._estimate_log_prob_resp = expectation
    gmm._m_step = maximization
    try:
        return gmm.fit(descriptors)
    finally:
        del gmm._estimate_log_prob_resp
        del gmm._m_step


def learn_gmm(descriptors, *, n_modes=32, gm_args=None):
    """Fit a diagonal Gaussian mixture to one or more descriptor matrices."""
    provider = _provider()
    if (provider is not None and type(n_modes) is int and 0 < n_modes <= 256
            and (gm_args is None or type(gm_args) is dict and
                 (not gm_args or gm_args.get("covariance_type") == "diag"))):
        data = descriptors
        if (type(data) is list and data and all(_supported(part) for part in data)
                and all(part.shape[1] == data[0].shape[1] for part in data)):
            data = np.vstack(data)
        if _supported(data):
            options = gm_args or {"covariance_type": "diag"}
            return _fit(provider[0](n_components=n_modes, **options), data)
    return fallback("feature", "learn_gmm", descriptors, n_modes=n_modes, gm_args=gm_args)
