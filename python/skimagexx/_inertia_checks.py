"""Verification-only default inertia policy; never used by the image algorithm.

The genuine original central moments are prepared outside benchmark timing.
The frozen moments allowance is propagated through the tiny tensor quotient,
using the same formula as tests/inertia_tensor_cases.py. No global tolerance
or candidate-derived scale replaces this cell-specific policy.
"""
import itertools
import math
import numpy as np
from ._moments_checks import default_limits as central_limits


def default_limits(image, mu, expected):
    bound = np.zeros(expected.shape, dtype=np.float64)
    if not np.all(np.isfinite(mu)):
        return bound
    d = image.ndim
    uncertainty = central_limits('measure.moments_central', image, mu)
    m = float(mu[(0,) * d])
    if m == 0:
        return bound
    b = float(uncertainty[(0,) * d])
    if not (math.isfinite(m) and b >= 0 and b < abs(m) / 4):
        raise ValueError('inertia default comparison exceeds the frozen mass frontier')
    corners = [tuple(2 if k == axis else 0 for k in range(d)) for axis in range(d)]
    q = [float(mu[index]) for index in corners]
    e = [float(uncertainty[index]) for index in corners]
    u = float(np.finfo(expected.real.dtype).eps) / 2
    for i, j in itertools.product(range(d), repeat=2):
        if i == j:
            a, delta, operations = sum(map(abs, q)) + abs(q[i]), sum(e) + e[i], d + 3
        else:
            index = tuple(1 if k in (i, j) else 0 for k in range(d))
            a, delta, operations = abs(float(mu[index])), float(uncertainty[index]), 3
        g = 2 * operations * u / (1 - 2 * operations * u)
        reference = abs(float(expected[i, j]))
        bound[i, j] = (delta + reference * b + g * (a + delta + reference * abs(m))) / (abs(m) - b)
    if d == 1:
        bound[0, 0] = 0
    finite = np.isfinite(expected)
    if not np.all(np.isfinite(bound[finite]) & (bound[finite] >= 0)):
        raise AssertionError('finite inertia reference has an invalid allowance')
    return bound


def default_error(image, actual, expected):
    # This is bound preparation only: public candidate code calls no upstream
    # moment algorithm, and neither this call nor the checker is timed.
    from skimage.measure import moments_central as reference_moments
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError('inertia output shape or dtype differs')
    with np.errstate(all='ignore'):
        mu = reference_moments(image, order=2)
        bound = default_limits(image, mu, expected)
    for mask in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(mask(actual), mask(expected))
    finite = np.isfinite(expected)
    difference = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
    if not np.all(difference <= bound[finite]):
        raise AssertionError('inertia values exceed the frozen cell-specific allowance')
    exact = (bound == 0) & finite
    np.testing.assert_array_equal(actual[exact], expected[exact])
    np.testing.assert_array_equal(np.signbit(actual[exact]), np.signbit(expected[exact]))
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(actual.T))
    return float(np.max(difference, initial=0.))


def eigvals_default_error(image, actual, expected):
    """Frozen solver envelope plus Weyl's bound from the original tensor."""
    if type(actual) is not list or type(expected) is not list or len(actual) != len(expected):
        raise AssertionError('inertia eigenvalues must preserve the scalar list')
    for a, e in zip(actual, expected):
        if type(a) is not type(e) or a.dtype != e.dtype:
            raise AssertionError('inertia eigenvalue scalar type or dtype differs')
    from skimage.measure import moments_central as reference_moments, inertia_tensor as reference_tensor
    a, e = np.asarray(actual), np.asarray(expected)
    with np.errstate(all='ignore'):
        mu = reference_moments(image, order=2)
        tensor = reference_tensor(image, mu=mu)
        bound = np.zeros(e.shape, dtype=np.float64)
        if np.all(np.isfinite(tensor)):
            tolerance = 2e-6 if e.dtype == np.float32 else 2e-13
            scale = float(np.max(np.abs(np.tril(tensor)), initial=0.))
            bound[:] = tolerance * np.abs(e.astype(np.float64)) + tolerance * scale
            cells = default_limits(image, mu, tensor)
            bound += float(np.max(np.sum(cells, axis=1), initial=0.))
            if image.ndim == 1:
                bound[:] = 0
    for mask in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(mask(a), mask(e))
    finite = np.isfinite(e)
    if not np.all(np.isfinite(bound[finite]) & (bound[finite] >= 0)):
        raise AssertionError('finite inertia eigenvalue has an invalid allowance')
    difference = np.abs(a[finite].astype(np.float64) - e[finite].astype(np.float64))
    if not np.all(difference <= bound[finite]):
        raise AssertionError('inertia eigenvalues exceed the frozen allowance')
    exact = finite & (bound == 0)
    np.testing.assert_array_equal(a[exact], e[exact])
    np.testing.assert_array_equal(np.signbit(a[exact]), np.signbit(e[exact]))
    if not np.all(a[finite] >= 0):
        raise AssertionError('inertia eigenvalues must be nonnegative')
    if np.all(finite) and not all(a[i] >= a[i + 1] for i in range(len(a) - 1)):
        raise AssertionError('inertia eigenvalues must be descending')
    return float(np.max(difference, initial=0.))
