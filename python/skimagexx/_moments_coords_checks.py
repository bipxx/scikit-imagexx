"""Frozen coordinate-monomial policy for untimed default verification only."""
import sys
import numpy as np


def default_limits(coords, calc, expected):
    """Use the captured typed source monomials; never infer an image extent."""
    if calc.shape != (coords.shape[0],) + expected.shape:
        raise AssertionError('coordinate moments monomial shape differs')
    if not np.all(np.isfinite(calc)) or not np.all(np.isfinite(expected)):
        return np.zeros(expected.shape, dtype=np.float64)
    u = float(np.finfo(expected.real.dtype).eps) / 2
    q = coords.shape[0] + coords.shape[1]
    if q * u >= 1:
        raise ValueError('coordinate moments default exceeds the frozen reduction frontier')
    gamma = q * u / (1 - q * u)
    scale = np.sum(np.abs(calc).astype(np.float64), axis=0, dtype=np.float64)
    bound = 2 * gamma * scale
    if not np.all(np.isfinite(bound) & (bound >= 0)):
        raise AssertionError('finite coordinate moments reference has an invalid allowance')
    return bound


def default_error(api, coords, actual, expected):
    """Capture genuine reference monomials outside any measured callback."""
    from skimage.measure import _moments as original
    captured = {}
    prior = sys.getprofile()
    def capture(frame, event, argument):
        if event == 'return' and frame.f_code is original.moments_coords_central.__code__:
            captured['calc'] = frame.f_locals.get('calc')
    sys.setprofile(capture)
    try:
        getattr(original, api.rsplit('.', 1)[1])(coords)
    finally:
        sys.setprofile(prior)
    if type(actual) is not np.ndarray or actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError('coordinate moments output shape or dtype differs')
    with np.errstate(all='ignore'):
        bound = default_limits(coords, captured['calc'], expected)
    for mask in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(mask(actual), mask(expected))
    finite = np.isfinite(expected)
    difference = np.abs(actual[finite] - expected[finite])
    if not np.all(difference <= bound[finite]):
        raise AssertionError('coordinate moments exceed the frozen monomial allowance')
    exact = finite & (bound == 0)
    np.testing.assert_array_equal(actual[exact], expected[exact])
    np.testing.assert_array_equal(np.signbit(actual[exact]), np.signbit(expected[exact]))
    return float(np.max(difference, initial=0.))
