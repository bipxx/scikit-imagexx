"""Frozen Shannon entropy policy for untimed default verification only."""
import inspect
import math
import sys
import numpy as np


def error_limit(base, nats, expected):
    """Propagate the existing NMI entropy-only budget through base conversion."""
    if not np.isfinite(expected) or nats == 0:
        return 0.
    denominator = 1. if base is None else abs(math.log(base))
    if math.isinf(denominator):
        return 0.
    assert denominator > 0 and math.isfinite(denominator)
    allowance = (2e-13 + 2e-12 * abs(float(nats))) / denominator
    if base is not None:
        allowance += float(np.finfo(np.float64).eps) / 2 * abs(float(expected))
    assert math.isfinite(allowance) and allowance >= 0
    return allowance


def default_error(image, actual, expected):
    """Capture the genuine reference nats before its default-base division."""
    from skimage.measure import entropy as original
    backend = inspect.unwrap(original.scipy_entropy).__code__
    captured = {}
    prior = sys.getprofile()
    def capture(frame, event, argument):
        if frame.f_code is backend and event == 'c_call' and argument is math.log:
            captured['nats'] = frame.f_locals['S']
    sys.setprofile(capture)
    try:
        original.shannon_entropy(image)
    finally:
        sys.setprofile(prior)
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.shape == e.shape == () and a.dtype == e.dtype == np.dtype(np.float64)
    for mask in (np.isnan, np.isposinf, np.isneginf):
        assert bool(mask(a)) == bool(mask(e))
    bound = error_limit(2, captured['nats'], e)
    if not np.isfinite(e):
        return 0.
    difference = abs(float(a) - float(e))
    assert difference <= bound, 'Shannon entropy exceeds the frozen allowance'
    if bound == 0:
        assert a == e and np.signbit(a) == np.signbit(e)
    return difference
