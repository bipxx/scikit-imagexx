"""Frozen moments policy for default per-function tests and benchmark checks.

This checker is not part of the image algorithm. Full keyword contracts retain
an independent frozen evaluator in tests/moments_cases.py. Default inputs have
unit spacing, raw order3 or automatic central order3; centroid uses order1.
"""
import itertools
import math
import numpy as np


def default_limits(api, image, expected):
    weights = image.astype(np.float64)
    if not np.all(np.isfinite(weights)):
        return np.zeros(expected.shape, dtype=np.float64)
    eps_in = float(np.finfo(np.float32 if image.dtype.kind == 'f' and image.dtype.itemsize <= 4 else np.float64).eps)
    eps_out = float(np.finfo(expected.real.dtype).eps)
    g = 8 * max(eps_in, eps_out) * max(1, sum(image.shape))
    reference_abs = np.abs(expected.astype(np.complex128 if expected.dtype.kind == 'c' else np.float64))
    coordinates = [np.arange(n, dtype=np.float64) for n in image.shape]
    absolute = np.abs(weights)
    mass, total = float(np.sum(weights, dtype=np.float64)), float(np.sum(absolute, dtype=np.float64))
    auto = api == 'measure.moments_central'
    if auto or api == 'measure.centroid':
        if mass == 0 or not math.isfinite(mass) or not math.isfinite(total):
            return np.zeros(expected.shape, dtype=np.float64)
        condition = total / abs(mass)
        if condition > 8 or g * condition >= .25:
            raise ValueError('moments default comparison exceeds the frozen conditioning frontier')
        center = []
        for d, x in enumerate(coordinates):
            shape = [1] * image.ndim
            shape[d] = image.shape[d]
            center.append(np.sum(weights * x.reshape(shape)) / mass)
        if api == 'measure.centroid':
            radius = np.asarray([max(np.abs(x), default=0.) for x in coordinates])
            return g * reference_abs + 2 * g * condition * (radius + np.abs(center)) / (1 - g * condition)
    else:
        center = (0.,) * image.ndim
        condition = 0.
    bound = np.empty(expected.shape, dtype=np.float64)
    for alpha in itertools.product(*(range(n) for n in expected.shape)):
        terms = absolute.copy()
        for d, degree in enumerate(alpha):
            distances = np.abs(coordinates[d]) + abs(center[d]) if auto else np.abs(coordinates[d] - center[d])
            shape = [1] * image.ndim
            shape[d] = image.shape[d]
            terms *= (distances ** degree).reshape(shape)
        scale = float(np.sum(terms, dtype=np.float64))
        bound[alpha] = g * float(reference_abs[alpha]) + g * (1 + 2 * sum(alpha) * condition) * scale
    return bound


def default_error(api, image, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError('moments output shape or dtype differs')
    with np.errstate(all='ignore'):
        bound = default_limits(api, image, expected)
    forced = np.zeros(expected.shape, dtype=bool)
    fast = False
    if api == 'measure.moments_central':
        order = expected.shape[0] - 1
        fast = expected.ndim in (2, 3) and order < 4
        for alpha in itertools.product(*(range(n) for n in expected.shape)):
            forced[alpha] = sum(alpha) > order or (fast and sum(alpha) == 1)
    if fast and expected.dtype.kind == 'c':
        if np.any(actual.imag != 0) or np.any(np.signbit(actual.imag)):
            raise AssertionError('moments structural imaginary zeros differ')
    components = ((actual.real, expected.real), (actual.imag, expected.imag)) if expected.dtype.kind == 'c' else ((actual, expected),)
    for a, e in components:
        for mask in (np.isnan, np.isposinf, np.isneginf):
            np.testing.assert_array_equal(mask(a), mask(e))
        finite = np.isfinite(e)
        if not np.all(np.isfinite(bound[finite]) & (bound[finite] >= 0)):
            raise AssertionError('finite moments reference has an invalid allowance')
        if not np.all(np.abs(a[finite] - e[finite]) <= bound[finite]):
            raise AssertionError('moments values exceed the frozen cell-specific allowance')
        exact_zero = forced | ((bound == 0) & (e == 0))
        np.testing.assert_array_equal(a[exact_zero], e[exact_zero])
        np.testing.assert_array_equal(np.signbit(a[exact_zero]), np.signbit(e[exact_zero]))
    finite = np.isfinite(expected)
    return float(np.max(np.abs(actual[finite] - expected[finite]), initial=0.))
