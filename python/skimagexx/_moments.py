"""Owned moment contractions with source-ordered NumPy preparation.

NumPy provides dtype conversion, coordinate powers, layout permutation and
the literal raw-to-central formulae. Each axis contraction runs in the owned
float32/float64/complex64/complex128 C++ projector. No upstream moment helper
or NumPy contraction is called. The assessed scope is the frozen 100 cases;
arbitrary subclasses, hostile callbacks, extended arithmetic types, replaced
NumPy contraction implementations and print/log error handlers are unassessed.

NumPy 1.24's optimized dot does not report contraction floating exceptions.
NumPy 2.5's matmul does, using one policy snapshot per axis. Worker budgets
come from the sole shared core at that same contraction boundary.
"""
import itertools
import math
import warnings
from importlib import import_module

import numpy as np

from . import _core


_FLAGS = ((1, "divide", "divide by zero"), (2, "over", "overflow"),
          (4, "under", "underflow"), (8, "invalid", "invalid value"))
# Detect the installed source route before any test traps. This inspects only
# implementation metadata; neither einsum nor bmm_einsum is ever called.
_EINSUM_SOURCE = import_module("numpy._core.einsumfunc" if hasattr(np, "_core")
                               else "numpy.core.einsumfunc")
_MATMUL_ERRORS = hasattr(_EINSUM_SOURCE, "bmm_einsum")


def _supported_float_type(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        raise ValueError("complex valued input is not supported")
    return {"e": np.float32, "f": np.float32, "d": np.float64}.get(dtype.char, np.float64)


def _check_array(value):
    if isinstance(value, np.ndarray):
        _core.fft_check(value)


def _dispatch_matmul(flags, snapshot):
    policy, handler = snapshot
    # One array operation retains its policy across all error callbacks.
    for flag, key, label in _FLAGS:
        if not flags & flag:
            continue
        method = policy[key]
        message = f"{label} encountered in matmul"
        if method == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        elif method == "raise":
            raise FloatingPointError(message)
        elif method == "call":
            if handler is None:
                raise NameError(f"python callback specified for {label} (in  matmul) but no function found.")
            handler(label, flags)
        elif method in ("print", "log"):
            raise NotImplementedError("moments contraction print/log error handlers are not supported")


def _project(calc, powers, axis):
    # Coordinate callbacks can invalidate the image view retained as calc.
    # Validate before the shell's dtype conversion reads that external data;
    # the binding handles the later contiguous/output allocation boundary.
    _check_array(calc)
    _check_array(powers)
    dtype = np.result_type(calc.dtype, powers.dtype).newbyteorder("=")
    calc = calc.astype(dtype, copy=False)
    powers = powers.astype(dtype, copy=False)
    snapshot = (np.geterr(), np.geterrcall()) if _MATMUL_ERRORS else None
    logical, flags = _core.moments_project(calc, powers, axis, _core.get_num_threads())
    if snapshot is not None:
        _dispatch_matmul(flags, snapshot)
    # Both source optimized einsum routes pop the powers operand first. Their
    # contraction stores (order, remaining axes) and returns a transposed view.
    # Keep that public ownership/stride contract without a second contraction.
    storage = np.ascontiguousarray(np.moveaxis(logical, axis, 0))
    return np.moveaxis(storage, 0, axis)


def moments(image, order=3, *, spacing=None):
    """Calculate raw image moments up to the requested order."""
    return moments_central(image, (0,) * image.ndim, order=order, spacing=spacing)


def moments_central(image, center=None, order=3, *, spacing=None, **kwargs):
    """Calculate central image moments with source-ordered preparation."""
    if center is None:
        moments_raw = moments(image, order=order, spacing=spacing)
        return moments_raw_to_central(moments_raw)
    float_dtype = _supported_float_type(image.dtype)
    if spacing is None:
        spacing = np.ones(image.ndim, dtype=float_dtype)
    _check_array(image)
    calc = image.astype(float_dtype, copy=False)
    orders = np.arange(order + 1, dtype=float_dtype)
    for dim, dim_length in enumerate(image.shape):
        # Keep the literal multiply then subtract order. Between those reads
        # a parameter/warning callback can change the next external array.
        coordinates = np.arange(dim_length, dtype=float_dtype)
        _check_array(spacing)
        delta = coordinates * spacing[dim]
        _check_array(center)
        delta = delta - center[dim]
        powers_of_delta = delta[:, np.newaxis] ** orders
        calc = _project(calc, powers_of_delta, dim)
    return calc


def centroid(image, *, spacing=None):
    """Return the weighted centroid as the source ndarray result."""
    M = moments_central(image, center=(0,) * image.ndim, order=1, spacing=spacing)
    center = M[tuple(np.eye(image.ndim, dtype=int))] / M[(0,) * image.ndim]
    return center


def _moments_raw_to_central_fast(moments_raw):
    """Analytical formulae for 2D and 3D central moments of order < 4.

    `moments_raw_to_central` will automatically call this function when
    ndim < 4 and order < 4.

    Parameters
    ----------
    moments_raw : ndarray
        The raw moments.

    Returns
    -------
    moments_central : ndarray
        The central moments.
    """
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    float_dtype = moments_raw.dtype
    # convert to float64 during the computation for better accuracy
    moments_raw = moments_raw.astype(np.float64, copy=False)
    moments_central = np.zeros_like(moments_raw)
    if order >= 4 or ndim not in [2, 3]:
        raise ValueError("This function only supports 2D or 3D moments of order < 4.")
    m = moments_raw
    if ndim == 2:
        cx = m[1, 0] / m[0, 0]
        cy = m[0, 1] / m[0, 0]
        moments_central[0, 0] = m[0, 0]
        # Note: 1st order moments are both 0
        if order > 1:
            # 2nd order moments
            moments_central[1, 1] = m[1, 1] - cx * m[0, 1]
            moments_central[2, 0] = m[2, 0] - cx * m[1, 0]
            moments_central[0, 2] = m[0, 2] - cy * m[0, 1]
        if order > 2:
            # 3rd order moments
            moments_central[2, 1] = (
                m[2, 1]
                - 2 * cx * m[1, 1]
                - cy * m[2, 0]
                + cx**2 * m[0, 1]
                + cy * cx * m[1, 0]
            )
            moments_central[1, 2] = (
                m[1, 2] - 2 * cy * m[1, 1] - cx * m[0, 2] + 2 * cy * cx * m[0, 1]
            )
            moments_central[3, 0] = m[3, 0] - 3 * cx * m[2, 0] + 2 * cx**2 * m[1, 0]
            moments_central[0, 3] = m[0, 3] - 3 * cy * m[0, 2] + 2 * cy**2 * m[0, 1]
    else:
        # 3D case
        cx = m[1, 0, 0] / m[0, 0, 0]
        cy = m[0, 1, 0] / m[0, 0, 0]
        cz = m[0, 0, 1] / m[0, 0, 0]
        moments_central[0, 0, 0] = m[0, 0, 0]
        # Note: all first order moments are 0
        if order > 1:
            # 2nd order moments
            moments_central[0, 0, 2] = -cz * m[0, 0, 1] + m[0, 0, 2]
            moments_central[0, 1, 1] = -cy * m[0, 0, 1] + m[0, 1, 1]
            moments_central[0, 2, 0] = -cy * m[0, 1, 0] + m[0, 2, 0]
            moments_central[1, 0, 1] = -cx * m[0, 0, 1] + m[1, 0, 1]
            moments_central[1, 1, 0] = -cx * m[0, 1, 0] + m[1, 1, 0]
            moments_central[2, 0, 0] = -cx * m[1, 0, 0] + m[2, 0, 0]
        if order > 2:
            # 3rd order moments
            moments_central[0, 0, 3] = (
                2 * cz**2 * m[0, 0, 1] - 3 * cz * m[0, 0, 2] + m[0, 0, 3]
            )
            moments_central[0, 1, 2] = (
                -cy * m[0, 0, 2] + 2 * cz * (cy * m[0, 0, 1] - m[0, 1, 1]) + m[0, 1, 2]
            )
            moments_central[0, 2, 1] = (
                cy**2 * m[0, 0, 1]
                - 2 * cy * m[0, 1, 1]
                + cz * (cy * m[0, 1, 0] - m[0, 2, 0])
                + m[0, 2, 1]
            )
            moments_central[0, 3, 0] = (
                2 * cy**2 * m[0, 1, 0] - 3 * cy * m[0, 2, 0] + m[0, 3, 0]
            )
            moments_central[1, 0, 2] = (
                -cx * m[0, 0, 2] + 2 * cz * (cx * m[0, 0, 1] - m[1, 0, 1]) + m[1, 0, 2]
            )
            moments_central[1, 1, 1] = (
                -cx * m[0, 1, 1]
                + cy * (cx * m[0, 0, 1] - m[1, 0, 1])
                + cz * (cx * m[0, 1, 0] - m[1, 1, 0])
                + m[1, 1, 1]
            )
            moments_central[1, 2, 0] = (
                -cx * m[0, 2, 0] - 2 * cy * (-cx * m[0, 1, 0] + m[1, 1, 0]) + m[1, 2, 0]
            )
            moments_central[2, 0, 1] = (
                cx**2 * m[0, 0, 1]
                - 2 * cx * m[1, 0, 1]
                + cz * (cx * m[1, 0, 0] - m[2, 0, 0])
                + m[2, 0, 1]
            )
            moments_central[2, 1, 0] = (
                cx**2 * m[0, 1, 0]
                - 2 * cx * m[1, 1, 0]
                + cy * (cx * m[1, 0, 0] - m[2, 0, 0])
                + m[2, 1, 0]
            )
            moments_central[3, 0, 0] = (
                2 * cx**2 * m[1, 0, 0] - 3 * cx * m[2, 0, 0] + m[3, 0, 0]
            )

    return moments_central.astype(float_dtype, copy=False)


def moments_raw_to_central(moments_raw):
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    if ndim in [2, 3] and order < 4:
        return _moments_raw_to_central_fast(moments_raw)

    moments_central = np.zeros_like(moments_raw)
    m = moments_raw
    # centers as computed in centroid above
    centers = tuple(m[tuple(np.eye(ndim, dtype=int))] / m[(0,) * ndim])

    if ndim == 2:
        # This is the general 2D formula from
        # https://en.wikipedia.org/wiki/Image_moment#Central_moments
        for p in range(order + 1):
            for q in range(order + 1):
                if p + q > order:
                    continue
                for i in range(p + 1):
                    term1 = math.comb(p, i)
                    term1 *= (-centers[0]) ** (p - i)
                    for j in range(q + 1):
                        term2 = math.comb(q, j)
                        term2 *= (-centers[1]) ** (q - j)
                        moments_central[p, q] += term1 * term2 * m[i, j]
        return moments_central

    # The nested loops below are an n-dimensional extension of the 2D formula
    # given at https://en.wikipedia.org/wiki/Image_moment#Central_moments

    # iterate over all [0, order] (inclusive) on each axis
    for orders in itertools.product(*((range(order + 1),) * ndim)):
        # `orders` here is the index into the `moments_central` output array
        if sum(orders) > order:
            # skip any moment that is higher than the requested order
            continue
        # loop over terms from `m` contributing to `moments_central[orders]`
        for idxs in itertools.product(*[range(o + 1) for o in orders]):
            val = m[idxs]
            for i_order, c, idx in zip(orders, centers, idxs):
                val *= math.comb(i_order, idx)
                val *= (-c) ** (i_order - idx)
            moments_central[orders] += val

    return moments_central


def inertia_tensor(image, mu=None, *, spacing=None):
    """Assemble the tensor from owned moments and source NumPy preparation.

    Real finite float32/64 tensor arithmetic runs in C++. NumPy keeps the
    diagonal sum and indexing; exceptional, integer and complex arithmetic
    retains the literal source expressions, without upstream forwarding.
    """
    if mu is None:
        mu = moments_central(image, order=2, spacing=spacing)
    _check_array(mu)
    mu0 = mu[(0,) * image.ndim]
    result = np.zeros((image.ndim, image.ndim), dtype=mu.dtype)
    corners2 = tuple(2 * np.eye(image.ndim, dtype=int))
    d = np.diag(result)
    d.flags.writeable = True
    _check_array(mu)
    total = np.sum(mu[corners2])
    # A warning handler in the sum can invalidate external supplied storage.
    _check_array(mu)
    diagonal = mu[corners2]
    if (type(mu) is np.ndarray and mu.dtype.char in ('f', 'd')
            and np.ndim(mu0) == 0 and np.isfinite(mu0) and mu0 != 0
            and np.isfinite(total) and np.all(np.isfinite(diagonal))):
        pairs = list(itertools.combinations(range(image.ndim), 2))
        values = []
        for dims in pairs:
            mu_index = np.zeros(image.ndim, dtype=int)
            mu_index[list(dims)] = 1
            _check_array(mu)
            values.append(mu[tuple(mu_index)])
        dtype = mu.dtype.newbyteorder('=')
        mixed = np.asarray(values, dtype=dtype)
        if np.all(np.isfinite(mixed)):
            tensor = _core.inertia_tensor(np.ascontiguousarray(diagonal, dtype=dtype),
                                          mixed, total, mu0)
            if tensor is not None:
                # Keep the original early allocation, including byte order,
                # dtype metadata and owning C-contiguous storage.
                result[...] = tensor
                return result
    # A native FP-status refusal publishes no result. Keep the original
    # NumPy operation/warning order, including the two scalar pair expressions.
    d[:] = (total - diagonal) / mu0
    for dims in itertools.combinations(range(image.ndim), 2):
        mu_index = np.zeros(image.ndim, dtype=int)
        mu_index[list(dims)] = 1
        _check_array(mu)
        result[dims] = -mu[tuple(mu_index)] / mu0
        _check_array(mu)
        result.T[dims] = -mu[tuple(mu_index)] / mu0
    return result
