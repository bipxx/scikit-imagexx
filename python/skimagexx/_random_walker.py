"""Native Jacobi-preconditioned random walker for ordinary grayscale arrays.

Interface and warning text follow scikit-image 0.26 (BSD-3-Clause,
third_party/scikit-image-LICENSE.txt).
"""
import warnings

import numpy as np

from . import _core
from ._compat import fallback


_WIDE_FLOAT32_GRADIENTS = np.lib.NumpyVersion(np.__version__) >= "2.0.0"


def _plain_number(value):
    return type(value) is float or (type(value) is int and abs(value) <= 2**53)


def random_walker(
    data, labels, beta=130, mode="cg_j", tol=1.0e-3, copy=True,
    return_full_prob=False, spacing=None, *, prob_tol=1e-3, channel_axis=None,
):
    """Segment a grayscale image from positive marker labels by graph diffusion."""
    if (
        type(data) is np.ndarray and type(labels) is np.ndarray
        and channel_axis is None and (mode is None or type(mode) is str and mode == "cg_j")
        and copy is True and type(return_full_prob) is bool
        and all(_plain_number(value) for value in (beta, tol, prob_tol))
    ):
        if spacing is None:
            native_spacing = (1.0, 1.0, 1.0)
        elif (
            type(spacing) in (list, tuple) and len(spacing) == data.ndim
            and len(spacing) in (2, 3) and all(_plain_number(value) for value in spacing)
        ):
            native_spacing = tuple(spacing) + (1.0,) * (3 - len(spacing))
        else:
            native_spacing = None
        # NumPy 1 can keep float32 edge gradients and sparse CG arithmetic
        # in single precision when dividing by a spacing scalar. The native
        # solver widens the gradients and uses a double precision Laplacian.
        wide_gradients = data.dtype.type is not np.float32 or _WIDE_FLOAT32_GRADIENTS
        if native_spacing is not None and wide_gradients:
            result = _core.random_walker(
                data, labels, beta, tol, prob_tol, native_spacing, return_full_prob,
            )
            if result is not None:
                output, convergence_warning, probability_warning = result
                if convergence_warning:
                    warnings.warn(
                        "Conjugate gradient convergence to tolerance not achieved. "
                        "Consider decreasing beta to improve system conditionning.",
                        stacklevel=2,
                    )
                if probability_warning:
                    warnings.warn(
                        "The probability range is outside [0, 1] given the tolerance "
                        "`prob_tol`. Consider decreasing `beta` and/or decreasing `tol`.",
                        stacklevel=2,
                    )
                return output
    return fallback(
        "segmentation", "random_walker", data, labels, beta=beta, mode=mode,
        tol=tol, copy=copy, return_full_prob=return_full_prob, spacing=spacing,
        prob_tol=prob_tol, channel_axis=channel_axis,
    )
