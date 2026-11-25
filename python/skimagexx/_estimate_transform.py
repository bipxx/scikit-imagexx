"""Dispatch geometric fitting to the matching public transform class."""

import numpy as np

from ._compat import fallback
from ._geometric import (
    AffineTransform,
    EuclideanTransform,
    ProjectiveTransform,
    SimilarityTransform,
    _center_and_normalize_points,
)


_TRANSFORMS = {
    'affine': AffineTransform,
    'euclidean': EuclideanTransform,
    'projective': ProjectiveTransform,
    'similarity': SimilarityTransform,
    'essential': 'essential',
    'fundamental': 'fundamental',
    'piecewise-affine': 'piecewise-affine',
    'polynomial': 'polynomial',
}


def _positive_determinant(matrix, dimensions):
    if dimensions == 2:
        (a, b), (c, d) = matrix
        return a * d - b * c > 0
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g) > 0


def _fit_rigid(model, src, dst, dimensions):
    # Full-rank, orientation-preserving Umeyama fits have an identity sign
    # matrix. Retain the owned solver for reflected or near-singular inputs.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    centered = src - src_mean
    covariance = (dst - dst_mean).T @ centered / len(src)
    u, singular, vh = np.linalg.svd(covariance)
    if singular[-1] <= 1e-4 * singular[0] or not _positive_determinant(covariance, dimensions):
        return None

    rotation = u @ vh
    scale = 1.0
    if model is SimilarityTransform:
        scale = 1.0 / centered.var(axis=0).sum() * singular.sum()
    matrix = np.eye(dimensions + 1)
    matrix[:-1, -1] = dst_mean - scale * (rotation @ src_mean)
    matrix[:-1, :-1] = scale * rotation
    return model(matrix)


def _fit_linear(model, src, dst, dimensions, weights):
    step = dimensions + 1
    affine = model is AffineTransform
    columns = dimensions * step + (1 if affine else step)
    count = len(src)
    if count * dimensions < columns:
        return None  # The wide system needs the full right nullspace.

    src_matrix, src = _center_and_normalize_points(src)
    dst_matrix, dst = _center_and_normalize_points(dst)
    if not np.all(np.isfinite(src_matrix + dst_matrix)):
        return None

    # Form only the fitted columns. Applying positive finite weights by row
    # avoids allocating and multiplying by a dense diagonal matrix.
    design = np.zeros((dimensions, count, columns))
    for axis, block in enumerate(design):
        start = axis * step
        block[:, start : start + dimensions] = src
        block[:, start + dimensions] = 1
        if not affine:
            block[:, -step:-1] = src * -dst[:, axis, None]
        block[:, -1] = dst[:, axis]
    if weights is not None:
        design *= np.sqrt(weights / np.max(weights))[None, :, None]
    _, singular, vh = np.linalg.svd(design.reshape(-1, columns), full_matrices=False)
    vector = vh[-1]
    if singular[-2] - singular[-1] <= 1e-7 * singular[0] or abs(vector[-1]) <= 1e-7:
        return None

    matrix = np.eye(step)
    size = step * dimensions if affine else step * step - 1
    matrix.ravel()[:size] = -vector[:-1] / vector[-1]
    matrix = np.linalg.inv(dst_matrix) @ matrix
    denormalized = matrix @ src_matrix
    divisor = denormalized[-1, -1]
    if abs(divisor) <= 1e-7 * (np.abs(matrix[-1]) @ np.abs(src_matrix[:, -1])):
        return None
    denormalized /= divisor
    return model(denormalized)


def _try_fast_estimate(model, src, dst, args, kwargs):
    if (type(src) is not np.ndarray or type(dst) is not np.ndarray
            or src.ndim != 2 or src.shape != dst.shape or src.shape[1] not in (2, 3)
            or src.shape[0] <= src.shape[1] or src.dtype != np.float64 or dst.dtype != np.float64
            or not (src.flags.c_contiguous and dst.flags.c_contiguous)):
        return None

    dimensions = src.shape[1]
    rigid = model is EuclideanTransform or model is SimilarityTransform
    weights = None
    if args or kwargs:
        if rigid:
            return None
        if len(args) == 1 and not kwargs:
            weights = args[0]
        elif not args and len(kwargs) == 1 and 'weights' in kwargs:
            weights = kwargs['weights']
        else:
            return None
        if weights is not None and (type(weights) is not np.ndarray or weights.dtype != np.float64
                or weights.shape != (len(src),) or not np.min(weights) > 0):
            return None

    # A trial that would emit an arithmetic warning is replayed through the
    # owned solver so exceptional inputs retain the runtime's normal policy.
    with np.errstate(all='raise'):
        try:
            if rigid:
                return _fit_rigid(model, src, dst, dimensions)
            return _fit_linear(model, src, dst, dimensions, weights)
        except (FloatingPointError, np.linalg.LinAlgError):
            return None


def estimate_transform(ttype, src, dst, *args, **kwargs):
    """Estimate a geometric transformation from corresponding coordinates.

    The case-insensitive transform type selects the model. Relative weights
    may be supplied for affine and projective models, and a polynomial order
    may be supplied for the polynomial model. Extra arguments are forwarded
    to the selected model's ``from_estimate`` method.

    The returned model maps ``src`` to ``dst``. A failed fit returns a falsy
    ``FailedEstimation`` with a diagnostic message.
    """
    ttype = ttype.lower()
    model = _TRANSFORMS.get(ttype)
    if model is None:
        raise ValueError(f"the transformation type '{ttype}' is not implemented")
    if isinstance(model, str) or any(
        isinstance(array, np.ndarray) and array.dtype.metadata is not None
        for array in (src, dst, *args, *kwargs.values())
    ):
        return fallback('transform', 'estimate_transform', ttype, src, dst, *args, **kwargs)
    result = _try_fast_estimate(model, src, dst, args, kwargs)
    if result is not None:
        return result
    try:
        if not (args or kwargs):
            return model.from_estimate(src, dst)
        return model.from_estimate(src, dst, *args, **kwargs)
    except NotImplementedError as error:
        # The dimensionality error from the owned class includes its module.
        local = f'Input for {model} should result in transform of >=2D'
        if error.args == (local,):
            upstream = f"<class 'skimage.transform._geometric.{model.__name__}'>"
            error.args = (f'Input for {upstream} should result in transform of >=2D',)
        raise
