"""Native quadratic interpolation and banded snake evolution."""
from . import _core
from ._compat import fallback


def active_contour(
    image,
    snake,
    alpha=0.01,
    beta=0.1,
    w_line=0.0,
    w_edge=1,
    gamma=0.01,
    max_px_move=1.0,
    max_num_iter=2500,
    convergence=0.1,
    *,
    boundary_condition='periodic',
):
    """Fit an open or closed snake to edges and intensities in a 2D image."""
    args = (image, snake, alpha, beta, w_line, w_edge, gamma, max_px_move, max_num_iter, convergence)
    result = _core.active_contour_finite(*args, boundary_condition)
    if result is not None:
        return result
    return fallback('segmentation', 'active_contour', *args, boundary_condition=boundary_condition)
