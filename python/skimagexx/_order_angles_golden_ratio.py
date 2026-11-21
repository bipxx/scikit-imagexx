"""Native ordering within a small input range, with first-resume fallback."""
import numpy as np

from . import _core
from ._compat import fallback

_WEAK_SCALARS = int(np.__version__.split('.')[0]) >= 2


def order_angles_golden_ratio(theta):
    """Yield indices into theta in golden-ratio order.

    The native path accepts up to 16 angles in [-360, 360], using zero or
    normal values and round-to-nearest. Other input domains use the configured
    scikit-image fallback on first resume. Once native iteration begins,
    ndarray values are read live and must remain within the native domain.
    """
    carrier = type(theta)
    if carrier is np.ndarray:
        array = theta
    elif carrier in (list, tuple) and len(theta) <= 16 and all(
        type(value) in (bool, int, float) for value in theta
    ):
        array = np.asarray(theta)
    else:
        yield from fallback('transform', 'order_angles_golden_ratio', theta)
        return

    state = _core._order_angles_begin(array, _WEAK_SCALARS)
    if state is None:
        yield from fallback('transform', 'order_angles_golden_ratio', theta)
        return

    yield np.int64(state.first)
    while _core._order_angles_resume(state):
        if carrier is not np.ndarray:
            # Preserve the original carrier's deferred fancy-indexing error.
            theta[state.remaining_indices]
        yield np.int64(_core._order_angles_next(state, theta))