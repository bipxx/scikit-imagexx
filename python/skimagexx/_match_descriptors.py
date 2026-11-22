"""Native matching for bounded canonical Boolean descriptors."""

from . import _core
from ._compat import fallback


_INF = float("inf")


def match_descriptors(
    descriptors1, descriptors2, metric=None, p=2, max_distance=_INF,
    cross_check=True, max_ratio=1.0,
):
    """Return matching left and right descriptor indices in ascending left order.

    Canonical C-order Boolean descriptors can use native Hamming matching.
    The configured scikit-image fallback handles other descriptor formats,
    metrics and unsupported threshold or array configurations.
    """
    if (
        (metric is None or type(metric) is str and metric == "hamming")
        and type(p) is int and p == 2
        and type(cross_check) is bool
        and type(max_distance) is float
        and type(max_ratio) is float
    ):
        result = _core.match_descriptors_bool(
            descriptors1, descriptors2, max_distance, cross_check, max_ratio,
        )
        if result is not None:
            return result
    return fallback(
        "feature", "match_descriptors", descriptors1, descriptors2, metric=metric,
        p=p, max_distance=max_distance, cross_check=cross_check, max_ratio=max_ratio,
    )
