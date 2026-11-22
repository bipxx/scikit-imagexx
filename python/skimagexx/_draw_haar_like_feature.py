"""Bounded native Haar rectangle drawing and explicit NumPy RNG provider."""
import numpy as np

from . import _core
from ._compat import fallback

_INTEGER_TYPES = (int, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
                  np.int64, np.uint64)
_RNG_TYPES = (*_INTEGER_TYPES, np.random.Generator, np.random.SeedSequence, type(None))


def _palette(value):
    return (type(value) in (tuple, list) and len(value) == 3
            and all(type(x) in (int, float) and (x == 0 or 2.0**-24 <= abs(x) <= 8) for x in value))


def draw_haar_like_feature(
    image, r, c, width, height, feature_coord,
    color_positive_block=(1.0, 0.0, 0.0), color_negative_block=(0.0, 1.0, 0.0),
    alpha=0.5, max_n_features=None, rng=None,
):
    """Overlay ordered Haar rectangles on a fresh three-channel image.

    Concrete C-order bool, uint8, float32 and float64 image storage uses
    the native overlay when geometry and colors are bounded. A sampled
    subset uses NumPy's generator and choice implementation. Other storage,
    geometry, palettes and protocol objects use compatibility. The native
    blend admits alpha 0, 1, or the interval [0.25, 0.75].
    """
    sampled = max_n_features is not None
    if (type(image) is np.ndarray and type(rng) in _RNG_TYPES
        and type(feature_coord) in (tuple, list, np.ndarray)
        and type(alpha) in (int,float) and (alpha in (0, 1) or .25 <= alpha <= .75)
        and _palette(color_positive_block) and _palette(color_negative_block)
        and (not sampled or (type(feature_coord) is np.ndarray
                            and type(max_n_features) in _INTEGER_TYPES))):
        # Inspect only concrete geometry before consuming RNG or calling a
        # provider. The source never reads width or height.
        prepared = _core._prepare_haar_drawing(image, r, c, feature_coord)
        if prepared is not None:
            generator = np.random.default_rng(rng)
            order = generator.choice(prepared.size, size=max_n_features, replace=False) if sampled else None
            return _core._draw_haar_prepared(
                prepared, order, color_positive_block, color_negative_block, alpha,
            )
    return fallback(
        'feature', 'draw_haar_like_feature', image, r, c, width, height, feature_coord,
        color_positive_block=color_positive_block, color_negative_block=color_negative_block,
        alpha=alpha, max_n_features=max_n_features, rng=rng,
    )
