"""Native descriptor paths with upstream dispatch for other inputs."""

import math

import numpy as np

from . import _core
from ._compat import fallback


_LBP_METHODS = {"default": 0, "ror": 1, "uniform": 2, "nri_uniform": 3, "var": 4}
_HOG_NORMS = {"L1": 0, "L1-sqrt": 1, "L2": 2, "L2-Hys": 3}
_DAISY_NORMS = {"off": 0, "l1": 1, "l2": 2, "daisy": 3}


def _integer(value):
    return type(value) is int and -(1 << 31) <= value < (1 << 31)


def _real(value):
    return type(value) in (int, float) and -1e100 <= value <= 1e100


def _pair(value):
    return type(value) in (tuple, list) and len(value) == 2 and all(map(_integer, value))


def local_binary_pattern(image, P, R, method="default"):
    if type(image) is np.ndarray and _integer(P) and _real(R) and type(method) is str:
        code = _LBP_METHODS.get(method.lower())
        if code is not None:
            result = _core.local_binary_pattern_u8(image, P, R, code)
            if result is not None:
                return result
    return fallback("feature", "local_binary_pattern", image, P, R, method=method)


def multiblock_lbp(int_image, r, c, width, height):
    if type(int_image) is np.ndarray:
        if type(r) is type(c) is type(width) is type(height) is int:
            if 0 <= r < 1 << 31 and 0 <= c < 1 << 31 and 0 < width < 1 << 31 and 0 < height < 1 << 31:
                result = _core.multiblock_lbp_native(int_image, r, c, width, height)
                if result is not None:
                    return result
        # The upstream Cython kernel disables bounds checking. Keep unsafe
        # geometry out of it even when a dtype or memory layout uses fallback.
        if all(isinstance(value, (int, np.integer)) for value in (r, c, width, height)):
            if int_image.ndim == 2 and (
                r < 0 or c < 0 or width <= 0 or height <= 0
                or int(r) + 3 * int(height) > int_image.shape[0]
                or int(c) + 3 * int(width) > int_image.shape[1]
            ):
                raise ValueError("the 3x3 LBP grid must fit inside the integral image")
    return fallback("feature", "multiblock_lbp", int_image, r, c, width, height)


def hog(
    image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
    block_norm="L2-Hys", visualize=False, transform_sqrt=False, feature_vector=True,
    *, channel_axis=None,
):
    if (
        type(image) is np.ndarray
        and ((channel_axis is None and image.ndim == 2)
             or (type(channel_axis) is int and channel_axis in (-1, 2) and image.ndim == 3))
        and _integer(orientations) and _pair(pixels_per_cell) and _pair(cells_per_block)
        and type(block_norm) is str and block_norm in _HOG_NORMS
        and type(visualize) is type(transform_sqrt) is type(feature_vector) is bool
    ):
        result = _core.hog_native(
            image, orientations, *pixels_per_cell, *cells_per_block, _HOG_NORMS[block_norm],
            visualize, transform_sqrt, feature_vector,
        )
        if result is not None:
            return result
    return fallback(
        "feature", "hog", image, orientations, pixels_per_cell, cells_per_block,
        block_norm, visualize, transform_sqrt, feature_vector, channel_axis=channel_axis,
    )


def _real_sequence(value):
    return type(value) in (list, tuple) and len(value) <= 65 and all(map(_real, value))


def daisy(
    image, step=4, radius=15, rings=3, histograms=8, orientations=8,
    normalization="l1", sigmas=None, ring_radii=None, visualize=False,
):
    if (
        type(image) is np.ndarray and visualize is False
        and all(map(_integer, (step, radius, rings, histograms, orientations)))
        and type(normalization) is str and normalization in _DAISY_NORMS
        and (ring_radii is None or (_real_sequence(ring_radii) and ring_radii and _integer(ring_radii[-1])))
        and (sigmas is None or (_real_sequence(sigmas) and len(sigmas) >= 2))
        and (sigmas is None or ring_radii is None or len(sigmas) == len(ring_radii) + 1)
    ):
        effective_radius = radius if ring_radii is None else ring_radii[-1]
        count = len(sigmas) - 1 if sigmas is not None else (
            rings if ring_radii is None else len(ring_radii)
        )
        if 0 < count <= 64:
            scales = sigmas if sigmas is not None else [
                effective_radius * i / (2 * count) for i in range(1, count + 1)
            ]
            radii = ring_radii if ring_radii is not None else [
                effective_radius * i / count for i in range(1, count + 1)
            ]
            result = _core.daisy_native(
                image, step, effective_radius, histograms, orientations,
                _DAISY_NORMS[normalization], [scales[0], *scales[:count]], radii,
            )
            if result is not None:
                return result
    return fallback(
        "feature", "daisy", image, step, radius, rings, histograms, orientations,
        normalization, sigmas, ring_radii, visualize,
    )


def match_template(image, template, pad_input=False, mode="constant", constant_values=0):
    if (
        type(image) is type(template) is np.ndarray
        and type(pad_input) is bool and type(mode) is str and mode == "constant"
        and _real(constant_values)
    ):
        result = _core.match_template_native(image, template, pad_input, constant_values)
        if result is not None:
            return result
    return fallback(
        "feature", "match_template", image, template, pad_input, mode, constant_values,
    )


def multiscale_basic_features(
    image, intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=16,
    num_sigma=None, workers=None, *, channel_axis=None,
):
    if (
        type(image) is np.ndarray
        and ((channel_axis is None and image.ndim == 2)
             or (type(channel_axis) is int and channel_axis in (-1, 2) and image.ndim == 3))
        and type(intensity) is type(edges) is type(texture) is bool
        and _real(sigma_min) and _real(sigma_max) and 0 < sigma_min <= 256 and 0 < sigma_max <= 256
        and (num_sigma is None or _integer(num_sigma))
        and (workers is None or (type(workers) is int and workers == 1))
    ):
        start, stop = math.log2(sigma_min), math.log2(sigma_max)
        count = int(stop - start + 1) if num_sigma is None else num_sigma
        if 0 < count <= 64:
            sigmas = np.logspace(start, stop, num=count, base=2).tolist()
            result = _core.multiscale_basic_features_native(image, intensity, edges, texture, sigmas)
            if result is not None:
                return result
    return fallback(
        "feature", "multiscale_basic_features", image, intensity, edges, texture,
        sigma_min, sigma_max, num_sigma, workers, channel_axis=channel_axis,
    )
