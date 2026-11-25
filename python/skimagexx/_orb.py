"""Oriented FAST detection and rBRIEF extraction with native octave kernels."""

import math

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _ordinary_storage
from ._resize import _scipy_profile
from .filters import _gaussian_cached_coefficients


_NO_FEATURES = (
    "ORB found no features. Try passing in an image containing "
    "greater intensity contrasts between adjacent pixels."
)
_DTYPES = (
    np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32,
    np.uint64, np.int64, np.float16, np.float32, np.float64,
)
_OPTIONS = ("downscale", "n_scales", "n_keypoints", "fast_n", "fast_threshold", "harris_k")


def _ordinary_array(value):
    return type(value) is np.ndarray and value.dtype.metadata is None and _ordinary_storage(value)


def _scalar(value, lower, upper):
    return type(value) in (int, float) and lower <= value <= upper


class ORB:
    """Detect oriented FAST keypoints and compute rotated BRIEF descriptors."""

    def __init__(self, downscale=1.2, n_scales=8, n_keypoints=500, fast_n=9,
                 fast_threshold=0.08, harris_k=0.04):
        self.downscale = downscale
        self.n_scales = n_scales
        self.n_keypoints = n_keypoints
        self.fast_n = fast_n
        self.fast_threshold = fast_threshold
        self.harris_k = harris_k
        self.keypoints = None
        self.scales = None
        self.responses = None
        self.orientations = None
        self.descriptors = None

    def _delegate(self, method, *args):
        original = fallback("feature", "ORB", **{name: getattr(self, name) for name in _OPTIONS})
        vars(original).update(vars(self))
        try:
            return getattr(original, method)(*args)
        finally:
            vars(self).update(vars(original))

    def _build_pyramid(self, image):
        return self._delegate("_build_pyramid", image)

    def _detect_octave(self, octave_image):
        return self._delegate("_detect_octave", octave_image)

    def _extract_octave(self, octave_image, keypoints, orientations):
        return self._delegate("_extract_octave", octave_image, keypoints, orientations)

    def _native_pyramid(self, image):
        if type(image) in (list, tuple):
            image = np.asarray(image)
        if not _ordinary_array(image) or image.dtype.type not in _DTYPES:
            return None
        if image.size == 0:
            raise ValueError("The parameter `image` cannot be an empty array")
        if image.ndim != 2 or min(image.shape) == 1:
            raise ValueError("The parameter `image` must be a 2-dimensional array")
        if (not image.flags.writeable and image.flags.c_contiguous
                and image.dtype in (np.float32, np.float64)):
            return None
        if type(self.downscale) in (int, float) and self.downscale <= 1:
            raise ValueError("scale factor must be greater than 1")
        if (not _scalar(self.downscale, 1, 16) or type(self.n_scales) is not int
                or np.geterr()["under"] != "ignore"):
            return None
        dtype = np.dtype(np.float32) if image.dtype.type is np.float16 else image.dtype.newbyteorder("=")
        current = _core.orb_image(np.require(image, dtype=dtype, requirements=("C", "A")))
        if current is None:
            return None

        pyramid = [current]
        sigma = 2 * self.downscale / 6.0
        weights = _gaussian_cached_coefficients(sigma, int(4 * sigma + 0.5))
        legacy_zoom, _ = _scipy_profile()
        octave = 0
        while octave != self.n_scales - 1:
            octave += 1
            shape = tuple(math.ceil(edge / self.downscale) for edge in current.shape)
            if shape == current.shape:
                break
            current = _core.orb_reduce(current, *shape, weights, legacy_zoom)
            pyramid.append(current)
        return pyramid

    def _native_detect_octave(self, image):
        if (type(self.fast_n) is not int or not -128 <= self.fast_n <= 127
                or not _scalar(self.fast_threshold, -4096, 4096)
                or not _scalar(self.harris_k, -4096, 4096)):
            return None
        weights = _gaussian_cached_coefficients(1, 4)
        return _core.orb_detect_octave(image, self.fast_n, self.fast_threshold, self.harris_k, weights)

    def _native_extract_octave(self, image, keypoints, orientations):
        if (orientations.dtype != image.dtype or not np.all(np.isfinite(orientations))):
            return None
        coords = np.ascontiguousarray(keypoints, dtype=np.float64)
        angles = np.ascontiguousarray(orientations)
        return _core.orb_extract_octave(image, coords, angles)

    def _detect(self, image, describe):
        method = "detect_and_extract" if describe else "detect"
        pyramid = self._native_pyramid(image)
        if pyramid is None or type(self.n_keypoints) is not int:
            return self._delegate(method, image)

        keypoints = []
        orientations = []
        responses = []
        scales = []
        descriptors = []
        for octave, level in enumerate(pyramid):
            if min(level.shape) == 1:
                break
            detected = self._native_detect_octave(level)
            if detected is None:
                return self._delegate(method, image)
            points, angles, scores = detected
            scale = self.downscale**octave
            if describe:
                if len(points) == 0:
                    keypoints.append(points)
                    responses.append(scores)
                    descriptors.append(np.zeros((0, 256), dtype=bool))
                    continue
                extracted = self._native_extract_octave(level, points, angles)
                if extracted is None:
                    return self._delegate(method, image)
                bits, mask = extracted
                descriptors.append(bits)
                points, angles, scores = points[mask], angles[mask], scores[mask]
                scales.append(scale * np.ones(len(points), dtype=np.intp))
            else:
                scales.append(np.full(len(points), scale, dtype=level.dtype))
            keypoints.append(points * scale)
            orientations.append(angles)
            responses.append(scores)

        if describe and not scales:
            raise RuntimeError(_NO_FEATURES)
        points = np.vstack(keypoints)
        angles = np.hstack(orientations)
        scores = np.hstack(responses)
        scale_array = np.hstack(scales)
        if describe:
            bits = np.vstack(descriptors).view(bool)
        if len(points) >= self.n_keypoints:
            indices = scores.argsort()[::-1][:self.n_keypoints]
            points, angles, scores, scale_array = (
                array[indices] for array in (points, angles, scores, scale_array)
            )
            if describe:
                bits = bits[indices]
        self.keypoints = points
        self.orientations = angles
        self.responses = scores
        self.scales = scale_array
        if describe:
            self.descriptors = bits

    def detect(self, image):
        """Detect oriented FAST keypoints and their corresponding scales."""
        return self._detect(image, False)

    def extract(self, image, keypoints, scales, orientations):
        """Extract rotated BRIEF descriptors for the supplied keypoints."""
        pyramid = self._native_pyramid(image)
        if (pyramid is None or not all(_ordinary_array(a) for a in (keypoints, scales, orientations))
                or keypoints.ndim != 2 or keypoints.shape[1] != 2
                or scales.shape != (len(keypoints),) or orientations.shape != scales.shape
                or keypoints.dtype not in (np.float16, np.float32, np.float64)
                or scales.dtype.kind not in "fiub"
                or orientations.dtype != pyramid[0].dtype):
            return self._delegate("extract", image, keypoints, scales, orientations)

        octaves = (np.log(scales) / np.log(self.downscale)).astype(np.intp)
        descriptors = []
        masks = []
        for octave, level in enumerate(pyramid):
            selected = octaves == octave
            if not np.any(selected):
                continue
            points = keypoints[selected]
            points /= self.downscale**octave
            extracted = self._native_extract_octave(level, points, orientations[selected])
            if extracted is None:
                return self._delegate("extract", image, keypoints, scales, orientations)
            bits, mask = extracted
            descriptors.append(bits)
            masks.append(mask)
        self.descriptors = np.vstack(descriptors).view(bool)
        self.mask_ = np.hstack(masks)

    def detect_and_extract(self, image):
        """Detect keypoints and extract their rotated BRIEF descriptors."""
        return self._detect(image, True)
