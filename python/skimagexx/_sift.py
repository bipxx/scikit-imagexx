"""Native SIFT detection and description with scikit-image class state."""

import math
import operator

import numpy as np

from . import _core
from ._compat import fallback


class SIFT:
    """Detect scale-invariant keypoints and compute SIFT descriptors."""

    def __init__(
        self, upsampling=2, n_octaves=8, n_scales=3, sigma_min=1.6, sigma_in=0.5,
        c_dog=0.04 / 3, c_edge=10, n_bins=36, lambda_ori=1.5, c_max=0.8,
        lambda_descr=6, n_hist=4, n_ori=8,
    ):
        if upsampling not in (1, 2, 4):
            raise ValueError("upsampling must be 1, 2 or 4")
        self.upsampling = upsampling
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.sigma_min = sigma_min / upsampling
        self.sigma_in = sigma_in
        if type(n_scales) is int and n_scales == 3:
            self.c_dog = 1.0 * c_dog
        else:
            self.c_dog = (2 ** (1 / n_scales) - 1) / (2 ** (1 / 3) - 1) * c_dog
        self.c_edge = c_edge
        self.n_bins = n_bins
        self.lambda_ori = lambda_ori
        self.c_max = c_max
        self.lambda_descr = lambda_descr
        self.n_hist = n_hist
        self.n_ori = n_ori
        self.delta_min = 1 / upsampling
        self.float_dtype = None
        self.scalespace_sigmas = None
        self.keypoints = None
        self.positions = None
        self.sigmas = None
        self.scales = None
        self.orientations = None
        self.octaves = None
        self.descriptors = None

    @property
    def deltas(self):
        """The sampling distances of all octaves."""
        return self.delta_min * np.power(
            2, np.arange(self.n_octaves), dtype=self.float_dtype,
        )

    def _fallback(self, method, *args):
        original = fallback("feature.sift", "SIFT")
        original.__dict__.update(self.__dict__)
        try:
            return getattr(original, method)(*args)
        finally:
            self.__dict__.update(original.__dict__)

    def _set_number_of_octaves(self, image_shape):
        maximum = int(math.log2(min(image_shape) * self.upsampling / 12) + 1)
        if maximum < self.n_octaves:
            self.n_octaves = maximum

    def _create_scalespace(self, image):
        return self._fallback("_create_scalespace", image)

    def _inrange(self, a, dim):
        return ((a[:, 0] > 0) & (a[:, 0] < dim[0] - 1)
                & (a[:, 1] > 0) & (a[:, 1] < dim[1] - 1))

    def _find_localize_evaluate(self, dogspace, img_shape):
        return self._fallback("_find_localize_evaluate", dogspace, img_shape)

    def _fit(self, h):
        return (h[0] - h[2]) / (2 * (h[0] + h[2] - 2 * h[1]))

    def _compute_orientation(
        self, positions_oct, scales_oct, sigmas_oct, octaves, gaussian_scalespace,
    ):
        return self._fallback("_compute_orientation", positions_oct, scales_oct, sigmas_oct,
                              octaves, gaussian_scalespace)

    def _rotate(self, row, col, angle):
        cos, sin = math.cos(angle), math.sin(angle)
        return cos * row + sin * col, -sin * row + cos * col

    def _compute_descriptor(self, gradient_space):
        return self._fallback("_compute_descriptor", gradient_space)

    def _preprocess(self, image):
        return self._fallback("_preprocess", image)

    def _options(self, image):
        try:
            upsampling = operator.index(self.upsampling)
            n_octaves = operator.index(self.n_octaves)
            n_scales = operator.index(self.n_scales)
            n_bins = operator.index(self.n_bins)
            n_hist = operator.index(self.n_hist)
            n_ori = operator.index(self.n_ori)
        except TypeError:
            return None
        if (type(image) is not np.ndarray or image.ndim != 2 or min(image.shape) < 3
                or max(image.shape) > 16384 or image.dtype.kind not in "buif"
                or image.dtype.itemsize not in (1, 2, 4, 8)
                or upsampling not in (1, 2, 4) or min(image.shape) * upsampling < 12
                or self.delta_min != 1 / upsampling or not 1 <= n_scales <= 12
                or n_octaves < 1 or not 6 <= n_bins <= 256 or not 1 <= n_hist <= 16
                or not 1 <= n_ori <= 64
                or image.size * upsampling**2 * (n_scales + 3) > (128 << 20)):
            return None
        real_options = (self.sigma_min, self.sigma_in, self.c_dog, self.c_edge,
                        self.lambda_ori, self.c_max, self.lambda_descr)
        if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in real_options):
            return None
        sigma_min, sigma_in, c_dog, c_edge, lambda_ori, c_max, lambda_descr = map(float, real_options)
        if not (0 <= sigma_in <= sigma_min and 0 < sigma_min * upsampling <= 64
                and 0 < c_dog < math.inf and 0 < c_edge < math.inf and 0 < c_max <= 1
                and 0 < lambda_ori <= 64 and 0 < lambda_descr <= 64):
            return None
        n_octaves = min(n_octaves, int(math.log2(min(image.shape) * upsampling / 12) + 1))
        c_dog *= (2 ** (1 / 3) - 1) / (2 ** (1 / n_scales) - 1)
        return (upsampling, n_octaves, n_scales, sigma_min * upsampling, sigma_in,
                c_dog, c_edge, n_bins, lambda_ori, c_max, lambda_descr, n_hist, n_ori)

    def _stored(self):
        if any(type(value) is not np.ndarray for value in (
            self.positions, self.sigmas, self.orientations, self.scales, self.octaves,
        )):
            return None
        if (self.scales.ndim != 1 or self.scales.dtype.kind not in "iu"
                or self.positions.shape != (len(self.scales), 2)
                or any(value.shape != self.scales.shape
                       for value in (self.sigmas, self.orientations, self.octaves))
                or any(value.dtype.kind not in "fiu"
                       for value in (self.positions, self.sigmas, self.orientations, self.octaves))):
            return None
        try:
            return np.column_stack((
                self.positions, self.sigmas, self.orientations, self.scales, self.octaves,
            )).astype(np.float64, copy=False)
        except (TypeError, ValueError):
            return None

    def _run(self, method, image, *, describe=False, extract=False):
        options = self._options(image)
        if options is None:
            return self._fallback(method, image)
        missing_scales = extract and self.scales is None
        stored = self._stored() if extract and not missing_scales else None
        if extract and stored is None and not missing_scales:
            return self._fallback(method, image)
        target_dtype = image.dtype.newbyteorder("=")
        if target_dtype == np.dtype(np.float16):
            target_dtype = np.dtype(np.float32)
        source = np.require(image, dtype=target_dtype, requirements=("C", "A"))
        old_dtype, old_octaves, old_sigmas = self.float_dtype, self.n_octaves, self.scalespace_sigmas
        self.float_dtype = np.float32 if source.dtype == np.dtype(np.float32) else np.float64
        self.n_octaves = options[1]
        sigmas = np.power(2, np.arange(self.n_scales + 3) / self.n_scales) * self.sigma_min
        self.scalespace_sigmas = self.deltas[:, None] / self.deltas[0] * sigmas[None, :]
        if missing_scales:
            raise TypeError("object of type 'NoneType' has no len()")
        try:
            result = _core.sift(source, *options, describe, stored)
        except ValueError:
            self.float_dtype, self.n_octaves, self.scalespace_sigmas = old_dtype, old_octaves, old_sigmas
            return self._fallback(method, image)
        if extract:
            self.descriptors = result
            return
        positions, sigmas, angles, scales, octaves, descriptors, promoted, multiple = result
        if not len(scales):
            raise RuntimeError(
                "SIFT found no features. Try passing in an image containing "
                "greater intensity contrasts between adjacent pixels."
            )
        self.positions = positions if promoted else positions.astype(self.float_dtype, copy=False)
        self.sigmas = sigmas
        self.orientations = angles
        self.scales = scales
        self.octaves = octaves if multiple else octaves.astype(np.float64)
        if describe:
            self.descriptors = descriptors
        self.keypoints = self.positions.round().astype(int)

    def detect(self, image):
        """Detect the keypoints in a two-dimensional image."""
        self._run("detect", image)

    def extract(self, image):
        """Extract descriptors at the previously detected keypoints."""
        self._run("extract", image, extract=True)

    def detect_and_extract(self, image):
        """Detect keypoints and extract their descriptors."""
        self._run("detect_and_extract", image, describe=True)
