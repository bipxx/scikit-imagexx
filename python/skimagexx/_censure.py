"""Stateful center-surround extrema detection."""

import math

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _ordinary_storage
from .util import img_as_float

try:
    from skimage.feature.util import FeatureDetector as _FeatureDetector
except ModuleNotFoundError as error:
    if error.name != 'skimage':
        raise

    class _FeatureDetector:
        def __init__(self):
            self.keypoints_ = np.array([])

        def detect(self, image):
            raise NotImplementedError()


_MODES = ('dob', 'octagon', 'star')
_SCALE_LIMITS = (64, 11, 12)
_SUPPORTED_DTYPES = {
    np.bool_, np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64,
}


def _threshold(value):
    return (type(value) is int and abs(value) <= 2**53
            or type(value) is float and math.isfinite(value))


class CENSURE(_FeatureDetector):
    """Detect grayscale keypoints as center-surround extrema at multiple scales."""

    def __init__(self, min_scale=1, max_scale=7, mode='DoB', non_max_threshold=0.15, line_threshold=10):
        mode = mode.lower()
        if mode not in _MODES:
            raise ValueError("`mode` must be one of 'DoB', 'Octagon', 'STAR'.")
        if min_scale < 1 or max_scale < 1 or max_scale - min_scale < 2:
            raise ValueError('The scales must be >= 1 and the number of scales should be >= 3.')
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mode = mode
        self.non_max_threshold = non_max_threshold
        self.line_threshold = line_threshold
        self.keypoints = None
        self.scales = None

    def detect(self, image):
        """Populate keypoints in row-column order and their corresponding scales."""
        array = np.asanyarray(image)
        if not array.size:
            raise ValueError('The parameter `image` cannot be an empty array')
        if array.ndim != 2 or min(array.shape) == 1:
            raise ValueError('The parameter `image` must be a 2-dimensional array')

        lo = self.min_scale
        hi = self.max_scale
        if (self.mode == 'dob' and (type(hi) is int or isinstance(hi, np.integer))
                and hi >= 1 and min(array.shape) <= 4 * int(hi)):
            raise ValueError('CENSURE DoB image is too small for max_scale')

        if (type(array) is np.ndarray and _ordinary_storage(array)
                and array.dtype.type in _SUPPORTED_DTYPES and array.dtype.isnative
                and array.dtype.metadata is None and max(array.shape) <= 32768
                and array.size <= 8 * 1024 * 1024
                and type(lo) is int and type(hi) is int
                and type(self.mode) is str and self.mode in _MODES
                and 1 <= lo <= hi - 2 and hi <= _SCALE_LIMITS[_MODES.index(self.mode)]
                and _threshold(self.non_max_threshold) and _threshold(self.line_threshold)
                and abs(self.line_threshold) <= 1e12):
            work = img_as_float(array)
            if work.dtype == np.float16:
                work = work.astype(np.float32)
            work = np.require(work, requirements=('C', 'A'))
            result = _core.censure_detect(
                work, lo, hi, _MODES.index(self.mode), self.non_max_threshold, self.line_threshold,
            )
            if result is not None:
                self.keypoints, self.scales = result
                return

        original = fallback('feature', 'CENSURE')
        original.__dict__.update(vars(self))
        original.detect(image)
        self.keypoints = original.keypoints
        self.scales = original.scales
