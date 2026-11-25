"""BRIEF descriptor extraction with a native intensity comparison kernel."""

from copy import deepcopy

import numpy as np

from . import _core
from ._compat import fallback
from .filters import gaussian
from .util import img_as_float

try:
    from skimage.feature.util import DescriptorExtractor
except ModuleNotFoundError as error:
    if error.name != "skimage":
        raise

    class DescriptorExtractor:
        """Descriptor extractor base for installations without scikit-image."""

        def __init__(self):
            self.descriptors_ = np.array([])

        def extract(self, image, keypoints):
            raise NotImplementedError


class BRIEF(DescriptorExtractor):
    """Extract binary descriptors by comparing pixel pairs near each keypoint."""

    def __init__(self, descriptor_size=256, patch_size=49, mode="normal", sigma=1, rng=1):
        """Configure descriptor width, patch sampling, smoothing and random state."""
        mode = mode.lower()
        if mode not in ("normal", "uniform"):
            raise ValueError("`mode` must be 'normal' or 'uniform'.")
        self.descriptor_size = descriptor_size
        self.patch_size = patch_size
        self.mode = mode
        self.sigma = sigma
        if isinstance(rng, np.random.Generator):
            bit_generator = rng.bit_generator
            seed_sequence = getattr(bit_generator, "seed_seq", bit_generator._seed_seq)
            (seed,) = seed_sequence.spawn(1)
            rng = type(bit_generator)(seed)
        self.rng = np.random.default_rng(rng)
        self.descriptors = None
        self.mask = None

    def extract(self, image, keypoints):
        """Set the Boolean mask and descriptors for row-column keypoints."""
        checked = np.asanyarray(image)
        if checked.size == 0:
            raise ValueError("The parameter `image` cannot be an empty array")
        if checked.ndim != 2:
            raise ValueError("The parameter `image` must be a 2-dimensional array")
        rng = deepcopy(self.rng)
        if type(checked) is not np.ndarray:
            image = fallback("feature.brief", "_prepare_grayscale_input_2D", image)
        else:
            image = np.squeeze(image)
            if image.ndim != 2:
                raise ValueError("The parameter `image` must be a 2-dimensional array")
            if not image.dtype.isnative:
                image = image.astype(image.dtype.newbyteorder("="))
            image = img_as_float(image)
            if image.dtype == np.float16:
                image = image.astype(np.float32)
            elif image.dtype.kind == "f" and image.dtype.type not in (np.float32, np.float64):
                image = image.astype(np.float64)
        if type(self.sigma) is int and self.sigma == 0:
            image = np.require(image, requirements=["C", "A"])
        else:
            image = np.ascontiguousarray(gaussian(image, sigma=self.sigma, mode="reflect"))

        width = self.descriptor_size
        patch = self.patch_size
        distance = patch // 2
        low = -(patch - 2) // 2
        if self.mode == "normal":
            offsets = ((patch / 5.0) * rng.standard_normal(width * 8)).astype(np.int32)
            offsets = offsets[(offsets < distance) & (offsets > low)]
            first = offsets[:2 * width].reshape(width, 2)
            second = offsets[2 * width:4 * width].reshape(width, 2)
        elif self.mode == "uniform":
            offsets = rng.integers(low, distance + 1, (2 * width, 2)).astype(np.int32)
            first, second = np.split(offsets, 2)
        else:
            raise UnboundLocalError("cannot access local variable 'pos1' where it is not associated with a value")

        rows, cols = image.shape
        row = keypoints[:, 0]
        col = keypoints[:, 1]
        self.mask = ((distance - 1 < row) & (row < rows - distance + 1)
                     & (distance - 1 < col) & (col < cols - distance + 1))
        selected = np.asarray(keypoints[self.mask, :], dtype=np.int64, order="C")
        self.descriptors = np.zeros((selected.shape[0], width), dtype=bool)
        _core.brief_sample(image, selected, first, second, self.descriptors)
