"""Color API compatible with scikit-image."""
import numpy as np
from . import _core
from ._compat import fallback, module_attribute, native_image

__all__ = ['rgb2gray', 'gray2rgb']


def rgb2gray(rgb, *, channel_axis=-1):
    if isinstance(channel_axis, (int, np.integer)) and isinstance(rgb, np.ndarray) and rgb.ndim == 3:
        moved = np.moveaxis(rgb, channel_axis, -1)
        if native_image(moved, rgb=True):
            return _core.rgb2gray(moved)
    return fallback('color', 'rgb2gray', rgb, channel_axis=channel_axis)


def gray2rgb(image, *, channel_axis=-1):
    if native_image(image) and isinstance(channel_axis, (int, np.integer)) and -3 <= channel_axis < 3:
        return np.moveaxis(_core.gray2rgb(image), -1, channel_axis)
    return fallback('color', 'gray2rgb', image, channel_axis=channel_axis)


def __getattr__(name):
    return module_attribute('color', name)
