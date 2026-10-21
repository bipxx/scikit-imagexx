"""Color API compatible with scikit-image."""
import numpy as np
from . import _core
from ._compat import fallback, module_attribute, native_image

__all__ = ['rgb2gray', 'gray2rgb', 'rgb2yuv', 'yuv2rgb', 'rgb2yiq', 'yiq2rgb',
           'rgb2ypbpr', 'ypbpr2rgb', 'rgb2ycbcr', 'ycbcr2rgb', 'rgb2ydbdr',
           'ydbdr2rgb', 'rgb2rgbcie', 'rgbcie2rgb', 'rgb2xyz', 'xyz2rgb']


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


_TRANSFORMS = {name: i for i, name in enumerate(__all__[2:])}
_COLOR_TYPES = {np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
                np.int32, np.uint64, np.int64, np.float32, np.float64}


def _transform(name, image, channel_axis):
    if isinstance(image, np.ndarray) and type(image) is not np.ndarray:
        return fallback('color', name, image, channel_axis=channel_axis)
    a = np.asarray(image)
    axis = channel_axis
    if axis is None:
        axis = -1
    elif not np.isscalar(axis):
        if len(axis) > 1:
            raise ValueError('only a single channel axis is currently supported')
        axis = axis[0]
    if name == 'ycbcr2rgb' and axis == -1 and not isinstance(image, np.ndarray):
        # Upstream copies/indexes before _prepare_colorarray for this function.
        image.copy()[..., 0]
    moved = a if axis == -1 else np.moveaxis(a, axis, -1)
    if name == 'ycbcr2rgb' and (moved.ndim == 0 or moved.shape[-1] < 3 or moved.dtype.type in (np.bool_, np.int8)):
        # Offset subtraction precedes upstream shape/type preparation. Check
        # its indexing and scalar-casting errors without touching image pixels.
        for j, amount in enumerate((16,128,128)):
            moved[...,j]
            if moved.dtype.type in (np.bool_, np.int8):
                probe = np.empty(0, dtype=moved.dtype)
                np.subtract(probe, amount, out=probe)
    if moved.shape[-1] != 3:
        raise ValueError(f'the input array must have size 3 along `channel_axis`, got {moved.shape}')
    if moved.dtype.type not in _COLOR_TYPES or not moved.dtype.isnative:
        return fallback('color', name, image, channel_axis=channel_axis)
    result = _core.color_transform(moved, _TRANSFORMS[name])
    return result if axis == -1 else np.moveaxis(result, -1, axis)


def rgb2yuv(rgb, *, channel_axis=-1):
    return _transform('rgb2yuv', rgb, channel_axis)


def yuv2rgb(yuv, *, channel_axis=-1):
    return _transform('yuv2rgb', yuv, channel_axis)


def rgb2yiq(rgb, *, channel_axis=-1):
    return _transform('rgb2yiq', rgb, channel_axis)


def yiq2rgb(yiq, *, channel_axis=-1):
    return _transform('yiq2rgb', yiq, channel_axis)


def rgb2ypbpr(rgb, *, channel_axis=-1):
    return _transform('rgb2ypbpr', rgb, channel_axis)


def ypbpr2rgb(ypbpr, *, channel_axis=-1):
    return _transform('ypbpr2rgb', ypbpr, channel_axis)


def rgb2ycbcr(rgb, *, channel_axis=-1):
    return _transform('rgb2ycbcr', rgb, channel_axis)


def ycbcr2rgb(ycbcr, *, channel_axis=-1):
    return _transform('ycbcr2rgb', ycbcr, channel_axis)


def rgb2ydbdr(rgb, *, channel_axis=-1):
    return _transform('rgb2ydbdr', rgb, channel_axis)


def ydbdr2rgb(ydbdr, *, channel_axis=-1):
    return _transform('ydbdr2rgb', ydbdr, channel_axis)


def rgb2rgbcie(rgb, *, channel_axis=-1):
    return _transform('rgb2rgbcie', rgb, channel_axis)


def rgbcie2rgb(rgbcie, *, channel_axis=-1):
    return _transform('rgbcie2rgb', rgbcie, channel_axis)


def rgb2xyz(rgb, *, channel_axis=-1):
    return _transform('rgb2xyz', rgb, channel_axis)


def xyz2rgb(xyz, *, channel_axis=-1):
    return _transform('xyz2rgb', xyz, channel_axis)
