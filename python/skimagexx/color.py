"""Color API compatible with scikit-image."""
import numpy as np
import warnings
from functools import lru_cache
from . import _core
from ._whitepoints import ILLUMINANTS as _ILLUMINANTS
from ._compat import fallback, module_attribute, native_image

__all__ = ['rgb2gray', 'gray2rgb', 'rgb2yuv', 'yuv2rgb', 'rgb2yiq', 'yiq2rgb',
           'rgb2ypbpr', 'ypbpr2rgb', 'rgb2ycbcr', 'ycbcr2rgb', 'rgb2ydbdr',
           'ydbdr2rgb', 'rgb2rgbcie', 'rgbcie2rgb', 'rgb2xyz', 'xyz2rgb']
__all__ += ['rgb2hsv','hsv2rgb','xyz2lab','lab2xyz','rgb2lab','lab2rgb',
            'lab2lch','lch2lab','xyz_tristimulus_values']
__all__ += ['xyz2luv','luv2xyz','rgb2luv','luv2rgb']


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


_TRANSFORMS = {name: i for i, name in enumerate(__all__[2:16])}
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


def xyz_tristimulus_values(*, illuminant, observer, dtype=float):
    illuminant = illuminant.upper()
    observer = observer.upper()
    try:
        return np.asarray(_ILLUMINANTS[illuminant][observer],dtype=dtype)
    except KeyError:
        raise ValueError(f'Unknown illuminant/observer combination (`{illuminant}`, `{observer}`)')


_PERCEPTUAL = {name:i for i,name in enumerate(('rgb2hsv','hsv2rgb','xyz2lab','lab2xyz','rgb2lab','lab2rgb',
                                             'lab2lch','lch2lab','xyz2luv','luv2xyz','rgb2luv','luv2rgb'))}


@lru_cache(maxsize=64)
def _luv_reference(white,dtype):
    # Three-value metadata only. NumPy's dot rounding affects singular inverse
    # denominators, so preserve its installed BLAS behavior without image work.
    w=np.asarray(white,dtype=dtype)
    denominator=np.array([1,15,3],dtype=dtype)@w
    return float(4*w[0]/denominator),float(9*w[1]/denominator)


def _perceptual(name, image, channel_axis, illuminant='D65', observer='2'):
    cie = name in ('xyz2lab','lab2xyz','rgb2lab','lab2rgb','xyz2luv','luv2xyz')
    options = {'channel_axis':channel_axis}
    if cie:
        options.update(illuminant=illuminant,observer=observer)
    if isinstance(image,np.ndarray) and type(image) is not np.ndarray:
        return fallback('color',name,image,**options)
    axis = channel_axis
    if axis is None:
        axis = -1
    elif not np.isscalar(axis):
        if len(axis) > 1:
            raise ValueError('only a single channel axis is currently supported')
        axis = axis[0]
    if name in ('rgb2hsv','xyz2luv') and axis == -1:
        image.ndim  # Upstream inspects this before coercing array-like input.
    a = np.asarray(image)
    moved = a if axis == -1 else np.moveaxis(a,axis,-1)
    polar = name in ('lab2lch','lch2lab')
    if polar and moved.shape[-1] < 3:
        raise ValueError('Input image has less than 3 channels.')
    if not polar and moved.shape[-1] != 3:
        shape = (1,)+moved.shape if name in ('rgb2hsv','xyz2luv') and moved.ndim == 1 else moved.shape
        raise ValueError(f'the input array must have size 3 along `channel_axis`, got {shape}')
    if moved.dtype.type not in _COLOR_TYPES or not moved.dtype.isnative:
        return fallback('color',name,image,**options)
    white = xyz_tristimulus_values(illuminant=illuminant,observer=observer) if cie or name in ('rgb2luv','luv2rgb') else (1.0,1.0,1.0)
    uv=_luv_reference(tuple(white),np.dtype(np.float32 if moved.dtype==np.float32 else np.float64).str) if 'luv' in name else None
    result,negative_z,unsupported = _core.perceptual_transform(moved,_PERCEPTUAL[name],white,uv)
    if unsupported:
        return fallback('color',name,image,**options)
    if negative_z:
        target = 'XYZ' if name == 'lab2xyz' else 'sRGB'
        via = '' if name == 'lab2xyz' else ', via XYZ'
        warnings.warn(f'Conversion from CIE-LAB{via} to {target} color space resulted in '
                      f'{negative_z} negative Z values that have been clipped to zero',UserWarning,stacklevel=3)
    return result if axis == -1 else np.moveaxis(result,-1,axis)


def rgb2hsv(rgb, *, channel_axis=-1):
    return _perceptual('rgb2hsv',rgb,channel_axis)


def hsv2rgb(hsv, *, channel_axis=-1):
    return _perceptual('hsv2rgb',hsv,channel_axis)


def xyz2lab(xyz, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('xyz2lab',xyz,channel_axis,illuminant,observer)


def lab2xyz(lab, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('lab2xyz',lab,channel_axis,illuminant,observer)


def rgb2lab(rgb, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('rgb2lab',rgb,channel_axis,illuminant,observer)


def lab2rgb(lab, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('lab2rgb',lab,channel_axis,illuminant,observer)


def lab2lch(lab, *, channel_axis=-1):
    return _perceptual('lab2lch',lab,channel_axis)


def lch2lab(lch, *, channel_axis=-1):
    return _perceptual('lch2lab',lch,channel_axis)


def xyz2luv(xyz, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('xyz2luv',xyz,channel_axis,illuminant,observer)


def luv2xyz(luv, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('luv2xyz',luv,channel_axis,illuminant,observer)


def rgb2luv(rgb, *, channel_axis=-1):
    return _perceptual('rgb2luv',rgb,channel_axis)


def luv2rgb(luv, *, channel_axis=-1):
    return _perceptual('luv2rgb',luv,channel_axis)
