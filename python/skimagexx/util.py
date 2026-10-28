"""Image utilities with native dispatch."""
import numpy as np
import warnings
from . import _core
from ._compare_images import compare_images
from ._map_array import map_array
from ._label_points import label_points
from ._unique_rows import unique_rows
from ._compat import fallback, module_attribute, native_image

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []

__all__ = ['unique_rows', 'label_points', 'map_array', 'compare_images', 'invert', 'img_as_float', 'img_as_float32', 'img_as_float64',
           'img_as_ubyte', 'img_as_uint', 'img_as_int', 'img_as_bool', 'dtype_limits']

_SUPPORTED = {np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32,
              np.uint64, np.int64, np.float32, np.float64}
_LIMITS = {dtype: (int(np.iinfo(dtype).min), int(np.iinfo(dtype).max))
           for dtype in _SUPPORTED if np.issubdtype(dtype, np.integer)}
_LIMITS.update({np.bool_: (False, True), np.float16: (-1, 1), np.float32: (-1, 1), np.float64: (-1, 1)})


def _convert(image, dtype, force_copy, name):
    image = np.asarray(image)
    # Upstream's identity check is on dtype.type, including nonnative byte order.
    if dtype is None and image.dtype.kind == 'f' or dtype is not None and image.dtype.type is np.dtype(dtype).type:
        return image.copy() if force_copy else image
    target = np.dtype(np.float64 if dtype is None else dtype)
    if image.dtype.type not in _SUPPORTED or not image.dtype.isnative:
        return fallback('util', name, image, force_copy)
    result, downcast, nan, overflow, underflow, invalid_cast, maximum = _core.convert(image, target)
    if image.ndim == 0 and image.dtype.kind == 'f' and target.kind in 'ui':
        # Preserve the pinned upstream scalar/out= behavior (after range checks).
        raise TypeError('return arrays must be of ArrayType')
    if downcast:
        bits = target.itemsize*8-(target.kind == 'i')
        message_dtype = f'int{bits+1}' if bits % 2 else f'uint{bits}'
        warnings.warn(f'Downcasting {image.dtype} to {message_dtype} without scaling because max '
                      f'value {maximum} fits in {message_dtype}', UserWarning, stacklevel=3)
    # Use one-element NumPy operations to honor warn/raise/call/print/log and
    # errstate, without delegating image conversion or rescanning the input.
    if overflow:
        if image.dtype.kind == 'f' and target.kind in 'ui':
            scale = np.iinfo(target).max if target.kind == 'u' else 32767.5
            np.multiply(np.asarray([np.finfo(image.dtype).max], dtype=image.dtype),
                        np.asarray(scale, dtype=image.dtype))
        else:
            np.asarray([np.finfo(np.float64).max]).astype(np.float32)
    if underflow:
        np.asarray([1e-300]).astype(np.float32)
    if nan:
        np.asarray([np.nan], dtype=image.dtype).astype(target)
    elif invalid_cast:
        bits = np.array([0x7f800001], dtype=np.uint32) if image.dtype.itemsize == 4 else np.array([0x7ff0000000000001], dtype=np.uint64)
        bits.view(image.dtype).astype(target)
    return result[()] if image.ndim == 0 and target.kind == 'b' else result


def img_as_float(image, force_copy=False):
    return _convert(image, None, force_copy, 'img_as_float')


def img_as_float32(image, force_copy=False):
    return _convert(image, np.float32, force_copy, 'img_as_float32')


def img_as_float64(image, force_copy=False):
    return _convert(image, np.float64, force_copy, 'img_as_float64')


def img_as_ubyte(image, force_copy=False):
    return _convert(image, np.uint8, force_copy, 'img_as_ubyte')


def img_as_uint(image, force_copy=False):
    return _convert(image, np.uint16, force_copy, 'img_as_uint')


def img_as_int(image, force_copy=False):
    return _convert(image, np.int16, force_copy, 'img_as_int')


def img_as_bool(image, force_copy=False):
    return _convert(image, np.bool_, force_copy, 'img_as_bool')


def dtype_limits(image, clip_negative=False):
    result = _LIMITS.get(image.dtype.type)
    if result is None:
        return fallback('util', 'dtype_limits', image, clip_negative)
    return (0, result[1]) if clip_negative else result


def invert(image, signed_float=False):
    if native_image(image):
        return _core.invert(image, signed_float)
    return fallback('util', 'invert', image, signed_float)


def __getattr__(name):
    if name == 'FailedEstimationAccessError':
        from ._geometric import _canonical_failure_error
        return _canonical_failure_error()
    return module_attribute('util', name)


def __dir__():
    return sorted(set(globals()) | set(__all__) | {'FailedEstimationAccessError'})
