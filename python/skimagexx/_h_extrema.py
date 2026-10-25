"""Owned H-extrema composition with explicit NumPy semantic preparation.

The integer-v1 route is preserved. Floating and promoted-height routes retain
source-ordered NumPy stages and use owned reconstruction plus an exact native
mask when admitted. Other final comparisons remain explicit provider work.
No upstream H-extrema or reconstruction algorithm is imported.
"""
import inspect
import functools
import warnings
import numpy as np
from . import _core
from ._reconstruction import reconstruction


def _admit():
    if getattr(_core, 'h_extrema_interface', None) != 'h-extrema-integer-v1':
        raise NotImplementedError('H-extrema compiled integer interface unavailable')
    for name in ('h_extrema_integer_range', 'h_extrema_integer_seed', 'h_extrema_integer_mask'):
        entry = getattr(_core, name, None)
        if not inspect.isbuiltin(entry) or entry.__name__ != name or entry.__module__ != _core.__name__:
            raise NotImplementedError('H-extrema requires owned compiled entries')


def _integer(image, h, footprint, minima):
    _admit()
    if (type(image) is not np.ndarray or image.dtype.kind not in 'iu'
            or image.itemsize not in (1, 2, 4, 8) or type(h) is not int or h < 0):
        raise NotImplementedError('H-extrema floating, promotion and protocol stages are pending')
    if not image.size:
        raise ValueError('zero-size array to reduction operation maximum which has no identity')
    bits = _core.h_extrema_integer_range(image)
    width = 8 * image.itemsize
    if image.dtype.kind == 'i' and bits >= 1 << (width - 1):
        bits -= 1 << width
    typed_range = image.dtype.type(bits)
    # Keep this original comparison as scalar provider work. NumPy1 and NumPy2
    # differ for large Python heights versus uint64; widening it is not valid.
    if h > typed_range:
        return np.zeros(image.shape, dtype=np.uint8)
    if h == 0:
        name = 'local_minima' if minima else 'local_maxima'
        raise ValueError('h = 0 is ambiguous, use ' + name + '() instead?')
    if h > (1 << width) - 1:
        word = 'added' if minima else 'subtracted'
        raise ValueError('The ' + word + ' constant is not compatiblewith the image data type.')
    method = 1 if minima else 0
    shifted, _ = _core.h_extrema_integer_seed(image, h, method)
    # Footprint conversion can mutate image; reconstruction's post-loop shape
    # assignment can invoke a warning callback. These real stages stay ordered.
    reconstructed = reconstruction(shifted, image, method='erosion' if minima else 'dilation', footprint=footprint)
    # The native binder revalidates the current image/root after those callbacks.
    result, _ = _core.h_extrema_integer_mask(image, reconstructed, float(h), method)
    return result


_warn = functools.partial(warnings.warn, stacklevel=2)
_LIMITS = {dtype: (int(np.iinfo(dtype).min), int(np.iinfo(dtype).max))
           for dtype in (np.uint8, np.int8, np.uint16, np.int16,
                         np.uint32, np.int32, np.uint64, np.int64,
                         np.ulonglong, np.longlong)}
_LIMITS.update({np.bool_: (False, True), np.float16: (-1, 1),
                np.float32: (-1, 1), np.float64: (-1, 1)})


def _admit_real():
    _admit()
    if getattr(_core, 'h_extrema_real_interface', None) != 'h-extrema-real-v1':
        raise NotImplementedError('H-extrema compiled real interface unavailable')
    entry = getattr(_core, 'h_extrema_real_mask', None)
    if (not inspect.isbuiltin(entry) or entry.__name__ != 'h_extrema_real_mask'
            or entry.__module__ != _core.__name__):
        raise NotImplementedError('H-extrema requires an owned compiled real stage')


def _storage(value):
    # type() does not invoke an instance's overridable __class__/base property.
    return _core._HExtremaStorage(value) if issubclass(type(value), np.ndarray) else None


def _check(*guards):
    for guard in guards:
        if guard is not None:
            guard.check()


def _clipped_seed(image, h, image_guard, height_guard, minima):
    # The literal source scalar/array expressions matter for uint64 mixed-sign
    # promotion and for array heights, which do not enter the float-image path.
    low, high = _LIMITS[image.dtype.type]
    _check(height_guard)
    if h > high - low:
        word = 'added' if minima else 'subtracted'
        raise ValueError('The ' + word + ' constant is not compatiblewith the image data type.')
    _check(image_guard, height_guard)
    result = image + h if minima else image - h
    result_guard = _storage(result)
    _check(height_guard)
    threshold = high - h if minima else h + low
    threshold_guard = _storage(threshold)
    _check(image_guard, threshold_guard)
    outside = image > threshold if minima else image < threshold
    outside_guard = _storage(outside)
    _check(result_guard, outside_guard)
    result[outside] = high if minima else low
    return result


def _provider_mask(image, reconstructed, h, image_guard, reconstructed_guard, height_guard, minima):
    # Real source operations preserve subclass construction and unusual scalar
    # promotion. This is explicitly provider work, never a native mask claim.
    _check(image_guard, reconstructed_guard)
    residue = reconstructed - image if minima else image - reconstructed
    residue_guard = _storage(residue)
    _check(residue_guard, height_guard)
    selected = residue >= h
    selected_guard = _storage(selected)
    _check(selected_guard)
    return selected.astype(np.uint8)


def _real(image, h, footprint, minima):
    _admit_real()
    # Capture array-height storage before the FIRST comparison, not just the
    # final native stage. No height value or custom scalar is admission-probed.
    image_guard, height_guard = _storage(image), _storage(h)
    _check(image_guard)
    span = np.ptp(image)
    span_guard = _storage(span)
    _check(height_guard, span_guard)
    if h > span:
        return np.zeros(image.shape, dtype=np.uint8)
    if np.issubdtype(type(h), np.floating) and np.issubdtype(image.dtype, np.integer):
        _check(height_guard)
        if (h % 1) != 0:
            _warn('possible precision loss converting image to '
                  'floating point. To silence this warning, '
                  'ensure image and h have same data type.', stacklevel=2)
            _check(image_guard)
            image = image.astype(float)
            image_guard = _storage(image)
        else:
            h = image.dtype.type(h)
            height_guard = _storage(h)
    _check(height_guard)
    if h == 0:
        name = 'local_minima' if minima else 'local_maxima'
        raise ValueError('h = 0 is ambiguous, use ' + name + '() instead?')
    if np.issubdtype(image.dtype, np.floating):
        # Do not fuse across any ufunc allocation or warning callback. In
        # particular, a resolution-underflow handler may mutate image before
        # the following shift; source rounding occurs after EACH operation.
        coefficient = 2 * np.finfo(image.dtype).resolution
        _check(image_guard)
        absolute = np.abs(image)
        absolute_guard = _storage(absolute)
        _check(absolute_guard)
        resolution = coefficient * absolute
        resolution_guard = _storage(resolution)
        _check(image_guard, height_guard)
        shifted = image + h if minima else image - h
        shifted_guard = _storage(shifted)
        _check(shifted_guard, resolution_guard)
        shifted = shifted + resolution if minima else shifted - resolution
    else:
        shifted = _clipped_seed(image, h, image_guard, height_guard, minima)
    shifted_guard = _storage(shifted)
    _check(shifted_guard, image_guard)
    reconstructed = reconstruction(shifted, image, method='erosion' if minima else 'dilation', footprint=footprint)
    reconstructed_guard = _storage(reconstructed)
    _check(image_guard, reconstructed_guard, height_guard)
    if type(image) is np.ndarray and type(reconstructed) is np.ndarray:
        # The binder allocates output, revalidates every owner, THEN captures
        # ordinary height bytes. It declines custom/inexact thresholds without
        # executing their conversion protocols or writing a partial mask.
        native = _core.h_extrema_real_mask(image, reconstructed, h, 1 if minima else 0)
        if native is not None:
            return native[0]
    return _provider_mask(image, reconstructed, h, image_guard, reconstructed_guard, height_guard, minima)


def _dispatch(image, h, footprint, minima):
    # Preserve the proved integer-v1 dispatch, reductions, seed, mask and counters.
    if (type(image) is np.ndarray and image.dtype.kind in 'iu'
            and image.itemsize in (1, 2, 4, 8) and type(h) is int and h >= 0):
        return _integer(image, h, footprint, minima)
    return _real(image, h, footprint, minima)


def h_maxima(image, h, footprint=None):
    return _dispatch(image, h, footprint, False)


def h_minima(image, h, footprint=None):
    return _dispatch(image, h, footprint, True)
