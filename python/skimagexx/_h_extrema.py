"""Owned H-extrema integer composition; floating/promotion stages remain open.

Native preparation and exact residue comparison reuse the repaired owned
reconstruction implementation. No upstream H-extrema or reconstruction algorithm
is imported. NumPy scalar comparison retains version-specific typed-ptp behavior.
"""
import inspect
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


def h_maxima(image, h, footprint=None):
    return _integer(image, h, footprint, False)


def h_minima(image, h, footprint=None):
    return _integer(image, h, footprint, True)
