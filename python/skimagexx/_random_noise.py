"""Preserve NumPy distributions and clipping around bounded native pixel transforms."""

import numpy as np

from . import _core, util
from ._compat import fallback
from ._expand_labels import _ordinary_storage


_UNSIGNED = (np.dtype(np.uint8), np.dtype(np.uint16))
_DOUBLE = np.dtype(np.float64)
_KWARGS = {'gaussian': ('mean', 'var'), 'speckle': ('mean', 'var'),
           's&p': ('amount', 'salt_vs_pepper'), 'pepper': ('amount',)}


def _number_in(value, lower, upper):
    return type(value) in (int, float) and lower <= value <= upper


def _controls(mode, kwargs):
    if mode not in _KWARGS or any(key not in _KWARGS[mode] for key in kwargs):
        return False
    if mode in ('gaussian', 'speckle'):
        return _number_in(kwargs.get('mean', 0.0), -1.0, 1.0) and _number_in(kwargs.get('var', 0.01), 0.0, 1.0)
    return (_number_in(kwargs.get('amount', 0.05), 0.0, 1.0)
            and _number_in(kwargs.get('salt_vs_pepper', 0.5), 0.0, 1.0))


def random_noise(image, mode='gaussian', rng=None, clip=True, **kwargs):
    """Apply a seeded NumPy noise distribution with native pixel transforms."""
    name = mode.lower() if type(mode) is str else None
    if (type(image) is np.ndarray and 1 <= image.ndim <= 3 and 0 < image.size <= 524288
            and (image.dtype == _DOUBLE or image.dtype in _UNSIGNED) and image.dtype.metadata is None
            and image.flags.c_contiguous and image.flags.aligned and _ordinary_storage(image)
            and type(clip) is bool and _controls(name, kwargs)
            and (type(rng) is int and 0 <= rng < 2**64
                 or type(rng) is np.random.Generator and type(rng.bit_generator) is np.random.PCG64)
            and np.geterr()['under'] == 'ignore' and _core.random_noise_round_to_nearest()):
        if image.dtype in _UNSIGNED:
            minimum = 0.0
            bounded = True
        else:
            minimum = image.min()
            bounded = -1.0 <= minimum <= image.max() <= 1.0
        if bounded:
            low = -1.0 if minimum < 0 else 0.0
            values = util.img_as_float(image)
            generator = np.random.default_rng(rng)
            if name in ('gaussian', 'speckle'):
                noise = generator.normal(kwargs.get('mean', 0.0), kwargs.get('var', 0.01) ** 0.5, values.shape)
                result = _core.random_noise_apply_normal(values, noise, name == 'speckle')
            else:
                p = kwargs.get('amount', 0.05)
                q = 0.0 if name == 'pepper' else kwargs.get('salt_vs_pepper', 0.5)
                flipped = None if p in (0, 1) else generator.random(values.shape)
                salted = None if q in (0, 1) else generator.random(values.shape)
                result = _core.random_noise_apply_impulse(values, flipped, salted, p, q, low)
            if clip:
                np.clip(result, low, 1.0, out=result)
            return result

    return fallback('util', 'random_noise', image, mode=mode, rng=rng, clip=clip, **kwargs)
