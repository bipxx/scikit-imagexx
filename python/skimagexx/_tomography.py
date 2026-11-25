"""Tomography wrappers for contiguous, writable, native-endian numeric arrays."""

import warnings

import numpy as np

from . import _core
from ._compat import fallback


_FILTERS = {name: i for i, name in enumerate(("ramp", "shepp-logan", "cosine", "hamming", "hann", None))}
_INTERPOLATIONS = {name: i for i, name in enumerate(("linear", "nearest", "cubic"))}
_FLOATS = (np.float32, np.float64)
_BOOLS = (bool, np.bool_)
_SIZES = (int, np.int8, np.int16, np.int32, np.int64, np.longlong,
          np.uint8, np.uint16, np.uint32, np.uint64, np.ulonglong)
_MAX_EXTENT = 16384
_WEAK_SCALARS = int(np.__version__.split(".")[0]) >= 2
_DEFAULT_ERRORS = dict(divide="warn", over="warn", under="ignore", invalid="warn")


def _array(value, ndim):
    return (
        type(value) is np.ndarray
        and value.ndim == ndim
        and 0 < min(value.shape) <= max(value.shape) <= _MAX_EXTENT
        and value.dtype.metadata is None
        and value.dtype.isnative
        and value.flags.c_contiguous
        and value.flags.aligned
        and value.flags.writeable
    )


def _owned_read(value):
    if not value.flags.owndata:
        _core._tomography_storage(value)


def _image(value, preserve_range):
    source = value.dtype
    if source.type in _FLOATS:
        return value
    if source.type is np.float16:
        _owned_read(value)
        return value.astype(np.float32)
    if source.kind not in "biu":
        return None
    _owned_read(value)
    if preserve_range or source.kind == "b":
        return value.astype(np.float64)
    result = np.multiply(value, 1.0 / np.iinfo(source).max, dtype=np.float64)
    if source.kind == "i":
        np.maximum(result, -1.0, out=result)
    return result


def _sequence(value, length=None):
    if type(value) in (list, tuple) and 0 < len(value) <= _MAX_EXTENT:
        if not all(type(item) in (int, float, bool) for item in value):
            return None
        value = np.asarray(value)
    if (
        not _array(value, 1)
        or (length is not None and len(value) != length)
        or (value.dtype.kind not in "biu" and value.dtype.type not in (*_FLOATS, np.float16))
    ):
        return None
    _owned_read(value)
    return value


def _trig(theta):
    radians = np.deg2rad(theta)
    # Use the same scalar ufuncs as stock. Vector trig can use different
    # approximations, which matters at nearest-neighbor half ties.
    cosine = np.fromiter((np.cos(angle) for angle in radians), radians.dtype, len(radians))
    sine = np.fromiter((np.sin(angle) for angle in radians), radians.dtype, len(radians))
    return cosine, sine


def _finite_angles(theta):
    return theta is not None and np.isfinite(theta).all()


def _rotation(theta, center):
    cosine, sine = _trig(theta)
    if not _WEAK_SCALARS and cosine.dtype.type is not np.float64:
        # NumPy 1.x promoted the scalar offset expressions in upstream's loop.
        offsets = [(-center * (c + s - 1), -center * (c - s - 1)) for c, s in zip(cosine, sine)]
        offset_x, offset_y = np.asarray(offsets).T
    else:
        offset_x = -center * (cosine + sine - 1)
        offset_y = -center * (cosine - sine - 1)
    return np.stack((cosine, sine, offset_x, offset_y), dtype=np.float64)


def radon(image, theta=None, circle=True, *, preserve_range=False):
    """Project a 2-D image along rotation angles in degrees."""
    if (_array(image, 2) and type(circle) in _BOOLS and type(preserve_range) in _BOOLS
            and np.geterr() == _DEFAULT_ERRORS):
        angles = np.arange(180) if theta is None else _sequence(theta)
        if _finite_angles(angles):
            prepared = _image(image, preserve_range)
            if prepared is not None:
                side = min(image.shape) if circle else int(np.ceil(np.sqrt(2) * max(image.shape)))
                result = _core._tomography_radon(prepared, _rotation(angles, side // 2), circle)
                if result is not None:
                    output, outside_circle = result
                    if outside_circle:
                        warnings.warn(
                            "Radon transform: image must be zero outside the reconstruction circle",
                            stacklevel=2,
                        )
                    return output
    return fallback("transform", "radon", image, theta, circle, preserve_range=preserve_range)


def iradon(
    radon_image, theta=None, output_size=None, filter_name="ramp", interpolation="linear",
    circle=True, preserve_range=True,
):
    """Reconstruct an image by filtered backprojection."""
    if (
        _array(radon_image, 2)
        and type(circle) in _BOOLS
        and type(preserve_range) in _BOOLS
        and (filter_name is None or type(filter_name) is str)
        and filter_name in _FILTERS
        and type(interpolation) is str
        and interpolation in _INTERPOLATIONS
        and (output_size is None or type(output_size) in _SIZES)
        and np.geterr() == _DEFAULT_ERRORS
    ):
        detectors, count = radon_image.shape
        size = output_size
        if size is None:
            size = detectors if circle else int(np.floor(np.sqrt(detectors ** 2 / 2.0)))
        angles = np.linspace(0, 180, count, endpoint=False) if theta is None else _sequence(theta, count)
        if _finite_angles(angles) and 0 < size <= _MAX_EXTENT:
            prepared = _image(radon_image, preserve_range)
            if prepared is not None:
                coordinates = np.stack(_trig(angles), dtype=np.float64)
                result = _core._tomography_iradon(
                    prepared, coordinates, int(size), _FILTERS[filter_name], _INTERPOLATIONS[interpolation], circle
                )
                if result is not None:
                    return result
    return fallback(
        "transform", "iradon", radon_image, theta, output_size, filter_name, interpolation, circle, preserve_range
    )


def _sart_dtype(value, source):
    if value is None:
        return source
    if value is float or any(value is item for item in _FLOATS) or type(value) is str or isinstance(value, np.dtype):
        try:
            return np.dtype(value)
        except TypeError:
            pass
    return None


def iradon_sart(
    radon_image, theta=None, image=None, projection_shifts=None, clip=None, relaxation=0.15, dtype=None,
):
    """Reconstruct an image with one ordered SART iteration."""
    if (_array(radon_image, 2) and radon_image.dtype.type in _FLOATS
            and type(relaxation) in (float, int) and -1 <= relaxation <= 1
            and np.geterr() == _DEFAULT_ERRORS):
        target = _sart_dtype(dtype, radon_image.dtype)
        if target is not None and target.type in _FLOATS and target.isnative and target.metadata is None:
            result = _sart(radon_image, theta, image, projection_shifts, clip, relaxation, target)
            if result is not None:
                return result
    return fallback("transform", "iradon_sart", radon_image, theta, image, projection_shifts, clip, relaxation, dtype)


def _sart(sinogram, theta, image, projection_shifts, clip, relaxation, dtype):
    size, count = sinogram.shape
    if image is not None and (
        not _array(image, 2) or image.dtype != dtype or image.shape != (size, size)
    ):
        return None
    angles = np.linspace(0, 180, count, endpoint=False, dtype=dtype) if theta is None else _sequence(theta, count)
    shifts = np.zeros(count, dtype=dtype) if projection_shifts is None else _sequence(projection_shifts, count)
    bounds = None if clip is None else _sequence(clip, 2)
    if angles is None or shifts is None or (clip is not None and bounds is None):
        return None
    if image is not None and any(
        arg is not None and np.may_share_memory(image, arg) for arg in (sinogram, angles, shifts, bounds)
    ):
        return None
    if sinogram.dtype != dtype:
        _owned_read(sinogram)
        sinogram = sinogram.astype(dtype)
    return _core._tomography_sart(
        sinogram, angles.astype(dtype, copy=False), shifts.astype(dtype, copy=False),
        image, None if bounds is None else bounds.astype(dtype, copy=False), relaxation, _WEAK_SCALARS,
    )
