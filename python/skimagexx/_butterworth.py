"""Butterworth orchestration with owned C++ Fourier transforms.

NumPy supplies padding, frequency-grid/mask arithmetic, spectrum multiplication
and cropping. Those stages retain NumPy promotion and warning semantics; this
is a native-partial FFT implementation, not an owned C++ filter-mask kernel.
Extended-precision FFT inputs remain an explicit binding refusal frontier.
"""
import functools

import numpy as np

from . import _core


_MASK_FLOAT_TYPE = {
    "f": np.float32,
    "d": np.float64,
    "F": np.complex64,
    "D": np.complex128,
    "e": np.float32,
    "g": np.float64,
    "G": np.complex128,
}


def _butterworth_mask(shape, factor, order, high_pass, real, dtype,
                      squared_butterworth):
    # Keep the source arithmetic order: cast the distance grid before power,
    # and multiply by q2 for high-pass instead of rewriting it as 1-lowpass.
    ranges = []
    for d in shape:
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
        axis **= 2
        midpoint = axis.size // 2
        ranges.append(np.concatenate((axis[midpoint:], axis[:midpoint])))
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    if len(ranges) == 2:
        q2 = ranges[0][:, None] + ranges[1][None, :]
    else:
        q2 = functools.reduce(np.add, (
            axis.reshape((1,) * i + (-1,) + (1,) * (len(ranges) - i - 1))
            for i, axis in enumerate(ranges)
        ))
    q2 = q2.astype(dtype, copy=False)
    # A cutoff of at least 1e-8 bounds the squared distance below the
    # float32 overflow threshold even across all 64 NumPy dimensions.
    if (type(order) in (int, float) and order == 2.0
            and isinstance(factor, (int, float, np.integer, np.floating))
            and factor >= 1e-8 and np.geterr()["under"] == "ignore"):
        np.multiply(q2, q2, out=q2)
    else:
        q2 = np.power(q2, order)
    if high_pass:
        wfilt = 1 + q2
    else:
        wfilt = q2
        np.add(1, wfilt, out=wfilt)
    np.true_divide(1, wfilt, out=wfilt)
    if high_pass:
        wfilt *= q2
    if not squared_butterworth:
        np.sqrt(wfilt, out=wfilt)
    return wfilt


def butterworth(image, cutoff_frequency_ratio=0.005, high_pass=True, order=2.0,
                channel_axis=None, *, squared_butterworth=True, npad=0):
    """Apply the source-ordered mask using owned forward/inverse FFT stages."""
    if npad < 0:
        raise ValueError("npad must be >= 0")
    elif npad > 0:
        center_slice = tuple(slice(npad, s + npad) for s in image.shape)
        if isinstance(image, np.ndarray):
            _core.fft_check(image)
        image = np.pad(image, npad, mode="edge")
    fft_shape = (
        image.shape if channel_axis is None else np.delete(image.shape, channel_axis)
    )
    is_real = np.isrealobj(image)
    float_dtype = _MASK_FLOAT_TYPE.get(np.dtype(image.dtype).char, np.float64)
    if cutoff_frequency_ratio < 0 or cutoff_frequency_ratio > 0.5:
        raise ValueError("cutoff_frequency_ratio should be in the range [0, 0.5]")
    wfilt = _butterworth_mask(
        fft_shape, cutoff_frequency_ratio, order, high_pass, is_real,
        float_dtype, squared_butterworth,
    )
    axes = tuple(range(image.ndim))
    if channel_axis is not None:
        abs_channel = channel_axis % image.ndim
        axes = axes[:abs_channel] + axes[abs_channel + 1:]
        post = image.ndim - abs_channel - 1
        sl = (slice(None),) * abs_channel + (np.newaxis,) + (slice(None),) * post
        wfilt = wfilt[sl]
    # Match the source FFT validation after mask construction and its warnings.
    transform_shape = [image.shape[axis] for axis in axes]
    if any(size < 1 for size in transform_shape):
        raise ValueError(f"invalid number of data points ({transform_shape}) specified")
    workers = _core.get_num_threads()
    if is_real:
        spectrum = _core.fft_r2c(image, axes, workers)
    else:
        spectrum = _core.fft_c2c(image, axes, True, workers)
    if (np.result_type(wfilt, spectrum) == spectrum.dtype
            and np.broadcast_shapes(wfilt.shape, spectrum.shape) == spectrum.shape):
        np.multiply(wfilt, spectrum, out=spectrum)
    else:
        spectrum = wfilt * spectrum
    if is_real:
        butterfilt = _core.fft_c2r(spectrum, image.shape, axes, workers)
    else:
        butterfilt = _core.fft_c2c(spectrum, axes, False, workers)
    if npad > 0:
        butterfilt = butterfilt[center_slice]
    return butterfilt
