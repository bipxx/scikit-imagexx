"""Wiener deconvolution, J-invariant masking and cycle spinning.

Reference algorithms and interface follow scikit-image 0.26 (BSD-3-Clause,
third_party/scikit-image-LICENSE.txt).
"""

from functools import partial, wraps
from importlib.util import find_spec
from itertools import product
from operator import index
import warnings

import numpy as np

from . import _core
from ._compat import fallback
from ._component_consumers import _DEPRECATED
from ._tv_bregman import _channel_as_last_axis, img_as_float


_DASK_AVAILABLE = find_spec('dask') is not None


def _plain_numeric(value):
    return type(value) is np.ndarray and value.dtype.kind in 'buifc'


def _float_dtype(value):
    return np.float32 if value.dtype.char in 'ef' else np.float64


def _native_dense(value):
    return np.require(value, requirements=['C', 'A'])


def _spectrum_inputs(image, psf, reg, is_real, *, fuse_laplacian=False):
    if (is_real is not True or not _plain_numeric(image) or image.dtype.kind == 'c'
            or not _plain_numeric(psf) or image.ndim == 0 or not all(image.shape)
            or psf.ndim != image.ndim or not all(psf.shape)):
        return None
    shape = image.shape
    spectrum_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    if reg is None:
        if min(shape) < 3:
            return None
    elif (not _plain_numeric(reg) or reg.ndim != image.ndim or not all(reg.shape)
          or (reg.dtype.kind == 'c' and reg.shape != spectrum_shape)
          or any(r > n for r, n in zip(reg.shape, shape))):
        return None
    if any(p > n for p, n in zip(psf.shape, shape)):
        return None
    dtype = _float_dtype(image)
    prepared = _native_dense(image.astype(dtype, copy=False))
    if reg is None:
        regularizer = (None if fuse_laplacian else
                       _core.restoration_extra_laplacian(shape, dtype is np.float32))
    elif reg.dtype.kind == 'c':
        regularizer = _native_dense(reg.real.astype(dtype, copy=False))
    else:
        regularizer = _core.restoration_extra_transfer(
            _native_dense(reg.astype(_float_dtype(reg), copy=False)), shape
        ).real.astype(dtype, order='C', copy=True)
    spread = _native_dense(psf.real.astype(dtype, copy=False))
    transfer = (spread.astype(np.result_type(dtype, np.complex64))
                if spread.shape == spectrum_shape
                else _core.restoration_extra_transfer(spread, shape))
    return prepared, transfer, regularizer


def wiener(image, psf, balance, reg=None, is_real=True, clip=True):
    """Restore a degraded image by Wiener–Hunt deconvolution."""
    if type(balance) in (int, float, np.float32, np.float64) and type(clip) is bool:
        inputs = _spectrum_inputs(image, psf, reg, is_real, fuse_laplacian=True)
        if inputs is not None and (inputs[0].dtype == np.float64
                                   or type(balance) is not np.float64):
            result = _core.restoration_extra_wiener(*inputs, balance, clip)
            if result is not None:
                return result
    return fallback('restoration', 'wiener', image, psf, balance, reg=reg,
                    is_real=is_real, clip=clip)


class _GibbsWorkspace:
    def __init__(self, data):
        shape = data.shape
        dtype = data.real.dtype
        self.sample = np.empty_like(data)
        self.residuals = np.empty((2, *shape), dtype=data.dtype)
        self.normals = np.empty((2, *shape), dtype=np.float64)
        self.real, self.imag = self.normals
        self.squared = np.empty((2, *shape), dtype=dtype)
        self.rows = np.empty((2, shape[0]), dtype=dtype)
        self.norms = np.empty(2, dtype=dtype)
        self.first_column = np.empty(2, dtype=dtype)
        self.total = np.zeros_like(data)
        self.difference = np.empty_like(data)
        self.mean = np.empty_like(data)
        self.magnitude = self.squared[0]

    def residual_norms(self):
        # Keep upstream's two axis reductions and its half-spectrum heuristic.
        np.absolute(self.residuals, out=self.squared)
        np.square(self.squared, out=self.squared)
        np.sum(self.squared, axis=-1, out=self.rows)
        np.sum(self.rows, axis=-1, out=self.norms)
        if self.sample.shape[0] != self.sample.shape[1]:
            self.rows[:] = self.squared[..., 0]
            np.sum(self.rows, axis=-1, out=self.first_column)
            return (2 * self.norms[0] - self.first_column[0],
                    2 * self.norms[1] - self.first_column[1])
        return self.norms

    def accumulate(self, count):
        if count == 1:
            np.add(self.total, self.sample, out=self.total)
            return np.nan
        np.divide(self.total, count - 1, out=self.difference)
        np.add(self.total, self.sample, out=self.total)
        np.divide(self.total, count, out=self.mean)
        np.subtract(self.mean, self.difference, out=self.difference)
        np.absolute(self.difference, out=self.magnitude)
        change = self.magnitude.sum()
        np.absolute(self.total, out=self.magnitude)
        return change / self.magnitude.sum() / count


def unsupervised_wiener(
    image, psf, reg=None, user_params=None, is_real=True, clip=True, *, rng=None
):
    """Estimate the Wiener–Hunt regularization with Gibbs sampling."""
    def original():
        return fallback('restoration', 'unsupervised_wiener', image, psf, reg=reg,
                        user_params=user_params, is_real=is_real, clip=clip, rng=rng)

    if (not _plain_numeric(image) or image.ndim != 2
            or type(clip) is not bool or (user_params is not None and type(user_params) is not dict)):
        return original()
    params = user_params or {}
    maximum = params.get('max_num_iter', 200)
    minimum = params.get('min_num_iter', 30)
    burnin = params.get('burnin', 15)
    threshold = params.get('threshold', 1e-4)
    callback = params.get('callback')
    if (any(type(value) is not int for value in (maximum, minimum, burnin))
            or maximum <= 0 or burnin < 0 or maximum <= burnin + 1
            or type(threshold) not in (float, int) or callback is not None):
        return original()
    inputs = _spectrum_inputs(image, psf, reg, is_real)
    if inputs is None:
        return original()
    image, transfer, regularizer = inputs
    data = _core.restoration_extra_gibbs_forward(image)
    power = np.abs(transfer) ** 2
    reg_power = regularizer ** 2
    initial_precision = power + reg_power
    if (not np.isfinite(data).all() or not np.isfinite(initial_precision).all()
            or np.any(initial_precision == 0)):
        return original()
    generator = np.random.default_rng(rng)
    workspace = _GibbsWorkspace(data)
    draw_normal = generator.standard_normal
    draw_gamma = generator.gamma
    sample = _core.restoration_extra_gibbs_step
    noise_shape, prior_shape = image.size / 2, (image.size - 1) / 2
    noise, prior = [1], [1]
    delta = np.nan
    for iteration in range(maximum):
        # Keep the reference's float64 fills and noise-before-prior gamma draws.
        draw_normal(out=workspace.real)
        draw_normal(out=workspace.imag)
        sample(data, transfer, regularizer, power, reg_power, noise[-1], prior[-1],
               workspace.normals, workspace.sample, workspace.residuals)
        noise_norm, prior_norm = workspace.residual_norms()
        noise.append(draw_gamma(noise_shape, 2 / noise_norm))
        prior.append(draw_gamma(prior_shape, 2 / prior_norm))
        count = iteration - burnin
        if count > 0:
            delta = workspace.accumulate(count)
        if iteration > minimum and delta < threshold:
            break
    np.divide(workspace.total, count, out=workspace.total)
    result = _core.restoration_extra_gibbs_inverse(workspace.total, image.shape)
    if clip:
        np.clip(result, -1, 1, out=result)
    return result, {'noise': noise, 'prior': prior}


def _grid(offset, dims, stride):
    phases = [0] * dims
    for axis in range(dims - 1, -1, -1):
        offset, phases[axis] = divmod(offset, stride)
    return tuple(slice(phase, None, stride) for phase in phases)


def _j_domain(image, stride, spatial_rank):
    return (_plain_numeric(image) and image.dtype.kind != 'c' and image.dtype.isnative
            and type(stride) in (int, np.int32, np.int64) and 0 < stride
            and spatial_rank > 0 and all(image.shape))


def _j_image(image):
    image = img_as_float(image)
    return image.astype(_float_dtype(image), copy=False)


def _masked(image, denoise_function, kwargs, spatial_rank, stride, masks):
    interp = _core.restoration_extra_interpolate(_native_dense(image), spatial_rank)
    output = np.zeros_like(image)
    if masks is None:
        masks = (_grid(i, spatial_rank, stride) for i in range(stride**spatial_rank))
    for mask in masks:
        input_image = image.copy()
        input_image[mask] = interp[mask]
        output[mask] = denoise_function(input_image, **kwargs)[mask]
    return output


def denoise_invariant(image, denoise_function, *, stride=4, masks=None, denoiser_kwargs=None):
    """Apply a denoiser to disjoint held-out pixels using mirror interpolation."""
    kwargs = {} if denoiser_kwargs is None else denoiser_kwargs
    spatial_rank = getattr(image, 'ndim', 0) - int(
        type(kwargs) is dict and kwargs.get('channel_axis') is not None
    )
    if not _j_domain(image, stride, spatial_rank) or type(kwargs) is not dict:
        return fallback('restoration', 'denoise_invariant', image, denoise_function,
                        stride=stride, masks=masks, denoiser_kwargs=denoiser_kwargs)
    return _masked(_j_image(image), denoise_function, kwargs, spatial_rank, int(stride), masks)


def calibrate_denoiser(
    image, denoise_function, denoise_parameters, *, stride=4, approximate_loss=True,
    extra_output=False
):
    """Select the J-invariant denoiser parameters minimizing held-out MSE."""
    if not _j_domain(image, stride, getattr(image, 'ndim', 0)) or type(denoise_parameters) is not dict:
        return fallback('restoration', 'calibrate_denoiser', image, denoise_function,
                        denoise_parameters, stride=stride, approximate_loss=approximate_loss,
                        extra_output=extra_output)

    image = img_as_float(image)
    prepared = _j_image(image)
    parameters = [dict(zip(denoise_parameters, values))
                  for values in product(*denoise_parameters.values())]
    losses = []
    for kwargs in parameters:
        spatial_rank = image.ndim - int(kwargs.get('channel_axis') is not None)
        if approximate_loss:
            mask = _grid(int(stride)**spatial_rank // 2, spatial_rank, int(stride))
            masks = [mask]
        else:
            mask = Ellipsis
            masks = None
        if spatial_rank == 0:
            denoised = denoise_invariant(image, denoise_function, stride=stride,
                                         masks=masks, denoiser_kwargs=kwargs)
        else:
            denoised = _masked(prepared, denoise_function, kwargs, spatial_rank, int(stride), masks)
        # MSE subtracts in image precision and accumulates in float64.
        losses.append(np.mean((image[mask].astype(prepared.dtype, copy=False) - denoised[mask])
                              ** 2, dtype=np.float64))
    result = partial(denoise_invariant, denoise_function=denoise_function, stride=stride,
                     denoiser_kwargs=parameters[np.argmin(losses)])
    return (result, (parameters, losses)) if extra_output else result


def _spin_dense(array):
    return (type(array) is np.ndarray and array.dtype.type in (np.float32, np.float64)
            and array.dtype.metadata is None and array.dtype.isnative and array.size > 0
            and array.flags.c_contiguous and array.flags.aligned)


def _cycle_shifts(value, ndim, multichannel, padding, name):
    if np.isscalar(value):
        return (value,) * (ndim - multichannel) + (padding,) * multichannel
    if multichannel and len(value) == ndim - 1:
        return (*value, padding)
    if len(value) != ndim:
        raise ValueError(f'{name} should have length ndim')
    return value


def _cycle_worker_alias(function):
    @wraps(function)
    def call(*args, **kwargs):
        if len(args) > 4:
            if 'num_workers' in kwargs:
                return function(*args, **kwargs)
            old = args[4]
            args = args[:4] + (_DEPRECATED,) + args[5:]
        else:
            old = kwargs.pop('num_workers', _DEPRECATED)
        if old is not _DEPRECATED:
            warnings.warn(
                'Parameter `num_workers` is deprecated since version 0.26 and will be '
                'removed in 0.28 (or later). To avoid this warning, please use the parameter '
                '`workers` instead. For more details, see the documentation of `cycle_spin`.',
                FutureWarning, stacklevel=2,
            )
            if 'workers' in kwargs and kwargs['workers'] is not _DEPRECATED:
                raise ValueError('Both deprecated parameter `num_workers` and new parameter '
                                 '`workers` are used. Use only the latter to avoid conflicting values.')
            kwargs['workers'] = old
        return function(*args, **kwargs)
    return call


@_cycle_worker_alias
@_channel_as_last_axis
def cycle_spin(
    x, func, max_shifts, shift_steps=1, num_workers=_DEPRECATED, func_kw=None, *,
    workers=None, channel_axis=None
):
    """Average the inverse-aligned outputs of a function over circular shifts."""
    serial = workers == 1 or (not _DASK_AVAILABLE and (workers is None or workers > 1))
    if not serial or not _plain_numeric(x) or x.ndim == 0:
        return fallback('restoration', 'cycle_spin', x, func, max_shifts, shift_steps,
                        func_kw=func_kw, channel_axis=channel_axis, workers=workers)
    multichannel = channel_axis is not None
    limits = _cycle_shifts(max_shifts, x.ndim, multichannel, 0, 'max_shifts')
    steps = _cycle_shifts(shift_steps, x.ndim, multichannel, 1, 'shift_steps')
    if any(step < 1 for step in steps):
        raise ValueError('shift_steps must all be >= 1')
    if multichannel and limits[-1] != 0:
        raise ValueError('Multichannel cycle spinning should not have shifts along the last axis.')
    ranges = tuple(range(0, limit + 1, index(step)) for limit, step in zip(limits, steps))
    if workers != 1:
        warnings.warn(
            'The optional dask dependency is not installed. '
            'The number of workers is set to 1. To silence '
            'this warning, install dask or explicitly set `workers=1` '
            'when calling the `cycle_spin` function',
            UserWarning, stacklevel=2,
        )
    axes = tuple(range(x.ndim))
    mean = None
    count = 0
    kwargs = {} if func_kw is None else func_kw
    dense = _spin_dense(x)
    roll = _core.restoration_extra_roll
    for shift in product(*ranges):
        shifted = (roll(x, tuple(s % n for s, n in zip(shift, x.shape))) if dense
                   else np.roll(x, shift, axis=axes))
        result = func(shifted, **kwargs)
        if (_spin_dense(result) and result.shape == x.shape
                and (mean is None or (_spin_dense(mean) and mean.shape == x.shape
                                      and mean.dtype == result.dtype))):
            mean = roll(result, tuple(-s % n for s, n in zip(shift, x.shape)), mean)
        else:
            aligned = np.roll(result, tuple(-s for s in shift), axis=axes)
            if mean is None:
                mean = aligned
            else:
                mean += aligned
        count += 1
    if not count:
        raise IndexError('list index out of range')
    mean /= count
    return mean
