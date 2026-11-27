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
from . import _tv_bregman, _tv_chambolle
from ._compat import fallback
from ._component_consumers import _DEPRECATED
from ._tv_bregman import _channel_as_last_axis, img_as_float


_DASK_AVAILABLE = find_spec('dask') is not None
_TV_CHAMBOLLE = _tv_chambolle.denoise_tv_chambolle
_TV_CHAMBOLLE_CONVERT = _tv_chambolle._img_as_float
_TV_BREGMAN = _tv_bregman.denoise_tv_bregman
_TV_BREGMAN_KERNEL = _tv_bregman._denoise_tv_bregman
_NUMPY_MEAN = np.mean


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
    power = transfer.real ** 2 + transfer.imag ** 2
    reg_power = regularizer ** 2
    initial_precision = power + reg_power
    if (not np.isfinite(data).all() or not np.isfinite(initial_precision).all()
            or np.any(initial_precision == 0)):
        return original()
    generator = np.random.default_rng(rng)
    normals = np.empty((2, *data.shape), dtype=np.float64)
    normal_real, normal_imag = normals
    total = np.zeros_like(data)
    scalar = data.real.dtype.type
    accumulate = _core.restoration_extra_gibbs_accumulate
    draw_normal = generator.standard_normal
    draw_gamma = generator.gamma
    combined_draw = type(generator) is np.random.Generator
    noise_shape, prior_shape = image.size / 2, (image.size - 1) / 2
    noise, prior = [1], [1]
    for iteration in range(maximum):
        # Keep the reference's float64 fills and noise-before-prior gamma draws.
        if combined_draw:
            draw_normal(out=normals)
        else:
            draw_normal(out=normal_real)
            draw_normal(out=normal_imag)
        count = iteration - burnin
        measure_delta = iteration > minimum and threshold > 0
        noise_norm, prior_norm, delta = accumulate(
            data, transfer, regularizer, power, reg_power, noise[-1], prior[-1],
            normals, total, max(0, count), measure_delta,
        )
        noise.append(draw_gamma(noise_shape, 2 / scalar(noise_norm)))
        prior.append(draw_gamma(prior_shape, 2 / scalar(prior_norm)))
        if measure_delta and delta < threshold:
            break
    np.divide(total, count, out=total)
    result = _core.restoration_extra_gibbs_inverse(total, image.shape)
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


def _original_tv(denoise_function):
    return (denoise_function is _TV_CHAMBOLLE
            and denoise_function is _tv_chambolle.denoise_tv_chambolle)


def _chambolle_batch_parameters(kwargs):
    if (kwargs.keys() - {'weight', 'eps', 'max_num_iter', 'channel_axis'}
            or kwargs.get('channel_axis') is not None):
        return None
    weight = kwargs.get('weight', 0.1)
    eps = kwargs.get('eps', 0.0002)
    iterations = kwargs.get('max_num_iter', 200)
    if (type(weight) is not float or type(eps) is not float or type(iterations) is not int
            or not 2.**-20 <= weight <= 2.**20 or not 0. <= eps <= 2.**20
            or not 1 <= iterations <= 20000
            or not hasattr(_core, 'tv_chambolle_finite_interface')
            or not _tv_chambolle._providers_pristine()):
        return None
    policy = np.geterr()
    if policy['under'] != 'ignore' or any(v not in ('ignore', 'warn', 'raise') for v in policy.values()):
        return None
    profile = _tv_chambolle._finite_profile()
    buffer_elements = np.getbufsize()
    if profile is None or type(buffer_elements) is not int or buffer_elements <= 0:
        return None
    _tv_chambolle._require_stages()
    _tv_chambolle._require_finite()
    return False, weight, eps, iterations, profile == '2.5.4', buffer_elements


def _original_bregman(denoise_function):
    return (denoise_function is _TV_BREGMAN
            and denoise_function is _tv_bregman.denoise_tv_bregman
            and _tv_bregman._denoise_tv_bregman is _TV_BREGMAN_KERNEL
            and _tv_bregman.img_as_float is img_as_float)


def _denoise_batch_parameters(image, denoise_function, kwargs):
    if image.ndim != 2 or not _spin_dense(image) or type(kwargs) is not dict:
        return None
    if _original_tv(denoise_function):
        return _chambolle_batch_parameters(kwargs)
    if (not _original_bregman(denoise_function) or min(image.shape) < 2
            or kwargs.keys() - {'weight', 'eps', 'max_num_iter', 'isotropic', 'channel_axis'}
            or kwargs.get('channel_axis') is not None):
        return None
    weight = kwargs.get('weight', 5.0)
    eps = kwargs.get('eps', 0.001)
    iterations = kwargs.get('max_num_iter', 100)
    isotropic = kwargs.get('isotropic', True)
    if (type(weight) not in (int, float) or not 2.**-20 <= weight <= 2.**20
            or type(eps) not in (int, float) or not 0. <= eps <= 2.**20
            or type(iterations) is not int or not 1 <= iterations <= 20000
            or type(isotropic) is not bool or not _tv_chambolle._providers_pristine()):
        return None
    _tv_bregman._require_kernel()
    return True, weight, eps, iterations, isotropic, 0


def _dense_phase(image, stride, phase):
    return all(offset < extent for offset, extent in zip(divmod(phase, stride), image.shape))


def _masked(image, denoise_function, kwargs, spatial_rank, stride, masks, interp=None,
            generated_phase=None, compact=False, phase_source_shape=None):
    generated_2d = (image.ndim == spatial_rank == 2 and stride <= 64
                    and (masks is None or generated_phase is not None) and _spin_dense(image)
                    and (phase_source_shape is None or image.shape == phase_source_shape))
    if generated_2d:
        parameters = _denoise_batch_parameters(image, denoise_function, kwargs)
        if parameters is not None:
            jobs = () if generated_phase is None else (divmod(generated_phase, stride),)
            result = _core.restoration_extra_denoise_batch(
                image, interp, jobs, stride, True, compact, *parameters,
            )
            if result is not None:
                return result
    if interp is None:
        source = _native_dense(image)
        if generated_2d and compact and _dense_phase(image, stride, generated_phase):
            phase_source_shape = image.shape
            interp = _core.restoration_extra_interpolate_phase(
                source, divmod(generated_phase, stride), stride,
            )
        else:
            interp = _core.restoration_extra_interpolate(source, spatial_rank)
    # Explicit mask iterators may have side effects between yields.
    reuse = ((masks is None or generated_phase is not None)
             and _original_tv(denoise_function) and _tv_chambolle._providers_pristine())
    if generated_2d and not reuse:
        return _masked_dense_2d(image, denoise_function, kwargs, stride, interp,
                                generated_phase, compact, phase_source_shape is not None)
    output = (np.empty(image[masks[0]].shape, dtype=image.dtype) if compact
              else np.zeros_like(image))
    if masks is None:
        masks = (_grid(i, spatial_rank, stride) for i in range(stride**spatial_rank))
    # A custom callback can mutate or retain its argument across calls.
    if reuse:
        input_image = image.copy()
    for mask in masks:
        if not reuse:
            input_image = image.copy()
        input_image[mask] = interp[mask] if phase_source_shape is None else interp
        output[Ellipsis if compact else mask] = denoise_function(input_image, **kwargs)[mask]
        if reuse:
            input_image[mask] = image[mask]
    return output


def _masked_dense_2d(image, denoise_function, kwargs, stride, interp, phase, compact,
                     compact_interp):
    initial_shape = image.shape
    if compact:
        output = np.empty(image[_grid(phase, 2, stride)].shape, dtype=image.dtype)
    else:
        output = np.empty_like(image) if phase is None else np.zeros_like(image)
    jobs = product(range(stride), repeat=2) if phase is None else (divmod(phase, stride),)
    prepare = _core.restoration_extra_phase_input
    store = _core.restoration_extra_phase_store
    for job in jobs:
        if image.shape == initial_shape and _spin_dense(image) and image.dtype == output.dtype:
            input_image = prepare(image, interp, job, stride)
        else:
            mask = tuple(slice(offset, None, stride) for offset in job)
            input_image = image.copy()
            input_image[mask] = interp if compact_interp else interp[mask]
        result = denoise_function(input_image, **kwargs)
        if (_spin_dense(result) and result.flags.owndata and result.shape == initial_shape
                and result.dtype == output.dtype and output.size):
            store(result, output, job, stride, compact)
        else:
            mask = tuple(slice(offset, None, stride) for offset in job)
            output[Ellipsis if compact else mask] = result[mask]
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
    prepared = image.astype(_float_dtype(image), copy=False)
    parameters = [dict(zip(denoise_parameters, values))
                  for values in product(*denoise_parameters.values())]
    ranks = [image.ndim - int(kwargs.get('channel_axis') is not None) for kwargs in parameters]
    stride_int = int(stride)
    source = _native_dense(prepared)
    interpolated = {}
    phase_source_shape = None
    # Preserve the original interpolation snapshot for arbitrary denoisers or
    # parameter protocols that may mutate the caller's array between trials.
    native_masks = (type(approximate_loss) is bool and prepared.ndim == 2
                    and stride_int <= 64 and _spin_dense(prepared)
                    and (_original_tv(denoise_function) or _original_bregman(denoise_function))
                    and all(type(key) is str for key in denoise_parameters)
                    and all(type(value) in (int, float, bool, type(None))
                            for kwargs in parameters for value in kwargs.values())
                    and _tv_chambolle._providers_pristine())
    for rank in dict.fromkeys(ranks):
        if rank <= 0:
            continue
        if rank == 2 and native_masks:
            continue
        if (approximate_loss is True and prepared.ndim == rank == 2 and stride_int <= 64
                and _spin_dense(prepared)):
            phase = stride_int**2 // 2
            phase_job = divmod(phase, stride_int)
            if _dense_phase(prepared, stride_int, phase):
                phase_source_shape = prepared.shape
                interpolated[rank] = _core.restoration_extra_interpolate_phase(
                    source, phase_job, stride_int,
                )
                continue
        interpolated[rank] = _core.restoration_extra_interpolate(source, rank)
    losses = []
    for kwargs, spatial_rank in zip(parameters, ranks):
        compact = False
        if approximate_loss:
            phase = stride_int**spatial_rank // 2
            mask = _grid(phase, spatial_rank, stride_int)
            masks = [mask]
            compact = (prepared.ndim == spatial_rank == 2 and stride_int <= 64
                       and _spin_dense(prepared))
        else:
            phase = None
            mask = Ellipsis
            masks = None
        if spatial_rank == 0:
            denoised = denoise_invariant(image, denoise_function, stride=stride,
                                         masks=masks, denoiser_kwargs=kwargs)
        else:
            denoised = _masked(prepared, denoise_function, kwargs, spatial_rank, stride_int, masks,
                               interpolated.get(spatial_rank), phase, compact,
                               phase_source_shape if spatial_rank == 2 else None)
        values = denoised if compact else denoised[mask]
        squared = None
        if (_spin_dense(image) and image.flags.owndata and _spin_dense(values)
                and image.dtype == values.dtype
                and image.ndim == 2 and (compact or phase is None)
                and (compact or image.shape == values.shape)):
            loss_phase = divmod(phase, stride_int) if compact else (0, 0)
            squared = _core.restoration_extra_squared_error(
                image, values, loss_phase, stride_int if compact else 1,
                np.__version__ == '2.5.4' and np.mean is _NUMPY_MEAN,
            )
            if type(squared) is float:
                losses.append(np.float64(squared))
                continue
        # Use NumPy's warnings/errstate handling for exceptional arithmetic.
        if squared is None:
            squared = (image[mask].astype(prepared.dtype, copy=False) - values) ** 2
        losses.append(np.mean(squared, dtype=np.float64))
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
    roll = _core.restoration_extra_roll
    if (x.ndim == 2 and all(ranges) and len(ranges[0]) * len(ranges[1]) <= 4096):
        prepared = x
        if (_original_tv(func) and type(kwargs) is dict and x.size and x.dtype.isnative
                and x.dtype.metadata is None and x.dtype.kind in 'bui'
                and _tv_chambolle._img_as_float is _TV_CHAMBOLLE_CONVERT):
            parameters = _chambolle_batch_parameters(kwargs)
            if parameters is not None:
                # Plain integer conversion commutes with the circular shift.
                prepared = _native_dense(_TV_CHAMBOLLE_CONVERT(x))
        else:
            parameters = _denoise_batch_parameters(prepared, func, kwargs)
        if parameters is not None:
            jobs = [(row % x.shape[0], col % x.shape[1]) for row, col in product(*ranges)]
            batch = _core.restoration_extra_denoise_batch(
                prepared, None, jobs, 1, False, False, *parameters,
            )
            if batch is not None:
                batch /= len(jobs)
                return batch
    for shift in product(*ranges):
        shape = x.shape
        reduced = (tuple(s % n for s, n in zip(shift, shape))
                   if len(shape) == len(axes) and _spin_dense(x) else None)
        if shift == (0, 0) and len(shape) == 2:
            shifted = x.copy(order='K')
        else:
            shifted = (roll(x, reduced) if reduced is not None
                       else np.roll(x, shift, axis=axes))
        result = func(shifted, **kwargs)
        if (_spin_dense(result) and result.ndim == len(axes) and result.shape == x.shape
                and (mean is None or (_spin_dense(mean) and mean.shape == x.shape
                                      and mean.dtype == result.dtype))):
            if shift == (0, 0) and mean is None:
                mean = result.copy()
            elif reduced is not None and x.shape == shape:
                mean = roll(result, reduced, mean, True)
            else:
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
