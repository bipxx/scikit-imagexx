"""Blur metric with owned uniform and Sobel sampling; NumPy reductions remain providers."""
import numpy as np
from . import _core

try:
    from numpy import AxisError
except ImportError:
    from numpy.exceptions import AxisError

_EPSILON = np.spacing(np.float64(1))


def _sobel(image, axis, owned):
    ndim = image.ndim
    if image.dtype.kind == 'f':
        from skimage._shared.utils import _supported_float_type
        image = image.astype(_supported_float_type(image.dtype), copy=False)
    else:
        from skimage.util import img_as_float
        image = img_as_float(image)
    output = np.zeros(image.shape, dtype=image.dtype)
    shape = [1] * ndim
    shape[axis] = 3
    kernel = np.array([1, 0, -1]).reshape(shape)
    for smooth_dim in list(set(range(ndim)) - {axis}):
        shape = [1] * ndim
        shape[smooth_dim] = 3
        kernel = kernel * (np.array([1, 2, 1]) / 4).reshape(shape)
    if owned:
        # SciPy convolve uses ordered double products/sums, then one output
        # cast. The sparse float32 path instead rounds at every tap, so feed
        # this existing primitive double storage even for float32 images.
        reflected = kernel[(slice(None, None, -1),) * ndim]
        indices = np.nonzero(reflected)
        weights = np.ascontiguousarray(reflected[indices], dtype=np.float64)
        coordinates = list(zip(*(index.tolist() for index in indices)))
        prepared = np.require(image, dtype=np.float64, requirements=('C', 'A', 'O'))
        ax_output, flags = _core.correlate_sparse(
            prepared, kernel.shape, coordinates, weights, 0, False)
        # SciPy performs this output cast in C without NumPy FP dispatch.
        with np.errstate(all='ignore'):
            ax_output = ax_output.astype(image.dtype, copy=False)
    else:
        from scipy import ndimage as ndi
        ax_output = ndi.convolve(image, kernel, mode='reflect', cval=0.0)
    output += ax_output
    return output


def blur_effect(image, h_size=11, channel_axis=None, reduce_func=np.max):
    if channel_axis is not None:
        try:
            image = np.moveaxis(image, channel_axis, -1)
        except AxisError:
            print('channel_axis must be one of the image array dimensions')
            raise
        except TypeError:
            print('channel_axis must be an integer')
            raise
        from skimage.color import rgb2gray
        image = rgb2gray(image)
    n_axes = image.ndim
    from skimage.util import img_as_float
    image = img_as_float(image)
    shape = image.shape
    if n_axes and any(size == 0 for size in shape):
        raise ValueError("blur filtering requires positive spatial axes; zero-axis compatibility is unassessed")
    B = []
    workers = _core.get_num_threads()
    slices = tuple([slice(2, s - 1) for s in shape])
    for ax in range(n_axes):
        filt_im = None
        if (type(image) is np.ndarray and image.dtype.char in 'fd'
                and 1 <= image.ndim <= 3 and all(image.shape)
                and isinstance(h_size, (int, np.integer)) and h_size > 0):
            prepared = np.require(image, dtype=image.dtype.newbyteorder('='),
                                  requirements=('C', 'A', 'O'))
            filt_im = _core.uniform_filter1d(prepared, ax, int(h_size), workers)
        owned = filt_im is not None
        if not owned:
            from scipy import ndimage as ndi
            filt_im = ndi.uniform_filter1d(image, h_size, axis=ax)
        im_sharp = np.abs(_sobel(image, ax, owned))
        im_blur = np.abs(_sobel(filt_im, ax, owned))
        im_sharp = np.maximum(_EPSILON, im_sharp)
        im_blur = np.maximum(_EPSILON, im_blur)
        T = np.maximum(0, im_sharp - im_blur)
        M1 = np.sum(im_sharp[slices])
        M2 = np.sum(T[slices])
        B.append(np.abs(M1 - M2) / M1)
    return B if reduce_func is None else reduce_func(B)
