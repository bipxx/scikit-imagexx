"""Owned GLCM count/property arithmetic; explicit NumPy validation/conversion boundary.

Ordering follows scikit-image0.26 feature/texture.py (BSD-3-Clause).
No upstream image algorithm/provider is called.
"""
import operator
import numpy as np
from . import _core


def _check_nd(array, ndim, arg_name='image'):
    array = np.asanyarray(array)
    if array.size == 0:
        raise ValueError('The parameter `%s` cannot be an empty array' % arg_name)
    if array.ndim != ndim:
        raise ValueError('The parameter `%s` must be a %s-dimensional array' % (arg_name, ndim))


def graycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False):
    _check_nd(image, 2)
    _check_nd(distances, 1, 'distances')
    _check_nd(angles, 1, 'angles')
    image = np.ascontiguousarray(image)
    # Source-ordered boundary extrema/comparison are explicit NumPy providers;
    # histogram, wrap, symmetry and normalization are owned native operations.
    image_max = image.max()
    if np.issubdtype(image.dtype, np.floating):
        raise ValueError('Float images are not supported by graycomatrix. Convert the image to an unsigned integer type.')
    if image.dtype not in (np.uint8, np.int8) and levels is None:
        raise ValueError('The levels argument is required for data types other than uint8. The resulting matrix will be at least levels ** 2 in size.')
    if np.issubdtype(image.dtype, np.signedinteger) and np.any(image < 0):
        raise ValueError('Negative-valued images are not supported.')
    if levels is None:
        levels = 256
    if image_max >= levels:
        raise ValueError('The maximum grayscale value in the image should be smaller than the number of levels.')
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    angles = np.ascontiguousarray(angles, dtype=np.float64)
    # Allocation integer coercion precedes Cython fused-buffer admission.
    levels = operator.index(levels)
    if image.dtype.kind not in 'biu':
        raise TypeError('No matching signature found')
    if not image.flags.writeable:
        raise ValueError('buffer source array is read-only')
    if not image.dtype.isnative:
        raise ValueError('Big-endian buffer not supported on little-endian compiler')
    workers = _core.get_num_threads()
    return _core.graycomatrix_apply(image, distances, angles, levels, bool(symmetric), bool(normed), workers)


def _policy(operation, flags):
    # Scalar metadata probes reuse the admitted IGG source-phase policy route.
    # The callback runs at the native operation boundary before later stages;
    # no provider sees an image or computes a property.
    if operation == 'log':
        if flags != 1: raise NotImplementedError('graycoprops logarithm status domain')
        np.log(np.array([-1.]))
    elif operation == 'sqrt':
        if flags != 1: raise NotImplementedError('graycoprops sqrt status domain')
        np.sqrt(np.array([-1.]))
    elif operation == 'divide':
        limits = np.finfo(np.float64); left=[]; right=[]
        if flags & 2: left.append(1.); right.append(0.)
        if flags & 4: left.append(limits.max); right.append(.5)
        if flags & 8: left.append(limits.tiny); right.append(limits.max)
        if flags & 1: left.append(0.); right.append(0.)
        np.divide(np.array(left), np.array(right))
    else:
        raise NotImplementedError('graycoprops unassessed '+operation+' policy status')


def graycoprops(P, prop='contrast'):
    _check_nd(P, 4, 'P')
    num_level, num_level2, num_dist, num_angle = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')
    # Required source conversion is a provider (complex warning and object
    # conversion included); no NumPy image-property arithmetic is delegated.
    prepared = P.astype(np.float64)
    # Property refusal occurs only after native normalization/policy delivery.
    return _core.graycoprops_apply(prepared, prop, _core.get_num_threads(), _policy)
