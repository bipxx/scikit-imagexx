"""Private native transform classes with explicit NumPy/SciPy solver providers.

Finite real coordinate application uses C++ projective-coordinate-v1. Matrix
inversion, composition, normalization, SVD/Umeyama and Euler construction retain
runtime NumPy/SciPy numerical policy. No upstream transform classes are called.

Class/solver algorithms ported from scikit-image 0.26.0.
Copyright: 2009-2022 the scikit-image team (BSD-3-Clause).
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
from abc import ABC, abstractmethod
from functools import wraps
import inspect
import math
import textwrap
import warnings
from typing import Self
import numpy as np
from scipy import spatial
from . import _core

NP_COPY_IF_NEEDED = None if np.lib.NumpyVersion(np.__version__) >= '2.0.0' else False


def _apply_numpy(matrix, points):
    points = np.array(points, copy=NP_COPY_IF_NEEDED, ndmin=2)
    homogeneous = np.hstack((points, np.ones((len(points), 1))))
    transformed = homogeneous @ matrix.T
    divisors = transformed[:, -1]
    divisors = np.where(divisors == 0, np.finfo(float).eps, divisors)
    return transformed[:, :-1] / divisors[:, None]


def _apply_homogeneous(matrix, points):
    points = np.array(points, copy=NP_COPY_IF_NEEDED, ndmin=2)
    policy = np.geterr()
    if (type(matrix) is not np.ndarray or matrix.ndim != 2 or points.ndim != 2
        or matrix.shape[0] != matrix.shape[1] or points.shape[1] + 1 != matrix.shape[0]
        or points.shape[1] < 1 or points.dtype.kind not in 'biuf' or matrix.dtype.kind not in 'biuf'
        or points.dtype.itemsize > 8 or matrix.dtype.itemsize > 8
        or policy != dict(divide='warn', over='warn', under='ignore', invalid='warn')):
        return _apply_numpy(matrix, points)
    points64 = np.asarray(points, dtype=np.float64)
    matrix64 = np.asarray(matrix, dtype=np.float64)
    result, provider = _core.projective_apply_checked(points64, matrix64)
    if provider:
        # Replay exceptional arithmetic through the runtime math provider so
        # warnings and configured NumPy error callbacks remain observable.
        return _apply_numpy(matrix, points)
    return result


def _deprecate_estimate(function, class_name=None):
    name = function.__qualname__.split('.')[0] if class_name is None else class_name
    @wraps(function)
    def estimate(*args, **kwargs):
        warnings.warn('`estimate` is deprecated since version 0.26 and will be removed in version 2.2. '
                      f'Please use `{name}.from_estimate` class constructor instead.', FutureWarning, stacklevel=2)
        return function(*args, **kwargs)
    return estimate


def _deprecate_inherited_estimate(cls):
    def estimate(self, *args, **kwargs):
        return self._estimate(*args, **kwargs) is None
    inherited = getattr(cls, 'estimate').__wrapped__
    estimate.__signature__ = inspect.signature(inherited)
    cls.estimate = _deprecate_estimate(estimate, cls.__name__)
    return cls


def _update_from_estimate_docstring(cls):
    inherited = getattr(cls, 'from_estimate')
    def from_estimate(cls, *args, **kwargs):
        return inherited(*args, **kwargs)
    from_estimate.__signature__ = inspect.signature(inherited)
    cls.from_estimate = classmethod(from_estimate)
    return cls


class FailedEstimationAccessError(AttributeError):
    pass


class _GeometricTransform(ABC):
    @abstractmethod
    def __call__(self, coords):
        raise NotImplementedError
    @property
    @abstractmethod
    def inverse(self):
        raise NotImplementedError
    def residuals(self, src, dst):
        return np.sqrt(np.sum((self(src) - dst) ** 2, axis=1))
    @classmethod
    @abstractmethod
    def identity(cls, dimensionality=None):
        raise NotImplementedError
    @classmethod
    def _prepare_estimation(cls, src, dst):
        src, dst = np.asarray(src), np.asarray(dst)
        return cls.identity(src.shape[1]), src, dst
    @classmethod
    def from_estimate(cls, src, dst, *args, **kwargs):
        return _from_estimate(cls, src, dst, *args, **kwargs)

class FailedEstimation:
    error_cls = FailedEstimationAccessError
    hint = "You can check for a failed estimation by truth testing the returned object. For failed estimations, `bool(estimation_result)` will be `False`. E.g.\n\n    if not estimation_result:\n        raise RuntimeError(f'Failed estimation: {estimation_result}')"

    def __init__(self, message):
        self.message = message

    def __bool__(self):
        return False

    def __repr__(self):
        return f'{type(self).__name__}({self.message!r})'

    def __str__(self):
        return self.message

    def __call__(self, *args, **kwargs):
        msg = f'{type(self).__name__} is not callable. {self.message}\n\nHint: {self.hint}'
        raise self.error_cls(msg)

    def __getattr__(self, name):
        msg = f'{type(self).__name__} has no attribute {name!r}. {self.message}\n\nHint: {self.hint}'
        raise self.error_cls(msg)

def _from_estimate(cls, src, dst, *args, **kwargs):
    tf, src, dst = cls._prepare_estimation(src, dst)
    msg = tf._estimate(src, dst, *args, **kwargs)
    return tf if msg is None else FailedEstimation(f'{cls.__name__}: {msg}')

def _calc_center_normalize(points, scaling='rms'):
    n, d = points.shape
    scaling = scaling.lower()
    matrix = np.eye(d + 1)
    if scaling == 'raw':
        return matrix
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    if scaling == 'rms':
        divisor = np.sqrt(np.mean(centered ** 2))
    elif scaling == 'mrs':
        divisor = np.mean(np.sqrt(np.sum(centered ** 2, axis=1))) / np.sqrt(d)
    else:
        raise ValueError(f'Unexpected "scaling" of "{scaling}"')
    if divisor == 0:
        return matrix + np.nan
    matrix[:d, d] = -centroid
    matrix[:d, :] /= divisor
    return matrix

def _center_and_normalize_points(points, scaling='rms'):
    matrix = _calc_center_normalize(points, scaling)
    if not np.all(np.isfinite(matrix)):
        return (matrix + np.nan, np.full_like(points, np.nan))
    return (matrix, _apply_numpy(matrix, points))

def _umeyama(src, dst, estimate_scale):
    src = np.asarray(src)
    dst = np.asarray(dst)
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    tol = S.max() * np.max(A.shape) * np.finfo(float).eps
    rank = np.count_nonzero(S > tol)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T

def _euler_rotation_matrix(angles, degrees=False):
    return spatial.transform.Rotation.from_euler('XYZ', angles=angles, degrees=degrees).as_matrix()

class _HMatrixTransform(_GeometricTransform):

    def __init__(self, matrix=None, *, dimensionality=None):
        if matrix is None:
            d = 2 if dimensionality is None else dimensionality
            matrix = np.eye(d + 1)
        else:
            matrix = np.asarray(matrix)
        self._check_matrix(matrix, dimensionality)
        self._check_dims(matrix.shape[0] - 1)
        self.params = matrix

    def _check_matrix(self, matrix, dimensionality):
        if dimensionality is not None:
            if dimensionality != matrix.shape[0] - 1:
                raise ValueError(f'Dimensionality {dimensionality} does not match matrix {matrix}')
        m = matrix.shape[0]
        if matrix.shape != (m, m):
            raise ValueError('Invalid shape of transformation matrix')

    def _check_dims(self, d):
        if d == 2:
            return
        raise NotImplementedError(f'Input for {type(self)} should result in 2D transform')

    @classmethod
    def identity(cls, dimensionality=None):
        d = 2 if dimensionality is None else dimensionality
        return cls(matrix=np.eye(d + 1))

    @property
    def dimensionality(self):
        return self.matrix.shape[0] - 1

class ProjectiveTransform(_HMatrixTransform):
    scaling = 'rms'

    @property
    def _coeff_inds(self):
        return range(self.params.size - 1)

    def _check_dims(self, d):
        if d >= 2:
            return
        raise NotImplementedError(f'Input for {type(self)} should result in transform of >=2D')

    @property
    def _inv_matrix(self):
        return np.linalg.inv(self.params)

    def __array__(self, dtype=None, copy=None):
        return self.params if dtype is None else self.params.astype(dtype)

    def __call__(self, coords):
        return _apply_homogeneous(self.params, coords)

    @property
    def inverse(self):
        return type(self)(matrix=self._inv_matrix)

    @classmethod
    def from_estimate(cls, src, dst, weights=None):
        return super().from_estimate(src, dst, weights)

    def _estimate(self, src, dst, weights=None):
        src = np.asarray(src)
        dst = np.asarray(dst)
        n, d = src.shape
        fail_matrix = np.full((d + 1, d + 1), np.nan)
        src_matrix, src = _center_and_normalize_points(src)
        dst_matrix, dst = _center_and_normalize_points(dst)
        if not np.all(np.isfinite(src_matrix + dst_matrix)):
            self.params = fail_matrix
            return 'Scaling generated NaN values'
        A = np.zeros((n * d, (d + 1) ** 2))
        for ddim in range(d):
            A[ddim * n:(ddim + 1) * n, ddim * (d + 1):ddim * (d + 1) + d] = src
            A[ddim * n:(ddim + 1) * n, ddim * (d + 1) + d] = 1
            A[ddim * n:(ddim + 1) * n, -d - 1:-1] = src
            A[ddim * n:(ddim + 1) * n, -1] = -1
            A[ddim * n:(ddim + 1) * n, -d - 1:] *= -dst[:, ddim:ddim + 1]
        A = A[:, list(self._coeff_inds) + [-1]]
        if weights is None:
            _, _, V = np.linalg.svd(A)
        else:
            weights = np.asarray(weights)
            W = np.diag(np.tile(np.sqrt(weights / np.max(weights)), d))
            _, _, V = np.linalg.svd(W @ A)
        H = np.zeros((d + 1, d + 1))
        if np.isclose(V[-1, -1], 0):
            self.params = fail_matrix
            return 'Right singular vector has 0 final element'
        H.flat[list(self._coeff_inds) + [-1]] = -V[-1, :-1] / V[-1, -1]
        H[d, d] = 1
        H = np.linalg.inv(dst_matrix) @ H @ src_matrix
        H /= H[-1, -1]
        self.params = H
        return None

    def __add__(self, other):
        if isinstance(other, ProjectiveTransform):
            if type(self) == type(other):
                tform = self.__class__
            else:
                tform = ProjectiveTransform
            return tform(other.params @ self.params)
        else:
            raise TypeError('Cannot combine transformations of differing types.')

    def __nice__(self):
        if not hasattr(self, 'params'):
            return '<not yet initialized>'
        npstring = np.array2string(self.params, separator=', ')
        return 'matrix=\n' + textwrap.indent(npstring, '    ')

    def __repr__(self):
        return f'<{type(self).__name__}({self.__nice__()}) at {hex(id(self))}>'

    def __str__(self):
        return f'<{type(self).__name__}({self.__nice__()})>'

    @property
    def dimensionality(self):
        return self.params.shape[0] - 1

    @classmethod
    def identity(cls, dimensionality=None):
        return super().identity(dimensionality=dimensionality)

    @_deprecate_estimate
    def estimate(self, src, dst, weights=None):
        return self._estimate(src, dst, weights) is None

@_update_from_estimate_docstring
@_deprecate_inherited_estimate
class AffineTransform(ProjectiveTransform):

    def __init__(self, matrix=None, *, scale=None, shear=None, rotation=None, translation=None, dimensionality=None):
        n_srst_none = sum((p is None for p in (scale, rotation, shear, translation)))
        if n_srst_none != 4:
            if matrix is not None:
                raise ValueError('Do not specify any implicit parameters when matrix is specified.')
            if dimensionality is not None and dimensionality > 2:
                raise ValueError('Implicit parameters only valid for 2D transforms')
            matrix = self._srst2matrix(scale, rotation, shear, translation)
            if matrix.shape[0] != 3:
                raise ValueError('Implicit parameters must give 2D transforms')
        super().__init__(matrix=matrix, dimensionality=dimensionality)

    @property
    def _coeff_inds(self):
        return range(self.dimensionality * (self.dimensionality + 1))

    def _srst2matrix(self, scale, rotation, shear, translation):
        scale = (1, 1) if scale is None else scale
        sx, sy = (scale, scale) if np.isscalar(scale) else scale
        rotation = 0 if rotation is None else rotation
        if not np.isscalar(rotation):
            raise ValueError('rotation must be scalar (2D rotation)')
        shear = 0 if shear is None else shear
        shear_x, shear_y = (shear, 0) if np.isscalar(shear) else shear
        translation = (0, 0) if translation is None else translation
        if np.isscalar(translation):
            raise ValueError('translation must be length 2')
        a2, b2 = translation
        a0 = sx * (math.cos(rotation) + math.tan(shear_y) * math.sin(rotation))
        a1 = -sy * (math.tan(shear_x) * math.cos(rotation) + math.sin(rotation))
        b0 = sx * (math.sin(rotation) - math.tan(shear_y) * math.cos(rotation))
        b1 = -sy * (math.tan(shear_x) * math.sin(rotation) - math.cos(rotation))
        return np.array([[a0, a1, a2], [b0, b1, b2], [0, 0, 1]])

    @property
    def scale(self):
        if self.dimensionality != 2:
            return np.sqrt(np.sum(self.params ** 2, axis=0))[:self.dimensionality]
        ss = np.sum(self.params ** 2, axis=0)
        ss[1] = ss[1] / (math.tan(self.shear) ** 2 + 1)
        return np.sqrt(ss)[:self.dimensionality]

    @property
    def rotation(self):
        if self.dimensionality != 2:
            raise NotImplementedError('The rotation property is only implemented for 2D transforms.')
        return math.atan2(self.params[1, 0], self.params[0, 0])

    @property
    def shear(self):
        if self.dimensionality != 2:
            raise NotImplementedError('The shear property is only implemented for 2D transforms.')
        beta = math.atan2(-self.params[0, 1], self.params[1, 1])
        return beta - self.rotation

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]

class EuclideanTransform(ProjectiveTransform):
    _estimate_scale = False

    def __init__(self, matrix=None, *, rotation=None, translation=None, dimensionality=None):
        n_rt_none = sum((p is None for p in (rotation, translation)))
        if n_rt_none != 2:
            if matrix is not None:
                raise ValueError('Do not specify any implicit parameters when matrix is specified.')
            n_dims, chk_msg = self._rt2ndims_msg(rotation, translation)
            if chk_msg is not None:
                raise ValueError(chk_msg)
            matrix = self._rt2matrix(rotation, translation, n_dims)
        super().__init__(matrix=matrix, dimensionality=dimensionality)

    def _rt2ndims_msg(self, rotation, translation):
        if rotation is not None:
            N = 1 if np.isscalar(rotation) else len(rotation)
            msg = '``rotations`` must be scalar (3D) or length 3 (3D)' if N not in (1, 3) else None
            return (2 if N == 1 else N, msg)
        if translation is not None:
            return (2 if np.isscalar(translation) else len(translation), None)
        return (None, None)

    def _rt2matrix(self, rotation, translation, n_dims):
        if translation is None:
            translation = (0,) * n_dims
        if rotation is None:
            rotation = 0 if n_dims == 2 else np.zeros(3)
        matrix = np.eye(n_dims + 1)
        if n_dims == 2:
            cos_r, sin_r = (math.cos(rotation), math.sin(rotation))
            matrix[:2, :2] = [[cos_r, -sin_r], [sin_r, cos_r]]
        elif n_dims == 3:
            matrix[:3, :3] = _euler_rotation_matrix(rotation)
        matrix[0:n_dims, n_dims] = translation
        return matrix

    @classmethod
    def from_estimate(cls, src, dst) -> Self | FailedEstimation:
        return _from_estimate(cls, src, dst)

    def _estimate(self, src, dst):
        self.params = _umeyama(src, dst, self._estimate_scale)
        return 'Poor conditioning for estimation' if np.any(np.isnan(self.params)) else None

    @property
    def rotation(self):
        if self.dimensionality == 2:
            return math.atan2(self.params[1, 0], self.params[1, 1])
        elif self.dimensionality == 3:
            return self.params[:3, :3]
        else:
            raise NotImplementedError('Rotation only implemented for 2D and 3D transforms.')

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]

    @_deprecate_estimate
    def estimate(self, src, dst):
        return self._estimate(src, dst) is None

@_update_from_estimate_docstring
@_deprecate_inherited_estimate
class SimilarityTransform(EuclideanTransform):
    _estimate_scale = True

    def __init__(self, matrix=None, *, scale=None, rotation=None, translation=None, dimensionality=None):
        n_srt_none = sum((p is None for p in (scale, rotation, translation)))
        if n_srt_none != 3:
            if matrix is not None:
                raise ValueError('Do not specify any implicit parameters when matrix is specified.')
            self._check_scale(scale, (rotation, translation), dimensionality)
            if scale is not None and (not np.isscalar(scale)):
                n_dims, chk_msg = (len(scale), None)
            else:
                n_dims, chk_msg = self._rt2ndims_msg(rotation, translation)
            if chk_msg is not None:
                raise ValueError(chk_msg)
            n_dims = n_dims if n_dims is not None else dimensionality if dimensionality is not None else 2
            matrix = self._rt2matrix(rotation, translation, n_dims)
            if scale not in (None, 1):
                matrix[:n_dims, :n_dims] *= scale
        super().__init__(matrix=matrix, dimensionality=dimensionality)

    def _check_scale(self, scale, other_params, dimensionality):
        if dimensionality in (None, 2) or scale is None or (not np.isscalar(scale)):
            return
        if all((p is None for p in other_params)):
            warnings.warn('In the future, it will be a ValueError to pass a scalar `scale` value with a ``dimensionality`` > 2\n,and without other implicit parameters to indicate the dimensionality of the transform.\nPlease indicate dimensionality by passing a vector of suitable length to `scale`.', FutureWarning, stacklevel=2)

    @property
    def scale(self):
        if self.dimensionality == 2:
            return np.sqrt(np.linalg.det(self.params))
        elif self.dimensionality == 3:
            return np.cbrt(np.linalg.det(self.params))
        else:
            raise NotImplementedError('Scale is only implemented for 2D and 3D.')

# Preserve the pinned inspect.signature return-annotation spelling while
# executable annotations and failed results refer to our own native class.
class _PinnedFailureAnnotation:
    def __repr__(self):
        return 'Union[Self, skimage._shared.utils.FailedEstimation]'
for _cls in (EuclideanTransform, SimilarityTransform):
    _method = _cls.__dict__['from_estimate'].__func__
    _method.__signature__ = inspect.signature(_method).replace(return_annotation=_PinnedFailureAnnotation())
del _cls, _method
