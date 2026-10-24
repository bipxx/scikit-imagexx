"""Private matrix_transform shell; exact intake and owned NumPy providers.

The checked projective C++ primitive is used only for certified small dyadic
real coordinates/matrices with pristine source providers. It is not evidence
of public function registration, arbitrary floating arithmetic or performance.
Source ordering follows scikit-image0.26 (BSD; third_party license).
"""
import ctypes
import importlib
from types import BuiltinFunctionType

import numpy as np
from . import _core

_COPY_IF_NEEDED = None if np.lib.NumpyVersion(np.__version__) >= '2.0.0' else False
_PREFIX = 'numpy._core' if int(np.__version__.split('.')[0]) >= 2 else 'numpy.core'
_MULTI = importlib.import_module(_PREFIX + '.multiarray')
_NUMERIC = importlib.import_module(_PREFIX + '.numeric')
_SHAPE = importlib.import_module(_PREFIX + '.shape_base')
_LIMITS = importlib.import_module(_PREFIX + '.getlimits')
_TYPES = importlib.import_module(_PREFIX + '.numerictypes')
_UMATH = importlib.import_module(_PREFIX + '.umath')
_ERRORS = importlib.import_module(_PREFIX + '._ufunc_config')
_TWODIM = importlib.import_module('numpy.lib._twodim_base_impl' if _PREFIX.endswith('_core') else 'numpy.lib.twodim_base')
_ARRAY = _MULTI.ndarray
_F64 = _TYPES.float64
_U64 = _TYPES.uint64
_REAL_TYPES = tuple(getattr(_TYPES, name) for name in ('bool_', 'uint8', 'int8', 'uint16', 'int16',
    'uint32', 'int32', 'uint64', 'int64', 'float16', 'float32', 'float64'))
_DTYPE_CLASSES = frozenset(type(_MULTI.dtype(t)) for t in _REAL_TYPES)
_PROVIDERS = dict(array=_MULTI.array, asarray=_MULTI.asarray, ones=_NUMERIC.ones,
                  hstack=_SHAPE.hstack, where=_MULTI.where, finfo=_LIMITS.finfo,
                  eye=_TWODIM.eye, dtype=_MULTI.dtype)
_ABS = _UMATH.absolute
_FINITE = _UMATH.isfinite
_FLOOR = _UMATH.floor
_MUL = _UMATH.multiply
_BITS_AND = _UMATH.bitwise_and
_GETERR = _ERRORS.geterr
_ALL = _NUMERIC.all
_ANY = _NUMERIC.any
_SIGN = _U64(0x8000000000000000)
_MAGNITUDE = _U64(0x7fffffffffffffff)
_MIN_GRID = _U64(0x3eb0000000000000)  # 2^-20 as double bits.


def _rounding_query():
    # Read-only platform metadata. Missing APIs disable acceleration; they do
    # not prevent execution through the literal owned NumPy provider.
    try:
        import os
        library = ctypes.CDLL('ucrtbase.dll') if os.name == 'nt' else ctypes.CDLL(None)
        function = library.fegetround
        function.argtypes = []
        function.restype = ctypes.c_int
        return function
    except (OSError, AttributeError):
        return None


_ROUND = _rounding_query()


def _pristine():
    return (np.__version__ in ('1.24.4', '2.5.4')
            and all(getattr(np, name, None) is real for name, real in _PROVIDERS.items())
            and _GETERR() == dict(divide='warn', over='warn', under='ignore', invalid='warn')
            and _ROUND is not None and _ROUND() == 0)


def _grid(array):
    """All dyadic products/sums have <=51 significant bits for D<=4.

    Values are bounded by16 and multiples of2^-20; a product is an integer
    <=2^48 times2^-40, and at most five terms fit within2^51. Every homogeneous
    dot operation is exact regardless of BLAS grouping or FMA. Division is the
    same correctly rounded double operation as NumPy. Zero results have a +0
    term from the non-negative-zero homogeneous coefficient. Reject negative
    zero explicitly. Integer bit checks reject subnormals before FP operations
    and make FTZ/DAZ irrelevant throughout this normal-range certificate.
    """
    words = array.view(_U64)
    magnitudes = _BITS_AND(words, _MAGNITUDE)
    if _ANY(words == _SIGN) or _ANY((magnitudes != 0) & (magnitudes < _MIN_GRID)):
        return False
    if not _ALL(_FINITE(array)) or not _ALL(_ABS(array) <= 16.):
        return False
    scaled = _MUL(array, 1048576.)
    return bool(_ALL(scaled == _FLOOR(scaled)))


def _checked_entry():
    function = getattr(_core, 'projective_apply_checked', None)
    if not (type(function) is BuiltinFunctionType and function.__name__ == 'projective_apply_checked'
            and function.__module__ == _core.__name__):
        return None
    return function


def _native(matrix, points):
    # Screening uses exact types and identity checks, never user dtype factories
    # or array/scalar protocols. Protocol intake has already happened once.
    if not _pristine() or type(matrix) is not _ARRAY or type(points) is not _ARRAY:
        return None
    if type(matrix.dtype) not in _DTYPE_CLASSES or type(points.dtype) not in _DTYPE_CLASSES:
        return None
    if (points.ndim != 2 or points.shape[0] == 0 or points.shape[1] not in (2, 3, 4)
            or matrix.shape != (points.shape[1]+1, points.shape[1]+1)
            or points.dtype.kind not in 'biuf' or matrix.dtype.kind not in 'biuf'
            or points.dtype.itemsize > 8 or matrix.dtype.itemsize > 8):
        return None
    entry = _checked_entry()
    if entry is None:
        return None
    points64 = _MULTI.asarray(points, dtype=_F64)
    matrix64 = _MULTI.asarray(matrix, dtype=_F64)
    if not _grid(points64) or not _grid(matrix64):
        return None
    result, provider = entry(points64, matrix64)
    # Checked primitives restore fenv; exceptional arithmetic is replayed by
    # the owned source stages so original warning/error policy remains visible.
    return None if provider else result


def _materialized(matrix, points):
    # No repeated np.array: points is the original single intake result.
    homogeneous = np.hstack((points, np.ones((len(points), 1))))
    transformed = homogeneous @ matrix.T
    divisors = transformed[:, -1]
    divisors = np.where(divisors == 0, np.finfo(float).eps, divisors)
    return transformed[:, :-1] / divisors[:, None]


def matrix_transform(coords, matrix):
    """Apply an array-like homogeneous matrix with source-faithful intake."""
    if matrix is None:
        matrix = np.eye(3)
    else:
        matrix = np.asarray(matrix)
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        raise ValueError('Invalid shape of transformation matrix')
    if not matrix.shape[0] - 1 >= 2:
        # This is nominal exception metadata, not an upstream class call. The
        # function's original private constructor names this exact class.
        raise NotImplementedError("Input for <class 'skimage.transform._geometric.ProjectiveTransform'> should result in transform of >=2D")
    points = np.array(coords, copy=_COPY_IF_NEEDED, ndmin=2)
    result = _native(matrix, points)
    return _materialized(matrix, points) if result is None else result
