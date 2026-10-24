"""Private matrix_transform shell; exact intake and owned NumPy providers.

The general finite C++ kernel uses authenticated pinned NumPy/OpenBLAS
arithmetic profiles with source-faithful intake and padding. Other domains use
owned NumPy stages. This private shell is not public registration/performance.
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
                  eye=_TWODIM.eye, dtype=_MULTI.dtype, getbufsize=_ERRORS.getbufsize)
_GETERR = _ERRORS.geterr


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
            and _ERRORS.getbufsize() == 8192
            and _ROUND is not None and _ROUND() == 0)


# Each pair pins the actual NumPy loop binary and its bundled OpenBLAS.
# Other builds/CPUs remain provider-only, even if they have the same version.
_RUNTIME_PINS = {
 ('2.5.4', 'f37332834048e3aef93a33ff53f558a319cad12009defb2231ee1ba4aa6105d8'):
  ('ed4f167a5330424524f45258e7ca2c8d7f03c0cba1f30ee4985553ac1f88ecf8', 'scipy_openblas_get_corename64_'),
 ('2.5.4', '405abe5c8f52c1d15b71bafbd155693237b6e1742bc598638b7fc0626bcc1576'):
  ('432e1c1739731468ba305771af96773939e43016643e2d630f58cb45b6670e96', 'scipy_openblas_get_corename64_'),
 ('1.24.4', 'db283346a443cde0c8cb4dcdde56490714d2c34e3ad16fe2c30ae8c20458fc86'):
  ('554bde1d8a0c71d8dc21ae74de05c44da4fff5dbc6791a819f6acf5adfe90bd9', 'openblas_get_corename64_'),
}
_THREAD_QUERY = None


@__import__('functools').lru_cache(maxsize=1)
def _runtime_profile():
    # Read-only file/dispatch authentication, never numerical calibration.
    import hashlib
    from pathlib import Path
    global _THREAD_QUERY
    try:
        umath = importlib.import_module(_PREFIX+'._multiarray_umath')
        umath_path=getattr(umath,'__file__',None);numpy_path=getattr(np,'__file__',None)
        if type(umath_path) is not str or type(numpy_path) is not str:
            return None
        key = (np.__version__, hashlib.sha256(Path(umath_path).read_bytes()).hexdigest())
        pin = _RUNTIME_PINS.get(key)
        if pin is None:
            return None
        for path in (Path(numpy_path).parent.parent/'numpy.libs').glob('*openblas*'):
            if hashlib.sha256(path.read_bytes()).hexdigest() == pin[0]:
                library = ctypes.CDLL(str(path))
                query = getattr(library, pin[1]); query.restype = ctypes.c_char_p
                if query() == b'Haswell':
                    _THREAD_QUERY=getattr(library,pin[1].replace('corename','num_threads'))
                    _THREAD_QUERY.restype=ctypes.c_int
                    return key[0]
    except (OSError, AttributeError):
        return None
    return None


def _checked_entry():
    function = getattr(_core, 'matrix_transform_run', None)
    if not (type(function) is BuiltinFunctionType and function.__name__ == 'matrix_transform_run'
            and function.__module__ == _core.__name__
            and getattr(_core, 'matrix_transform_interface', None) == 'matrix-transform-math-v1'
            and getattr(_core, 'matrix_transform_capabilities', 0) == 31):
        return None
    return function


def _blasable(a, rows, columns):
    return (a[1] == 8 and a[0] % 8 == 0 and a[0]//8 >= columns
            and a[0]//8 <= (1 << 63)-2)


def _arithmetic_profile(h, matrix, version):
    # h and matrix.T are the original actual padded operands, before ANY
    # convenience copy. NumPy1.24's non-BLASable path differs from2.5's copy.
    transposed = matrix.T
    m = matrix.shape[0]
    c = _blasable(transposed.strides, m, m)
    f = _blasable(transposed.strides[::-1], m, m)
    if h.shape[0] == 1:
        if not (c or f):
            return 2 if version == '1.24.4' else 5
        # The ufunc buffers an unaligned operand in traversal order before its
        # GEMV call. The exact pinned iterator/loop bytes and all unaligned
        # fixed cohorts are separately authenticated in private qualification.
        return 3 if f and matrix.flags.aligned else 4
    return 2 if version == '1.24.4' and not (c or f) else 1


def _has_nonfinite_matrix(matrix):
    if matrix.dtype.kind != 'f':
        return False
    unsigned=_MULTI.dtype('u'+str(matrix.itemsize)).newbyteorder(matrix.dtype.byteorder)
    words=matrix.view(unsigned)
    mask={2:0x7c00,4:0x7f800000,8:0x7ff0000000000000}[matrix.itemsize]
    return bool(_NUMERIC.any(_UMATH.bitwise_and(words,mask)==mask))


def _dyadic(array):
    # Preserve the earlier exact real-matrix dtype coverage. This certificate
    # is association-independent; it does not pretend a general cast profile.
    words=array.view(_U64);mag=_UMATH.bitwise_and(words,_U64(0x7fffffffffffffff))
    if _NUMERIC.any(words==_U64(0x8000000000000000)) or _NUMERIC.any((mag!=0)&(mag<_U64(0x3eb0000000000000))):
        return False
    if not _NUMERIC.all(_UMATH.isfinite(array)) or not _NUMERIC.all(_UMATH.absolute(array)<=16.):
        return False
    scaled=_UMATH.multiply(array,1048576.)
    return bool(_NUMERIC.all(scaled==_UMATH.floor(scaled)))


def _native(matrix, homogeneous, trusted=True):
    if not trusted or not _pristine() or type(matrix) is not _ARRAY or type(homogeneous) is not _ARRAY:
        return None
    if (type(matrix.dtype) not in _DTYPE_CLASSES or type(homogeneous.dtype) not in _DTYPE_CLASSES
            or matrix.dtype.kind not in 'biuf' or matrix.dtype.itemsize>8
            or homogeneous.dtype.type is not _F64
            or not homogeneous.dtype.isnative
            or homogeneous.ndim != 2 or not homogeneous.shape[0]
            or homogeneous.shape[1] not in (3,4,5)
            or matrix.shape != (homogeneous.shape[1], homogeneous.shape[1])):
        return None
    entry = _checked_entry()
    if entry is None:
        return None
    # A protocol callback may temporarily replace ones/hstack, restore the
    # public provider identity, and return altered padding. Certify the ACTUAL
    # padded operand, not only the provider identities observed afterwards.
    if not (homogeneous.flags.c_contiguous or homogeneous.flags.f_contiguous):
        return None
    if not _NUMERIC.all(homogeneous[:, -1].view(_U64)==_U64(0x3ff0000000000000)):
        return None
    version = _runtime_profile()
    if version is None:
        return None
    if _THREAD_QUERY is None or _THREAD_QUERY()!=1:
        return None
    if matrix.dtype.type is not _F64 or not matrix.dtype.isnative:
        # Never quiet a signaling NaN or emit a cast warning merely to decide
        # that the literal owned matmul provider is required.
        if _has_nonfinite_matrix(matrix):
            return None
        converted=_MULTI.asarray(matrix,dtype=_F64)
        if not _dyadic(converted) or not _dyadic(homogeneous):
            return None
        profile=1 if homogeneous.shape[0]>1 else 2
        matrix=converted
    else:
        profile=_arithmetic_profile(homogeneous,matrix,version)
    result, status = entry(homogeneous[:, :-1], matrix, profile)
    return result if status == 0 else None


def _materialized(matrix, homogeneous):
    transformed = homogeneous @ matrix.T
    divisors = transformed[:, -1]
    divisors = np.where(divisors == 0, np.finfo(float).eps, divisors)
    return transformed[:, :-1] / divisors[:, None]


def matrix_transform(coords, matrix):
    """Apply an array-like homogeneous matrix with source-faithful intake."""
    trusted = _pristine()
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
    homogeneous = np.hstack((points, np.ones((len(points), 1))))
    result = _native(matrix, homogeneous, trusted)
    return _materialized(matrix, homogeneous) if result is None else result
