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
_RUNTIME_CORENAME = None
_FORCED_CPU = ('OPENBLAS_CORETYPE', 'NPY_DISABLE_CPU_FEATURES',
               'NPY_ENABLE_CPU_FEATURES', 'MKL_DEBUG_CPU_TYPE', 'MKL_SERVICE_FORCE_INTEL')
_RUNTIME_CORES = {
    'ed4f167a5330424524f45258e7ca2c8d7f03c0cba1f30ee4985553ac1f88ecf8': frozenset(('Haswell',)),
    '432e1c1739731468ba305771af96773939e43016643e2d630f58cb45b6670e96': frozenset(('Haswell', 'SkylakeX')),
    '554bde1d8a0c71d8dc21ae74de05c44da4fff5dbc6791a819f6acf5adfe90bd9': frozenset(('Haswell', 'Cooperlake', 'Zen')),
}


def _unforced():
    import os
    return (not any(name in os.environ for name in _FORCED_CPU)
            and not any(os.environ.get(name) for name in ('LD_PRELOAD', 'LD_AUDIT')))


def _pe_imports(data):
    # Parse the already hash-qualified UMATH import table, never load a candidate
    # DLL merely because its file happens to be in numpy.libs.
    import struct
    if data[:2] != b'MZ':
        raise ValueError('not PE')
    off = struct.unpack_from('<I', data, 0x3c)[0]
    if data[off:off+4] != b'PE\0\0':
        raise ValueError('not PE')
    count = struct.unpack_from('<H', data, off+6)[0]
    size = struct.unpack_from('<H', data, off+20)[0]
    opt = off+24
    if struct.unpack_from('<H', data, opt)[0] != 0x20b:
        raise ValueError('not PE32+')
    rva = struct.unpack_from('<I', data, opt+120)[0]
    sections = []
    for i in range(count):
        vs, va, rs, rp = struct.unpack_from('<IIII', data, opt+size+i*40+8)
        sections.append((va, max(vs, rs), rp))
    def address(value):
        for va, length, raw in sections:
            result = raw+value-va
            if va <= value < va+length and 0 <= result < len(data):
                return result
        raise ValueError('RVA outside file')
    pos = address(rva)
    names = []
    for _ in range(1000):
        row = struct.unpack_from('<IIIII', data, pos)
        pos += 20
        if not any(row):
            return names
        n = address(row[3])
        names.append(data[n:data.index(b'\0', n)].decode('ascii'))
    raise ValueError('unterminated import table')


def _loaded_blas(umath_path, umath_bytes):
    import os
    from pathlib import Path
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        kernel.GetModuleFileNameW.restype = ctypes.c_uint32
        kernel.GetModuleHandleExW.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        kernel.GetModuleHandleExW.restype = ctypes.c_int
        names = [n for n in _pe_imports(umath_bytes) if 'openblas' in n.lower()]
        if len(names) != 1:
            return None
        handle = kernel.GetModuleHandleW(names[0])
        if not handle:
            return None
        def module_path(value):
            buf = ctypes.create_unicode_buffer(32768)
            if not kernel.GetModuleFileNameW(value, buf, len(buf)):
                raise OSError('loaded DLL path unavailable')
            return Path(buf.value).resolve()
        path = module_path(handle)
        library = ctypes.CDLL(None, handle=handle)
        def resolve(function):
            actual = ctypes.c_void_p()
            if not kernel.GetModuleHandleExW(6, ctypes.cast(function, ctypes.c_void_p), ctypes.byref(actual)):
                raise OSError('loaded symbol owner unavailable')
            return module_path(actual)
        return library, path, resolve
    # UMATH's real loaded dependency group, plus dladdr on EACH getter.
    library = ctypes.CDLL(str(umath_path))
    class Info(ctypes.Structure):
        _fields_ = [('name', ctypes.c_char_p), ('base', ctypes.c_void_p),
                    ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]
    query = ctypes.CDLL(None).dladdr
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info)]
    query.restype = ctypes.c_int
    def resolve(function):
        info = Info()
        if not query(ctypes.cast(function, ctypes.c_void_p), ctypes.byref(info)) or not info.name:
            raise OSError('dladdr unavailable')
        return Path(os.fsdecode(info.name)).resolve()
    return library, None, resolve


@__import__('functools').lru_cache(maxsize=1)
def _authenticate_runtime():
    import hashlib
    import struct
    from pathlib import Path
    try:
        umath = importlib.import_module(_PREFIX+'._multiarray_umath')
        u = getattr(umath, '__file__', None)
        n = getattr(np, '__file__', None)
        if type(u) is not str or type(n) is not str:
            return None
        upath = Path(u).resolve()
        raw = upath.read_bytes()
        key = (np.__version__, hashlib.sha256(raw).hexdigest())
        pin = _RUNTIME_PINS.get(key)
        if pin is None:
            return None
        loaded = _loaded_blas(upath, raw)
        if loaded is None:
            return None
        library, path, resolve = loaded
        getters = []
        for name in ('corename', 'num_threads', 'config'):
            function = getattr(library, pin[1].replace('corename', name))
            origin = resolve(function)
            if path is None:
                path = origin
            if origin != path:
                return None
            getters.append(function)
        if path.parent != (Path(n).resolve().parent.parent/'numpy.libs').resolve():
            return None
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin[0]:
            return None
        core, threads, config = getters
        core.argtypes = []; core.restype = ctypes.c_char_p
        threads.argtypes = []; threads.restype = ctypes.c_int
        config.argtypes = []; config.restype = ctypes.c_char_p
        description = config()
        if not description or b'USE64BITINT' not in description:
            return None
        # Retain handles/functions and exact file identities. Metadata queries
        # are read-only; no per-call numeric calibration or vendor algorithms.
        identities = tuple((str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in (upath, path))
        return key[0], pin[0], core, threads, library, identities
    except (OSError, AttributeError, ValueError, UnicodeError, struct.error):
        return None


def _runtime_profile():
    from pathlib import Path
    global _THREAD_QUERY, _RUNTIME_CORENAME
    _THREAD_QUERY = None
    _RUNTIME_CORENAME = None
    if not _unforced():
        return None
    certificate = _authenticate_runtime()
    if certificate is None:
        return None
    version, digest, query, threads, library, identities = certificate
    try:
        if any((Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != (size, stamp)
               for p, size, stamp in identities):
            return None
        actual = query()
        if actual is None:
            return None
        corename = actual.decode('ascii')
        if corename not in _RUNTIME_CORES[digest] or threads() != 1:
            return None
    except (OSError, ValueError, UnicodeError):
        return None
    _THREAD_QUERY = threads
    _RUNTIME_CORENAME = corename
    return version


_runtime_profile.cache_clear = _authenticate_runtime.cache_clear


def _checked_entry(profile=1):
    if profile == 6:
        function = getattr(_core, 'matrix_transform_run_v2', None)
        if (type(function) is BuiltinFunctionType and function.__name__ == 'matrix_transform_run_v2'
                and function.__module__ == _core.__name__
                and getattr(_core, 'matrix_transform_v2_interface', None) == 'matrix-transform-math-v2'
                and getattr(_core, 'matrix_transform_v2_capabilities', 0) == 63):
            return function
        return None
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
        # The gufunc makes an ALIGNED whole temporary: its omitted core
        # dimensions receive C strides. This occurs before the matmul loop,
        # unlike the matrix-matrix explicit orientation-preserving copy.
        return 3 if f and matrix.flags.aligned else 4
    return 2 if version == '1.24.4' and not (c or f) else 1


def _prepared_profile(h, matrix, version, corename):
    # Arithmetic SHAPE selection only; this never authenticates a runtime.
    profile = _arithmetic_profile(h, matrix, version)
    if profile == 4 and corename in ('Cooperlake', 'SkylakeX'):
        return 6
    return profile


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
        profile=_prepared_profile(homogeneous,matrix,version,_RUNTIME_CORENAME)
    entry = _checked_entry(profile)
    if entry is None:
        return None
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
