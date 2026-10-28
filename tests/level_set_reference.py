"""Select exact witnessed LevelSet records without changing their contents."""
import ctypes
import hashlib
import json
from pathlib import Path
import platform

WINDOWS_IDS=('checkerboard_level_set/square-wide','disk_level_set/radius-longdouble')
SUPPLEMENT=Path(__file__).parents[1]/'coverage/contracts/level-set-windows.json'


def _windows_abi(np):
    import scipy
    import skimage
    from skimage.segmentation import morphsnakes
    dtype=np.dtype(np.longdouble)
    return dict(system=platform.system(),machine=platform.machine(),python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,
        c_long_bytes=ctypes.sizeof(ctypes.c_long),pointer_bytes=ctypes.sizeof(ctypes.c_void_p),
        longdouble_itemsize=dtype.itemsize,longdouble_nmant=int(np.finfo(np.longdouble).nmant),
        longdouble_is_float64=np.longdouble is np.float64,longdouble_dtype=str(dtype),
        longdouble_dtype_type_module=dtype.type.__module__,longdouble_dtype_type_name=dtype.type.__name__,
        original_source_sha256=hashlib.sha256(Path(morphsnakes.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest())


def reference_observation(record,case_id,np):
    system=platform.system()
    if case_id not in WINDOWS_IDS or system=='Linux':
        return next(row for row in record['original_observations'][np.__version__] if row['id']==case_id)
    if system!='Windows':raise AssertionError('Unassessed LevelSet reference platform: '+system)
    supplement=json.loads(SUPPLEMENT.read_bytes())
    if supplement['schema']!='level-set-windows-original-v1' or tuple(supplement['ids'])!=WINDOWS_IDS:
        raise AssertionError('Unexpected LevelSet Windows supplement scope')
    actual=_windows_abi(np)
    if actual!=supplement['abi']:
        raise AssertionError('Unassessed LevelSet Windows ABI: '+repr(actual))
    if tuple(row['id'] for row in supplement['observations'])!=WINDOWS_IDS:
        raise AssertionError('Incomplete LevelSet Windows original records')
    return next(row for row in supplement['observations'] if row['id']==case_id)
