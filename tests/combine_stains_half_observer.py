"""Bounded original observer; preserves raw words and full array metadata."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings
if __package__:
    from .combine_stains_half_contract import API, MAX_OUTPUT_ELEMENTS, fixture
    from .draw_coordinates_observer import encode, scientific_profile
else:
    from combine_stains_half_contract import API, MAX_OUTPUT_ELEMENTS, fixture
    from draw_coordinates_observer import encode, scientific_profile
SOURCE_SHA = 'fc5f49ac194c229325f21ddbe4c49f7df4307b9627dfbb5fcd2825133144fdfe'
UTILS_SHA = 'b0a232570c8e990ab76cd6fe6f54a67161a691214dbf6c529656e13a799eb1f1'
DTYPE_SHA = 'd4f99f88966e4d0dbbe80a3c0b8acb3890644cc5fbfb707ba9b5a90e9119e627'
SIGNATURE = '(stains, conv_matrix, *, channel_axis=-1)'


def reference_runtime():
    np = scientific_profile()
    module = importlib.import_module('skimage.color.colorconv')
    utils = importlib.import_module('skimage._shared.utils')
    dtype = importlib.import_module('skimage.util.dtype')
    for item, pin in ((module, SOURCE_SHA), (utils, UTILS_SHA), (dtype, DTYPE_SHA)):
        assert hashlib.sha256(Path(item.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pin
    binary = importlib.import_module('numpy._core._multiarray_umath' if np.__version__.startswith('2.') else 'numpy.core._multiarray_umath')
    call = module.combine_stains
    assert str(inspect.signature(call)) == SIGNATURE
    return np, call, dict(python=sys.version, numpy=np.__version__, skimage='0.26.0',
        original_color_source_sha256=SOURCE_SHA, utils_source_sha256=UTILS_SHA, dtype_source_sha256=DTYPE_SHA,
        numpy_binary_path=str(binary.__file__), numpy_binary_sha256=hashlib.sha256(Path(binary.__file__).read_bytes()).hexdigest(),
        calls_per_spec=1, native_calls=0, timings=False)


def observe(call, spec, np):
    args, options, buffers = fixture(spec, np)
    before = encode((args, options, buffers), np)
    pointers = {name: array.ctypes.data for name, array in buffers.items()}
    old_policy, old_handler, old_buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    events = []
    result = None
    error = None
    def handler(message, flags):
        events.append(dict(event='FP-handler', message=message, flags=flags))
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        with np.errstate(**spec['policy'], call=handler):
            policy = np.geterr().copy()
            try:
                result = call(*args, **options)
            except Exception as exc:
                error = [type(exc).__module__, type(exc).__name__, str(exc)]
            stable = np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == old_buffer
    if error is None:
        assert type(result) is np.ndarray and result.size <= MAX_OUTPUT_ELEMENTS
        assert result.dtype.type in (np.float32, np.float64)
        assert np.isfinite(result).all() and (result >= 0).all() and (result <= 1).all()
    row = dict(id=spec['id'], api=API, input_before=before,
        input_after=encode((args, options, buffers), np), result=encode(result, np), error=error,
        warnings=[[type(w.message).__module__, type(w.message).__name__, str(w.message)] for w in notices],
        events=events, output_input_aliases=[] if error is not None else [dict(input=name,
            same_object=result is array, shares_memory=bool(np.shares_memory(result, array)),
            same_base=result.base is array.base) for name, array in buffers.items()],
        checks=dict(policy_unchanged=stable,
            state_restored=np.geterr() == old_policy and np.geterrcall() is old_handler and np.getbufsize() == old_buffer,
            pointers_stable=all(buffers[name].ctypes.data == pointer for name, pointer in pointers.items())))
    assert all(row['checks'].values()) and before == row['input_after']
    json.dumps(row, allow_nan=False)
    return row
