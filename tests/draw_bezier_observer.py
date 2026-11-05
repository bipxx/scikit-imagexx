"""One-call Bezier observation using the existing frozen raw draw encoder."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings
if __package__:
    from .draw_bezier_contract import API, MAX_OUTPUT_ELEMENTS, fixture
    from .draw_coordinates_observer import encode, scientific_profile, SOURCE_SHA, BINARY_SHA
else:
    from draw_bezier_contract import API, MAX_OUTPUT_ELEMENTS, fixture
    from draw_coordinates_observer import encode, scientific_profile, SOURCE_SHA, BINARY_SHA

SIGNATURE = '(r0, c0, r1, c1, r2, c2, weight, shape=None)'
SOURCE_PYX_SHA = 'fe7f5f475d2643633f494befe433b5cd1c4bae6d6c141ed4d1bb5415e80207e4'


def reference_runtime():
    np = scientific_profile()
    module = importlib.import_module('skimage.draw.draw')
    binary = importlib.import_module('skimage.draw._draw')
    assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == SOURCE_SHA
    assert hashlib.sha256(Path(binary.__file__).read_bytes()).hexdigest() == BINARY_SHA[np.__version__]
    call = module.bezier_curve
    assert str(inspect.signature(call)) == SIGNATURE
    return np, call, dict(python=sys.version, numpy=np.__version__, source_path=str(module.__file__),
        source_canonical_lf_sha256=SOURCE_SHA, binary_path=str(binary.__file__),
        binary_sha256=BINARY_SHA[np.__version__], calls_per_spec=1, native_calls=0, timings=False)


def observe(call, spec, np):
    args, options, events, buffers = fixture(spec, np)
    before = encode((args,options,buffers),np); assert not events
    pointers = {name:value.ctypes.data for name,value in buffers.items()}
    old_policy, old_handler, old_buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    result = None; error = None
    def handler(message, flags):
        events.append(dict(event='FP-handler',message=message,flags=flags))
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        with np.errstate(**spec['policy'],call=handler):
            policy = np.geterr().copy()
            try:
                result = call(*args,**options)
            except Exception as exc:
                error = [type(exc).__module__,type(exc).__name__,str(exc)]
            stable = np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == old_buffer
    outputs = [] if error is not None else list(result)
    if error is None:
        assert type(result) is tuple and len(outputs) == 2
        assert all(type(value) is np.ndarray and value.dtype == np.dtype(np.intp) for value in outputs)
        assert sum(value.size for value in outputs) <= MAX_OUTPUT_ELEMENTS
    row = dict(id=spec['id'],api=API,input_before=before,input_after=encode((args,options,buffers),np),
        result=encode(result,np),error=error,
        warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)] for w in notices],
        events=encode(events,np),
        output_input_aliases=[[value is buf,bool(np.shares_memory(value,buf))] for value in outputs for buf in buffers.values()],
        output_pair_aliases=[[a is b,bool(np.shares_memory(a,b)),a.base is b.base] for a in outputs for b in outputs],
        checks=dict(policy_unchanged=stable,state_restored=np.geterr()==old_policy and np.geterrcall() is old_handler and np.getbufsize()==old_buffer,
                    pointers_stable=all(buffers[name].ctypes.data==pointer for name,pointer in pointers.items())))
    assert all(row['checks'].values()); json.dumps(row,allow_nan=False)
    return row
