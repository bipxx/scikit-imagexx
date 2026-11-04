"""Prospective line_nd observer; importing it never invokes draw algorithms."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings
if __package__:
    from .line_nd_contract import API,MAX_OUTPUT_ELEMENTS,fixture
    from .draw_coordinates_observer import encode,scientific_profile
else:
    from line_nd_contract import API,MAX_OUTPUT_ELEMENTS,fixture
    from draw_coordinates_observer import encode,scientific_profile
SOURCE_SHA = '44a1b51878ec4f25cdf09da75891950dea2a1b7b2ba8fbaa99345e66c26cb7fe'
SIGNATURE = '(start, stop, *, endpoint=False, integer=True)'
LINSPACE_SOURCE_SHA = {'1.24.4':'35849b320cedb57fdec533fc360cca7595e349c28082af1caeed95ee2f975e71',
                       '2.5.4':'97925f5f2a271088cbff838a149a6312335d8dac80ac0314f0c85a31b442c1b5'}


def reference_runtime():
    """Future capture gate only: imports/pins, not called by prospective tests."""
    np = scientific_profile()
    module = importlib.import_module('skimage.draw.draw_nd')
    prefix = 'numpy._core' if np.__version__.startswith('2.') else 'numpy.core'
    linspace = importlib.import_module(prefix+'.function_base')
    binary = importlib.import_module(prefix+'._multiarray_umath')
    assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == SOURCE_SHA
    assert hashlib.sha256(Path(linspace.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == LINSPACE_SOURCE_SHA[np.__version__]
    assert str(inspect.signature(module.line_nd)) == SIGNATURE
    return np,module.line_nd,dict(python=sys.version,numpy=np.__version__,skimage='0.26.0',
        source_path=module.__file__,source_canonical_lf_sha256=SOURCE_SHA,
        numpy_linspace_source_sha256=LINSPACE_SOURCE_SHA[np.__version__],numpy_binary_path=binary.__file__,
        numpy_binary_sha256=hashlib.sha256(Path(binary.__file__).read_bytes()).hexdigest(),calls_per_spec=1,native_calls=0,timings=False)


def observe(call,spec,np):
    args,options,events,buffers = fixture(spec,np)
    before = encode((args,options,buffers),np)
    assert not events
    pointers = {name:a.ctypes.data for name,a in buffers.items()}
    old_policy,old_handler,old_buffer = np.geterr(),np.geterrcall(),np.getbufsize()
    def handler(message,flags):
        events.append(dict(event='FP-handler',message=message,flags=flags))
    result,error = None,None
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        with np.errstate(**spec['policy'],call=handler):
            policy = np.geterr().copy()
            try:
                result = call(*args,**options)
            except Exception as exc:
                error = [type(exc).__module__,type(exc).__name__,str(exc)]
            stable = np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == old_buffer
    arrays = []
    if error is None:
        assert type(result) is tuple and len(result) <= MAX_OUTPUT_ELEMENTS
        for index,value in enumerate(result):
            assert isinstance(value,(np.ndarray,np.generic))
            if isinstance(value,np.ndarray):
                assert value.ndim == 1 and value.dtype.kind in 'fiu'
                arrays.append((index,value))
        assert sum(a.size for _,a in arrays) <= MAX_OUTPUT_ELEMENTS
    row = dict(id=spec['id'],api=API,input_before=before,input_after=encode((args,options,buffers),np),
        result=encode(result,np),error=error,events=encode(events,np),
        warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)] for w in notices],
        output_pair_aliases=[dict(first=i,second=j,same_object=a is b,shares_memory=bool(np.shares_memory(a,b)),same_base=a.base is b.base) for i,a in arrays for j,b in arrays],
        output_input_aliases=[dict(output=i,input=name,same_object=a is b,shares_memory=bool(np.shares_memory(a,b))) for i,a in arrays for name,b in buffers.items()],
        checks=dict(input_unchanged=before == encode((args,options,buffers),np),pointers_stable=all(buffers[n].ctypes.data == p for n,p in pointers.items()),
            policy_unchanged=stable,state_restored=np.geterr() == old_policy and np.geterrcall() is old_handler and np.getbufsize() == old_buffer))
    assert all(row['checks'].values())
    json.dumps(row,allow_nan=False)
    return row
