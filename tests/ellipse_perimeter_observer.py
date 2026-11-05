"""Pinned source observer; import and intake never call original algorithms."""
import hashlib,importlib,inspect,json,sys,warnings
import importlib.metadata
from pathlib import Path
if __package__:
    from .ellipse_perimeter_contract import API,MAX_OUTPUT_ELEMENTS,fixture
    from .draw_coordinates_observer import encode,scientific_profile,SOURCE_SHA,BINARY_SHA
else:
    from ellipse_perimeter_contract import API,MAX_OUTPUT_ELEMENTS,fixture
    from draw_coordinates_observer import encode,scientific_profile,SOURCE_SHA,BINARY_SHA
SIGNATURE='(r, c, r_radius, c_radius, orientation=0, shape=None)'
CYTHON_SHA='fe7f5f475d2643633f494befe433b5cd1c4bae6d6c141ed4d1bb5415e80207e4'

def reference_runtime():
    np=scientific_profile();module=importlib.import_module('skimage.draw.draw');binary=importlib.import_module('skimage.draw._draw')
    assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA
    assert hashlib.sha256(Path(binary.__file__).read_bytes()).hexdigest()==BINARY_SHA[np.__version__]
    function=module.ellipse_perimeter;assert str(inspect.signature(function))==SIGNATURE
    return np,function,dict(python=sys.version,numpy=np.__version__,scipy=importlib.metadata.version('scipy'),
        reference='0.26.0',source_path=module.__file__,source_sha256=SOURCE_SHA,binary_path=binary.__file__,
        binary_sha256=BINARY_SHA[np.__version__],original_calls=0,native_calls=0,timings=False)

def observe(call,spec,np):
    args,options,events,buffers,bound=fixture(spec,np);before=encode((args,options,buffers),np)
    pointers={n:a.ctypes.data for n,a in buffers.items()};old_policy,old_handler,old_buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    def handler(message,flags):events.append(dict(event='FP-handler',message=message,flags=flags))
    result=None;error=None
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        with np.errstate(**spec['policy'],call=handler):
            policy=np.geterr().copy()
            try:result=call(*args,**options)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
            stable=np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==old_buffer
    outputs=[]
    if error is None:
        assert type(result) is tuple and len(result)==2 and all(type(a)is np.ndarray for a in result)
        outputs=list(result);assert all(a.dtype==np.dtype(np.intp) for a in outputs)
        assert outputs[0].shape==outputs[1].shape and sum(a.size for a in outputs)<=min(MAX_OUTPUT_ELEMENTS,bound['max_output_elements'])
        assert all(a.ndim==(0 if bound.get('early_0D_before_shape') else 1) for a in outputs)
    row=dict(id=spec['id'],api=API,input_before=before,input_after=encode((args,options,buffers),np),
        result=encode(result,np),error=error,warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)]for w in notices],events=encode(events,np),
        output_input_aliases=[[a is b,bool(np.shares_memory(a,b))]for a in outputs for b in buffers.values()],
        output_pair_aliases=[[a is b,bool(np.shares_memory(a,b)),a.base is b.base]for a in outputs for b in outputs],
        checks=dict(inputs_unchanged=before==encode((args,options,buffers),np),policy_unchanged=stable,
            state_restored=np.geterr()==old_policy and np.geterrcall()is old_handler and np.getbufsize()==old_buffer,
            pointers_stable=all(buffers[n].ctypes.data==p for n,p in pointers.items())))
    assert all(row['checks'].values());json.dumps(row,allow_nan=False);return row
