"""Exact public line_aa observer, reusing frozen draw encoder/runtime pins."""
import hashlib,importlib,inspect,json,sys,warnings
from pathlib import Path
if __package__:
    from .line_aa_contract import API,MAX_OUTPUT_ELEMENTS,fixture
    from .draw_coordinates_observer import encode,scientific_profile,SOURCE_SHA,BINARY_SHA
else:
    from line_aa_contract import API,MAX_OUTPUT_ELEMENTS,fixture
    from draw_coordinates_observer import encode,scientific_profile,SOURCE_SHA,BINARY_SHA
SOURCE_PYX_SHA='fe7f5f475d2643633f494befe433b5cd1c4bae6d6c141ed4d1bb5415e80207e4'
SIGNATURE='(r0, c0, r1, c1)'


def reference_runtime():
    np=scientific_profile();module=importlib.import_module('skimage.draw.draw');binary=importlib.import_module('skimage.draw._draw')
    assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA
    assert hashlib.sha256(Path(binary.__file__).read_bytes()).hexdigest()==BINARY_SHA[np.__version__]
    call=module.line_aa;assert str(inspect.signature(call))==SIGNATURE
    return np,call,dict(python=sys.version,numpy=np.__version__,source_path=str(module.__file__),source_canonical_lf_sha256=SOURCE_SHA,binary_path=str(binary.__file__),binary_sha256=BINARY_SHA[np.__version__],calls_per_spec=1,native_calls=0,timings=False)


def observe(call,spec,np):
    args,options,events,buffers=fixture(spec,np)
    before=encode((args,options,buffers),np);assert not events
    pointers={n:a.ctypes.data for n,a in buffers.items()}
    old_policy,old_handler,old_buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    result=None;error=None
    def handler(message,flags):events.append(dict(event='FP-handler',message=message,flags=flags))
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always')
        with np.errstate(**spec['policy'],call=handler):
            policy=np.geterr().copy()
            try:result=call(*args,**options)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
            stable=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==old_buffer
    outputs=[]
    if error is None:
        assert type(result)is tuple and len(result)==3
        assert all(type(a)is np.ndarray for a in result)
        outputs=list(result);assert sum(a.size for a in outputs)<=MAX_OUTPUT_ELEMENTS
        assert outputs[0].dtype==outputs[1].dtype==np.dtype(np.intp)and outputs[2].dtype==np.dtype('float64')
        assert len({a.shape for a in outputs})==1 and outputs[0].ndim==1
        assert np.isfinite(outputs[2]).all()and ((outputs[2]>=0)&(outputs[2]<=1)).all()
    row=dict(id=spec['id'],api=API,input_before=before,input_after=encode((args,options,buffers),np),result=encode(result,np),error=error,warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)]for w in notices],events=encode(events,np),output_input_aliases=[[a is b,bool(np.shares_memory(a,b))]for a in outputs for b in buffers.values()],output_pair_aliases=[[a is b,bool(np.shares_memory(a,b)),a.base is b.base]for a in outputs for b in outputs],checks=dict(policy_unchanged=stable,state_restored=np.geterr()==old_policy and np.geterrcall()is old_handler and np.getbufsize()==old_buffer,pointers_stable=all(buffers[n].ctypes.data==v for n,v in pointers.items())))
    assert all(row['checks'].values());json.dumps(row,allow_nan=False)
    return row
