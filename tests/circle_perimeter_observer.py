"""Pinned original-only observer; one target call per prospective case."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings
from circle_perimeter_contract import API,MAX_OUTPUT_ELEMENTS,fixture
from draw_coordinates_observer import SOURCE_SHA,BINARY_SHA,scientific_profile,encode as draw_encode

SIGNATURE="(r, c, radius, method='bresenham', shape=None)"
CYTHON_SOURCE_SHA='fe7f5f475d2643633f494befe433b5cd1c4bae6d6c141ed4d1bb5415e80207e4'


def encode(value,np):
    if isinstance(value,bytes):return dict(bytes_hex=value.hex())
    if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[encode(item,np)for item in value])
    if isinstance(value,dict):return {key:encode(item,np)for key,item in value.items()}
    return draw_encode(value,np)


def reference_runtime():
    np=scientific_profile()
    module=importlib.import_module('skimage.draw.draw')
    binary=importlib.import_module('skimage.draw._draw')
    assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA
    assert hashlib.sha256(Path(binary.__file__).read_bytes()).hexdigest()==BINARY_SHA[np.__version__]
    call=module.circle_perimeter
    assert str(inspect.signature(call))==SIGNATURE
    return np,call,dict(python=sys.version,numpy=np.__version__,skimage='0.26.0',
                       source_canonical_lf_sha256=SOURCE_SHA,binary_sha256=BINARY_SHA[np.__version__],
                       original_calls_per_spec=1,native_calls=0,timings=False)


def observe(call,spec,np):
    args,options,events,buffers=fixture(spec,np)
    before=encode((args,options,buffers),np)
    assert not events
    pointers={name:value.ctypes.data for name,value in buffers.items()}
    policy,handler,buffer_size=np.geterr(),np.geterrcall(),np.getbufsize()
    result=None;error=None
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always')
        try:result=call(*args,**options)
        except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
    outputs=[]
    if error is None:
        assert isinstance(result,tuple)and len(result)==2
        assert all(isinstance(value,np.ndarray)and value.ndim==1 and value.dtype==np.dtype(np.intp)for value in result)
        assert result[0].size==result[1].size and sum(value.size for value in result)<=MAX_OUTPUT_ELEMENTS
        outputs=list(result)
    row=dict(id=spec['id'],api=API,input_before=before,input_after=encode((args,options,buffers),np),
             result=encode(result,np),error=error,events=encode(events,np),
             warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)]for w in notices],
             output_input_aliases=[[value is item,bool(np.shares_memory(value,item))]for value in outputs for item in buffers.values()],
             output_pair_aliases=[[a is b,bool(np.shares_memory(a,b)),a.base is b.base]for a in outputs for b in outputs],
             checks=dict(policy_unchanged=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer_size,
                         pointers_stable=all(buffers[name].ctypes.data==pointer for name,pointer in pointers.items())))
    assert all(row['checks'].values())
    json.dumps(row,allow_nan=False)
    return row
