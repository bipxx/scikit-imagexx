"""Prospective exact fill observer; caller authorizes originals separately."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings
if __package__:
    from .draw_polygon_contract import APIS,fixture
    from .draw_coordinates_observer import encode,scientific_profile
else:
    from draw_polygon_contract import APIS,fixture
    from draw_coordinates_observer import encode,scientific_profile
DRAW_SOURCE_SHA='fc2e5f3d5174770a667405f0c3baf6fdf5ffdf47efec85686d3af8bd1ae68f49'
MASK_SOURCE_SHA='63575285de3c5916f1a51aaf3c4bd444f3f1c0529a47483e458d75fdea4aa325'
DRAW_BINARY={'1.24.4':'32673fec979236e2f83543ad289a8128059b96e6562a9ea30114c4e73c17139b','2.5.4':'ae81e3d9b252c264cac3236a22d96e7b5242d0f9819950dbe1f73464380869d4'}
GEOMETRY_BINARY={'1.24.4':'1e3c9d10521cfc941841bb03a77e393b5df06916b7005207a7a28a506bff7278','2.5.4':'388e1559a7bea0d05dd9db1ab8e9f6ab3d6a253b989b9ae39eb60c917ea26458'}
SIGNATURES={'draw.polygon':'(r, c, shape=None)','draw.polygon2mask':'(image_shape, polygon)'}


def reference_runtime():
    np=scientific_profile();draw=importlib.import_module('skimage.draw.draw');mask=importlib.import_module('skimage.draw._polygon2mask')
    for module,pin in [(draw,DRAW_SOURCE_SHA),(mask,MASK_SOURCE_SHA)]:
        assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    binaries={}
    for name,pins in [('skimage.draw._draw',DRAW_BINARY),('skimage._shared.geometry',GEOMETRY_BINARY)]:
        path=Path(importlib.import_module(name).__file__);pin=hashlib.sha256(path.read_bytes()).hexdigest();assert pin==pins[np.__version__]
        binaries[name]=dict(path=str(path),sha256=pin)
    functions={APIS[0]:draw.polygon,APIS[1]:mask.polygon2mask}
    assert {api:str(inspect.signature(call))for api,call in functions.items()}==SIGNATURES
    return np,functions,dict(python=sys.version,numpy=np.__version__,reference='0.26.0',source_sha256=dict(draw=DRAW_SOURCE_SHA,polygon2mask=MASK_SOURCE_SHA),binaries=binaries,calls_per_spec=1,native_calls=0,timings=False)


def observe(call,spec,np):
    # The factory validates this external recipe once; bounds/typed values below
    # retain that established input invariant. No retries or original recapture.
    args,options,buffers,bounds=fixture(spec,np)
    before=encode((args,options),np);pointers={k:a.ctypes.data for k,a in buffers.items()}
    descriptor={k:(a.shape,a.strides,a.dtype,a.dtype.metadata,a.flags.writeable,a.flags.owndata,a.flags.aligned)for k,a in buffers.items()}
    old_policy,old_handler,old_buffer=np.geterr(),np.geterrcall(),np.getbufsize();events=[];result=None;error=None
    def handler(message,flags):events.append(dict(message=message,flags=flags))
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always')
        with np.errstate(all='warn',call=handler):
            policy=np.geterr().copy()
            try:result=call(*args,**options)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
            stable=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==old_buffer
    if error is None:
        outputs=list(result)if spec['api']==APIS[0]else[result]
        assert all(type(a)is np.ndarray for a in outputs)
        if spec['api']==APIS[0]:
            assert type(result)is tuple and len(outputs)==2 and all(a.ndim==1 and a.dtype==np.dtype(np.intp)for a in outputs)
            assert outputs[0].shape==outputs[1].shape and outputs[0].size<=bounds['max_output_pairs']
        else:assert result.dtype==np.dtype(bool)and result.shape==tuple(spec['shape'])and result.size<=bounds['max_mask_elements']
    else:outputs=[]
    after=encode((args,options),np)
    checks=dict(input_bytes_metadata_unchanged=before==after,original_pointers_unchanged=all(a.ctypes.data==pointers[k]for k,a in buffers.items()),
        original_descriptors_unchanged=all((a.shape,a.strides,a.dtype,a.dtype.metadata,a.flags.writeable,a.flags.owndata,a.flags.aligned)==descriptor[k]for k,a in buffers.items()),
        fp_policy_unchanged=stable,state_restored=np.geterr()==old_policy and np.geterrcall()is old_handler and np.getbufsize()==old_buffer)
    assert all(checks.values())
    row=dict(id=spec['id'],api=spec['api'],inputs_before=before,inputs_after=after,result=encode(result,np),error=error,
        warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)]for w in notices],FP_events=events,
        output_pair_aliases=[[a is b,bool(np.shares_memory(a,b)),a.base is b.base]for a in outputs for b in outputs],
        output_input_aliases=[[a is b,bool(np.shares_memory(a,b))]for a in outputs for b in buffers.values()],checks=checks)
    json.dumps(row,allow_nan=False);return row
