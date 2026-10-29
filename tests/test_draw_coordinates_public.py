"""Original-first coordinate contracts; capture is separately authorized."""
import ast
import base64
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import sys
import warnings
import zlib
import pytest
if __package__:
    from .draw_coordinates_contract import APIS,MAX_OUTPUT_ELEMENTS,UNASSESSED,ScalarProvider,ArrayProvider,cases,fixture,benchmark_cases
else:
    from draw_coordinates_contract import APIS,MAX_OUTPUT_ELEMENTS,UNASSESSED,ScalarProvider,ArrayProvider,cases,fixture,benchmark_cases

CASES=cases();MATRIX=Path(__file__).parents[1]/'coverage/contracts/draw-coordinates.json'
SOURCE_SHA='fc2e5f3d5174770a667405f0c3baf6fdf5ffdf47efec85686d3af8bd1ae68f49'
SIGNATURES={'draw.line':'(r0, c0, r1, c1)','draw.rectangle':'(start, end=None, extent=None, shape=None)',
            'draw.disk':'(center, radius, *, shape=None)','draw.ellipse':'(r, c, r_radius, c_radius, shape=None, rotation=0.0)'}
BINARY_SHA={'1.24.4':'32673fec979236e2f83543ad289a8128059b96e6562a9ea30114c4e73c17139b',
            '2.5.4':'ae81e3d9b252c264cac3236a22d96e7b5242d0f9819950dbe1f73464380869d4'}


def scientific_profile():
    import numpy as np,scipy
    assert(np.__version__,scipy.__version__)in(('1.24.4','1.15.3'),('2.5.4','1.18.1'))
    assert importlib.metadata.version('scikit-image')=='0.26.0'
    return np


def reference_runtime():
    """Import/pin only; the authorized external journal controller calls once/spec."""
    np=scientific_profile();module=importlib.import_module('skimage.draw.draw');binary=importlib.import_module('skimage.draw._draw')
    assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA
    assert hashlib.sha256(Path(binary.__file__).read_bytes()).hexdigest()==BINARY_SHA[np.__version__]
    functions={api:getattr(module,api.split('.')[1])for api in APIS}
    assert {api:str(inspect.signature(call))for api,call in functions.items()}==SIGNATURES
    return np,functions,dict(python=sys.version,numpy=np.__version__,scipy=importlib.metadata.version('scipy'),skimage='0.26.0',source_path=str(module.__file__),source_canonical_lf_sha256=SOURCE_SHA,binary_path=str(binary.__file__),binary_sha256=BINARY_SHA[np.__version__],calls_per_spec=1,native_calls=0,timings=False)


def encode(value,np):
    if isinstance(value,ScalarProvider):return dict(protocol='ScalarProvider',label=value.label,value=value.value)
    if isinstance(value,ArrayProvider):return dict(protocol='ArrayProvider',array=encode(value.array,np))
    if isinstance(value,np.ndarray):
        assert not value.dtype.hasobject and not value.dtype.fields and value.dtype.itemsize<=8
        raw=value.tobytes(order='C')
        return dict(type=type(value).__name__,dtype=value.dtype.str,dtype_descr=encode(value.dtype.descr,np),dtype_metadata=None if value.dtype.metadata is None else dict(value.dtype.metadata),shape=list(value.shape),strides=list(value.strides),own=bool(value.flags.owndata),writeable=bool(value.flags.writeable),aligned=bool(value.flags.aligned),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),base_type=type(value.base).__name__,base_shape=None if not hasattr(value.base,'shape')else list(value.base.shape),base_strides=None if not hasattr(value.base,'strides')else list(value.base.strides),raw_sha256=hashlib.sha256(raw).hexdigest(),raw_zlib_base64=base64.b64encode(zlib.compress(raw)).decode())
    if isinstance(value,np.generic):return dict(type=type(value).__name__,dtype=value.dtype.str,raw=value.tobytes().hex())
    if isinstance(value,float)and not math.isfinite(value):return dict(python_float='nan'if math.isnan(value)else'positive-infinity'if value>0 else'negative-infinity')
    if isinstance(value,dict):return {key:encode(item,np)for key,item in value.items()}
    if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[encode(item,np)for item in value])
    if value is None or isinstance(value,(bool,int,float,str)):return value
    raise TypeError('Unsupported coordinate observation: '+type(value).__name__)


def observe(call,spec,np):
    args,options,events,buffers=fixture(spec,np);before=encode((args,options,buffers),np);assert not events
    pointers={name:value.ctypes.data for name,value in buffers.items()};old_policy,old_handler,old_buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    def handler(message,flags):events.append(dict(event='FP-handler',message=message,flags=flags))
    result=None;error=None
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always')
        with np.errstate(**spec['policy'],call=handler):
            policy=np.geterr().copy()
            try:result=call(*args,**options)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
            stable=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==old_buffer
    outputs=[]
    if error is None:
        assert isinstance(result,(list,tuple))and all(isinstance(v,np.ndarray)for v in result)
        outputs=list(result);assert sum(v.size for v in outputs)<=MAX_OUTPUT_ELEMENTS
        assert all(v.dtype.kind=='i'and v.dtype.itemsize==np.dtype(np.intp).itemsize for v in outputs)
    row=dict(id=spec['id'],api=spec['api'],input_before=before,input_after=encode((args,options,buffers),np),result=encode(result,np),error=error,warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)]for w in notices],events=encode(events,np),output_input_aliases=[[v is a,bool(np.shares_memory(v,a))]for v in outputs for a in buffers.values()],output_pair_aliases=[[a is b,bool(np.shares_memory(a,b)),a.base is b.base]for a in outputs for b in outputs],checks=dict(policy_unchanged=stable,state_restored=np.geterr()==old_policy and np.geterrcall()is old_handler and np.getbufsize()==old_buffer,pointers_stable=all(buffers[name].ctypes.data==pointer for name,pointer in pointers.items())))
    assert all(row['checks'].values());json.dumps(row,allow_nan=False)
    return row


def test_draw_coordinates_prospective_manifest():
    matrix=json.loads(MATRIX.read_bytes());assert matrix['apis']==list(APIS)and matrix['cases']==CASES
    assert len(CASES)==88 and len({s['id']for s in CASES})==88
    assert matrix['signatures']==SIGNATURES and matrix['source_canonical_lf_sha256']==SOURCE_SHA and matrix['binary_sha256']==BINARY_SHA
    assert matrix['native_implementation']=='pending'and matrix['unassessed']==list(UNASSESSED)and matrix['benchmark_cases']==benchmark_cases()
    for name,pin in matrix['source_freeze'].items():assert hashlib.sha256((MATRIX.parents[2]/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
def test_draw_coordinates_fixture_without_original(spec):
    np=scientific_profile();args,options,events,buffers=fixture(spec,np)
    assert not events and all(value.size<=32 for value in buffers.values());json.dumps(encode((args,options,buffers),np),allow_nan=False)
    assert 'skimage.draw.draw'not in sys.modules and 'skimage.draw._draw'not in sys.modules


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_draw_coordinates_saved_original_rows(profile):
    rows=json.loads(MATRIX.read_bytes())['original_observations']
    if not rows:pytest.skip('Prospective only; original capture awaits review')
    observed=rows[profile];assert len(observed)==len(CASES)and[r['id']for r in observed]==[s['id']for s in CASES]
    for row in observed:
        assert all(row['checks'].values());pending=[row]
        while pending:
            value=pending.pop()
            if isinstance(value,dict):
                if 'raw_zlib_base64'in value:assert hashlib.sha256(zlib.decompress(base64.b64decode(value['raw_zlib_base64']))).hexdigest()==value['raw_sha256']
                pending.extend(value.values())
            elif isinstance(value,list):pending.extend(value)


@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_draw_coordinates_native_public_contract(spec,workers):
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    if spec['api']not in native_functions():pytest.skip('Native coordinate generation pending')
    import numpy as np
    expected=next(r for r in json.loads(MATRIX.read_bytes())['original_observations'][np.__version__]if r['id']==spec['id'])
    old=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    try:
        call=getattr(xx.draw,spec['api'].split('.')[1]);assert str(inspect.signature(call))==SIGNATURES[spec['api']]
        assert observe(call,spec,np)==expected
    finally:xx.set_num_threads(old);xx.set_fallback_enabled(fallback)
