"""Original-first geometry contracts; target calls require separate authorization."""
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
    from .rolling_ball_kernels_contract import APIS,MAX_ELEMENTS,STAGES,UNASSESSED,IndexValue,RadiusValue,ArrayInput,IntensityValue,cases,fixture,output_bound
else:
    from rolling_ball_kernels_contract import APIS,MAX_ELEMENTS,STAGES,UNASSESSED,IndexValue,RadiusValue,ArrayInput,IntensityValue,cases,fixture,output_bound

CASES=cases();MATRIX=Path(__file__).parents[1]/'coverage/contracts/rolling-ball-kernels.json'
SOURCE_SHA='5572c656ad2a6e48518ee7188c4cea96e7817405a941fbf76932268ddd8540f6'
SIGNATURES={'restoration.ball_kernel':'(radius, ndim)','restoration.ellipsoid_kernel':'(shape, intensity)'}
FUNCTION_SHAS={'ball_kernel':'94238cd483eba493167343cff76c96309beeda669e53385f3850cc1001ef226e','ellipsoid_kernel':'81269f8617faf10f8a0db54bd1c2dd507fbfbd1f8d5496e0d0af49df344f0f02'}


def scientific_profile():
    import numpy as np,scipy
    assert (np.__version__,scipy.__version__)in(('1.24.4','1.15.3'),('2.5.4','1.18.1'))
    assert importlib.metadata.version('scikit-image')=='0.26.0'
    return np


def reference_runtime():
    """Capture runner calls this only after root authorizes once-only originals."""
    np=scientific_profile();module=importlib.import_module('skimage.restoration._rolling_ball')
    source=Path(module.__file__);raw=source.read_bytes().replace(b'\r\n',b'\n');text=raw.decode('utf-8')
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    funcs={api:getattr(module,api.split('.')[-1])for api in APIS}
    for api,call in funcs.items():assert str(inspect.signature(call))==SIGNATURES[api]
    for node in ast.parse(text).body:
        if isinstance(node,ast.FunctionDef)and node.name in FUNCTION_SHAS:
            assert hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()==FUNCTION_SHAS[node.name]
    return np,funcs,dict(python=sys.version,numpy=np.__version__,scipy=importlib.metadata.version('scipy'),skimage='0.26.0',source_path=str(source),source_canonical_lf_sha256=SOURCE_SHA,function_segment_sha256=FUNCTION_SHAS,signatures=SIGNATURES,calls_per_spec=1,native_calls=0,timings=False)


def encode(value,np):
    for cls in(IndexValue,RadiusValue,IntensityValue):
        if isinstance(value,cls):return dict(protocol=cls.__name__,value=encode(value.value,np))
    if isinstance(value,ArrayInput):return dict(protocol='ArrayInput',value=encode(value.value,np))
    if isinstance(value,np.dtype):return dict(dtype_object=value.str,descr=encode(value.descr,np),metadata=None if value.metadata is None else dict(value.metadata))
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject or value.dtype.fields or(value.dtype.kind in 'fc'and value.dtype.itemsize not in(2,4,8,16)):
            raise TypeError('Unsupported object/padded geometry representation')
        if value.dtype.kind=='f'and value.dtype.itemsize>8:raise TypeError('Longdouble bytes are outside this contract')
        raw=value.tobytes()
        return dict(type=type(value).__name__,dtype=value.dtype.str,dtype_descr=encode(value.dtype.descr,np),dtype_metadata=None if value.dtype.metadata is None else dict(value.dtype.metadata),shape=list(value.shape),strides=list(value.strides),own=bool(value.flags.owndata),writeable=bool(value.flags.writeable),aligned=bool(value.flags.aligned),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),base_type=type(value.base).__name__,raw_sha256=hashlib.sha256(raw).hexdigest(),raw_zlib_base64=base64.b64encode(zlib.compress(raw)).decode())
    if isinstance(value,np.generic):
        if value.dtype.hasobject or value.dtype.fields or(value.dtype.kind=='f'and value.dtype.itemsize>8):raise TypeError('Unsupported scalar representation')
        return dict(type=type(value).__name__,dtype=value.dtype.str,raw=value.tobytes().hex())
    if isinstance(value,float)and not math.isfinite(value):return dict(python_float='nan'if math.isnan(value)else'positive-infinity'if value>0 else'negative-infinity')
    if isinstance(value,complex):return dict(python_complex=[encode(value.real,np),encode(value.imag,np)])
    if isinstance(value,dict):return {key:encode(item,np)for key,item in value.items()}
    if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[encode(item,np)for item in value])
    if value is None or isinstance(value,(bool,int,float,str)):return value
    raise TypeError('Unsupported geometry contract value: '+type(value).__name__)


def observe(call,spec,np):
    args,options,events,buffers=fixture(spec,np);before=encode((args,options,buffers),np)
    json.dumps(before,allow_nan=False)
    pointers={key:value.ctypes.data for key,value in buffers.items()}
    old_policy,old_handler,old_buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    def handler(message,flags):events.append(dict(event='FP-handler',message=message,flags=flags))
    result=None;error=None
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always')
        with np.errstate(**spec['policy'],call=handler):
            policy=np.geterr().copy()
            try:result=call(*args,**options)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
            stable=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==old_buffer
    if error is None:
        assert type(result)is np.ndarray and result.size<=spec['max_successful_elements']
        assert not result.dtype.hasobject and not result.dtype.fields and result.dtype.kind in 'fciub'
        assert not(result.dtype.kind=='f'and result.dtype.itemsize>8)
    row=dict(id=spec['id'],api=spec['api'],input_before=before,input_after=encode((args,options,buffers),np),result=encode(result,np),error=error,warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)]for w in notices],events=events,result_is_input=any(result is arg for arg in args),result_shares_input=isinstance(result,np.ndarray)and any(np.shares_memory(result,value)for value in buffers.values()),checks=dict(policy_unchanged=stable,state_restored=np.geterr()==old_policy and np.geterrcall()is old_handler and np.getbufsize()==old_buffer,pointers_stable=all(buffers[key].ctypes.data==ptr for key,ptr in pointers.items())))
    assert all(row['checks'].values())
    if error is None:assert not row['result_is_input']and not row['result_shares_input']
    json.dumps(row,allow_nan=False)
    return row


def test_geometry_prospective_manifest():
    matrix=json.loads(MATRIX.read_bytes())
    assert matrix['apis']==list(APIS)and matrix['cases']==CASES and len(CASES)==50
    assert matrix['source_canonical_lf_sha256']==SOURCE_SHA and matrix['function_segment_sha256']==FUNCTION_SHAS
    assert matrix['signatures']==SIGNATURES and matrix['native_implementation']=='pending'
    assert matrix['unassessed']==list(UNASSESSED)and matrix['max_successful_elements']==8192
    assert all(callable(globals()[stage])for stage in STAGES)
    for name,pin in matrix['source_freeze'].items():assert hashlib.sha256((MATRIX.parents[2]/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


def test_prospective_allocation_bounds_and_rank_frontier_are_explicit():
    assert all(0<=output_bound(spec)<=MAX_ELEMENTS for spec in CASES)
    by_id={spec['id']:spec for spec in CASES}
    assert output_bound(by_id['ball-8d'])==output_bound(by_id['ellipsoid-8d'])==6561
    assert all(output_bound(by_id['ball-rank'+str(rank)+'-zero'])==1 for rank in(31,32,63,64))
    assert {spec['params']['ndim']for spec in CASES if spec['id']in('ball-5d','ball-8d')}=={5,8}


@pytest.mark.parametrize('spec',CASES,ids=[spec['id']for spec in CASES])
def test_geometry_fixture_serializes_without_original_calls(spec):
    np=scientific_profile();args,options,events,buffers=fixture(spec,np)
    json.dumps(encode((args,options,buffers),np),allow_nan=False)
    assert not events and all(value.size<=8192 for value in buffers.values())
    assert 'skimage.restoration._rolling_ball'not in sys.modules


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_geometry_saved_original_rows(profile):
    matrix=json.loads(MATRIX.read_bytes());observed=matrix['original_observations']
    if not observed:pytest.skip('Prospective freeze; original capture is not authorized')
    rows=observed[profile];assert len(rows)==50 and [r['id']for r in rows]==[s['id']for s in CASES]
    for row in rows:
        assert all(row['checks'].values())
        if row['result']is not None:
            raw=zlib.decompress(base64.b64decode(row['result']['raw_zlib_base64']))
            assert hashlib.sha256(raw).hexdigest()==row['result']['raw_sha256']


@pytest.mark.parametrize('mutate',(False,True))
def test_geometry_observer_standin_retains_nonfinite_words_and_input_evidence(mutate):
    np=scientific_profile();spec=next(s for s in CASES if s['id']=='ellipsoid-int16-shape')
    def standin(shape,intensity):
        if mutate:shape.fill(1)
        return np.array([float('inf'),float('nan'),-0.],np.float64)
    row=observe(standin,spec,np)
    assert row['error']is None and(row['input_before']!=row['input_after'])is mutate
    assert all(row['checks'].values());json.dumps(row,allow_nan=False)


@pytest.mark.parametrize('spec',CASES,ids=[spec['id']for spec in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_geometry_native_public_contract(spec,workers):
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    if spec['api']not in native_functions():pytest.skip('Native geometry remains pending')
    import numpy as np
    matrix=json.loads(MATRIX.read_bytes());expected=next(row for row in matrix['original_observations'][np.__version__]if row['id']==spec['id'])
    old=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    try:
        call=getattr(xx.restoration,spec['api'].split('.')[-1]);assert str(inspect.signature(call))==SIGNATURES[spec['api']]
        assert observe(call,spec,np)==expected
    finally:xx.set_num_threads(old);xx.set_fallback_enabled(fallback)
