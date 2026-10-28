"""Original-first exact RDP contract. Prospective tests never import the target."""
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
    from .approximate_polygon_contract import API,MAX_POINTS,UNASSESSED,CoordinatesProvider,ToleranceProvider,cases,fixture
else:
    from approximate_polygon_contract import API,MAX_POINTS,UNASSESSED,CoordinatesProvider,ToleranceProvider,cases,fixture

CASES=cases();MATRIX=Path(__file__).parents[1]/'coverage/contracts/approximate-polygon.json'
SOURCE_SHA='26a27d89545d158f6b542857297995639b251ed3af0428c8c86a56c84928aabe'
FUNCTION_SHA='086f1d9030bfa69e166649b6c67ba211b597b62e9c0b87294f59a678d71db12e'
SIGNATURE='(coords, tolerance)'


def scientific_profile():
    import numpy as np,scipy
    assert(np.__version__,scipy.__version__)in(('1.24.4','1.15.3'),('2.5.4','1.18.1'))
    assert importlib.metadata.version('scikit-image')=='0.26.0'
    return np


def reference_runtime():
    """Only a separately authorized once-only capture runner calls this."""
    np=scientific_profile();module=importlib.import_module('skimage.measure._polygon');source=Path(module.__file__);raw=source.read_bytes().replace(b'\r\n',b'\n');text=raw.decode()
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='approximate_polygon')
    assert hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()==FUNCTION_SHA
    call=module.approximate_polygon;assert str(inspect.signature(call))==SIGNATURE
    return np,call,dict(python=sys.version,numpy=np.__version__,scipy=importlib.metadata.version('scipy'),skimage='0.26.0',source_path=str(source),source_canonical_lf_sha256=SOURCE_SHA,function_segment_sha256=FUNCTION_SHA,signature=SIGNATURE,calls_per_spec=1,native_calls=0,timings=False)


def encode(value,np):
    if isinstance(value,CoordinatesProvider):return dict(protocol='CoordinatesProvider',array=encode(value.array,np))
    if isinstance(value,ToleranceProvider):return dict(protocol='ToleranceProvider',value=encode(value.value,np),mutating=value.array is not None)
    if isinstance(value,np.ndarray):
        assert not value.dtype.fields and not(value.dtype.kind=='f'and value.dtype.itemsize>8)
        result=dict(type=type(value).__name__,dtype=value.dtype.str,dtype_descr=encode(value.dtype.descr,np),dtype_metadata=None if value.dtype.metadata is None else dict(value.dtype.metadata),shape=list(value.shape),strides=list(value.strides),own=bool(value.flags.owndata),writeable=bool(value.flags.writeable),aligned=bool(value.flags.aligned),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),base_type=type(value.base).__name__)
        if value.dtype.hasobject:result['object_values']=encode(value.tolist(),np)
        else:
            raw=value.tobytes();result.update(raw_sha256=hashlib.sha256(raw).hexdigest(),raw_zlib_base64=base64.b64encode(zlib.compress(raw)).decode())
        return result
    if isinstance(value,np.generic):
        assert not value.dtype.hasobject and not value.dtype.fields and not(value.dtype.kind=='f'and value.dtype.itemsize>8)
        return dict(type=type(value).__name__,dtype=value.dtype.str,raw=value.tobytes().hex())
    if isinstance(value,float)and not math.isfinite(value):return dict(python_float='nan'if math.isnan(value)else'positive-infinity'if value>0 else'negative-infinity')
    if isinstance(value,complex):return dict(python_complex=[encode(value.real,np),encode(value.imag,np)])
    if isinstance(value,dict):return {key:encode(item,np)for key,item in value.items()}
    if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[encode(item,np)for item in value])
    if value is None or isinstance(value,(bool,int,float,str)):return value
    raise TypeError('Unsupported RDP observation value: '+type(value).__name__)


def observe(call,spec,np):
    args,options,events,buffers=fixture(spec,np);before=encode((args,options,buffers),np);assert not events
    pointers={name:value.ctypes.data for name,value in buffers.items()}
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
    if error is None and isinstance(result,np.ndarray):assert result.ndim<=2 and result.size<=MAX_POINTS*3
    row=dict(id=spec['id'],api=API,input_before=before,input_after=encode((args,options,buffers),np),result=encode(result,np),error=error,warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)]for w in notices],events=encode(events,np),result_is_input=result is args[0],result_shares_input=isinstance(result,np.ndarray)and any(np.shares_memory(result,value)for value in buffers.values()),checks=dict(policy_unchanged=stable,state_restored=np.geterr()==old_policy and np.geterrcall()is old_handler and np.getbufsize()==old_buffer,pointers_stable=all(buffers[name].ctypes.data==pointer for name,pointer in pointers.items())))
    assert all(row['checks'].values())
    json.dumps(row,allow_nan=False)
    return row


def test_approximate_polygon_prospective_manifest():
    matrix=json.loads(MATRIX.read_bytes())
    assert matrix['api']==API and matrix['cases']==CASES and len(CASES)==60 and len({s['id']for s in CASES})==60
    assert matrix['source_canonical_lf_sha256']==SOURCE_SHA and matrix['function_segment_sha256']==FUNCTION_SHA and matrix['signature']==SIGNATURE
    assert matrix['native_implementation']=='pending'and matrix['unassessed']==list(UNASSESSED)
    for name,pin in matrix['source_freeze'].items():assert hashlib.sha256((MATRIX.parents[2]/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


@pytest.mark.parametrize('spec',CASES,ids=[spec['id']for spec in CASES])
def test_approximate_polygon_fixture_without_original(spec):
    np=scientific_profile();args,options,events,buffers=fixture(spec,np)
    assert not events and all(value.ndim==0 or value.shape[0]<=MAX_POINTS for value in buffers.values())
    json.dumps(encode((args,options,buffers),np),allow_nan=False)
    assert 'skimage.measure._polygon'not in sys.modules


@pytest.mark.parametrize('kind',('zero-tolerance','tolerance-mutates-coordinates','object-numbers'))
def test_approximate_polygon_observer_standin(kind):
    np=scientific_profile();spec=next(s for s in CASES if s['id']==kind)
    def standin(coords,tolerance):
        if kind=='zero-tolerance':return coords
        if kind=='tolerance-mutates-coordinates':tolerance<=0
        return coords[:2].copy()
    row=observe(standin,spec,np)
    assert row['error']is None and all(row['checks'].values())
    assert row['result_is_input']is(kind=='zero-tolerance')
    assert(row['input_before']!=row['input_after'])is(kind=='tolerance-mutates-coordinates')


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_approximate_polygon_saved_original_rows(profile):
    rows=json.loads(MATRIX.read_bytes())['original_observations']
    if not rows:pytest.skip('Prospective freeze; original capture is not authorized')
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
def test_approximate_polygon_native_public_contract(spec,workers):
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    if API not in native_functions():pytest.skip('Native RDP implementation remains pending')
    import numpy as np
    expected=next(r for r in json.loads(MATRIX.read_bytes())['original_observations'][np.__version__]if r['id']==spec['id'])
    old=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    try:
        assert str(inspect.signature(xx.measure.approximate_polygon))==SIGNATURE
        assert observe(xx.measure.approximate_polygon,spec,np)==expected
    finally:xx.set_num_threads(old);xx.set_fallback_enabled(fallback)
