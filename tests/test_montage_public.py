"""Prospective original-first montage contract. Capture is not yet authorized."""
import base64
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import warnings
import zlib

import pytest

if __package__:
    from .montage_contract import API,STAGES,UNASSESSED,ArrayInput,GridValue,IntValue,PaddingValue,FillValue,cases,fixture
else:
    from montage_contract import API,STAGES,UNASSESSED,ArrayInput,GridValue,IntValue,PaddingValue,FillValue,cases,fixture

CASES=cases()
MATRIX=Path(__file__).parents[1]/'coverage/contracts/montage.json'
SOURCE_SHA='c386560986cda3ba262beec03d1e08bcf55ef7219223031682289e747d526b66'
DECORATOR_SHA='b0a232570c8e990ab76cd6fe6f54a67161a691214dbf6c529656e13a799eb1f1'
SIGNATURE="(arr_in, fill='mean', rescale_intensity=False, grid_shape=None, padding_width=0, *, channel_axis=None)"


def runtime():
    import numpy as np,scipy,skimage
    module=importlib.import_module('skimage.util._montage')
    utils=importlib.import_module('skimage._shared.utils')
    assert skimage.__version__=='0.26.0'
    assert (np.__version__,scipy.__version__)in(('1.24.4','1.15.3'),('2.5.4','1.18.1'))
    assert str(inspect.signature(module.montage))==SIGNATURE
    for source,pin in ((module,SOURCE_SHA),(utils,DECORATOR_SHA)):
        assert hashlib.sha256(Path(source.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    return np,module.montage,dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,source_canonical_lf_sha256=SOURCE_SHA,decorator_canonical_lf_sha256=DECORATOR_SHA,signature=SIGNATURE,calls_per_spec=1,native_calls=0,timings=False)


def encode(value,np):
    if isinstance(value,ArrayInput):return dict(protocol='ArrayInput',value=encode(value.value,np))
    if isinstance(value,GridValue):return dict(protocol='GridValue',rows=2,columns=2)
    if isinstance(value,IntValue):return dict(protocol='IntValue',label=value.label,value=value.value)
    if isinstance(value,PaddingValue):return dict(protocol='PaddingValue',value=value.value)
    if isinstance(value,FillValue):return dict(protocol='FillValue',value=encode(value.value,np))
    if isinstance(value,np.dtype):return dict(dtype_object=value.str,descr=encode(value.descr,np),metadata=None if value.metadata is None else dict(value.metadata))
    if isinstance(value,np.ndarray):
        result=dict(type=type(value).__name__,dtype=value.dtype.str,dtype_descr=encode(value.dtype.descr,np),dtype_metadata=None if value.dtype.metadata is None else dict(value.dtype.metadata),shape=list(value.shape),strides=list(value.strides),own=bool(value.flags.owndata),writeable=bool(value.flags.writeable),aligned=bool(value.flags.aligned),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),base_type=type(value.base).__name__)
        if value.dtype.hasobject:result['object_values']=encode(value.tolist(),np)
        else:
            raw=value.tobytes();result.update(raw_sha256=hashlib.sha256(raw).hexdigest(),raw_zlib_base64=base64.b64encode(zlib.compress(raw)).decode())
        return result
    if isinstance(value,np.generic):
        return dict(type=type(value).__name__,dtype=value.dtype.str,raw=value.tobytes().hex())
    if isinstance(value,float)and not math.isfinite(value):return dict(python_float='nan'if math.isnan(value)else'positive-infinity'if value>0 else'negative-infinity')
    if isinstance(value,complex):return dict(python_complex=[encode(value.real,np),encode(value.imag,np)])
    if isinstance(value,bytes):return dict(python_bytes_hex=value.hex())
    if isinstance(value,dict):return {key:encode(item,np)for key,item in value.items()}
    if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[encode(item,np)for item in value])
    if value is None or isinstance(value,(bool,int,float,str)):return value
    raise TypeError('Unsupported montage contract value: '+type(value).__name__)


def observe(call,spec,np):
    args,options,events,buffers=fixture(spec,np)
    named=dict(arr_in=args[0],**options);before=encode((named,buffers),np)
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
            state=np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==old_buffer
    row=dict(id=spec['id'],api=API,input_before=before,input_after=encode((named,buffers),np),result=encode(result,np),error=error,warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)]for w in notices],events=events,result_is_input=result is args[0],result_shares_input=isinstance(result,np.ndarray)and any(np.shares_memory(result,value)for value in buffers.values()),checks=dict(policy_unchanged=state,state_restored=np.geterr()==old_policy and np.geterrcall()is old_handler and np.getbufsize()==old_buffer,pointers_stable=all(buffers[key].ctypes.data==value for key,value in pointers.items())))
    assert all(row['checks'].values()),spec['id']
    if error is None:assert type(result)is np.ndarray and not row['result_is_input']and not row['result_shares_input']
    json.dumps(row,allow_nan=False)
    return row


def test_montage_prospective_manifest():
    matrix=json.loads(MATRIX.read_bytes())
    assert matrix['api']==API and matrix['cases']==CASES and len(CASES)==58
    assert matrix['unassessed']==list(UNASSESSED)and matrix['native_implementation']=='pending'
    assert matrix['source_canonical_lf_sha256']==SOURCE_SHA and matrix['decorator_canonical_lf_sha256']==DECORATOR_SHA
    assert matrix['signature']==SIGNATURE and all(callable(globals()[name])for name in STAGES)
    for name,pin in matrix['source_freeze'].items():
        assert hashlib.sha256((MATRIX.parents[2]/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


@pytest.mark.parametrize('spec',CASES,ids=[spec['id']for spec in CASES])
def test_montage_fixture_serializes_before_original_calls(spec):
    np,_,_=runtime()
    args,options,events,buffers=fixture(spec,np)
    json.dumps(encode((args,options,buffers),np),allow_nan=False)
    assert not events
    assert all(value.nbytes<=8192 for value in buffers.values())


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_montage_saved_original_rows(profile):
    observed=json.loads(MATRIX.read_bytes())['original_observations']
    if not observed:pytest.skip('Prospective contract; original capture is not authorized yet')
    rows=observed[profile]
    assert len(rows)==58 and [row['id']for row in rows]==[spec['id']for spec in CASES]
    assert all(all(row['checks'].values())for row in rows)


@pytest.mark.parametrize('mutate',(False,True))
def test_montage_observer_serializes_nonfinites_and_preserves_mutation_evidence(mutate):
    np,_,_=runtime()
    spec=next(s for s in CASES if s['id']=='fill-nan')
    def standin(arr,**options):
        assert math.isnan(options['fill'])
        if mutate:arr.fill(7)
        return arr.copy()
    row=observe(standin,spec,np)
    assert row['error']is None and (row['input_before']!=row['input_after'])is mutate
    assert all(row['checks'].values())
    json.dumps(row,allow_nan=False)
