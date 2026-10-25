"""Original-only reconstruction benchmark contract; no product/timer/controller."""
import ast
import base64
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import sys
import warnings
import zlib
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frozen_reconstruction_cases',ROOT/'benchmarks/reconstruction_cases.py')
CASES=importlib.util.module_from_spec(spec);spec.loader.exec_module(CASES)
MATRIX=ROOT/'coverage/contracts/reconstruction-benchmark.json'


def profile():return 'windows' if os.name=='nt' else 'minimum' if np.__version__=='1.24.4' else 'modern'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def matrix():return json.loads(MATRIX.read_bytes())


def packed(raw,assets):
    key=digest(raw)
    if key not in assets:assets[key]=dict(raw_bytes=len(raw),codec='zlib-base64',payload=base64.b64encode(zlib.compress(raw,9)).decode())
    return dict(raw_sha256=key,raw_bytes=len(raw))


def array(a,assets,include_base=True):
    if a is None:return None
    if not isinstance(a,np.ndarray):return dict(type=type(a).__name__)
    out=dict(type=type(a).__name__,dtype=a.dtype.str,shape=list(a.shape),strides=list(a.strides),own=bool(a.flags.owndata),
        writeable=bool(a.flags.writeable),C=bool(a.flags.c_contiguous),F=bool(a.flags.f_contiguous),data=packed(a.tobytes(),assets))
    b=a.base
    if include_base:
        out['base']=None if b is None else array(b,assets,False)
        out['offset']=0 if b is None else int(a.ctypes.data-b.ctypes.data)
    return out


def source_auth(primary):
    import skimage
    from skimage.morphology import grayreconstruct as source
    assert skimage.__version__=='0.26.0'
    for name,want in CASES.PRIMARY_SOURCE_PINS.items():assert digest((primary/name).read_bytes())==want
    tag=ast.parse((primary/'grayreconstruct.py').read_bytes())
    declared=next(n for n in tag.body if isinstance(n,ast.FunctionDef) and n.name=='reconstruction')
    installed=ast.parse(inspect.getsource(source.reconstruction)).body[0]
    assert ast.dump(declared,include_attributes=False)==ast.dump(installed,include_attributes=False)
    from skimage.morphology import _grayreconstruct as compiled
    return source,dict(version=skimage.__version__,signature=str(inspect.signature(source.reconstruction)),
        installed_wrapper=str(Path(source.__file__).resolve()),installed_wrapper_raw_sha256=digest(Path(source.__file__).read_bytes()),
        Cython_binary=str(Path(compiled.__file__).resolve()),Cython_binary_raw_sha256=digest(Path(compiled.__file__).read_bytes()),
        numpy=np.__version__,python=sys.version,executable=sys.executable,primary_source_AST_exact=True)


def prepared_certificate(row,seed,mask,options,ranks,prev,next_,strides,start,plane):
    cert=CASES.certificate(row,seed,mask,options);assert plane==cert['plane']
    count=2*plane
    assert ranks.dtype==np.dtype('u4') and prev.dtype==next_.dtype==strides.dtype==np.dtype('i4')
    assert ranks.shape==prev.shape==next_.shape==(count,) and strides.ndim==1
    assert all(a.flags.c_contiguous and a.flags.aligned for a in (ranks,prev,next_,strides))
    assert 0<=start<count and np.all(prev>=-1) and np.all(prev<count) and np.all(next_>=-1) and np.all(next_<count)
    assert np.all(ranks<count) and np.all(strides>-plane) and np.all(strides<plane)
    padded=tuple(cert['padded_shape']);radius=row['radius'];inside=tuple(slice(radius,radius+s) for s in seed.shape)
    outside=np.ones(padded,bool);outside[inside]=False
    a=ranks[:plane].reshape(padded);b=ranks[plane:].reshape(padded)
    assert np.all(a[outside]==0) and np.all(b[outside]==0) and np.all(a[inside]<=b[inside])
    return dict(case_id=row['id'],plane=plane,rank_count=count,neighbors=int(strides.size),index_dtype=str(prev.dtype),
        rank_before_sha256=digest(ranks.tobytes()),links_before_sha256=[digest(prev.tobytes()),digest(next_.tobytes())],
        strides_raw=strides.tobytes().hex(),start=int(start),certified=True)


def observe(row,source,assets):
    seed,mask,options=CASES.fixture(row);cert=CASES.certificate(row,seed,mask,options)
    before=dict(seed=array(seed,assets),mask=array(mask,assets),footprint=array(options['footprint'],assets),seed_is_mask=seed is mask)
    calls=[];entry=source.reconstruction_loop
    def checked(ranks,prev,next_,strides,start,plane):
        call=prepared_certificate(row,seed,mask,options,ranks,prev,next_,strides,start,plane);calls.append(call)
        result=entry(ranks,prev,next_,strides,start,plane);call['completed']=True;call['rank_after_sha256']=digest(ranks.tobytes());return result
    source.reconstruction_loop=checked
    result=None;error=None
    try:
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='warn'):
            warnings.simplefilter('always')
            try:result=source.reconstruction(seed,mask,**options)
            except Exception as exc:error=dict(module=type(exc).__module__,type=type(exc).__name__,message=str(exc))
        notices=[dict(module=x.category.__module__,category=x.category.__name__,message=str(x.message)) for x in caught]
    finally:source.reconstruction_loop=entry
    after=dict(seed=array(seed,assets),mask=array(mask,assets),footprint=array(options['footprint'],assets),seed_is_mask=seed is mask)
    return dict(id=row['id'],before=before,after=after,result=array(result,assets),error=error,warnings=notices,
        aliases=dict(seed=result is seed,mask=result is mask,shares_seed=isinstance(result,np.ndarray) and np.shares_memory(result,seed),shares_mask=isinstance(result,np.ndarray) and np.shares_memory(result,mask)),
        changed_elements=None if result is None else int(np.count_nonzero(result!=seed)),source_backend=calls,certificate=cert)


def independent_fixed_point(row):
    seed,mask,options=CASES.fixture(row);dtype=np.float32 if mask.dtype==np.float32 else np.float64
    out=seed.astype(dtype);limit=mask.astype(dtype);fp=options['footprint'];radius=row['radius']
    offsets=[tuple(i-radius for i in index) for index in np.ndindex(fp.shape) if fp[index] and any(i!=radius for i in index)]
    for _ in range(2*seed.size+1):
        changed=False
        for point in np.ndindex(out.shape):
            value=out[point]
            for offset in offsets:
                other=tuple(a+b for a,b in zip(point,offset))
                if all(0<=p<s for p,s in zip(other,out.shape)):
                    candidate=min(out[other],limit[point]) if row['method']=='dilation' else max(out[other],limit[point])
                    value=max(value,candidate) if row['method']=='dilation' else min(value,candidate)
            if value!=out[point]:out[point]=value;changed=True
        if not changed:return out
    raise AssertionError('finite tiny fixed-point oracle did not converge')


def record(primary,output):
    data=matrix();assert data['case_manifest']==CASES.case_manifest()
    source,auth=source_auth(primary);assets={};observations={}
    for row in CASES.specifications():
        observed=observe(row,source,assets);observations[row['id']]=observed
        # Store the complete failure immediately. Validation happens only after
        # each row has been preserved; no case is removed or repaired in place.
        report=dict(profile=profile(),source=auth,observations=observations,assets=assets,native_acceptance=False,timing_rows=0)
        output.write_bytes((json.dumps(report,indent=2)+'\n').encode())
        assert observed['error'] is None,row['id']
        assert len(observed['source_backend'])==1 and observed['source_backend'][0]['completed']
        assert observed['before']==observed['after'] and not any(observed['aliases'].values())
        if 'flood' in row['group']:assert observed['changed_elements']>0,row['id']
        if row['group']=='identity':assert observed['changed_elements']==0
        if row['group']=='oracle':
            oracle=independent_fixed_point(row);assert observed['result']['dtype']==oracle.dtype.str
            assert observed['result']['data']['raw_sha256']==digest(oracle.tobytes())
    return report


def test_specs_frozen_and_complete():
    data=matrix();assert data['specifications']==CASES.specifications() and data['case_manifest']==CASES.case_manifest()
    assert len(data['specifications'])==64 and sum(s['intended_future_measurement'] for s in data['specifications'])==48
    assert sum(s['group']=='signedzero' for s in data['specifications'])==4
    assert data['native_acceptance'] is False and data['accepted_product_pins'] is None
    assert CASES.contract_hash(data['case_manifest'])==data['case_contract_sha256']


def test_native_pending_before_fixture(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('fixture created before admission')
    monkeypatch.setattr(CASES,'fixture',forbidden)
    with pytest.raises(RuntimeError,match='pending genuine public200'):CASES.native_admission()


def test_source_text_pins():
    data=matrix()
    for name,want in data['source_text_sha256'].items():assert digest((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==want
    assert data['reference_matrix_sha256']==CASES.REFERENCE_MATRIX_SHA256
    tree=ast.parse((ROOT/'benchmarks/reconstruction_cases.py').read_bytes())
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(a.name.split('.')[0] in ('time','timeit','skimagexx') for a in n.names) for n in ast.walk(tree))


@pytest.mark.parametrize('row',CASES.specifications(),ids=lambda s:s['id'])
def test_original_frozen_contract(row):
    data=matrix();primary=Path(os.environ['SKIMAGEXX_RECONSTRUCTION_PRIMARY_SOURCE']);source,_=source_auth(primary);assets={}
    actual=observe(row,source,assets);assert actual==data['profiles'][profile()]['observations'][row['id']]
    for key,asset in assets.items():assert digest(zlib.decompress(base64.b64decode(asset['payload'])))==key


@pytest.mark.parametrize('row',[r for r in CASES.specifications() if r['group']=='oracle'],ids=lambda s:s['id'])
def test_independent_fixed_point_exact(row):
    expected=matrix()['profiles'][profile()]['observations'][row['id']]['result'];out=independent_fixed_point(row)
    assert expected['dtype']==out.dtype.str and expected['data']['raw_sha256']==digest(out.tobytes())


@pytest.mark.parametrize('fault',('empty','rank','dtype','nonfinite','order','footprint','spec'))
def test_safety_certificate_rejects_before_Cython(fault):
    row=next(r for r in CASES.specifications() if r['dtype']==('f4' if fault=='nonfinite' else 'u1'));seed,mask,kw=CASES.fixture(row)
    if fault=='empty':seed=seed[:0];mask=mask[:0]
    elif fault=='rank':seed=seed.reshape(-1);mask=mask.reshape(-1)
    elif fault=='dtype':seed=seed.astype('c8')
    elif fault=='nonfinite':seed=seed.copy();seed.flat[0]=np.nan
    elif fault=='order':seed=seed.copy();seed.flat[0]=255;mask=mask.copy();mask.flat[0]=0
    elif fault=='footprint':kw['footprint']=np.ones((2,2),bool)
    else:row=dict(row,shape=[0,0])
    with pytest.raises(AssertionError):CASES.certificate(row,seed,mask,kw)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--primary-source',type=Path,required=True);parser.add_argument('--record',type=Path,required=True)
    args=parser.parse_args();assert not args.record.exists();record(args.primary_source,args.record)
