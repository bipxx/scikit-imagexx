"""Pinned0.26 Bregman reference contracts; no native implementation admission.

Only certified finite prepared buffers reach the unchecked Cython solver.
Python intake/error protocols use an explicit blocked entry and remain separate
from genuine numerical observations. Upstream source is BSD-3-Clause.
"""
import ast
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import struct
import sys
import warnings

import numpy as np
import pytest
import scipy
import skimage
import skimagexx as xx
from skimage.restoration import denoise_tv_bregman as reference
from skimagexx.coverage import native_functions

ROOT=Path(__file__).parents[1]
MATRIX=ROOT/'coverage/contracts/tv-bregman.json'
API='restoration.denoise_tv_bregman'
original_module=importlib.import_module('skimage.restoration._denoise')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def source_sha(function):return hashlib.sha256(inspect.getsource(function).replace('\r\n','\n').encode()).hexdigest()
def profile():return 'windows' if os.name=='nt' else 'minimum' if np.__version__.split('.')[0]=='1' else 'modern'


def cases():
    rows=[]
    for dtype in ('f4','f8'):
        for shape in ([4,4],[4,5],[7,9]):
            for iso in (False,True):
                for iterations in (1,2,5):
                    r=dict(id=f'core/{dtype}/{shape[0]}x{shape[1]}/{int(iso)}/i{iterations}',mode='compiled',dtype=dtype,shape=shape,pattern='ramp',layout='C',iterations=iterations,weight=0.5,eps=1e-6,isotropic=iso);rows.append(r)
        for layout in ('F','reverse','step','broadcast','tagged','readonly-F'):
            r=dict(id=f'layout/{dtype}/{layout}',mode='compiled',dtype=dtype,shape=[4,5],pattern='ramp',layout=layout,iterations=2,weight=1.,eps=1e-3,isotropic=False);rows.append(r)
        for pattern in ('zero','constant','impulse','checker'):
            for iso in (False,True):rows.append(dict(id=f'pattern/{dtype}/{pattern}/{int(iso)}',mode='compiled',dtype=dtype,shape=[7,9],pattern=pattern,layout='C',iterations=8,weight=0.25,eps=1e-6,isotropic=iso))
        for weight in (0.25,5.,8.):
            for eps in (1e-6,1.):rows.append(dict(id=f'parameter/{dtype}/w{weight}/eps{eps}',mode='compiled',dtype=dtype,shape=[4,5],pattern='ramp',layout='C',iterations=5,weight=weight,eps=eps,isotropic=True))
        for axis in (None,-1,0,1):
            for channels in (1,3):rows.append(dict(id=f'channels/{dtype}/axis{axis}/c{channels}',mode='compiled',dtype=dtype,shape=[4,5,channels],axis=axis,pattern='channels',layout='C',iterations=2,weight=0.5,eps=1e-6,isotropic=True))
        for iso in (False,True):rows.append(dict(id=f'bounded/{dtype}/64x64/{int(iso)}',mode='compiled',dtype=dtype,shape=[64,64],pattern='checker',layout='C',iterations=5,weight=0.5,eps=1e-3,isotropic=iso))
    for dtype in ('b1','u1','i1','u2','i2','u4','i4','u8','i8','>u2'):
        rows.append(dict(id=f'conversion/{dtype}',mode='compiled',dtype=dtype,shape=[4,5],pattern='integer',layout='C',iterations=2,weight=0.5,eps=1e-6,isotropic=False))
    # Literal first-sweep RMSE values from the source-ordered typed residual
    # fold, independently reconstructed from the frozen4x5 one-sweep outputs.
    for dtype,hexadecimal in (('f4','0x1.cf90c9e73f172p-2'),('f8','0x1.cf90c9f943110p-2')):
        boundary=float.fromhex(hexadecimal)
        for relation,eps in (('below',math.nextafter(boundary,0.)),('equal',boundary),('above',math.nextafter(boundary,math.inf))):
            rows.append(dict(id=f'stopping/{dtype}/{relation}',mode='compiled',dtype=dtype,shape=[4,5],pattern='ramp',layout='C',iterations=2,weight=0.5,eps=eps,isotropic=False))
    for fault in ('complex','string','axis-many','axis-empty','axis-range','axis-float','array-raises','weight-raises','zeros-raises','contiguous-raises','unexpected-keyword','positional-channel'):
        rows.append(dict(id='python-reject/'+fault,mode='python-reject',fault=fault,dtype='f8',shape=[4,5],pattern='ramp',layout='C',iterations=2,weight=0.5,eps=1e-6,isotropic=True))
    for protocol in ('f16','endian-f4','unaligned-f4','readonly-C','weight-event','max-index','eps-float','isotropic-int','axis-tuple','axis-numpy-int','allocation-order'):
        rows.append(dict(id='intake-only/'+protocol,mode='entry-blocked',protocol=protocol,dtype='f8',shape=[4,5],pattern='ramp',layout='C',iterations=2,weight=0.5,eps=1e-6,isotropic=True))
    assert len({r['id'] for r in rows})==len(rows)
    return tuple(rows)


CASES=cases()


class Tagged(np.ndarray):pass
class ProviderError(RuntimeError):pass
class EntryBlocked(RuntimeError):pass


def array(value):
    a=np.asarray(value)
    return dict(dtype=a.dtype.str,shape=list(a.shape),strides=list(a.strides),C=bool(a.flags.c_contiguous),F=bool(a.flags.f_contiguous),writeable=bool(a.flags.writeable),own=bool(a.flags.owndata),aligned=bool(a.flags.aligned),bytes=a.tobytes().hex())


def scalar(value):
    if isinstance(value,np.generic):return dict(type=type(value).__name__,dtype=value.dtype.str,bytes=value.tobytes().hex())
    if isinstance(value,(int,float,bool,str,type(None))):return dict(type=type(value).__name__,value=value.hex() if type(value) is float else value)
    return dict(type=type(value).__name__)


def fixture(spec):
    events=[];shape=tuple(spec['shape']);dtype=np.dtype(spec['dtype']);n=math.prod(shape)
    values=np.arange(n,dtype=np.int64).reshape(shape)
    pattern=spec['pattern']
    if dtype.kind=='b':image=(values%3==0)
    elif dtype.kind in 'iu':image=(values%13-(4 if dtype.kind=='i' else 0)).astype(dtype)
    else:
        data=((values%13)-6)/8.
        if pattern=='zero':data=np.zeros(shape)
        elif pattern=='constant':data=np.full(shape,0.25)
        elif pattern=='impulse':data=np.zeros(shape);data.reshape(-1)[n//2]=1.
        elif pattern=='checker':data=(values%2)*1.0-0.5
        elif pattern=='channels':data=data+np.arange(shape[-1])[None,None,:]/8.
        image=np.asarray(data,dtype=dtype)
    layout=spec['layout'];owner=image
    if layout=='F' or layout=='readonly-F':image=np.array(image,order='F');owner=image
    elif layout=='reverse':image=image[::-1,::-1];owner=image.base
    elif layout=='step':
        owner=np.zeros(tuple(2*s for s in shape),dtype=dtype);image=owner[tuple(slice(None,None,2) for _ in shape)];image[...] =np.asarray(values%13,dtype=dtype)/16. if dtype.kind=='f' else values%13
    elif layout=='broadcast':owner=image[:1,:1].copy();image=np.broadcast_to(owner,shape)
    elif layout=='unaligned':
        owner=np.zeros(n*dtype.itemsize+1,dtype=np.uint8);image=np.ndarray(shape,dtype,buffer=owner,offset=1);image[...] =np.asarray(((values%13)-6)/8.,dtype)
    elif layout=='tagged':image=image.view(Tagged);owner=image.base
    if layout=='readonly-F':image.flags.writeable=False
    options=dict(weight=spec['weight'],max_num_iter=spec['iterations'],eps=spec['eps'],isotropic=spec['isotropic'])
    if 'axis' in spec:
        axis=spec['axis'];options['channel_axis']=axis
        if axis is not None and axis!=-1:image=np.moveaxis(image,-1,axis)
    passed=image;fault=spec.get('fault');protocol=spec.get('protocol')
    if fault=='complex':passed=image.astype(np.complex128);image=owner=passed
    if fault=='string':passed=np.full(shape,'x',dtype='U1');image=owner=passed
    if fault in ('axis-many','axis-empty','axis-range','axis-float'):options['channel_axis']={'axis-many':(0,1),'axis-empty':(),'axis-range':99,'axis-float':0.5}[fault]
    if fault=='array-raises':
        class ArrayRaises:
            def __array__(self,*args,**kwargs):events.append('image.array');raise ProviderError('image array rejected')
        passed=ArrayRaises()
    if fault=='weight-raises':
        class WeightRaises:
            def __float__(self):events.append('weight.float');raise ProviderError('weight conversion rejected')
        options['weight']=WeightRaises()
    if fault=='unexpected-keyword':options['unrecognized']=True
    if protocol=='f16':passed=image.astype(np.float16);image=owner=passed
    if protocol=='endian-f4':passed=image.astype('>f4');image=owner=passed
    if protocol=='unaligned-f4':
        owner=np.zeros(n*4+1,dtype=np.uint8);view=np.ndarray(shape,np.float32,buffer=owner,offset=1);view[...] =image;passed=image=view
    if protocol=='readonly-C':image.flags.writeable=False
    if protocol=='weight-event':
        class WeightEvent:
            def __float__(self):events.append('weight.float');return 0.5
        options['weight']=WeightEvent()
    if protocol in ('max-index','eps-float','isotropic-int'):
        class IndexProvider:
            def __index__(self):events.append('parameter.index');raise ProviderError('deferred index')
            def __int__(self):events.append('parameter.int');raise ProviderError('deferred int')
            def __float__(self):events.append('parameter.float');raise ProviderError('deferred float')
        options[{'max-index':'max_num_iter','eps-float':'eps','isotropic-int':'isotropic'}[protocol]]=IndexProvider()
    if protocol=='axis-tuple':options['channel_axis']=(-1,)
    if protocol=='axis-numpy-int':options['channel_axis']=np.int64(-1)
    return passed,image,owner,options,events


def certify_entry(image,weight,maximum,eps,isotropic,out):
    # This certificate is evaluated BEFORE invoking the unchecked original.
    assert type(image) is np.ndarray and image.ndim==3
    assert image.dtype in (np.dtype('f4'),np.dtype('f8')) and image.dtype.isnative
    assert image.flags.c_contiguous and image.flags.writeable and image.flags.aligned
    r,c,d=image.shape;assert 4<=r<=64 and 4<=c<=64 and 1<=d<=3
    assert np.isfinite(image).all() and np.abs(image).max()<=2
    assert type(weight) in (np.float32,np.float64) and 0.25<=weight<=8
    assert type(maximum) is int and 1<=maximum<=8
    assert type(eps) is float and 1e-6<=eps<=1
    assert type(isotropic) is bool
    assert type(out) is np.ndarray and out.dtype==image.dtype and out.shape==(r+2,c+2,d)
    assert out.flags.c_contiguous and out.flags.writeable and out.flags.aligned and not np.shares_memory(out,image)
    assert (image.nbytes+out.nbytes*5)<8*1024*1024


def observe(function,spec):
    passed,image,owner,options,events=fixture(spec);before=dict(image=array(image),owner=array(owner));pointer=image.__array_interface__['data'][0]
    entry=original_module._denoise_tv_bregman;img_float=original_module.img_as_float;zeros=np.zeros;contiguous=np.ascontiguousarray;atleast=np.atleast_3d
    entries=[];provider_calls=[]
    def traced_float(value):
        provider_calls.append('img_as_float');return img_float(value)
    def traced_zeros(*args,**kwargs):
        provider_calls.append('zeros')
        if spec.get('fault')=='zeros-raises':raise ProviderError('zeros rejected')
        return zeros(*args,**kwargs)
    def traced_contiguous(*args,**kwargs):
        provider_calls.append('ascontiguousarray')
        if spec.get('fault')=='contiguous-raises':raise ProviderError('contiguous rejected')
        return contiguous(*args,**kwargs)
    def traced_atleast(*args,**kwargs):provider_calls.append('atleast_3d');return atleast(*args,**kwargs)
    def guarded(image,weight,maximum,eps,isotropic,out):
        row=dict(image=array(image),weight=scalar(weight),maximum=scalar(maximum),eps=scalar(eps),isotropic=scalar(isotropic),out_before=array(out),image_shares_input=bool(np.shares_memory(image,passed)) if isinstance(passed,np.ndarray) else False)
        entries.append(row)
        if spec['mode']!='compiled':raise EntryBlocked('compiled original entry intentionally blocked')
        certify_entry(image,weight,maximum,eps,isotropic,out)
        result=entry(image,weight,maximum,eps,isotropic,out);row['out_after']=array(out);row['return_is_none']=result is None;return result
    result=None;error=None;old=np.seterr(all='warn')
    try:
        original_module._denoise_tv_bregman=guarded;original_module.img_as_float=traced_float;np.zeros=traced_zeros;np.ascontiguousarray=traced_contiguous;np.atleast_3d=traced_atleast
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                if spec.get('fault')=='positional-channel':result=function(passed,0.5,2,1e-6,True,None)
                else:result=function(passed,**options)
            except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
            captured=[dict(type=type(w.message).__name__,message=str(w.message)) for w in caught]
    finally:
        original_module._denoise_tv_bregman=entry;original_module.img_as_float=img_float;np.zeros=zeros;np.ascontiguousarray=contiguous;np.atleast_3d=atleast;np.seterr(**old)
    outcome=None
    if result is not None:
        outcome=dict(array=array(result),type=type(result).__name__,shares_input=bool(np.shares_memory(result,image)),is_input=result is image,base=array(result.base) if isinstance(result.base,np.ndarray) else None)
    return dict(before=before,after=dict(image=array(image),owner=array(owner)),pointer_stable=pointer==image.__array_interface__['data'][0],result=outcome,error=error,warnings=captured,events=events,providers=provider_calls,entries=entries)


@pytest.fixture(scope='module')
def frozen():return json.loads(MATRIX.read_text())


def test_source_signature_audited_domain_and_explicit_native0(frozen):
    assert skimage.__version__=='0.26.0' and frozen['native_checks']==0
    assert str(inspect.signature(reference))==frozen['signature']
    assert reference is original_module.denoise_tv_bregman
    assert source_sha(reference)==frozen['source_sha256']
    assert digest(CASES)==frozen['case_contract_sha256'] and list(CASES)==frozen['cases']
    assert frozen['profiles'][profile()]['numpy']==np.__version__ and frozen['profiles'][profile()]['scipy']==scipy.__version__
    assert hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==frozen['test_source_canonical_lf_sha256']
    assert frozen['sources']['src/skimage/restoration/_denoise_cy.pyx']['sha256']=='737fa38f080af85c41156d2dfe758c499bc22790b9a8a50c976fbbe8653aa124'
    for row in frozen['sources'].values():assert hashlib.sha256(row['utf8'].encode()).hexdigest()==row['sha256']


@pytest.mark.parametrize('spec',CASES,ids=lambda s:s['id'])
def test_original_safe_numeric_or_blocked_intake_observation(spec,frozen):
    expected=frozen['observations'][frozen['profiles'][profile()]['snapshots'][spec['id']]]
    actual=observe(reference,spec);assert actual==expected,spec['id']
    assert actual['before']==actual['after'] and actual['pointer_stable']
    if spec['mode']=='compiled':
        assert actual['error'] is None and actual['entries'] and not actual['warnings']
        assert not actual['result']['shares_input'] and not actual['result']['array']['own']
        assert actual['result']['base']['own']
    elif spec['mode']=='python-reject':assert not actual['entries'] and actual['error']['type']!='EntryBlocked'
    else:assert len(actual['entries'])==1 and actual['error']['type']=='EntryBlocked'


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',CASES,ids=lambda s:s['id'])
def test_native_pending_never_counts_upstream_forwarding(spec,workers,monkeypatch):
    assert API not in native_functions()
    def forbidden(*args,**kwargs):raise AssertionError('original Cython entry forbidden in native-pending tests')
    monkeypatch.setattr(original_module,'_denoise_tv_bregman',forbidden)
    before=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    try:
        passed,_,_,options,_=fixture(spec)
        with pytest.raises(NotImplementedError):getattr(xx.restoration,'denoise_tv_bregman')(passed,**options)
    finally:xx.set_num_threads(before);xx.set_fallback_enabled(fallback)



def independent_sweeps(image,weight,steps,isotropic):
    """Typed scalar finite-grid definition; no upstream/provider algorithm call."""
    r,c=image.shape
    cast=(lambda v:struct.unpack('<f',struct.pack('<f',float(v)))[0]) if image.dtype.itemsize==4 else float
    add=lambda a,b:cast(float(a)+float(b));sub=lambda a,b:cast(float(a)-float(b));mul=lambda a,b:cast(float(a)*float(b));div=lambda a,b:cast(float(a)/float(b))
    grids=[[[cast(0) for _ in range(c+2)] for _ in range(r+2)] for _ in range(5)]
    out,dx,dy,bx,by=grids
    for y in range(r):
        for x in range(c):out[y+1][x+1]=cast(image[y,x])
    for x in range(c):out[0][x+1]=cast(image[1,x]);out[-1][x+1]=cast(image[-1,x])
    for y in range(r):out[y+1][0]=cast(image[y,1]);out[y+1][-1]=cast(image[y,-1])
    w=cast(weight);lam=mul(2,w);norm=add(w,mul(4,lam))
    for _ in range(steps):
        for y in range(1,r+1):
            for x in range(1,c+1):
                previous=out[y][x];ux=sub(out[y][x+1],previous);uy=sub(out[y+1][x],previous)
                total=add(out[y+1][x],out[y-1][x]);total=add(total,out[y][x+1]);total=add(total,out[y][x-1])
                for operator,value in ((add,dx[y][x-1]),(sub,dx[y][x]),(add,dy[y-1][x]),(sub,dy[y][x]),(sub,bx[y][x-1]),(add,bx[y][x]),(sub,by[y-1][x]),(add,by[y][x])):total=operator(total,value)
                out[y][x]=div(add(mul(lam,total),mul(w,cast(image[y-1,x-1]))),norm)
                tx=add(ux,bx[y][x]);ty=add(uy,by[y][x])
                if isotropic:
                    length=cast(math.sqrt(add(mul(tx,tx),mul(ty,ty))))
                    denominator=float(mul(length,lam))+1.0  # Cython literal promotes denominator to double.
                    dxx=div(mul(mul(length,lam),tx),denominator);dyy=div(mul(mul(length,lam),ty),denominator)
                else:
                    threshold=div(1,lam)
                    shrink=lambda v:sub(v,threshold) if v>threshold else add(v,threshold) if v<-threshold else cast(0)
                    dxx=shrink(tx);dyy=shrink(ty)
                dx[y][x]=dxx;dy[y][x]=dyy;bx[y][x]=add(bx[y][x],sub(ux,dxx));by[y][x]=add(by[y][x],sub(uy,dyy))
    return np.array([row[1:-1] for row in out[1:-1]],dtype=image.dtype)


@pytest.mark.parametrize('dtype',('f4','f8'))
@pytest.mark.parametrize('isotropic',(False,True))
@pytest.mark.parametrize('steps',(1,2))
def test_independent_small_gauss_seidel_definition(dtype,isotropic,steps):
    spec=dict(id='independent',mode='compiled',dtype=dtype,shape=[4,5],pattern='ramp',layout='C',iterations=steps,weight=0.5,eps=1e-6,isotropic=isotropic)
    image=fixture(spec)[1];expected=independent_sweeps(image,0.5,steps,isotropic);observed=observe(reference,spec)
    assert observed['error'] is None
    actual=observed['result']['array'];assert actual['dtype']==expected.dtype.str and actual['shape']==list(expected.shape)
    assert actual['bytes']==expected.tobytes().hex()


@pytest.mark.parametrize('dtype',('f4','f8'))
def test_fixed_asymmetric_halos_and_padded_return_owner(dtype):
    spec=dict(id='halo',mode='compiled',dtype=dtype,shape=[4,5],pattern='ramp',layout='C',iterations=2,weight=0.5,eps=1e-6,isotropic=False)
    image=fixture(spec)[1];observed=observe(reference,spec);base=observed['result']['base'];pad=np.frombuffer(bytes.fromhex(base['bytes']),np.dtype(base['dtype'])).reshape(base['shape'])
    assert pad.shape==(6,7,1) and base['own']
    assert pad[0,1:-1,0].tobytes()==image[1].tobytes()
    assert pad[-1,1:-1,0].tobytes()==image[-1].tobytes()
    assert pad[1:-1,0,0].tobytes()==image[:,1].tobytes()
    assert pad[1:-1,-1,0].tobytes()==image[:,-1].tobytes()
    assert not np.any(pad[np.ix_([0,5],[0,6],[0])])
    assert observed['result']['array']['strides']==[7*image.dtype.itemsize,image.dtype.itemsize]
    assert not observed['result']['array']['C'] and not observed['result']['shares_input']


@pytest.mark.parametrize('dtype',('f4','f8'))
def test_strict_rmse_stopping_boundary_is_source_ordered(dtype,frozen):
    snapshots=frozen['profiles'][profile()]['snapshots'];records=frozen['observations']
    first=records[snapshots[f'core/{dtype}/4x5/0/i1']]
    info=first['result']['array'];code='f' if dtype=='f4' else 'd';itemsize=4 if code=='f' else 8
    unpack=lambda value:struct.unpack('<'+code*(len(bytes.fromhex(value['bytes']))//itemsize),bytes.fromhex(value['bytes']))
    cast=(lambda v:struct.unpack('<f',struct.pack('<f',v))[0]) if code=='f' else float
    total=0.
    for x,y in zip(unpack(info),unpack(first['before']['image'])):
        delta=cast(x-y);total+=cast(delta*delta)
    boundary=math.sqrt(total/math.prod(info['shape']))
    spec=next(c for c in CASES if c['id']==f'stopping/{dtype}/equal');assert spec['eps']==boundary
    def output(relation):return records[snapshots[f'stopping/{dtype}/{relation}']]['result']['array']['bytes']
    assert output('equal')==output('above')==info['bytes']
    assert output('below')!=info['bytes']  # while rmse>eps, not >=eps.

def record(path):
    observations={};snapshots={};repeat=[]
    for spec in CASES:
        first=observe(reference,spec);second=observe(reference,spec)
        if first!=second:repeat.append(spec['id'])
        key=digest(first);observations[key]=first;snapshots[spec['id']]=key
    value=dict(profile=profile(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,signature=str(inspect.signature(reference)),source_sha256=source_sha(reference),cases=list(CASES),case_contract_sha256=digest(CASES),observations=observations,snapshots=snapshots,repeat_differences=repeat,test_source_canonical_lf_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),native_checks=0)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(profile=profile(),cases=len(CASES),repeat_differences=repeat,native_checks=0)))
    return int(bool(repeat))


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--record':raise SystemExit('use --record OUTPUT')
    raise SystemExit(record(Path(sys.argv[2])))
