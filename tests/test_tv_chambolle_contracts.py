"""Pinned original TV-Chambolle observations and owned public parity checks.

All reference calls are bounded NumPy code, not unchecked Cython probes. The
closed-form first dual update supplies an independent two-iteration oracle.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch
import warnings

import numpy as np
import pytest
import scipy
import skimage
from skimage.restoration import denoise_tv_chambolle as reference
import skimage.restoration._denoise as original_module
import skimagexx as xx
from skimagexx.coverage import native_functions

ROOT=Path(__file__).parents[1]
MATRIX=ROOT/'coverage/contracts/tv-chambolle.json'
API='restoration.denoise_tv_chambolle'
REAL=('b1','u1','i1','u2','i2','u4','i4','u8','i8','f2','f4','f8')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def source_sha(function):return hashlib.sha256(inspect.getsource(function).replace('\r\n','\n').encode()).hexdigest()
def profile():return 'windows' if os.name=='nt' else 'minimum' if np.__version__.split('.')[0]=='1' else 'modern'


def cases():
    result=[]
    def put(key,**options):
        spec=dict(id=key,dtype='f8',shape=[3,4],iterations=3);spec.update(options);result.append(spec)
    for dtype in REAL:
        for shape in ([3,4],[2,3,4]):
            for count in (1,2,7):put(f'base/{dtype}/{len(shape)}d/i{count}',dtype=dtype,shape=shape,iterations=count)
    for dtype in ('u1','i1','u2','i2','u4','i4','u8','i8'):
        for count in (1,3):put(f'extremes/{dtype}/i{count}',dtype=dtype,pattern='extremes',iterations=count)
    for dtype in ('f4','f8'):
        for layout in ('F','reverse','step','broadcast','readonly','unaligned','tagged','endian'):
            for count in (1,4):put(f'layout/{dtype}/{layout}/i{count}',dtype=dtype,layout=layout,iterations=count)
        for axis in (None,0,1,2,-1,-2,-3):
            for count in (1,2,7):put(f'channels/{dtype}/{axis}/i{count}',dtype=dtype,shape=[3,4,2],axis=axis,iterations=count)
        for axis in (0,1,-1):put(f'channels4d/{dtype}/{axis}',dtype=dtype,shape=[2,3,4,2],axis=axis)
    for shape in ([],[0],[0,3],[2,0,3],[1,1],[1,3,1]):
        for count in (0,1,2):put('shape/'+str(shape)+f'/i{count}',shape=shape,iterations=count)
    for shape,axis in (([0,2,3],0),([2,0,3],1),([2,3,0],-1)):
        for count in (0,1,3):put('zerochannels/'+str(shape)+f'/i{count}',shape=shape,axis=axis,iterations=count)
    for value in (-1,False,True,1.25,2.5,None,'bad','np-i2','np-f2','nan'):
        put('iteration/'+str(value),iterations=value)
    for value in (0,-0.0,-1,0.01,1,1e-300,1e300,'nan','inf','-inf',None,'bad','array0','array1','array2'):
        put('weight/'+str(value),weight=value,shape=[3,2])
    for value in (-1,0,1,1e10,'nan','inf','-inf',None,'bad','array0','array2'):
        put('eps/'+str(value),eps=value)
    for value in (20,-20,True,1.5,'bad','np-i2','np-f2'):
        put('axis/'+str(value),axis=value)
    for dtype in ('c8','c16','O','U2'):
        put('unsupported/'+dtype,dtype=dtype)
    put('input/list',input='list');put('input/none',input='none')
    put('default/small',shape=[2,3],iterations='default')
    put('default/constant',shape=[2,3],pattern='constant',iterations='default')
    for dtype in ('f4','f8'):
        for pattern in ('nonfinite','large'):
            for policy in ('warn','ignore','raise','call','log'):
                put(f'fp/{dtype}/{pattern}/{policy}',dtype=dtype,pattern=pattern,policy=policy)
            for policy in ('warn','raise','call'):
                put(f'oneiter/{dtype}/{pattern}/{policy}',dtype=dtype,pattern=pattern,policy=policy,iterations=1)
    for key,options in [('eps-bad',dict(eps='bad')),('eps-nan',dict(eps='nan')),('weight-none',dict(weight=None)),('weight-zero',dict(weight=0))]:
        put('oneiter/'+key,iterations=1,**options)
    put('zeroiter/weight-none',iterations=0,weight=None)
    put('zeroiter/eps-bad',iterations=0,eps='bad')
    for action in ('fill','raise'):
        put('warning-hook/'+action,pattern='nonfinite',warning_hook=action)
        put('fp-hook/'+action,pattern='nonfinite',policy='call',fp_action=action)
    put('snan/f4',dtype='f4',pattern='snan')
    for dtype in ('f4','u1'):
        for intake in ('trace','dtype-1','dtype-2','array','astype','astype-fill'):
            put(f'intake/{dtype}/{intake}',dtype=dtype,intake=intake)
    for token in ('count','count-fail0','count-fail1','count-fill1','axis','axis-fail','weight','weight-mul-fail','weight-div-fail','weight-fill','eps','eps-fail'):
        put('parameter/'+token,parameter=token)
    put('parameter/count-zero-energy-strict-stop',parameter='count',pattern='constant',eps=100.)
    for axis in (None,-1):
        for allocation in ('trace','fail-1','fail-2','fail-3','fill-first'):
            put(f'allocation/{axis}/{allocation}',shape=[2,3,2],axis=axis,allocation=allocation)
        for allocation in ('trace','fail-1','fail-3'):
            put(f'allocation-zeroiter/{axis}/{allocation}',shape=[2,3,2],axis=axis,allocation=allocation,iterations=0)
    assert len({p['id'] for p in result})==len(result)
    return tuple(result)


CASES=cases()


class Tagged(np.ndarray):
    def __array_finalize__(self,parent):self.marker=getattr(parent,'marker','tv-tag')


class SourceProviderError(RuntimeError):pass


def scalar(value):
    if isinstance(value,np.generic):
        return dict(type=type(value).__name__,dtype=value.dtype.str,bytes=value.tobytes().hex())
    if isinstance(value,float):return dict(type='float',hex=value.hex())
    return dict(type=type(value).__name__,value=value)


def array(value):
    if not isinstance(value,np.ndarray):return dict(type=type(value).__name__,value=value)
    result=dict(type=type(value).__name__,dtype=value.dtype.str,shape=list(value.shape),strides=list(value.strides),
                flags={n:bool(value.flags[n]) for n in ('C_CONTIGUOUS','F_CONTIGUOUS','OWNDATA','WRITEABLE','ALIGNED')},
                base_type=type(value.base).__name__,marker=getattr(value,'marker',None))
    if value.dtype.hasobject:result['values']=value.tolist()
    else:result['bytes']=value.tobytes(order='C').hex()
    return result


def token(value):
    if value=='np-i2':return np.int64(2)
    if value=='np-f2':return np.float64(2)
    if value=='nan':return float('nan')
    if value=='inf':return float('inf')
    if value=='-inf':return -float('inf')
    if value=='array0':return np.array(0.1)
    if value=='array1':return np.array([0.1])
    if value=='array2':return np.array([0.1,0.2])
    return value


def fixture(spec):
    dtype=np.dtype(spec['dtype']);shape=tuple(spec['shape']);size=math.prod(shape);pattern=spec.get('pattern','ramp')
    if pattern=='extremes':
        limits=np.iinfo(dtype);values=[limits.min,limits.max,0,1]*(size//4+1)
    elif dtype.kind=='b':values=[i%3==0 for i in range(size)]
    elif dtype.kind=='u':values=[(i*7+3)%17 for i in range(size)]
    elif dtype.kind=='i':values=[(i*7)%23-11 for i in range(size)]
    elif dtype.kind in 'fc':values=[((i*7)%23-11)/8 for i in range(size)]
    elif dtype.kind=='O':values=[i%3 for i in range(size)]
    else:values=['a']*size
    if pattern=='constant':values=[0.25]*size
    elif pattern=='nonfinite':values=[float('nan'),float('inf'),-float('inf'),-0.]*(size//4+1)
    elif pattern=='large':values=[np.finfo(dtype).max,-np.finfo(dtype).max,0.,1.]*(size//4+1)
    if pattern=='snan':
        bits=np.array(([0x7f800001,0x3f800000,0x80000000,0x7fc01234]*(size//4+1))[:size],np.uint32)
        image=bits.view(np.float32).reshape(shape)
    else:image=np.array(values[:size],dtype=dtype).reshape(shape)
    layout=spec.get('layout','C');owner=image
    if layout=='F':image=np.asfortranarray(image);owner=image
    elif layout=='reverse':image=image[...,::-1]
    elif layout=='step':
        owner=np.zeros((shape[0]*2,)+shape[1:],dtype);owner[::2]=image;image=owner[::2]
    elif layout=='broadcast':image=np.broadcast_to(image[:1],shape)
    elif layout=='readonly':image.setflags(write=False)
    elif layout=='unaligned':
        owner=np.zeros(image.nbytes+1,np.uint8);view=np.ndarray(shape,dtype,buffer=owner,offset=1);view[...]=image;image=view
    elif layout=='tagged':image=image.view(Tagged)
    elif layout=='endian':image=image.astype(dtype.newbyteorder('>' if sys.byteorder=='little' else '<'));owner=image
    if spec.get('input')=='list':image=image.tolist();owner=image
    elif spec.get('input')=='none':image=None;owner=None
    events=[];passed=image
    if 'intake' in spec:
        mode=spec['intake']
        class Intake:
            reads=0
            @property
            def dtype(self):
                self.reads+=1;events.append('dtype:'+str(self.reads))
                if mode=='dtype-'+str(self.reads):raise SourceProviderError(mode)
                return image.dtype
            def __array__(self,dtype=None,copy=None):
                events.append(dict(array_dtype=None if dtype is None else np.dtype(dtype).str,copy=copy))
                if mode=='array':raise SourceProviderError(mode)
                result=image if dtype is None else image.astype(dtype,copy=False)
                return result.copy() if copy else result
            def astype(self,dtype,copy=True):
                events.append(dict(astype=np.dtype(dtype).str,copy=copy))
                if mode=='astype':raise SourceProviderError(mode)
                if mode=='astype-fill':image.fill(0)
                return image.astype(dtype,copy=copy)
        passed=Intake()
    options={k:token(spec[k]) for k in ('weight','eps') if k in spec}
    if spec['iterations']!='default':options['max_num_iter']=token(spec['iterations'])
    if 'axis' in spec:options['channel_axis']=token(spec['axis'])
    parameter=spec.get('parameter')
    if parameter:
        class Count:
            def __gt__(self,value):
                events.append('count>'+str(value))
                if parameter=='count-fail'+str(value):raise SourceProviderError(parameter)
                if parameter=='count-fill1' and value==1:image.fill(0)
                return value<2
        class Axis:
            def __mod__(self,value):
                events.append('axis%'+str(value))
                if parameter=='axis-fail':raise SourceProviderError(parameter)
                return 1%value
        class Weight:
            def __mul__(self,value):
                events.append(dict(weight_mul=scalar(value)))
                if parameter=='weight-mul-fail':raise SourceProviderError(parameter)
                if parameter=='weight-fill':image.fill(0)
                return 0.5*value
            def __rtruediv__(self,value):
                events.append(dict(weight_div=scalar(value)))
                if parameter=='weight-div-fail':raise SourceProviderError(parameter)
                return value/0.5
        class Eps:
            def __mul__(self,value):
                events.append(dict(eps_mul=scalar(value)))
                if parameter=='eps-fail':raise SourceProviderError(parameter)
                return 0.*value
        if parameter.startswith('count'):options['max_num_iter']=Count()
        elif parameter.startswith('axis'):options['channel_axis']=Axis()
        elif parameter.startswith('weight'):options['weight']=Weight()
        else:options['eps']=Eps()
    return passed,image,owner,options,events


@contextmanager
def allocation_hooks(spec,image,events):
    mode=spec.get('allocation')
    if mode is None:yield;return
    count=0;zeros=np.zeros;zeros_like=np.zeros_like
    def action(name,shape,dtype):
        nonlocal count
        count+=1;events.append(dict(allocation=name,shape=list(shape),dtype=np.dtype(dtype).str))
        if mode=='fail-'+str(count):raise SourceProviderError(mode)
        if mode=='fill-first' and count==1:image.fill(0)
    def create(shape,*args,**kwargs):action('zeros',shape,kwargs.get('dtype',float));return zeros(shape,*args,**kwargs)
    def like(other,*args,**kwargs):action('zeros_like',other.shape,kwargs.get('dtype',other.dtype));return zeros_like(other,*args,**kwargs)
    with patch.object(original_module.np,'zeros',create),patch.object(original_module.np,'zeros_like',like):yield


def observe(function,spec):
    passed,image,owner,options,events=fixture(spec)
    before=dict(image=array(image),owner=array(owner));pointer=None if not isinstance(image,np.ndarray) else image.ctypes.data
    old=np.geterr();oldcall=np.geterrcall();fired=False
    def action(kind):
        nonlocal fired
        if fired:return
        fired=True
        if kind=='fill':image.fill(0)
        elif kind=='raise':raise SourceProviderError('policy-callback')
    class Handler:
        def __call__(self,message,flags):events.append(dict(fp_call=message,flags=int(flags)));action(spec.get('fp_action'))
        def write(self,message):events.append(dict(fp_log=message.strip()))
    try:
        np.seterrcall(Handler());np.seterr(all=spec.get('policy','warn'))
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');previous=warnings.showwarning
            def warning(message,category,*args,**kwargs):
                events.append(dict(warning_hook=str(message)))
                action(spec['warning_hook']);return previous(message,category,*args,**kwargs)
            if 'warning_hook' in spec:warnings.showwarning=warning
            try:
                with allocation_hooks(spec,image,events):value=function(passed,**options)
                error=None
            except Exception as caught:value=None;error=dict(type=type(caught).__name__,message=str(caught))
            finally:warnings.showwarning=previous
        result=None
        if value is not None:
            result=dict(array=array(value),identity_image=value is image,identity_owner=value is owner,
                        shares_image=isinstance(image,np.ndarray) and bool(np.shares_memory(value,image)),
                        shares_owner=isinstance(owner,np.ndarray) and bool(np.shares_memory(value,owner)),
                        base_is_image=isinstance(value,np.ndarray) and value.base is image,
                        base_is_owner=isinstance(value,np.ndarray) and value.base is owner)
        return dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],events=events,before=before,after=dict(image=array(image),owner=array(owner)),pointer_stable=pointer is None or image.ctypes.data==pointer)
    finally:np.seterr(**old);np.seterrcall(oldcall)


def result_nan_encoding_contract(raw):
    """tv-result-nan-encoding-v1: only result NaN words are unpromised.

    NumPy1.24 original replays choose different NaN signs/payloads for some
    SIMD/alignment paths. Raw observations remain stored separately. All NaN
    positions and every non-NaN byte (including infinities/signed zero), input
    payloads, result metadata, ownership, events and warnings stay exact.
    """
    result=copy.deepcopy(raw)
    if result.get('result') is not None:
        info=result['result']['array'];dtype=np.dtype(info['dtype'])
        if dtype.kind=='f':
            value=np.frombuffer(bytes.fromhex(info['bytes']),dtype).copy()
            value[np.isnan(value)]=np.array(np.nan,dtype=dtype)
            info['bytes']=value.tobytes().hex()
    return result


@pytest.fixture(scope='module')
def frozen():return json.loads(MATRIX.read_text())


def test_source_signature_safety_and_native0(frozen):
    assert skimage.__version__==frozen['skimage']=='0.26.0'
    assert frozen['native_checks']==0 and frozen['public_paths']==[API]
    assert str(inspect.signature(reference))==frozen['signature']
    assert reference is original_module.denoise_tv_chambolle
    assert source_sha(reference)==frozen['source_sha256']
    assert source_sha(original_module._denoise_tv_chambolle_nd)==frozen['helper_sha256']
    migration=json.loads((ROOT/'coverage/evidence/tv-chambolle-public-assertion-migration.json').read_bytes())
    assert migration['schema']=='tv-chambolle-public-assertion-migration-v1'
    assert migration['source_file']=='tests/test_tv_chambolle_contracts.py'
    assert migration['original_source_sha256']==frozen['test_source_canonical_lf_sha256']
    assert migration['reference_matrix_sha256']==hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    assert hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==migration['public_source_sha256']
    assert list(CASES)==frozen['cases'] and digest(CASES)==frozen['case_contract_sha256']
    assert frozen['safety']['max_elements']==max(math.prod(p['shape']) for p in CASES)
    assert all(math.prod(p['shape'])<=64 and len(p['shape'])<=4 for p in CASES)
    assert frozen['profiles'][profile()]['numpy']==np.__version__ and frozen['profiles'][profile()]['scipy']==scipy.__version__
    assert set(frozen['profiles'][profile()]['snapshots'])=={p['id'] for p in CASES}
    assert all(digest(v)==k for k,v in frozen['observations'].items())
    assert frozen['comparison']=='tv-result-nan-encoding-v1'


@pytest.mark.parametrize('spec',CASES,ids=[p['id'] for p in CASES])
def test_original_frozen_values_errors_warnings_providers_ownership(spec,frozen):
    expected=frozen['observations'][frozen['profiles'][profile()]['snapshots'][spec['id']]]
    assert result_nan_encoding_contract(observe(reference,spec))==result_nan_encoding_contract(expected),spec['id']


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',CASES,ids=[p['id'] for p in CASES])
def test_strict_native_owned_values_without_forwarding(spec,workers,frozen):
    assert API in native_functions(),'actual public registration is required; no automatic skip or source adapter'
    policy=xx.set_fallback_enabled(False);threads=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        function=importlib.import_module('skimagexx.restoration').denoise_tv_chambolle
        native=importlib.import_module('skimagexx._tv_chambolle')
        assert function is native.denoise_tv_chambolle and function is not reference
        assert function.__module__=='skimagexx._tv_chambolle'
        assert inspect.signature(function)==inspect.signature(reference)
        forbidden={reference.__code__,original_module._denoise_tv_chambolle_nd.__code__}
        forwarded=[]
        previous=sys.getprofile()
        def guard(frame,event,arg):
            if event=='call' and frame.f_code in forbidden:
                forwarded.append(frame.f_code.co_name)
                raise AssertionError('upstream TV algorithm/helper forwarding')
            if previous is not None:previous(frame,event,arg)
        try:
            sys.setprofile(guard)
            actual=observe(function,spec)
        finally:sys.setprofile(previous)
        assert not forwarded,'upstream TV algorithm/helper forwarding'
        expected=frozen['observations'][frozen['profiles'][profile()]['snapshots'][spec['id']]]
        assert result_nan_encoding_contract(actual)==result_nan_encoding_contract(expected),spec['id']
    finally:xx.set_num_threads(threads);xx.set_fallback_enabled(policy)


def dual_first_update(values,shape,weight):
    """Independent finite-grid definition; explicit FP order, no TV helper."""
    points=list(np.ndindex(shape));lookup=dict(zip(points,map(float,values)));ndim=len(shape);tau=1./(2.*ndim);dual={}
    for point in points:
        gradient=[]
        for axis in range(ndim):
            neighbor=list(point);neighbor[axis]+=1
            gradient.append(lookup[tuple(neighbor)]-lookup[point] if neighbor[axis]<shape[axis] else 0.)
        squared=0.
        for value in gradient:squared+=value*value
        denominator=math.sqrt(squared)*(tau/weight)+1.
        dual[point]=[(0.-tau*value)/denominator for value in gradient]
    expected=[]
    for point in points:
        total=0.
        for component in dual[point]:total+=component
        divergence=-total
        for axis in range(ndim):
            if point[axis]:
                neighbor=list(point);neighbor[axis]-=1;divergence+=dual[tuple(neighbor)][axis]
        expected.append(lookup[point]+divergence)
    return np.array(expected,np.float64).reshape(shape)


@pytest.mark.parametrize('shape',((2,),(1,2),(1,1,2),(2,3),(2,2,3)))
@pytest.mark.parametrize('weight',(0.5,1.,2.))
def test_independent_two_iteration_dual_stencil(shape,weight):
    values=[float((i*3+i//2)%5-2) for i in range(math.prod(shape))]
    image=np.array(values,np.float64).reshape(shape)
    expected=dual_first_update(values,shape,weight)
    actual=reference(image,weight=weight,eps=0.,max_num_iter=2)
    assert actual.dtype==expected.dtype and actual.shape==shape
    assert actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('ndim',(1,2,3))
@pytest.mark.parametrize('weight',(0.5,1.,2.))
def test_independent_two_pixel_energy_and_return_stage(ndim,weight):
    image=np.array([0.,1.]).reshape((1,)*(ndim-1)+(2,));energy=[]
    class Eps:
        def __mul__(self,initial):energy.append(float(initial));return 0.*initial
    actual=reference(image,weight=weight,eps=Eps(),max_num_iter=2)
    assert energy==[weight/2.]
    tau=1./(2.*ndim);step=tau/(1.+tau/weight)
    expected=np.array([step,1.-step]).reshape(image.shape)
    assert actual.tobytes()==expected.tobytes() and not np.shares_memory(actual,image)
    one=reference(image,weight=weight,max_num_iter=1)
    assert one is image and one.tobytes()==image.tobytes()


@pytest.mark.parametrize('field',('nan-finite','nan-inf','finite','signedzero','metadata','input-nan-payload','warning','event','alias'))
def test_narrow_nan_encoding_contract_does_not_hide_defined_differences(field):
    values=np.array([0x7ff8000000000001,0x3ff0000000000000,0x8000000000000000],np.uint64).view(np.float64)
    raw=dict(result=dict(array=array(values),identity_image=False),before=dict(image=array(values)),warnings=[],events=[])
    changed=copy.deepcopy(raw)
    if field in ('nan-finite','nan-inf','finite','signedzero'):
        bits=np.frombuffer(bytes.fromhex(changed['result']['array']['bytes']),np.uint64).copy()
        if field=='nan-finite':bits[0]=0
        elif field=='nan-inf':bits[0]=0x7ff0000000000000
        elif field=='finite':bits[1]=0x4000000000000000
        else:bits[2]=0
        changed['result']['array']['bytes']=bits.tobytes().hex()
    elif field=='metadata':changed['result']['array']['strides']=[99]
    elif field=='input-nan-payload':changed['before']['image']['bytes']='000000000000f87f'+raw['before']['image']['bytes'][16:]
    elif field=='warning':changed['warnings']=['unexpected']
    elif field=='event':changed['events']=['unexpected']
    else:changed['result']['identity_image']=True
    assert result_nan_encoding_contract(raw)!=result_nan_encoding_contract(changed)


def test_narrow_nan_encoding_contract_ignores_only_result_nan_words():
    a=np.array([0x7ff8000000000001,0xfff8000000000012,0x8000000000000000],np.uint64).view(np.float64)
    b=np.array([0xfff8000000000022,0x7ff8000000000011,0x8000000000000000],np.uint64).view(np.float64)
    raw=dict(result=dict(array=array(a)),before=dict(image=array(a)),warnings=[],events=[])
    changed=copy.deepcopy(raw);changed['result']['array']['bytes']=b.tobytes().hex()
    assert raw!=changed and result_nan_encoding_contract(raw)==result_nan_encoding_contract(changed)


def record(output):
    require=not Path(output).exists();assert require,'never overwrite a raw reference trial'
    raw={};differences={};defined_differences={}
    for spec in CASES:
        first=observe(reference,spec);second=observe(reference,spec);raw[spec['id']]=first
        if first!=second:differences[spec['id']]=dict(first=first,second=second)
        if result_nan_encoding_contract(first)!=result_nan_encoding_contract(second):defined_differences[spec['id']]=dict(first=first,second=second)
    result=dict(scope='Original-only bounded reference observations; native0',comparison='tv-result-nan-encoding-v1',profile=profile(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,signature=str(inspect.signature(reference)),source_sha256=source_sha(reference),helper_sha256=source_sha(original_module._denoise_tv_chambolle_nd),source_utf8=inspect.getsource(reference).replace('\r\n','\n'),helper_utf8=inspect.getsource(original_module._denoise_tv_chambolle_nd).replace('\r\n','\n'),test_source_canonical_lf_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),case_contract_sha256=digest(CASES),cases=list(CASES),observations=raw,replay_differences=differences,defined_replay_differences=defined_differences,native_checks=0)
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    assert not defined_differences,'defined repeated-call differences retained: '+','.join(defined_differences)
    print(json.dumps(dict(profile=profile(),cases=len(CASES),native_checks=0,raw=output)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--record',required=True);record(parser.parse_args().record)
