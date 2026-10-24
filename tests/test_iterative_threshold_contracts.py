"""Pinned0.26 ORIGINAL-only threshold contracts. Native coverage remains zero.

Unchecked multi-Otsu domains are intercepted BEFORE the original Cython body.
Those rows are intake witnesses, never numerical/error claims for that body.
"""
import ast
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import struct
import sys
import warnings

import numpy as np
import pytest
import skimage

ROOT=Path(__file__).resolve().parents[1]
MATRIX=ROOT/'coverage/contracts/iterative-thresholds.json'
NAMES=('threshold_li','threshold_multiotsu')
DTYPES=('bool','u1','i1','u2','i2','u4','i4','u8','i8','f2','f4','f8')
LAYOUTS=('C','F','step','reverse','endian','readonly')
SOURCE=importlib.import_module('skimage.filters.thresholding')
ORIGINAL={n:getattr(SOURCE,n) for n in NAMES}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def profile():
    return 'windows' if os.name=='nt' else 'minimum' if np.__version__=='1.24.4' else 'modern'


def value(a):
    if isinstance(a,np.ndarray):
        return dict(type=type(a).__name__,dtype=a.dtype.str,metadata=dict(a.dtype.metadata or {}),shape=list(a.shape),strides=list(a.strides),
            raw=a.tobytes().hex(),own=bool(a.flags.owndata),writeable=bool(a.flags.writeable),C=bool(a.flags.c_contiguous),F=bool(a.flags.f_contiguous),base_type=None if a.base is None else type(a.base).__name__)
    if isinstance(a,np.generic):return dict(type=type(a).__name__,dtype=a.dtype.str,raw=a.tobytes().hex())
    if type(a) is float:return dict(type='float',raw=struct.pack('<d',a).hex())
    if type(a) in (str,int,bool) or a is None:return a
    if isinstance(a,(list,tuple)):return dict(type=type(a).__name__,items=[value(v) for v in a])
    if isinstance(a,dict):return {k:value(v) for k,v in a.items()}
    return dict(type=type(a).__name__)


class Tagged(np.ndarray):
    pass


class SafetyBoundary(RuntimeError):
    pass


def cases():
    rows=[]
    def add(name,identifier,**kwargs):rows.append(dict(id=name+'/'+identifier,function=name,**kwargs))
    for name in NAMES:
        for dtype,layout in itertools.product(DTYPES,LAYOUTS):add(name,'default-'+dtype+'-'+layout,dtype=dtype,layout=layout)
    for dtype in ('i2','f4','f8'):
        for initial in ('scalar','mean','quantile','mutate','raise','invalid'):
            add('threshold_li','initial-'+dtype+'-'+initial,dtype=dtype,initial=initial,callback='record')
        for tolerance in ('zero','explicit','probe','nan','array-one','array-two','negative-abort'):
            add('threshold_li','tolerance-'+dtype+'-'+tolerance,dtype=dtype,tolerance=tolerance,callback='abort2' if tolerance=='negative-abort' else 'record')
        for callback in ('abort1','abort2','mutate','invalid'):
            add('threshold_li','callback-'+dtype+'-'+callback,dtype=dtype,initial='mean',callback=callback)
    for kind in ('empty','scalar','constant','all-nan','mixed-nan','inf','both-inf','finite-inf','signed-zero','full-u8','3D','list','none'):
        add('threshold_li','boundary-'+kind,kind=kind,dtype='u1' if kind=='full-u8' else 'f8')
    for name in NAMES:
        for dtype in ('u1','f4','f8'):
            for classes in (2,3,4):
                if name=='threshold_multiotsu':add(name,'full256-'+dtype+'-c'+str(classes),kind='full-u8' if dtype=='u1' else 'full-float',dtype=dtype,classes=classes)
    for dtype,layout in itertools.product(('u1','u2','u8','f4','f8'),('C','step','reverse','endian','readonly','unaligned')):
        add('threshold_multiotsu','hist-'+dtype+'-'+layout,hist='counts',dtype=dtype,layout=layout,classes=3)
    for dtype in DTYPES[1:]:
        add('threshold_multiotsu','centers-'+dtype,hist='tuple',dtype='f4',centers_dtype=dtype,classes=3)
    for hist in ('trim','sparse','zero','empty','short-centers','wrong-pair','negative','nan','infinite'):
        add('threshold_multiotsu','hist-boundary-'+hist,hist=hist,dtype='f4',classes=3,blocked=hist in ('negative','nan','infinite'))
    for kind in ('empty','scalar','constant','all-nan','mixed-nan','inf','both-inf','finite-inf','3D','list','none'):
        # Empty floating histograms normalize to NaNs and reach unchecked Cython.
        add('threshold_multiotsu','image-boundary-'+kind,kind=kind,dtype='f8',blocked=kind=='empty')
    for nbins in (0,-1,1,2,8,32):add('threshold_multiotsu','nbins-'+str(nbins),dtype='f4',nbins=nbins)
    for classes in (1,0,-1):add('threshold_multiotsu','blocked-classes-'+str(classes),hist='counts',dtype='f4',classes=classes,blocked=True)
    add('threshold_multiotsu','constant-class1',kind='constant',dtype='u1',classes=1)
    add('threshold_multiotsu','zero-hist-class0',hist='zero',dtype='f4',classes=0)
    add('threshold_multiotsu','none-no-hist',kind='none',dtype='u1')
    for mode in ('index','int-subclass','rgb-spy','rgb-warning-error'):
        add('threshold_multiotsu','provider-'+mode,hist='tuple',dtype='f4',protocol=mode)
    for classes in (2,3,4):
        for fallback in (False,True):add('threshold_multiotsu','solver-c'+str(classes)+'-'+('fallback' if fallback else 'lut'),hist='counts',dtype='f4',classes=classes,fallback=fallback)
    assert len({r['id'] for r in rows})==len(rows)
    return rows


def layout(a,mode):
    if mode=='F':return np.asfortranarray(a)
    if mode=='reverse':return a[(slice(None,None,-1),)*a.ndim]
    if mode=='step':
        owner=np.zeros(tuple(2*n for n in a.shape),a.dtype);out=owner[(slice(None,None,2),)*a.ndim];out[...] =a;return out
    if mode=='endian':return a.astype(a.dtype.newbyteorder('>'))
    if mode=='readonly':a.flags.writeable=False
    if mode=='unaligned':
        owner=np.zeros(a.nbytes+1,'u1');out=np.ndarray(a.shape,dtype=a.dtype,buffer=owner,offset=1);out[...] =a;return out
    return a


def fixture(spec,events):
    dtype=np.dtype(spec.get('dtype','u1'));kind=spec.get('kind','ordinary');data=[0,0,1,2,3,4,4,5,7,9,11,12]
    if kind=='empty':image=np.empty((0,3),dtype)
    elif kind=='scalar':image=np.array(3,dtype)
    elif kind=='constant':image=np.full((4,4),3,dtype)
    elif kind=='all-nan':image=np.full((3,4),np.nan,dtype)
    elif kind=='mixed-nan':image=np.array([0,1,np.nan,2,5,8],dtype)
    elif kind=='inf':image=np.array([np.inf,np.inf],dtype)
    elif kind=='both-inf':image=np.array([-np.inf,np.inf],dtype)
    elif kind=='finite-inf':image=np.array([-np.inf,0,1,2,5,8,np.inf],dtype)
    elif kind=='signed-zero':image=np.array([-0.,0.,-0.,0.],dtype)
    elif kind in ('full-u8','full-float'):image=np.arange(256,dtype=dtype).reshape(16,16)
    elif kind=='3D':image=np.array(data*2,dtype).reshape(2,4,3)
    elif kind=='list':image=data
    elif kind=='none':image=None
    else:image=np.array(data,dtype).reshape(3,4)
    if isinstance(image,np.ndarray):image=layout(image,spec.get('layout','C'))
    kwargs={};hist=spec.get('hist')
    if hist:
        if hist=='trim':counts=np.array([0,0,3,1,4,2,0],dtype)
        elif hist=='sparse':counts=np.array([0,3,0,2,0,4,0],dtype)
        elif hist=='zero':counts=np.zeros(6,dtype)
        elif hist=='empty':counts=np.empty(0,dtype)
        elif hist=='negative':counts=np.array([2,-1,3,2,1,4],dtype)
        elif hist=='nan':counts=np.array([2,np.nan,3,2,1,4],dtype)
        elif hist=='infinite':counts=np.array([2,np.inf,3,2,1,4],dtype)
        else:counts=np.array([4,1,3,2,4,1],dtype)
        counts=layout(counts,spec.get('layout','C'));centers=np.arange(counts.size,dtype=spec.get('centers_dtype','f8'))
        if hist=='short-centers':centers=centers[:1]
        kwargs['hist']=(counts,centers) if hist in ('tuple','trim','short-centers') else (counts,) if hist=='wrong-pair' else counts
    for key in ('classes','nbins'):
        if key in spec:kwargs[key]=spec[key]
    work={}
    class Guess:
        def __call__(self,a):
            events.append(['initial_guess',value(a),bool(np.shares_memory(a,image))]);work['image']=a
            mode=spec['initial']
            if mode=='raise':raise LookupError('initial-guess-stop')
            if mode=='mutate':a.flat[0]=a.max();events.append(['initial_mutation',value(a)])
            return np.quantile(a,.75) if mode=='quantile' else np.mean(a)
    initial=spec.get('initial')
    if initial:kwargs['initial_guess']=1.25 if initial=='scalar' else [1] if initial=='invalid' else Guess()
    if spec.get('callback'):
        callback=spec['callback'];calls=[0]
        def cb(v):
            calls[0]+=1;events.append(['iter_callback',value(v)])
            if callback=='mutate' and 'image' in work:work['image'].flat[0]=0;events.append(['callback_mutation',value(work['image'])])
            if callback=='abort1' or callback=='abort2' and calls[0]==2 or calls[0]>=32:raise RuntimeError('callback-stop')
        kwargs['iter_callback']=7 if callback=='invalid' else cb
    class Tolerance(float):
        def __bool__(self):events.append(['tolerance_bool']);return True
        def __rmul__(self,other):events.append(['tolerance_rmul',other]);return float(self)*other
    tol=spec.get('tolerance')
    if tol:kwargs['tolerance']={'zero':0.,'explicit':.125,'probe':Tolerance(.5),'nan':np.nan,'array-one':np.array([.5]),'array-two':np.array([.5,.5]),'negative-abort':-.5}[tol]
    class Index:
        def __init__(self,n):self.n=n
        def __index__(self):events.append(['threshold_count_index',self.n]);return self.n
    class Classes:
        def __gt__(self,x):events.append(['classes_gt',int(x)]);return 3>x
        def __eq__(self,x):events.append(['classes_eq',int(x)]);return 3==x
        def __sub__(self,x):events.append(['classes_sub',x]);return Index(3-x)
    class IntClasses(int):
        def __sub__(self,x):events.append(['int_classes_sub',x]);return super().__sub__(x)
    class ImageSpy:
        @property
        def ndim(self):events.append(['image_ndim']);return 3
        @property
        def shape(self):events.append(['image_shape']);return (4,4,3)
    protocol=spec.get('protocol')
    if protocol=='index':kwargs['classes']=Classes()
    if protocol=='int-subclass':kwargs['classes']=IntClasses(3)
    if protocol in ('rgb-spy','rgb-warning-error'):image=ImageSpy()
    return image,kwargs


def observe(spec,functions=None):
    """Full original observation, with separately marked internal path witnesses."""
    events=[];backend=[];image,kwargs=fixture(spec,events);before=value(image);options_before=value(kwargs)
    histogram=SOURCE.histogram;entries={n:getattr(SOURCE,n) for n in ('_get_multiotsu_thresh_indices_lut','_get_multiotsu_thresh_indices')}
    def hist(*args,**options):
        row=dict(stage='histogram',input=value(args),options=value(options));backend.append(row)
        result=histogram(*args,**options);row['result']=value(result);return result
    def wrap(name):
        def call(prob,count):
            row=dict(stage=name,prob=value(prob),count=value(count),completed=False,actual_Cython=False);backend.append(row)
            if spec.get('blocked'):row['source_unsafe_blocked']=True;raise SafetyBoundary('source-unsafe Cython intake blocked')
            k=count.n if type(count).__name__=='Index' else int(count)
            a=np.asarray(prob)
            assert a.ndim==1 and 1<=k<=3 and 3<=a.size<=256
            assert np.isfinite(a).all() and (a>=0).all() and 0<float(a.sum())<=100000
            assert np.count_nonzero(a)>k+1 and math.comb(a.size-1,k)<=4000000
            if spec.get('fallback') and name.endswith('_lut'):row['forced_allocation_failure']=True;raise MemoryError('controlled-LUT-allocation-failure')
            row['actual_Cython']=True
            result=entries[name](prob,count);row['completed']=True;row['result']=value(result);return result
        return call
    SOURCE.histogram=hist
    for name in entries:setattr(SOURCE,name,wrap(name))
    error=None;result=None
    try:
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='warn'):
            warnings.simplefilter('error' if spec.get('protocol')=='rgb-warning-error' else 'always')
            try:result=(functions or ORIGINAL)[spec['function']](image,**kwargs)
            except Exception as e:error=dict(type=type(e).__name__,message=str(e))
        warning_rows=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]
    finally:
        SOURCE.histogram=histogram
        for name,entry in entries.items():setattr(SOURCE,name,entry)
    return dict(input_before=before,input_after=value(image),options_before=options_before,options_after=value(kwargs),result=value(result),error=error,warnings=warning_rows,events=events,source_backend=backend,
        source_unsafe=bool(spec.get('blocked')),numeric_Cython_completed=sum(bool(r.get('completed')) for r in backend),input_result_alias=bool(isinstance(result,np.ndarray) and isinstance(image,np.ndarray) and np.shares_memory(result,image)))


def public_observation(observation):
    # Internal source histogram/Cython calls are handoff witnesses, not fake CPP calls.
    return {k:v for k,v in observation.items() if k not in ('source_backend','numeric_Cython_completed')}


def multiotsu_definition(counts,classes,lut=True):
    """Independent exhaustive partitions with the pinned F32/source quirks."""
    p=np.array(counts,'f4');n=len(p);omega=np.empty(n,'f4');moment=np.empty(n,'f4');omega[0]=p[0];moment[0]=p[0]
    for i in range(1,n):omega[i]=omega[i-1]+p[i];moment[i]=moment[i-1]+i*p[i]
    def score(i,j):
        if lut and i==j==0:return np.float32(0)
        a=omega[j] if i==0 else np.float32(omega[j]-omega[i-1]);b=moment[j] if i==0 else np.float32(moment[j]-moment[i-1])
        return np.float32(float(b)**2/float(a)) if a>0 else np.float32(0)
    best=np.float32(0);winner=None
    for thresholds in itertools.combinations(range(n-1),classes-1):
        sigma=np.float32(score(0,thresholds[0])+score(thresholds[-1]+1,n-1))
        for i in range(classes-2):sigma=np.float32(sigma+score(thresholds[i]+1,thresholds[i+1]))
        if sigma>best:best=sigma;winner=thresholds
    assert winner is not None
    return np.array(winner,dtype=np.intp)


def li_definition(data):
    """Independent small-integer tally/partition equation; no upstream helper."""
    offset=min(data);a=[int(v)-offset for v in data];t=np.float64(sum(a)/len(a));old=np.float64(-1)
    for _ in range(64):
        if not abs(t-old)>.5:return np.float64(t+offset)
        old=t;f=[v for v in a if v>t];b=[v for v in a if v<=t]
        mf=np.float64(sum(f)/len(f));mb=np.float64(sum(b)/len(b))
        if mb==0:return np.float64(t+offset)
        t=(mb-mf)/(np.log(mb)-np.log(mf))
    raise AssertionError('bounded independent Li witness did not converge')


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_original_fixed_contract(spec):
    data=json.loads(MATRIX.read_bytes());expected=data['profiles'][profile()]['observations'][spec['id']]
    actual=observe(spec);assert actual==expected
    assert actual['input_before']==actual['input_after']


def test_source_signatures_integrity_and_native0():
    data=json.loads(MATRIX.read_bytes());assert skimage.__version__=='0.26.0'
    assert data['native_implemented']==0 and data['status']=='original-reference/native-pending'
    assert data['cases']==cases() and len({s['id'] for s in cases()})==len(cases())
    assert data['test_source_canonical_lf_sha256']==hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    for name in NAMES:
        assert str(inspect.signature(ORIGINAL[name]))==data['signatures'][name]
        assert getattr(importlib.import_module('skimage.filters'),name) is ORIGINAL[name]
        node=ast.parse(inspect.getsource(ORIGINAL[name])).body[0]
        assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==data['source_function_AST'][str(sys.version_info.minor)][name]


@pytest.mark.parametrize('spec',[s for s in cases() if s.get('blocked')],ids=lambda s:s['id'])
def test_source_unsafe_intakes_never_enter_compiled_algorithm(spec,monkeypatch):
    attempted=[]
    def forbidden(*args):attempted.append(args);raise AssertionError('unsafe compiled original executed')
    for name in ('_get_multiotsu_thresh_indices_lut','_get_multiotsu_thresh_indices'):
        monkeypatch.setattr(SOURCE,name,forbidden)
    row=observe(spec)
    assert not attempted and row['source_unsafe'] and row['numeric_Cython_completed']==0
    assert row['error']==dict(type='SafetyBoundary',message='source-unsafe Cython intake blocked')


@pytest.mark.parametrize('classes',(2,3,4))
@pytest.mark.parametrize('fallback',(False,True))
def test_independent_exhaustive_multiotsu_definition(classes,fallback):
    spec=dict(id='definition',function='threshold_multiotsu',hist='counts',dtype='f4',classes=classes,fallback=fallback)
    observed=observe(spec);assert observed['error'] is None
    expected=multiotsu_definition([4,1,3,2,4,1],classes,not fallback)
    assert observed['result']==value(expected)


@pytest.mark.parametrize('data',([0,0,1,2,4,4,5,9],[1,1,2,3,5,5,6,10],[0,1,1,2,3,7,8,9]))
def test_independent_small_integer_li_partition(data):
    actual=ORIGINAL['threshold_li'](np.array(data,'u1'))
    assert value(actual)==value(li_definition(data))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('workers',(1,4))
def test_native_pending_rejects_forwarding(name,workers):
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert 'filters.'+name not in native_functions()
    previous=xx.set_fallback_enabled(False);old=xx.get_num_threads();xx.set_num_threads(workers)
    code=ORIGINAL[name].__code__;profile_before=sys.getprofile();attempts=[]
    def reject(frame,event,arg):
        if event=='call' and frame.f_code is code:attempts.append(name);raise AssertionError('upstream forwarding is not native')
    sys.setprofile(reject)
    try:
        with pytest.raises(NotImplementedError):getattr(xx.filters,name)(np.arange(16,dtype='u1').reshape(4,4))
        assert not attempts
    finally:sys.setprofile(profile_before);xx.set_num_threads(old);xx.set_fallback_enabled(previous)
