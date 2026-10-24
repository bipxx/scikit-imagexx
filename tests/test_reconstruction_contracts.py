"""Source-safe0.26 reconstruction reference; native admission remains separate."""
import hashlib,importlib,inspect,itertools,json,os,struct,warnings
from pathlib import Path
import numpy as np
import pytest
import skimage
ROOT=Path(__file__).resolve().parents[1]
MATRIX=ROOT/'coverage/contracts/reconstruction.json'
SOURCE=importlib.import_module('skimage.morphology.grayreconstruct')
ORIGINAL=SOURCE.reconstruction
DTYPES=('bool','u1','i1','u2','i2','u4','i4','u8','i8','f2','f4','f8')
def profile():return 'windows' if os.name=='nt' else 'minimum' if np.__version__=='1.24.4' else 'modern'
def value(a):
    if isinstance(a,np.ndarray):
        b=a.base
        return dict(type=type(a).__name__,dtype=a.dtype.str,metadata=dict(a.dtype.metadata or {}),shape=list(a.shape),strides=list(a.strides),raw=a.tobytes().hex(),own=bool(a.flags.owndata),writeable=bool(a.flags.writeable),C=bool(a.flags.c_contiguous),F=bool(a.flags.f_contiguous),base=None if b is None else dict(type=type(b).__name__,shape=list(b.shape),strides=list(b.strides),dtype=b.dtype.str,raw=b.tobytes().hex(),offset=int(a.ctypes.data-b.ctypes.data)))
    if isinstance(a,np.generic):return dict(type=type(a).__name__,dtype=a.dtype.str,raw=a.tobytes().hex())
    if type(a) in (str,int,bool) or a is None:return a
    if type(a) is float:return dict(type='float',raw=struct.pack('<d',a).hex())
    if isinstance(a,(list,tuple)):return dict(type=type(a).__name__,items=[value(x) for x in a])
    if isinstance(a,dict):return {k:value(v) for k,v in a.items()}
    return dict(type=type(a).__name__)
class Tagged(np.ndarray):pass
class SourceUnsafeBoundary(RuntimeError):pass
def cases():
    rows=[]
    def add(id,**kw):rows.append(dict(id=id,**kw))
    for d,l,m in itertools.product(DTYPES,('C','F','reverse','step','endian','readonly'),('dilation','erosion')):add('dtype/'+d+'/'+l+'/'+m,dtype=d,layout=l,method=m)
    for d,l,m in itertools.product(('u2','f4','f8'),('unaligned','tagged'),('dilation','erosion')):add('storage/'+d+'/'+l+'/'+m,dtype=d,layout=l,method=m)
    for r,d,m in itertools.product((1,2,3,4),('u1','f4','f8'),('dilation','erosion')):add('nd/'+str(r)+'/'+d+'/'+m,rank=r,dtype=d,method=m)
    for f,d,m in itertools.product(('cross','zero','center','directed','wide','float','cross3','zero3','even-left','even-right','wide1','isolated'),('u1','f4'),('dilation','erosion')):add('footprint/'+f+'/'+d+'/'+m,footprint=f,dtype=d,method=m,rank=3 if f.endswith('3') else 1 if f in ('even-left','even-right','wide1') else 2)
    for pair,m in itertools.product(('u1:f4','f8:u1','i2:f2','u8:f8','f8:f4','f2:i4'),('dilation','erosion')):
        a,b=pair.split(':');add('mixed/'+pair+'/'+m,dtype=a,mask_dtype=b,method=m)
    for d,m in itertools.product(('f4','f8'),('dilation','erosion')):add('zero/'+d+'/'+m,kind='signed-zero',dtype=d,method=m)
    for d,m in itertools.product(('u8','i8','f4','f8'),('dilation','erosion')):add('extreme/'+d+'/'+m,kind='extreme',dtype=d,method=m)
    for name in ('shape','order-dilation','order-erosion','method','method-order','even','offset-ndim','offset-range','offset-list','empty','scalar','list','complex','fp-raise'):add('preentry/'+name,preentry=name,dtype='f4')
    for name in ('nan','wrong-footprint-rank','custom-bad-footprint'):add('blocked/'+name,blocked=name,dtype='f4')
    for m,p in itertools.product(('dilation','erosion'),('fp-record','fp-mutate','offset-record')):add('provider/'+p+'/'+m,provider=p,dtype='f4',method=m,rank=1 if p=='offset-record' else 2)
    assert len(rows)==275 and len({s['id'] for s in rows})==275
    return rows
def layout(a,name):
    if name=='F':return np.asfortranarray(a)
    if name=='reverse':return a[(slice(None,None,-1),)*a.ndim]
    if name=='step':
        owner=np.zeros(tuple(n*2 for n in a.shape),a.dtype);out=owner[(slice(None,None,2),)*a.ndim];out[...]=a;return out
    if name=='endian':return a.astype(a.dtype.newbyteorder('>'))
    if name=='readonly':a.flags.writeable=False
    if name=='tagged':return a.astype(np.dtype(a.dtype,metadata={'reconstruction':1})).view(Tagged)
    if name=='unaligned':
        owner=np.zeros(a.nbytes+1,'u1');out=np.ndarray(a.shape,dtype=a.dtype,buffer=owner,offset=1);out[...]=a;return out
    return a
def fixture(s,events):
    rank=s.get('rank',2);shape={1:(7,),2:(4,5),3:(3,4,5),4:(2,3,2,3)}[rank];n=int(np.prod(shape));d=np.dtype(s['dtype']);base=(np.arange(n)%13).reshape(shape)
    if d.kind=='b':low=(base%3==0);high=np.ones(shape,bool)
    else:low=base.astype(d);high=(base+3).astype(s.get('mask_dtype',d))
    if s.get('kind')=='signed-zero':low=np.zeros(shape,d);low.flat[::2]=-0.;high=low.copy();high.flat[::2]=0.;high.flat[1::2]=-0.
    if s.get('kind')=='extreme':
        if d.kind=='u':low=(np.arange(n,dtype=np.uint64)%4)*4096+np.uint64(2**63);high=low+np.uint64(4096)
        elif d.kind=='i':low=np.arange(n,dtype=np.int64)%4*4096+np.int64(-2**63);high=low+4096
        else:low=(base-6).astype(d)*d.type(1e20);high=low+d.type(1e20)
        low=low.reshape(shape);high=high.reshape(shape)
    method=s.get('method','dilation');seed,mask=(high,low) if method=='erosion' else (low,high)
    seed=layout(np.asarray(seed),s.get('layout','C'));mask=layout(np.asarray(mask),s.get('layout','C'));kw=dict(method=method);f=s.get('footprint')
    if f:
        if f in ('even-left','even-right'):fp=np.ones(2,bool);kw['offset']=np.array([0 if f=='even-left' else 1])
        elif f=='wide1':fp=np.ones(5,bool)
        elif f=='center':fp=np.ones((1,1),bool)
        elif f=='wide':fp=np.ones((5,5),bool)
        elif f=='cross':fp=np.eye(3,dtype=bool)|np.eye(3,dtype=bool)[::-1]
        elif f=='cross3':fp=np.zeros((3,3,3),bool);fp[:,1,1]=True;fp[1,:,1]=True;fp[1,1,:]=True
        elif f in ('zero','zero3'):fp=np.zeros((3,)*rank,bool)
        elif f=='directed':fp=np.array([[0,0,0],[0,1,1],[0,1,0]],bool)
        elif f=='isolated':fp=np.array([[1,0,0],[0,0,0],[0,0,1]],bool)
        else:fp=np.ones((3,3),'f4');fp[0,0]=np.nan
        kw['footprint']=fp
    p=s.get('preentry')
    if p=='shape':mask=mask[:2]
    if p in ('order-dilation','method-order'):seed=mask+1
    if p=='order-erosion':kw['method']='erosion'
    if p in ('method','method-order'):kw['method']='other'
    if p=='even':kw['footprint']=np.ones((2,2),bool)
    if p=='offset-ndim':kw['offset']=np.array([1,1])
    if p=='offset-range':seed=seed[0];mask=mask[0];kw['footprint']=np.ones(3,bool);kw['offset']=np.array([3])
    if p=='offset-list':kw['offset']=[1,1]
    if p=='empty':seed=np.empty((0,3),d);mask=seed.copy()
    if p=='scalar':seed=np.array(1,d);mask=np.array(2,d)
    if p=='list':seed=seed.tolist();mask=mask.tolist()
    if p=='complex':seed=seed.astype('c8');mask=mask.astype('c8')
    class Footprint:
        def astype(self,dtype,copy):
            events.append(['footprint_astype',str(np.dtype(dtype)),copy])
            if p=='fp-raise':raise LookupError('footprint-stop')
            if s.get('provider')=='fp-mutate':seed.flat[0]=seed.flat[1];events.append(['provider_seed_after',value(seed)])
            return np.ones((3,)*seed.ndim,bool)
    if p=='fp-raise' or s.get('provider') in ('fp-record','fp-mutate'):kw['footprint']=Footprint()
    class Offset:
        @property
        def ndim(self):events.append(['offset_ndim']);return 1
        def __iter__(self):events.append(['offset_iter']);return iter([1])
    if s.get('provider')=='offset-record':kw['offset']=Offset()
    return seed,mask,kw

def observe(s,function=None):
    events=[];seed,mask,kw=fixture(s,events);before=[value(seed),value(mask),value(kw)];calls=[];entry=SOURCE.reconstruction_loop
    def wrapped(ranks,prev,next,strides,start,image_stride):
        row=dict(ranks_before=value(ranks),prev=value(prev),next=value(next),strides=value(strides),start=value(start),image_stride=value(image_stride),completed=False);calls.append(row)
        if s.get('preentry'):raise AssertionError('preentry witness unexpectedly reached unchecked Cython')
        assert ranks.ndim==prev.ndim==next.ndim==strides.ndim==1 and ranks.size==2*image_stride and prev.size==next.size==ranks.size
        assert 0<=start<ranks.size and ranks.size<100000 and ranks.dtype.kind=='u' and prev.dtype==next.dtype==strides.dtype
        fp=kw.get('footprint');fshape=(3,)*seed.ndim if fp is None or not isinstance(fp,np.ndarray) else fp.shape
        offset=[0 if s.get('footprint')=='even-left' else 1] if seed.ndim==1 and 'offset' in kw else [x//2 for x in fshape]
        padded=tuple(a+b-1 for a,b in zip(seed.shape,fshape));assert int(np.prod(padded))==image_stride
        inside=tuple(slice(o,o+n) for o,n in zip(offset,seed.shape));outside=np.ones(padded,bool);outside[inside]=False
        a=ranks[:image_stride].reshape(padded);b=ranks[image_stride:].reshape(padded)
        assert np.all(a[outside]==0) and np.all(b[outside]==0) and np.all(a[inside]<=b[inside])
        result=entry(ranks,prev,next,strides,start,image_stride);row['completed']=True;row['ranks_after']=value(ranks);return result
    error=None;result=None;SOURCE.reconstruction_loop=wrapped
    try:
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='warn'):
            warnings.simplefilter('always')
            try:
                if s.get('blocked'):raise SourceUnsafeBoundary('unchecked source intake deliberately not executed')
                result=(function or ORIGINAL)(seed,mask,**kw)
            except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
        warning_rows=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]
    finally:SOURCE.reconstruction_loop=entry
    return dict(before=before,after=[value(seed),value(mask),value(kw)],result=value(result),error=error,warnings=warning_rows,events=events,original_Cython_attempts=len(calls),original_Cython_completed=sum(c['completed'] for c in calls),source_backend=calls,aliases=dict(seed=isinstance(result,np.ndarray) and isinstance(seed,np.ndarray) and np.shares_memory(result,seed),mask=isinstance(result,np.ndarray) and isinstance(mask,np.ndarray) and np.shares_memory(result,mask)))
def public_observation(row):return {k:v for k,v in row.items() if k not in ('source_backend','original_Cython_attempts','original_Cython_completed')}
@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_original_contract(spec):
    data=json.loads(MATRIX.read_bytes());assert observe(spec)==data['profiles'][profile()]['observations'][spec['id']]
def test_source_signature_and_fixed_scope():
    d=json.loads(MATRIX.read_bytes());assert d['cases']==cases() and d['new_native_paths']==0 and skimage.__version__=='0.26.0'
    assert str(inspect.signature(ORIGINAL))=="(seed, mask, method='dilation', footprint=None, offset=None)"
    assert d['test_canonical_lf_sha256']==hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def independent_reconstruction(seed,mask,method,footprint=None):
    dtype=np.float32 if mask.dtype.type in (np.float16,np.float32) else np.float64
    out=seed.astype(dtype,copy=True);limit=mask.astype(dtype,copy=True)
    fp=np.ones((3,)*out.ndim,bool) if footprint is None else footprint.astype(bool,copy=True)
    center=tuple(d//2 for d in fp.shape);fp[center]=False
    offsets=[tuple(i-c for i,c in zip(index,center)) for index in np.ndindex(fp.shape) if fp[index]]
    for _ in range(out.size*32+1):
        changed=False
        for index in np.ndindex(out.shape):
            for delta in offsets:
                neighbor=tuple(i+d for i,d in zip(index,delta))
                if any(j<0 or j>=n for j,n in zip(neighbor,out.shape)):continue
                candidate=min(out[index],limit[neighbor]) if method=='dilation' else max(out[index],limit[neighbor])
                if candidate>out[neighbor] if method=='dilation' else candidate<out[neighbor]:out[neighbor]=candidate;changed=True
        if not changed:return out
    raise AssertionError('bounded monotone reconstruction oracle did not reach fixed point')

@pytest.mark.parametrize('dtype',('u1','f4','f8'))
@pytest.mark.parametrize('method',('dilation','erosion'))
@pytest.mark.parametrize('footprint',(None,'directed','cross','zero'))
def test_independent_monotone_fixed_point(dtype,method,footprint):
    spec=dict(dtype=dtype,method=method)
    if footprint:spec['footprint']=footprint
    seed,mask,options=fixture(spec,[]);expected=independent_reconstruction(seed,mask,method,options.get('footprint'))
    row=observe(spec);assert row['error'] is None
    assert row['result']['dtype']==expected.dtype.str and row['result']['raw']==expected.tobytes().hex()

@pytest.mark.parametrize('dtype',('f4','f8'))
@pytest.mark.parametrize('method',('dilation','erosion'))
def test_signed_zero_repeat_words_are_stable(dtype,method):
    spec=dict(dtype=dtype,method=method,kind='signed-zero')
    assert observe(spec)==observe(spec)

@pytest.mark.parametrize('workers',(1,4))
def test_native_pending_is_not_upstream_forwarding(workers):
    import sys
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert 'morphology.reconstruction' not in native_functions()
    before=xx.get_num_threads();previous=xx.set_fallback_enabled(False);old_profile=sys.getprofile();attempts=[]
    class Unsupported:
        @property
        def shape(self):raise AssertionError('unsupported seed inspected before explicit rejection')
    def guard(frame,event,arg):
        if event=='call' and frame.f_code is ORIGINAL.__code__:attempts.append(True);raise AssertionError('upstream reconstruction forwarding forbidden')
    try:
        xx.set_num_threads(workers);sys.setprofile(guard)
        with pytest.raises(NotImplementedError):xx.morphology.reconstruction(Unsupported(),Unsupported())
        assert not attempts
    finally:sys.setprofile(old_profile);xx.set_num_threads(before);xx.set_fallback_enabled(previous)

