"""Pinned h-extrema contracts; unchecked reconstruction intakes certified first."""
import ast,hashlib,importlib,importlib.util,inspect,itertools,json,sys,warnings
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];MATRIX=ROOT/'coverage/contracts/h-extrema.json'
sp=importlib.util.spec_from_file_location('h_frozen_reconstruction_reference',ROOT/'tests/test_reconstruction_contracts.py');R=importlib.util.module_from_spec(sp);sp.loader.exec_module(R)
SOURCE=importlib.import_module('skimage.morphology.extrema');ORIGINAL={n:getattr(SOURCE,n) for n in ('h_maxima','h_minima')};ORIGINAL_REC=SOURCE.grayreconstruct.reconstruction
value=R.value
class SourceUnsafeBoundary(RuntimeError):pass

def cases():
    rows=[]
    def add(function,id,**kw):rows.append(dict(function=function,id=function+'/'+id,**kw))
    for function in ORIGINAL:
        for dtype,layout in itertools.product(R.DTYPES,('C','F','reverse','step','endian','readonly')):add(function,'dtype/'+dtype+'/'+layout,dtype=dtype,layout=layout)
        for dtype,layout in itertools.product(('u2','f4','f8'),('unaligned','tagged')):add(function,'storage/'+dtype+'/'+layout,dtype=dtype,layout=layout)
        for dtype,height in itertools.product(('u1','i1','u2','i2','u8','i8','f2','f4','f8'),('zero','negative','two','out-of-range','fractional','float-integral','float-fractional','u64','array-one','array-two')):add(function,'height/'+dtype+'/'+height,dtype=dtype,height=height)
        for rank,dtype in itertools.product((1,2,3,4),('u1','f4','f8')):add(function,'rank/'+str(rank)+'/'+dtype,rank=rank,dtype=dtype)
        for footprint,dtype in itertools.product(('cross','directed','zero','center','even','wrongrank','raise','record','mutate'),('u1','f4','f8')):add(function,'footprint/'+footprint+'/'+dtype,footprint=footprint,dtype=dtype)
        for kind,dtype in itertools.product(('plateau','border','negative','extreme','resolution'),('u1','f4','f8')):add(function,'pattern/'+kind+'/'+dtype,kind=kind,dtype=dtype)
        for kind in ('empty','scalar','list','nan','inf','both-inf','complex'):add(function,'boundary/'+kind,kind=kind,dtype='f4')
        for height in ('nan','infinity','negative-infinity','probe','index-only'):add(function,'provider/'+height,dtype='f4',height=height)
    assert len(rows)==468 and len({s['id'] for s in rows})==468
    return rows

def fixture(s,events):
    rank=s.get('rank',2);shape={1:(9,),2:(5,7),3:(3,4,5),4:(2,3,2,3)}[rank];dtype=np.dtype(s['dtype']);n=int(np.prod(shape));data=((np.arange(n)*7)%13).reshape(shape)
    image=data.astype(dtype);kind=s.get('kind')
    if kind=='plateau':image=np.full(shape,4,dtype);image.flat[3:7]=8
    if kind=='border':image=np.zeros(shape,dtype);image.flat[0]=image.flat[-1]=8;image.flat[n//2]=4
    if kind=='negative' and dtype.kind=='f':image=(data.astype(dtype)-dtype.type(6))
    if kind=='extreme':
        if dtype.kind=='u':image=(data%3*100).astype(dtype)
        else:image=(data.astype(dtype)-dtype.type(6))*dtype.type(1e20)
    if kind=='resolution' and dtype.kind=='f':image=(np.ones(shape,dtype)*dtype.type(1000))+data.astype(dtype)*dtype.type(np.finfo(dtype).resolution*1000)
    if kind=='empty':image=np.empty((0,3),dtype)
    if kind=='scalar':image=np.array(3,dtype)
    if kind=='list':image=image.tolist()
    if kind=='nan':image=image.copy();image.flat[0]=np.nan
    if kind=='inf':image=image.copy();image.flat[0]=np.inf
    if kind=='both-inf':image=image.copy();image.flat[0]=-np.inf;image.flat[-1]=np.inf
    if kind=='complex':image=image.astype('c8')
    if isinstance(image,np.ndarray):image=R.layout(image,s.get('layout','C'))
    class Height(float):
        def __gt__(self,x):events.append(['height_gt',value(x)]);return super().__gt__(x)
        def __eq__(self,x):events.append(['height_eq',value(x)]);return super().__eq__(x)
        def __rsub__(self,x):events.append(['height_rsub',value(x)]);return x-float(self)
        def __radd__(self,x):events.append(['height_radd',value(x)]);return x+float(self)
    class Index:
        def __index__(self):events.append(['height_index']);return 1
    height=s.get('height','one');h={'one':1,'zero':0,'negative':-1,'two':2,'out-of-range':1000,'fractional':1.5,'float-integral':np.float32(2),'float-fractional':np.float64(1.5),'u64':np.uint64(2),'array-one':np.array([1.5]),'array-two':np.array([1.,2.]),'nan':np.nan,'infinity':np.inf,'negative-infinity':-np.inf,'probe':Height(1.5),'index-only':Index()}[height]
    kw={};name=s.get('footprint')
    if name:
        if name=='cross':fp=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
        elif name=='directed':fp=np.array([[0,0,0],[0,1,1],[0,1,0]],bool)
        elif name=='zero':fp=np.zeros((3,3),bool)
        elif name=='center':fp=np.ones((1,1),bool)
        elif name=='even':fp=np.ones((2,2),bool)
        elif name=='wrongrank':fp=np.ones(3,bool)
        else:
            class Footprint:
                def astype(self,dtype,copy):
                    events.append(['footprint_astype',str(np.dtype(dtype)),copy])
                    if name=='raise':raise LookupError('h-footprint-stop')
                    if name=='mutate':
                        if s['function']=='h_maxima':image.flat[0]=image.flat[1]
                        else:image.flat[1]=image.flat[0]
                        events.append(['image_after_footprint',value(image)])
                    return np.ones((3,3),bool)
            fp=Footprint()
        kw['footprint']=fp
    return image,h,kw

def observe(s,function=None):
    events=[];image,h,kw=fixture(s,events);before=[value(image),value(h),value(kw)];reconstruction=[];cython=[];rec=SOURCE.grayreconstruct.reconstruction;loop=R.SOURCE.reconstruction_loop
    def guarded_loop(ranks,prev,next,strides,start,image_stride):
        row=dict(ranks_before=value(ranks),strides=value(strides),start=value(start),image_stride=value(image_stride),completed=False);cython.append(row)
        call=reconstruction[-1];seed=call['_seed'];mask=call['_mask'];fp=call['_footprint'];fp_shape=(3,)*seed.ndim if fp is None or not isinstance(fp,np.ndarray) else fp.shape
        assert seed.ndim==mask.ndim==len(fp_shape) and all(d%2 for d in fp_shape)
        shape=tuple(a+b-1 for a,b in zip(seed.shape,fp_shape));inside=tuple(slice(d//2,d//2+n) for d,n in zip(fp_shape,seed.shape));outside=np.ones(shape,bool);outside[inside]=False
        assert ranks.size==2*image_stride and int(np.prod(shape))==image_stride and ranks.size<100000
        a=ranks[:image_stride].reshape(shape);b=ranks[image_stride:].reshape(shape);assert np.all(a[outside]==0) and np.all(b[outside]==0) and np.all(a[inside]<=b[inside])
        result=loop(ranks,prev,next,strides,start,image_stride);row['completed']=True;row['ranks_after']=value(ranks);return result
    def guarded_rec(seed,mask,*args,**kwargs):
        row=dict(seed=value(seed),mask=value(mask),args=value(args),kwargs=value(kwargs),completed=False,_seed=seed,_mask=mask,_footprint=kwargs.get('footprint'));reconstruction.append(row)
        method=kwargs.get('method','dilation');wrong_order=np.any(seed>mask if method=='dilation' else seed<mask)
        if not wrong_order and (not np.isfinite(seed).all() or not np.isfinite(mask).all()):row['source_unsafe']='nonfinite unchecked rank/padding';raise SourceUnsafeBoundary('nonfinite reconstruction intake not executed')
        if s.get('footprint')=='wrongrank' and not wrong_order:row['source_unsafe']='footprint rank mismatch';raise SourceUnsafeBoundary('footprint rank mismatch not executed')
        result=rec(seed,mask,*args,**kwargs);row['completed']=True;row['result']=value(result);return result
    SOURCE.grayreconstruct.reconstruction=guarded_rec;R.SOURCE.reconstruction_loop=guarded_loop;result=None;error=None
    try:
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='warn'):
            warnings.simplefilter('always')
            try:result=(function or ORIGINAL[s['function']])(image,h,**kw)
            except Exception as e:error=dict(type=type(e).__name__,message=str(e))
        warning_rows=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]
    finally:SOURCE.grayreconstruct.reconstruction=rec;R.SOURCE.reconstruction_loop=loop
    for row in reconstruction:
        for k in ('_seed','_mask','_footprint'):row.pop(k)
    return dict(before=before,after=[value(image),value(h),value(kw)],result=value(result),error=error,warnings=warning_rows,events=events,reconstruction=reconstruction,cython=cython,original_numeric_completed=sum(bool(r['completed']) for r in cython),source_unsafe=any('source_unsafe' in r for r in reconstruction),input_alias=isinstance(result,np.ndarray) and isinstance(image,np.ndarray) and np.shares_memory(result,image))

def public_observation(row):return {k:v for k,v in row.items() if k not in ('reconstruction','cython','original_numeric_completed','source_unsafe')}
@pytest.mark.parametrize('s',cases(),ids=lambda s:s['id'])
def test_original_fixed_contract(s):
    d=json.loads(MATRIX.read_bytes());assert observe(s)==d['profiles'][R.profile()]['observations'][s['id']]
def test_source_scope_and_signatures():
    d=json.loads(MATRIX.read_bytes());assert d['cases']==cases() and d['new_native_paths']==0
    assert all(str(inspect.signature(f))=='(image, h, footprint=None)' for f in ORIGINAL.values())
    spec=importlib.util.spec_from_file_location('h_exact_source_provenance',ROOT/'tests/h_extrema_source_provenance.py')
    provenance=importlib.util.module_from_spec(spec);spec.loader.exec_module(provenance)
    record=provenance.authenticate_source(ROOT,'tests/test_h_extrema_contracts.py',d['test_canonical_lf_sha256'])
    assert record['historical_canonical_lf_sha256']==d['test_canonical_lf_sha256']
@pytest.mark.parametrize('name',ORIGINAL)
@pytest.mark.parametrize('workers',(1,4))
def test_strict_native_pending(name,workers):
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    declared=native_functions();path='morphology.'+name
    expected_exception=NotImplementedError;expected_source_error=None
    if path in declared:
        import inspect
        from skimagexx import _h_extrema as owned
        core=xx._core;public=getattr(xx.morphology,name)
        assert owned._core is core and core.__name__=='skimagexx._core'
        assert public is vars(owned)[name] and public.__module__=='skimagexx._h_extrema'
        assert inspect.signature(public)==inspect.signature(ORIGINAL[name])
        assert declared.count(path)==1 and xx.morphology.__all__.count(name)==1
        assert core.h_extrema_interface=='h-extrema-integer-v1'
        assert core.h_extrema_integer_widths==(8,16,32,64) and core.h_extrema_full_API is False
        for entry_name in ('h_extrema_validate_metadata','h_extrema_integer_range','h_extrema_integer_seed','h_extrema_integer_mask','h_extrema_counts'):
            entry=getattr(core,entry_name)
            assert inspect.isbuiltin(entry) and entry.__module__==core.__name__ and entry.__name__==entry_name
        assert core.h_extrema_real_interface=='h-extrema-real-v1'
        real_entry=core.h_extrema_real_mask
        assert inspect.isbuiltin(real_entry) and real_entry.__module__==core.__name__ and real_entry.__name__=='h_extrema_real_mask'
        raw_preentry=(ROOT/'coverage/contracts/h-extrema-object-preentry.json').read_bytes()
        assert hashlib.sha256(raw_preentry).hexdigest()=='10383844bf54762cf09293f5270cd6d49713371bba5d558b172f0293e512dff8'
        preentry=json.loads(raw_preentry);assert preentry['schema']=='h-exact-object-original-preentry-contract-v1'
        raw_profile=preentry['profiles'][R.profile()]
        profile_bytes=raw_profile['raw_report_utf8'].encode('utf-8')
        assert len(profile_bytes)==raw_profile['raw_report_bytes'] and hashlib.sha256(profile_bytes).hexdigest()==raw_profile['raw_report_sha256']
        profile_record=json.loads(profile_bytes);assert profile_record['profile']==R.profile()
        source_observation=profile_record['observations'][name+'/preentry/exact-object-height1']
        assert source_observation['original_H_entry_count']==1 and not source_observation['reconstruction_denial_attempts'] and source_observation['warnings']==[]
        expected_source_error=source_observation['error']
        assert expected_source_error['module']=='builtins' and expected_source_error['type']=='TypeError'
        expected_exception=TypeError
    else:
        assert path not in declared
    old=xx.get_num_threads();policy=xx.set_fallback_enabled(False);before=sys.getprofile();attempts=[]
    def guard(frame,event,arg):
        if event=='call' and frame.f_code is ORIGINAL[name].__code__:attempts.append(name);raise AssertionError('upstream extrema forwarding forbidden')
    try:
        xx.set_num_threads(workers);sys.setprofile(guard)
        with pytest.raises(expected_exception) as caught:getattr(xx.morphology,name)(object(),1)
        if expected_source_error is not None:
            assert dict(module=type(caught.value).__module__,type=type(caught.value).__name__,message=str(caught.value))==expected_source_error
        assert not attempts
    finally:sys.setprofile(before);xx.set_num_threads(old);xx.set_fallback_enabled(policy)
