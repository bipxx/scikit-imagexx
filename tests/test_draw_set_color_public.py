import importlib,json,sys
from pathlib import Path
import numpy as np
import pytest
from draw_set_color_contract import CASES,F32_CASES
from draw_set_color_observer import observe
from skimagexx import _core,set_fallback_enabled
from skimagexx._draw_set_color import set_color
ROOT=Path(__file__).resolve().parents[1]
SUCCESS=[x for x in CASES if not x['id'].startswith('error-')]
@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('spec',SUCCESS,ids=lambda x:x['id'])
def test_saved_native(spec,workers,monkeypatch):
    expected=next(x for x in json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_text())['observations'][np.__version__]if x['id']==spec['id'])
    def denied(*a,**k):raise AssertionError('original algorithm forbidden')
    monkeypatch.setattr(importlib.import_module('skimage.draw'),'set_color',denied)
    monkeypatch.setattr(importlib.import_module('skimage.draw.draw'),'set_color',denied)
    calls=[];results=[]
    def profile(frame,event,arg):
        if event=='c_return'and getattr(arg,'__name__','')in('draw_set_color','get_num_threads'):calls.append(arg.__name__)
        if event=='return'and frame.f_code is set_color.__code__:results.append(frame.f_locals.get('native_result'))
    previous=_core.get_num_threads();set_fallback_enabled(False);_core.set_num_threads(workers)
    sys.setprofile(profile)
    try:actual=observe(set_color,spec,np)
    finally:sys.setprofile(None);_core.set_num_threads(previous);set_fallback_enabled(True)
    assert actual==expected
    assert calls.count('get_num_threads')==1 and calls.count('draw_set_color')==1 and results==[True]

@pytest.mark.parametrize('spec',[x for x in CASES if x['id'].startswith('error-')],ids=lambda x:x['id'])
def test_error_frontier(spec,monkeypatch):
    def denied(*a,**k):raise AssertionError('original algorithm forbidden')
    monkeypatch.setattr(importlib.import_module('skimage.draw'),'set_color',denied)
    monkeypatch.setattr(importlib.import_module('skimage.draw.draw'),'set_color',denied)
    set_fallback_enabled(False)
    try:row=observe(set_color,spec,np)
    finally:set_fallback_enabled(True)
    if spec['id']=='error-color':
        expected=next(x for x in json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_text())['observations'][np.__version__]if x['id']==spec['id'])
        assert row==expected
    else:assert row['error'][1]=='NotImplementedError'

def test_refusal_does_not_write():
    image=np.ones((2,2),dtype=np.float64);rr=np.array([0],dtype=np.intp);cc=rr.copy();c=np.array([float('nan')]);a=np.array([.5]);before=image.copy()
    assert _core.draw_set_color(image,rr,cc,c,a,1)is False
    assert np.array_equal(image,before)

def test_default_restore():
    assert _core.get_num_threads()==1

@pytest.mark.parametrize('kind',['f16','protocol'])
def test_enabled_provider_preserves_boundary(kind,monkeypatch):
    from skimagexx._compat import FallbackWarning
    calls=[]
    class ArrayProvider:
        def __array__(self,*a,**k):raise AssertionError('unsupported protocol must not be normalized')
    image=np.zeros((2,2),dtype=np.float16)if kind=='f16'else ArrayProvider()
    coords=(np.array([0],dtype=np.intp),np.array([0],dtype=np.intp));color=.5
    def provider(*args,**kwargs):calls.append((args,kwargs));return None
    monkeypatch.setattr(importlib.import_module('skimage.draw'),'set_color',provider)
    set_fallback_enabled(True)
    with pytest.warns(FallbackWarning):assert set_color(image,coords,color)is None
    assert len(calls)==1 and calls[0][0][0]is image and calls[0][0][1]is coords

@pytest.mark.parametrize('kind',['color-f32','color-longdouble','alpha-infinity','alpha-huge-int'])
def test_unassessed_conversion_frontier(kind,monkeypatch):
    image=np.zeros((2,2));coords=(np.array([0],dtype=np.intp),np.array([0],dtype=np.intp));color=np.array([.5]);alpha=.5
    if kind=='color-f32':color=color.astype(np.float32)
    if kind=='color-longdouble':color=np.array([np.finfo(np.longdouble).max],dtype=np.longdouble)
    if kind=='alpha-infinity':alpha=float('inf')
    if kind=='alpha-huge-int':alpha=2**1024
    calls=[]
    def provider(*args,**kwargs):calls.append((args,kwargs));return None
    monkeypatch.setattr(importlib.import_module('skimage.draw'),'set_color',provider)
    from skimagexx._compat import FallbackWarning
    with pytest.warns(FallbackWarning)as notices:assert set_color(image,coords,color,alpha=alpha)is None
    assert len(notices)==1 and len(calls)==1 and calls[0][0][2]is color and calls[0][1]['alpha']is alpha
    assert np.count_nonzero(image)==0

@pytest.mark.parametrize('kind',['swapped-image','f16-image','overlap-image','alpha-length','unaligned-color','readonly-image','invalid-workers'])
def test_private_boundary_no_write(kind):
    image=np.ones((2,2));rr=np.array([0],dtype=np.intp);cc=rr.copy();color=np.array([.5]);alpha=np.array([.5]);workers=1
    if kind=='swapped-image':image=image.astype('>f8')
    if kind=='f16-image':image=image.astype(np.float16)
    if kind=='overlap-image':image=np.ndarray((2,2),dtype=np.float64,buffer=image,strides=(0,8))
    if kind=='alpha-length':alpha=np.array([],dtype=np.float64)
    if kind=='unaligned-color':color=np.ndarray((1,),dtype=np.float64,buffer=np.zeros(9,dtype=np.uint8),offset=1)
    if kind=='readonly-image':image.flags.writeable=False
    if kind=='invalid-workers':workers=0
    before=image.tobytes()
    assert _core.draw_set_color(image,rr,cc,color,alpha,workers)is False
    assert image.tobytes()==before

def test_default_alpha_owned():
    image=np.full((2,2),.5);coords=(np.array([0],dtype=np.intp),np.array([1],dtype=np.intp));set_fallback_enabled(False)
    try:assert set_color(image,coords,.25)is None
    finally:set_fallback_enabled(True)
    assert image[0,1]==.25 and image[1,0]==.5

@pytest.mark.parametrize('workers',[1,4])
def test_integer_stage_saved_original_holdout(workers,monkeypatch):
    contract=json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_text())
    profile='minimum'if np.__version__=='1.24.4'else'modern'
    gold=contract['integer_stage_supplement'][profile]
    original=next(row for row in gold['records']if row['role']=='original')
    def denied(*args,**kwargs):raise AssertionError('no original recapture')
    monkeypatch.setattr(importlib.import_module('skimage.draw'),'set_color',denied)
    monkeypatch.setattr(importlib.import_module('skimage.draw.draw'),'set_color',denied)
    calls=[];accepted=[]
    def trace(frame,event,value):
        if event=='c_return'and getattr(value,'__name__','')in('draw_set_color','get_num_threads'):calls.append(value.__name__)
        if event=='return'and frame.f_code is set_color.__code__:accepted.append(frame.f_locals.get('native_result'))
    image=np.array([[-0.0]]);coords=(np.array([0],dtype=np.intp),np.array([0],dtype=np.intp));color=np.array([0],dtype=np.int64)
    previous=_core.get_num_threads();_core.set_num_threads(workers);set_fallback_enabled(False);prior=sys.getprofile();sys.setprofile(trace)
    try:result=set_color(image,coords,color,alpha=-1)
    finally:sys.setprofile(prior);_core.set_num_threads(previous);set_fallback_enabled(True)
    assert result is None and image.tobytes().hex()==original['image_words']=='0000000000000000'
    assert calls==['get_num_threads','draw_set_color']and accepted==[True]

@pytest.mark.parametrize('dtype',['int8','int16','int32','int64','uint8','uint16','uint32','uint64','float64'])
@pytest.mark.parametrize('integer_alpha',[False,True])
@pytest.mark.parametrize('workers',[1,4])
def test_integer_stage_ufunc_kind_and_words(dtype,integer_alpha,workers):
    contract=json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_text())
    fact=next(row for row in contract['integer_stage_operation_witnesses'][np.__version__]['rows']if row['color_dtype']==dtype and row['alpha_kind']==('builtin-int'if integer_alpha else'builtin-float'))
    # Values are independently frozen NumPy-operation facts, not candidate output.
    expected_zero='0000000000000000'if fact['weighted_dtype']=='<i8'else'0000000000000080'
    expected=expected_zero+np.array([-1.],dtype=np.float64).tobytes().hex()
    image=np.full((1,1,2),-0.);coords=(np.array([0],dtype=np.intp),np.array([0],dtype=np.intp));color=np.array([0,1],dtype=dtype)
    previous=_core.get_num_threads();_core.set_num_threads(workers);set_fallback_enabled(False)
    try:assert set_color(image,coords,color,alpha=-1 if integer_alpha else-1.)is None
    finally:_core.set_num_threads(previous);set_fallback_enabled(True)
    assert image.tobytes().hex()==expected

@pytest.mark.parametrize('value',[-16,-1,0,1,16])
def test_integer_stage_bounds_and_zero(value):
    image=np.array([[-0.0 if value==0 else .5]]);coords=(np.array([0],dtype=np.intp),np.array([0],dtype=np.intp));color=np.array([-16 if value else-1],dtype=np.int64)
    set_fallback_enabled(False)
    try:assert set_color(image,coords,color,alpha=value)is None
    finally:set_fallback_enabled(True)
    expected=0.0 if value==0 else .5*(1-value)+(-16*value)
    assert image.tobytes()==np.array([[expected]],dtype=np.float64).tobytes()

@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('spec',F32_CASES,ids=lambda x:x['id'])
def test_float32_saved_native(spec,workers,monkeypatch):
    gold=next(x for x in json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_text())['float32_original_observations'][np.__version__]if x['id']==spec['id'])
    def denied(*args,**kwargs):raise AssertionError('originalrecaptureforbidden')
    monkeypatch.setattr(importlib.import_module('skimage.draw'),'set_color',denied)
    monkeypatch.setattr(importlib.import_module('skimage.draw.draw'),'set_color',denied)
    calls=[];accepted=[]
    def profile(frame,event,value):
        if event=='c_return'and getattr(value,'__name__','')in('draw_set_color','get_num_threads'):calls.append(value.__name__)
        if event=='return'and frame.f_code is set_color.__code__:accepted.append(frame.f_locals.get('native_result'))
    previous=_core.get_num_threads();_core.set_num_threads(workers);set_fallback_enabled(False);old=sys.getprofile();sys.setprofile(profile)
    try:row=observe(set_color,spec,np)
    finally:sys.setprofile(old);_core.set_num_threads(previous);set_fallback_enabled(True)
    assert row==gold and calls==['get_num_threads','draw_set_color']and accepted==[True]

@pytest.mark.parametrize('dtype',['float64','float32'])
@pytest.mark.parametrize('operand',['coords','color','alpha'])
@pytest.mark.parametrize('workers',[1,4])
def test_float32_alignment_uses_known_gold(dtype,operand,workers):
    from draw_set_color_contract import fixture
    name='gray'if dtype=='float64'else'f32-gray';spec={'id':name}if dtype=='float64'else{'id':name,'partition':'float32'}
    args,kw=fixture(spec,np);image,coords,color=args;rr,cc=coords
    contract=json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_text());section='observations'if dtype=='float64'else'float32_original_observations';gold=next(x for x in contract[section][np.__version__]if x['id']==name)
    def unaligned(value):
        result=np.ndarray(value.shape,dtype=value.dtype,buffer=np.zeros(value.nbytes+1,dtype=np.uint8),offset=1);result[:]=value
        assert result.flags.c_contiguous and not result.flags.aligned
        return result
    if operand=='coords':rr,cc=unaligned(rr),unaligned(cc)
    elif operand=='color':color=unaligned(color)
    else:kw['alpha']=unaligned(np.full(rr.shape,kw['alpha'],dtype=np.float64))
    previous=_core.get_num_threads();_core.set_num_threads(workers);set_fallback_enabled(False)
    try:assert set_color(image,(rr,cc),color,**kw)is None
    finally:_core.set_num_threads(previous);set_fallback_enabled(True)
    import zlib,base64
    encoded=gold['after']['items'][0]['items'][0]
    assert image.tobytes()==zlib.decompress(base64.b64decode(encoded['raw_zlib_base64']))

@pytest.mark.parametrize('kind',['subnormal-input','snan-input','tiny-output'])
def test_float32_private_window_refuses_without_write(kind):
    image=np.zeros((1,1),dtype=np.float32);r=c=np.array([0],dtype=np.intp);color=np.array([.25]);alpha=np.array([.5])
    if kind=='subnormal-input':image.view(np.uint32)[0,0]=1
    if kind=='snan-input':image.view(np.uint32)[0,0]=0x7f800001
    if kind=='tiny-output':color[:]=2.**-128;alpha[:]=2.**-128
    before=image.tobytes();events=[]
    with np.errstate(all='call',call=lambda *args:events.append(args)):
        assert _core.draw_set_color(image,r,c,color,alpha,1)is False
    assert image.tobytes()==before and events==[]

@pytest.mark.parametrize('dtype',['float32','float64'])
def test_float32_aligned_fastpath_and_cached_copy_policy(dtype,monkeypatch):
    image=np.full((1,1),.5,dtype=dtype);coords=(np.array([0],dtype=np.intp),np.array([0],dtype=np.intp));color=np.array([.25]);alpha=np.array([.5])
    def denied(*args,**kwargs):raise AssertionError('alignedcallmustnotcopyorparseversion')
    # Scope the spy to this boundary; unchanged CAPI storage helpers may query
    # the real NumPy version independently of the wrapper copy policy.
    import types
    class BoundaryNumPy:
        def __getattr__(self,name):
            if name=='require':return denied
            if name=='lib':return types.SimpleNamespace(NumpyVersion=denied)
            return getattr(np,name)
    monkeypatch.setitem(set_color.__globals__,'np',BoundaryNumPy())
    set_fallback_enabled(False)
    try:assert set_color(image,coords,color,alpha=alpha)is None
    finally:set_fallback_enabled(True)
    assert image[0,0]==.375
