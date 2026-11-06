"""Saved originals only; explicit component proof before shared public wiring."""
import ctypes
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import weakref
import gc
import numpy as np
import pytest
from find_boundaries_contract import cases,fixture
from find_boundaries_observer import observe,reference_runtime
ROOT=Path(__file__).resolve().parents[1]
MATRIX=json.loads((ROOT/'coverage/contracts/find-boundaries-v1.json').read_bytes())

@pytest.fixture(scope='module')
def owned():
    directory=os.environ.get('FIND_BOUNDARIES_COMPONENT_DIR')
    if not directory:return importlib.import_module('skimagexx._find_boundaries')
    # A REAL compiled private builtin component, not a fake full-product core.
    sys.path.insert(0,directory)
    core=importlib.import_module('_find_boundaries_core')
    package=types.ModuleType('find_boundaries_component');package.__path__=[str(ROOT/'python/skimagexx')]
    sys.modules[package.__name__]=package;sys.modules[package.__name__+'._core']=core
    return importlib.import_module(package.__name__+'._find_boundaries')

def original_codes(original):
    m=importlib.import_module('skimage.segmentation.boundaries');codes=set()
    gray=importlib.import_module('skimage.morphology.gray')
    for function in(original,m._find_boundaries_subpixel,m.dilation,m.erosion,m.ndi.generate_binary_structure,gray._iterate_gray_func,m.view_as_windows,m.ndi.grey_erosion,m.ndi.grey_dilation,m.ndi.minimum_filter,m.ndi.maximum_filter):
        while True:
            codes.add(function.__code__)
            if not hasattr(function,'__wrapped__'):break
            function=function.__wrapped__
    return codes

@pytest.fixture(scope='module')
def reference():
    numpy,original,runtime=reference_runtime()
    return numpy,original,runtime,original_codes(original)

@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_find_boundaries_saved_public(spec,workers,owned,reference):
    _,original,_,denied=reference;expected=next(r for r in MATRIX['original_observations'][np.__version__]if r['id']==spec['id'])
    returns=[];accepted=[];core=owned._core
    def guard(frame,event,value):
        if event=='call'and frame.f_code in denied:raise AssertionError('Original/dependency code forbidden in saved parity')
        if event=='c_return'and value is core.find_boundaries_into:returns.append(True)
        if event=='return'and frame.f_code is owned.find_boundaries.__code__:accepted.append(frame.f_locals.get('native_result'))
    previous=sys.getprofile();old=core.get_num_threads();core.set_num_threads(workers);sys.setprofile(guard)
    try:row=observe(owned.find_boundaries,spec,np)
    finally:sys.setprofile(previous);core.set_num_threads(old)
    if spec['id'].endswith('signed-inrange-rounded-words'):
        assert row['error']['type']=='NotImplementedError'and row['result']is None
        assert returns==[True]and accepted==[None]
        assert row['before']==expected['before']and row['after']==expected['after']
        return # Explicit CONTROL-ONLY source partition, no parity/native credit.
    assert row=={key:value for key,value in expected.items()if key!='controller_fenv'}
    if expected['error']is None:assert returns==[True]and accepted==[True]
    else:assert returns==[]and accepted==[None]
    assert core.get_num_threads()==old

@pytest.mark.parametrize('kind',['float32','swapped','rank9','readonly_output','wrong_output_dtype','aliased_output','nonexact64','outer64'])
def test_find_boundaries_private_refusal_before_write(kind,owned):
    a=np.array([[0,1],[1,0]],dtype=np.uint16);out=np.zeros(a.shape,dtype=bool);mode=0
    if kind=='float32':a=a.astype(np.float32)
    elif kind=='swapped':a=a.astype(a.dtype.newbyteorder('>'))
    elif kind=='rank9':a=np.zeros((1,)*9,dtype=np.uint8);out=np.zeros(a.shape,dtype=bool)
    elif kind=='readonly_output':out.flags.writeable=False
    elif kind=='wrong_output_dtype':out=np.zeros(a.shape,dtype=np.uint8)
    elif kind=='aliased_output':a=a.astype(bool);out=a
    elif kind=='nonexact64':a=np.array([[0,2**53+1],[1,0]],dtype=np.uint64)
    elif kind=='outer64':a=a.astype(np.int64);mode=2
    before=out.tobytes();assert owned._core.find_boundaries_into(a,out,1,mode,0,True,1)is None
    assert out.tobytes()==before

def test_find_boundaries_invalid_strides_and_external_storage(owned):
    root=np.arange(4,dtype=np.uint8)
    bad=np.lib.stride_tricks.as_strided(root,shape=(2,2),strides=(100,1))
    out=np.zeros((2,2),dtype=bool)
    with pytest.raises(ValueError,match='storage|allocation|root'):owned._core.find_boundaries_into(bad,out,1,0,0,True,1)
    assert not out.any()
    a=np.ndarray((2,2),dtype=np.uint8,buffer=bytearray([0,1,1,0]));out.fill(False)
    with pytest.raises(ValueError,match='storage'):owned._core.find_boundaries_into(a,out,1,0,0,True,1)
    assert not out.any() # Existing ordinary external-buffer policy gap, explicit.

def test_find_boundaries_current_descriptor_after_allocation(owned,monkeypatch):
    a=np.array([[0,1],[1,0]],dtype=np.uint8);allocate=np.empty_like
    def changing(*args,**kwargs):
        result=allocate(*args,**kwargs);a.shape=(4,);return result
    monkeypatch.setattr(owned.np,'empty_like',changing)
    with pytest.raises(NotImplementedError,match='frontier'):owned.find_boundaries(a)

def test_find_boundaries_owner_lifetime_and_no_array_reductions(owned,monkeypatch):
    a=np.array([[0,0],[0,1]],dtype=np.uint8);owner=weakref.ref(a)
    def deny(*args,**kwargs):raise AssertionError('NumPy reduction/unique algorithm forbidden')
    for name in('min','max','amin','amax','unique'):monkeypatch.setattr(np,name,deny)
    out=owned.find_boundaries(a);assert out.dtype==np.dtype(bool)and out.flags.owndata
    assert not np.shares_memory(a,out);del a;gc.collect();assert owner()is None
    assert out.tolist()==[[False,True],[True,True]]

def test_find_boundaries_permuted_owning_layout_and_current_pixels(owned,monkeypatch):
    a=np.empty_like(np.arange(12,dtype=np.uint16).reshape(2,2,3).transpose(2,0,1));a.fill(1)
    assert a.flags.owndata and not a.flags.c_contiguous and not a.flags.f_contiguous
    out=owned.find_boundaries(a);assert out.strides==tuple(s//2 for s in a.strides)and out.flags.owndata
    assert not out.any()
    allocate=np.empty_like
    def mutate(*args,**kwargs):
        result=allocate(*args,**kwargs);a.fill(0);return result
    monkeypatch.setattr(owned.np,'empty_like',mutate);assert not owned.find_boundaries(a).any()

def test_find_boundaries_np_policy_and_fenv(owned):
    libc=ctypes.CDLL(None)
    if not hasattr(libc,'fetestexcept'):pytest.skip('C fenv entry unavailable')
    libc.fetestexcept.argtypes=[ctypes.c_int];libc.fetestexcept.restype=ctypes.c_int
    a=np.array([[0,1],[1,0]],dtype=np.uint32);policy=np.geterr();handler=np.geterrcall()
    try:
        np.seterr(all='raise');flags=libc.fetestexcept(0x3d);rounding=libc.fegetround()
        owned.find_boundaries(a);assert libc.fetestexcept(0x3d)==flags and libc.fegetround()==rounding
        assert np.geterr()==dict(divide='raise',over='raise',under='raise',invalid='raise')and np.geterrcall()is handler
    finally:np.seterr(**policy);np.seterrcall(handler)
def test_find_boundaries_protocols_refused_without_conversion_replay(owned):
    calls=[]
    class ArrayProtocol:
        @property
        def dtype(self):calls.append('dtype');return np.dtype(np.uint8)
        def __array__(self,*args,**kwargs):calls.append('array');raise AssertionError('array conversion forbidden')
    with pytest.raises(NotImplementedError):owned.find_boundaries(ArrayProtocol())
    assert calls==['dtype'];calls.clear()
    class ScalarProtocol:
        def __index__(self):calls.append('index');return 1
        def __int__(self):calls.append('int');return 1
        def __eq__(self,other):calls.append('eq');return False
    a=np.zeros((2,2),dtype=np.uint8);out=np.zeros(a.shape,dtype=bool)
    for name in('mode','connectivity','background'):
        with pytest.raises(NotImplementedError):owned.find_boundaries(a,**{name:ScalarProtocol()})
    assert calls==[]
    with pytest.raises(TypeError):owned._core.find_boundaries_into(a,out,ScalarProtocol(),0,0,True,1)
    assert calls==[]and not out.any()
