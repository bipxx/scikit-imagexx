"""Saved-original parity through the real wrapper and private compiled stages.

No original functions are invoked. Class identity and Python frame provenance
are separate explicit frontiers, never rewritten into a claimed equality.
"""
import importlib
import importlib.util
import inspect
import ctypes
import json
import os
from pathlib import Path
import sys
import types
import numpy as np
import pytest
from join_segmentations_contract import cases
from join_segmentations_observer import observe

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def target():
    stage=Path(os.environ['JOIN_COMPONENT_DIR'])
    def extension(name,directory):
        path=next(directory.glob(name+'*.so'))
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    new=extension('_join_segmentations_core',stage)
    old=extension('_component_consumers_core',Path(os.environ['JOIN_CONSUMER_DIR']))
    package=types.ModuleType('_join_test_package');package.__path__=[str(ROOT/'python/skimagexx')]
    core=types.ModuleType(package.__name__+'._core')
    for module in (old,new):
        for key,value in vars(module).items():
            if not key.startswith('__'):setattr(core,key,value)
    package._core=core;sys.modules[package.__name__]=package;sys.modules[core.__name__]=core
    compat=importlib.import_module(package.__name__+'._compat');compat.set_fallback_enabled(False)
    module=importlib.import_module(package.__name__+'._join_segmentations')
    assert inspect.isbuiltin(core.join_encode) and inspect.isbuiltin(core.join_divmod)
    assert core.get_num_threads()==1
    return module,core


@pytest.mark.parametrize('spec',cases(),ids=lambda spec:spec['id'])
def test_saved_original(target,spec):
    module,core=target
    profile='minimum' if np.__version__.startswith('1.') else 'modern'
    fixture=json.loads((ROOT/'tests/fixtures/join_segmentations_originals_v1.json').read_text())
    expected=next(row for row in fixture['profiles'][profile]['rows'] if row['id']==spec['id'])
    # The retained Linux capture pins glibc x86-64 fenv_t to32bytes; all100
    # original rows preserved its complete bytes, including sticky flags.
    assert sys.platform=='linux'
    libc=ctypes.CDLL(None);before=(ctypes.c_ubyte*32)();after=(ctypes.c_ubyte*32)()
    assert libc.fegetenv(ctypes.byref(before))==0
    try:
        actual=observe(module.join_segmentations,spec,np)
        assert libc.fegetenv(ctypes.byref(after))==0
    finally:assert libc.fesetenv(ctypes.byref(before))==0
    actual['controller_fenv']=dict(before=bytes(before).hex(),after=bytes(after).hex())
    retained=Path(os.environ['JOIN_RESULT_DIR']);retained.mkdir(exist_ok=True)
    (retained/(spec['id'].split('/')[-1]+'.json')).write_text(json.dumps(actual,indent=2)+'\n')
    assert expected['controller_fenv']['before']==expected['controller_fenv']['after']
    assert bytes(before)==bytes(after),'raw FENV preservation'
    for key in ('error','input_before','input_after','returned_map_aliases','output_aliases','output_input_aliases','warnings','FP_events'):
        assert actual[key]==expected[key],key
    assert all(actual['checks'].values())
    if actual['returned_map_aliases'] is None:
        assert actual['result']==expected['result']
    else:
        assert actual['result']['joined']==expected['result']['joined']
        for left,right in zip(actual['result']['maps'],expected['result']['maps']):
            assert left['type']==['_join_test_package._component_consumers','ArrayMap']
            assert right['type']==['skimage.util._map_array','ArrayMap']
            assert {k:v for k,v in left.items() if k!='type'}=={k:v for k,v in right.items() if k!='type'}
    # Native stages need not synthesize original Python helper-frame counters.
    for left,right in zip(actual['map_protocols'],expected['map_protocols']):
        assert left['action']==right['action'] and left['error']==right['error']
        if 'maps' in left:
            for a,b in zip(left['maps'],right['maps']):
                assert {k:v for k,v in a.items() if k!='type'}=={k:v for k,v in b.items() if k!='type'}
            assert left['shared_in_values_after']==right['shared_in_values_after']
        else:assert left['result']==right['result']


def test_wrapping_and_worker_spans(target):
    _,core=target
    # Distinguishes wrapping the product BEFORE promotion from pair equivalence.
    left=np.array([2,3,4],dtype=np.uint8);right=np.array([1,2,3],dtype=np.uint16)
    actual=core.join_encode(left,right,200,np.dtype('uint8'),np.dtype('uint16'))
    assert actual.tolist()==[145,90,35]
    signed=core.join_encode(left,right,100,np.dtype('int8'),np.dtype('int16'))
    assert signed.tolist()==[-55,46,-109]
    left=np.arange(300001,dtype=np.uint64)%12345;right=left%17
    before=core.get_num_threads()
    try:
        core.set_num_threads(1);one=core.join_encode(left,right,123,np.dtype('uint64'),np.dtype('uint64'))
        core.set_num_threads(4);four=core.join_encode(left,right,123,np.dtype('uint64'),np.dtype('uint64'))
        assert np.array_equal(one,four)
        q,r=core.join_divmod(four,123,np.dtype('uint64'))
        assert np.array_equal(q,left) and np.array_equal(r,right)
    finally:core.set_num_threads(before)


def test_strict_unsupported_and_binary_arity(target):
    module,_=target
    with pytest.raises(TypeError):module.join_segmentations(np.zeros(1,dtype=np.int8))
    for dtype in ('complex64','complex128','object','S3','U3','datetime64[D]','timedelta64[D]','V3'):
        a=np.zeros(3,dtype=dtype)
        with pytest.raises(NotImplementedError,match='fallback is disabled'):module.join_segmentations(a,a)


def test_ordinary_sparse_image_and_callback_phase(target):
    module,core=target
    # Large, sparse original labels exceed the reference recipe bounds. Distinct
    # pairs are checked only in this nonoverflowing domain; never the wrap oracle.
    left=((np.arange(300019,dtype=np.int32)%37)*101).reshape(1,-1)
    right=((np.arange(300019,dtype=np.int32)%23)*79).reshape(1,-1)
    before=left.copy(),right.copy();budget=core.get_num_threads();seen=[]
    class Flag:
        def __bool__(self):
            seen.append(core.get_num_threads());core.set_num_threads(4);return True
    try:
        joined,one,two=module.join_segmentations(left,right,Flag())
        assert seen==[budget] and core.get_num_threads()==4
        assert one.in_values is two.in_values
        assert np.array_equal(one[joined],left) and np.array_equal(two[joined],right)
        assert np.array_equal(left,before[0]) and np.array_equal(right,before[1])
        assert len(np.unique(joined))==37*23
    finally:core.set_num_threads(budget)


def test_sdk_fixture_verbatim():
    from join_segmentations_sdk_fixture import render
    assert (ROOT/'tests/fixtures/join_segmentations_originals_v1.hpp').read_text()==render()


def test_no_array_algorithm_provider_forwarding(target,monkeypatch):
    module,_=target
    a=np.array([[0,997],[997,0]],dtype=np.int32);b=np.array([[0,79],[0,79]],dtype=np.int32)
    def refused(*args,**kwargs):raise RuntimeError('ARRAY_PROVIDER_REFUSED')
    for name in ('max','unique','multiply','add','divmod'):monkeypatch.setattr(np,name,refused)
    joined,one,two=module.join_segmentations(a,b,True)
    assert one.in_values is two.in_values
    assert joined.tolist()==[[0,3],[2,1]]


@pytest.mark.parametrize('dtype',('float16','float32','float64'))
def test_advertised_float_error_order(target,dtype):
    module,_=target
    positive=np.array([0,1],dtype=dtype);negative=np.array([-1,1],dtype=dtype)
    with pytest.raises(TypeError,match='label_field must have an integer dtype'):module.join_segmentations(positive,positive)
    with pytest.raises(ValueError,match='Cannot relabel array that contains negative values'):module.join_segmentations(negative,positive)


def test_warning_exception_precedes_factor_and_late_flag(target,monkeypatch):
    import warnings
    module,core=target
    a=np.array([0,1],dtype=np.int16)
    seen=[]
    class Flag:
        def __bool__(self):seen.append('flag');return True
    def forbidden(*args):raise RuntimeError('factor must follow relabel warnings')
    monkeypatch.setattr(core,'consumer_max',forbidden)
    consumers=importlib.import_module('_join_test_package._component_consumers')
    monkeypatch.setattr(consumers,'_mapping_notice',lambda:warnings.warn('JOIN_TEST_SHAPE_NOTICE',UserWarning))
    with warnings.catch_warnings():
        warnings.simplefilter('error',UserWarning)
        with pytest.raises(UserWarning,match='JOIN_TEST_SHAPE_NOTICE'):module.join_segmentations(a,a,Flag())
    assert seen==[]
