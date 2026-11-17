"""Saved original parity; private family bridge is explicit, never genuine credit."""
import builtins
import ctypes
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types
import pytest

ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'tests/fixtures/flood_fill_original_v1.json'


@pytest.fixture(scope='module')
def target():
    import numpy as np
    path=os.environ.get('FLOOD_FILL_CORE_EXTENSION')
    if path:
        package=types.ModuleType('flood_fill_component');package.__path__=[]
        sys.modules[package.__name__]=package
        spec=importlib.util.spec_from_file_location(package.__name__+'._core',path)
        core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
        sys.modules[spec.name]=core
        spec=importlib.util.spec_from_file_location(package.__name__+'.frontend',ROOT/'python/skimagexx/_flood_fill.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    else:
        module=importlib.import_module('skimagexx.morphology')
        segmentation=importlib.import_module('skimagexx.segmentation')
        assert module.flood is segmentation.flood and module.flood_fill is segmentation.flood_fill
        core=importlib.import_module('skimagexx._core')
        assert callable(core.flood_mask) and callable(core.flood_assign)
    assert all(inspect.isbuiltin(getattr(core,name)) for name in ('flood_mask','flood_assign','crop_copy_into'))
    return np,module


from flood_fill_contract import recipes


@pytest.mark.parametrize('spec',recipes(),ids=lambda s:s['id'])
def test_saved_original(target,spec,monkeypatch):
    from flood_fill_observer import observe
    from util_views_fenv import FEnv,backend
    np,module=target
    profile='minimum' if np.__version__.split('.')[0]=='1' else 'modern'
    saved=json.loads(SAVED.read_bytes())['profiles'][profile]['rows']
    expected=next(row for row in saved if row['id']==spec['id'])
    libc,helper=backend()
    def state():
        value=FEnv();assert libc.fegetenv(ctypes.byref(value))==0;return value
    def restore(value):assert helper.reference_fenv_restore(ctypes.byref(value),32)==0
    def numerical(value):return dict(control=value.control,status=value.status,mxcsr=value.mxcsr)
    outer=state()
    # Source input envelope only: copy trusted numerical state into LIVE storage.
    # Instruction/data pointers and reserved words always remain current-process.
    incoming=state()
    for key in ('control','status','mxcsr'):
        setattr(incoming,key,expected['target_immediate']['before'][key])
    observed={}
    original_import=builtins.__import__
    def refused(name,*args,**kwargs):
        if name=='skimage' or name.startswith('skimage.'):
            raise AssertionError('original request forbidden: '+name)
        return original_import(name,*args,**kwargs)
    def call(*args,**options):
        # Factory/observer setup is outside this exact captured target input.
        restore(incoming);observed['before']=numerical(state())
        try:return getattr(module,spec['api'])(*args,**options)
        finally:observed['after']=numerical(state())
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(builtins,'__import__',refused)
            actual=observe(call,spec,np)
        assert actual=={key:expected[key]for key in actual}
        assert observed=={when:{key:expected['target_immediate'][when][key]
                               for key in ('control','status','mxcsr')}
                          for when in ('before','after')}
    finally:
        restore(outer);assert bytes(state())==bytes(outer)


@pytest.mark.parametrize('dtype',['object',[('field','i4')]])
def test_private_assignment_refuses_non_pod_without_mutation(target,dtype):
    np,module=target
    core=module._core if hasattr(module,'_core')else importlib.import_module('skimagexx._core')
    a=np.array([1,2],dtype=dtype);before=a.copy();mask=np.ones(2,dtype=bool);values=np.array([3,4],dtype=dtype)
    with pytest.raises(TypeError,match='numeric POD dtype'):core.flood_assign(a,mask,values)
    assert np.array_equal(a,before)


@pytest.mark.parametrize('tolerance',[0,1])
def test_bool_tolerance_pinned_numeric_limits(target,tolerance):
    np,module=target
    a=np.array([[False,False],[True,True]])
    actual=module.flood(a,(0,0),tolerance=tolerance)
    expected=np.ones((2,2),dtype=bool)if tolerance else np.array([[True,True],[False,False]])
    assert np.array_equal(actual,expected)


@pytest.fixture(autouse=True)
def genuine_global_worker_and_caller(target,request):
    if os.environ.get('FLOOD_FILL_CORE_EXTENSION'):
        yield
        return
    from util_views_fenv import FEnv,backend
    import hashlib
    core=importlib.import_module('skimagexx._core')
    libc,helper=backend();outer=FEnv();assert libc.fegetenv(ctypes.byref(outer))==0
    assert core.get_num_threads()==1
    workers=1 if sum(map(ord,request.node.nodeid))%2 else 4
    core.set_num_threads(workers);assert core.get_num_threads()==workers
    try:
        yield
        assert core.get_num_threads()==workers
    finally:
        core.set_num_threads(1);assert core.get_num_threads()==1
        assert helper.reference_fenv_restore(ctypes.byref(outer),32)==0
        actual=FEnv();assert libc.fegetenv(ctypes.byref(actual))==0
        assert bytes(actual)==bytes(outer)
        directory=os.environ.get('FLOOD_FILL_CALLER_RECORD_DIR')
        if directory:
            out=Path(directory);out.mkdir(exist_ok=True,parents=True)
            record=dict(node=request.node.nodeid,workers=workers,saved=bytes(outer).hex(),actual=bytes(actual).hex(),exact32=True,scope='Serial BFS leaves shared global unchanged; one selected1or4 setting per row, not two-worker per-case matrix')
            (out/(hashlib.sha256(request.node.nodeid.encode()).hexdigest()+'.json')).write_text(json.dumps(record))
