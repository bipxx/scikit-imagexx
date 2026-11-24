"""Strict saved originals; private component or installed namespace, no originals."""
import json
import os
from pathlib import Path
import pytest
from graycomatrix_graycoprops_contract import recipes
CASES=recipes()

@pytest.fixture(scope='module')
def gray_public():
    import ctypes
    import importlib
    import importlib.util
    import sys
    import types
    import numpy as np
    from util_views_fenv import FEnv,backend
    libc,helper=backend()
    def state():
        value=FEnv();assert libc.fegetenv(ctypes.byref(value))==0;return value
    def restore(value):
        assert helper.reference_fenv_restore(ctypes.byref(value),32)==0
    path=os.environ.get('GRAY_TEXTURE_CORE')
    if path:
        package=types.ModuleType('_gray_texture_native')
        package.__path__=[str(Path(__file__).parents[1]/'python/skimagexx')]
        sys.modules[package.__name__]=package
        spec=importlib.util.spec_from_file_location(package.__name__+'._core',path)
        core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
        sys.modules[spec.name]=core
        public=importlib.import_module(package.__name__+'._gray_texture')
    else:
        from skimagexx import feature as public,_core as core
    profile=os.environ['SKIMAGEXX_REFERENCE_PROFILE']
    saved=json.loads((Path(__file__).parent/'fixtures/gray_texture_original_v1.json').read_bytes())['profiles'][profile]
    return np,core,public,state,restore,{row['id']:row for row in saved}

@pytest.mark.parametrize('case',CASES,ids=[c['id']for c in CASES])
@pytest.mark.parametrize('workers',[int(x)for x in os.environ.get('GRAY_TEXTURE_TEST_WORKERS','1,4').split(',')])
def test_saved_gray_texture(gray_public,case,workers):
    from graycomatrix_graycoprops_observer import observe
    np,core,public,state,restore,saved=gray_public
    row=saved[case['id']];caller=state();old=core.get_num_threads();immediate={}
    try:
        core.set_num_threads(workers)
        def call(*args,**options):
            incoming=state()
            for key in ('control','status','mxcsr'):
                setattr(incoming,key,row['target_immediate']['before'][key])
            restore(incoming)
            immediate['before']={k:getattr(state(),k)for k in ('control','status','mxcsr')}
            try:
                return getattr(public,case['api'].split('.')[-1])(*args,**options)
            finally:
                immediate['after']={k:getattr(state(),k)for k in ('control','status','mxcsr')}
        actual=observe(call,case,np)
        assert actual=={k:row[k]for k in actual}
        assert immediate=={p:{k:row['target_immediate'][p][k]for k in ('control','status','mxcsr')}for p in ('before','after')}
    finally:
        core.set_num_threads(old);restore(caller);assert bytes(state())==bytes(caller)


def test_owned_native_entries(gray_public):
    import inspect
    np,core,public,state,restore,saved=gray_public
    assert inspect.isbuiltin(core.graycomatrix_apply)
    assert inspect.isbuiltin(core.graycoprops_apply)
    assert core.get_num_threads()==1
    assert str(inspect.signature(public.graycomatrix))=='(image, distances, angles, levels=None, symmetric=False, normed=False)'
    assert str(inspect.signature(public.graycoprops))=="(P, prop='contrast')"


def test_normalization_precedes_invalid_property(gray_public):
    np,core,public,state,restore,saved=gray_public
    caller=state();policy=np.geterr()
    try:
        p=np.ones((2,2,1,1));p[0,0,0,0]=float('inf')
        np.seterr(invalid='raise')
        with pytest.raises(FloatingPointError,match='divide'):
            public.graycoprops(p,'INVALID')
    finally:
        np.seterr(**policy);restore(caller);assert bytes(state())==bytes(caller)
