"""Strict saved Linux originals and live Windows parity."""
import json
import os
from pathlib import Path
import sys
import pytest
from graycomatrix_graycoprops_contract import recipes
CASES=recipes()

@pytest.fixture(scope='module')
def gray_public():
    import importlib
    import importlib.util
    import types
    import numpy as np
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
    return np,core,public


def saved_linux():
    import ctypes
    from util_views_fenv import FEnv,backend
    libc,helper=backend()
    def state():
        value=FEnv();assert libc.fegetenv(ctypes.byref(value))==0;return value
    def restore(value):
        assert helper.reference_fenv_restore(ctypes.byref(value),32)==0

    profile=os.environ['SKIMAGEXX_REFERENCE_PROFILE']
    saved=json.loads((Path(__file__).parent/'fixtures/gray_texture_original_v1.json').read_bytes())['profiles'][profile]
    return state,restore,{row['id']:row for row in saved}


@pytest.fixture(scope='module')
def linux_reference():
    return None if sys.platform == 'win32' else saved_linux()

@pytest.mark.parametrize('case',CASES,ids=[c['id']for c in CASES])
@pytest.mark.parametrize('workers',[int(x)for x in os.environ.get('GRAY_TEXTURE_TEST_WORKERS','1,4').split(',')])
def test_saved_gray_texture(gray_public,linux_reference,case,workers):
    from graycomatrix_graycoprops_observer import observe
    np,core,public=gray_public
    name=case['api'].split('.')[-1]
    if sys.platform == 'win32':
        from skimage import feature

        old=core.get_num_threads()
        try:
            core.set_num_threads(workers)
            assert core.get_num_threads()==workers
            expected=observe(getattr(feature,name),case,np)
            actual=observe(getattr(public,name),case,np)
            assert actual==expected
        finally:
            core.set_num_threads(old)
        return

    state,restore,saved=linux_reference
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
                return getattr(public,name)(*args,**options)
            finally:
                immediate['after']={k:getattr(state(),k)for k in ('control','status','mxcsr')}
        actual=observe(call,case,np)
        assert actual=={k:row[k]for k in actual}
        assert immediate=={p:{k:row['target_immediate'][p][k]for k in ('control','status','mxcsr')}for p in ('before','after')}
    finally:
        core.set_num_threads(old);restore(caller);assert bytes(state())==bytes(caller)


def test_owned_native_entries(gray_public):
    import inspect
    np,core,public=gray_public
    assert inspect.isbuiltin(core.graycomatrix_apply)
    assert inspect.isbuiltin(core.graycoprops_apply)
    assert core.get_num_threads()==1
    assert str(inspect.signature(public.graycomatrix))=='(image, distances, angles, levels=None, symmetric=False, normed=False)'
    assert str(inspect.signature(public.graycoprops))=="(P, prop='contrast')"


@pytest.mark.parametrize('prop', ['ASM', 'energy', 'correlation'])
def test_fortran_histogram_reduction(gray_public, prop):
    np, _, public = gray_public
    if sys.platform == 'win32':
        from skimage.feature import graycoprops as reference
    else:
        def reference(array, name):
            return public.graycoprops(array.astype(np.float64), name)

    rng = np.random.default_rng(7)
    for shape in [(3, 3, 2, 3), (9, 9, 3, 2), (64, 64, 2, 2)]:
        array = np.array(rng.integers(0, 1000, size=shape, dtype=np.uint32), order='F')
        array[:, :, 0, 0] = 0
        array[:, :, 1, 0] = 0
        array[0, :, 1, 0] = 1
        for values in (array, array[::-1]):
            expected = reference(values, prop)
            actual = public.graycoprops(values, prop)
            assert actual.dtype == expected.dtype and actual.strides == expected.strides
            assert actual.tobytes() == expected.tobytes()


def test_normalization_precedes_invalid_property(gray_public,linux_reference):
    np,core,public=gray_public
    if linux_reference is not None:
        state,restore,_=linux_reference
        caller=state()
    try:
        p=np.ones((2,2,1,1));p[0,0,0,0]=float('inf')
        with np.errstate(invalid='raise'):
            with pytest.raises(FloatingPointError,match='divide'):
                public.graycoprops(p,'INVALID')
    finally:
        if linux_reference is not None:
            restore(caller);assert bytes(state())==bytes(caller)
