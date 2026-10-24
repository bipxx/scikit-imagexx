"""Actual public nested imports, shared core, and strict pinned rank parity."""
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import warnings
import numpy as np
import pytest
import skimagexx as xx
from skimage.filters import rank as reference
from skimage.filters.rank import generic as reference_generic
from skimagexx.filters import rank
from skimagexx.filters.rank import generic
from skimagexx import _core,_rank
from skimagexx.coverage import RANK_CANONICAL_FUNCTIONS,RANK_IDENTITY_ALIASES,rank_contract,native_functions

NAMES=('minimum','maximum','mean','sum','pop','gradient','median','modal','entropy')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

v1=load('rank_public_v1_contracts',Path(__file__).with_name('test_rank_neighborhood_contracts.py'))
v2=load('rank_public_v2_contracts',Path(__file__).with_name('test_rank_distribution_contracts.py'))

@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=xx.set_fallback_enabled(False);threads=xx.get_num_threads()
    xx.set_num_threads(request.param)
    yield request.param
    xx.set_num_threads(threads);xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('name',NAMES)
def test_own_wrapper_identity_signature_and_registry(name):
    assert rank is xx.filters.rank is importlib.import_module('skimagexx.filters.rank')
    assert generic is rank.generic is importlib.import_module('skimagexx.filters.rank.generic')
    assert getattr(rank,name) is getattr(generic,name) is getattr(_rank,name)
    assert getattr(rank,name) is not getattr(reference,name)
    assert getattr(reference_generic,name) is getattr(reference,name)
    assert inspect.signature(getattr(rank,name))==inspect.signature(getattr(reference,name))
    assert rank.__all__==_rank.__all__
    assert generic.__all__==[n for n in rank.__all__ if n in reference_generic.__all__]
    assert rank.__all__[:len(NAMES)]==list(NAMES)
    assert tuple('filters.rank.'+n for n in rank.__all__)==RANK_CANONICAL_FUNCTIONS
    assert _rank._core is _core
    assert xx.set_num_threads is _core.set_num_threads
    assert xx.get_num_threads is _core.get_num_threads
    assert _core.rank_histogram_capabilities & 511 == 511
    canonical='filters.rank.'+name;alias='filters.rank.generic.'+name
    assert canonical in native_functions() and alias in native_functions()
    assert RANK_IDENTITY_ALIASES[alias]==canonical
    assert rank_contract(alias)['identity_alias']
    assert not rank_contract(canonical)['identity_alias']


@pytest.mark.parametrize('surface',[rank,generic],ids=['canonical','generic-identity-alias'])
@pytest.mark.parametrize('case',v1.CASES,ids=[c[0] for c in v1.CASES])
def test_public_v1_frozen_contract(surface,case):
    key,name,parameters=case
    assert v1.observe(getattr(surface,name),parameters)==v1.observe(getattr(reference,name),parameters),key


@pytest.mark.parametrize('surface',[rank,generic],ids=['canonical','generic-identity-alias'])
@pytest.mark.parametrize('case',v2.CASES,ids=[c[0] for c in v2.CASES])
def test_public_v2_safe_frozen_contract(surface,case):
    key,name,parameters=case
    assert v2.observe(getattr(surface,name),parameters)==v2.observe(getattr(reference,name),parameters),key


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
def test_public_no_forwarding(name,ndim,dtype,monkeypatch):
    shape=(5,7) if ndim==2 else (3,4,5)
    image=np.arange(int(np.prod(shape)),dtype=dtype).reshape(shape)
    fp=np.ones((3,)*ndim,np.uint8)
    expected=getattr(reference,name)(image,fp)
    def forbidden(*args,**kwargs):raise AssertionError('upstream rank algorithm called')
    for operation in NAMES:
        monkeypatch.setattr(reference,operation,forbidden)
        monkeypatch.setattr(reference_generic,operation,forbidden)
    actual=getattr(rank,name)(image,fp)
    assert actual.dtype==expected.dtype and actual.strides==expected.strides
    assert actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('name',['gradient','median','modal','entropy'])
def test_pending_r2_alias_is_explicit_safely_rejected(name):
    image=np.arange(35,dtype=np.uint8).reshape(5,7)
    if name=='entropy':
        # Active integral entropy feedback has no defined original replay.
        before=image.copy()
        with pytest.raises(NotImplementedError,match='Overlapping buffers'):
            getattr(rank,name)(image,np.ones((3,3),np.uint8),out=image.view())
        np.testing.assert_array_equal(image,before)
    else:
        spec=dict(ndim=2,dtype='u1',output='u1',kind='view')
        closed=load('rank_public_closed_alias',Path(__file__).with_name('test_rank_alias_contracts.py'))
        assert closed.observe(getattr(rank,name),spec)==closed.observe(getattr(reference,name),spec)
    assert rank_contract('filters.rank.'+name)['alias_interface']=='rank-alias-v1'


def test_unported_names_preserve_policy_and_are_not_native_aliases():
    assert len(RANK_CANONICAL_FUNCTIONS)==len(rank.__all__)
    expected_aliases={f'filters.rank.{module}.'+name:'filters.rank.'+name
                      for module in ('generic','bilateral') for name in rank.__all__
                      if hasattr(importlib.import_module('skimage.filters.rank.'+module),name)
                      and getattr(importlib.import_module('skimage.filters.rank.'+module),name) is getattr(reference,name)}
    assert RANK_IDENTITY_ALIASES==expected_aliases
    assert 'majority' not in rank.__all__ and 'majority' not in generic.__all__
    image=np.arange(35,dtype=np.uint8).reshape(5,7);fp=np.ones((3,3),np.uint8)
    for module in (rank,generic):
        with pytest.raises(NotImplementedError,match='fallback is disabled'):module.majority(image,fp)
    previous=xx.set_fallback_enabled(True)
    try:
        with pytest.warns(xx.FallbackWarning):actual=rank.majority(image,fp)
        assert actual.tobytes()==reference.majority(image,fp).tobytes()
    finally:xx.set_fallback_enabled(previous)


def test_fresh_process_native_imports_without_scikit_image():
    # Derive aliases from pinned ORIGINAL parent references independently of
    # the native registry. Canonical-only percentile/bilateral names must not
    # acquire generic attributes merely because they are native callables.
    expected_generic=tuple(name for name in rank.__all__
                           if hasattr(reference_generic,name)
                           and getattr(reference_generic,name) is getattr(reference,name))
    assert len(expected_generic)==17
    script="expected_generic="+repr(expected_generic)+"\n"+r'''
import importlib.abc,sys
class NoUpstream(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='skimage' or fullname.startswith('skimage.'):
            raise ModuleNotFoundError('upstream deliberately unavailable',name=fullname)
sys.meta_path.insert(0,NoUpstream())
import numpy as np
import skimagexx as xx
import skimagexx.filters.rank as rank
import skimagexx.filters.rank.generic as generic
from skimagexx import _core,_rank
assert xx.get_num_threads()==1
assert _rank._core is _core
assert _core.rank_histogram_capabilities & 511 == 511
xx.set_fallback_enabled(False)
image=np.arange(35,dtype=np.uint8).reshape(5,7)
for workers in (1,4):
    xx.set_num_threads(workers)
    assert xx.get_num_threads()==workers
    for name in rank.__all__:
        function=getattr(rank,name)
        assert function is getattr(_rank,name)
        if name in expected_generic:
            assert function is getattr(generic,name)
        else:
            assert name not in generic.__dict__ and name not in generic.__all__
        expected=image.shape+(int(image.max())+1,) if name=='windowed_histogram' else image.shape
        assert function(image,np.ones((3,3),np.uint8)).shape==expected
    if 'mean_bilateral' in rank.__all__:
        import skimagexx.filters.rank.bilateral as bilateral
        assert bilateral is rank.bilateral
        assert bilateral.__all__==['mean_bilateral','pop_bilateral','sum_bilateral']
        for name in bilateral.__all__:
            assert getattr(bilateral,name) is getattr(rank,name)
            assert getattr(bilateral,name)(image,np.ones((3,3),np.uint8)).shape==image.shape
assert not any(n=='skimage' or n.startswith('skimage.') for n in sys.modules)
print('native canonical/default1/shared core/workers1+4, independent17generic aliases passed without skimage')
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,env=os.environ.copy())
    assert result.returncode==0,(result.stdout,result.stderr)
