"""Actual public R3 exports, sealed observations and shared production core.

No development adapter, alternate backend or upstream image forwarding is
accepted here. Private alias support does not broaden public R3 admission.
"""
import hashlib
import importlib
import importlib.util
import inspect
import json
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
from skimagexx import _rank
from skimagexx.coverage import native_functions,rank_contract


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


contracts=load('rank_intensity_public_contracts',Path(__file__).with_name('test_rank_intensity_contracts.py'))
NAMES=contracts.NAMES
MATRIX=json.loads(contracts.MATRIX.read_text(encoding='utf-8'))
policy=load('rank_intensity_public_policy',Path(__file__).with_name('test_rank_policies.py'))
integer=load('rank_intensity_public_integer',Path(__file__).with_name('test_rank_integer_warning.py'))


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=xx.set_fallback_enabled(False);threads=xx.get_num_threads()
    xx.set_num_threads(request.param)
    yield request.param
    xx.set_num_threads(threads);xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('name',NAMES)
def test_public_identity_signature_registry_and_shared_core(name):
    assert _rank._core is xx._core
    assert xx.set_num_threads is xx._core.set_num_threads and xx.get_num_threads is xx._core.get_num_threads
    assert xx._core.rank_histogram_capabilities & 8191==8191
    assert getattr(rank,name) is getattr(generic,name) is getattr(_rank,name)
    assert getattr(reference,name) is getattr(reference_generic,name)
    assert getattr(rank,name) is not getattr(reference,name)
    assert inspect.signature(getattr(rank,name))==inspect.signature(getattr(reference,name))
    for path in ('filters.rank.'+name,'filters.rank.generic.'+name):
        assert path in native_functions()
        facts=rank_contract(path)
        assert facts['version']=='rank-histogram-v3' and facts['capability']==8191
        assert facts['operation']==9+NAMES.index(name)
        assert facts['identity_alias']==('.generic.' in path)
        assert facts['parity_evidence'] is facts['performance_evidence'] is None
    contracts.test_source_signature_and_public_generic_identity(name,MATRIX)


@pytest.mark.parametrize('surface',[rank,generic],ids=['canonical','generic-identity-alias'])
@pytest.mark.parametrize('case',contracts.CASES,ids=[c[0] for c in contracts.CASES])
def test_every_sealed_observation_through_actual_public_export(surface,case):
    key,name,parameters=case
    expected=MATRIX['observations'][MATRIX['snapshots'][contracts.profile()][key]]
    assert contracts.observe(getattr(surface,name),parameters)==expected,key
    assert contracts.observe(getattr(reference,name),parameters)==expected,key


@pytest.mark.parametrize('surface',[rank,generic])
@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
@pytest.mark.parametrize('output',[None,np.float32,np.float64])
@pytest.mark.parametrize('pattern',['ties','asymmetric','zero-window','center-below','center-above'])
def test_independent_definition_on_public_exports(surface,name,ndim,dtype,output,pattern,monkeypatch):
    monkeypatch.setattr(contracts,'reference',surface)
    contracts.test_independent_histogram_intensity_definition(name,ndim,dtype,output,pattern)


@pytest.mark.parametrize('surface',[rank,generic])
@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('output',[np.float32,np.float64])
@pytest.mark.parametrize('direction',['below','above'])
def test_wide_signed_quotient_and_modular_integer_store(surface,ndim,output,direction,monkeypatch):
    monkeypatch.setattr(contracts,'reference',surface)
    contracts.test_autolevel_signed_wide_quotient_is_observable_in_float_output(ndim,output,direction)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('stage',['image','mask','footprint'])
@pytest.mark.parametrize('error_policy',['warn','raise','call'])
@pytest.mark.parametrize('signal',[False,True])
def test_inherited_materialized_dtype_policy(name,dtype,stage,error_policy,signal,capfd):
    assert policy.observe(rank,name,dtype,stage,error_policy,'none',signal,capfd)==policy.observe(reference,name,dtype,stage,error_policy,'none',signal,capfd)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('stage',['image','mask','footprint'])
@pytest.mark.parametrize('error_policy',['call','log'])
@pytest.mark.parametrize('action',['fill','shape','reentrant'])
def test_inherited_stateful_callbacks(name,dtype,stage,error_policy,action,capfd):
    assert policy.observe(rank,name,dtype,stage,error_policy,action,True,capfd)==policy.observe(reference,name,dtype,stage,error_policy,action,True,capfd)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('stage',['image','mask'])
@pytest.mark.parametrize('hook',['showwarning','stream'])
@pytest.mark.parametrize('action',['high','shape','dtype'])
def test_integer_warning_capture_on_each_new_operation(name,stage,hook,action):
    ndim=3 if action=='dtype' else 2
    assert integer.observe(rank,name,stage,np.int16,ndim,hook,action)==integer.observe(reference,name,stage,np.int16,ndim,hook,action)


@pytest.mark.parametrize('surface',[rank,generic])
@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('fallback',[False,True])
def test_never_forward_original_rank_algorithm(surface,name,fallback,monkeypatch):
    image=np.arange(35,dtype=np.uint8).reshape(5,7);fp=np.ones((3,3),np.uint8)
    expected=getattr(reference,name)(image,fp)
    def forbidden(*args,**kwargs):raise AssertionError('upstream rank algorithm forwarding')
    for module in (reference,reference_generic):
        for operation in NAMES:monkeypatch.setattr(module,operation,forbidden)
    xx.set_fallback_enabled(fallback)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always');actual=getattr(surface,name)(image,fp)
    assert not any(issubclass(w.category,xx.FallbackWarning) for w in observed)
    assert actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('name',NAMES)
def test_missing_linked_operation_rejects_before_preparation(name,monkeypatch):
    class OldCore:rank_histogram_capabilities=511
    monkeypatch.setattr(_rank,'_core',OldCore())
    class NoInspection:
        @property
        def dtype(self):raise AssertionError('unavailable native operation inspected caller')
        def __array__(self,*args,**kwargs):raise AssertionError('unavailable native operation converted caller')
    with pytest.raises(NotImplementedError,match='rank-histogram-v3'):
        getattr(rank,name)(NoInspection(),NoInspection())


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('fallback',[False,True])
def test_r3_prepared_overlap_stays_explicitly_excluded(name,fallback):
    xx.set_fallback_enabled(fallback)
    image=np.arange(35,dtype=np.uint8).reshape(5,7);before=image.copy()
    with pytest.raises(NotImplementedError,match='Overlapping buffers'):
        getattr(rank,name)(image,np.ones((3,3),np.uint8),out=image.view())
    np.testing.assert_array_equal(image,before)


@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
def test_native_footprint_rank_label_pipeline_worker_bits(ndim,dtype):
    shape=(513,519) if ndim==2 else (3,299,301)
    image=((np.arange(np.prod(shape))*37)%71).astype(dtype).reshape(shape)
    from skimage import morphology,measure
    def pipeline(fpmod,rankmod,labelmod):
        fp=fpmod.footprint_rectangle((3,)*ndim)
        equal=rankmod.equalize(image,fp)
        scaled=rankmod.autolevel(equal,fp)
        contrast=rankmod.enhance_contrast(scaled,fp)
        threshold=rankmod.threshold(contrast,fp)
        return equal,scaled,contrast,threshold,labelmod.label(threshold)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');expected=pipeline(morphology,reference,measure)
        one=pipeline(xx.morphology,rank,xx.measure)
        old=xx.get_num_threads()
        try:
            xx.set_num_threads(1);serial=pipeline(xx.morphology,rank,xx.measure)
            xx.set_num_threads(4);parallel=pipeline(xx.morphology,rank,xx.measure)
        finally:xx.set_num_threads(old)
    for actual_a,actual_b,actual_c,original in zip(one,serial,parallel,expected):
        assert actual_a.dtype==actual_b.dtype==actual_c.dtype==original.dtype
        assert actual_a.tobytes()==actual_b.tobytes()==actual_c.tobytes()==original.tobytes()


def test_public_intensity_metadata_is_separate_from_measurements():
    record=json.loads((Path(__file__).parents[1]/'coverage/contracts/rank-intensity-public.json').read_text())
    assert record['pinned_skimage']=='0.26.0' and record['operation_ids']==dict(zip(NAMES,range(9,13)))
    assert record['frozen_cases']==1012 and record['independent_definition_oracles']==248
    assert record['public_test']=='tests/test_rank_intensity_public.py'
    assert record['performance_evidence'] is None
    assert hashlib.sha256(contracts.MATRIX.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==record['reference_contract_sha256_posix_lf']


def test_four_new_public_functions_without_upstream_dependency():
    code='''
import importlib.abc,sys
class NoSkimage(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='skimage' or fullname.startswith('skimage.'):
            raise ModuleNotFoundError('upstream unavailable',name=fullname)
sys.meta_path.insert(0,NoSkimage())
import numpy as np
import skimagexx as xx
from skimagexx.filters import rank
from skimagexx.filters.rank import generic
xx.set_fallback_enabled(False)
assert xx.get_num_threads()==1
for workers in (1,4):
    xx.set_num_threads(workers)
    for name in ('autolevel','equalize','enhance_contrast','threshold'):
        assert getattr(rank,name) is getattr(generic,name)
        image=np.arange(35,dtype=np.uint8).reshape(5,7)
        assert getattr(rank,name)(image,np.ones((3,3),np.uint8)).shape==image.shape
assert not any(n=='skimage' or n.startswith('skimage.') for n in sys.modules)
'''
    result=subprocess.run([sys.executable,'-c',code],env=os.environ.copy(),capture_output=True,text=True)
    assert result.returncode==0,(result.stdout,result.stderr)
