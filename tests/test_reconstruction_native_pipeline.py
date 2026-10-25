"""Actual private builtin reconstruction pipeline; never public admission."""
import hashlib,importlib.util,inspect,json,os,sys,types
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('reconstruction_fixed_contract',ROOT/'tests/test_reconstruction_contracts.py');CONTRACT=importlib.util.module_from_spec(sp);sp.loader.exec_module(CONTRACT)

@pytest.fixture(scope='module')
def backend():
    path=os.environ.get('SKIMAGEXX_RECONSTRUCTION_CORE')
    if not path:pytest.skip('private reconstruction core explicitly required; no native/public claim')
    path=Path(path).resolve();assert path.is_file()
    expected=os.environ['SKIMAGEXX_RECONSTRUCTION_CORE_SHA256'];assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    spec=importlib.util.spec_from_file_location('_reconstruction_core',path);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    assert core.reconstruction_interface=='reconstruction-v1' and type(core.reconstruction_capabilities) is int and core.reconstruction_capabilities==3
    assert core.get_num_threads()==1 and core.reconstruction_parallel_flood is False
    for name in ('reconstruction_run','reconstruction_ordered','reconstruction_negative_zero'):
        f=getattr(core,name);assert inspect.isbuiltin(f) and f.__module__==core.__name__ and f.__name__==name
    package=types.ModuleType('_reconstruction_private');package.__path__=[str(ROOT/'python/skimagexx')];package._core=core;sys.modules[package.__name__]=package
    source=ROOT/'python/skimagexx/_reconstruction.py';s=importlib.util.spec_from_file_location(package.__name__+'._reconstruction',source);wrapper=importlib.util.module_from_spec(s);s.loader.exec_module(wrapper)
    yield core,wrapper
    core.set_num_threads(1)

@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',CONTRACT.cases(),ids=lambda s:s['id'])
def test_actual_private_pipeline(backend,workers,spec):
    core,wrapper=backend
    if workers==4 and os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')=='1':
        with pytest.raises(ValueError,match='no OpenMP'):core.set_num_threads(4)
        assert core.get_num_threads()==1
        pytest.skip('portable worker4 explicitly rejected; no parallel parity claim')
    core.set_num_threads(workers);before=core.reconstruction_counts(True);attempts=[];previous=sys.getprofile();code=CONTRACT.ORIGINAL.__code__;rank_code=CONTRACT.SOURCE.rank_order.__code__
    def reject(frame,event,arg):
        if event=='call' and frame.f_code in (code,rank_code):attempts.append(frame.f_code.co_name);raise AssertionError('upstream reconstruction/rank_order forwarding forbidden')
    sys.setprofile(reject)
    try:actual=CONTRACT.observe(spec,wrapper.reconstruction)
    finally:sys.setprofile(previous)
    expected=json.loads(CONTRACT.MATRIX.read_bytes())['profiles'][CONTRACT.profile()]['observations'][spec['id']]
    assert CONTRACT.public_observation(actual)==CONTRACT.public_observation(expected)
    assert not attempts and actual['original_Cython_attempts']==actual['original_Cython_completed']==0
    counts=core.reconstruction_counts(False)
    if expected['original_Cython_completed']:
        assert counts['run']==dict(attempted=1,completed=1)
    else:assert counts['run']['completed']==0


def test_private_signatures_and_no_public_claim(backend):
    core,wrapper=backend
    assert str(inspect.signature(wrapper.reconstruction))==str(inspect.signature(CONTRACT.ORIGINAL))
    assert not hasattr(core,'native_functions') and core.get_num_threads() in (1,4)
    if os.environ.get('SKIMAGEXX_EXPECT_GIL_ASSERTIONS')=='1':assert core.gil_assertions is True

@pytest.mark.parametrize('method',('dilation','erosion'))
def test_actual_large_queue_and_workers(backend,method):
    import numpy as np
    core,wrapper=backend
    seed=np.zeros((257,259),np.uint16);mask=np.full(seed.shape,65535,np.uint16);seed[0,0]=60000
    if method=='erosion':seed=65535-seed;mask=65535-mask
    for workers in (1,4):
        if workers==4 and os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')=='1':continue
        core.set_num_threads(workers);core.reconstruction_counts(True);result=wrapper.reconstruction(seed,mask,method)
        assert np.all(result==(5535 if method=='erosion' else 60000)) and result.dtype==np.float64
        assert core.reconstruction_counts(False)['run']==dict(attempted=1,completed=1)
