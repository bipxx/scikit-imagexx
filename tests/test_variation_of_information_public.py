"""Legacy scalar saved words and distinct current public provider controls."""
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types
import numpy as np
import pytest
from label_pair_metrics_contract import cases
from label_pair_metrics_observer import observe
ROOT=Path(__file__).resolve().parents[1]
SPECS=[s for s in cases()if s['api']=='metrics.variation_of_information']


@pytest.fixture(scope='module')
def target():
    if not os.environ.get('VI_COMPONENT_DIR'):
        import skimagexx as xx
        module=importlib.import_module('skimagexx._variation_of_information')
        core=xx._core
        assert module._core is core and xx.metrics.variation_of_information is module.variation_of_information
        assert inspect.isbuiltin(core.variation_of_information_csr)and core.__name__=='skimagexx._core'
        assert core.get_num_threads()==1
        assert core.has_avx2()==(os.environ.get('SKIMAGEXX_DISABLE_SIMD')!='1')
        return module,core
    stage=Path(os.environ['VI_COMPONENT_DIR']);path=next(stage.glob('_core*.so'))
    package=types.ModuleType('_vi_family');package.__path__=[str(ROOT/'python/skimagexx')]
    sys.modules[package.__name__]=package
    spec=importlib.util.spec_from_file_location(package.__name__+'._core',path)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    sys.modules[spec.name]=core;package._core=core
    module=importlib.import_module(package.__name__+'._variation_of_information')
    assert inspect.isbuiltin(core.variation_of_information_csr)
    assert inspect.isbuiltin(core.label_pair_counts_preweighted)
    assert str(inspect.signature(module.variation_of_information))=='(image0=None, image1=None, *, table=None, ignore_labels=())'
    assert core.get_num_threads()==1
    assert core.has_avx2()==(os.environ.get('SKIMAGEXX_DISABLE_SIMD')!='1')
    return module,core


@pytest.fixture(autouse=True,params=[1,4])
def workers(target,request):
    core=target[1];before=core.get_num_threads();assert before==1
    envelope=None
    if not os.environ.get('VI_COMPONENT_DIR'):
        import ctypes
        from util_views_fenv import backend,FEnv
        libc,helper=backend();saved=FEnv();assert libc.fegetenv(ctypes.byref(saved))==0
        envelope=(ctypes,libc,helper,FEnv,saved)
    core.set_num_threads(request.param)
    try:yield request.param
    finally:
        core.set_num_threads(before)
        if envelope is not None:
            ctypes,libc,helper,FEnv,saved=envelope
            assert helper.reference_fenv_restore(ctypes.byref(saved),32)==0
            actual=FEnv();assert libc.fegetenv(ctypes.byref(actual))==0 and bytes(actual)==bytes(saved)
            output=os.environ.get('VI_CALLER_RECORD_DIR')
            if output:
                import hashlib
                p=Path(output);p.mkdir(exist_ok=True)
                (p/(hashlib.sha256(request.node.nodeid.encode()).hexdigest()+'.json')).write_text(json.dumps(dict(node=request.node.nodeid,workers=request.param,scope='Caller restoration only; old VI rows contain no original hardware-FP witness',saved=bytes(saved).hex(),actual=bytes(actual).hex(),exact32=True))+'\n')


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_legacy_scalar_saved_original_fields(target,spec,workers):
    version=np.__version__;assert version in('1.24.4','2.5.4')
    data=json.loads((ROOT/'coverage/contracts/label-pair-metrics-v1.json').read_bytes())
    expected=next(r for r in data['original_observations'][version]if r['id']==spec['id'])
    # These immutable CPU-specific words qualify the unchanged scalar SDK
    # gateway. Current public providers have independent same-runtime pairs.
    from test_variation_of_information_large_saved import legacy_scalar_gateway
    actual=observe(legacy_scalar_gateway(*target).variation_of_information,spec,np)
    if os.environ.get('VI_RESULT_DIR'):
        path=Path(os.environ['VI_RESULT_DIR']);path.mkdir(exist_ok=True)
        (path/(spec['id'].split('/')[-1]+'-workers'+str(workers)+'.json')).write_text(json.dumps(actual,indent=2)+'\n')
    # Includes raw f64 bytes, signed NaNs/zeros, class/dtype/layout/ownership,
    # all error text/type, callbacks/warnings, input/table/cache mutation/aliases.
    assert actual==expected
    assert target[1].get_num_threads()==workers


def test_owned_entropy_uses_only_explicit_log2_and_matmul(target,monkeypatch):
    import scipy.sparse as sparse
    original=np.log2,np.matmul;batches=[]
    def log2(values):
        assert type(values)is np.ndarray and values.dtype==np.dtype('float64')
        assert values.ndim==1 and values.flags.owndata and values.flags.c_contiguous
        batches.append(('log2',values.shape))
        return original[0](values)
    def matmul(left,right):
        for value in(left,right):
            assert type(value)is np.ndarray and value.dtype==np.dtype('float64')
            assert value.ndim==1 and value.flags.owndata and value.flags.c_contiguous
        assert left.shape==right.shape
        batches.append(('matmul',left.shape))
        result=original[1](left,right);assert type(result)is np.float64
        return result
    def forbidden(*a,**k):raise AssertionError('Other entropy arithmetic provider forbidden')
    monkeypatch.setattr(np,'log2',log2);monkeypatch.setattr(np,'matmul',matmul);monkeypatch.setattr(np,'log',forbidden)
    monkeypatch.setattr(np,'sum',forbidden)
    for cls in(sparse.csr_matrix,sparse.csr_array):
        monkeypatch.setattr(cls,'__matmul__',forbidden);monkeypatch.setattr(cls,'sum',forbidden)
    a=np.array([[0,1,1],[2,2,2]],dtype=np.int16)
    actual=target[0].variation_of_information(a,a)
    assert actual.tobytes()==bytes(16)and actual.flags.owndata
    assert [name for name,shape in batches]==['log2','matmul','log2','matmul']
    assert batches[0]==batches[2]and batches[1]==batches[3]
    primary=RuntimeError('VI_PROVIDER_PRIMARY')
    def refused(values):raise primary
    monkeypatch.setattr(np,'log2',refused)
    with pytest.raises(RuntimeError)as error:target[0].variation_of_information(a,a)
    assert error.value is primary


@pytest.mark.parametrize('phase',range(4))
def test_current_provider_primary_stops_later_callbacks(target,monkeypatch,phase):
    originals=np.log2,np.matmul;events=[];primary=RuntimeError('VI_PROVIDER_PRIMARY_'+str(phase))
    def call(index,*args):
        current=len(events);events.append(index)
        if current==phase:raise primary
        return originals[index](*args)
    monkeypatch.setattr(np,'log2',lambda value:call(0,value))
    monkeypatch.setattr(np,'matmul',lambda left,right:call(1,left,right))
    a=np.array([[0,1],[1,0]],dtype=np.int64)
    with pytest.raises(RuntimeError)as error:target[0].variation_of_information(a,a)
    assert error.value is primary and events==[0,1,0,1][:phase+1]


@pytest.mark.parametrize('kind',('python-float','float32','array','subclass'))
def test_current_matmul_rejects_nonplain_scalar_transport(target,monkeypatch,kind):
    class Derived(np.float64):pass
    wrong={'python-float':lambda:0.,'float32':lambda:np.float32(0),
        'array':lambda:np.array(0.,dtype=np.float64),'subclass':lambda:Derived(0)}
    calls=[]
    def bad(left,right):calls.append((left.shape,right.shape));return wrong[kind]()
    monkeypatch.setattr(np,'matmul',bad)
    a=np.array([[0,1],[1,0]],dtype=np.int64)
    with pytest.raises(TypeError,match='plain native float64 scalar'):target[0].variation_of_information(a,a)
    assert len(calls)==1


def test_plain_CSR_buffers_and_caches_are_not_mutated(target):
    import scipy.sparse as sparse
    data=np.array([1.,2.,3.,0.]);indices=np.array([1,0,1,2],dtype=np.int32);indptr=np.array([0,2,4],dtype=np.int32)
    table=sparse.csr_array((data,indices,indptr),shape=(2,3),copy=False)
    before=(data.tobytes(),indices.tobytes(),indptr.tobytes(),dict(table.__dict__))
    a=np.zeros((2,3),dtype=np.int16);result=target[0].variation_of_information(a,a,table=table)
    assert (data.tobytes(),indices.tobytes(),indptr.tobytes())==before[:3]
    assert set(table.__dict__)==set(before[3])
    assert table.__dict__.get('_has_sorted_indices')is None and table.__dict__.get('_has_canonical_format')is None
    assert result.flags.owndata and not np.shares_memory(result,data)


@pytest.mark.parametrize('fault',('dtype','shape','index','rowptr'))
def test_native_typed_boundary_refuses_malformed_buffers(target,fault):
    core=target[1];data=np.array([1.]);indices=np.array([0],dtype=np.int64);indptr=np.array([0,1],dtype=np.int64)
    if fault=='dtype':data=data.astype(np.float32)
    elif fault=='shape':data=data.reshape(1,1)
    elif fault=='index':indices[0]=1
    elif fault=='rowptr':indptr[0]=1
    assert core.variation_of_information_csr(data,indices,indptr,1,1,core.get_num_threads())is None
