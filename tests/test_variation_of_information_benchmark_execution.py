"""Pure/inert authority and actual sampler-entry refusal; zero target bodies."""
import ast
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/variation-of-information-benchmark-execution-v5.json'
DATA=json.loads((ROOT/CONTEXT).read_bytes())
H=lambda raw:hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()


def load(path):
    spec=importlib.util.spec_from_file_location('vi_benchmark_pure_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def shared_root():
    # Final integration uses ROOT. This private source packet records missing
    # dependencies explicitly and tests their exact retained frozen sources.
    root=ROOT if(ROOT/'benchmarks/mark_boundaries_execution.py').is_file()else Path(os.environ['VI_BENCH_COMMON_ROOT'])
    assert H((root/'benchmarks/mark_boundaries_execution.py').read_bytes())==DATA['common_boundary_canonical_sha256']
    assert H((root/'benchmarks/next_util_execution.py').read_bytes())==DATA['reused_executor_canonical_sha256']
    assert H((root/'tests/test_mark_boundaries_benchmark_execution.py').read_bytes())==DATA['metadata_fixture_canonical_sha256']
    return root


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks'))
    return load(ROOT/'benchmarks/variation_of_information_consumer.py'),load(ROOT/'benchmarks/variation_of_information_execution.py')


def source_overlay(tmp_path):
    root=tmp_path/'controller';root.mkdir();external=shared_root()
    pins={**DATA['source_files_canonical_sha256'],**DATA['FENV_dependency']['source_files_raw_sha256']}
    for name,digest in pins.items():
        choices=[ROOT/name,external/name];source=next(p for p in choices if p.is_file()and(H(p.read_bytes())==digest or hashlib.sha256(p.read_bytes()).hexdigest()==digest))
        out=root/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(source.read_bytes())
    for name in(CONTEXT,'CMakeLists.txt'):
        source=ROOT/name if name==CONTEXT else external/name
        out=root/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(source.read_bytes())
    return root


def synthetic(tmp_path,monkeypatch,fault='none'):
    root=source_overlay(tmp_path)
    # Reuse the admitted MARK metadata factory, not another binding framework.
    fixtures=runpy.run_path(str(shared_root()/'tests/test_mark_boundaries_benchmark_execution.py'))
    fn=fixtures['synthetic'];fn.__globals__['ROOT']=root
    path,b,c=fn(tmp_path,monkeypatch,fault,root/CONTEXT)
    # The inherited synthetic MARK factory has one named header. VI additionally
    # requires its existing count header; these remain metadata-only stand-ins.
    if fault!='SDK':
        b['SDK_members']['include/skimagexx/label_pair_metrics.hpp']='f'*64
        runtime=Path(b['runtime_path']);payload=json.loads(runtime.read_bytes())
        payload['runtime']['sdk_sha256']=b['SDK_members'];runtime.write_text(json.dumps(payload))
        if fault!='runtime':b['runtime_sha256']=hashlib.sha256(runtime.read_bytes()).hexdigest()
        path.write_text(json.dumps(b));go=Path(b['root_go_path']);value=json.loads(go.read_bytes())
        if fault!='go':value['binding_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        go.write_text(json.dumps(value))
    return root,path,b,c


def test_exact_cases_bounds_source_method_and_unbound(modules):
    consumer,execution=modules
    old=json.loads((ROOT/'coverage/contracts/variation-of-information-benchmark-v1.json').read_bytes())
    assert DATA['cases']==old['cases']==consumer.case_specs()and len(DATA['cases'])==4
    assert all(v is None for v in DATA['unbound'].values())
    for spec in DATA['cases']:
        bounds=consumer.case_bounds(spec);assert bounds['pixels']<=65536 and bounds['maximum_logical_NNZ']<=4096
    for name,pin in DATA['source_files_canonical_sha256'].items():
        if name in('src/label_pair_metrics.cpp','src/execution.hpp','benchmarks/next_util_execution.py','benchmarks/mark_boundaries_execution.py'):path=shared_root()/name
        else:path=ROOT/name
        assert H(path.read_bytes())==pin,name
    common=load(shared_root()/'benchmarks/next_util_execution.py')
    measure=runpy.run_path(str(ROOT/'tools/montage_sustained.py'))['measure']
    import inspect
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest()==DATA['method']['method_sha256']
    assert(DATA['method']['repeat'],DATA['method']['target_ms'],DATA['method']['warmup_min_ms'],DATA['method']['threads'],DATA['method']['pass_index'])==(7,100,250,1,0)
    assert DATA['raw_samples']==56 and DATA['warm_groups']==8


@pytest.mark.parametrize('cached',(False,True))
def test_real_CLI_unbound_refuses_even_cached_science(modules,monkeypatch,cached):
    consumer,execution=modules;seen=[];old=builtins.__import__
    if cached:
        for name in('numpy','skimage','skimagexx'):monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    def guarded(name,*a,**k):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):seen.append(name);raise AssertionError('SCIENCE_REQUEST')
        return old(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded);monkeypatch.setattr(sys,'argv',['VI-consumer'])
    with pytest.raises(SystemExit,match='VI_UNBOUND'):consumer.main()
    assert not seen


@pytest.mark.parametrize('fault',('none','extra-cases','extra-method','core','runtime','context','revision','producer','git-root','root-admission','root-stager','SDK','route','interpreter','helper-digest','helper-receipt','helper-mode','helper-legacy','CMake','go','expired','go-budget','git-dirty'))
def test_actual_shared_binding_authority_metadata_only(modules,tmp_path,monkeypatch,fault):
    root,path,b,c=synthetic(tmp_path,monkeypatch,fault)
    shared=load(shared_root()/'benchmarks/mark_boundaries_execution.py')
    if fault=='none':
        result=shared.validate_binding(path,root=root,context_path=CONTEXT)
        assert result[0]==b and result[1]==c
    else:
        with pytest.raises(AssertionError):shared.validate_binding(path,root=root,context_path=CONTEXT)


def test_positive_inert_bound_real_CLI_reaches_REQUEST_not_body(modules,tmp_path,monkeypatch):
    consumer,execution=modules;root,path,b,c=synthetic(tmp_path,monkeypatch)
    monkeypatch.setattr(execution,'ROOT',root);seen=[];previous=builtins.__import__
    def refuse(name,*a,**k):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):seen.append(name);raise RuntimeError('INERT_IMPORT_REQUEST_STOP')
        return previous(name,*a,**k)
    monkeypatch.setattr(sys,'argv',['VI','--root-binding',str(path),'--first-cohort','--case',DATA['cases'][0]['id'],'--output',str(tmp_path/'results/never.json')])
    monkeypatch.setattr(builtins,'__import__',refuse)
    with pytest.raises(RuntimeError,match='INERT_IMPORT_REQUEST_STOP'):execution.execute(consumer.__dict__)
    assert seen==['numpy']and not(tmp_path/'results/never.batches.jsonl').exists()


@pytest.mark.parametrize('fault',('none','missing','cases','scope','cohort','binding','deadline'))
def test_separate_VI_measurement_authority_before_bodies(modules,tmp_path,monkeypatch,fault):
    root,path,b,c=synthetic(tmp_path,monkeypatch);pin=hashlib.sha256(path.read_bytes()).hexdigest()
    shared=load(shared_root()/'benchmarks/mark_boundaries_execution.py')
    done=dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',route='vi',source=b['source_revision'],root_binding_sha256=pin,counts=dict(original=1,native=1),sampler_body_calls=0,timed_batches=0,outer_state_restored=True,poststop_source_authority_unchanged=True)
    if fault=='cohort':done['timed_batches']=1
    cohort=tmp_path/'INERT-cohort.json';cohort.write_text(json.dumps(done))
    m=dict(schema='next-metrics-measurement-v1',status='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON',prepared_source=b['prepared_source'],producer=b['producer'],core_sha256=b['core_sha256'],runtime_sha256=b['runtime_sha256'],families={c['report_family']:dict(contract=CONTEXT,cases=4,case_ids=[s['id']for s in c['cases']],untimed_calls_per_case=1)},execution_scope=dict(workers=1,repeat=7,target_ms=100,warmup_min_ms=250,single_host_same_installed_runtime=True,no_automatic_retry=True,retain_all_losses=True),completed_first_cohort=dict(path=str(cohort),sha256=hashlib.sha256(cohort.read_bytes()).hexdigest(),binding_sha256=pin),deadline_utc='2099-01-01T00:00:00+00:00')
    if fault=='cases':m['families'][c['report_family']]['cases']=3
    elif fault=='scope':m['execution_scope']['workers']=4
    elif fault=='binding':m['completed_first_cohort']['binding_sha256']='0'*64
    elif fault=='deadline':m['deadline_utc']='2000-01-01T00:00:00+00:00'
    target=tmp_path/'INERT-measurement.json';target.write_text(json.dumps(m))
    if fault=='none':assert shared.validate_measurement(target,b,c,pin,context_path=CONTEXT)[0]==hashlib.sha256(target.read_bytes()).hexdigest()
    else:
        with pytest.raises(AssertionError):shared.validate_measurement(None if fault=='missing'else target,b,c,pin,context_path=CONTEXT)


@pytest.mark.parametrize('kind',('absent','non-stopping','replaces-at-entry','replaces-on-any-event','admitted-stopping'))
def test_actual_reused_sampler_entry_zero_body(modules,tmp_path,kind):
    shared=load(shared_root()/'benchmarks/mark_boundaries_execution.py');common=shared.support(shared_root(),DATA)
    measure=runpy.run_path(str(ROOT/'tools/montage_sustained.py'))['measure'];body=[];calls=[];oldprofile,oldtrace=sys.getprofile(),sys.gettrace()
    class Stopped(BaseException):pass
    def observer(frame,event,value):
        if kind=='replaces-on-any-event':sys.setprofile(lambda*a:None)
        if event=='call'and frame.f_code is measure.__code__:
            assert frame.f_locals['pass_index']==frame.f_locals['case_index']==0
            assert frame.f_locals['case_id']==DATA['cases'][0]['id']
            if kind=='replaces-at-entry':sys.setprofile(lambda*a:None)
            if kind=='admitted-stopping':raise Stopped('EXTERNAL_STOP')
    def trace(frame,event,value):
        if frame.f_code is measure.__code__ and event=='line':body.append(frame.f_lineno)
        return trace
    cli=types.SimpleNamespace(first_cohort=True,output=tmp_path/'never.json')
    try:
        sys.settrace(trace);sys.setprofile(None if kind=='absent'else observer)
        with pytest.raises(Stopped if kind=='admitted-stopping'else AssertionError):
            shared.sample_entry(measure,lambda:calls.append('original'),lambda:calls.append('native'),cli,0,DATA['cases'][0],'2099-01-01T00:00:00+00:00',common)
    finally:sys.setprofile(oldprofile);sys.settrace(oldtrace)
    assert body==calls==[]and not cli.output.with_suffix('.batches.jsonl').exists()
    assert sys.getprofile()is oldprofile and sys.gettrace()is oldtrace


@pytest.mark.parametrize('own_init',(False,True))
def test_candidate_provider_scope_is_local_and_restored(modules,own_init):
    execution=modules[1];events=[]
    class Base:
        def __init__(self,arg,*a,**kw):events.append(arg);self.arg=arg
    class CSR(Base):pass
    if own_init:CSR.__init__=Base.__init__
    before=dict(CSR.__dict__);initialize=CSR.__init__
    log=lambda*a:None;np=types.SimpleNamespace(log=log,log2=log);sparse=types.SimpleNamespace(csr_array=CSR)
    with execution.candidate_providers(np,sparse):
        assert sparse.csr_array is CSR and isinstance(CSR((1,2,3)),sparse.csr_array)
        with pytest.raises(AssertionError,match='coordinate/coalescing'):sparse.csr_array((1,(2,3)))
        with pytest.raises(AssertionError):np.log(2)
        with pytest.raises(AssertionError):np.log2(2)
    assert sparse.csr_array is CSR and CSR.__init__ is initialize and dict(CSR.__dict__)==before
    assert np.log is np.log2 is log and events==[(1,2,3)]
    with pytest.raises(RuntimeError,match='INERT_FAILURE'):
        with execution.candidate_providers(np,sparse):raise RuntimeError('INERT_FAILURE')
    assert dict(CSR.__dict__)==before and np.log is np.log2 is log


def test_both_installed_SDK_header_dependencies_required(modules):
    fn=modules[1].require_sdk_dependencies
    members={'include/skimagexx/variation_of_information.hpp':'a','include/skimagexx/label_pair_metrics.hpp':'b'}
    fn(dict(SDK_members=members))
    for name in members:
        with pytest.raises(AssertionError):fn(dict(SDK_members={k:v for k,v in members.items()if k!=name}))


def test_exact_native_preflight_and_uninstrumented_public_sampler_source(modules):
    consumer,execution=modules
    assert consumer.native_requirements(DATA['cases'][0])==DATA['strict_preflight']['typed_native_C_returns']
    source=Path(execution.__file__).read_text();tree=ast.parse(source)
    assert 'suffix.append(frame.f_locals.get(\'native_result\')is not None)'in source
    assert 'accepted==suffix==[True]'in source and 'expected.tobytes()==actual.tobytes()'in source
    assert 'reference_before==native_before and reference_FP==native_FP'in source
    assert 'shared.sample_entry(measure,lambda:reference(*args,**options),lambda:native(*args,**options)'in source
    assert 'finally:sys.setprofile(prior);restore(caller)'in source and 'finally:restore(sample_state)'in source
    assert 'old_fallback' in source and 'sys.path[:]=oldpath'in source
    assert not any(isinstance(n,ast.FunctionDef)and n.name=='measure'for n in ast.walk(tree))


@pytest.mark.parametrize('matrix',(False,True))
def test_actual_frontend_CSR_identity_reaches_inert_native_suffix(modules,matrix):
    # Execute only unchanged Python adapter definitions against inert metadata.
    # No NumPy/SciPy import, provider arithmetic or native body is available.
    events=[];execution=modules[1]
    class Buffer:
        def __init__(self,label):
            self.label=label;self.size=4;self.flags=types.SimpleNamespace(aligned=True)
        def astype(self,dtype,copy=False):events.append(('index_cast',self.label,copy));return self
    data,indices,indptr=(Buffer(n)for n in('data','indices','indptr'))
    class CSR:
        def __init__(self,arg,*,shape,copy=False):
            events.append(('structural',len(arg)));self.data,self.indices,self.indptr=arg;self.shape=shape
            self.dtype=types.SimpleNamespace(kind='f',itemsize=8,isnative=True)
    class Matrix:
        def __init__(self,array):self.__dict__.update(array.__dict__)
    sparse=types.SimpleNamespace(csr_array=CSR,csr_matrix=Matrix)
    class Image:
        shape=(2,2)
    first,second=Image(),Image();result=object()
    class Core:
        def label_pair_check_storage(self,buffer):events.append(('storage',buffer.label))
        def get_num_threads(self):return 1
        def variation_of_information_csr(self,*args):
            assert args==(data,indices,indptr,2,2,1);events.append('native_entropy_suffix');return result
    np=types.SimpleNamespace(ndarray=Image,float64='f8',int64='i8',ascontiguousarray=lambda a,dtype:a,
                             log=lambda*a:None,log2=lambda*a:None)
    scope=dict(np=np,sparse=sparse,_core=Core(),get_index_dtype=lambda *a,**kw:'index-dtype')
    def definitions(path,names):
        tree=ast.parse((ROOT/path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
        assert {n.name for n in nodes}==set(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),scope)
    definitions('python/skimagexx/_contingency_table.py',('_structural_csr',))
    def accepted_counts(a,b,**kw):
        assert (a,b)==(first,second)and kw==dict(ignore_labels=(),normalize=True)
        native_result=((2,2),indptr,indices,data);events.append('inert_counts_native_result_non_none')
        table=scope['_structural_csr'](native_result,a,b)
        return Matrix(table)if matrix else table
    scope['contingency_table']=accepted_counts
    definitions('python/skimagexx/_variation_of_information.py',('_aligned','variation_of_information'))
    before=dict(CSR.__dict__)
    with execution.candidate_providers(np,sparse):
        assert sparse.csr_array is CSR and sparse.csr_matrix is Matrix
        assert scope['variation_of_information'](first,second)is result
    assert dict(CSR.__dict__)==before
    assert events.count('native_entropy_suffix')==events.count('inert_counts_native_result_non_none')==1
    assert events.count(('structural',3))==1


def test_provider_guard_successor_preserves_methods_and_authentication_history():
    previous=DATA['predecessor_context'];oldraw=(ROOT/previous['path']).read_bytes()
    assert hashlib.sha256(oldraw).hexdigest()==previous['raw_sha256'];old=json.loads(oldraw)
    assert DATA['cases']==old['cases']and DATA['method']==old['method']and DATA['strict_preflight']==old['strict_preflight']
    for name,row in DATA['provider_guard_predecessors'].items():
        raw=(ROOT/name).read_bytes().replace(b'\r\n',b'\n');assert hashlib.sha256(raw).hexdigest()==row['current_canonical_sha256']
        for fragment in row['inverse_fragments']:
            current=fragment['current'].encode();assert raw.count(current)==1;raw=raw.replace(current,fragment['predecessor'].encode(),1)
        assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_sha256']
    assert all(v is None for v in DATA['unbound'].values())
