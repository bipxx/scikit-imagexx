"""Inert current log2/matmul guards. No scientific/native/helper execution."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
POLICY='coverage/contracts/vi-public-matmul-provider-v1.json'
ADAPTER='benchmarks/vi_public_matmul_provider_execution.py'
CONTEXT='coverage/contracts/vi-public-matmul-provider-benchmark-v1.json'
ORDER=('numpy.log2','numpy.matmul','numpy.log2','numpy.matmul')
H=lambda raw:hashlib.sha256(raw).hexdigest()


def load(name):
    spec=importlib.util.spec_from_file_location('_vi_matmul_inert_'+Path(name).stem,ROOT/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_policy_is_explicit_unbound_and_literal_current():
    data=json.loads((ROOT/POLICY).read_bytes())
    assert data['execution_authority']is None
    assert not any(data[k]for k in('compiled','public_parity','SDK_parity','release_qualified','performance'))
    assert tuple(data['provider_order'])==ORDER
    assert data['SDK_legacy']=='Unchanged scalar-libm and log2-only ABI/graphs/golds'
    for name,pin in data['source_raw_sha256'].items():assert H((ROOT/name).read_bytes())==pin,name
    for name,pin in data['frozen_contracts_raw_sha256'].items():assert H((ROOT/name).read_bytes())==pin,name


def test_entire_legacy_CPP_and_header_prefixes_remain_exact():
    data=json.loads((ROOT/POLICY).read_bytes())
    text=(ROOT/'src/variation_of_information.cpp').read_text()
    marker='\nbool variation_of_information_compatible(VariationTableView a,VariationInformation& output,\n    VariationLog2Batch logarithm,VariationInnerProduct inner_product,int workers)'
    old=text[:text.index(marker)]+'} // namespace skimagexx\n'
    assert H(old.encode())==data['legacy_CPP_raw_sha256']
    text=(ROOT/'include/skimagexx/variation_of_information.hpp').read_text()
    old=text[:text.index('// Explicit dense inner-product provider.')]+ '}\n'
    assert H(old.encode())==data['legacy_header_raw_sha256']


def assert_native_current(text):
    start=text.index('VariationLog2Batch logarithm,VariationInnerProduct inner_product,int workers)')
    body=text[start:text.index('bool variation_of_information_compatible(const detail::LabelPairInput& input,',start)]
    required=('!logarithm.function||!inner_product.function','!environment()||!validate(a)',
        'VariationInformation result;result.captured_workers=snapshot;',
        'batch_entropy(a,ix,true,logarithm,row_entropy)',
        'inner_product.function(px.data(),row_entropy.data(),a.rows,',
        'batch_entropy(a,iy,false,logarithm,column_entropy)',
        'inner_product.function(column_entropy.data(),py.data(),a.columns,',
        'output=result;return true;')
    positions=[body.index(token)for token in required]
    assert positions==sorted(positions)
    assert body.count('inner_product.function(')==2
    assert 'for(auto& v:px)v=-v;'in body and 'for(auto& v:column_entropy)v=-v;'in body
    assert body.count('&result.values[')==2 and 'source_dot('not in body
    for bad in('catch(','fesetenv','feclearexcept','getenv(','#pragma omp','case_id','std::log2'):
        assert bad not in body
    assert body[body.index('VariationInformation result;'):].count('))return false;')==4


def test_current_native_source_order_publication_and_no_fallback():
    assert_native_current((ROOT/'src/variation_of_information.cpp').read_text())


@pytest.mark.parametrize('old,new',(('!logarithm.function||!inner_product.function','!logarithm.function'),('&result.values[0]','&output.values[0]'),('for(auto& v:px)v=-v;',''),('output=result;return true;','return true;')))
def test_native_source_control_rejects_partial_or_missing_provider(old,new):
    text=(ROOT/'src/variation_of_information.cpp').read_text()
    # Limit mutation to the additive host implementation, not legacy bodies.
    pos=text.index('VariationLog2Batch logarithm,VariationInnerProduct inner_product,int workers)')
    assert old in text[pos:]
    with pytest.raises((AssertionError,ValueError)):assert_native_current(text[:pos]+text[pos:].replace(old,new,1))


def test_binding_uses_public_GIL_buffer_transport_and_strict_scalar_type():
    text=(ROOT/'src/bind_variation_of_information.cpp').read_text()
    body=text[text.index('struct InnerProductProvider'):text.index('void bind_variation_of_information')]
    assert 'py::object callable;'in body
    assert body.index('py::gil_scoped_acquire acquire;')<body.index('py::array_t<double> a(count),b(count);')<body.index('PyObject_CallFunctionObjArgs(owner.callable.ptr(),a.ptr(),b.ptr(),nullptr)')
    assert 'if(!returned)throw py::error_already_set();'in body
    assert 'py::reinterpret_steal<py::object>(returned)'in body
    assert 'Py_TYPE(result.ptr()))!=scalar_type.ptr()'in body
    assert 'py::reinterpret_borrow<py::buffer>(result).request()'in body
    assert 'buffer.ndim!=0||buffer.size!=1||buffer.itemsize!=sizeof(double)'in body
    assert 'buffer.format!=py::format_descriptor<double>::format()'in body
    assert body.index('buffer.format!=')<body.index('std::memcpy(output,buffer.ptr,sizeof(double))')
    assert text.index('InnerProductProvider product{std::move(matmul_provider)}')<text.index('py::gil_scoped_release release;')
    assert 'if(!matmul_provider.is_none()&&log2_provider.is_none())'in text
    assert 'py::arg("log2_provider")=py::none(),py::arg("matmul_provider")=py::none()'in text
    assert 'else accepted=supplied'in text and 'variation_of_information(table,result,workers)'in text
    for bad in('PyUFuncObject','dlvsym','_get_strided_loop','NPY_DISABLE_CPU_FEATURES','PyFloat_AsDouble','forcecast','catch('):assert bad not in text


def frontend():
    tree=ast.parse((ROOT/'python/skimagexx/_variation_of_information.py').read_bytes())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    return compile(ast.fix_missing_locations(tree),'<current VI frontend>','exec')


def test_actual_frontend_passes_both_current_providers_and_preserves_primary():
    class Image:shape=(2,2)
    class CSR:
        dtype=SimpleNamespace(kind='f',itemsize=8,isnative=True);shape=(2,2)
        data=indices=indptr=SimpleNamespace(flags=SimpleNamespace(aligned=True))
    log=object();dot=object();result=object();calls=[];table=CSR()
    np=SimpleNamespace(ndarray=Image,float64='f8',int64='i8',log2=log,matmul=dot,ascontiguousarray=lambda v,dtype:v)
    class Core:
        def label_pair_check_storage(self,v):pass
        def get_num_threads(self):return 1
        def variation_of_information_csr(self,*args):calls.append(args);return result
    core=Core();env=dict(np=np,sparse=SimpleNamespace(csr_array=CSR,csr_matrix=type('Matrix',(),{})),_core=core,contingency_table=lambda*a,**k:table)
    exec(frontend(),env);fn=env['variation_of_information']
    assert str(inspect.signature(fn))=='(image0=None, image1=None, *, table=None, ignore_labels=())'
    assert fn(Image(),Image())is result and calls[-1][6:]==(log,dot)
    log2=object();dot2=object();np.log2=log2;np.matmul=dot2
    assert fn(Image(),Image(),table=table)is result and calls[-1][6:]==(log2,dot2)
    primary=RuntimeError('INERT_CURRENT_PROVIDER_PRIMARY')
    def fail(*args):raise primary
    core.variation_of_information_csr=fail
    with pytest.raises(RuntimeError)as error:fn(Image(),Image(),table=table)
    assert error.value is primary


class Vector:
    ndim=1
    dtype=SimpleNamespace(kind='f',itemsize=8,isnative=True,str='<f8')
    flags=SimpleNamespace(c_contiguous=True,owndata=True)
    def __init__(self,n=3):self.shape=(n,);self.size=n


class Scalar:
    shape=()
    dtype=Vector.dtype


def fake_providers():
    calls=[]
    def log(value):calls.append('log2');return value
    def dot(left,right):calls.append('matmul');return Scalar()
    np=SimpleNamespace(ndarray=Vector,float64=Scalar,log2=log,matmul=dot,log=log,sum=log,dot=dot)
    class Base:
        def __init__(self,arg,*a,**k):self.arg=arg
        def sum(self):return 'historical'
    class CSR(Base):pass
    class Matrix(Base):pass
    return np,SimpleNamespace(csr_array=CSR,csr_matrix=Matrix),calls


def invoke(np):
    np.log2(Vector());np.matmul(Vector(2),Vector(2));np.log2(Vector(0));np.matmul(Vector(4),Vector(4))


def test_four_actual_guard_events_and_all_global_identities_restore():
    module=load(ADAPTER);np,sparse,calls=fake_providers();old=np.log2,np.matmul,np.sum
    types=dict(sparse.csr_array.__dict__),dict(sparse.csr_matrix.__dict__)
    with module.candidate_providers(np,sparse)as events:
        invoke(np)
        assert tuple(e['provider']for e in events)==ORDER
        assert events[2]['shape']==[0]and events[1]['left_shape']==events[1]['right_shape']==[2]
        with pytest.raises(AssertionError):np.dot(Vector(),Vector())
        with pytest.raises(AssertionError):np.sum(Vector())
        with pytest.raises(AssertionError):sparse.csr_array((1,(2,3)))
        with pytest.raises(AssertionError):sparse.csr_array((1,2,3)).sum()
    assert (np.log2,np.matmul,np.sum)==old and calls==['log2','matmul','log2','matmul']
    assert types==(dict(sparse.csr_array.__dict__),dict(sparse.csr_matrix.__dict__))


@pytest.mark.parametrize('fault',('missing','order','many','rank','dtype','native','owned','shape','scalar-type'))
def test_guard_refuses_missing_reordered_or_bad_transport(fault):
    module=load(ADAPTER);np,sparse,calls=fake_providers();old=np.log2,np.matmul
    if fault=='scalar-type':np.matmul=lambda*a:object();old=np.log2,np.matmul
    with pytest.raises(AssertionError):
        with module.candidate_providers(np,sparse):
            if fault=='missing':return
            if fault=='order':np.matmul(Vector(),Vector());return
            value=Vector()
            if fault=='rank':value.ndim=2
            if fault=='dtype':value.dtype=SimpleNamespace(kind='f',itemsize=4,isnative=True)
            if fault=='native':value.dtype=SimpleNamespace(kind='f',itemsize=8,isnative=False)
            if fault=='owned':value.flags=SimpleNamespace(c_contiguous=True,owndata=False)
            np.log2(value)
            np.matmul(Vector(),Vector(2)if fault=='shape'else Vector())
            np.log2(Vector());np.matmul(Vector(),Vector())
            if fault=='many':np.log2(Vector())
    assert (np.log2,np.matmul)==old


@pytest.mark.parametrize('phase',range(4))
def test_actual_guard_stops_first_primary_and_restores_current_handles(phase):
    module=load(ADAPTER);np,sparse,calls=fake_providers();primary=RuntimeError('INERT_PHASE_'+str(phase));events=[]
    def call(name,*a):
        index=len(events);events.append(name)
        if index==phase:raise primary
        return Vector()if name=='log2'else Scalar()
    np.log2=lambda*a:call('log2',*a);np.matmul=lambda*a:call('matmul',*a);old=np.log2,np.matmul
    with pytest.raises(RuntimeError)as error:
        with module.candidate_providers(np,sparse)as retained:invoke(np)
    assert error.value is primary and events==['log2','matmul','log2','matmul'][:phase+1]
    assert len(retained)==phase+1 and (np.log2,np.matmul)==old


def test_direct_current_counts_and_legacy_saved_scope_are_explicit():
    source=(ROOT/'tests/test_variation_of_information_current_counts_public.py').read_text()
    assert source.count('workers,np.log2,np.matmul)')==2
    old=(ROOT/'tests/test_variation_of_information_public.py').read_text()
    assert 'observe(legacy_scalar_gateway(*target).variation_of_information,spec,np)'in old
    assert 'assert actual==expected'in old
    current=(ROOT/'tests/test_vi_public_log2_provider_large.py').read_text()
    for token in('benchmarks/vi_public_matmul_provider_execution.py','len(batches)==4','expected.tobytes()==actual.tobytes()','np.matmul is true_matmul','restore(natural)'):
        assert token in current


def test_current_benchmark_preserves_exact_method_cases_and_execution_body(monkeypatch):
    context='coverage/contracts/vi-public-matmul-provider-benchmark-ci-h-extrema-v1.json'
    adapter=load(ADAPTER);current=load('benchmarks/vi_public_matmul_provider_execution_v2.py').executor();prior=load('benchmarks/variation_of_information_execution.py')
    assert current.CONTEXT==context and inspect.getsource(current.candidate_providers)==inspect.getsource(adapter.candidate_providers)
    assert current.execute.__code__.co_code==prior.execute.__code__.co_code
    data=json.loads((ROOT/context).read_bytes());old=json.loads((ROOT/'coverage/contracts/vi-public-log2-provider-benchmark-v2.json').read_bytes())
    for key in('cases','case_count','method','raw_samples','warm_groups','FENV_dependency','strict_preflight','timed_ownership','execution_authority','unbound'):
        assert data[key]==old[key],key
    assert data['raw_samples']==56 and data['method']['retain_all_losses']
    assert data['route']['driver']=='benchmarks/vi_public_matmul_provider_consumer_v2.py'
    assert data['route']['contract']==context and data['provider_policy']==POLICY
    assert data['source_binding']['mode']=='literal-current-canonical-LF'
    for name,pin in data['source_files_canonical_sha256'].items():assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin,name
    assert all(value is None for value in data['unbound'].values())and data['execution_authority']is None
    assert current.boundary(ROOT).current_context(ROOT,context)==data
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks'))
    consumer=load('benchmarks/vi_public_matmul_provider_consumer_v2.py')
    monkeypatch.setattr(sys,'argv',['current-matmul-consumer'])
    with pytest.raises(SystemExit,match='VI_UNBOUND'):current.execute(consumer.__dict__)


@pytest.mark.parametrize('fault',(None,'current','snapshot','prepared','old-contract'))
def test_current_binding_requires_literal_current_source_not_history(tmp_path,monkeypatch,fault):
    context_path='coverage/contracts/vi-public-matmul-provider-benchmark-ci-h-extrema-v1.json';adapter='benchmarks/vi_public_matmul_provider_execution_v2.py'
    current=load(adapter).executor();factory=load('tests/test_mark_boundaries_benchmark_execution.py')
    monkeypatch.setattr(factory,'source_fixture_bytes',lambda name,context_path:(ROOT/name).read_bytes())
    path,binding,context=factory.synthetic(tmp_path,monkeypatch,context_path=ROOT/context_path)
    binding['SDK_members']['include/skimagexx/label_pair_metrics.hpp']='f'*64
    runtime=Path(binding['runtime_path']);record=json.loads(runtime.read_bytes());record['runtime']['sdk_sha256']=binding['SDK_members']
    def refresh():
        record['source']=binding['prepared_source'];runtime.write_text(json.dumps(record),encoding='utf-8')
        binding['runtime_sha256']=H(runtime.read_bytes());path.write_text(json.dumps(binding),encoding='utf-8')
        go=Path(binding['root_go_path']);value=json.loads(go.read_bytes());value['binding_sha256']=H(path.read_bytes());go.write_text(json.dumps(value),encoding='utf-8')
    refresh();shared=current.boundary(ROOT)
    accepted=shared.validate_binding(path,root=ROOT,context_path=context_path)
    assert accepted[0]==binding and accepted[1]==context;current.require_sdk_dependencies(binding)
    name=adapter;expected=context['source_files_canonical_sha256'][name];snapshot=Path(binding['source_root'])
    assert H(shared.source_bytes(ROOT/name))==H(shared.source_bytes(snapshot/name))==binding['prepared_source']['files_sha256'][name]==expected
    if fault is None:return
    if fault=='current':
        original=shared.source_bytes
        monkeypatch.setattr(shared,'source_bytes',lambda p:original(p)+b'bad'if Path(p)==ROOT/name else original(p))
    elif fault=='snapshot':(snapshot/name).write_bytes((snapshot/name).read_bytes()+b'bad')
    elif fault=='prepared':binding['prepared_source']['files_sha256'][name]='0'*64;refresh()
    else:binding['base_contract_sha256']=H((ROOT/'coverage/contracts/vi-public-log2-provider-benchmark-v2.json').read_bytes());refresh()
    with pytest.raises(AssertionError):shared.validate_binding(path,root=ROOT,context_path=context_path)


def test_small_current_pairs_preserve_all_eight_domains_and_natural_FP():
    source=(ROOT/'tests/test_vi_public_matmul_provider_public.py').read_text()
    tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='test_current_public_contract_original_pair')
    assert "pytest.mark.parametrize('spec', SPECS, ids=lambda s: s['id'])"in [ast.unparse(n)for n in fn.decorator_list]
    for token in("reference_runtime()","observe(checked(reference,'reference'),spec,np)","observe(checked(native,'native'),spec,np)",'restore(natural)','actual==expected',"events==[[]]and returned==[]and suffix==[]and accepted==[False]",'VI_PUBLIC_MATMUL_SMALL_RECORD_DIR','outputs[\'expected\'].tobytes()==outputs[\'actual\'].tobytes()','bytes(actual)==bytes(value)'):
        assert token in source
    assert all(token not in source for token in('target_immediate','NPY_DISABLE_CPU_FEATURES','allclose','isclose','pytest.skip','timing=True'))


@pytest.mark.parametrize('partial',(False,True))
def test_strict_primary_retains_already_captured_target_FP_without_new_reads(partial):
    source=(ROOT/'tests/test_vi_public_matmul_provider_public.py').read_text();tree=ast.parse(source)
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='retain_target_FP')
    namespace={};exec(compile(ast.Module(body=[helper],type_ignores=[]),'<actual failure FP encoder>','exec'),namespace)
    class State:
        control=895;status=32;mxcsr=8096
        def __init__(self,word):self.word=word
        def __bytes__(self):return bytes([self.word])*32
    states={'reference':[State(1),State(2)],'native':[State(3)]if partial else[State(3),State(4)]}
    record={};primary=AssertionError('STRICT_WORD_PRIMARY')
    with pytest.raises(AssertionError)as error:
        try:raise primary
        finally:namespace['retain_target_FP'](record,states)
    assert error.value is primary
    assert [row['raw32']for row in record['target_FP']['reference']]==['01'*32,'02'*32]
    assert [row['raw32']for row in record['target_FP']['native']]==(['03'*32]if partial else['03'*32,'04'*32])
    target=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='test_current_public_contract_original_pair')
    outer=next(n for n in target.body if isinstance(n,ast.Try))
    closeout=ast.unparse(ast.Module(body=outer.finalbody,type_ignores=[]))
    assert closeout.index('retain_target_FP(record, snapshots)')<closeout.index('retain_record(path, record, outputs, array_words)')
    assert 'except BaseException as error:'in closeout and "sys.stderr.write('VI_SMALL_DIAGNOSTIC_SECONDARY '"in closeout


@pytest.mark.parametrize('checkpoint',('post-target','finally'))
def test_actual_large_source_guard_rejects_either_identity_checkpoint_removal(checkpoint):
    guard=load('tests/test_vi_public_log2_large_source.py')
    source=(ROOT/'tests/test_vi_public_log2_provider_large.py').read_text()
    guard.assert_current_source(source)
    token='np.matmul is true_matmul'
    position=source.index(token)if checkpoint=='post-target'else source.rindex(token)
    changed=source[:position]+'True'+source[position+len(token):]
    assert token in changed  # The other checkpoint is deliberately still there.
    with pytest.raises(AssertionError):guard.assert_current_source(changed)
