"""Complete inert/source VI provider controls; no scientific/native/helper calls."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shlex
import shutil
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/vi-public-log2-provider-source-v1.json'
POLICY='coverage/contracts/vi-public-log2-provider-v1.json'
H=lambda b:hashlib.sha256(b).hexdigest()


def load(path):
    spec=importlib.util.spec_from_file_location('_vi_provider_source_'+path.stem,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def reference_bridge(root,name,raw,canonical=False):
    from tools.mark_lbp_fp_context import VI_PUBLIC_REFERENCE_CONTEXT_SHA256
    from tools.vi_public_reference_context_v2 import predecessor_bytes
    return predecessor_bytes(root,name,raw,VI_PUBLIC_REFERENCE_CONTEXT_SHA256,canonical)


def sdk_bridge(root,name,raw,canonical=False):
    raw=reference_bridge(root,name,raw,canonical)
    from tools.mark_lbp_fp_context import VI_PROVIDER_SDK_CONTEXT_SHA256
    from tools.vi_provider_sdk_context import predecessor_bytes
    return predecessor_bytes(root,name,raw,VI_PROVIDER_SDK_CONTEXT_SHA256,canonical)


def bridge(root,name,raw,canonical=False):
    raw=sdk_bridge(root,name,raw,canonical)
    from tools.mark_lbp_fp_context import VI_PUBLIC_LOG2_CONTEXT_SHA256
    from tools.vi_public_log2_context import predecessor_bytes
    return predecessor_bytes(root,name,raw,VI_PUBLIC_LOG2_CONTEXT_SHA256,canonical)


def test_current_source_context_and_scalar_ABI_preserved():
    data=json.loads((ROOT/CONTEXT).read_bytes())
    for name,row in data['sources'].items():
        raw=bridge(ROOT,name,(ROOT/name).read_bytes())
        assert H(raw)==row['predecessor_raw_sha256'],name
    cpp=(ROOT/'src/variation_of_information.cpp').read_text()
    old=bridge(ROOT,'src/variation_of_information.cpp',cpp.encode()).decode()
    def body(text,start):
        a=text.index(start);brace=text.index('{',a);depth=1;i=brace+1
        while depth:
            depth+=(text[i]=='{')-(text[i]=='}');i+=1
        return text[a:i]
    for start in('bool variation_of_information(VariationTableView','bool variation_of_information(const detail::LabelPairInput&'):
        assert body(cpp,start)==body(old,start)
    header=(ROOT/'include/skimagexx/variation_of_information.hpp').read_text()
    assert 'using Function=bool(*)(const double*,double*,std::size_t,void*);'in header
    assert 'scalar platform-libm' in header and 'Exceptions propagate' in header
    # The shared raw-reader correction remains untouched by this successor.
    assert H((ROOT/'tools/join_source_context.py').read_bytes())==data['unchanged_join_source_raw_sha256']


def test_provider_order_zero_coalescing_and_owned_graphs_source():
    text=(ROOT/'src/variation_of_information.cpp').read_text()
    fn=text[text.index('bool variation_of_information_compatible(VariationTableView'):]
    assert fn.index('batch_entropy(a,ix,true')<fn.index('result.values[0]=source_dot')<fn.index('batch_entropy(a,iy,false')<fn.index('result.values[1]=source_dot')<fn.index('output=result;return true;')
    batch=text[text.index('bool batch_entropy('):text.index('} // namespace')]
    assert 'scaled_row(a,r,inverse,left,coefficients,seen)'in batch
    assert 'if(coefficients[c]!=0.)'in batch and 'offsets[r+1]=values.size()'in batch
    assert batch.index('provider.function(values.data(),logarithms.data(),values.size(),provider.context)')<batch.index('values[i]*=logarithms[i]')<batch.index('entropy[r]=minor_sum')
    assert 'entropy[columns[i]]+=values[i]'in batch
    assert 'if(!provider.function' in fn and '#pragma omp'not in fn and 'case_id'not in fn
    assert 'feclearexcept'not in text and 'fesetenv'not in text and 'seed_state'not in text


def test_binding_GIL_lifetime_error_and_public_API_source():
    text=(ROOT/'src/bind_variation_of_information.cpp').read_text()
    callback=text[text.index('static bool batch('):text.index('void bind_variation_of_information')]
    assert 'py::object callable;'in text
    assert callback.index('py::gil_scoped_acquire acquire;')<callback.index('py::array_t<double> input(count)')<callback.index('PyObject_CallOneArg(owner.callable.ptr(),input.ptr())')
    assert 'if(!returned)throw py::error_already_set();'in callback
    assert 'py::reinterpret_steal<py::object>(returned)'in callback
    assert 'Py_TYPE(result.ptr())!=Py_TYPE(input.ptr())'in callback
    assert 'array.owndata()'in callback and 'array.dtype().attr("isnative")'in callback
    assert '(count&&array.strides(0)'in callback  # Empty arrays are not forced to stride8.
    assert 'if(count)std::memcpy(logarithms,array.data(),count*sizeof(double));'in callback
    assert text.index('Log2Provider provider{std::move(log2_provider)}')<text.index('py::gil_scoped_release release')
    assert 'accepted=supplied' in text and 'variation_of_information(table,result,workers)'in text
    assert all(word not in text for word in('PyUFuncObject','_get_strided_loop','_resolve_dtypes','dlvsym','NPY_DISABLE_CPU_FEATURES','thread_local','catch('))


def frontend():
    tree=ast.parse((ROOT/'python/skimagexx/_variation_of_information.py').read_bytes())
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    return compile(ast.fix_missing_locations(tree),'<actual VI provider frontend>','exec')


def test_actual_frontend_forwards_current_provider_without_call_or_cache():
    seen=[];result=object();provider=object()
    class Image:shape=(2,2)
    class Buffer:
        flags=SimpleNamespace(aligned=True)
    class CSR:
        dtype=SimpleNamespace(kind='f',itemsize=8,isnative=True);shape=(2,2)
        data=Buffer();indices=Buffer();indptr=Buffer()
    table=CSR();np=SimpleNamespace(ndarray=Image,float64='f8',int64='i8',log2=provider,ascontiguousarray=lambda x,dtype:x)
    class Core:
        def label_pair_check_storage(self,v):seen.append(('storage',v))
        def get_num_threads(self):return 1
        def variation_of_information_csr(self,*a):
            assert a[:6]==(table.data,table.indices,table.indptr,2,2,1)
            seen.append(('provider',a[6]));return result
    env=dict(np=np,sparse=SimpleNamespace(csr_array=CSR,csr_matrix=type('Matrix',(),{})),_core=Core(),contingency_table=lambda*a,**k:table)
    exec(frontend(),env)
    fn=env['variation_of_information'];assert str(inspect.signature(fn))=='(image0=None, image1=None, *, table=None, ignore_labels=())'
    assert fn(Image(),Image())is result
    replacement=object();np.log2=replacement
    assert fn(Image(),Image(),table=table)is result
    assert [v for k,v in seen if k=='provider']==[provider,replacement]
    primary=RuntimeError('INERT_PROVIDER_ERROR')
    def fail(*a):raise primary
    env['_core'].variation_of_information_csr=fail
    with pytest.raises(RuntimeError)as error:fn(Image(),Image(),table=table)
    assert error.value is primary


class Batch:
    ndim=1
    dtype=SimpleNamespace(kind='f',itemsize=8,isnative=True,str='<f8')
    flags=SimpleNamespace(c_contiguous=True,owndata=True)
    def __init__(self,n=3):self.size=n;self.shape=(n,)


def guard_fixture():
    events=[]
    def provider(value):events.append(value);return value
    np=SimpleNamespace(ndarray=Batch,log2=provider,log=provider,sum=provider)
    class Base:
        def __init__(self,arg,*a,**k):self.arg=arg
        def sum(self,*a):return 'old-sum'
    class CSR(Base):pass
    class Matrix(Base):pass
    return np,SimpleNamespace(csr_array=CSR,csr_matrix=Matrix),events


def adapter():return load(ROOT/'benchmarks/vi_public_log2_provider_execution.py')


def test_provider_guard_counts_two_whole_batches_and_restores_inherited_methods():
    m=adapter();np,sparse,events=guard_fixture();before=(dict(sparse.csr_array.__dict__),dict(sparse.csr_matrix.__dict__));old=np.log2
    with m.candidate_providers(np,sparse)as batches:
        assert np.log2(Batch())is events[0]
        assert np.log2(Batch(0))is events[1]
        assert [b['shape']for b in batches]==[[3],[0]]
        with pytest.raises(AssertionError):np.sum(Batch())
        with pytest.raises(AssertionError):np.log(Batch())
        with pytest.raises(AssertionError):sparse.csr_array((1,(2,3)))
        value=sparse.csr_array((1,2,3))
        with pytest.raises(AssertionError):value.sum()
    assert np.log2 is old and before==(dict(sparse.csr_array.__dict__),dict(sparse.csr_matrix.__dict__))


@pytest.mark.parametrize('fault',('missing','too-many','not-owned','dtype','rank','primary'))
def test_provider_guard_refuses_bypasses_and_preserves_primary(fault):
    m=adapter();np,sparse,events=guard_fixture();old=np.log2;primary=RuntimeError('INERT_PRIMARY')
    with pytest.raises(RuntimeError if fault=='primary'else AssertionError)as error:
        with m.candidate_providers(np,sparse):
            if fault=='primary':raise primary
            if fault=='missing':pass
            else:
                value=Batch()
                if fault=='not-owned':value.flags=SimpleNamespace(c_contiguous=True,owndata=False)
                if fault=='dtype':value.dtype=SimpleNamespace(kind='f',itemsize=4,isnative=True)
                if fault=='rank':value.ndim=2
                np.log2(value)
                if fault=='too-many':np.log2(Batch());np.log2(Batch())
    if fault=='primary':assert error.value is primary
    assert np.log2 is old


def test_new_executor_is_isolated_strict_predecessor_and_unbound(monkeypatch):
    m=adapter();old=load(ROOT/'benchmarks/variation_of_information_execution.py');current=m.executor()
    assert old.CONTEXT=='coverage/contracts/variation-of-information-benchmark-execution-v6.json'
    assert current.CONTEXT=='coverage/contracts/vi-public-log2-provider-benchmark-v1.json'
    assert current.candidate_providers is m.candidate_providers
    assert current.execute.__code__.co_code==old.execute.__code__.co_code
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks'))
    consumer=load(ROOT/'benchmarks/vi_public_log2_provider_consumer.py')
    monkeypatch.setattr(sys,'argv',['provider-consumer'])
    with pytest.raises(SystemExit,match='VI_UNBOUND'):current.execute(consumer.__dict__)
    old_context=json.loads((ROOT/old.CONTEXT).read_bytes());new=json.loads((ROOT/current.CONTEXT).read_bytes())
    assert old_context['cases']==new['cases'] and old_context['method']==new['method']
    assert new['route']['contract']==current.CONTEXT and new['route']['driver']=='benchmarks/vi_public_log2_provider_consumer.py'
    assert all(x is None for x in new['unbound'].values()) and new['execution_authority']is None


@pytest.mark.parametrize('field',('contract','driver'))
def test_new_executor_refuses_old_route_before_loading_predecessor(tmp_path,monkeypatch,field):
    module=load(ROOT/'benchmarks/vi_public_log2_provider_execution.py')
    data=json.loads((ROOT/module.CONTEXT).read_bytes())
    data['route'][field]='frozen-old-route'
    path=tmp_path/module.CONTEXT;path.parent.mkdir(parents=True);path.write_text(json.dumps(data),encoding='utf-8')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    with pytest.raises(AssertionError):module.executor()


@pytest.mark.parametrize('canonical',(False,True))
def test_source_bridge_refuses_changed_and_double_projected_inputs(canonical):
    data=json.loads((ROOT/CONTEXT).read_bytes())
    for name,row in data['sources'].items():
        raw=(ROOT/name).read_bytes()
        if canonical:raw=raw.replace(b'\n',b'\r\n')
        previous=bridge(ROOT,name,raw,canonical)
        assert H(previous)==row['predecessor_canonical_sha256'if canonical else'predecessor_raw_sha256']
        with pytest.raises(AssertionError):bridge(ROOT,name,raw+b' ',canonical)
        with pytest.raises(AssertionError):bridge(ROOT,name,previous,canonical)


def source_root(tmp_path):
    names=['tools/vi_public_reference_context.py','coverage/contracts/vi-public-log2-reference-source-v1.json',
        'tools/vi_public_reference_context_v2.py','coverage/contracts/vi-public-log2-reference-source-v2.json','tools/vi_provider_sdk_context.py','coverage/contracts/vi-provider-sdk-source-v1.json',
        'tools/vi_public_log2_context.py','tools/mark_lbp_fp_context.py',CONTEXT,
        'coverage/contracts/mark-lbp-controller-source-successor-v1.json',
        'coverage/contracts/mark-lbp-fp-integration-v2.json','coverage/contracts/mark-lbp-fp-integration-v1.json',
        'coverage/contracts/rolling-ball-sentinel-v1.json','coverage/contracts/current281-sdk-family-source-v2.json',
        'coverage/contracts/current281-projection-v2.json','coverage/contracts/current281-integration-fixes-v1.json',
        'coverage/contracts/rolling-ball-enabled-context-v1.json','coverage/contracts/vi-reduction-integration-v2.json']
    for name in names:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    return tmp_path


@pytest.mark.parametrize('fault',('context','helper','old-parent','old-SDK'))
def test_actual_old_entry_keeps_current_and_old_context_gates(tmp_path,fault):
    root=source_root(tmp_path);name='tools/public_qualification.py';raw=(ROOT/name).read_bytes()
    from tools.join_source_context import correction_predecessor_bytes
    expected=correction_predecessor_bytes(ROOT,name,raw)
    assert correction_predecessor_bytes(root,name,raw)==expected
    if fault=='context':p=root/CONTEXT;p.write_bytes(p.read_bytes()+b' ')
    elif fault=='helper':p=root/'tools/vi_public_log2_context.py';p.write_bytes(p.read_bytes()+b' ')
    elif fault=='old-parent':p=root/'coverage/contracts/mark-lbp-fp-integration-v2.json';p.write_bytes(p.read_bytes()+b' ')
    else:
        p=root/'coverage/contracts/current281-sdk-family-source-v2.json';value=json.loads(p.read_bytes());value['predecessor_source']='tampered';p.write_text(json.dumps(value))
    with pytest.raises(AssertionError):correction_predecessor_bytes(root,name,raw)


def test_retained_predecessor_bytes_are_bound_even_with_rehashed_context(tmp_path):
    root=source_root(tmp_path);name='src/variation_of_information.cpp';data=json.loads((root/CONTEXT).read_bytes())
    data['sources'][name]['inverse'][0]['predecessor']+=' '
    encoded=(json.dumps(data,indent=2)+'\n').encode();(root/CONTEXT).write_bytes(encoded)
    from tools.vi_public_log2_context import predecessor_bytes
    with pytest.raises(AssertionError):predecessor_bytes(root,name,(ROOT/name).read_bytes(),H(encoded))


def test_collector_and_real_workflow_include_complete_new_guard():
    q=load(ROOT/'tools/public_qualification.py');data=json.loads((ROOT/CONTEXT).read_bytes())
    assert set(data['sources'])|set(data['new_collector_inputs'])<=set(q.DRAW_SOURCE_PATHS)
    historical_workflow=sdk_bridge(ROOT,'.github/workflows/public-qualification.yml',(ROOT/'.github/workflows/public-qualification.yml').read_bytes())
    line=next(x.strip()for x in historical_workflow.decode().splitlines()if x.strip().startswith('python -m pytest -q --capture=sys'))
    args=shlex.split(line);selectors=[a for a in args[:args.index('|')]if a.startswith('tests/')]
    assert len(selectors)==len(set(selectors))==data['source_selector_count']
    assert 'tests/test_vi_public_log2_provider_source.py'in selectors
    assert data['execution_authority']is None
    policy=json.loads((ROOT/POLICY).read_bytes())
    assert policy['compiled']is policy['public_parity']is policy['SDK_parity']is policy['release_qualified']is False
    assert policy['provider']['name']=='numpy.log2' and policy['SDK_default']=='platform-libm scalar; existing ABI'


def test_installed_SDK_controls_separate_scalar_default_host_callback_and_no_original():
    source=(ROOT/'tests/cmake_consumer/variation_of_information_enabled_consumer.cpp').read_text()
    assert 'vi_current_counts_controls()'in source and 'vi_large_saved_controls()'in source
    assert 'variation_of_information_compatible' in source and 'sdk_log2' in source
    assert 'audit.calls!=2' in source and 'audit.empty!=2' in source
    assert 'result.values[0]!=123.' in source and 'result.values[1]!=456.' in source
    assert 'not evidence of NumPy/SVML semantics' in source
    assert 'host_scalar_log2_compatible=' in source


@pytest.mark.parametrize('canonical',(False,True))
def test_SDK_successor_exact_current_and_predecessor_source_domains(canonical):
    data=json.loads((ROOT/'coverage/contracts/vi-provider-sdk-source-v1.json').read_bytes())
    for name,row in data['sources'].items():
        raw=(ROOT/name).read_bytes()
        transported=raw.replace(b'\n',b'\r\n')if canonical else raw
        old=sdk_bridge(ROOT,name,transported,canonical)
        assert H(old)==row['predecessor_canonical_sha256'if canonical else'predecessor_raw_sha256']
        with pytest.raises(AssertionError):sdk_bridge(ROOT,name,transported+b' ',canonical)
        with pytest.raises(AssertionError):sdk_bridge(ROOT,name,old,canonical)


@pytest.mark.parametrize('fault',('context','helper'))
def test_SDK_successor_rejects_changed_authentication_inputs(tmp_path,fault):
    root=source_root(tmp_path)
    name='tools/public_qualification.py';raw=(ROOT/name).read_bytes()
    assert sdk_bridge(root,name,raw)==sdk_bridge(ROOT,name,raw)
    path=root/('coverage/contracts/vi-provider-sdk-source-v1.json'if fault=='context'else'tools/vi_provider_sdk_context.py')
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(AssertionError):sdk_bridge(root,name,raw)


def test_SDK_successor_bound_predecessor_cannot_be_rewritten(tmp_path):
    root=source_root(tmp_path);path=root/'coverage/contracts/vi-provider-sdk-source-v1.json'
    data=json.loads(path.read_bytes());name='tests/cmake_consumer/variation_of_information_enabled_consumer.cpp'
    data['sources'][name]['inverse'][0]['predecessor']+=' '
    encoded=(json.dumps(data,indent=2)+'\n').encode();path.write_bytes(encoded)
    from tools.vi_provider_sdk_context import predecessor_bytes
    with pytest.raises(AssertionError):predecessor_bytes(root,name,(ROOT/name).read_bytes(),H(encoded))


def test_current_SDK_and_capture_guards_and_collector_are_complete():
    data=json.loads((ROOT/'coverage/contracts/vi-provider-sdk-source-v1.json').read_bytes())
    qualifier=load(ROOT/'tools/public_qualification.py')
    assert set(data['sources'])|set(data['new_collector_inputs'])<=set(qualifier.DRAW_SOURCE_PATHS)
    prior=reference_bridge(ROOT,'.github/workflows/public-qualification.yml',(ROOT/'.github/workflows/public-qualification.yml').read_bytes())
    line=next(x.strip()for x in prior.decode().splitlines()if x.strip().startswith('python -m pytest -q --capture=sys'))
    args=shlex.split(line);selectors=[a for a in args[:args.index('|')]if a.startswith('tests/')]
    assert len(selectors)==len(set(selectors))==data['source_selector_count']==172
    assert selectors==data['prior_source_selectors']+data['added_source_selectors']
    assert data['added_source_selectors']==['tests/test_vi_owning_sdk_source.py','tests/test_vi4_failure_capture.py']
    assert data['execution_authority']is None


@pytest.mark.parametrize('canonical',(False,True))
def test_reference_successor_exact_123_source_domains(canonical):
    data=json.loads((ROOT/'coverage/contracts/vi-public-log2-reference-source-v2.json').read_bytes())
    assert data['predecessor_source']=='1237655d691740d20b35fe28109b4dc08642896a'
    for name,row in data['sources'].items():
        raw=(ROOT/name).read_bytes();transported=raw.replace(b'\n',b'\r\n')if canonical else raw
        old=reference_bridge(ROOT,name,transported,canonical)
        assert H(old)==row['predecessor_canonical_sha256'if canonical else'predecessor_raw_sha256']
        with pytest.raises(AssertionError):reference_bridge(ROOT,name,transported+b' ',canonical)
        with pytest.raises(AssertionError):reference_bridge(ROOT,name,old,canonical)


@pytest.mark.parametrize('fault',('context','helper','parent','old-reference'))
def test_reference_successor_rejects_authentication_changes(tmp_path,fault):
    root=source_root(tmp_path);name='tools/public_qualification.py';raw=(ROOT/name).read_bytes()
    assert reference_bridge(root,name,raw)==reference_bridge(ROOT,name,raw)
    path=root/({'context':'coverage/contracts/vi-public-log2-reference-source-v2.json','helper':'tools/vi_public_reference_context_v2.py','parent':'coverage/contracts/vi-provider-sdk-source-v1.json','old-reference':'coverage/contracts/vi-public-log2-reference-source-v1.json'}[fault])
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(AssertionError):reference_bridge(root,name,raw)


def test_reference_successor_retains_old172_and_adds_complete173_guard():
    q=load(ROOT/'tools/public_qualification.py');data=json.loads((ROOT/'coverage/contracts/vi-public-log2-reference-source-v2.json').read_bytes())
    assert set(data['sources'])|set(data['new_collector_inputs'])<=set(q.DRAW_SOURCE_PATHS)
    assert {'tools/vi_cohort_value_diagnostic.py','tests/test_vi_cohort_value_diagnostic.py'}<=set(data['new_collector_inputs'])
    current=(ROOT/'.github/workflows/public-qualification.yml').read_text()
    def selectors(text):
        args=shlex.split(next(x.strip()for x in text.splitlines()if x.strip().startswith('python -m pytest -q --capture=sys')))
        return [a for a in args[:args.index('|')]if a.startswith('tests/')]
    assert selectors(current)==data['prior_source_selectors']+['tests/test_vi_public_log2_large_source.py']
    assert len(selectors(current))==len(set(selectors(current)))==173
    assert data['execution_authority']is None
    contract=q.current_contract(lambda name:(ROOT/name).read_bytes())
    assert q.CURRENT_SCOPE_CONTRACT=='coverage/contracts/current-public-qualification-v7.json'
    assert len(contract['required_selectors'])==160
    assert contract['post203_contracts']['metrics.variation_of_information']=='tests/test_vi_public_log2_provider_large.py'
    for profile in('minimum','modern','debug'):
        groups=q.applicable_public_groups(q.GROUPS,profile)
        assert groups['public-VI-large-saved-linux']==('test_variation_of_information_large_saved.py','test_vi_public_log2_provider_large.py')
    assert 'public-VI-large-saved-linux'not in q.applicable_public_groups(q.GROUPS,'windows')
    text=(ROOT/'tools/public_qualification.py').read_text()
    assert "os.environ['VI_PUBLIC_LOG2_LARGE_RECORD_DIR']=str(out/(mode+'-VI-current-large-rows'))"in text
    assert "require(row['tests']==16,'Full8 legacy scalar SDK plus8 fresh current-public VI pair nodes required')"in text
    assert "validate_VI_current_large_rows(out/(mode+'-VI-current-large-rows'))"in text


def test_current_owning_SDK_archive_extractor_keeps_all_included_headers():
    text=(ROOT/'tools/public_qualification.py').read_text()
    sdk=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='sdk')
    predicate=next(n.test for n in ast.walk(sdk)if isinstance(n,ast.If)and 'tests/fixtures/bezier_curve_originals_v1.bin'in ast.unparse(n.test))
    expression=compile(ast.Expression(predicate),'<actual SDK source archive extractor>','eval')
    for name in('tests/variation_of_information_provider_controls.hpp','tests/variation_of_information_current_counts_controls.hpp','tests/variation_of_information_large_controls.hpp','tests/fixtures/variation_of_information_saved_v1.hpp','tests/cmake_consumer/variation_of_information_enabled_consumer.cpp'):
        assert (ROOT/name).is_file()and eval(expression,{'name':name}),name
    for name in('tests/variation_of_information_provider_controls.hpp/','tests/unassessed_provider.hpp','python/skimagexx/_variation_of_information.py'):
        assert not eval(expression,{'name':name}),name


@pytest.mark.parametrize('fault',('missing-current','missing-legacy','wrong-map','wrong-policy'))
def test_reference_current_contract_rejects_scope_reduction(fault):
    q=load(ROOT/'tools/public_qualification.py');data=json.loads((ROOT/q.CURRENT_SCOPE_CONTRACT).read_bytes())
    if fault=='missing-current':data['required_selectors'].remove('tests/test_vi_public_log2_provider_large.py')
    elif fault=='missing-legacy':data['required_selectors'].remove('tests/test_variation_of_information_large_saved.py')
    elif fault=='wrong-map':data['post203_contracts']['metrics.variation_of_information']='tests/test_variation_of_information_large_saved.py'
    else:data['VI_current_runtime_large_reference']['current_nodes_per_mode']=4
    with pytest.raises(q.QualificationError):q.current_contract(lambda name:json.dumps(data).encode())


@pytest.mark.parametrize('fault',(None,'missing','duplicate','words','FP','caller','provider','ownership','markers','diagnostic','inputs','software'))
def test_reference_seal_requires_all8_independent_current_records(tmp_path,fault):
    q=load(ROOT/'tools/public_qualification.py');raw='00000000000000000000000000000000';fp={'control':895,'status':32,'mxcsr':8098}
    value=dict(primary=None,timing=False,diagnostics=[],counts=dict(original=1,native=1),accepted=[True],suffix=[True],requirements={'native_marker':11},returned=['native_marker']*11,
        expected=dict(type=['numpy','ndarray'],dtype='<f8',shape=[2],nbytes=16,raw_hex=raw,payload_sha256=H(bytes.fromhex(raw)),flags={'owndata':True}),
        caller32=dict(saved='00'*32,actual='00'*32,exact=True),natural_caller_raw32='00'*32,
        reference_FP=dict(before=fp,after=fp,before_raw32='00'*32),native_FP=dict(before=fp,after=fp,before_raw32='00'*32),
        provider='current true numpy.log2 ufunc computation',provider_type='ufunc',provider_batches=[{'provider':'numpy.log2'}]*2,
        output_pair_alias=False,output_input_aliases=False,input_before={'saved':1},input_after={'saved':1},pointers_before={'a':4},pointers_after={'a':4},pointers_stable=True,software_context_unchanged=True,namespace_identity_unchanged=True)
    rows=[json.loads(json.dumps(dict(value,index=i,workers=w)))for i in range(4)for w in(1,4)]
    for row in rows:row['actual']=dict(row['expected'])
    if fault=='missing':rows.pop()
    elif fault=='duplicate':rows[0]['index']=rows[1]['index'];rows[0]['workers']=rows[1]['workers']
    elif fault=='words':rows[0]['actual']['raw_hex']='01'*16
    elif fault=='FP':rows[0]['native_FP']['after']['status']=33
    elif fault=='caller':rows[0]['caller32']['exact']=False
    elif fault=='provider':rows[0]['provider_batches']=[]
    elif fault=='ownership':rows[0]['accepted']=[False]
    elif fault=='markers':rows[0]['returned'].pop()
    elif fault=='diagnostic':rows[0]['diagnostics']=[{'error':'failed'}]
    elif fault=='inputs':rows[0]['input_after']={}
    elif fault=='software':rows[0]['software_context_unchanged']=False
    for i,row in enumerate(rows):(tmp_path/(str(i)+'.json')).write_text(json.dumps(row),encoding='utf-8')
    if fault:
        with pytest.raises(q.QualificationError):q.validate_VI_current_large_rows(tmp_path)
    else:assert len(q.validate_VI_current_large_rows(tmp_path))==8
