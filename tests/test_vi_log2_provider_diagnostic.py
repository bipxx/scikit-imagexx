"""Inert provider controls: no scientific imports, F1 loads or math primitives."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from variation_of_information_source_context import provider_predecessor_source

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_inert_log2_provider', ROOT/'tools/vi_log2_provider_diagnostic.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def base_features():
    return dict(baseline=['SSE','SSE2','SSE3'], dispatch=['AVX2','AVX512F','AVX512_SKX','AVX512_CLX','AVX512_ICL'],
                enabled=dict(AVX2=True, AVX512F=True, AVX512_SKX=True, AVX512_CLX=True,
                             AVX512_ICL=False, AVX512BW=True, AVX512DQ=True),
                numpy_version='1.24.4', numpy_binary='/same/numpy.so')


def test_exact_retained_inputs_no_float_reconstruction():
    raw = D.inputs()
    assert [len(p) for p in raw] == [448,448]
    assert D.sha(D.FIXTURE) == '144808de2c3dfb7c49eec0abcbed59a9783eeab20dfbfd359c0057a28e151bd6'
    assert hashlib.sha256(b''.join(raw)).hexdigest() == '31729e0dcb882ce246ddaf996a6733cf7d00006b2dfc4509c6d0d6af4384d78e'
    assert raw[0][:8] == int('3fb87fffffffffe7',16).to_bytes(8,'little')
    assert len({raw[0][i:i+8] for i in range(0,448,8)}) == 13
    assert len({raw[1][i:i+8] for i in range(0,448,8)}) == 34


@pytest.mark.parametrize('mutation', ['order','count','dtype','nonpositive'])
def test_input_changes_rejected(tmp_path, monkeypatch, mutation):
    fixture = json.loads(D.FIXTURE.read_bytes())
    vector = fixture['vectors'][0]
    if mutation == 'order': vector['words_uint64_hex'].reverse()
    if mutation == 'count': vector['words_uint64_hex'].pop()
    if mutation == 'dtype': vector['dtype'] = '<f4'
    if mutation == 'nonpositive': vector['words_uint64_hex'][0] = 'bfb87fffffffffe7'
    target = tmp_path/'fixture.json'
    target.write_text(json.dumps(fixture),encoding='utf-8')
    context = tmp_path/'context.json'
    context.write_text(json.dumps(dict(input_sha256=D.sha(target))),encoding='utf-8')
    monkeypatch.setattr(D,'FIXTURE',target)
    monkeypatch.setattr(D,'CONTRACT',context)
    with pytest.raises(AssertionError): D.inputs()


def test_disables_explicit_compiled_enabled_groups_not_raw_subgroups():
    base = base_features()
    targets = D.disabled_targets(base)
    assert targets == ['AVX512F','AVX512_CLX','AVX512_SKX']
    disabled = deepcopy(base)
    for name in targets: disabled['enabled'][name] = False
    D.check_intervention(base,disabled,targets)
    assert disabled['enabled']['AVX512BW'] is True


@pytest.mark.parametrize('case', ['missing-compiled-SKX','unsupported-SKX','AVX512F-only','nonAVX512-change','compiled-change'])
def test_feature_intervention_rejects_inadequate_or_confounded_maps(case):
    base = base_features()
    disabled = deepcopy(base)
    targets = D.disabled_targets(base)
    for name in targets: disabled['enabled'][name] = False
    if case == 'missing-compiled-SKX': base['dispatch'].remove('AVX512_SKX')
    if case == 'unsupported-SKX': base['enabled']['AVX512_SKX'] = False
    if case == 'AVX512F-only': disabled['enabled']['AVX512_SKX'] = True
    if case == 'nonAVX512-change': disabled['enabled']['AVX2'] = False
    if case == 'compiled-change': disabled['dispatch'].remove('AVX2')
    with pytest.raises(AssertionError): D.check_intervention(base,disabled,targets)


def test_seed_changes_only_three_semantic_fields_of_actual_snapshot():
    caller = bytearray(range(32))
    caller[8:10] = b'\xff\xff'
    expected = dict(control=895,status=32,mxcsr=8098)
    seed = D.seed_state(bytes(caller),expected)
    assert D.semantic(seed) == expected
    assert seed[2:4] == caller[2:4] and seed[6:28] == caller[6:28]
    assert bytes(caller) != seed
    with pytest.raises(AssertionError): D.seed_state(bytes(32),expected)


@pytest.mark.parametrize('fail', [False,True])
def test_primitive_snapshots_then_restores_without_encoding_or_second_call(fail):
    events,rows = [],[]
    caller,seed,after = b'C'*32,b'S'*32,b'A'*32
    def restore(raw): events.append(('restore',raw)); return raw
    def save(): events.append(('save',after)); return after
    def primitive():
        events.append(('primitive',None))
        if fail: raise ValueError('preserved primitive failure')
        return object()
    if fail:
        with pytest.raises(ValueError,match='preserved primitive failure'):
            D.observe(rows,'inert',primitive,save,restore,seed,caller)
    else: D.observe(rows,'inert',primitive,save,restore,seed,caller)
    assert events == [('restore',seed),('primitive',None),('save',after),('restore',caller)]
    assert len(rows) == 1 and rows[0]['before'] == seed and rows[0]['after'] == after
    assert rows[0]['caller_restored'] == caller
    assert ('value' in rows[0]) is not fail


class Array:
    def __init__(self,raw):
        self.raw=raw;self.dtype=SimpleNamespace(str='<f8');self.shape=(56,);self.strides=(8,);self.nbytes=448
        self.flags=SimpleNamespace(owndata=True,c_contiguous=True)
    def tobytes(self): return self.raw
    def copy(self): return Array(self.raw)


def test_codec_preserves_all_words_and_FP_bytes():
    raw=D.inputs()[0]
    before=bytes.fromhex('7f03000020000000ffff000000000000000000000000000000000000a21f0000')
    result=D.encode_observations([dict(provider='numpy-0',before=before,after=before,caller_restored=before,value=Array(raw))],SimpleNamespace(ndarray=Array))
    assert bytes.fromhex(result[0]['raw_hex']) == raw
    assert result[0]['before'] == before.hex()
    assert result[0]['before_semantic'] == dict(control=895,status=32,mxcsr=8098)
    assert len(result[0]['words_uint64_hex']) == 56


def test_fixed_bitwise_metric_does_not_relax_equality():
    a=['bff0000000000000']*56;b=a.copy();b[27]='bff0000000000001'
    assert D.word_differences(a,a) == []
    assert D.word_differences(a,b) == [dict(index=27,left=a[27],right=b[27])]
    with pytest.raises(AssertionError): D.word_differences(a[:-1],b)


@pytest.mark.parametrize('status', ['UNBOUND','SOURCE_ONLY_NO_EXECUTION_AUTHORITY','ROOT_GO_PRIVATE_VI16_ONE_PAIR_VALUE_WITNESS'])
def test_old_or_unbound_authority_rejected_before_platform_or_helper(status):
    from datetime import datetime,timezone
    with pytest.raises(AssertionError): D.check_go(dict(status=status),runner_hash='unused',now=datetime.now(timezone.utc))


@pytest.mark.parametrize('mode', ['private-baseline','hosted-feature'])
def test_campaign_fresh_child_limits_and_only_environment_delta(tmp_path,monkeypatch,mode):
    parent=tmp_path/'go.json';parent.write_text('{}',encoding='utf-8')
    output=tmp_path/'outputs'
    go=dict(mode=mode,output_root=str(output),disabled_targets=['AVX512F','AVX512_SKX'])
    monkeypatch.setattr(D,'check_go',lambda *a,**k:None)
    monkeypatch.delenv('NPY_DISABLE_CPU_FEATURES',raising=False)
    commands=[]
    def run(command,*,env,stdout,stderr):
        role=command[-1];commands.append((role,env))
        rows=[dict(provider=prefix+str(i),words_uint64_hex=['bff0000000000000']*56)for i in range(2)for prefix in ('numpy-','core-scalar-')]
        D.write_json(output/(role+'.json'),dict(observations=rows,cpu=dict(model='same')))
        stdout.write(json.dumps(dict(status='CHILD_PASS_WITH_FINAL32_RESTORE',role=role,
            counts=dict(numpy_log2_array_calls=2,scalar_log2_values=112 if role=='base'else 0,
                        FP_save=15 if role=='base'else 9,FP_restore=10 if role=='base'else 6))).encode())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(D.subprocess,'run',run)
    D.campaign(go,parent,D.sha(parent))
    assert [role for role,env in commands] == (['base','disabled'] if mode=='hosted-feature' else ['base'])
    if mode=='hosted-feature':
        base,off=commands[0][1],commands[1][1]
        assert {k:(base.get(k),off.get(k)) for k in set(base)|set(off) if base.get(k)!=off.get(k)} == {'NPY_DISABLE_CPU_FEATURES':(None,'AVX512F,AVX512_SKX')}
    summary=json.loads((output/'summary.json').read_bytes())
    assert summary['VI_original_calls']==summary['VI_native_calls']==summary['timed_batches']==summary['retries']==0
    assert not summary['old_9V45_cause_proved'] and not summary['per_call_kernel_selection_proved']
    with pytest.raises(FileExistsError): D.campaign(go,parent,D.sha(parent))


def test_failed_first_child_preserves_exit_and_stops_without_retry(tmp_path,monkeypatch):
    parent=tmp_path/'go.json';parent.write_text('{}',encoding='utf-8')
    go=dict(mode='hosted-feature',output_root=str(tmp_path/'outputs'),disabled_targets=['AVX512_SKX'])
    monkeypatch.setattr(D,'check_go',lambda *a,**k:None)
    monkeypatch.delenv('NPY_DISABLE_CPU_FEATURES',raising=False)
    calls=[]
    def fail(*args,**kwargs):calls.append(args[0]);return SimpleNamespace(returncode=19)
    monkeypatch.setattr(D.subprocess,'run',fail)
    with pytest.raises(AssertionError,match='Stop dependent child'): D.campaign(go,parent,D.sha(parent))
    folder=Path(go['output_root'])
    assert len(calls)==1 and json.loads((folder/'base-exit.json').read_bytes())['exit_code']==19
    assert not (folder/'disabled-command.json').exists() and not (folder/'summary.json').exists()


@pytest.mark.parametrize('failure', [None,'scalar','late-authority'])
def test_child_full32_counts_and_failure_values_with_inert_providers(tmp_path,monkeypatch,failure,capsys):
    """Run the actual child control flow using memory-only F1/NumPy stand-ins."""
    import ctypes
    import sys
    from types import ModuleType
    contract=json.loads(D.CONTRACT.read_bytes())
    caller=bytes.fromhex('7f03000000000000ffff000000000000000000000000000000000000801f0000')
    state=SimpleNamespace(raw=caller,numpy_calls=0,scalar_calls=0,go_checks=0)
    def get(pointer):
        for i in range(8):pointer._obj[i]=int.from_bytes(state.raw[i*4:i*4+4],'little')
        return 0
    def restore(pointer,length):
        assert length==32;state.raw=bytes(pointer._obj);return 0
    fake_fenv=SimpleNamespace(backend=lambda:(SimpleNamespace(fegetenv=get),SimpleNamespace(reference_fenv_restore=restore)))
    real_spec=D.importlib.util.spec_from_file_location
    real_module=D.importlib.util.module_from_spec
    helper_spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None))
    monkeypatch.setattr(D.importlib.util,'spec_from_file_location',lambda name,path:helper_spec if name=='_provider_existing_F1'else real_spec(name,path))
    monkeypatch.setattr(D.importlib.util,'module_from_spec',lambda spec:fake_fenv if spec is helper_spec else real_module(spec))
    np=ModuleType('numpy');np.__version__='1.24.4';np.ndarray=Array
    np.core=SimpleNamespace(_multiarray_umath=SimpleNamespace(__cpu_baseline__=['SSE2'],__cpu_dispatch__=['AVX2'],__cpu_features__={'AVX2':True},__file__='/inert/numpy.so'))
    np.frombuffer=lambda raw,*,dtype:Array(raw)
    np.geterr=lambda:dict(divide='warn',over='warn',under='ignore',invalid='warn')
    np.getbufsize=lambda:8192
    def numpy_log2(array):
        state.numpy_calls+=1
        state.raw=state.raw[:4]+b'!\0'+state.raw[6:]
        return Array(array.raw)
    np.log2=numpy_log2
    monkeypatch.setitem(sys.modules,'numpy',np)
    def scalar(value):
        state.scalar_calls+=1
        if failure=='scalar'and state.scalar_calls==3:raise ValueError('inert scalar failure')
        state.raw=state.raw[:4]+b'!\0'+state.raw[6:]
        return value.value
    monkeypatch.setattr(D,'scalar_provider',lambda:(scalar,dict(inert=True),None))
    def check(*args,**kwargs):
        state.go_checks+=1
        if failure=='late-authority'and state.go_checks==2:raise ValueError('inert authority changed')
        return contract
    monkeypatch.setattr(D,'check_go',check)
    monkeypatch.setattr(D,'cpu_record',lambda:{'model name':contract['private_CPU_model']})
    monkeypatch.delenv('NPY_DISABLE_CPU_FEATURES',raising=False)
    go_path=tmp_path/'go.json';go_path.write_text('{}',encoding='utf-8')
    go=dict(mode='private-baseline',source_root=str(tmp_path),output_root=str(tmp_path))
    original_write=D.write_json
    def write(path,value):
        original_write(path,value)
        state.raw=b'W'*32  # Final restore must also undo a writer's ambient mutation.
    monkeypatch.setattr(D,'write_json',write)
    if failure:
        with pytest.raises(ValueError,match='inert '):D.child(go,go_path,D.sha(go_path),'base')
    else:D.child(go,go_path,D.sha(go_path),'base')
    assert state.raw==caller
    record=json.loads((tmp_path/'base.json').read_bytes())
    assert record['caller_raw32']==caller.hex() and record['full32_restored_before_codec']
    assert record['observations'][0]['raw_hex']==D.inputs()[0].hex()
    assert all(v['before_semantic']==contract['incoming_FP_semantic']for v in record['observations'])
    assert all(v['caller_restored']==caller.hex()for v in record['observations'])
    if failure=='scalar':
        assert state.numpy_calls==1 and state.scalar_calls==3
        assert len(record['observations'])==2 and 'raw_hex'not in record['observations'][1]
        assert record['failure']['error']=='inert scalar failure'
    else:
        assert state.numpy_calls==2 and state.scalar_calls==112
        assert len(record['observations'])==4
        if failure=='late-authority':assert record['authority_failure']['error']=='inert authority changed'
        else:
            terminal=json.loads(capsys.readouterr().out)
            assert terminal['counts']==dict(numpy_log2_array_calls=2,scalar_log2_values=112,FP_save=15,FP_restore=10)
            assert terminal['status']=='CHILD_PASS_WITH_FINAL32_RESTORE'


def test_source_scope_and_current_pins():
    context=json.loads(D.CONTRACT.read_bytes())
    assert context['status']=='SOURCE_ONLY_NO_EXECUTION_AUTHORITY' and context['actual_authority'] is None
    assert context['runner_sha256']==D.sha(ROOT/context['runner'])
    assert context['limits']==D.LIMITS and context['core_loaded'] is False
    assert context['limits']['scalar_log2_values_max']==128
    assert context['expected']['private_baseline']['numpy_arrays']==2
    assert context['expected']['hosted_feature']['numpy_arrays']==4
    assert context['scalar_symbol']['version']=='GLIBC_2.29'
    assert context['scalar_symbol']['NumPy_binary_import']=='log2@GLIBC_2.2.5'
    source=(ROOT/context['runner']).read_text()
    tree=ast.parse(source)
    assert source.count('np.log2(array)')==1
    assert 'math.log2(' not in source and 'variation_of_information(' not in source
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom))and any(a.name.startswith(('scipy','skimage'))for a in getattr(n,'names',[]))for n in ast.walk(tree))


def test_workflow_capacity_gate_feature_gate_helpers_and_unchanged_installation():
    original=(ROOT/'.github/workflows/vi-value-diagnostic.yml').read_text()
    workflow=(ROOT/'.github/workflows/vi-log2-provider-diagnostic.yml').read_text()
    assert workflow.index('Require scratch capacity')<workflow.index('actions/checkout@v4')
    assert workflow.index('--feature-gate')<workflow.index("probe['prepare_host_helper']")
    assert "if: env.VI_HOST_ELIGIBLE == '1' && env.LOG2_NUMPY_ELIGIBLE == '1'" in workflow
    assert "probe['prepare_cohorts']" not in workflow and 'vi_cohort_value_diagnostic.py' not in workflow
    start='          python -m venv "$RUNNER_TEMP/runtime"\n'
    stop='      - name: Prepare the authenticated small F1 helper'
    install=original[original.index(start):original.index(stop)]
    assert install in workflow
    assert workflow.count("probe['prepare_host_helper'](")==1
    assert workflow.count("'--wheel-dir',str(workspace/'stage/wheels')")==1
    assert "archive.unlink()" in workflow and 'archive_bytes_locally_available=False' in workflow
    # Compile every inline Python block on the actual test interpreter, never execute it.
    blocks=[];current=None
    for line in workflow.splitlines():
        if " <<'PY'" in line:current=[]
        elif current is not None and line=='          PY':blocks.append('\n'.join(current));current=None
        elif current is not None:
            assert line.startswith('          ')
            current.append(line[10:])
    assert len(blocks)>=6 and current is None
    for block in blocks:compile(block,'<inert-workflow>','exec')


def bound_go_fixture(tmp_path,monkeypatch,mode='hosted-feature'):
    """Inert trusted-metadata stand-ins; exercise the real check_go body."""
    from datetime import datetime,timedelta,timezone
    context=json.loads(D.CONTRACT.read_bytes())
    snapshot=tmp_path/'prepared16-source'
    for name,pin in context['prepared_dependencies'].items():
        raw=provider_predecessor_source(ROOT,name);assert hashlib.sha256(raw).hexdigest()==pin,name
        path=snapshot/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    now=datetime(2026,11,15,19,40,tzinfo=timezone.utc)
    prefix=context['hosted_loader_environment']['base_prefix']
    interpreter='/home/runner/work/_temp/runtime/bin/python'
    private_python='3.11.14 (main, Feb 12 2026, 00:42:50) [Clang 21.1.4 ]'
    python=context['genuine_python']if mode=='hosted-feature'else private_python
    runtime=tmp_path/'runtime.json'
    runtime.write_text(json.dumps(dict(source=dict(revision=D.SOURCE),runtime=dict(python=python,sdk_sha256={str(i):'inert'for i in range(108)}))),encoding='utf-8')
    binding=tmp_path/'private-binding.json'
    binding.write_text(json.dumps(dict(kind='PRIVATE_LOCAL_31114_RETAINED_16CORE_NOT_GENUINE',source_revision=D.SOURCE,
        core=dict(sha256=context['core_sha256']),python_version=private_python,interpreter=interpreter,helper=dict(path='/inert/helper.so'))),encoding='utf-8')
    helper=tmp_path/'helper-binding.json'
    env=dict(UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT='/inert/receipt.json',UTIL_VIEWS_FENV_RECEIPT_SHA256='1'*64)
    helper.write_text(json.dumps(dict(env=env)),encoding='utf-8')
    real_sha=D.sha
    # These tiny synthetic metadata files stand in for already authenticated
    # runtime975/private-binding identities, not for any scientific execution.
    identities={runtime:context['genuine_runtime_sha256'],binding:context['private_binding_sha256']}
    monkeypatch.setattr(D,'sha',lambda path:identities.get(Path(path),real_sha(path)))
    for key in('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT','OPENBLAS_CORETYPE','SKIMAGEXX_DISABLE_SIMD','UTIL_VIEWS_FENV_HELPER','UTIL_VIEWS_FENV_RECEIPT'):
        monkeypatch.delenv(key,raising=False)
    for key in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'):monkeypatch.setenv(key,'1')
    monkeypatch.setattr(D.sys,'platform','linux')
    monkeypatch.setattr(D.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(D.sys,'executable',interpreter)
    monkeypatch.setattr(D.sys,'version',python)
    monkeypatch.setattr(D.sys,'base_prefix',prefix)
    if mode=='hosted-feature':
        for key,value in env.items():monkeypatch.setenv(key,value)
        monkeypatch.setenv('LD_LIBRARY_PATH',context['hosted_loader_environment']['LD_LIBRARY_PATH'])
    else:
        monkeypatch.setenv('UTIL_VIEWS_FENV_MODE','private-fixed-v1')
        monkeypatch.setenv('UTIL_VIEWS_FENV_HELPER','/inert/helper.so')
        monkeypatch.delenv('LD_LIBRARY_PATH',raising=False)
    go=dict(status=D.AUTHORITY,limits=D.LIMITS,mode=mode,runner_sha256=D.sha(D.__file__),contract_sha256=D.sha(D.CONTRACT),input_sha256=D.sha(D.FIXTURE),
        source_revision=D.SOURCE,source_root=str(snapshot),output_root=str(tmp_path/'outputs'),core_sha256=context['core_sha256'],interpreter=interpreter,
        created_utc=now.isoformat(),deadline_utc=(now+timedelta(hours=1)).isoformat(),
        runtime_path=str(runtime),runtime_sha256=context['genuine_runtime_sha256'],helper_binding_path=str(helper),helper_binding_sha256=D.sha(helper),
        interpreter_base_prefix=prefix,allowed_LD_LIBRARY_PATH=context['hosted_loader_environment']['LD_LIBRARY_PATH'],
        private_binding_path=str(binding),private_binding_sha256=context['private_binding_sha256'])
    return go,context,now


def test_real_check_go_accepts_observed_authenticated_hosted_loader_path(tmp_path,monkeypatch):
    go,context,now=bound_go_fixture(tmp_path,monkeypatch)
    assert D.os.environ['LD_LIBRARY_PATH']=='/opt/hostedtoolcache/Python/3.11.17/x64/lib'
    assert D.check_go(go,runner_hash=D.sha(D.__file__),now=now)==context


@pytest.mark.parametrize('path', ['/tmp/lib','lib','/opt/hostedtoolcache/Python/3.11.17/x64/lib:/tmp/lib',
    '/opt/hostedtoolcache/Python/3.11.17/x64/lib:','/opt/hostedtoolcache/Python/3.11.16/x64/lib',
    '/opt/hostedtoolcache/Python/3.11.17/x64/../x64/lib',''])
def test_real_check_go_rejects_unbound_hosted_loader_paths(tmp_path,monkeypatch,path):
    go,context,now=bound_go_fixture(tmp_path,monkeypatch)
    monkeypatch.setenv('LD_LIBRARY_PATH',path)
    with pytest.raises(AssertionError):D.check_go(go,runner_hash=D.sha(D.__file__),now=now)


@pytest.mark.parametrize('changed', ['actual-prefix','GO-prefix','GO-path','consistent-wrong-prefix','runtime-pin','python-version'])
def test_real_check_go_requires_prefix_GO_and_genuine_runtime_binding(tmp_path,monkeypatch,changed):
    go,context,now=bound_go_fixture(tmp_path,monkeypatch)
    wrong='/opt/hostedtoolcache/Python/3.11.16/x64'
    if changed=='actual-prefix':monkeypatch.setattr(D.sys,'base_prefix',wrong)
    if changed=='GO-prefix':go['interpreter_base_prefix']=wrong
    if changed=='GO-path':go['allowed_LD_LIBRARY_PATH']='/tmp/lib'
    if changed=='consistent-wrong-prefix':
        monkeypatch.setattr(D.sys,'base_prefix',wrong);go['interpreter_base_prefix']=wrong
        go['allowed_LD_LIBRARY_PATH']=wrong+'/lib';monkeypatch.setenv('LD_LIBRARY_PATH',wrong+'/lib')
    if changed=='runtime-pin':go['runtime_sha256']='0'*64
    if changed=='python-version':monkeypatch.setattr(D.sys,'version','3.11.14 PRIVATE')
    with pytest.raises(AssertionError):D.check_go(go,runner_hash=D.sha(D.__file__),now=now)


@pytest.mark.parametrize('key', ['LD_PRELOAD','LD_AUDIT','OPENBLAS_CORETYPE','PYTHONPATH','PYTHONHOME','SKIMAGEXX_DISABLE_SIMD'])
def test_real_check_go_keeps_other_injection_barriers(tmp_path,monkeypatch,key):
    go,context,now=bound_go_fixture(tmp_path,monkeypatch)
    monkeypatch.setenv(key,'unbound-injection')
    with pytest.raises(AssertionError):D.check_go(go,runner_hash=D.sha(D.__file__),now=now)


def test_real_check_go_private_blank_loader_requirement_unchanged(tmp_path,monkeypatch):
    go,context,now=bound_go_fixture(tmp_path,monkeypatch,'private-baseline')
    assert D.check_go(go,runner_hash=D.sha(D.__file__),now=now)==context
    monkeypatch.setenv('LD_LIBRARY_PATH',context['hosted_loader_environment']['LD_LIBRARY_PATH'])
    with pytest.raises(AssertionError):D.check_go(go,runner_hash=D.sha(D.__file__),now=now)


def test_v2_loader_context_preserves_v1_and_fixed_science_scope():
    current=json.loads(D.CONTRACT.read_bytes())
    old_path=ROOT/current['predecessor']['path']
    assert D.sha(old_path)==current['predecessor']['sha256']=='f78e32ffa0b79fcf6560002616a544c066da38ac56297e89636470d5b17b27cc'
    old=json.loads(old_path.read_bytes())
    for key in old:
        if key not in('schema','runner_sha256'):assert current[key]==old[key],key
    workflow=(ROOT/'.github/workflows/vi-log2-provider-diagnostic.yml').read_text()
    assert 'coverage/contracts/vi-log2-provider-v1.json'not in workflow
    assert D.sha(D.CONTRACT)in workflow and D.sha(ROOT/current['runner'])in workflow
    assert "interpreter_base_prefix=sys.base_prefix,allowed_LD_LIBRARY_PATH=os.environ['LD_LIBRARY_PATH']"in workflow
