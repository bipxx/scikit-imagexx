"""Current VI4 matmul route/source controls; scientific targets are never loaded."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/vi4-public-matmul-capture-ci-fenv-v1.json'
PROVIDER='coverage/contracts/vi-public-matmul-provider-benchmark-ci-fenv-v1.json'
RUNNER='tools/vi4_public_matmul_capture.py'
ADAPTER='benchmarks/vi_public_matmul_provider_execution_v2.py'
CONSUMER='benchmarks/vi_public_matmul_provider_consumer_v2.py'
OLD_CONTEXT='coverage/contracts/vi4-failure-capture-v2.json'
OLD_PROVIDER='coverage/contracts/vi-public-matmul-provider-benchmark-v1.json'
H=lambda raw:hashlib.sha256(raw).hexdigest()


def data(name):return json.loads((ROOT/name).read_bytes())


def load(name):
    spec=importlib.util.spec_from_file_location('_inert_vi4_matmul_'+Path(name).stem,ROOT/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def function(text,name):
    return ast.get_source_segment(text,next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name))


def test_unbound_current_route_exact_dependencies_and_caps():
    module=load(RUNNER);current=data(CONTEXT);capture=current['capture'];provider=data(PROVIDER)
    preceding_capture=data('coverage/contracts/vi4-public-matmul-capture-v6.json');preceding_provider=data('coverage/contracts/vi-public-matmul-provider-benchmark-v7.json')
    assert capture['source_binding_successor']['metadata_predecessor_context']==dict(path='coverage/contracts/vi4-public-matmul-capture-v6.json',sha256=H((ROOT/'coverage/contracts/vi4-public-matmul-capture-v6.json').read_bytes()))
    assert provider['source_binding']['metadata_predecessor_contract']==dict(path='coverage/contracts/vi-public-matmul-provider-benchmark-v7.json',raw_sha256=H((ROOT/'coverage/contracts/vi-public-matmul-provider-benchmark-v7.json').read_bytes()))
    assert preceding_capture['capture']['source_binding_successor']['metadata_predecessor_context']==dict(path='coverage/contracts/vi4-public-matmul-capture-v5.json',sha256=H((ROOT/'coverage/contracts/vi4-public-matmul-capture-v5.json').read_bytes()))
    assert preceding_provider['source_binding']['metadata_predecessor_contract']==dict(path='coverage/contracts/vi-public-matmul-provider-benchmark-v6.json',raw_sha256=H((ROOT/'coverage/contracts/vi-public-matmul-provider-benchmark-v6.json').read_bytes()))
    previous_capture=data('coverage/contracts/vi4-public-matmul-capture-v5.json')
    previous_provider=data('coverage/contracts/vi-public-matmul-provider-benchmark-v6.json')
    assert previous_capture['capture']['source_binding_successor']['metadata_predecessor_context']==dict(path='coverage/contracts/vi4-public-matmul-capture-v4.json',sha256=H((ROOT/'coverage/contracts/vi4-public-matmul-capture-v4.json').read_bytes()))
    assert previous_provider['source_binding']['metadata_predecessor_contract']==dict(path='coverage/contracts/vi-public-matmul-provider-benchmark-v5.json',raw_sha256=H((ROOT/'coverage/contracts/vi-public-matmul-provider-benchmark-v5.json').read_bytes()))
    latest_capture=data('coverage/contracts/vi4-public-matmul-capture-v4.json')
    latest_provider=data('coverage/contracts/vi-public-matmul-provider-benchmark-v5.json')
    assert latest_capture['capture']['source_binding_successor']['metadata_predecessor_context']==dict(path='coverage/contracts/vi4-public-matmul-capture-v3.json',sha256=H((ROOT/'coverage/contracts/vi4-public-matmul-capture-v3.json').read_bytes()))
    assert latest_provider['source_binding']['metadata_predecessor_contract']==dict(path='coverage/contracts/vi-public-matmul-provider-benchmark-v4.json',raw_sha256=H((ROOT/'coverage/contracts/vi-public-matmul-provider-benchmark-v4.json').read_bytes()))
    retained_capture=data('coverage/contracts/vi4-public-matmul-capture-v3.json')
    retained_provider=data('coverage/contracts/vi-public-matmul-provider-benchmark-v4.json')
    assert retained_capture['capture']['source_binding_successor']['metadata_predecessor_context']==dict(path='coverage/contracts/vi4-public-matmul-capture-v2.json',sha256=H((ROOT/'coverage/contracts/vi4-public-matmul-capture-v2.json').read_bytes()))
    assert retained_provider['source_binding']['metadata_predecessor_contract']==dict(path='coverage/contracts/vi-public-matmul-provider-benchmark-v3.json',raw_sha256=H((ROOT/'coverage/contracts/vi-public-matmul-provider-benchmark-v3.json').read_bytes()))
    prior_capture=data('coverage/contracts/vi4-public-matmul-capture-v2.json')
    prior_provider=data('coverage/contracts/vi-public-matmul-provider-benchmark-v3.json')
    assert prior_capture['capture']['source_binding_successor']['metadata_predecessor_context']==dict(path='coverage/contracts/vi4-public-matmul-capture-v1.json',sha256=H((ROOT/'coverage/contracts/vi4-public-matmul-capture-v1.json').read_bytes()))
    assert prior_provider['source_binding']['metadata_predecessor_contract']==dict(path='coverage/contracts/vi-public-matmul-provider-benchmark-v2.json',raw_sha256=H((ROOT/'coverage/contracts/vi-public-matmul-provider-benchmark-v2.json').read_bytes()))
    assert module.context(ROOT)==current
    with pytest.raises(AssertionError,match='UNBOUND'):module.read_manifest(ROOT)
    assert module.CONTEXT==CONTEXT and module.PROVIDER==PROVIDER
    assert capture['provider_context']==PROVIDER
    route=current['first_cohort']['routes']['vi']
    assert route['contract']==PROVIDER and route['driver_route']==provider['route']
    assert current['families'][provider['report_family']]['contract']==PROVIDER
    assert current['benchmark_source_sha256']==provider['source_files_canonical_sha256']
    for name,pin in current['benchmark_source_sha256'].items():
        assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin,name
    for name,pin in current['first_cohort']['source_files_raw_sha256'].items():
        assert H((ROOT/name).read_bytes())==pin,name
    assert all(current[k]is None for k in('producer','prepared_source','SDK_members','core_sha256','runtime_sha256','benchmark_fingerprint'))
    assert all(current['execution_authority'][k]is None for k in('controller_root_admission','created_utc','deadline_utc','cohort_GO','measurement_GO'))
    budget=capture['campaign_budget']
    assert (budget['mandatory_cohort_pairs'],budget['measurement_preflight_pairs'])==(1,4)
    assert (budget['original_untimed'],budget['native_untimed'])==(5,5)
    assert (budget['native_log2_untimed_batches'],budget['native_matmul_untimed_calls'])==(10,10)
    assert (budget['accepted_raw_batches'],budget['warmup_groups'])==(56,8)
    assert current['execution_scope']==data(OLD_CONTEXT)['execution_scope']
    assert all(v is None for v in provider['unbound'].values())and provider['execution_authority']is None


def test_prior_scope_and_every_catcher_body_remain_exact():
    old=data(OLD_CONTEXT);current=data(CONTEXT);capture=current['capture']
    predecessor=capture['source_binding_successor']['predecessor_context']
    assert predecessor==dict(path=OLD_CONTEXT,sha256='42c44333aca6dc38298cc928ef815a1f36152d6660fd65ede4338f7f8342e232')
    assert H((ROOT/OLD_CONTEXT).read_bytes())==predecessor['sha256']
    for name in('cases','phase_lines','phase_rule','envelopes','failure_only_budget','success_extra_budget','failure_outputs','unavailable','cold_source_projection'):
        assert capture[name]==old['capture'][name],name
    budget=dict(capture['campaign_budget']);assert budget.pop('native_matmul_untimed_calls')==10
    assert budget==old['capture']['campaign_budget']
    prior=(ROOT/'tools/vi4_failure_capture.py').read_text(encoding='utf-8')
    actual=(ROOT/RUNNER).read_text(encoding='utf-8')
    replacements=[('coverage/contracts/vi4-failure-capture-v2.json',CONTEXT),('coverage/contracts/vi-public-log2-provider-benchmark-v2.json',PROVIDER),('tools/vi4_failure_capture.py',RUNNER),("capture['schema']=='vi4-failure-capture-v2'","capture['schema']=='vi4-public-matmul-capture-v1'"),('benchmarks/vi_public_log2_provider_execution.py',ADAPTER),('benchmarks/vi_public_log2_provider_consumer.py',CONSUMER)]
    for before,after in replacements:prior=prior.replace(before,after)
    assert actual==prior  # No capture/FP/sampler/error/identity gate was edited.
    frozen=(ROOT/'tools/vi4_failure_capture.py').read_text(encoding='utf-8')
    for name in('capture_failure','descriptor','failure_path','run_catching','run_cohorts','run_measurements','scoped_measurement','_post_identity'):
        assert function(actual,name)==function(frozen,name),name


def test_versioned_metadata_preserves_method_words_FP_engine_and_provider_policy():
    current=data(PROVIDER);old=data(OLD_PROVIDER)
    for name in('cases','case_count','method','raw_samples','warm_groups','FENV_dependency','providers','strict_preflight','timed_ownership','provider_guard_scope','reduction_scope','execution_authority','unbound','reused_execution_canonical_sha256','provider_policy'):
        assert current[name]==old[name],name
    assert current['route']['driver']==CONSUMER and current['route']['contract']==PROVIDER
    assert current['source_binding']['mode']=='literal-current-canonical-LF'
    assert current['source_binding']['predecessor_contract']==dict(path=OLD_PROVIDER,raw_sha256='98aa288de7a3bedb230f74e8629af20c86b9d90246cf98e0154c90980e1f4e1b')
    for name,pin in current['FENV_dependency']['source_files_raw_sha256'].items():assert H((ROOT/name).read_bytes())==pin,name
    assert H((ROOT/'benchmarks/vi_public_matmul_provider_execution.py').read_bytes())=='7a6a84c4caba8acdbc2f04850205f283a80460bd484743142229c12aa32777fa'
    assert H((ROOT/'benchmarks/variation_of_information_execution.py').read_bytes())=='93fe29097b3050947c45da2719dbb0f7f987efde72be259a2d43af93b89a8c4d'
    text=(ROOT/'benchmarks/vi_public_matmul_provider_execution.py').read_text(encoding='utf-8')
    assert "ORDER=('numpy.log2','numpy.matmul','numpy.log2','numpy.matmul')"in text


def test_new_namespace_uses_exact_existing_engine_and_guard_without_importing_science(monkeypatch):
    route=load(ADAPTER);current=route.executor();old=load('benchmarks/vi_public_matmul_provider_execution.py').executor()
    assert current.CONTEXT==PROVIDER and old.CONTEXT==OLD_PROVIDER
    assert current.execute.__code__.co_code==old.execute.__code__.co_code
    assert current.candidate_providers.__code__.co_code==old.candidate_providers.__code__.co_code
    assert current.boundary(ROOT).current_context(ROOT,PROVIDER)==data(PROVIDER)
    capture=load(RUNNER);fresh=capture.bound_probe(ROOT);prior=load('tools/vi_first_cohort_probe.py')
    assert fresh.CONTRACT==CONTEXT and fresh.DRIVER_CONTEXT==PROVIDER
    assert prior.CONTRACT=='coverage/contracts/vi-measurement-v5.json'
    for name in('main','verify_source_state','verify_helper_state','validate_native_entry','finish_probe','prepare_cohorts','measurement_commands','validate_reports'):
        assert getattr(fresh,name).__code__.co_code==getattr(prior,name).__code__.co_code,name
    engine=fresh.driver_adapter(ROOT)
    assert engine.CONTEXT==PROVIDER and engine.candidate_providers.__code__.co_code==old.candidate_providers.__code__.co_code
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks'));consumer=load(CONSUMER)
    monkeypatch.setattr(sys,'argv',['VI4-matmul-unbound'])
    with pytest.raises(SystemExit,match='VI_UNBOUND'):current.execute(consumer.__dict__)


@pytest.mark.parametrize('fault',('runner','dependency','schema'))
def test_current_capture_dependency_tampering_refuses(tmp_path,fault):
    module=load(RUNNER);current=data(CONTEXT)
    names=set(current['capture']['dependencies'])|{CONTEXT,RUNNER}
    for name in names:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    if fault=='schema':
        current['capture']['schema']='old-authority';(tmp_path/CONTEXT).write_text(json.dumps(current),encoding='utf-8')
    else:
        name=RUNNER if fault=='runner'else ADAPTER
        target=tmp_path/name;target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(AssertionError):module.context(tmp_path)


@pytest.mark.parametrize('fault',(None,'current','snapshot','prepared','old-contract'))
def test_actual_current_binding_requires_literal_current_not_predecessor(tmp_path,monkeypatch,fault):
    current=load(ADAPTER).executor();factory=load('tests/test_mark_boundaries_benchmark_execution.py')
    monkeypatch.setattr(factory,'source_fixture_bytes',lambda name,context_path:(ROOT/name).read_bytes())
    path,binding,contract=factory.synthetic(tmp_path,monkeypatch,context_path=ROOT/PROVIDER)
    binding['SDK_members']['include/skimagexx/label_pair_metrics.hpp']='f'*64
    runtime=Path(binding['runtime_path']);record=json.loads(runtime.read_bytes());record['runtime']['sdk_sha256']=binding['SDK_members']
    def refresh():
        record['source']=binding['prepared_source'];runtime.write_text(json.dumps(record),encoding='utf-8')
        binding['runtime_sha256']=H(runtime.read_bytes());path.write_text(json.dumps(binding),encoding='utf-8')
        go=Path(binding['root_go_path']);value=json.loads(go.read_bytes());value['binding_sha256']=H(path.read_bytes());go.write_text(json.dumps(value),encoding='utf-8')
    refresh();shared=current.boundary(ROOT)
    accepted=shared.validate_binding(path,root=ROOT,context_path=PROVIDER)
    assert accepted[0]==binding and accepted[1]==contract;current.require_sdk_dependencies(binding)
    expected=contract['source_files_canonical_sha256'][ADAPTER];snapshot=Path(binding['source_root'])
    assert H(shared.source_bytes(ROOT/ADAPTER))==H(shared.source_bytes(snapshot/ADAPTER))==binding['prepared_source']['files_sha256'][ADAPTER]==expected
    if fault is None:return
    if fault=='current':
        original=shared.source_bytes
        monkeypatch.setattr(shared,'source_bytes',lambda p:original(p)+b'bad'if Path(p)==ROOT/ADAPTER else original(p))
    elif fault=='snapshot':(snapshot/ADAPTER).write_bytes((snapshot/ADAPTER).read_bytes()+b'bad')
    elif fault=='prepared':binding['prepared_source']['files_sha256'][ADAPTER]='0'*64;refresh()
    else:binding['base_contract_sha256']=H((ROOT/OLD_PROVIDER).read_bytes());refresh()
    with pytest.raises(AssertionError):shared.validate_binding(path,root=ROOT,context_path=PROVIDER)


@pytest.mark.parametrize('bad_expected',(False,True))
def test_both_output_words_and_existing_caller_envelope_survive_failure(tmp_path,bad_expected):
    module=load(RUNNER);factory=load('tests/test_vi4_failure_capture.py')
    error,events,values=factory.observed_error(3,136,bad_expected=bad_expected)
    path=tmp_path/'failure.json';module.capture_failure(error,ROOT,'measurement',path,data(CONTEXT)['capture'])
    record=json.loads(path.read_bytes())
    assert record['primary']['type']=='AssertionError'and record['retry']is False
    assert record['actual']['words']['raw_hex']==values['actual'].raw.hex()
    assert ('words'in record['expected'])is(not bad_expected)
    assert values['args'][0].nbytes==524288 and len(record['input_after'])==2
    assert events.index('actual')<events.index('input0')
    assert (events.count('save'),events.count('restore'))==(3,2)  # Fakes only.
    assert record['FP']['before_codecs']is True and record['FP']['after_codecs']is True
    assert record['added_target_calls']==record['added_fixture_calls']==record['added_sampler_calls']==0


def test_success_path_has_no_capture_or_instrumentation(tmp_path,monkeypatch):
    module=load(RUNNER);events=[];script=tmp_path/'driver.py';argv=sys.argv;path=list(sys.path)
    monkeypatch.setattr(module.runpy,'run_path',lambda *a,**k:(events.append(a)or{'success':True}))
    monkeypatch.setattr(module,'capture_failure',lambda *a:pytest.fail('Success entered capture'))
    monkeypatch.setattr(sys,'setprofile',lambda *a:pytest.fail('Success changed profile'))
    assert module.run_catching(script,[str(script)],ROOT,'measurement',tmp_path/'never.json',data(CONTEXT)['capture'])=={'success':True}
    assert len(events)==1 and sys.argv is argv and sys.path==path and not(tmp_path/'never.json').exists()


def test_current_all_four_measurement_command_is_bounded_and_refuses_replay(tmp_path,monkeypatch):
    module=load(RUNNER);probe=module.bound_probe(ROOT)
    controller=tmp_path/'controller';snapshot=tmp_path/'snapshot';results=tmp_path/'results';results.mkdir();(controller/'coverage/contracts').mkdir(parents=True)
    manifest=data(CONTEXT);manifest['status']='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON';(controller/CONTEXT).write_text(json.dumps(manifest),encoding='utf-8')
    binding=results/'binding.json';binding.write_text('{}');authority={'binding':str(binding),'measurement':str(results/'measurement.json')}
    terminal=dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',api='metrics.variation_of_information',root_binding_sha256=module.sha(binding),counts=dict(original=1,native=1),timed_batches=0,sampler_body_calls=0,outer_state_restored=True,poststop_source_authority_unchanged=True)
    folder=results/'first-cohort/vi';folder.mkdir(parents=True);(folder/'terminal.json').write_text(json.dumps(terminal),encoding='utf-8');(results/'first-cohort-summary.json').write_text(json.dumps(dict(status='PASS',postbatch_offline_identity_verified=True,routes=[terminal])),encoding='utf-8')
    monkeypatch.setattr(module,'bound_probe',lambda root:probe);events=[]
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:events.append('before-verifier'))
    monkeypatch.setattr(module,'_child',lambda *a:events.append(a[-1]))
    module.run_measurements(snapshot,results,manifest,'python',{'vi':authority},'2099-01-01T00:00:00+00:00')
    command=events[1]
    assert events[0]=='before-verifier'and command[1]==str(snapshot/CONSUMER)
    assert '--case'not in command and '--first-cohort'not in command
    assert command[command.index('--threads')+1]=='1'and command[command.index('--repeat')+1]=='7'
    with pytest.raises(AssertionError):module.run_measurements(snapshot,results,manifest,'python',{'vi':authority},'2099-01-01T00:00:00+00:00')
    assert len(events)==2



def test_manual_workflow_is_exact_frozen_transform_and_retains_current_sources():
    # This predecessor view proves unchanged workflow bodies only. Current
    # runtime acceptance above always reads actual ROOT and literal V2 pins.
    from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
    from tools.vi_public_matmul_context import predecessor_bytes
    name='.github/workflows/vi-initial-failure-capture.yml';actual=(ROOT/name).read_bytes()
    previous=predecessor_bytes(ROOT,name,actual,VI_PUBLIC_MATMUL_CONTEXT_SHA256,canonical=False)
    assert H(previous)=='b0117717c438f5ccfe274228285927ec1c3e6f8082f22f65015c04ccd9001f4e'
    replacements=[(b'coverage/contracts/vi4-failure-capture-v2.json',CONTEXT.encode()),(b'tools/vi4_failure_capture.py',RUNNER.encode()),(b'tests/test_vi4_failure_capture.py',b'tests/test_vi4_public_matmul_capture.py')]
    assert [previous.count(old)for old,new in replacements]==[4,4,1]
    for old,new in replacements:previous=previous.replace(old,new)
    assert actual==previous
    text=actual.decode('utf-8')
    assert text.index('ROOT_PREPARATION_CAPACITY_BEGIN')<text.index('actions/checkout')<text.index('actions/artifacts/')
    assert "['capture']['dependencies']"in text and 'if: always()'in text and 'path: results/'in text
    assert "probe['run_cohorts']"in text and "probe['run_measurements']"in text and "probe['validate_reports']"in text
    assert 'NPY_DISABLE_CPU_FEATURES='not in text
    assert name in data(PROVIDER)['source_files_canonical_sha256']and name in data(CONTEXT)['capture']['dependencies']
