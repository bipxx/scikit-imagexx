"""New manual workflow source/inert writer controls, zero target/helper calls."""
import argparse
import ast
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import shutil
import subprocess
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=ROOT/'.github/workflows/vi-value-diagnostic.yml'


def python_blocks(path, *, include_capacity=False):
    blocks=[];current=None
    for line in path.read_text().splitlines():
        if line.startswith('          ')and " - <<'PY'"in line:
            assert current is None;current=[]
        elif current is not None:
            if line=='          PY':blocks.append('\n'.join(current)+'\n');current=None
            else:
                assert line.startswith('          ')
                current.append(line[10:])
    assert current is None
    if include_capacity:
        return blocks
    return [block for block in blocks if '# ROOT_PREPARATION_CAPACITY_BEGIN' not in block]


@pytest.mark.parametrize('workspace_gib,runtime_gib,passed', [(32,32,True),(29,40,False),(40,29,False)])
def test_capacity_guard_refuses_before_checkout_and_transport(tmp_path,monkeypatch,workspace_gib,runtime_gib,passed):
    source=WORKFLOW.read_text(encoding='utf-8')
    initial_source=(ROOT/'.github/workflows/vi-initial.yml').read_text(encoding='utf-8')
    assert '          mkdir results\n' not in source and '          mkdir results\n' not in initial_source
    assert source.count('          mkdir -p results\n')==initial_source.count('          mkdir -p results\n')==2
    assert source.index('- name: Require scratch capacity before preparation') < source.index('- uses: actions/checkout@v4')
    assert source.index('- uses: actions/checkout@v4') < source.index('gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$ARTIFACT_ID/zip"')
    blocks=python_blocks(WORKFLOW,include_capacity=True)
    capacity=[block for block in blocks if '# ROOT_PREPARATION_CAPACITY_BEGIN' in block]
    assert len(blocks)==5 and len(capacity)==1
    initial=python_blocks(ROOT/'.github/workflows/vi-initial.yml',include_capacity=True)
    assert len(initial)==6 and [block for block in initial if '# ROOT_PREPARATION_CAPACITY_BEGIN' in block]==capacity
    workspace=tmp_path/'workspace';runtime=tmp_path/'runtime'
    workspace.mkdir();runtime.mkdir();(workspace/'results').mkdir()
    monkeypatch.chdir(workspace);monkeypatch.setenv('RUNNER_TEMP',str(runtime))
    observed=[]
    def disk_usage(path):
        path=Path(path)
        assert path in (workspace,runtime)
        observed.append(path)
        return SimpleNamespace(free=(runtime_gib if path==runtime else workspace_gib)*1024**3)
    monkeypatch.setattr(shutil,'disk_usage',disk_usage)
    if passed:
        exec(compile(capacity[0],'<actual capacity guard>','exec'),{})
    else:
        with pytest.raises(AssertionError,match='Insufficient scratch capacity'):
            exec(compile(capacity[0],'<actual capacity guard>','exec'),{})
    record=json.loads((workspace/'results/preparation-capacity.json').read_bytes())
    assert record['passed'] is passed and observed==[workspace,runtime]
    assert record['estimated_peak_bytes']==4*1024**3 and record['reserve_bytes']==20*1024**3
    assert record['required_free_bytes']==record['minimum_free_floor_bytes']==30*1024**3
    assert record['original_calls']==record['native_calls']==0


def test_actual_inline_Python_compiles_and_setup_is_preserved():
    new=python_blocks(WORKFLOW);old=python_blocks(ROOT/'.github/workflows/vi-initial.yml')
    assert len(new)==4 and len(old)==5
    for index,text in enumerate(new):compile(text,'<new actual workflow block '+str(index)+'>','exec')
    assert new[1:3]==old[1:3]  # exact offline14wheel/stager+F1 helper steps
    first=new[0].split('# ROOT_CURRENT_PRODUCT_HOST_GATE_BEGIN\n',1)[0].replace('import hashlib, json, os, re, subprocess','import json, os, re, subprocess')
    guards=[line for line in first.splitlines(True)if line.startswith("assert os.environ['DIAGNOSTIC_AUTHORITY']")or line.startswith("assert hashlib.sha256(Path('controller/")]
    assert len(guards)==3
    for line in guards:first=first.replace(line,'',1)
    assert first==old[0]  # all existing API/product/root/runtime/core/SDK gates exact
    source=WORKFLOW.read_text()
    assert '        default: UNBOUND\n'in source and '  workflow_dispatch:\n'in source
    assert '  diagnose:\n'in source and 'matrix:'not in source
    assert 'name: vi-value-diagnostic-minimum'in source and '          path: results/\n'in source
    assert '        if: always()\n'in source and 'continue-on-error'not in source
    assert 'run_measurements'not in source and 'validate_reports'not in source and "probe['run_cohorts']"not in source
    assert 'raise SystemExit(done.returncode if done.returncode else verified.returncode)'in new[3]
    assert new[3].count("verified=subprocess.run(")==1
    assert source.count("        if: env.VI_HOST_ELIGIBLE == '1'\n")==4
    assert new[3].count("probe['prepare_cohorts'](")==1
    tree=ast.parse(new[3]);runs=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='run']
    assert len(runs)==2


def test_default_UNBOUND_real_first_block_refuses_before_source_API_or_helpers(monkeypatch):
    monkeypatch.setenv('DIAGNOSTIC_AUTHORITY','UNBOUND')
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:pytest.fail('API/process reached'))
    with pytest.raises(AssertionError,match='UNBOUND'):
        exec(compile(python_blocks(WORKFLOW)[0],'<actual first workflow block>','exec'),{})


@pytest.mark.parametrize('mode',('failure-exit','success-stop','success-Z','expired','UNBOUND','ineligible-host','verify-failure','primary-and-verify-failure'))
def test_actual_host_GO_writer_one_way_wrapper_interface_and_exit_retention(tmp_path,monkeypatch,mode):
    workspace=tmp_path/'workspace';controller=workspace/'controller';snapshot=workspace/'snapshot';results=workspace/'results'
    for directory in (controller,snapshot,results):directory.mkdir(parents=True)
    context_path='coverage/contracts/vi-cohort-value-diagnostic-v4.json'
    context=json.loads((ROOT/context_path).read_bytes())
    for name in [*context['controller_dependencies'],'tools/vi_cohort_value_diagnostic.py',context_path]:
        destination=controller/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,destination)
    for name in context['prepared_dependencies']:
        destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,destination)
    created=datetime.now(timezone.utc)-timedelta(minutes=1)
    deadline=datetime.now(timezone.utc)+timedelta(minutes=30)
    if mode=='expired':created=datetime.now(timezone.utc)-timedelta(minutes=31);deadline=datetime.now(timezone.utc)-timedelta(minutes=1)
    clock=dict(created_utc=created.isoformat(),deadline_utc=deadline.isoformat())
    (results/'execution-clock.json').write_text(json.dumps(clock))
    (results/'host-eligibility.json').write_text(json.dumps(dict(eligible=mode!='ineligible-host')))
    monkeypatch.setenv('GITHUB_WORKSPACE',str(workspace));monkeypatch.setenv('GITHUB_SHA','a'*40)
    monkeypatch.setenv('EXECUTION_CREATED_UTC',clock['created_utc'].replace('+00:00','Z')if mode=='success-Z'else clock['created_utc']);monkeypatch.setenv('DEADLINE_UTC',clock['deadline_utc'].replace('+00:00','Z')if mode=='success-Z'else clock['deadline_utc'])
    monkeypatch.setenv('DIAGNOSTIC_AUTHORITY','UNBOUND'if mode=='UNBOUND'else'ROOT_AUTHORIZED_ONE_VI_VALUE_DIAGNOSTIC')
    manifest=json.loads((controller/'coverage/contracts/vi-measurement-v5.json').read_bytes());setup=[]
    def prepare(c,s,r,m,python,site,until):
        setup.append('metadata-setup-only')
        assert m is manifest and until==clock['deadline_utc']
        folder=results/'authority-vi';folder.mkdir()
        binding=folder/'binding.json';binding.write_text(json.dumps(dict(source_root=str(snapshot),source_revision=context['qualified_source'])))
        go=folder/'root-go.json';go.write_text('{"inert_cohort_GO_only":true}')
        return {'vi':dict(binding=str(binding),go=str(go),measurement=str(folder/'unused-measurement.json'))}
    def old_namespace(path):
        assert Path(path)==controller/'tools/vi_first_cohort_probe.py'
        return dict(read_manifest=lambda root:(controller/'coverage/contracts/vi-measurement-v5.json',manifest),prepare_cohorts=prepare)
    monkeypatch.setattr(runpy,'run_path',old_namespace)
    spec=importlib.util.spec_from_file_location('_unchanged_wrapper_interface',ROOT/'tools/vi_cohort_value_diagnostic.py')
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    calls=[];verifiers=[]
    def inert_child(command,**kwargs):
        if command[1]==str(snapshot/'tools/ci_measurement_artifacts.py'):
            assert command==[__import__('sys').executable,str(snapshot/'tools/ci_measurement_artifacts.py'),'verify',
                             '--wheel-dir',str(workspace/'stage/wheels'),'--artifact',str(workspace/'stage/runtime.json')]
            verifiers.append(command)
            return SimpleNamespace(returncode=7 if mode in('verify-failure','primary-and-verify-failure')else 0)
        calls.append(command)
        pairs=dict(zip(command[2::2],command[3::2]))
        args=argparse.Namespace(binding=Path(pairs['--binding']),root_go=Path(pairs['--root-go']),
            diagnostic_go=Path(pairs['--diagnostic-go']),diagnostic_go_sha256=pairs['--diagnostic-go-sha256'],output=Path(pairs['--output']))
        # Execute the actual new wrapper's boundary, no old probe/FP/helper body.
        assert wrapper.validate_authority(args,controller)[1]==snapshot
        decision=json.loads(args.diagnostic_go.read_bytes())
        assert decision['maximum_original_topcalls']==decision['maximum_native_topcalls']==1
        assert decision['sampler_body_calls']==decision['timed_batches']==0
        assert 'diagnostic_go_sha256'not in json.loads(args.binding.read_bytes())
        args.output.write_text('{"inert_only":true,"actual_target_calls":0}')
        return SimpleNamespace(returncode=1 if mode in('failure-exit','primary-and-verify-failure')else 0)
    monkeypatch.setattr(subprocess,'run',inert_child)
    if mode in('expired','UNBOUND','ineligible-host'):
        with pytest.raises(AssertionError):exec(compile(python_blocks(WORKFLOW)[3],'<actual diagnostic workflow block>','exec'),{})
        assert not setup and not calls and not verifiers and not(results/'vi-value-diagnostic-exit.json').exists()
    else:
        with pytest.raises(SystemExit)as exit_status:exec(compile(python_blocks(WORKFLOW)[3],'<actual diagnostic workflow block>','exec'),{})
        expected_diagnostic=1 if mode in('failure-exit','primary-and-verify-failure')else 0
        expected_verifier=7 if mode in('verify-failure','primary-and-verify-failure')else 0
        assert exit_status.value.code==(expected_diagnostic or expected_verifier)
        assert json.loads((results/'vi-value-diagnostic-exit.json').read_bytes())==dict(exit=expected_diagnostic)
        assert json.loads((results/'vi-postbatch-identity-exit.json').read_bytes())==dict(exit=expected_verifier)
        assert setup==['metadata-setup-only']and len(calls)==len(verifiers)==1
        assert (results/'vi-value-diagnostic-command.json').is_file()
        assert not(results/'summary.json').exists()and not list(results.rglob('*.batches.jsonl'))


@pytest.mark.parametrize('variant',('eligible','other-model','missing-feature','other-machine','other-system'))
def test_current_genuine_host_gate_is_metadata_only_and_zero_target(tmp_path,monkeypatch,variant):
    import os
    monkeypatch.chdir(tmp_path)
    (tmp_path/'results').mkdir()
    relative='coverage/contracts/vi-cohort-value-diagnostic-v4.json'
    context=tmp_path/'controller'/relative;context.parent.mkdir(parents=True)
    shutil.copyfile(ROOT/relative,context)
    policy=json.loads(context.read_bytes())['host_eligibility_proposal']
    assert policy['purpose']=='CURRENT16_GENUINE_PRODUCT_VALUE_WITNESS'
    features=list(policy['required_features'])
    if variant=='missing-feature':features.pop()
    model='Different supported x86 CPU'if variant=='other-model'else'Intel(R) Xeon(R) 6975P-C'
    cpu='\n'.join(('vendor_id : GenuineIntel','model name : '+model,
        'cpu family : 6','model : 173','stepping : 1','flags : '+' '.join(features)))
    read_text=Path.read_text
    def read(path,*args,**kwargs):
        return cpu if path==Path('/proc/cpuinfo')else read_text(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    host=SimpleNamespace(sysname='Darwin'if variant=='other-system'else'Linux',machine='aarch64'if variant=='other-machine'else'x86_64')
    monkeypatch.setattr(os,'uname',lambda:host,raising=False)
    monkeypatch.setenv('GITHUB_ENV',str(tmp_path/'github-env'))
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('target/verifier/process reached'))
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:pytest.fail('API/process reached'))
    block=python_blocks(WORKFLOW)[0].split('# ROOT_CURRENT_PRODUCT_HOST_GATE_BEGIN\n',1)[1]
    exec(compile(block,'<actual current-product metadata host gate>','exec'),dict(Path=Path,json=json,os=os))
    record=json.loads((tmp_path/'results/host-eligibility.json').read_bytes())
    eligible=variant in('eligible','other-model')
    assert record['eligible']is eligible and record['prior_failed_host_causal_credit']is False
    assert record['original_calls']==record['native_calls']==record['sampler_body_calls']==record['timed_batches']==0
    assert (tmp_path/'github-env').read_text()=='VI_HOST_ELIGIBLE='+('1'if eligible else'0')+'\n'
    assert not(tmp_path/'stage').exists()and not(tmp_path/'results/authority-vi').exists()


def test_old_failed_host_policy_and_local_private_runtime_are_not_relabelled():
    old=json.loads((ROOT/'coverage/contracts/vi-cohort-value-diagnostic-v2.json').read_bytes())
    current=json.loads((ROOT/'coverage/contracts/vi-cohort-value-diagnostic-v4.json').read_bytes())
    assert old['host_eligibility_proposal']['failed_CPU']['model']=='AMD EPYC 9V45 96-Core Processor'
    assert current['host_eligibility_proposal']['historical_failed_CPU']==old['host_eligibility_proposal']['failed_CPU']
    assert current['host_eligibility_proposal']['prior_failed_host_causal_credit']is False
    assert current['local_private_witness']['python']=='3.11.14'and current['local_private_witness']['genuine_runtime_credit']is False
    assert current['qualified_source']=='16a5b50193125199ad26cf5a89acec5babf5290c'
    assert current['actual_authority']is None and current['original_budget_now']==0
