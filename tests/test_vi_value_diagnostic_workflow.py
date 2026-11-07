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


def python_blocks(path):
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
    return blocks


def test_actual_inline_Python_compiles_and_setup_is_preserved():
    new=python_blocks(WORKFLOW);old=python_blocks(ROOT/'.github/workflows/vi-initial.yml')
    assert len(new)==4 and len(old)==5
    for index,text in enumerate(new):compile(text,'<new actual workflow block '+str(index)+'>','exec')
    assert new[1:3]==old[1:3]  # exact offline14wheel/stager+F1 helper steps
    first=new[0].replace('import hashlib, json, os, re, subprocess','import json, os, re, subprocess')
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
    assert 'raise SystemExit(done.returncode)'in new[3]
    assert new[3].count("probe['prepare_cohorts'](")==1
    tree=ast.parse(new[3]);runs=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='run']
    assert len(runs)==1


def test_default_UNBOUND_real_first_block_refuses_before_source_API_or_helpers(monkeypatch):
    monkeypatch.setenv('DIAGNOSTIC_AUTHORITY','UNBOUND')
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:pytest.fail('API/process reached'))
    with pytest.raises(AssertionError,match='UNBOUND'):
        exec(compile(python_blocks(WORKFLOW)[0],'<actual first workflow block>','exec'),{})


@pytest.mark.parametrize('mode',('failure-exit','success-stop','success-Z','expired','UNBOUND'))
def test_actual_host_GO_writer_one_way_wrapper_interface_and_exit_retention(tmp_path,monkeypatch,mode):
    workspace=tmp_path/'workspace';controller=workspace/'controller';snapshot=workspace/'snapshot';results=workspace/'results'
    for directory in (controller,snapshot,results):directory.mkdir(parents=True)
    context_path='coverage/contracts/vi-cohort-value-diagnostic-v1.json'
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
    monkeypatch.setenv('GITHUB_WORKSPACE',str(workspace));monkeypatch.setenv('GITHUB_SHA','a'*40)
    monkeypatch.setenv('EXECUTION_CREATED_UTC',clock['created_utc'].replace('+00:00','Z')if mode=='success-Z'else clock['created_utc']);monkeypatch.setenv('DEADLINE_UTC',clock['deadline_utc'].replace('+00:00','Z')if mode=='success-Z'else clock['deadline_utc'])
    monkeypatch.setenv('DIAGNOSTIC_AUTHORITY','UNBOUND'if mode=='UNBOUND'else'ROOT_AUTHORIZED_ONE_VI_VALUE_DIAGNOSTIC')
    manifest=json.loads((controller/'coverage/contracts/vi-measurement-v3.json').read_bytes());setup=[]
    def prepare(c,s,r,m,python,site,until):
        setup.append('metadata-setup-only')
        assert m is manifest and until==clock['deadline_utc']
        folder=results/'authority-vi';folder.mkdir()
        binding=folder/'binding.json';binding.write_text(json.dumps(dict(source_root=str(snapshot),source_revision=context['qualified_source'])))
        go=folder/'root-go.json';go.write_text('{"inert_cohort_GO_only":true}')
        return {'vi':dict(binding=str(binding),go=str(go),measurement=str(folder/'unused-measurement.json'))}
    def old_namespace(path):
        assert Path(path)==controller/'tools/vi_first_cohort_probe.py'
        return dict(read_manifest=lambda root:(controller/'coverage/contracts/vi-measurement-v3.json',manifest),prepare_cohorts=prepare)
    monkeypatch.setattr(runpy,'run_path',old_namespace)
    spec=importlib.util.spec_from_file_location('_unchanged_wrapper_interface',ROOT/'tools/vi_cohort_value_diagnostic.py')
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    calls=[]
    def inert_child(command,**kwargs):
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
        return SimpleNamespace(returncode=1 if mode=='failure-exit'else 0)
    monkeypatch.setattr(subprocess,'run',inert_child)
    if mode in('expired','UNBOUND'):
        with pytest.raises(AssertionError):exec(compile(python_blocks(WORKFLOW)[3],'<actual diagnostic workflow block>','exec'),{})
        assert not setup and not calls and not(results/'vi-value-diagnostic-exit.json').exists()
    else:
        with pytest.raises(SystemExit)as exit_status:exec(compile(python_blocks(WORKFLOW)[3],'<actual diagnostic workflow block>','exec'),{})
        expected=1 if mode=='failure-exit'else 0
        assert exit_status.value.code==expected
        assert json.loads((results/'vi-value-diagnostic-exit.json').read_bytes())==dict(exit=expected)
        assert setup==['metadata-setup-only']and len(calls)==1
        assert (results/'vi-value-diagnostic-command.json').is_file()
        assert not(results/'summary.json').exists()and not list(results.rglob('*.batches.jsonl'))
