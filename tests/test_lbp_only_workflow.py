"""Manual LBP-only workflow source and inert host-writer controls, no science."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/lbp-only-initial.yml'
RECEIPT = 'coverage/evidence/lbp-only-comparison-root-source-admission-v1.json'
RECEIPT_SHA = '806378f6a67e9dd082f490b83b4490392c6bb532ba1b507bc66ef4c80b759f6d'


def run_blocks(path):
    blocks = []; current = None
    for line in path.read_text().splitlines():
        if line == '        run: |':
            assert current is None; current = []
        elif current is not None:
            if line.startswith('          '): current.append(line[10:])
            else:
                blocks.append('\n'.join(current)+'\n'); current = None
    if current is not None: blocks.append('\n'.join(current)+'\n')
    return blocks


def python_blocks(path):
    output = []
    for shell in run_blocks(path):
        lines = shell.splitlines(); start = next(i for i,s in enumerate(lines) if " - <<'PY'" in s)
        end = lines.index('PY', start+1)
        output.append('\n'.join(lines[start+1:end])+'\n')
    return output


def test_actual_inline_compile_offline_setup_and_parent_contract_inverse():
    new = python_blocks(WORKFLOW); old = python_blocks(ROOT/'.github/workflows/vi-initial.yml')
    assert len(new) == len(old) == 5
    for i, text in enumerate(new): compile(text, '<actual LBP workflow block '+str(i)+'>', 'exec')
    assert new[1] == old[1].replace('coverage/contracts/vi-measurement-v3.json', 'coverage/contracts/mark-lbp-measurement-v5.json')
    first = new[0].replace('import hashlib, json, os, re, subprocess','import json, os, re, subprocess')
    guards = [line for line in first.splitlines(True) if line.startswith("assert os.environ['LBP_ONLY_AUTHORITY']") or line.startswith("assert hashlib.sha256(Path('controller/")]
    assert len(guards) == 4
    for guard in guards: first = first.replace(guard, '', 1)
    assert first == old[0].replace('coverage/contracts/vi-measurement-v3.json', 'coverage/contracts/mark-lbp-measurement-v5.json')
    # All original archive/14wheel/root/stager/runtime/SDK/identity checks stay exact.
    assert run_blocks(WORKFLOW)[1] == run_blocks(ROOT/'.github/workflows/vi-initial.yml')[1].replace('coverage/contracts/vi-measurement-v3.json','coverage/contracts/mark-lbp-measurement-v5.json')
    assert hashlib.sha256((ROOT/RECEIPT).read_bytes()).hexdigest() == RECEIPT_SHA
    receipt = json.loads((ROOT/RECEIPT).read_bytes())
    assert receipt['admitted_controller_sha256'] == hashlib.sha256((ROOT/'tools/lbp_only_comparison.py').read_bytes()).hexdigest()
    source = WORKFLOW.read_text()
    assert '        default: UNBOUND\n' in source and '  workflow_dispatch:\n' in source
    assert 'push:' not in source and 'matrix:' not in source and 'continue-on-error' not in source
    assert 'name: lbp-only-initial-minimum' in source and '          path: results/\n' in source
    assert '        if: always()\n' in source and '          include-hidden-files: true\n' in source
    assert 'prepare_cohorts' not in source and 'run_cohorts' not in source and 'run_measurements' not in source
    assert new[3].count("adapter['prepare_cohort'](") == new[3].count("adapter['run_cohort'](") == 1
    assert new[4].count("adapter['run_measurement'](") == 1


def test_all_actual_shell_blocks_parse_without_execution(tmp_path):
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    assert bash.is_file(), 'Existing native Git Bash required; never install'
    for index, shell in enumerate(run_blocks(WORKFLOW)):
        path = tmp_path / ('actual-step-'+str(index)+'.sh')
        path.write_text(shell, encoding='utf-8', newline='\n')
        result = subprocess.run([str(bash), '-n', str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_default_UNBOUND_first_block_refuses_before_API_source_or_helpers(monkeypatch):
    monkeypatch.setenv('LBP_ONLY_AUTHORITY', 'UNBOUND')
    monkeypatch.setattr(subprocess,'check_output', lambda *a,**k:pytest.fail('No API/process authorized'))
    with pytest.raises(AssertionError,match='UNBOUND'):
        exec(compile(python_blocks(WORKFLOW)[0],'<actual upfront refusal>','exec'),{})


def prepared_controller(tmp_path, monkeypatch, mode):
    workspace = tmp_path/'workspace'; controller = workspace/'controller'
    snapshot = workspace/'snapshot'; results = workspace/'results'
    for folder in (controller,snapshot,results): folder.mkdir(parents=True)
    context = json.loads((ROOT/'coverage/contracts/lbp-only-comparison-v1.json').read_bytes())
    names = [*context['dependencies_sha256'], 'tools/lbp_only_comparison.py',
             'coverage/contracts/lbp-only-comparison-v1.json', RECEIPT,
             'coverage/contracts/mark-boundaries-benchmark-execution-v6.json']
    for name in names:
        target = controller/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,target)
    now = datetime.now(timezone.utc)
    created, deadline = now-timedelta(minutes=1), now+timedelta(minutes=20)
    if mode == 'expired': created,deadline = now-timedelta(minutes=20),now-timedelta(minutes=1)
    clock = dict(created_utc=created.isoformat(),deadline_utc=deadline.isoformat())
    (results/'execution-clock.json').write_text(json.dumps(clock))
    monkeypatch.setenv('GITHUB_WORKSPACE',str(workspace)); monkeypatch.setenv('GITHUB_SHA','f'*40)
    monkeypatch.setenv('GITHUB_ENV',str(results/'github-env'))
    monkeypatch.setenv('LBP_ONLY_AUTHORITY','UNBOUND' if mode=='UNBOUND' else 'ROOT_AUTHORIZED_LBP_ONLY_COMPARISON')
    for key,name in [('EXECUTION_CREATED_UTC','created_utc'),('DEADLINE_UTC','deadline_utc')]:
        monkeypatch.setenv(key,clock[name].replace('+00:00','Z') if mode=='Z' else clock[name])
    if mode == 'receipt-tampered': (controller/RECEIPT).write_text('{}')
    return workspace,controller,snapshot,results


@pytest.mark.parametrize('mode', ['authorized','Z','UNBOUND','expired','receipt-tampered'])
def test_actual_root_writer_authenticates_before_helper_with_one_way_GO(tmp_path,monkeypatch,mode):
    _,controller,snapshot,results = prepared_controller(tmp_path,monkeypatch,mode)
    original = runpy.run_path; calls = []
    def namespace(path):
        assert Path(path) == controller/'tools/lbp_only_comparison.py'
        adapter = original(str(path)); authorize = adapter['authorize']
        def checked(root,go,pin):
            # Run the actual admitted authorize with real copied parent/context/receipt bytes.
            outer,selected,decision = authorize(root,go,pin)
            assert selected['execution_scope']['canonical_cases'] == 4
            assert selected['execution_scope']['canonical_raw_samples'] == 56
            assert selected['execution_scope']['warmup_groups'] == 8
            assert set(selected['first_cohort']['routes']) == {'lbp'}
            assert decision['MARK_authorized'] is False
            assert 'root_GO_sha256' not in decision and 'binding_sha256' not in decision
            def helper(s,r,python,manifest):
                assert s == snapshot and r == results and manifest == selected
                calls.append('inert-helper-substitute')
                return dict(UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT='inert',UTIL_VIEWS_FENV_RECEIPT_SHA256='0'*64)
            outer.prepare_host_helper = helper
            return outer,selected,decision
        adapter['authorize'] = checked
        return adapter
    monkeypatch.setattr(runpy,'run_path',namespace)
    if mode in ('UNBOUND','expired','receipt-tampered'):
        with pytest.raises(AssertionError): exec(compile(python_blocks(WORKFLOW)[2],'<actual conditional writer>','exec'),{})
        assert not calls and not (results/'lbp-only-root-GO.json').exists()
    else:
        exec(compile(python_blocks(WORKFLOW)[2],'<actual conditional writer>','exec'),{})
        assert calls == ['inert-helper-substitute']
        path = results/'lbp-only-root-GO.json'
        assert (results/'lbp-only-root-GO.sha256').read_text().strip() == hashlib.sha256(path.read_bytes()).hexdigest()
        assert json.loads(path.read_bytes())['workflow_source'] == 'f'*40
        pins=json.loads((results/'controller-source-pins.json').read_bytes())['files_raw_sha256']
        assert len(pins)==5
        for name,pin in pins.items():
            assert (results/'controller-source'/name).read_bytes()==(controller/name).read_bytes()
            assert hashlib.sha256((results/'controller-source'/name).read_bytes()).hexdigest()==pin
        assert 'UTIL_VIEWS_FENV_MODE=source-built-v1' in (results/'github-env').read_text()
        with pytest.raises(FileExistsError): exec(compile(python_blocks(WORKFLOW)[2],'<no writer replay>','exec'),{})
        assert len(calls) == 1


@pytest.mark.parametrize('cohort_pass', [True, False])
def test_actual_host_steps_fixed_LBP_interface_no_MARK_or_timing_before_cohort(tmp_path,monkeypatch,cohort_pass):
    _,controller,snapshot,results = prepared_controller(tmp_path,monkeypatch,'authorized')
    (results/'lbp-only-root-GO.json').write_text('{}')
    (results/'lbp-only-root-GO.sha256').write_text('external-pin')
    calls = []; authority = dict(binding='only-lbp-binding',go='only-lbp-GO',measurement='only-lbp-measurement')
    def prepare(c,s,r,python,site,go,pin):
        assert (c,s,r)==(controller,snapshot,results) and pin=='external-pin'
        calls.append('prepare-lbp-only')
        (results/'cohort-authorities.json').write_text(json.dumps({'lbp':authority}))
        return authority
    def cohort(c,r,python,value,go,pin):
        assert value == authority
        calls.append('real-entry-cohort-substitute')
        assert cohort_pass, 'C1/postbatch failed: timing forbidden'
        (results/'first-cohort-summary.json').write_text('{"inert_postbatch":true}')
    def measure(c,s,r,python,value,go,pin):
        assert value == authority and (results/'first-cohort-summary.json').exists()
        calls.append('four-case-measurement-substitute')
    monkeypatch.setattr(runpy,'run_path',lambda path:dict(prepare_cohort=prepare,run_cohort=cohort,run_measurement=measure))
    if cohort_pass:
        exec(compile(python_blocks(WORKFLOW)[3],'<actual cohort step>','exec'),{})
        exec(compile(python_blocks(WORKFLOW)[4],'<actual timing step>','exec'),{})
        assert calls==['prepare-lbp-only','real-entry-cohort-substitute','four-case-measurement-substitute']
    else:
        with pytest.raises(AssertionError): exec(compile(python_blocks(WORKFLOW)[3],'<actual failed cohort>','exec'),{})
        assert calls==['prepare-lbp-only','real-entry-cohort-substitute']
        assert not (results/'first-cohort-summary.json').exists()
