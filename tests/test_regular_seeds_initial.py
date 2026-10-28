"""Inert one-product timing transport checks; no scientific imports or calls."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import statistics
import textwrap
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / '.github/workflows/regular-seeds-initial.yml').read_text()
SCRIPTS = [textwrap.dedent(s) for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n", WORKFLOW, re.S)]


def test_isolated_one_product_transport_and_fixed_closure():
    assert len(SCRIPTS) == 3
    for s in SCRIPTS: compile(s, 'workflow-inline', 'exec')
    for name in ('producer_run', 'producer_revision', 'artifact_id', 'artifact_sha256'):
        assert name + ':\n        type: string\n        required: true' in WORKFLOW
    assert 'default:' not in WORKFLOW
    timed = WORKFLOW.split('      - name: Measure', 1)[1].split('      - name:', 1)[0]
    assert timed.count('benchmarks/regular_seeds.py --threads 1 --repeat 7') == 1
    assert timed.count('ci_measurement_artifacts.py verify') == 2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__|setup.py', timed)
    assert '--no-index --no-deps --only-binary=:all: --require-hashes' in WORKFLOW
    assert "== 168" in SCRIPTS[1] and "== 79" in SCRIPTS[1]
    assert "== 242" in SCRIPTS[1] and "== 3" in SCRIPTS[1]
    assert 'expected_nodes = len(factory_cases) * 2 + 3' in SCRIPTS[1]
    assert '== expected_nodes' in SCRIPTS[1]
    assert 'if: always()' in WORKFLOW and 'name: regular-seeds-initial-minimum' in WORKFLOW
    assert 'lscpu > results/cpu.txt' in WORKFLOW
    assert 'core.regular_seeds_apply' not in WORKFLOW  # no entry replacement
    assert '5db8fb5751ba962440cc79cdf6990a6a447f97d3877b28893376c62cee9f22de' in SCRIPTS[1]
    assert '09c4dd70815a31667539904d582a3a9bcc736c4b82c956f78cadb184d416a3e3' in SCRIPTS[1]


@pytest.mark.parametrize('bad', [None, 'revision', 'run', 'repository', 'workflow', 'failed', 'artifact', 'digest', 'expired', 'foreign-artifact'])
def test_actual_producer_authentication(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path); (tmp_path / 'results').mkdir()
    env = dict(PRODUCER_RUN='123', PRODUCER_REVISION='a'*40, ARTIFACT_ID='456', ARTIFACT_SHA256='b'*64, GITHUB_REPOSITORY='bipxx/scikit-imagexx')
    record = dict(id=123, head_sha='a'*40, head_repository={'full_name':env['GITHUB_REPOSITORY']}, path='.github/workflows/ci.yml', status='completed', conclusion='success')
    item = dict(id=456, name='ci-performance-minimum', digest='sha256:'+'b'*64, expired=False, size_in_bytes=42, workflow_run=dict(id=123, head_sha='a'*40))
    if bad == 'revision': record['head_sha']='c'*40
    elif bad == 'run': record['id']=124
    elif bad == 'repository': record['head_repository']['full_name']='other/repo'
    elif bad == 'workflow': record['path']='other.yml'
    elif bad == 'failed': record['conclusion']='failure'
    elif bad == 'artifact': item['id']=457
    elif bad == 'digest': item['digest']='sha256:'+'c'*64
    elif bad == 'expired': item['expired']=True
    elif bad == 'foreign-artifact': item['workflow_run']['head_sha']='c'*40
    replies=[record,dict(total_count=1,artifacts=[item])]
    import subprocess
    monkeypatch.setattr(subprocess,'check_output',lambda args: json.dumps(replies.pop(0)).encode())
    for k,v in env.items(): monkeypatch.setenv(k,v)
    if bad:
        with pytest.raises(AssertionError): exec(SCRIPTS[0], {})
    else:
        exec(SCRIPTS[0], {}); assert json.loads((tmp_path/'results/selection.json').read_text()) == item


def validator():
    node = next(n for n in ast.parse(SCRIPTS[2]).body if isinstance(n, ast.FunctionDef))
    ns = dict(math=math, statistics=statistics)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-report-validator','exec'),ns)
    return ns['validate_report']


def fixture():
    source=dict(source_sha256='fixed',source_hash_schema='schema',source_hash_scope='scope')
    meta=dict(**source, source_before=deepcopy(source), threads=1,repeat=7,target_ms=19,raw_samples=126,reference_pool_limits={'OMP':'1'},fallback_enabled=False,selected_apis=['util.regular_seeds'],harness_sha256='h',original_source_sha256='o',core_sha256='c')
    contract=dict(original_python_source_sha256='o',cases=[dict(case_id=str(i)) for i in range(9)])
    manifest=dict(source=dict(files_sha256={'benchmarks/regular_seeds.py':'h'}),runtime=dict(packages_sha256={'skimagexx':{'_core.cpython-311-x86_64-linux-gnu.so':'c'}}))
    rows=[]
    for i in range(9):
        current=2. if i==0 else .5
        rows.append(dict(case_id=str(i),threads=1,native_accepted=True,actual_C_returns=['regular_seeds_apply'],native_entry_module='skimagexx._core',max_abs_error=0,output_sha256='x',reference_sha256='x',oracle_sha256='x',samples_ms=[[1.]*7,[current]*7],loops=[1,1],reference_ms=1.,cpp_ms=current,speedup=1/current))
    return dict(metadata=meta,results=rows),contract,manifest,source


@pytest.mark.parametrize('bad',[None,'source','harness','core','threads','pool','dtype-case','entry','owner','oracle','samples','nan','loops','median','ratio','forwarded'])
def test_actual_all_nine_report_and_loss_validation(bad):
    r,c,m,s=fixture(); x=r['results'][0]
    if bad=='source': r['metadata']['source_before']={}
    elif bad=='harness': r['metadata']['harness_sha256']='wrong'
    elif bad=='core': r['metadata']['core_sha256']='wrong'
    elif bad=='threads': r['metadata']['threads']=4
    elif bad=='pool': r['metadata']['reference_pool_limits']['OMP']='4'
    elif bad=='dtype-case': x['case_id']='changed'
    elif bad=='entry': x['actual_C_returns']=[]
    elif bad=='owner': x['native_entry_module']='fake'
    elif bad=='oracle': x['oracle_sha256']='wrong'
    elif bad=='samples': x['samples_ms'][0].pop()
    elif bad=='nan': x['samples_ms'][0][0]=float('nan')
    elif bad=='loops': x['loops'][0]=0
    elif bad=='median': x['cpp_ms']=3.
    elif bad=='ratio': x['speedup']=3.
    elif bad=='forwarded': x['native_accepted']=False
    if bad:
        with pytest.raises(AssertionError): validator()(r,c,m,s)
    else:
        out=validator()(r,c,m,s)
        assert out['raw_samples']==126 and out['losses']==['0'] and len(out['wins'])==8
        assert out['geometric_mean']>1  # a loss must survive a favorable aggregate


def test_frozen_factory_derives_both_109_node_censuses():
    import runpy
    cases = runpy.run_path(str(ROOT / 'tests/regular_seeds_contract.py'))['cases']()
    frozen = json.loads((ROOT / 'coverage/contracts/regular-seeds.json').read_bytes())['cases']
    assert [c['id'] for c in cases] == [c['id'] for c in frozen]
    assert len(cases) == 53
    # Execute the actual transport's arithmetic after the independent factory check.
    node = next(n for n in ast.parse(SCRIPTS[1]).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'expected_nodes')
    ns = dict(factory_cases=cases)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual-parity-count', 'exec'), ns)
    assert ns['expected_nodes'] == 109
