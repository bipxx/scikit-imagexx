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
WORKFLOW = (ROOT / '.github/workflows/montage-initial.yml').read_text()
SCRIPTS = [textwrap.dedent(s) for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n", WORKFLOW, re.S)]


def test_isolated_one_product_transport_and_fixed_closure():
    assert len(SCRIPTS) == 3
    for s in SCRIPTS: compile(s, 'workflow-inline', 'exec')
    for name in ('producer_run', 'producer_revision', 'artifact_id', 'artifact_sha256'):
        assert name + ':\n        type: string\n        required: true' in WORKFLOW
    assert 'default:' not in WORKFLOW
    timed = WORKFLOW.split('      - name: Measure', 1)[1].split('      - name:', 1)[0]
    assert timed.count('benchmarks/montage.py --threads 1 --repeat 7') == 1
    assert timed.count('ci_measurement_artifacts.py verify') == 2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__|setup.py', timed)
    assert '--no-index --no-deps --only-binary=:all: --require-hashes' in WORKFLOW
    assert "== 170" in SCRIPTS[1] and "== 80" in SCRIPTS[1]
    assert "== 243" in SCRIPTS[1] and "== 3" in SCRIPTS[1]
    assert "'test_montage_public': len(factory_cases) + 5" in SCRIPTS[1]
    assert "'test_montage_integration': len(factory_cases) * 2 + 21" in SCRIPTS[1]
    assert "'test_montage_pipeline': len(factory_cases) * 2 + 6" in SCRIPTS[1]
    assert 'dict(actual_files) == expected_files' in SCRIPTS[1] and '== 1439' in SCRIPTS[1]
    assert 'if: always()' in WORKFLOW and 'name: montage-initial-minimum' in WORKFLOW
    assert 'lscpu > results/cpu.txt' in WORKFLOW
    assert "len(manifest['wheels']) - len(products) == 13" in SCRIPTS[1]
    assert 'core.montage_apply' not in WORKFLOW  # no entry replacement
    assert '93a0dc6c08da84a705880d7f01e751a47f11e12728aa3d28c19112da5d236bdd' in SCRIPTS[1]
    assert '434d7540206cd2adcf3253bc097e36db7131a15b0c6054fa06b2c97fd5da3d68' in SCRIPTS[1]


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
    meta=dict(**source, source_before=deepcopy(source), threads=1,repeat=7,target_ms=19,raw_samples=126,reference_pool_limits={'OMP':'1'},fallback_enabled=False,selected_apis=['util.montage'],harness_sha256='h',original_source_sha256='o',core_sha256='c')
    contract=dict(original_python_source_sha256='o',max_output_elements=1000000,cases=[dict(case_id=str(i),shape=[4,2,3],dtype='uint8',grid=[2,2],padding=1,channel_axis=None)for i in range(9)])
    manifest=dict(source=dict(files_sha256={'benchmarks/montage.py':'h'}),runtime=dict(packages_sha256={'skimagexx':{'_core.cpython-311-x86_64-linux-gnu.so':'c'}}))
    rows=[]
    for i in range(9):
        if i in (1,2):contract['cases'][i].update(shape=[4,2,3,3]if i==1 else[4,3,2,3],channel_axis=-1 if i==1 else 1)
        color=i in(1,2); itemsize=1; channels=3 if color else 1
        height,width=2,3; out_rows,out_cols=7,9
        descriptor=dict(dtype='|u1',metadata=None,shape=[out_rows,out_cols]+([3]if color else[]),strides=[out_cols*channels,channels]+([1]if color else[]),own=color,writeable=True,c=True,f=False,base_type='NoneType'if color else'ndarray')
        current=2. if i==0 else .5
        rows.append(dict(**contract['cases'][i],threads=1,native_accepted=True,actual_C_returns=['montage_apply'],native_entry_module='skimagexx._core',max_abs_error=0,output_sha256='x',oracle_sha256='x',output_metadata=descriptor,samples_ms=[[1.]*7,[current]*7],loops=[1,1],reference_ms=1.,cpp_ms=current,speedup=1/current))
    return dict(metadata=meta,results=rows),contract,manifest,source


@pytest.mark.parametrize('bad',[None,'source','harness','core','threads','pool','dtype-case','entry','owner','oracle','samples','nan','loops','median','ratio','forwarded','output-shape','output-dtype','output-metadata','output-strides','output-owner','output-base','output-readonly','output-fortran','output-bound'])
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
    elif bad=='output-shape': x['output_metadata']['shape'].append(1)
    elif bad=='output-dtype': x['output_metadata']['dtype']='<f4'
    elif bad=='output-metadata': x['output_metadata']['metadata']={'changed':True}
    elif bad=='output-strides': x['output_metadata']['strides'][0]+=1
    elif bad=='output-owner': x['output_metadata']['own']=True
    elif bad=='output-base': x['output_metadata']['base_type']='NoneType'
    elif bad=='output-readonly': x['output_metadata']['writeable']=False
    elif bad=='output-fortran': x['output_metadata']['c']=False
    elif bad=='output-bound': c['max_output_elements']=1
    if bad:
        with pytest.raises(AssertionError): validator()(r,c,m,s)
    else:
        out=validator()(r,c,m,s)
        assert out['raw_samples']==126 and out['losses']==['0'] and len(out['wins'])==8
        assert out['geometric_mean']>1  # a loss must survive a favorable aggregate


def test_frozen_factory_derives_actual_reference_integration_pipeline_censuses():
    import runpy
    cases = runpy.run_path(str(ROOT / 'tests/montage_contract.py'))['cases']()
    frozen = json.loads((ROOT / 'coverage/contracts/montage.json').read_bytes())['cases']
    assert [c['id'] for c in cases] == [c['id'] for c in frozen]
    assert len(cases) == 58
    # Execute the actual transport's arithmetic after the independent factory check.
    node = next(n for n in ast.parse(SCRIPTS[1]).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'expected_files')
    ns = dict(factory_cases=cases)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual-parity-count', 'exec'), ns)
    assert ns['expected_files'] == {'test_montage_public':63,'test_montage_integration':137,'test_montage_pipeline':122}
    assert sum(ns['expected_files'].values())==322


@pytest.mark.parametrize('fault',(None,'missing-pipeline','substituted-pipeline','failure','skipped','wrong-module','scalar-missing'))
def test_actual_combined_xml_requires_all_three_test_files(tmp_path,fault):
    from collections import Counter
    import xml.etree.ElementTree as ET
    expected={'test_montage_public':63,'test_montage_integration':137,'test_montage_pipeline':122}
    for mode in('normal','scalar'):
        tree=ET.Element('testsuite')
        for module,count in expected.items():
            for index in range(count):
                if fault=='missing-pipeline'and module=='test_montage_pipeline'and index==0:continue
                if fault=='scalar-missing'and mode=='scalar'and module=='test_montage_pipeline'and index==0:continue
                label=module
                if module=='test_montage_pipeline'and index==0:
                    if fault=='substituted-pipeline':label='test_montage_integration'
                    elif fault=='wrong-module':label='test_different'
                case=ET.SubElement(tree,'testcase',classname='tests.'+label,name=str(index))
                if fault in('failure','skipped')and module=='test_montage_pipeline'and index==0:ET.SubElement(case,fault)
        ET.ElementTree(tree).write(tmp_path/('montage-'+mode+'.xml'),encoding='utf-8')
    parsed=ast.parse(SCRIPTS[1])
    declaration=next(n for n in parsed.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='expected_files')
    loop=next(n for n in parsed.body if isinstance(n,ast.For)and isinstance(n.iter,ast.Tuple)and 'montage-normal.xml'in ast.unparse(n.iter))
    ns=dict(factory_cases=[None]*58,prepared=tmp_path,ET=ET,Counter=Counter)
    code=compile(ast.Module(body=[declaration,loop],type_ignores=[]),'actual-Montage-parity-preflight','exec')
    if fault:
        with pytest.raises(AssertionError):exec(code,ns)
    else:exec(code,ns)
