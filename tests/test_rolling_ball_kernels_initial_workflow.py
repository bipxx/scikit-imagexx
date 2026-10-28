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
WORKFLOW = (ROOT / '.github/workflows/rolling-ball-kernels-initial.yml').read_text()
SCRIPTS = [textwrap.dedent(s) for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n", WORKFLOW, re.S)]


def test_isolated_one_product_transport_and_fixed_closure():
    assert len(SCRIPTS) == 3
    for s in SCRIPTS: compile(s, 'workflow-inline', 'exec')
    for name in ('producer_run', 'producer_revision', 'artifact_id', 'artifact_sha256'):
        assert name + ':\n        type: string\n        required: true' in WORKFLOW
    assert 'default:' not in WORKFLOW
    timed = WORKFLOW.split('      - name: Measure', 1)[1].split('      - name:', 1)[0]
    assert timed.count('benchmarks/rolling_ball_kernels.py --threads 1 --repeat 7') == 1
    assert timed.count('ci_measurement_artifacts.py verify') == 2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__|setup.py', timed)
    assert '--no-index --no-deps --only-binary=:all: --require-hashes' in WORKFLOW
    assert "== 172" in SCRIPTS[1] and "== 81" in SCRIPTS[1]
    assert "== 245" in SCRIPTS[1] and "== 3" in SCRIPTS[1]
    assert "'test_rolling_ball_kernels_integration': len(factory_cases) * 2 + 18" in SCRIPTS[1]
    assert "'test_rolling_ball_kernels_pipeline': len(factory_cases) * 2 + 12" in SCRIPTS[1]
    assert 'dict(actual_files) == expected_files' in SCRIPTS[1] and '== 1443' in SCRIPTS[1]
    assert 'if: always()' in WORKFLOW and 'name: rolling-ball-kernels-initial-minimum' in WORKFLOW
    assert 'lscpu > results/cpu.txt' in WORKFLOW
    assert "len(manifest['wheels']) - len(products) == 13" in SCRIPTS[1]
    assert 'core.ball_kernel_apply' not in WORKFLOW  # no entry replacement
    assert '417841595337d4afe2148aedbab54731d0de2d43caa11e7e7a5805d24249790a' in SCRIPTS[1]
    assert 'da1133fd274c84d8811af2fe9f3bd4afc5da4169cdfe4dc2f23def728e4aaf30' in SCRIPTS[1]


@pytest.mark.parametrize('bad', [None, 'revision', 'run', 'repository', 'workflow', 'failed', 'artifact', 'digest', 'expired', 'foreign-artifact','timing-ran','timing-missing','producer-job-failed'])
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
    jobs=[dict(name='numpy-minimum',conclusion='success',steps=[]),dict(name='measure-minimum',conclusion='success',steps=[dict(name='Run python benchmarks/'+str(i),conclusion='skipped')for i in range(27)])]
    if bad=='timing-ran':jobs[1]['steps'][0]['conclusion']='success'
    elif bad=='timing-missing':jobs[1]['steps'].pop()
    elif bad=='producer-job-failed':jobs[0]['conclusion']='failure'
    replies=[record,dict(total_count=1,artifacts=[item]),dict(total_count=2,jobs=jobs)]
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
    apis=['restoration.ball_kernel','restoration.ellipsoid_kernel']
    meta=dict(**source,source_before=deepcopy(source),threads=1,repeat=7,target_ms=19,raw_samples=168,reference_pool_limits={'OMP':'1'},fallback_enabled=False,selected_apis=apis,harness_sha256='h',original_source_sha256='o',core_sha256='c',numpy_error_policy=dict(divide='warn',over='warn',under='ignore',invalid='warn'),numpy_error_handler_type='NoneType',numpy_buffer_elements=8192)
    contract=dict(original_python_source_sha256='o',max_output_elements=1000000,apis=apis,cases=[])
    manifest=dict(source=dict(files_sha256={'benchmarks/rolling_ball_kernels.py':'h'}),runtime=dict(packages_sha256={'skimagexx':{'_core.cpython-311-x86_64-linux-gnu.so':'c'}}))
    rows=[]
    for i in range(12):
        ball=i<6; shape=[3]*(1+i%4);params=dict(radius=1.,ndim=len(shape))if ball else dict(shape=[2]*len(shape),intensity=2.5)
        spec=dict(case_id=str(i),api=apis[0 if ball else 1],parameters=params,expected_output_shape=shape,expected_elements=math.prod(shape));contract['cases'].append(spec)
        strides=[];stride=8
        for n in reversed(shape):strides.insert(0,stride);stride*=n
        descriptor=dict(dtype='<f8',dtype_metadata=None,shape=shape.copy(),strides=strides,own=True,writeable=True,c=True,f=len(shape)==1,base_type='NoneType')
        current=2. if i==0 else .5
        rows.append(dict(**deepcopy(spec),threads=1,native_accepted=True,actual_C_returns=['ball_kernel_apply'if ball else'ellipsoid_kernel_apply'],native_entry_module='skimagexx._core',max_abs_error=0,output_sha256='x',reference_sha256='x',oracle_sha256='x',output_metadata=descriptor,samples_ms=[[1.]*7,[current]*7],loops=[1,1],reference_ms=1.,cpp_ms=current,speedup=1/current))
    return dict(metadata=meta,results=rows),contract,manifest,source


@pytest.mark.parametrize('bad',[None,'source','harness','core','threads','pool','dtype-case','entry','owner','oracle','samples','nan','loops','median','ratio','forwarded','output-shape','output-dtype','output-metadata','output-strides','output-owner','output-base','output-readonly','output-fortran','output-bound','reference-word-hash','fp-handler','fp-policy','fp-buffer','wrong-api-entry','second-output-fortran'])
def test_actual_all_twelve_report_and_loss_validation(bad):
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
    elif bad=='output-metadata': x['output_metadata']['dtype_metadata']={'changed':True}
    elif bad=='output-strides': x['output_metadata']['strides'][0]+=1
    elif bad=='output-owner': x['output_metadata']['own']=False
    elif bad=='output-base': x['output_metadata']['base_type']='ndarray'
    elif bad=='output-readonly': x['output_metadata']['writeable']=False
    elif bad=='output-fortran': x['output_metadata']['c']=False
    elif bad=='output-bound': c['max_output_elements']=1
    if bad=='reference-word-hash': x['reference_sha256']='wrong'
    elif bad=='fp-handler':r['metadata']['numpy_error_handler_type']='callable'
    elif bad=='fp-policy':r['metadata']['numpy_error_policy']['invalid']='changed'
    elif bad=='fp-buffer':r['metadata']['numpy_buffer_elements']=0
    elif bad=='wrong-api-entry':r['results'][6]['actual_C_returns']=['ball_kernel_apply']
    elif bad=='second-output-fortran':r['results'][1]['output_metadata']['f']=True
    if bad:
        with pytest.raises(AssertionError): validator()(r,c,m,s)
    else:
        out=validator()(r,c,m,s)
        assert out['raw_samples']==168 and out['losses']==['0'] and len(out['wins'])==11
        assert out['geometric_mean']>1  # a loss must survive a favorable aggregate


def test_frozen_factory_derives_actual_reference_integration_pipeline_censuses():
    import runpy
    cases = runpy.run_path(str(ROOT / 'tests/rolling_ball_kernels_contract.py'))['cases']()
    frozen = json.loads((ROOT / 'coverage/contracts/rolling-ball-kernels.json').read_bytes())['cases']
    assert [c['id'] for c in cases] == [c['id'] for c in frozen]
    assert len(cases) == 50
    # Execute the actual transport's arithmetic after the independent factory check.
    node = next(n for n in ast.parse(SCRIPTS[1]).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'expected_files')
    ns = dict(factory_cases=cases)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual-parity-count', 'exec'), ns)
    assert ns['expected_files'] == {'test_rolling_ball_kernels_integration':118,'test_rolling_ball_kernels_pipeline':112}
    assert sum(ns['expected_files'].values())==230


@pytest.mark.parametrize('fault',(None,'missing-pipeline','substituted-pipeline','failure','skipped','wrong-module','scalar-missing'))
def test_actual_combined_xml_requires_both_runtime_test_files(tmp_path,fault):
    from collections import Counter
    import xml.etree.ElementTree as ET
    expected={'test_rolling_ball_kernels_integration':118,'test_rolling_ball_kernels_pipeline':112}
    for mode in('normal','scalar'):
        tree=ET.Element('testsuite')
        for module,count in expected.items():
            for index in range(count):
                if fault=='missing-pipeline'and module=='test_rolling_ball_kernels_pipeline'and index==0:continue
                if fault=='scalar-missing'and mode=='scalar'and module=='test_rolling_ball_kernels_pipeline'and index==0:continue
                label=module
                if module=='test_rolling_ball_kernels_pipeline'and index==0:
                    if fault=='substituted-pipeline':label='test_rolling_ball_kernels_integration'
                    elif fault=='wrong-module':label='test_different'
                case=ET.SubElement(tree,'testcase',classname='tests.'+label,name=str(index))
                if fault in('failure','skipped')and module=='test_rolling_ball_kernels_pipeline'and index==0:ET.SubElement(case,fault)
        ET.ElementTree(tree).write(tmp_path/('rolling-ball-kernels-'+mode+'.xml'),encoding='utf-8')
    parsed=ast.parse(SCRIPTS[1])
    declaration=next(n for n in parsed.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='expected_files')
    loop=next(n for n in parsed.body if isinstance(n,ast.For)and isinstance(n.iter,ast.Tuple)and 'rolling-ball-kernels-normal.xml'in ast.unparse(n.iter))
    ns=dict(factory_cases=[None]*50,prepared=tmp_path,ET=ET,Counter=Counter)
    code=compile(ast.Module(body=[declaration,loop],type_ignores=[]),'actual-kernel-parity-preflight','exec')
    if fault:
        with pytest.raises(AssertionError):exec(code,ns)
    else:exec(code,ns)


@pytest.mark.parametrize('fault',(None,'missing-sdk','foreign-sdk','missing-wheel-member','extra-wheel-member','foreign-core'))
def test_actual_source_derived_install_closure(tmp_path,monkeypatch,fault):
    import subprocess,zipfile
    monkeypatch.chdir(tmp_path)
    root=tmp_path/'snapshot';wheel_dir=root/'build/prepared/wheels';wheel_dir.mkdir(parents=True)
    sdk={f'include/skimagexx/{i}.hpp'for i in range(81)}
    members=sdk|{f'skimagexx/module{i}.py'for i in range(90)}
    layout=dict(product_count=172,SDK_count=81,SDK_members=sorted(sdk),members_without_core=sorted(members))
    actual_sdk=set(sdk);actual_members=members|{'skimagexx/_core.cpython-311-x86_64-linux-gnu.so'}
    if fault=='missing-sdk':actual_sdk.pop()
    elif fault=='foreign-sdk':actual_sdk.remove(next(iter(sdk)));actual_sdk.add('foreign.hpp')
    elif fault=='missing-wheel-member':actual_members.remove('skimagexx/module0.py')
    elif fault=='extra-wheel-member':actual_members.add('skimagexx/foreign.py')
    elif fault=='foreign-core':actual_members.remove('skimagexx/_core.cpython-311-x86_64-linux-gnu.so');actual_members.add('skimagexx/fake.so')
    with zipfile.ZipFile(wheel_dir/'product.whl','w')as z:
        for name in actual_members:z.writestr(name,b'')
    manifest=dict(runtime=dict(sdk_sha256={name:'x'for name in actual_sdk}),wheels={'product.whl':dict(name='scikit-imagexx'),**{f'dep{i}.whl':dict(name=str(i))for i in range(13)}})
    monkeypatch.setattr(subprocess,'check_output',lambda args:b'pyproject.toml\0')
    runpy=SimpleNamespace(run_path=lambda path:{'current_install_layout':lambda read,names,profile:layout})
    segment=SCRIPTS[1].split('source_root =',1)[1].split('inventory =',1)[0]
    code='source_root ='+segment
    ns=dict(Path=Path,subprocess=subprocess,runpy=runpy,manifest=manifest,zipfile=zipfile)
    if fault:
        with pytest.raises(AssertionError):exec(code,ns)
    else:exec(code,ns)
