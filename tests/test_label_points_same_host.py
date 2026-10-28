"""Inert contract/transport checks: synthetic latencies, no scientific calls."""
import ast
import base64
import zlib
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import textwrap
import pytest

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=(ROOT/'.github/workflows/label-points-same-host.yml').read_text(encoding='utf-8')
SCRIPTS=[textwrap.dedent(s) for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",WORKFLOW,re.S)]
PREPARATION=ast.parse(SCRIPTS[1])
PAIR=next(n for n in PREPARATION.body if isinstance(n,ast.If) and ast.unparse(n.test)=="role == 'candidate'")
MUTABLE={'src/bind_label_points.cpp','src/label_points.cpp','src/label_points.hpp','tests/test_label_points_native.cpp'}
CONTRACT='coverage/contracts/label-points-fused-candidate.json'


def test_frozen_transport_and_no_guard_or_harness_substitution():
    assert len(SCRIPTS)==3
    for script in SCRIPTS:compile(script,'workflow-inline','exec')
    assert '40689772340' in SCRIPTS[0] and '40693751374' in SCRIPTS[0]
    assert '12819131293' in SCRIPTS[0] and '510cf996df465d303cba7b6241cafa31bdb03517562cbbad513e987e0fccdc3b' in SCRIPTS[0]
    assert '12817656081' in SCRIPTS[0] and '142fd759cc792b0c34485a496756f48614f273574250af87e6b439467ae570d6' in SCRIPTS[0]
    assert 'fd8b792b82ac1e9635d5d1431776275f5dc882bc' in SCRIPTS[0]
    timed=WORKFLOW.split('      - name: Measure',1)[1].split('      - name:',1)[0]
    assert 'for pass in 1-baseline 2-candidate 3-candidate 4-baseline' in timed
    assert 'benchmarks/label_points.py --threads 1 --repeat 7' in timed
    assert timed.count('ci_measurement_artifacts.py verify')==2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__',timed)
    assert 'name: label-points-same-host-minimum' in WORKFLOW and 'if: always()' in WORKFLOW
    assert 'Normalize only' not in WORKFLOW and 'no normalization authorized' in SCRIPTS[1]
    loops=[ast.literal_eval(n.iter) for n in PREPARATION.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)]
    assert sum(c*2 for name,c in loops[0] if name!='color-transforms-regression')==6270
    assert len(loops[0])==29 and ('label-points-pipeline',100) in loops[0]
    assert "{'passed': 1427}" in SCRIPTS[1] and '== 243' in SCRIPTS[1]


def manifests():
    runtime=dict(python='same',implementation='same',cache_tag='same',machine='same',system='same',dependency_libraries_sha256={'blas':'same'},versions={'numpy':'1.24.4','pytest':'same','scikit-imagexx':'same'},packages_sha256={k:{'source':'a'} for k in ('numpy','scipy','skimage')})
    wheels={f'dep{i}':dict(name=f'dep{i}',sha256='a') for i in range(13)}
    wheels['product']=dict(name='scikit-imagexx',sha256='b')
    fixed={'python/skimagexx/_label_points.py','src/simd_label_points_avx2.cpp','CMakeLists.txt','benchmarks/label_points.py','tests/label_points_contract.py','src/fft.cpp'}
    old=dict(runtime=runtime,wheels=wheels,source=dict(files_sha256=dict.fromkeys(fixed|MUTABLE,'a')))
    new=deepcopy(old)
    for name in MUTABLE:new['source']['files_sha256'][name]='changed'
    new['source']['files_sha256'][CONTRACT]='7c0f0269bf7ffc60cf7c34a1c919444e9554df2b3d327870c021161356d6354d'
    new['wheels']['product']['sha256']='changed'
    return old,new


@pytest.mark.parametrize('bad',[None,'frontend','simd-zero','cmake','harness','factory','fft','new-file','dependency','pytest','package','runtime','missing-leaf','contract'])
def test_source_and_dependency_closure(tmp_path,monkeypatch,bad):
    old,new=manifests()
    mapping={'frontend':'python/skimagexx/_label_points.py','simd-zero':'src/simd_label_points_avx2.cpp','cmake':'CMakeLists.txt','harness':'benchmarks/label_points.py','factory':'tests/label_points_contract.py','fft':'src/fft.cpp','new-file':'new.cpp'}
    if bad in mapping:new['source']['files_sha256'][mapping[bad]]='changed'
    elif bad=='dependency':new['wheels']['dep0']['sha256']='changed'
    elif bad=='pytest':new['runtime']['versions']['pytest']='changed'
    elif bad=='package':new['runtime']['packages_sha256']['numpy']['source']='changed'
    elif bad=='runtime':new['runtime']['python']='changed'
    elif bad=='missing-leaf':del new['source']['files_sha256']['src/label_points.cpp']
    elif bad=='contract':new['source']['files_sha256'][CONTRACT]='changed'
    # Execute the actual pair validator. Contract-file digest and ABI source
    # predicate have separate concrete tests, so isolate those two I/O lines.
    body=[n for n in PAIR.body[1:] if not (isinstance(n,ast.Assert) and 'Path(' in ast.unparse(n)) and not isinstance(n,ast.Expr)]
    code=compile(ast.Module(body=body,type_ignores=[]),'actual-source-closure','exec')
    if bad:
        with pytest.raises(AssertionError):exec(code,dict(old=old,new=new))
    else:exec(code,dict(old=old,new=new))


@pytest.mark.parametrize('bad',[None,'descriptor','return','entry','binding-signature','binding-args'])
def test_actual_public_c_signature_and_private_return_transition(tmp_path,monkeypatch,bad):
    monkeypatch.chdir(tmp_path)
    old='struct LabelPointsInput {const void* axes;int dimensions;};\nvoid label_points_into(const LabelPointsInput&,unsigned long*,int);\n'
    new=old.replace('void label_points_into','bool label_points_into')
    binder='module.def("label_points_apply", [](const py::tuple& indices,const py::tuple& shape,int workers)->py::object {return {};}, py::arg("indices").noconvert(), py::arg("shape").noconvert(), py::arg("workers"));'
    new_binder=binder
    if bad=='descriptor':new=new.replace('int dimensions','long dimensions')
    elif bad=='return':new=new.replace('bool label_points_into','void label_points_into')
    elif bad=='entry':new=new.replace('long*,int','long*,long')
    elif bad=='binding-signature':new_binder=new_binder.replace('int workers','long workers')
    elif bad=='binding-args':new_binder=new_binder.replace('py::arg("shape")','py::arg("wrong")')
    for role,header,body in [('baseline',old,binder),('candidate',new,new_binder)]:
        folder=tmp_path/'snapshots'/role/'src';folder.mkdir(parents=True)
        (folder/'label_points.hpp').write_text(header);(folder/'bind_label_points.cpp').write_text(body)
    node=next(n for n in PREPARATION.body if isinstance(n,ast.FunctionDef))
    ns=dict(Path=Path,re=re);exec(compile(ast.Module(body=[node],type_ignores=[]),'public-C-ABI','exec'),ns)
    if bad:
        with pytest.raises(AssertionError):ns['assert_existing_sources']()
    else:ns['assert_existing_sources']()


def run_decision(tmp_path,monkeypatch,scenario):
    import datetime
    real=datetime.datetime
    class FixedDatetime(real):
        @classmethod
        def now(cls,tz=None):return real(2026,10,28,15 if scenario=='expired' else 13,tzinfo=datetime.timezone.utc)
    monkeypatch.setattr(datetime,'datetime',FixedDatetime);monkeypatch.chdir(tmp_path)
    original=json.loads((ROOT/'benchmarks/label-points-minimum-initial.json').read_bytes())
    fixed_contract=tmp_path/'snapshots/baseline/coverage/contracts/label-points-benchmark.json';fixed_contract.parent.mkdir(parents=True);fixed_contract.write_bytes((ROOT/'coverage/contracts/label-points-benchmark.json').read_bytes())
    identities={};fingerprints={}
    for role in ('baseline','candidate'):
        source=tmp_path/'snapshots'/role;prepared=source/'build/prepared/ci-artifacts';prepared.mkdir(parents=True);(source/'benchmarks').mkdir()
        value=dict(source=dict(revision=('c' if role=='candidate' else 'b')*40),runtime=dict(python=original['metadata']['python'],packages_sha256={'skimagexx':{'_core.so':role*8}}))
        raw=json.dumps(value).encode();(prepared/'measurement.json').write_bytes(raw)
        fingerprints[role]=dict(source_sha256=role,source_hash_schema='fixture',source_hash_scope='fixture')
        (source/'benchmarks/provenance.py').write_text('def source_fingerprint(root):return '+repr(fingerprints[role])+'\n')
        identities[role]=dict(status='verify',revision=value['source']['revision'],wheel_count=14,artifact_sha256=hashlib.sha256(raw).hexdigest())
    for name in ('1-baseline','2-candidate','3-candidate','4-baseline'):
        role=name.split('-')[1];candidate=role=='candidate';folder=tmp_path/'results'/name;folder.mkdir(parents=True)
        report=deepcopy(original);report['metadata'].update(fingerprints[role]);report['metadata']['source_before']=fingerprints[role];report['metadata']['core_sha256']=role*8
        for row in report['results']:
            cpp=5.0 if candidate else 10.0;ref=10.0
            if scenario=='global-threshold' and candidate:cpp=10/(1.11 if row['case_id']=='dense-64x64x64-i64' else 1.02)
            if scenario=='normalized-global' and candidate:
                ref=cpp*(1.11 if row['case_id']=='dense-64x64x64-i64' else 1.02 if row['case_id']=='dense-1024x1024-f64' else 1.0)
            if scenario=='target-repeat' and name=='3-candidate' and row['case_id']=='dense-64x64x64-i64':cpp=10/1.08
            if scenario=='preserved-loss' and row['case_id']=='dense-1024x1024-f64':ref=4.0
            if scenario=='remaining-loss' and row['case_id']=='sparse-64x64-i64':ref=4.0
            if scenario=='other-direct-floor' and candidate and row['case_id']=='sparse-64x64-i64':cpp=10/0.96
            if scenario=='other-normalized-floor' and candidate and row['case_id']=='sparse-64x64-i64':ref=4.8
            if scenario=='geometric-repeat':cpp=8.0 if candidate else (8.0 if name=='1-baseline' else 12.5)
            row.update(reference_ms=ref,cpp_ms=cpp,speedup=ref/cpp,samples_ms=[[ref]*7,[cpp]*7])
        if name=='3-candidate':
            row=report['results'][0]
            if scenario=='input-drift':row['inputs'][0]['sha256']='f'*64
            elif scenario=='output-drift':row['output_sha256']='f'*64
            elif scenario=='wrong-C-entry':row['actual_C_returns']=['fake']
            elif scenario=='wrong-core':report['metadata']['core_sha256']='f'*64
        (folder/'label-points.json').write_text(json.dumps(report))
        for when in ('before','after'):(folder/('identity-'+when+'.log')).write_text(json.dumps(identities[role]))
    exec(compile(SCRIPTS[2],'actual-decision','exec'),{})
    return json.loads((tmp_path/'results/comparison.json').read_bytes())


@pytest.mark.parametrize('scenario,expected',[('keep','KEEP'),('geometric-repeat','KEEP'),('remaining-loss','KEEP'),('global-threshold','DISCARD'),('normalized-global','DISCARD'),('target-repeat','DISCARD'),('preserved-loss','DISCARD'),('other-direct-floor','DISCARD'),('other-normalized-floor','DISCARD'),('expired','DISCARD')])
def test_fixed_decision_metrics(tmp_path,monkeypatch,scenario,expected):
    result=run_decision(tmp_path,monkeypatch,scenario)
    assert result['status']==expected and result['raw_samples']==504 and len(result['rows'])==9
    if scenario=='remaining-loss':assert result['balanced_original_losses']==['sparse-64x64-i64']


@pytest.mark.parametrize('scenario',['input-drift','output-drift','wrong-C-entry','wrong-core'])
def test_complete_raw_metadata_and_actual_C_guard(tmp_path,monkeypatch,scenario):
    with pytest.raises(AssertionError):run_decision(tmp_path,monkeypatch,scenario)


@pytest.mark.parametrize('mutate',[False,True])
def test_contract_actual_file_digest_gate(tmp_path,monkeypatch,mutate):
    monkeypatch.chdir(tmp_path)
    path=tmp_path/'snapshots/candidate'/CONTRACT;path.parent.mkdir(parents=True)
    raw=CONTRACT_RAW;path.write_bytes(raw+(b'changed' if mutate else b''))
    new=dict(source=dict(files_sha256={CONTRACT:hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()}))
    node=next(n for n in PAIR.body if isinstance(n,ast.Assert) and 'Path(' in ast.unparse(n))
    code=compile(ast.Module(body=[node],type_ignores=[]),'actual-contract-file-authentication','exec')
    namespace=dict(hashlib=hashlib,Path=Path,new=new,contract=CONTRACT)
    if mutate:
        with pytest.raises(AssertionError):exec(code,namespace)
    else:exec(code,namespace)

# Exact candidate git blob, retained as compressed inert contract metadata.
CONTRACT_RAW=zlib.decompress(base64.b64decode('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'))


def test_actual_contract_raw_vs_source_canonical_fingerprints():
    assert CONTRACT_RAW.count(b"\r\n")==0
    assert hashlib.sha256(CONTRACT_RAW).hexdigest()=="7c0f0269bf7ffc60cf7c34a1c919444e9554df2b3d327870c021161356d6354d"
    assert hashlib.sha256(CONTRACT_RAW.replace(b"\r\n",b"\n")).hexdigest()=="7c0f0269bf7ffc60cf7c34a1c919444e9554df2b3d327870c021161356d6354d"
