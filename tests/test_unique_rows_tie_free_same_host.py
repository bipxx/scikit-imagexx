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
WORKFLOW=(ROOT/'.github/workflows/unique-rows-tie-free-same-host.yml').read_text(encoding='utf-8')
SCRIPTS=[textwrap.dedent(s) for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",WORKFLOW,re.S)]
PREPARATION=ast.parse(SCRIPTS[1])
PAIR=next(n for n in PREPARATION.body if isinstance(n,ast.If) and ast.unparse(n.test)=="role == 'candidate'")
MUTABLE={'src/unique_rows.cpp','src/unique_rows.hpp','tests/test_unique_rows_native.cpp'}
CONTRACT='coverage/contracts/unique-rows-tie-free-candidate.json'


def test_frozen_transport_and_no_guard_or_harness_substitution():
    assert len(SCRIPTS)==3
    for script in SCRIPTS:compile(script,'workflow-inline','exec')
    assert '40702637014' in SCRIPTS[0] and '40712229136' in SCRIPTS[0]
    assert '12828765895' in SCRIPTS[0] and 'b9f233f4b6ed05efe1b844ab19f1e4d8454d31f53dc56fa285e1860fb1321087' in SCRIPTS[0]
    assert '12823857042' in SCRIPTS[0] and '355736e04190ae6efa5eabd99659c5b4066ae0145eb1b049bb0f2e7177e130d7' in SCRIPTS[0]
    assert '7aa059ddb1c2f196c3dbba9cf4d638ae1a761e67' in SCRIPTS[0]
    timed=WORKFLOW.split('      - name: Measure',1)[1].split('      - name:',1)[0]
    assert 'for pass in 1-baseline 2-candidate 3-candidate 4-baseline' in timed
    assert 'benchmarks/unique_rows.py --threads 1 --repeat 7' in timed
    assert timed.count('ci_measurement_artifacts.py verify')==2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__',timed)
    assert 'name: unique-rows-tie-free-same-host-minimum' in WORKFLOW and 'if: always()' in WORKFLOW
    assert 'Normalize only' not in WORKFLOW and 'no normalization authorized' in SCRIPTS[1]
    loops=[ast.literal_eval(n.iter) for n in PREPARATION.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)]
    assert sum(c*2 for name,c in loops[0] if name!='color-transforms-regression')==6698
    assert len(loops[0])==31 and ('unique-rows-pipeline',107) in loops[0]
    assert "{'passed': 1431}" in SCRIPTS[1] and '== 244' in SCRIPTS[1]


def manifests():
    runtime=dict(python='same',implementation='same',cache_tag='same',machine='same',system='same',dependency_libraries_sha256={'blas':'same'},versions={'numpy':'1.24.4','pytest':'same','scikit-imagexx':'same'},packages_sha256={k:{'source':'a'} for k in ('numpy','scipy','skimage')})
    wheels={f'dep{i}':dict(name=f'dep{i}',sha256='a') for i in range(13)}
    wheels['product']=dict(name='scikit-imagexx',sha256='b')
    fixed={'python/skimagexx/_unique_rows.py','src/bind_unique_rows.cpp','CMakeLists.txt','benchmarks/unique_rows.py','tests/unique_rows_contract.py','src/fft.cpp','src/simd_unique_rows_avx2.cpp'}
    old=dict(runtime=runtime,wheels=wheels,source=dict(files_sha256=dict.fromkeys(fixed|MUTABLE,'a')))
    new=deepcopy(old)
    for name in MUTABLE:new['source']['files_sha256'][name]='changed'
    new['source']['files_sha256'][CONTRACT]='f2bc8c10b2ae017a98c8b6dbc936be6d54b2b9d48acc12f7e2d98b5cebba05ea'
    new['wheels']['product']['sha256']='changed'
    return old,new


@pytest.mark.parametrize('bad',[None,'frontend','binder','cmake','harness','factory','fft','new-file','dependency','pytest','package','runtime','missing-leaf','contract','simd'])
def test_source_and_dependency_closure(tmp_path,monkeypatch,bad):
    old,new=manifests()
    mapping={'frontend':'python/skimagexx/_unique_rows.py','binder':'src/bind_unique_rows.cpp','cmake':'CMakeLists.txt','harness':'benchmarks/unique_rows.py','factory':'tests/unique_rows_contract.py','fft':'src/fft.cpp','new-file':'new.cpp','simd':'src/simd_unique_rows_avx2.cpp'}
    if bad in mapping:new['source']['files_sha256'][mapping[bad]]='changed'
    elif bad=='dependency':new['wheels']['dep0']['sha256']='changed'
    elif bad=='pytest':new['runtime']['versions']['pytest']='changed'
    elif bad=='package':new['runtime']['packages_sha256']['numpy']['source']='changed'
    elif bad=='runtime':new['runtime']['python']='changed'
    elif bad=='missing-leaf':del new['source']['files_sha256']['src/unique_rows.cpp']
    elif bad=='contract':new['source']['files_sha256'][CONTRACT]='changed'
    # Execute the actual pair validator. Contract-file digest and ABI source
    # predicate have separate concrete tests, so isolate those two I/O lines.
    body=[n for n in PAIR.body[1:] if not (isinstance(n,ast.Assert) and 'Path(' in ast.unparse(n)) and not isinstance(n,ast.Expr)]
    code=compile(ast.Module(body=body,type_ignores=[]),'actual-source-closure','exec')
    if bad:
        with pytest.raises(AssertionError):exec(code,dict(old=old,new=new))
    else:exec(code,dict(old=old,new=new))


@pytest.mark.parametrize('bad',[None,'input','output','entry'])
def test_actual_public_header_signature_preserved(tmp_path,monkeypatch,bad):
    monkeypatch.chdir(tmp_path)
    old='struct UniqueRowsInput {const unsigned char* rows;int row_count;int row_bytes;};\nstd::size_t unique_rows_indices(const UniqueRowsInput&,long*,int);\n'
    new=old+'void private_simd(int);\n'
    if bad=='input':new=new.replace('int row_count','long row_count')
    elif bad=='output':new=new.replace('int row_bytes','long row_bytes')
    elif bad=='entry':new=new.replace('long*,int','long*,long')
    for role,text in [('baseline',old),('candidate',new)]:
        p=tmp_path/'snapshots'/role/'src/unique_rows.hpp';p.parent.mkdir(parents=True);p.write_text(text)
    node=next(n for n in PREPARATION.body if isinstance(n,ast.FunctionDef))
    ns=dict(Path=Path,re=re);exec(compile(ast.Module(body=[node],type_ignores=[]),'ABI','exec'),ns)
    if bad:
        with pytest.raises(AssertionError):ns['assert_existing_sources']()
    else:ns['assert_existing_sources']()


def run_decision(tmp_path,monkeypatch,scenario):
    import datetime
    real=datetime.datetime
    class FixedDatetime(real):
        @classmethod
        def now(cls,tz=None):return real(2026,10,28,17 if scenario=='expired' else 15,tzinfo=datetime.timezone.utc)
    monkeypatch.setattr(datetime,'datetime',FixedDatetime);monkeypatch.chdir(tmp_path)
    original=json.loads((ROOT/'benchmarks/unique-rows-minimum-initial.json').read_bytes())
    fixed_contract=tmp_path/'snapshots/baseline/coverage/contracts/unique-rows-benchmark.json';fixed_contract.parent.mkdir(parents=True);fixed_contract.write_bytes((ROOT/'coverage/contracts/unique-rows-benchmark.json').read_bytes())
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
            if scenario=='global-threshold' and candidate:cpp=10/1.02
            if scenario=='target-repeat' and name=='3-candidate' and row['case_id']=='u8-readonly-duplicates':cpp=10/1.19
            if scenario=='preserved-loss' and candidate and row['case_id']=='u8-short-small-unique':ref=4.0
            if scenario=='other-original-loss' and row['case_id']=='f32-coordinates-large-unique':ref=4.0
            if scenario=='target-original-loss' and row['case_id']=='u8-readonly-duplicates':ref=4.0
            if scenario=='wide-cpp-direct-below' and candidate and row['case_id']=='u8-wide32-common-prefix':cpp=10/.96;ref=40.0
            if scenario=='float-direct-below' and candidate and row['case_id']=='f32-coordinates-large-unique':cpp=10/.96;ref=40.0
            if scenario=='float-normalized-below' and candidate and row['case_id']=='f32-strided-duplicates':ref=2.4
            if scenario=='geometric-repeat':cpp=8.2 if candidate else (8.0 if name=='1-baseline' else 12.5)
            row.update(reference_ms=ref,cpp_ms=cpp,speedup=ref/cpp,samples_ms=[[ref]*7,[cpp]*7])
        if name=='3-candidate':
            row=report['results'][0]
            if scenario=='input-drift':row['inputs'][0]['sha256']='f'*64
            elif scenario=='output-drift':row['output_sha256']='f'*64
            elif scenario=='native-index-drift':row['native_indices']['sha256']='f'*64
            elif scenario=='native-index-layout':row['native_indices']['strides']=[16]
            elif scenario=='wrong-C-entry':row['actual_C_returns']=['fake']
            elif scenario=='wrong-core':report['metadata']['core_sha256']='f'*64
        (folder/'unique-rows.json').write_text(json.dumps(report))
        for when in ('before','after'):(folder/('identity-'+when+'.log')).write_text(json.dumps(identities[role]))
    exec(compile(SCRIPTS[2],'actual-decision','exec'),{})
    return json.loads((tmp_path/'results/comparison.json').read_bytes())


@pytest.mark.parametrize('scenario,expected',[('keep','KEEP'),('geometric-repeat','KEEP'),('other-original-loss','DISCARD'),('target-original-loss','DISCARD'),('wide-cpp-direct-below','DISCARD'),('global-threshold','DISCARD'),('target-repeat','DISCARD'),('preserved-loss','DISCARD'),('expired','DISCARD'),('float-direct-below','DISCARD'),('float-normalized-below','DISCARD')])
def test_fixed_decision_metrics(tmp_path,monkeypatch,scenario,expected):
    result=run_decision(tmp_path,monkeypatch,scenario)
    assert result['status']==expected and result['raw_samples']==504 and len(result['rows'])==9


@pytest.mark.parametrize('scenario',['input-drift','output-drift','wrong-C-entry','wrong-core','native-index-drift','native-index-layout'])
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


@pytest.mark.parametrize('direct,normalized,target,control_direct,control_normalized,expected',[
    (1.02,2,True,1,1,False),(2,1.02,True,1,1,False),
    (2,2,True,.97,.97,True),(2,2,True,.969,.97,False),
    (2,2,True,.97,.969,False),(2,2,False,1,1,False)])
def test_exact_strict_global_and_inclusive_cpp_control_boundaries(direct,normalized,target,control_direct,control_normalized,expected):
    tree=ast.parse(SCRIPTS[2]);assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='keep')
    namespace=dict(direct=direct,normalized=normalized,targets=[dict(repeat_direct=target,repeat_normalized=target,original_repeat=True)],preserved=[dict(original_repeat=True)],cpp_controls=[dict(direct_gain=control_direct,normalized_gain=control_normalized)],completed=1,deadline=2)
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),'actual-keep-expression','exec'),namespace)
    assert namespace['keep']==expected

# Immutable prospective contract blob from candidate 7aa059d, no scientific payload.
CONTRACT_RAW=zlib.decompress(base64.b64decode('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'))

def test_actual_tie_free_contract_raw_and_canonical_lf_pin():
    assert b"\r\n" not in CONTRACT_RAW
    assert hashlib.sha256(CONTRACT_RAW).hexdigest()=="f2bc8c10b2ae017a98c8b6dbc936be6d54b2b9d48acc12f7e2d98b5cebba05ea"


def test_case_ids_and_thresholds_match_immutable_prospective_contract():
    contract=json.loads(CONTRACT_RAW);tree=ast.parse(SCRIPTS[2])
    values={ast.unparse(n.targets[0]):n.value for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1}
    assert ast.literal_eval(values['target_ids'])=={contract['target']}
    assert ast.literal_eval(values['preserved_ids'])==set(contract['preserve_original_wins'])
    assert ast.unparse(values['cpp_control_ids'])=='preserved_ids'
    assert 'direct > 1.02 and normalized > 1.02' in SCRIPTS[2]
    assert SCRIPTS[2].count(') > 1.20')==2
    assert "and r['original_repeat'] for r in targets" in SCRIPTS[2]
    assert '2026-10-28T16:57:38.559206+00:00' in SCRIPTS[2]
