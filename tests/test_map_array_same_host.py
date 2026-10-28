"""Inert contract/transport checks: synthetic latencies, no scientific calls."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import textwrap
import pytest

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=(ROOT/'.github/workflows/map-array-same-host.yml').read_text(encoding='utf-8')
SCRIPTS=[textwrap.dedent(s) for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",WORKFLOW,re.S)]
PREPARATION=ast.parse(SCRIPTS[1])
PAIR=next(n for n in PREPARATION.body if isinstance(n,ast.If) and ast.unparse(n.test)=="role == 'candidate'")
MUTABLE={'src/map_array.cpp','src/map_array.hpp','src/simd_map_array_avx2.cpp','tests/test_map_array_native.cpp'}
CONTRACT='coverage/contracts/map-array-small-native-candidate.json'


def test_frozen_transport_and_no_guard_or_harness_substitution():
    assert len(SCRIPTS)==3
    for script in SCRIPTS:compile(script,'workflow-inline','exec')
    assert '40685511797' in SCRIPTS[0] and '40688421947' in SCRIPTS[0]
    assert '12814229633' in SCRIPTS[0] and 'b006fa18d41fdb350d0e754829b13bad9a1fe09687df84f807e4a1ff72e5ee8d' in SCRIPTS[0]
    assert 'ef5d9ae4647c844ab948a2a36ddfdb86de665ea3' in SCRIPTS[0]
    timed=WORKFLOW.split('      - name: Measure',1)[1].split('      - name:',1)[0]
    assert 'for pass in 1-baseline 2-candidate 3-candidate 4-baseline' in timed
    assert 'benchmarks/map_array.py --threads 1 --repeat 7' in timed
    assert timed.count('ci_measurement_artifacts.py verify')==2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__',timed)
    assert 'name: map-array-same-host-minimum' in WORKFLOW and 'if: always()' in WORKFLOW
    assert 'Normalize only' not in WORKFLOW and 'no normalization authorized' in SCRIPTS[1]
    loops=[ast.literal_eval(n.iter) for n in PREPARATION.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)]
    assert sum(c*2 for name,c in loops[0] if name!='color-transforms-regression')==5872
    assert len(loops[0])==27 and ('map-array-pipeline',103) in loops[0]
    assert "{'passed': 1423}" in SCRIPTS[1] and '== 242' in SCRIPTS[1]


def manifests():
    runtime=dict(python='same',implementation='same',cache_tag='same',machine='same',system='same',dependency_libraries_sha256={'blas':'same'},versions={'numpy':'1.24.4','pytest':'same','scikit-imagexx':'same'},packages_sha256={k:{'source':'a'} for k in ('numpy','scipy','skimage')})
    wheels={f'dep{i}':dict(name=f'dep{i}',sha256='a') for i in range(13)}
    wheels['product']=dict(name='scikit-imagexx',sha256='b')
    fixed={'python/skimagexx/_map_array.py','src/bind_map_array.cpp','CMakeLists.txt','benchmarks/map_array.py','tests/map_array_contract.py','src/fft.cpp'}
    old=dict(runtime=runtime,wheels=wheels,source=dict(files_sha256=dict.fromkeys(fixed|MUTABLE,'a')))
    new=deepcopy(old)
    for name in MUTABLE:new['source']['files_sha256'][name]='changed'
    new['source']['files_sha256'][CONTRACT]='eb9305a7ec836e79899c9ad8558a4e588beb2a1d1eeff885b75f7c2faad32dad'
    new['wheels']['product']['sha256']='changed'
    return old,new


@pytest.mark.parametrize('bad',[None,'frontend','binder','cmake','harness','factory','fft','new-file','dependency','pytest','package','runtime','missing-leaf','contract'])
def test_source_and_dependency_closure(tmp_path,monkeypatch,bad):
    old,new=manifests()
    mapping={'frontend':'python/skimagexx/_map_array.py','binder':'src/bind_map_array.cpp','cmake':'CMakeLists.txt','harness':'benchmarks/map_array.py','factory':'tests/map_array_contract.py','fft':'src/fft.cpp','new-file':'new.cpp'}
    if bad in mapping:new['source']['files_sha256'][mapping[bad]]='changed'
    elif bad=='dependency':new['wheels']['dep0']['sha256']='changed'
    elif bad=='pytest':new['runtime']['versions']['pytest']='changed'
    elif bad=='package':new['runtime']['packages_sha256']['numpy']['source']='changed'
    elif bad=='runtime':new['runtime']['python']='changed'
    elif bad=='missing-leaf':del new['source']['files_sha256']['src/map_array.cpp']
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
    old='struct MapArrayInput {const void* data;int bits;};\nstruct MapArrayOutput {void* data;int stride;};\nvoid map_array_into(const MapArrayInput&,MapArrayOutput,int);\n'
    new=old+'void private_simd(int);\n'
    if bad=='input':new=new.replace('int bits','long bits')
    elif bad=='output':new=new.replace('int stride','long stride')
    elif bad=='entry':new=new.replace('Output,int','Output,long')
    for role,text in [('baseline',old),('candidate',new)]:
        p=tmp_path/'snapshots'/role/'src/map_array.hpp';p.parent.mkdir(parents=True);p.write_text(text)
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
        def now(cls,tz=None):return real(2026,10,28,15 if scenario=='expired' else 13,tzinfo=datetime.timezone.utc)
    monkeypatch.setattr(datetime,'datetime',FixedDatetime);monkeypatch.chdir(tmp_path)
    original=json.loads((ROOT/'benchmarks/map-array-minimum-initial.json').read_bytes())
    fixed_contract=tmp_path/'snapshots/baseline/coverage/contracts/map-array-benchmark.json';fixed_contract.parent.mkdir(parents=True);fixed_contract.write_bytes((ROOT/'coverage/contracts/map-array-benchmark.json').read_bytes())
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
            if scenario=='global-threshold' and candidate:cpp=10/1.03
            if scenario=='target-repeat' and name=='3-candidate' and row['case_id']=='signed64-sparse-f64-256':cpp=10/1.02
            if scenario=='preserved-loss' and candidate and row['case_id']=='dense-u8-u16-64':ref=4.0
            if scenario=='remaining-loss' and candidate and row['case_id']=='uint64-wide-c128-256':ref=4.0
            if scenario=='geometric-repeat':cpp=9.6 if candidate else (8.0 if name=='1-baseline' else 12.5)
            row.update(reference_ms=ref,cpp_ms=cpp,speedup=ref/cpp,samples_ms=[[ref]*7,[cpp]*7])
        if name=='3-candidate':
            row=report['results'][0]
            if scenario=='input-drift':row['inputs'][0]['sha256']='f'*64
            elif scenario=='output-drift':row['output_sha256']='f'*64
            elif scenario=='wrong-C-entry':row['actual_C_returns']=['fake']
            elif scenario=='wrong-core':report['metadata']['core_sha256']='f'*64
        (folder/'map-array.json').write_text(json.dumps(report))
        for when in ('before','after'):(folder/('identity-'+when+'.log')).write_text(json.dumps(identities[role]))
    exec(compile(SCRIPTS[2],'actual-decision','exec'),{})
    return json.loads((tmp_path/'results/comparison.json').read_bytes())


@pytest.mark.parametrize('scenario,expected',[('keep','KEEP'),('geometric-repeat','KEEP'),('remaining-loss','KEEP'),('global-threshold','DISCARD'),('target-repeat','DISCARD'),('preserved-loss','DISCARD'),('expired','DISCARD')])
def test_fixed_decision_metrics(tmp_path,monkeypatch,scenario,expected):
    result=run_decision(tmp_path,monkeypatch,scenario)
    assert result['status']==expected and result['raw_samples']==504 and len(result['rows'])==9
    if scenario=='remaining-loss':assert result['balanced_original_losses']==['uint64-wide-c128-256']


@pytest.mark.parametrize('scenario',['input-drift','output-drift','wrong-C-entry','wrong-core'])
def test_complete_raw_metadata_and_actual_C_guard(tmp_path,monkeypatch,scenario):
    with pytest.raises(AssertionError):run_decision(tmp_path,monkeypatch,scenario)


@pytest.mark.parametrize('mutate',[False,True])
def test_contract_actual_file_digest_gate(tmp_path,monkeypatch,mutate):
    monkeypatch.chdir(tmp_path)
    path=tmp_path/'snapshots/candidate'/CONTRACT;path.parent.mkdir(parents=True)
    raw=b'prospective contract bytes';path.write_bytes(raw+(b'changed' if mutate else b''))
    new=dict(source=dict(files_sha256={CONTRACT:hashlib.sha256(raw).hexdigest()}))
    node=next(n for n in PAIR.body if isinstance(n,ast.Assert) and 'Path(' in ast.unparse(n))
    code=compile(ast.Module(body=[node],type_ignores=[]),'actual-contract-file-authentication','exec')
    namespace=dict(hashlib=hashlib,Path=Path,new=new,contract=CONTRACT)
    if mutate:
        with pytest.raises(AssertionError):exec(code,namespace)
    else:exec(code,namespace)
