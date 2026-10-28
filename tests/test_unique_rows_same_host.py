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
WORKFLOW=(ROOT/'.github/workflows/unique-rows-same-host.yml').read_text(encoding='utf-8')
SCRIPTS=[textwrap.dedent(s) for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n",WORKFLOW,re.S)]
PREPARATION=ast.parse(SCRIPTS[1])
PAIR=next(n for n in PREPARATION.body if isinstance(n,ast.If) and ast.unparse(n.test)=="role == 'candidate'")
MUTABLE={'src/unique_rows.cpp','src/unique_rows.hpp','src/simd_unique_rows_avx2.cpp','tests/test_unique_rows_native.cpp'}
CONTRACT='coverage/contracts/unique-rows-prefix-candidate.json'


def test_frozen_transport_and_no_guard_or_harness_substitution():
    assert len(SCRIPTS)==3
    for script in SCRIPTS:compile(script,'workflow-inline','exec')
    assert '40695006125' in SCRIPTS[0] and '40702637014' in SCRIPTS[0]
    assert '12823857042' in SCRIPTS[0] and '355736e04190ae6efa5eabd99659c5b4066ae0145eb1b049bb0f2e7177e130d7' in SCRIPTS[0]
    assert '12820230837' in SCRIPTS[0] and '8f209d1ea6ce66efae8ce3709cf896e05c57ca2decfafb59f82ec9246c88ce8c' in SCRIPTS[0]
    assert 'd8adeb419d72bf284f1b4a04ce20e32a9be52074' in SCRIPTS[0]
    timed=WORKFLOW.split('      - name: Measure',1)[1].split('      - name:',1)[0]
    assert 'for pass in 1-baseline 2-candidate 3-candidate 4-baseline' in timed
    assert 'benchmarks/unique_rows.py --threads 1 --repeat 7' in timed
    assert timed.count('ci_measurement_artifacts.py verify')==2
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel|__module__',timed)
    assert 'name: unique-rows-same-host-minimum' in WORKFLOW and 'if: always()' in WORKFLOW
    assert 'Normalize only' not in WORKFLOW and 'no normalization authorized' in SCRIPTS[1]
    loops=[ast.literal_eval(n.iter) for n in PREPARATION.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)]
    assert sum(c*2 for name,c in loops[0] if name!='color-transforms-regression')==6698
    assert len(loops[0])==31 and ('unique-rows-pipeline',107) in loops[0]
    assert "{'passed': 1431}" in SCRIPTS[1] and '== 244' in SCRIPTS[1]


def manifests():
    runtime=dict(python='same',implementation='same',cache_tag='same',machine='same',system='same',dependency_libraries_sha256={'blas':'same'},versions={'numpy':'1.24.4','pytest':'same','scikit-imagexx':'same'},packages_sha256={k:{'source':'a'} for k in ('numpy','scipy','skimage')})
    wheels={f'dep{i}':dict(name=f'dep{i}',sha256='a') for i in range(13)}
    wheels['product']=dict(name='scikit-imagexx',sha256='b')
    fixed={'python/skimagexx/_unique_rows.py','src/bind_unique_rows.cpp','CMakeLists.txt','benchmarks/unique_rows.py','tests/unique_rows_contract.py','src/fft.cpp'}
    old=dict(runtime=runtime,wheels=wheels,source=dict(files_sha256=dict.fromkeys(fixed|MUTABLE,'a')))
    new=deepcopy(old)
    for name in MUTABLE:new['source']['files_sha256'][name]='changed'
    new['source']['files_sha256'][CONTRACT]='2c4f2ed8fbeaca18d686e896ac9f6a24f612dc341331f12974ffd4f55c0468a2'
    new['wheels']['product']['sha256']='changed'
    return old,new


@pytest.mark.parametrize('bad',[None,'frontend','binder','cmake','harness','factory','fft','new-file','dependency','pytest','package','runtime','missing-leaf','contract'])
def test_source_and_dependency_closure(tmp_path,monkeypatch,bad):
    old,new=manifests()
    mapping={'frontend':'python/skimagexx/_unique_rows.py','binder':'src/bind_unique_rows.cpp','cmake':'CMakeLists.txt','harness':'benchmarks/unique_rows.py','factory':'tests/unique_rows_contract.py','fft':'src/fft.cpp','new-file':'new.cpp'}
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
        def now(cls,tz=None):return real(2026,10,28,16 if scenario=='expired' else 14,tzinfo=datetime.timezone.utc)
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
            if scenario=='global-threshold' and candidate:cpp=10/1.05
            if scenario=='target-repeat' and name=='3-candidate' and row['case_id']=='f64-wide64-all-equal':cpp=10/1.09
            if scenario=='preserved-loss' and candidate and row['case_id']=='u8-short-small-unique':ref=4.0
            if scenario=='remaining-loss' and row['case_id']=='f32-coordinates-large-unique':ref=4.0
            if scenario=='float-direct-below' and candidate and row['case_id']=='f32-coordinates-large-unique':cpp=10/.96;ref=40.0
            if scenario=='float-normalized-below' and candidate and row['case_id']=='f32-strided-duplicates':ref=2.4
            if scenario=='geometric-repeat':cpp=8.9 if candidate else (8.0 if name=='1-baseline' else 12.5)
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


@pytest.mark.parametrize('scenario,expected',[('keep','KEEP'),('geometric-repeat','KEEP'),('remaining-loss','KEEP'),('global-threshold','DISCARD'),('target-repeat','DISCARD'),('preserved-loss','DISCARD'),('expired','DISCARD'),('float-direct-below','DISCARD'),('float-normalized-below','DISCARD')])
def test_fixed_decision_metrics(tmp_path,monkeypatch,scenario,expected):
    result=run_decision(tmp_path,monkeypatch,scenario)
    assert result['status']==expected and result['raw_samples']==504 and len(result['rows'])==9
    if scenario=='remaining-loss':assert result['balanced_original_losses']==['f32-coordinates-large-unique']


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


# Exact prospective contract blob from immutable candidate d8adeb4.
CONTRACT_RAW=zlib.decompress(base64.b64decode('eJyVWFtv3bgRfs+vIPzUIuci6q4ku0CSYosFdlMDbR/aohAocngOax1SS0q2Txb73zskj6Rjx1lkX2xLHM7lm5lvRv71FSE3jh/hxG7ekJtJq18m2Frz4LZO6UMP28GCVI9bzrRQgo2wvac3m3BrZOPk/K3BGjcAH9U9tB1IY6HlRkALQo0O/zwNqmejMro1tl0UtWYah2l0UVvHHPRKQ2vhXjmU9YqBizTNU6gLWua0o1kFPGG0bMo0q7K0S0AUspb1cxWTv50nZVMkSUnT4ukxs6OSjI8oQ9M6TdIsqbPqZZnWHVlalN6ZWqZJIyiwkkNZgmRQc8iqpOGybkpICl5UnKUCuGSyKxpZp8CbNC95jYI1j05Kaz6Dbpm3foO2yy1Ntmn9D5q9KYo3WbOr6orWzeskeZMk8YoAJrxTz28UX70xqhO0vTqpsT2ayfokpeHgnlnF9Ohf0PDCPGgQrWYhd1L14I/+gyc+wZbvY0G0viB2x2EI6l844k+PnDqJ9uq8ZfeP6bXQCG50e//ziVj0Iwii3H+fpuTiJNaP0Yqzfk3Nr1/3FyETddlh6lA0aQqo6rTkMq2brCiqtMwxPbwrBaaLF0UHdZdktMKSyqq6KGjCZP57MaP6khWspJhw2mS0FkmaAyZbUoGWJM1pk6YNCMmSumiELDuRpyKVWcJQd8VE9224oaFclKLGcmW1xJKWeVaXIukkJE1XZXXDU1Qtci5LTiVGKIou4UledbwsOGW0+Hbs0RgGQGnNGtqxui6LDComadZkOWedwHpPoUhlkqZFWhdZU/CuqGiTVxIdbJoOcp+/32LBq0essCOzGlxgiw40P56YvXNP4BzOsXYvki/lGbFmpZBAMUtpSvM0K2XFOe3QqaJjnCMUWUJZlmeswc6UBaJMAdMAvGt4J2+uPHow9q43TCA/6dFGOrjh5h4sO8B+fjn7GClxcX33P4cMFbR9oae17OHKZ8pkg9h0KRVNnSB11OifFE0GeQVdkVVFjrTB6kayXMgmYxVUIMqiZjxPyyqW3w3HLvDoNZF6AYTvaSSDxQewa1dbM2nhHyOrjcweYGxPQSBqGJiyCIKxAuza8XOvzaWycPVXXyw3ln4N0bPTEKmkSPLwlvUHY9V4PK3tyvq+hV8m1nuUPuKQYBYIMH4kiDVhB6a0G4lU1o0b4kYzEHZ5JCflTmzkx7dESYJ6SHgicaAQpQU8JpjCSY+UxIHkpQzHEtcHcgdnR4wlzlj/vCP/Bmu8UUcsjJPVyW4OLw4/rEWckN7Nv3366V9esu3O2EDvvqPlW3KPk8/Yd5PSY5m34/fkAeP8RACrQYHbkEk7dUCWJZ06bAHxY5oMjN95V4xcItJ/WvRu6j+T8AcxmgMZwHqbG6INwToYg/MSa24kLGAKo+I78v4SodF7pdWoWK8+x8fZmsUpj6AiQE4JIA9H06P2qesVJzgyPBbPAh+ZCun5wdg17Ga3oyXhl4xFpV797HJ/RtWgSUBNRLhDot+Sd9/Vy0X/u4cRruQW8w/oX6CFL4x/j6Z7YPeAIKiD0qwnver4/u8//vyXkFGCJRrNqfF8McYwQZgHfmT6AGJD4JHDMPqa2AZJLDuLXsfs78gn451ZSsZoD3hvzN00ECymLN16V0jEiIS+N71bnA9NhYB43/85J7+HR09mASQSJDbRy1DtcyzbULuYDAhhxNownE/Wgi+FJYa3hB8nfTfX8P4E9uB/RyYgg8Gcnlfx3crHkQyuxrws863HG38tgMyhTPXW4vbhc7oV04BKsffd1ak7ogPb3uvcRqZcmQDxcWBxZM/RtQ9Yeavh5To2Mxq+XN+sXnGDQOFFNPmyde82ZgOTfDL6sqleK8BkISXr68srT81124ZWanvjh84ll1foBP2rH09C3awyDrtdgHjR1B3A0HJsVSwLthJgmDVIggKpGMfGAcwJmG69Jj725/aA0I+BnukuKTZPL2ljT77DkcS/7aJn1vYyCcLf6yI+MAz94sW9a5eN63cV02/Ue+XoH9btA5W4wLZzKYmntfSiwZdhuPb2eeovWUfneoZdJmYsfFUAEi5qSHZN9YdUXIX9shof2+D5+9yi/y0Wjx79w8HXTogE5Uc7wVxkaM4d5++rq28CL7N0NzfWu37ZteZFHlkqrnckurV3SEXM7gfsD9b1sHfMT4zPyBwn/HBbWgx9rEvikG3FSreRJdEKNoVbKeaKusejq/b1vtnTYk/LPa32Gd1n6T7Lno6XC/kKJWUgTD9tVvqOE9k/fsmO+J7jR6cLJLgyAsoEtvU39z/c4pBkB/caowgLEVpwF4J0dJ/vD73BdMUXX4QRv9OQ49U9JiRP6+BJHJbhKdjHmalO0+m1R83qC7qvI7pr/3sOch5rPxH+ChpXhcv0xS9nMfGRzG0RjCz17Mco9M7PK9wraUYEDLhA4CQ4xyPCkbUmC5s1DeSyPwdNeMsD+fi4wwLFkftxG5NHDhOzOAclttURlzU4GpwzcSd8/+HD+x25xQmtR+I/xuEh6BLKDXHLwmGwu96kcU+Pn5HoCn4zw1p4Ea5Yeu8//LgsjT6Pdn66PY9HoxFvPWJwN2u5s/7TdLo9Y0McV+mPP7O7hXjnr4onOfOf7cae91hiEKbmcj4i4paFN1djYIgj1Ofm1vuLq3vfN3vcXC9r7MZvAhJDML1Ylw7MPlxQ1mcNDx9vby8kYAGZx/n/YeDVmbnmi3Omo12/a/jZJ5EewO5xk7HMnsNcw9wydboArU3rd5G+jWvaNTWsZ3JCSnHizv+XBR1qQd8rRPWEiXwm708XIl1WcyTSvp9Vv/rt1f8Bx//0kg=='))

def test_actual_contract_raw_and_canonical_are_same_LF_bytes():
    assert b"\r\n" not in CONTRACT_RAW
    assert hashlib.sha256(CONTRACT_RAW).hexdigest()=="2c4f2ed8fbeaca18d686e896ac9f6a24f612dc341331f12974ffd4f55c0468a2"
    assert hashlib.sha256(CONTRACT_RAW.replace(b"\r\n",b"\n")).hexdigest()=="2c4f2ed8fbeaca18d686e896ac9f6a24f612dc341331f12974ffd4f55c0468a2"


@pytest.mark.parametrize('direct,normalized,target,control_direct,control_normalized,expected',[
    (1.05,2,True,1,1,False),(2,1.05,True,1,1,False),
    (2,2,True,.97,.97,True),(2,2,True,.969,.97,False),
    (2,2,True,.97,.969,False),(2,2,False,1,1,False)])
def test_exact_strict_global_and_inclusive_float_control_boundaries(direct,normalized,target,control_direct,control_normalized,expected):
    tree=ast.parse(SCRIPTS[2]);assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='keep')
    namespace=dict(direct=direct,normalized=normalized,targets=[dict(repeat_direct=target,repeat_normalized=target)],preserved=[dict(original_repeat=True)],float_controls=[dict(direct_gain=control_direct,normalized_gain=control_normalized)],completed=1,deadline=2)
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),'actual-keep-expression','exec'),namespace)
    assert namespace['keep']==expected
