"""Inert orchestration guards; these tests never execute image algorithms."""
import ast
import base64
import copy
import csv
import gzip
import fnmatch
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import struct
import sys
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('matrix_profile_controller',ROOT/'tools/matrix_transform_profiles.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


@pytest.mark.parametrize('name',c.FORBIDDEN)
@pytest.mark.parametrize('value',['','Haswell'])
def test_forcing_is_rejected_even_empty(name,value):
    with pytest.raises(ValueError,match='override'):c.validate_default_environment({name:value})


def test_no_silent_preload_or_feature_override(monkeypatch):
    for k in c.FORBIDDEN:monkeypatch.delenv(k,raising=False)
    monkeypatch.setenv('LD_PRELOAD','unrelated.so')
    with pytest.raises(ValueError,match='preloaded'):c.environment()


def test_only_our_scalar_control_and_explicit_thread_pools(monkeypatch):
    for k in (*c.FORBIDDEN,'LD_PRELOAD'):monkeypatch.delenv(k,raising=False)
    monkeypatch.setenv('SKIMAGEXX_SOMETHING','bad')
    env=c.environment('scalar')
    assert env['SKIMAGEXX_DISABLE_SIMD']=='1'
    assert 'SKIMAGEXX_SOMETHING' not in env
    assert all(env[k]=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'))
    assert not any(k in env for k in c.FORBIDDEN)


@pytest.mark.parametrize('mutation',['missing','extra','hash'])
def test_source_closure_failclosed(mutation):
    r=dict(c.SOURCE_HASHES)
    if mutation=='missing':r.pop(next(iter(r)))
    elif mutation=='extra':r['unrelated.cpp']='0'*64
    else:r[next(iter(r))]='0'*64
    with pytest.raises(ValueError):c.validate_source_records(r)


def test_actual_frozen_git_source_and_complete_metadata_only_cohorts():
    state,blobs=c.source_state(ROOT,c.SOURCE_REVISION)
    assert state['materialized'] is False
    fixed=c.cohort_spec(blobs['tests/test_matrix_transform_native.cpp'])
    cases=list(c.case_specs(fixed))
    assert len(cases)==4000
    assert len({(x['case']['id'],x['n'],x['points_layout'],x['matrix_layout']) for x in cases})==4000
    assert {x['n'] for x in cases}>={0,1,2,64}
    for profile in c.PROFILES:
        config=c.pinned_config(blobs,profile)
        assert config['runtime']['scikit-image']=='0.26.0'
        assert config['runtime']['numpy']==('1.24.4' if profile=='minimum' else '2.5.4')
        assert len(config['runtime_wheels'])==(14 if profile=='windows' else 13)
        assert not any('scikit-imagexx' in n for n in config['runtime'])


def test_real_frozen_cpp_quoted_include_closure_is_retained():
    _,blobs=c.source_state(ROOT,c.SOURCE_REVISION)
    visited=set();pending=[n for n in c.SOURCE_HASHES if n.endswith('.cpp')]
    while pending:
        name=pending.pop()
        if name in visited:continue
        visited.add(name)
        for include in re.findall(r'^\s*#include\s*"([^"]+)"',blobs[name].decode(),re.M):
            candidates=['include/'+include] if include.startswith('skimagexx/') else [str(PurePosixPath(name).parent/include),'src/'+include]
            matches=[n for n in candidates if n in blobs]
            assert matches,(name,include,'missing actual transitive source dependency')
            pending.append(matches[0])
    assert {'include/skimagexx/convert.hpp','include/skimagexx/distance.hpp','include/skimagexx/local_threshold.hpp'}<=visited


@pytest.mark.parametrize('expression',["danger()","dict(x=danger())","__import__('os')","x+1"])
def test_config_literal_cannot_execute_code(expression):
    with pytest.raises(ValueError):c.literal(ast.parse(expression,mode='eval').body)


def test_native_trial_is_hosted_only_before_any_build(monkeypatch,tmp_path):
    monkeypatch.delenv('GITHUB_ACTIONS',raising=False)
    monkeypatch.setattr(c.venv,'EnvBuilder',lambda **kw:pytest.fail('must not create environment'))
    with pytest.raises(ValueError,match='hosted-only'):c.prepare(ROOT,c.SOURCE_REVISION,'modern',tmp_path/'absent')
    assert not (tmp_path/'absent').exists()


@pytest.mark.parametrize('paths',[
    (r'D:\a\source with spaces',r'C:\Users\agent\python.exe',r'D:\tools\é\pybind11'),
    ('/home/runner/source','/opt/python/bin/python','/opt/pybind11/share/cmake'),
])
def test_all_three_cmake_definitions_avoid_eval_path_escapes_without_mutating_facts(paths):
    original=tuple(paths)
    definitions=c.cmake_path_definitions(*paths)
    assert definitions==['-D'+name+'='+value.replace('\\','/') for name,value in
                         zip(('SOURCE','Python_EXECUTABLE','pybind11_DIR'),paths)]
    assert len(definitions)==3 and all('\\' not in value for value in definitions)
    assert paths==original
    # Exact callsite uses this tested encoder; unrelated generator/tool arguments
    # and raw path identity capture retain their existing behavior.
    source=ast.parse((ROOT/'tools/matrix_transform_profiles.py').read_text())
    prepare=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
    assert any(isinstance(n,ast.Starred) and isinstance(n.value,ast.Call) and
               isinstance(n.value.func,ast.Name) and n.value.func.id=='cmake_path_definitions' for n in ast.walk(prepare))


@pytest.mark.parametrize('wrong',['outside','unrecorded','changed'])
def test_actual_library_origin_and_wheel_authentication(tmp_path,wrong):
    libs=tmp_path/'numpy.libs';libs.mkdir();p=libs/'real.so';p.write_bytes(b'real loaded library')
    records={str(p.resolve()):c.sha(p)}
    if wrong=='outside':p=tmp_path/'unrelated.so';p.write_bytes(b'real loaded library')
    elif wrong=='unrecorded':records={}
    else:p.write_bytes(b'different loaded library')
    with pytest.raises(ValueError):c.validate_library_origin(p,libs,records)


def test_origin_records_actual_bytes(tmp_path):
    libs=tmp_path/'numpy.libs';libs.mkdir();p=libs/'actual.so';p.write_bytes(b'not executed')
    r=c.validate_library_origin(p,libs,{str(p.resolve()):c.sha(p)})
    assert r['sha256']==c.sha(p) and r['path']==str(p.resolve())


def instance(version,**kw):
    return dict(installationVersion=version,installationPath='C:/VS/'+version,isComplete=True,isLaunchable=True,**kw)


@pytest.mark.parametrize('major',[17,18])
def test_actual_available_vs_generator_not_assumed_2022(major):
    caps={'generators':[{'name':f'Visual Studio {major} example'}]}
    got=c.windows_generator(caps,[instance(f'{major}.1.2.3')])
    assert got==['-G',f'Visual Studio {major} example','-A','x64',f'-DCMAKE_GENERATOR_INSTANCE=C:/VS/{major}.1.2.3']


@pytest.mark.parametrize('problem',['missing','incomplete','unlaunchable','mismatch'])
def test_vs_ineligible_instance_fails(problem):
    i=instance('18.1.0')
    if problem=='missing':i.pop('isComplete')
    elif problem=='incomplete':i['isComplete']=False
    elif problem=='unlaunchable':i['isLaunchable']=False
    else:i['installationVersion']='17.0.0'
    with pytest.raises(ValueError):c.windows_generator({'generators':[{'name':'Visual Studio 18 2026'}]},[i])


def test_pe_import_ownership_parser(tmp_path):
    b=bytearray(1024);b[:2]=b'MZ';struct.pack_into('<I',b,0x3c,128);b[128:132]=b'PE\0\0'
    struct.pack_into('<H',b,134,1);struct.pack_into('<H',b,148,240)
    struct.pack_into('<H',b,152,0x20b);struct.pack_into('<I',b,272,0x1000)
    struct.pack_into('<IIII',b,400,512,0x1000,512,512)
    struct.pack_into('<IIIII',b,512,1,0,0,0x1060,1)
    name=b'libscipy_openblas64_test.dll\0';b[608:608+len(name)]=name
    p=tmp_path/'fake.pyd';p.write_bytes(b)
    assert c.pe_imports(p)==[name[:-1].decode()]
    b[152:154]=b'\x0b\x01';p.write_bytes(b)
    with pytest.raises(ValueError,match='x64'):c.pe_imports(p)


def result():
    return dict(schema=c.SCHEMA,original_cases=4000,rows=8000,workers=[1,4],operation_timings=0,
        source={'revision':c.SOURCE_REVISION,'canonical_members':dict(c.SOURCE_HASHES)},public_acceptance=False,
        profile_admission='pending-source-tree-audit',counts={str(i):dict(completed=8000,exact=7999,different=1,rejected=0) for i in range(1,6)})


@pytest.mark.parametrize('key,value',[('public_acceptance',True),('profile_admission','accepted'),('rows',7999),
    ('workers',[1]),('operation_timings',1),('original_cases',3999)])
def test_discovery_never_accepted_or_cherry_picked(key,value):
    r=result();r[key]=value
    with pytest.raises(ValueError):c.validate_discovery(r)


def test_adverse_discovery_counts_are_valid_and_retained():
    c.validate_discovery(result())
    r=result();r['counts']['1']['different']=0
    with pytest.raises(ValueError,match='accounting'):c.validate_discovery(r)


def test_wheel_full_record_rejects_unhashed_member(tmp_path):
    p=tmp_path/'toy.whl'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('toy-1.dist-info/METADATA','Name: toy\nVersion: 1\n')
        z.writestr('toy.py','not executed')
        z.writestr('toy-1.dist-info/RECORD','toy.py,,\ntoy-1.dist-info/METADATA,,\ntoy-1.dist-info/RECORD,,\n')
    with pytest.raises(ValueError,match='unhashed'):c.wheel_metadata(p)


def test_source_and_harness_baseline_and_leaf_are_separate():
    assert c.CMAKE.count('${SOURCE}/src/image.cpp')==1
    assert '-mno-avx2 -mno-fma' in c.CMAKE and '-mavx2;-mfma' in c.CMAKE
    assert 'NO_EXTRAS' in c.CMAKE and 'OpenMP REQUIRED' in c.CMAKE
    assert 'bind_matrix_transform(m)' in c.MODULE
    assert 'matrix_transform_into' not in c.MODULE  # real binder, not a substitute implementation


@pytest.mark.parametrize('problem',[None,'missing','extra','changed'])
def test_controller_source_closure_is_authenticated(tmp_path,problem):
    records={}
    for name in c.CONTROLLER_PATHS:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
        records[name]=c.sha(p)
    if problem=='missing':records.pop(next(iter(records)))
    elif problem=='extra':records['unrelated.py']='0'*64
    elif problem=='changed':(tmp_path/c.CONTROLLER_PATHS[0]).write_bytes(b'changed')
    if problem is None:c.validate_controller_members(tmp_path,records)
    else:
        with pytest.raises(ValueError):c.validate_controller_members(tmp_path,records)


def test_no_algorithm_modules_or_timing_functions_imported():
    tree=ast.parse((ROOT/'tools/matrix_transform_profiles.py').read_text())
    top=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all('numpy' not in ast.unparse(n) and 'skimage' not in ast.unparse(n) for n in top)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                   n.func.attr in ('perf_counter','timeit','monotonic','process_time') for n in ast.walk(tree))


def test_workflow_is_manual_four_fresh_hosts_and_never_dispatches():
    text=(ROOT/'.github/workflows/matrix-transform-profiles.yml').read_text()
    for value in ('workflow_dispatch:','windows','modern','minimum','debug','source_revision','if: always()'):
        assert value in text
    assert 'schedule:' not in text and 'gh workflow run' not in text
    assert 'OPENBLAS_CORETYPE:' not in text and 'NPY_DISABLE_CPU_FEATURES:' not in text


def test_upload_keeps_authenticated_hidden_controller_provenance_only_within_allowlist():
    text=(ROOT/'.github/workflows/matrix-transform-profiles.yml').read_text()
    upload=text.split('uses: actions/upload-artifact@v4',1)[1]
    assert re.search(r'^\s+include-hidden-files:\s+true\s*$',upload,re.M)
    paths=[p.strip().replace('${{ matrix.profile }}','modern') for p in upload.split('path: |',1)[1].splitlines() if p.strip()]
    assert paths==['build/matrix-default-modern/artifact/**',*[f'build/matrix-default-modern/*.{e}' for e in ('json','stdout','stderr','gz')]]
    provenance='build/matrix-default-modern/artifact/controller/.github/workflows/matrix-transform-profiles.yml'
    assert any(fnmatch.fnmatchcase(provenance,p) for p in paths)
    # Both the artifact producer and explicit upload selection exclude entire
    # checkout/environment trees. Hidden-file inclusion preserves a known source
    # member, rather than widening capture to the parent repository.
    assert '.github/workflows/matrix-transform-profiles.yml' in c.CONTROLLER_PATHS
    for forbidden in ('build/matrix-profile-source/.git/config',
                      'build/matrix-default-modern/runtime-env/.env',
                      'build/matrix-default-modern/build-env/.git/config'):
        assert not any(fnmatch.fnmatchcase(forbidden,p) for p in paths)


def test_artifact_undeclared_nested_verification_is_not_ignored(tmp_path):
    a=tmp_path/'artifact';a.mkdir();(a/'nested').mkdir();(a/'nested/verification.json').write_text('{}')
    c.dump(a/'manifest.json',dict(schema=c.SCHEMA,records={},source={'canonical_members':c.SOURCE_HASHES},
        operation_timings=0,all_commands_terminal=True,public_acceptance=False,profile_admission='pending-source-tree-audit'))
    with pytest.raises(ValueError,match='closure'):c.verify(a)


@pytest.mark.parametrize('fault',[None,'duplicate','false_exact','bad_counter','missing_profile','wrong_worker'])
def test_all_raw_rows_reconcile_without_dropping_mismatches(tmp_path,monkeypatch,fault):
    # Intentionally inert integer metadata, not image data or native execution.
    monkeypatch.setattr(c,'case_specs',lambda spec:iter([
        dict(case={'id':str(i)},n=1,points_layout='C',matrix_layout='F') for i in range(4000)]))
    path=tmp_path/'rows.gz'
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for w in (1,4):
            for i in range(4000):
                different=i==0 and w==1
                row=dict(id=str(i),requested_rows=1,points_layout='C',matrix_layout='F',workers=w,mode='normal',
                    expected_words=[[1]],candidates=[dict(profile=p,status=0,completed=1,
                    words=[[2 if different else 1]],exact=not different,admission='UNACCEPTED-arithmetic-candidate') for p in range(1,6)])
                if i==1 and w==1:
                    if fault=='duplicate':row['id']='0'
                    elif fault=='false_exact':row['candidates'][0]['exact']=False
                    elif fault=='bad_counter':row['candidates'][0]['completed']=0
                    elif fault=='missing_profile':row['candidates'].pop()
                    elif fault=='wrong_worker':row['workers']=2
                stream.write(json.dumps(row)+'\n')
    if fault is None:c.validate_rows(path,result(),{},'normal')
    else:
        with pytest.raises(ValueError):c.validate_rows(path,result(),{},'normal')
