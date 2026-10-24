"""Tooling-only guards. Synthetic artifacts/modules never prove public parity."""
import base64
import builtins
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qualification_tool_under_test', ROOT / 'tools/public_qualification.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def wheel(path, entries, corrupt=None):
    entries = dict(entries)
    rows = []
    record = 'scikit_imagexx-0.1.0.dist-info/RECORD'
    for name, value in entries.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b'=').decode()
        rows.append([name, 'sha256=' + digest, str(len(value))])
    rows.append([record, '', ''])
    stream = io.StringIO(); csv.writer(stream).writerows(rows)
    entries[record] = stream.getvalue().encode()
    if corrupt:
        entries[corrupt] += b'changed'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return entries


@pytest.fixture
def complete_fake_artifact(tmp_path):
    # An explicitly synthetic complete census exercises metadata guards only.
    source_entries = {'python/skimagexx/__init__.py': b'# synthetic\n'}
    source_entries.update({'include/skimagexx/h' + str(i) + '.hpp': b'// synthetic\n' for i in range(59)})
    entries = {n.removeprefix('python/'): v for n,v in source_entries.items()}
    entries.update({'skimagexx/_core.synthetic.so': b'NOT A NATIVE LIBRARY',
                    'scikit_imagexx-0.1.0.dist-info/METADATA': b'Name: scikit-imagexx\nVersion: 0.1.0\n'})
    for i in range(103 - len(entries)):
        entries['scikit_imagexx-0.1.0.dist-info/synthetic-' + str(i)] = b'metadata'
    product = tmp_path / 'synthetic.whl'; wheel(product, entries)
    source = tmp_path / 'source.zip'; revision = 'a' * 40
    with zipfile.ZipFile(source, 'w') as archive:
        archive.comment = revision.encode()
        for name,value in source_entries.items():
            archive.writestr(name,value)
    return product, source, revision


def test_complete_wheel_inspection_is_metadata_only(complete_fake_artifact):
    result = q.inspect_wheel(*complete_fake_artifact)
    assert len(result['members_sha256']) == 104
    assert len(result['SDK_sha256']) == 59
    assert result['record_verified']
    assert 'no algorithm or whole8 acceptance' in result['scope']


def test_source_revision_is_authenticated(complete_fake_artifact):
    product,source,_ = complete_fake_artifact
    with pytest.raises(q.QualificationError,match='revision'):
        q.inspect_wheel(product,source,'b'*40)


@pytest.mark.parametrize('name', ['python/skimagexx/__init__.py','include/skimagexx/h0.hpp'])
def test_source_bytes_cannot_replace_wheel_bytes(complete_fake_artifact,tmp_path,name):
    product,source,revision = complete_fake_artifact
    replacement = tmp_path / 'replacement.zip'
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(replacement,'w') as new:
        new.comment = old.comment
        for member in old.namelist():
            new.writestr(member,old.read(member)+(b'wrong' if member==name else b''))
    with pytest.raises(q.QualificationError,match='source bytes differ'):
        q.inspect_wheel(product,replacement,revision)


def test_source_header_census_cannot_be_partial(complete_fake_artifact,tmp_path):
    product,source,revision = complete_fake_artifact
    replacement=tmp_path/'extra.zip'
    with zipfile.ZipFile(source) as old,zipfile.ZipFile(replacement,'w') as new:
        new.comment=old.comment
        for member in old.namelist():new.writestr(member,old.read(member))
        new.writestr('include/skimagexx/unshipped.hpp',b'extra')
    with pytest.raises(q.QualificationError,match='file set'):
        q.inspect_wheel(product,replacement,revision)


@pytest.mark.parametrize('unsafe',['../escaped.py','/absolute.py','dir\\escape.py','C:/escaped.py'])
def test_wheel_paths_cannot_escape(tmp_path,unsafe):
    product=tmp_path/'bad.whl';wheel(product,{unsafe:b'bad'})
    with pytest.raises(q.QualificationError,match='unsafe wheel path'):
        q.wheel_members(product)


def test_record_authenticates_bytes(tmp_path):
    product=tmp_path/'bad.whl';wheel(product,{'safe.py':b'original'},'safe.py')
    with pytest.raises(q.QualificationError,match='RECORD bytes'):
        q.wheel_members(product)


def test_duplicate_wheel_member_is_rejected(tmp_path):
    path=tmp_path/'duplicate.whl'
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(path,'w') as z:z.writestr('x',b'a');z.writestr('x',b'b')
    with pytest.raises(q.QualificationError,match='duplicate'):
        q.wheel_members(path)


def test_package_extras_including_pyc_are_not_silently_ignored(tmp_path):
    (tmp_path/'p.py').write_text('a')
    expected={'p.py':q.sha(tmp_path/'p.py')}
    q.verify_tree(tmp_path,expected)
    (tmp_path/'p.pyc').write_bytes(b'extra')
    with pytest.raises(q.QualificationError,match='file set'):
        q.verify_tree(tmp_path,expected)


@pytest.mark.parametrize('profile',q.PROFILES)
def test_runtime_and_build_header_profiles_are_separate(profile):
    assert q.runtime_pins(profile)['scikit-image']=='0.26.0'
    assert q.PROFILES[profile]['headers'].startswith('2.')
    if profile=='minimum':
        assert q.runtime_pins(profile)['numpy']=='1.24.4'
        assert q.PROFILES[profile]['headers']=='2.4.6'
    else:assert q.runtime_pins(profile)['numpy']=='2.5.4'
    assert ('colorama' in q.runtime_pins(profile))==(profile=='windows')


@pytest.mark.parametrize('profile',q.PROFILES)
def test_retained_runtime_hash_closure_is_complete(profile):
    key='modern' if profile=='debug' else profile
    rows=q.RUNTIME_WHEELS[key]
    assert set(rows)==set(q.runtime_pins(profile))
    assert len({row['filename'] for row in rows.values()})==len(rows)
    assert all(len(row['sha256'])==64 for row in rows.values())
    assert ('cp311' if profile=='minimum' else 'cp313') in rows['numpy']['filename']
    assert set(q.BUILD_HEADER_TREES[key]['trees'])=={'numpy','pybind11'}


def test_environment_eliminates_all_candidate_and_shadowing_selectors(monkeypatch,tmp_path):
    for key in ('SKIMAGEXX_METRICS_PROVEN_BASELINE','SKIMAGEXX_METRICS_CANDIDATE','SKIMAGEXX_RANK_BACKEND','PYTHONHOME','GIT_DIR'):
        monkeypatch.setenv(key,'must disappear')
    env=q.environment(tmp_path,True)
    assert {k for k in env if k.startswith('SKIMAGEXX_')}=={'SKIMAGEXX_DISABLE_SIMD'}
    assert env['PYTHONPATH']==str(tmp_path.resolve())
    assert all(env[k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'))
    assert env['OMP_WAIT_POLICY']=='PASSIVE'
    assert env['PYTHONDONTWRITEBYTECODE']=='1'
    assert not any(k in env for k in ('PYTHONHOME','GIT_DIR'))


@pytest.mark.parametrize('family',q.CORPORA)
def test_corpus_command_is_always_strict_check_only(tmp_path,family):
    (tmp_path/'wheels').mkdir()
    command=q.corpus_command(Path(sys.executable),family,tmp_path,tmp_path/'result.json',True)
    assert '--check-only' in command and '--prepare-capsule' in command
    assert command[command.index('--threads')+1:command.index('--threads')+3]==['1','4']
    assert not {'--reference-only','--repeat','--target-ms'} & set(command)


def test_absent_capsule_fails_without_running_fixtures(tmp_path):
    with pytest.raises(q.QualificationError,match='capsule absent'):
        q.corpus_command(Path(sys.executable),'statistics',tmp_path,tmp_path/'out.json')


def valid_percentile_report():
    return dict(status='complete-parity-check',checks=[dict(status='passed',actual_by_workers={'1':{},'4':{}}) for _ in range(2552)],failures=[],results=[],
                alias_checks=[dict(reference_identity=True,native_identity=None) for _ in range(9)],
                provider_checks=[dict(actual_by_workers={'1':{},'4':{}}) for _ in range(54)],
                policy_checks=[dict(actual_by_workers={'1':{},'4':{}}) for _ in range(54)],
                metadata=dict(check_only=True,reference_only=False,requested_workers=[1,4]))


def test_complete_synthetic_report_passes_metadata_guard_only(tmp_path):
    path=tmp_path/'report.json';path.write_text(json.dumps(valid_percentile_report()))
    q.validate_corpus(path,'percentile')


@pytest.mark.parametrize('mutation', ['timed','reference','partial','failed','aux','worker','rowstatus','private-native'])
def test_corpus_guard_cannot_accept_weakened_report(tmp_path,mutation):
    report=valid_percentile_report()
    if mutation=='timed':report['results']=[{}]
    elif mutation=='reference':report['metadata']['reference_only']=True
    elif mutation=='partial':report['checks'].pop()
    elif mutation=='failed':report['failures']=[{}]
    elif mutation=='aux':report['policy_checks'].pop()
    elif mutation=='worker':report['checks'][0]['actual_by_workers'].pop('4')
    elif mutation=='rowstatus':report['checks'][0]['status']='pending'
    else:report['alias_checks'][0]['native_identity']=True
    path=tmp_path/'report.json';path.write_text(json.dumps(report))
    with pytest.raises(q.QualificationError):q.validate_corpus(path,'percentile')


@pytest.mark.parametrize('expected_error',[False,True])
def test_metrics_case_requires_both_success_or_error_observations(tmp_path,expected_error):
    row=dict(status='passed',expected_error=expected_error,actual_scalar_by_workers={} if expected_error else {'1':{},'4':{}},error_parity_workers=[1,4] if expected_error else [])
    report=dict(status='complete-parity-check',checks=[dict(row) for _ in range(1231)],failures=[],results=[],metadata=dict(check_only=True,reference_only=False,requested_workers=[1,4]))
    path=tmp_path/'metrics.json';path.write_text(json.dumps(report));q.validate_metrics(path)
    if expected_error:report['checks'][0]['error_parity_workers']=[1]
    else:report['checks'][0]['actual_scalar_by_workers']={'1':{}}
    path.write_text(json.dumps(report))
    with pytest.raises(q.QualificationError,match='both workers'):q.validate_metrics(path)


def test_journal_retains_failure_and_forbids_restart(tmp_path):
    journal=q.Journal(tmp_path)
    assert journal.run('failure',[sys.executable,'-c',"print('retained');raise SystemExit(13)"])==13
    before=(tmp_path/'failure.log').read_bytes()
    row=json.loads((tmp_path/'commands.json').read_text())[0]
    assert row['state']=='terminal' and row['returncode']==13 and row['pid']>0
    with pytest.raises(q.QualificationError,match='never overwrite'):
        journal.run('failure',[sys.executable,'-c',"raise SystemExit(0)"])
    assert (tmp_path/'failure.log').read_bytes()==before


def test_missing_admission_stops_before_pytest_or_fixture_import(tmp_path,monkeypatch):
    state={'source':{}};(tmp_path/'state.json').write_text(json.dumps(state))
    monkeypatch.setattr(q,'source_state',lambda _: {})
    def reject(*_):raise q.QualificationError('missing genuine capability')
    monkeypatch.setattr(q,'admission',reject)
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        if name=='pytest':raise AssertionError('fixture engine reached before admission')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    with pytest.raises(q.QualificationError,match='genuine capability'):
        q.child(tmp_path,'public-rank')


@pytest.mark.parametrize('state',['running','terminal'])
def test_live_or_partial_command_set_cannot_be_sealed(tmp_path,monkeypatch,state):
    (tmp_path/'state.json').write_text(json.dumps({'source':{}}))
    (tmp_path/'commands.json').write_text(json.dumps([dict(label='one',state=state,returncode=0)]))
    monkeypatch.setattr(q,'source_state',lambda _: {})
    with pytest.raises(q.QualificationError,match='live/incomplete|missing required'):
        q.seal(tmp_path)
    assert not (tmp_path/'manifest.json').exists()


def test_blocked_source_does_not_start_build(tmp_path,monkeypatch):
    monkeypatch.setattr(q,'source_state',lambda _: {'revision':'a'*40})
    monkeypatch.setattr(q,'plan',lambda *_: {'blockers':['R5 absent']})
    def forbidden(*_,**__):raise AssertionError('a build was launched')
    monkeypatch.setattr(q.subprocess,'Popen',forbidden)
    with pytest.raises(q.QualificationError,match='R5 absent'):
        q.prepare(tmp_path/'fresh','modern',None)


def test_only_canonical_r5_and_source_derived_count_in_plan(tmp_path,monkeypatch):
    monkeypatch.setattr(q,'registry_expectation',lambda _: dict(functions=['filters.rank.'+n for n in q.R5],image_count=987))
    (tmp_path/'tests/cmake_consumer').mkdir(parents=True)
    (tmp_path/'tests/cmake_consumer/CMakeLists.txt').write_text('\n'.join('add_executable('+f+'_consumer x.cpp)' for f in q.FAMILIES))
    for names in q.GROUPS.values():
        for name in names:(tmp_path/'tests'/name.split('::')[0]).touch()
    (tmp_path/'tests/test_arraymap_dense_contracts.py').touch()
    (tmp_path/'benchmarks').mkdir()
    for name,_ in q.CORPORA.values():(tmp_path/'benchmarks'/name).touch()
    assert not q.plan(tmp_path,'modern',987)['blockers']
    assert q.plan(tmp_path,'modern',180)['blockers']==['explicit image count differs from source registry']
    monkeypatch.setattr(q,'registry_expectation',lambda _: dict(functions=['filters.rank.'+n for n in q.R5]+['filters.rank.generic.percentile'],image_count=987))
    assert any('must not register generic' in x for x in q.plan(tmp_path,'modern')['blockers'])


def test_real_foreach_sdk_declaration_is_recognized_without_proving_execution():
    declaration='''foreach(_family IN ITEMS rank rank_alias rank_percentile)
    add_executable(${_family}_consumer ../test_${_family}_native.cpp)
    add_test(NAME installed_${_family}_consumer COMMAND ${_family}_consumer)
endforeach()
add_executable(projective_consumer ../test_projective_native.cpp)'''
    assert q.declared_sdk_families(declaration)=={'rank','rank_alias','rank_percentile','projective'}


def test_tool_does_not_import_or_call_timing_code():
    import ast
    tree=ast.parse((ROOT/'tools/public_qualification.py').read_text())
    names={node.func.id if isinstance(node.func,ast.Name) else node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,(ast.Name,ast.Attribute))}
    assert not names & {'measure','perf_counter','perf_counter_ns','timeit','process_time'}


def test_workflow_is_four_check_only_jobs_with_retained_failures():
    text=(ROOT/'.github/workflows/public-qualification.yml').read_text()
    assert text.count('profile: ')==4
    assert 'fail-fast: false' in text and 'if: always()' in text
    assert 'include-hidden-files: true' in text
    assert 'measure:' not in text
    assert 'source_revision' in text


def test_fresh_child_tmp_path_fixture_creates_only_owned_parent(tmp_path,monkeypatch):
    """Real pytest tmp_path setup, with metadata admission stubbed (no images)."""
    tree=tmp_path/'source';(tree/'tests').mkdir(parents=True)
    (tree/'tests/test_owned_temporary.py').write_text('def test_owned(tmp_path):\n    p=tmp_path/"owned.txt"\n    p.write_text("ok")\n    assert p.read_text()=="ok"\n')
    out=tmp_path/'output';out.mkdir();(out/'state.json').write_text(json.dumps({'source':{}}))
    monkeypatch.setattr(q,'ROOT',tree)
    monkeypatch.setattr(q,'GROUPS',{'tooling-temp-smoke':('test_owned_temporary.py',)})
    monkeypatch.setattr(q,'source_state',lambda _: {})
    monkeypatch.setattr(q,'admission',lambda *_: {'scalar':False})
    monkeypatch.setattr(q.sys,'path',list(sys.path))
    assert not (out/'pytest-tmp').exists()
    assert q.child(out,'tooling-temp-smoke')==0
    assert (out/'pytest-tmp/normal-tooling-temp-smoke').is_dir()
    assert list((out/'pytest-tmp/normal-tooling-temp-smoke').rglob('owned.txt'))
    with pytest.raises(q.QualificationError,match='never reuse'):
        q.child(out,'tooling-temp-smoke')


@pytest.fixture
def simulated_interpreter_link(tmp_path,monkeypatch):
    # Windows need not have symlink privilege: simulate link metadata only.
    # Both actual file bytes and the outside interpreter bytes are still real.
    out=tmp_path/'output';path=out/'runtime-env/bin/python';path.parent.mkdir(parents=True);path.write_bytes(b'interpreter')
    target=tmp_path/'known-base-python';target.write_bytes(b'interpreter')
    base=dict(path=str(target),sha256=q.sha(target),version='synthetic',major_minor='3.13',cache_tag='synthetic',os_name='posix')
    name='runtime-env/bin/python';link=dict(raw_target=str(target),resolved_target=str(target),sha256=q.sha(target))
    provenance=dict(base=base,links={name:link});current={'target':target,'raw':str(target)}
    real_resolve=Path.resolve;real_is_symlink=Path.is_symlink;real_readlink=q.os.readlink
    def resolve(self,strict=False):
        if self==path:return current['target']
        if self==path.parent and current.get('parent_escape'):return out.parent
        return real_resolve(self,strict=strict)
    monkeypatch.setattr(Path,'resolve',resolve)
    monkeypatch.setattr(Path,'is_symlink',lambda self: self==path or real_is_symlink(self))
    monkeypatch.setattr(q.os,'readlink',lambda p: current['raw'] if Path(p)==path else real_readlink(p))
    monkeypatch.setattr(q,'base_interpreter',lambda: dict(base))
    return out,path,name,provenance,current


def test_known_external_interpreter_is_pinned_to_owner_target_and_bytes(simulated_interpreter_link):
    out,path,name,provenance,_=simulated_interpreter_link
    q.validate_interpreter_provenance(out,provenance)
    q.verify_record(out,name,dict(bytes=path.stat().st_size,sha256=q.sha(path)),provenance)


@pytest.mark.parametrize('change',['different-target-same-bytes','raw-target','bytes','owner','base-provenance'])
def test_external_interpreter_exception_does_not_admit_arbitrary_escapes(simulated_interpreter_link,change):
    out,path,name,provenance,current=simulated_interpreter_link
    if change=='different-target-same-bytes':
        other=out.parent/'other-interpreter';other.write_bytes(path.read_bytes());current.update(target=other,raw=str(other))
    elif change=='raw-target':current['raw']='a-different-link-expression'
    elif change=='bytes':path.write_bytes(b'changed interpreter')
    elif change=='owner':provenance['links']['runtime-env/bin/arbitrary-secret']=provenance['links'].pop(name)
    else:provenance['base']=dict(provenance['base'],path=str(out.parent/'forged-base'))
    with pytest.raises(q.QualificationError):q.validate_interpreter_provenance(out,provenance)


@pytest.mark.parametrize('name',['../secret','/absolute','C:/outside','runtime-env\\bin\\python'])
def test_record_traversal_guard_remains_strict(tmp_path,name):
    with pytest.raises(q.QualificationError,match='escapes output'):
        q.verify_record(tmp_path,name,dict(bytes=0,sha256='0'*64),dict(base={},links={}))


def test_unapproved_outside_file_still_rejected_even_with_interpreter_bytes(simulated_interpreter_link):
    out,path,name,provenance,_=simulated_interpreter_link
    with pytest.raises(q.QualificationError,match='escapes output'):
        q.verify_record(out,name,dict(bytes=path.stat().st_size,sha256=q.sha(path)),dict(base=provenance['base'],links={}))


def test_interpreter_owner_directory_cannot_escape(simulated_interpreter_link):
    out,_,_,provenance,current=simulated_interpreter_link;current['parent_escape']=True
    with pytest.raises(q.QualificationError,match='owner directory escapes'):
        q.validate_interpreter_provenance(out,provenance)


def test_owned_regular_file_digest_still_authenticated(tmp_path):
    path=tmp_path/'owned';path.write_bytes(b'original')
    record=dict(bytes=path.stat().st_size,sha256=q.sha(path));provenance=dict(base={},links={})
    q.verify_record(tmp_path,'owned',record,provenance)
    path.write_bytes(b'different')
    with pytest.raises(q.QualificationError,match='artifact changed'):
        q.verify_record(tmp_path,'owned',record,provenance)
