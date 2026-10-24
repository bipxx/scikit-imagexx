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
    result = q.inspect_wheel(*complete_fake_artifact,scope="rank14")
    assert len(result['members_sha256']) == 104
    assert len(result['SDK_sha256']) == 59
    assert result['record_verified']
    assert 'no algorithm or whole8 acceptance' in result['scope']


@pytest.fixture
def complete_tv_fake_artifact(complete_fake_artifact,tmp_path):
    product,source,revision=complete_fake_artifact
    additions={'include/skimagexx/'+n+'.hpp':b'// synthetic TV header\n'
               for n in ('tv_chambolle','tv_reductions','tv_denoise')}
    additions['skimagexx/_tv_chambolle.py']=b'# synthetic TV helper\n'
    with zipfile.ZipFile(product) as archive:
        entries={n:archive.read(n) for n in archive.namelist() if not n.endswith('/RECORD')}
    entries.update(additions);new_product=tmp_path/'tv-synthetic.whl';wheel(new_product,entries)
    new_source=tmp_path/'tv-source.zip'
    with zipfile.ZipFile(source) as old,zipfile.ZipFile(new_source,'w') as archive:
        archive.comment=old.comment
        for n in old.namelist():archive.writestr(n,old.read(n))
        for n,value in additions.items():archive.writestr('python/'+n if n.startswith('skimagexx/') else n,value)
    return new_product,new_source,revision


def test_historical_tv108_62_requires_explicit_scope_and_original104_stays_explicit(complete_fake_artifact,complete_tv_fake_artifact):
    result=q.inspect_wheel(*complete_tv_fake_artifact,scope='tv17')
    assert result['product_scope']=='tv17' and len(result['members_sha256'])==108 and len(result['SDK_sha256'])==62
    with pytest.raises(q.QualificationError,match='complete110'):
        q.inspect_wheel(*complete_fake_artifact)
    with pytest.raises(q.QualificationError,match='complete104'):
        q.inspect_wheel(*complete_tv_fake_artifact,scope='rank14')


@pytest.mark.parametrize('member',['include/skimagexx/tv_chambolle.hpp','include/skimagexx/tv_reductions.hpp',
                                  'include/skimagexx/tv_denoise.hpp','python/skimagexx/_tv_chambolle.py'])
def test_tv_source_cannot_omit_any_new_header_or_helper(complete_tv_fake_artifact,tmp_path,member):
    product,source,revision=complete_tv_fake_artifact;replacement=tmp_path/'missing-TV.zip'
    with zipfile.ZipFile(source) as old,zipfile.ZipFile(replacement,'w') as archive:
        archive.comment=old.comment
        for n in old.namelist():
            if n!=member:archive.writestr(n,old.read(n))
    with pytest.raises(q.QualificationError,match='TV product scope/source'):
        q.inspect_wheel(product,replacement,revision,scope='tv17')


def test_source_revision_is_authenticated(complete_fake_artifact):
    product,source,_ = complete_fake_artifact
    with pytest.raises(q.QualificationError,match='revision'):
        q.inspect_wheel(product,source,'b'*40,scope='rank14')


@pytest.mark.parametrize('name', ['python/skimagexx/__init__.py','include/skimagexx/h0.hpp'])
def test_source_bytes_cannot_replace_wheel_bytes(complete_fake_artifact,tmp_path,name):
    product,source,revision = complete_fake_artifact
    replacement = tmp_path / 'replacement.zip'
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(replacement,'w') as new:
        new.comment = old.comment
        for member in old.namelist():
            new.writestr(member,old.read(member)+(b'wrong' if member==name else b''))
    with pytest.raises(q.QualificationError,match='source bytes differ'):
        q.inspect_wheel(product,replacement,revision,scope="rank14")


def test_source_header_census_cannot_be_partial(complete_fake_artifact,tmp_path):
    product,source,revision = complete_fake_artifact
    replacement=tmp_path/'extra.zip'
    with zipfile.ZipFile(source) as old,zipfile.ZipFile(replacement,'w') as new:
        new.comment=old.comment
        for member in old.namelist():new.writestr(member,old.read(member))
        new.writestr('include/skimagexx/unshipped.hpp',b'extra')
    with pytest.raises(q.QualificationError,match='file set'):
        q.inspect_wheel(product,replacement,revision,scope="rank14")


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
    monkeypatch.setattr(q,'registry_expectation',lambda _: dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH}),image_count=987))
    (tmp_path/'tests/cmake_consumer').mkdir(parents=True)
    (tmp_path/'tests/cmake_consumer/CMakeLists.txt').write_text('\n'.join('add_executable('+f+'_consumer x.cpp)' for f in q.FAMILIES))
    for names in q.GROUPS.values():
        for name in names:(tmp_path/'tests'/name.split('::')[0]).touch()
    (tmp_path/'tests/test_arraymap_dense_contracts.py').touch()
    for name in q.R6_SOURCE_PATHS+q.R7_SOURCE_PATHS+q.TV_SOURCE_PATHS+q.BREGMAN_SOURCE_PATHS:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    for family in q.FAMILIES:(tmp_path/'tests'/('test_'+family+'_native.cpp')).touch()
    (tmp_path/'benchmarks').mkdir()
    for name,_ in q.CORPORA.values():(tmp_path/'benchmarks'/name).touch()
    assert not q.plan(tmp_path,'modern',987)['blockers']
    assert q.plan(tmp_path,'modern',180)['blockers']==['explicit image count differs from source registry']
    monkeypatch.setattr(q,'registry_expectation',lambda _: dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH})+['filters.rank.generic.percentile'],image_count=987))
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


def msvc_provenance_fixture(major=18):
    year,tools,compiler,platform_toolset=(2026,'14.51.36231','19.51.36260.0','v145') if major==18 else (2022,'14.44.35207','19.44.35217.0','v143')
    instance=f'C:/VS/{major}/Enterprise';executable=instance+f'/VC/Tools/MSVC/{tools}/bin/Hostx64/x64/cl.exe'
    cache=f'CMAKE_GENERATOR:INTERNAL=Visual Studio {major} {year}\nCMAKE_GENERATOR_INSTANCE:INTERNAL={instance}\nCMAKE_GENERATOR_PLATFORM:INTERNAL=x64\n'
    fields=f'set(CMAKE_CXX_COMPILER "{executable}")\nset(CMAKE_CXX_COMPILER_ID "MSVC")\nset(CMAKE_CXX_COMPILER_VERSION "{compiler}")\nset(CMAKE_CXX_COMPILER_ARCHITECTURE_ID "x64")\n'
    project=f'''<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
      <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'"><PlatformToolset>{platform_toolset}</PlatformToolset></PropertyGroup>
      <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'"><ClCompile><RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary></ClCompile></ItemDefinitionGroup>
      <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'"><ClCompile><RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary></ClCompile></ItemDefinitionGroup>
    </Project>'''
    model=q.windows_toolchain_model(cache,fields,project,'Release')
    capabilities=dict(generators=[dict(name=model['generator'],platformSupport=True,toolsetSupport=True,supportedPlatforms=['x64'])])
    instances=[dict(installationPath=instance,installationVersion=f'{major}.0.12345',isComplete=True,isLaunchable=True)]
    return model,capabilities,instances,(cache,fields,project)


@pytest.mark.parametrize('major',[17,18])
def test_sdk_uses_actual_successful_wheel_instance_and_exact_toolset(major):
    model,capabilities,instances,_=msvc_provenance_fixture(major)
    selected=q.select_visual_studio(model,capabilities,instances)
    assert selected['generator']==model['generator']
    assert selected['instance']==model['instance']
    assert selected['architecture']=='x64'
    assert selected['runtime']=='MultiThreadedDLL'
    assert selected['toolset']==model['platform_toolset']+',version='+model['toolset_version']+',host=x64'


def test_vs2026_only_does_not_require_uninstalled_vs2022():
    model,capabilities,instances,_=msvc_provenance_fixture(18)
    assert q.select_visual_studio(model,capabilities,instances)['generator']=='Visual Studio 18 2026'


def test_vs2022_wheel_does_not_silently_select_newer_installed_instance():
    model,capabilities,instances,_=msvc_provenance_fixture(17)
    other,other_cap,other_instances,_=msvc_provenance_fixture(18)
    capabilities['generators']+=other_cap['generators'];instances+=other_instances
    selected=q.select_visual_studio(model,capabilities,instances)
    assert selected['generator']=='Visual Studio 17 2022' and selected['instance']==model['instance']


@pytest.mark.parametrize('failure',['missing-generator','missing-instance','wrong-version','incomplete','ambiguous','wrong-compiler-owner','unsupported-architecture'])
def test_unavailable_or_mismatched_sdk_toolchain_is_never_skipped(failure):
    model,capabilities,instances,_=msvc_provenance_fixture()
    if failure=='missing-generator':capabilities['generators']=[]
    elif failure=='missing-instance':instances=[]
    elif failure=='wrong-version':instances[0]['installationVersion']='17.0.1'
    elif failure=='incomplete':instances[0]['isComplete']=False
    elif failure=='ambiguous':instances*=2
    elif failure=='unsupported-architecture':capabilities['generators'][0]['supportedPlatforms']=['ARM64']
    else:model['compiler']='C:/another-instance/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe'
    with pytest.raises(q.QualificationError):q.select_visual_studio(model,capabilities,instances)


@pytest.mark.parametrize('field',['compiler_version','toolset_version','runtime','architecture','compiler_sha256','instance'])
def test_sdk_configure_must_match_sealed_wheel_before_compiling_clients(field):
    model,_,_,_=msvc_provenance_fixture();model['compiler_sha256']='a'*64
    changed=dict(model);changed[field]='mismatch'
    with pytest.raises(q.QualificationError,match='SDK/wheel toolchain mismatch'):
        q.require_same_windows_toolchain(model,changed)


def test_compiler_architecture_and_runtime_are_observed_not_assumed():
    _,_,_,(cache,fields,project)=msvc_provenance_fixture()
    with pytest.raises(q.QualificationError,match='architecture'):
        q.windows_toolchain_model(cache,fields.replace('"x64"','"x86"'),project,'Release')
    with pytest.raises(q.QualificationError,match='runtime'):
        q.windows_toolchain_model(cache,fields,project.replace('MultiThreadedDLL','UnknownRuntime'),'Release')


# These synthetic modules test metadata rejection only, never native/public proof.
def bilateral_metadata(monkeypatch):
    # The positive admission object is a deliberately marked harness stub. Only
    # that exact object is classified as builtin here; negatives still exercise
    # the real inspect.isbuiltin predicate, and no image call is allowed.
    def entry(*args):raise AssertionError('metadata stub cannot execute native work')
    entry.__name__='rank_bilateral_run';entry.__module__='builtins'
    original=q.inspect.isbuiltin
    monkeypatch.setattr(q.inspect,'isbuiltin',lambda value:value is entry or original(value))
    core=types.SimpleNamespace(__name__='builtins',rank_bilateral_interface='rank-bilateral-v1',
        rank_bilateral_capabilities=0x1c000000,rank_bilateral_run=entry,get_num_threads=lambda:1)
    wrapper=types.SimpleNamespace(_core=core)
    rank=types.SimpleNamespace(__all__=list(q.R6));nested=types.SimpleNamespace(__all__=list(q.R6))
    generic=types.SimpleNamespace(__all__=[]);rank.bilateral=nested
    for name in q.R6:
        def function(*args,**kwargs):raise AssertionError('tooling stub cannot execute an image')
        function.__module__='skimagexx._rank'
        for module in (wrapper,rank,nested):setattr(module,name,function)
    return core,rank,nested,generic,wrapper,set(q.R6_PATHS)


def test_bilateral_metadata_guard_has_no_fixture_calls(monkeypatch):
    q.bilateral_admission(*bilateral_metadata(monkeypatch))


@pytest.mark.parametrize('failure',['version','capability','partial-bits','entry','entry-owner','registry','identity','core','worker','forwarding','generic','nested-exports'])
def test_bilateral_admission_rejects_incomplete_or_private_metadata(failure,monkeypatch):
    args=list(bilateral_metadata(monkeypatch));core,rank,nested,generic,wrapper,registered=args
    if failure=='version':core.rank_bilateral_interface='rank-bilateral-v0'
    elif failure=='capability':del core.rank_bilateral_capabilities
    elif failure=='partial-bits':core.rank_bilateral_capabilities=0x4000000
    elif failure=='entry':core.rank_bilateral_run=lambda:None
    elif failure=='entry-owner':core.__name__='not_the_entry_owner'
    elif failure=='registry':registered.pop()
    elif failure=='identity':nested.mean_bilateral=lambda:None
    elif failure=='core':wrapper._core=object()
    elif failure=='worker':core.get_num_threads=lambda:4
    elif failure=='forwarding':rank.mean_bilateral.__module__='skimage.filters.rank.bilateral'
    elif failure=='generic':generic.mean_bilateral=rank.mean_bilateral
    else:nested.__all__.pop()
    with pytest.raises(q.QualificationError):q.bilateral_admission(*args)


def valid_bilateral_report():
    row=dict(status='passed',reference={'defined':'observation'},actual_by_workers={'1':{'defined':'observation'},'4':{'defined':'observation'}})
    aliases=[dict(row,canonical='filters.rank.'+n,alias='filters.rank.bilateral.'+n,reference_identity=True,native_identity=True) for n in q.R6]
    return dict(status='complete-parity-check',checks=[dict(row) for _ in range(1008)],failures=[],results=[],
                alias_checks=aliases,provider_checks=[dict(row) for _ in range(96)],policy_checks=[dict(row) for _ in range(18)],
                metadata=dict(check_only=True,reference_only=False,requested_workers=[1,4],validation_scope='strict actual public R6'))


def test_bilateral_complete_synthetic_report_is_metadata_only(tmp_path):
    path=tmp_path/'r6.json';path.write_text(json.dumps(valid_bilateral_report()));q.validate_corpus(path,'bilateral')


@pytest.mark.parametrize('failure',['raster-count','providers','policies','aliases','worker','identity','alias-path','private','timed','observation'])
def test_bilateral_report_cannot_seal_partial_or_substituted_observations(tmp_path,failure):
    r=valid_bilateral_report()
    if failure=='raster-count':r['checks'].pop()
    elif failure=='providers':r['provider_checks'].pop()
    elif failure=='policies':r['policy_checks'].pop()
    elif failure=='aliases':r['alias_checks'].pop()
    elif failure=='worker':r['provider_checks'][0]['actual_by_workers']={'1':{}}
    elif failure=='identity':r['alias_checks'][0]['native_identity']=False
    elif failure=='alias-path':r['alias_checks'][0]['alias']='filters.rank.generic.mean_bilateral'
    elif failure=='private':r['metadata']['validation_scope']='direct private-core prototype'
    elif failure=='timed':r['results']=[{}]
    else:r['checks'][0]['reference']={'different':'observation'}
    path=tmp_path/'r6.json';path.write_text(json.dumps(r))
    with pytest.raises(q.QualificationError):q.validate_corpus(path,'bilateral')


@pytest.mark.parametrize('missing',q.R6_SOURCE_PATHS)
def test_bilateral_source_helper_omission_is_a_plan_blocker(tmp_path,monkeypatch,missing):
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|{q.TV_PATH,q.BREGMAN_PATH}),image_count=186))
    (tmp_path/'tests/cmake_consumer').mkdir(parents=True)
    (tmp_path/'tests/cmake_consumer/CMakeLists.txt').write_text('\n'.join('add_executable('+f+'_consumer x.cpp)' for f in q.FAMILIES))
    for name in q.R6_SOURCE_PATHS:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    (tmp_path/missing).unlink()
    assert 'missing required R6 source: '+missing in q.plan(tmp_path,'modern')['blockers']


def test_bilateral_sdk_thirteenth_client_is_mandatory_in_plan(tmp_path,monkeypatch):
    assert len(q.ORIGINAL_FAMILIES)==14 and q.ORIGINAL_FAMILIES[-2:] == ('rank_bilateral','rank_decisions')
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|{q.TV_PATH,q.BREGMAN_PATH}),image_count=186))
    (tmp_path/'tests/cmake_consumer').mkdir(parents=True)
    (tmp_path/'tests/cmake_consumer/CMakeLists.txt').write_text('\n'.join('add_executable('+f+'_consumer x.cpp)' for f in q.FAMILIES if f!='rank_bilateral'))
    assert 'missing required installed SDK client: rank_bilateral' in q.plan(tmp_path,'modern')['blockers']


def test_bilateral_groups_are_public_and_old_scopes_remain():
    assert q.GROUPS['public-bilateral']==('test_rank_bilateral_contracts.py','test_rank_bilateral_public.py','test_rank_bilateral_masked_center_public.py','test_rank_bilateral_endian_public.py')
    assert not any('pipeline' in n or 'supplement.py' in n for n in q.GROUPS['public-bilateral'])
    assert q.CORPORA['bilateral']==('rank_bilateral.py',1008)
    assert {'statistics','percentile','consumers'}<=q.CORPORA.keys()
    assert {'public-rank','public-consumers','metrics-leaf-half','metrics-policies','shared'}<=q.GROUPS.keys()


def test_registry_separates_registered_image_and_metadata_counts(tmp_path):
    path=tmp_path/'python/skimagexx';path.mkdir(parents=True)
    (path/'coverage.py').write_text('def native_functions(): return [str(i) for i in range(189)]\ndef execution_kind(n): return "native" if int(n)<186 else "metadata"\n')
    r=q.registry_expectation(tmp_path)
    assert r['registered_count']==189 and r['image_count']==186 and r['metadata_count']==3


@pytest.mark.parametrize('change',['missing-thirteenth','duplicate','unexpected'])
def test_actual_sdk_discovery_cannot_accept_twelve_or_substituted_clients(change):
    listing=dict(tests=[dict(name='installed_'+f+'_consumer') for f in q.FAMILIES])
    q.validate_sdk_listing(listing)
    if change=='missing-thirteenth':listing['tests'].pop()
    elif change=='duplicate':listing['tests'].append(dict(listing['tests'][0]))
    else:listing['tests'][-1]['name']='installed_private_adapter_consumer'
    with pytest.raises(q.QualificationError,match='SDK test discovery'):q.validate_sdk_listing(listing)


@pytest.mark.parametrize('failure',['float-capability','integer-subclass','boolean-capability','borrowed-builtin','spoof-class','duplicate-exports','reordered-exports'])
def test_bilateral_strict_metadata_rejection_precedes_fixture_engine(tmp_path,monkeypatch,failure):
    args=list(bilateral_metadata(monkeypatch));core,rank,nested,generic,wrapper,registered=args
    if failure=='float-capability':core.rank_bilateral_capabilities=float(0x1c000000)
    elif failure=='integer-subclass':
        class Capability(int):pass
        core.rank_bilateral_capabilities=Capability(0x1c000000)
    elif failure=='boolean-capability':core.rank_bilateral_capabilities=True
    elif failure=='borrowed-builtin':core.rank_bilateral_run=len
    elif failure=='spoof-class':
        Spoof=type('builtin_function_or_method',(),{'__module__':'builtins','__name__':'rank_bilateral_run'})
        core.rank_bilateral_run=Spoof()
    elif failure=='duplicate-exports':nested.__all__.append(q.R6[0])
    else:nested.__all__.reverse()
    (tmp_path/'state.json').write_text(json.dumps({'source':{}}))
    monkeypatch.setattr(q,'source_state',lambda _: {})
    monkeypatch.setattr(q,'admission',lambda *_: q.bilateral_admission(*args))
    original=builtins.__import__
    def guard(name,*a,**kw):
        if name=='pytest' or 'bilateral_contract' in name:raise AssertionError('fixture engine reached before strict R6 rejection')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guard)
    with pytest.raises(q.QualificationError):q.child(tmp_path,'public-bilateral')
    assert not (tmp_path/'pytest-tmp').exists()

# R7 fixtures below are deliberately synthetic metadata models. Their entry
# must never execute and none of these unit tests prove native image parity.
def decisions_metadata(monkeypatch):
    def entry(*args):raise AssertionError('metadata entry cannot execute native work')
    entry.__name__='rank_decisions_run';entry.__module__='builtins'
    original=q.inspect.isbuiltin
    monkeypatch.setattr(q.inspect,'isbuiltin',lambda value:value is entry or original(value))
    core=types.SimpleNamespace(__name__='builtins',rank_decisions_interface='rank-decisions-v1',
        rank_decisions_capabilities=0x60000000,rank_decisions_run=entry,get_num_threads=lambda:1)
    wrapper=types.SimpleNamespace(_core=core)
    rank=types.SimpleNamespace(__all__=['otsu','majority']);generic=types.SimpleNamespace(__all__=['otsu'])
    for name in q.R7:
        def function(*a,**kw):raise AssertionError('metadata function cannot execute an image')
        function.__name__=name;function.__module__='skimagexx._rank'
        for module in (rank,generic,wrapper):setattr(module,name,function)
    return core,rank,generic,wrapper,list(q.R7_PATHS)


def test_decisions_metadata_only_own_aliases_and_nonstar_majority(monkeypatch):
    q.decisions_admission(*decisions_metadata(monkeypatch))


@pytest.mark.parametrize('failure',['version','missing-cap','partial-cap','float-cap','bool-cap','subclass-cap',
    'python-entry','borrowed-entry','entry-name','entry-owner','spoof-builtin','missing-path','duplicate-path','extra-path',
    'core','worker','identity','lazy-attribute','forwarding','name','canonical-star','duplicate-star','generic-otsu','generic-majority'])
def test_decisions_strict_rejection_before_fixture_engine(tmp_path,monkeypatch,failure):
    args=list(decisions_metadata(monkeypatch));core,rank,generic,wrapper,registered=args
    if failure=='version':core.rank_decisions_interface='rank-decisions-v0'
    elif failure=='missing-cap':del core.rank_decisions_capabilities
    elif failure=='partial-cap':core.rank_decisions_capabilities=0x20000000
    elif failure=='float-cap':core.rank_decisions_capabilities=float(0x60000000)
    elif failure=='bool-cap':core.rank_decisions_capabilities=True
    elif failure=='subclass-cap':
        class Capability(int):pass
        core.rank_decisions_capabilities=Capability(0x60000000)
    elif failure=='python-entry':core.rank_decisions_run=lambda:None
    elif failure=='borrowed-entry':core.rank_decisions_run=len
    elif failure=='entry-name':core.rank_decisions_run.__name__='borrowed'
    elif failure=='entry-owner':core.__name__='another_core'
    elif failure=='spoof-builtin':
        Fake=type('builtin_function_or_method',(),{'__module__':'builtins','__name__':'rank_decisions_run'})
        core.rank_decisions_run=Fake()
    elif failure=='missing-path':registered.pop()
    elif failure=='duplicate-path':registered.append(registered[0])
    elif failure=='extra-path':registered.append('filters.rank.bilateral.otsu')
    elif failure=='core':wrapper._core=object()
    elif failure=='worker':core.get_num_threads=lambda:4
    elif failure=='identity':generic.otsu=lambda:None
    elif failure=='lazy-attribute':del generic.majority
    elif failure=='forwarding':rank.otsu.__module__='skimage.filters.rank.generic'
    elif failure=='name':rank.otsu.__name__='another_algorithm'
    elif failure=='canonical-star':rank.__all__.remove('majority')
    elif failure=='duplicate-star':rank.__all__.append('otsu')
    elif failure=='generic-otsu':generic.__all__.remove('otsu')
    else:generic.__all__.append('majority')
    (tmp_path/'state.json').write_text(json.dumps({'source':{}}))
    monkeypatch.setattr(q,'source_state',lambda _: {})
    monkeypatch.setattr(q,'admission',lambda *_:q.decisions_admission(*args))
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name=='pytest' or 'decisions_contract' in name:raise AssertionError('fixture imported before R7 metadata rejection')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(q.QualificationError):q.child(tmp_path,'public-decisions')


def valid_decisions_report():
    source=json.loads((q.ROOT/'coverage/contracts/rank-decisions-benchmark.json').read_text())
    reference={'synthetic':'metadata equality only'}
    def row():return dict(status='passed',timed=False,reference=reference,actual_by_workers={'1':dict(reference),'4':dict(reference)})
    checks=[dict(row(),case_index=i,timing_eligible=item['timed_candidate'],**item) for i,item in enumerate(source['case_inventory'])]
    return dict(status='complete-parity-check',checks=checks,failures=[],results=[],
        alias_checks=[dict(row(),canonical='filters.rank.'+n,alias='filters.rank.generic.'+n,reference_identity=True,native_identity=True) for n in q.R7],
        provider_checks=[dict(row(),api='filters.rank.'+n,case=f'{d}/{stage}') for n in q.R7 for d in (2,3) for stage in ('success','footprint','shift_x','shift_y','shift_z','bad-out','bad-image','image-array-only')],
        policy_checks=[dict(row(),api='filters.rank.'+n,policy=policy) for n in q.R7 for policy in ('warn','raise','ignore','call','log','print')],
        metadata=dict(check_only=True,reference_only=False,requested_workers=[1,4],validation_scope='strict actual public R7',
            algorithm_count=2,scalar_api_count=2,vector_api_count=0,aliases_timed=False,fallback_enabled=False,
            initial_default_workers=1,canonical_paths=['filters.rank.'+n for n in q.R7],public_alias_paths=['filters.rank.generic.'+n for n in q.R7],operation_ids={'otsu':29,'majority':30},case_contract_sha256=source['case_contract_sha256']))


def test_decisions_complete_synthetic_report_has_694_and_auxiliary46(tmp_path):
    report=valid_decisions_report();assert len(report['checks'])==694
    assert sum(r['timing_class']=='source-metadata-layout' for r in report['checks'])==16
    p=tmp_path/'r7.json';p.write_text(json.dumps(report));q.validate_corpus(p,'decisions')


@pytest.mark.parametrize('failure',['missing-case','duplicate-case','wrong-spec','metadata-timed','metadata-dropped',
    'provider-count','policy-count','alias-count','wrong-alias','duplicate-alias','private-alias','aux-worker',
    'case-observation','provider-observation','policy-observation','alias-observation','scope','fallback','default-worker',
    'bool-worker','algorithms','canonical','opcodes','case-hash','reference-only','timed','duplicate-provider','duplicate-policy','aux-timed','bool-default','typed-observation'])
def test_decisions_report_cannot_hide_missing_rows_or_substitute_observations(tmp_path,failure):
    r=valid_decisions_report()
    if failure=='missing-case':r['checks'].pop()
    elif failure=='duplicate-case':r['checks'][1]=r['checks'][0]
    elif failure=='wrong-spec':r['checks'][0]['fixture']={}
    elif failure=='metadata-timed':next(v for v in r['checks'] if v['timing_class']=='source-metadata-layout')['timing_eligible']=True
    elif failure=='metadata-dropped':next(v for v in r['checks'] if v['timing_class']=='source-metadata-layout')['timing_class']='defined-nonalias-raster'
    elif failure=='provider-count':r['provider_checks'].pop()
    elif failure=='policy-count':r['policy_checks'].pop()
    elif failure=='alias-count':r['alias_checks'].pop()
    elif failure=='wrong-alias':r['alias_checks'][0]['alias']='filters.rank.bilateral.otsu'
    elif failure=='duplicate-alias':r['alias_checks'][1]=r['alias_checks'][0]
    elif failure=='private-alias':r['alias_checks'][0]['native_identity']=None
    elif failure=='aux-worker':r['provider_checks'][0]['actual_by_workers'].pop('4')
    elif failure in ('case-observation','provider-observation','policy-observation','alias-observation'):
        key={'case-observation':'checks','provider-observation':'provider_checks','policy-observation':'policy_checks','alias-observation':'alias_checks'}[failure]
        r[key][0]['actual_by_workers']['4']={'wrong':'observation'}
    elif failure=='scope':r['metadata']['validation_scope']='private source wrapper'
    elif failure=='fallback':r['metadata']['fallback_enabled']=True
    elif failure=='default-worker':r['metadata']['initial_default_workers']=4
    elif failure=='bool-worker':r['metadata']['requested_workers']=[True,4]
    elif failure=='algorithms':r['metadata']['algorithm_count']=4
    elif failure=='canonical':r['metadata']['canonical_paths'].reverse()
    elif failure=='opcodes':r['metadata']['operation_ids']['majority']=29
    elif failure=='case-hash':r['metadata']['case_contract_sha256']='0'*64
    elif failure=='reference-only':r['metadata']['reference_only']=True
    elif failure=='duplicate-provider':r['provider_checks'][1]=r['provider_checks'][0]
    elif failure=='duplicate-policy':r['policy_checks'][1]=r['policy_checks'][0]
    elif failure=='aux-timed':r['provider_checks'][0]['timed']=True
    elif failure=='bool-default':r['metadata']['initial_default_workers']=True
    elif failure=='typed-observation':
        r['checks'][0]['reference']={'flag':True};r['checks'][0]['actual_by_workers']={'1':{'flag':True},'4':{'flag':1}}
    else:r['results']=[{'timed':'forbidden'}]
    p=tmp_path/'r7.json';p.write_text(json.dumps(r))
    with pytest.raises(q.QualificationError):q.validate_corpus(p,'decisions')


def synthetic_complete_plan(root):
    (root/'tests/cmake_consumer').mkdir(parents=True)
    (root/'tests/cmake_consumer/CMakeLists.txt').write_text('\n'.join('add_executable('+f+'_consumer x.cpp)' for f in q.FAMILIES))
    for group in q.GROUPS.values():
        for name in group:(root/'tests'/name.split('::')[0]).touch()
    (root/'tests/test_arraymap_dense_contracts.py').touch()
    for name in q.R6_SOURCE_PATHS+q.R7_SOURCE_PATHS+q.TV_SOURCE_PATHS+q.BREGMAN_SOURCE_PATHS:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    for family in q.FAMILIES:(root/'tests'/('test_'+family+'_native.cpp')).touch()
    (root/'benchmarks').mkdir()
    for name,_ in q.CORPORA.values():(root/'benchmarks'/name).touch()


@pytest.mark.parametrize('missing',q.R7_SOURCE_PATHS)
def test_decisions_source_migration_and_public_paths_are_plan_blockers(tmp_path,monkeypatch,missing):
    synthetic_complete_plan(tmp_path)
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH}),image_count=190,registered_count=193,metadata_count=3))
    assert not q.plan(tmp_path,'modern',190)['blockers']
    (tmp_path/missing).unlink()
    assert 'missing required R7 source: '+missing in q.plan(tmp_path,'modern',190)['blockers']


@pytest.mark.parametrize('failure',['missing','duplicate','wrong','source'])
def test_sdk14_requires_exact_clients_and_existing_sources(tmp_path,monkeypatch,failure):
    synthetic_complete_plan(tmp_path)
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH}),image_count=190))
    path=tmp_path/'tests/cmake_consumer/CMakeLists.txt';text=path.read_text()
    if failure=='missing':text=text.replace('add_executable(rank_decisions_consumer x.cpp)','')
    elif failure=='duplicate':text+='\nadd_executable(rank_decisions_consumer x.cpp)'
    elif failure=='wrong':text+='\nadd_executable(wrong_family_consumer x.cpp)'
    else:(tmp_path/'tests/test_rank_decisions_native.cpp').unlink()
    path.write_text(text);blockers=q.plan(tmp_path,'modern')['blockers']
    assert any({'missing':'missing required installed SDK client: rank_decisions','duplicate':'duplicate installed SDK client: rank_decisions','wrong':'unknown installed SDK client: wrong_family','source':'missing required SDK client source: rank_decisions'}[failure]==v for v in blockers)


def test_decisions_group_is_public_and_sdk14_keeps_old13():
    assert q.GROUPS['public-decisions']==('test_rank_decisions_contracts.py','test_rank_decisions_public.py','test_rank_decisions_endian_public.py')
    assert not any('pipeline' in n or 'supplement.py' in n for n in q.GROUPS['public-decisions'])
    assert q.CORPORA['decisions']==('rank_decisions.py',694)
    assert len(q.ORIGINAL_FAMILIES)==14 and q.ORIGINAL_FAMILIES[-1]=='rank_decisions' and q.ORIGINAL_FAMILIES[-2]=='rank_bilateral'
    listing={'tests':[{'name':'installed_'+f+'_consumer'} for f in q.FAMILIES]};q.validate_sdk_listing(listing)
    listing['tests'][-1]={'name':listing['tests'][0]['name']}
    with pytest.raises(q.QualificationError):q.validate_sdk_listing(listing)


def test_registered193_counts_are190_images_plus3_metadata(tmp_path):
    path=tmp_path/'python/skimagexx';path.mkdir(parents=True)
    (path/'coverage.py').write_text('def native_functions():return tuple("image"+str(i) for i in range(190))+tuple("metadata"+str(i) for i in range(3))\ndef execution_kind(name):return "native" if name.startswith("image") else "metadata-native"\n')
    r=q.registry_expectation(tmp_path)
    assert r['registered_count']==193 and r['image_count']==190 and r['metadata_count']==3


def test_source_duplicate_registration_rejected(tmp_path):
    path=tmp_path/'python/skimagexx';path.mkdir(parents=True)
    (path/'coverage.py').write_text('def native_functions():return ("image", "image")\ndef execution_kind(name):return "native"\n')
    with pytest.raises(q.QualificationError,match='duplicate source registry'):q.registry_expectation(tmp_path)

@pytest.mark.parametrize('selector',['SKIMAGEXX_RANK_DECISIONS_PROTOTYPE_CORE','SKIMAGEXX_RANK_DECISIONS_ENDIAN_PROTOTYPE_CORE'])
def test_private_r7_selectors_rejected_before_any_package_or_fixture_import(tmp_path,monkeypatch,selector):
    (tmp_path/'package.json').write_text(json.dumps({'members_sha256':{}}))
    monkeypatch.setattr(q,'verify_tree',lambda *_:None)
    monkeypatch.setattr(q,'validate_interpreter_provenance',lambda *_:None)
    monkeypatch.setenv(selector,'private_core')
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name in ('skimagexx','pytest'):raise AssertionError('package/fixture imported before selector rejection')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(q.QualificationError,match='private selectors forbidden'):q.admission(tmp_path,{'interpreter_provenance':{}})


def test_shared_global_worker_checks_remain_before_any_fixtures():
    import ast
    import inspect
    admission=ast.parse(inspect.getsource(q.admission))
    worker=next(n for n in ast.walk(admission) if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple) and ast.literal_eval(n.iter)==(4,1))
    text=ast.unparse(worker)
    assert 'xx.set_num_threads(worker)' in text and 'xx._core.get_num_threads() == worker' in text
    assert 'xx.set_num_threads is xx._core.set_num_threads' in inspect.getsource(q.admission)
    assert 'xx.get_num_threads is xx._core.get_num_threads' in inspect.getsource(q.admission)
    child=ast.parse(inspect.getsource(q.child));guard=next(n for n in ast.walk(child) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='admission')
    fixtures=next(n for n in ast.walk(child) if isinstance(n,ast.Import) and any(a.name=='pytest' for a in n.names))
    assert guard.lineno<fixtures.lineno


def test_build_identity_capture_is_after_product_before_admission_and_seal():
    import ast
    import inspect
    source=inspect.getsource(q.prepare);tree=ast.parse(source)
    build=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='must_run' and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='build-whole-product')
    capture=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='capture_build_identity')
    admission=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='must_run' and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='initial-public-admission')
    assert build.lineno<capture.lineno<admission.lineno
    assert '--verbose' in source
    assert 'CMAKE_MODULE_LINKER_FLAGS=-save-temps=obj -Wl,--dependency-file=' in source
    assert 'native-build/core-link.d' in source and 'CMAKE_EXPORT_COMPILE_COMMANDS=ON' in source
    assert 'build.tool-args=-d;keeprsp;-d;keepdepfile' in source
    assert 'validate_build_identity(out, state)' in inspect.getsource(q.seal)
    assert "'gather-build-identity-facts','capture-build-identity'" in inspect.getsource(q.seal)


def fake_identity_journal(out,profile,revision,failure=None):
    class MetadataJournal:
        rows=[]
        def run(self,label,command,env=None):
            self.rows.append((label,list(map(str,command)),env))
            if label==failure:return 1
            if label=='gather-build-identity-facts':
                (out/'build-identity-facts.json').write_text(json.dumps(dict(schema='rank-statistics-build-input-v2',revision=revision,profile=profile,source_root=str(q.ROOT),build_root=str(out))))
            else:(out/'build-identity.json').write_text(json.dumps(dict(schema='rank-statistics-build-identity-v2')))
            return 0
    return MetadataJournal()


@pytest.mark.parametrize('profile',['windows','modern','minimum','debug'])
def test_identity_capture_uses_journal_and_exact_source_root_profile(tmp_path,profile):
    journal=fake_identity_journal(tmp_path,profile,'a'*40);env={'NO_IMAGE_CALLS':'metadata-fixture-only'}
    record=q.capture_build_identity(tmp_path,profile,{'revision':'a'*40},journal,Path('buildpython'),env)
    assert [r[0] for r in journal.rows]==['gather-build-identity-facts','capture-build-identity']
    assert all(r[2] is env for r in journal.rows)
    argv=journal.rows[1][1];assert argv[2:]==['capture-build-identity','--facts',str(tmp_path/'build-identity-facts.json'),'--output',str(tmp_path/'build-identity.json')]
    q.validate_build_identity(tmp_path,{'build_identity':record})
    with pytest.raises(q.QualificationError,match='retrofit'):q.capture_build_identity(tmp_path,profile,{'revision':'a'*40},journal,Path('buildpython'),env)


@pytest.mark.parametrize('label',['gather-build-identity-facts','capture-build-identity'])
def test_failed_identity_child_stops_before_any_later_stage(tmp_path,label):
    journal=fake_identity_journal(tmp_path,'modern','a'*40,label)
    with pytest.raises(q.QualificationError,match='command failed'):q.capture_build_identity(tmp_path,'modern',{'revision':'a'*40},journal,'python',{})
    assert journal.rows[-1][0]==label and not (tmp_path/'manifest.json').exists()


@pytest.mark.parametrize('change',['missing','oldschema','facts','identity'])
def test_identity_absent_old_or_changed_cannot_be_sealed(tmp_path,change):
    journal=fake_identity_journal(tmp_path,'modern','a'*40)
    record=q.capture_build_identity(tmp_path,'modern',{'revision':'a'*40},journal,'python',{})
    if change=='missing':state={}
    else:
        state={'build_identity':record}
        if change=='oldschema':record['schema']='rank-statistics-build-identity-v1'
        else:(tmp_path/('build-identity-facts.json' if change=='facts' else 'build-identity.json')).write_text('{}')
    with pytest.raises(q.QualificationError):q.validate_build_identity(tmp_path,state)


@pytest.mark.parametrize('key,value',[('revision','b'*40),('profile','minimum'),('source_root','/wrong'),('build_root','/wrong'),('schema','rank-statistics-build-input-v1')])
def test_wrong_identity_facts_reject_before_producer(tmp_path,key,value):
    journal=fake_identity_journal(tmp_path,'modern','a'*40);run=journal.run
    def modified(label,command,env=None):
        result=run(label,command,env)
        if label=='gather-build-identity-facts':
            path=tmp_path/'build-identity-facts.json';d=json.loads(path.read_text());d[key]=value;path.write_text(json.dumps(d))
        return result
    journal.run=modified
    with pytest.raises(q.QualificationError,match='source/profile/roots differ'):q.capture_build_identity(tmp_path,'modern',{'revision':'a'*40},journal,'python',{})
    assert [r[0] for r in journal.rows]==['gather-build-identity-facts']


def test_counts_after_group_is_conditional_and_keeps_every_original_group(tmp_path):
    before=q.groups_for_source(tmp_path);assert before==q._BASE_GROUPS and 'public-counts' not in before
    path=tmp_path/'tests/test_rank_windowed_counts_pipeline.py';path.parent.mkdir();path.write_text('# metadata fixture, never collected')
    after=q.groups_for_source(tmp_path)
    assert after['public-counts']==('test_rank_windowed_counts_pipeline.py',)
    assert {k:v for k,v in after.items() if k!='public-counts'}==before
    assert before==q._BASE_GROUPS  # no mutation of the before configuration


def test_counts_directory_does_not_masquerade_as_a_test_file(tmp_path):
    (tmp_path/'tests/test_rank_windowed_counts_pipeline.py').mkdir(parents=True)
    assert 'public-counts' not in q.groups_for_source(tmp_path)


def test_plan_selects_full_public_counts_only_in_the_source_that_has_it(tmp_path,monkeypatch):
    synthetic_complete_plan(tmp_path)
    # The fixture builder also follows an after checkout's available selectors.
    # Explicitly create the absent-file before variant in either checkout.
    counts_source=tmp_path/'tests/test_rank_windowed_counts_pipeline.py'
    if counts_source.exists():counts_source.unlink()
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH}),image_count=190,registered_count=193,metadata_count=3))
    before=q.plan(tmp_path,'modern',190);assert not before['blockers'] and 'public-counts' not in before['public_groups']
    (tmp_path/'tests/test_rank_windowed_counts_pipeline.py').touch()
    after=q.plan(tmp_path,'modern',190);assert not after['blockers']
    assert after['public_groups']['public-counts']==('test_rank_windowed_counts_pipeline.py',)
    assert {k:v for k,v in after['public_groups'].items() if k!='public-counts'}==before['public_groups']
    assert after['corpora']==before['corpora'] and after['SDK_families']==before['SDK_families']


def test_counts_private_selector_rejected_before_package_or_fixture_import(tmp_path,monkeypatch):
    (tmp_path/'package.json').write_text(json.dumps({'members_sha256':{}}))
    monkeypatch.setattr(q,'verify_tree',lambda *_:None);monkeypatch.setattr(q,'validate_interpreter_provenance',lambda *_:None)
    monkeypatch.setenv('SKIMAGEXX_WINDOWED_COUNTS_CORE','historical_private_core')
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name in ('skimagexx','pytest'):raise AssertionError('package/fixture imported before private selector rejection')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(q.QualificationError,match='private selectors forbidden'):q.admission(tmp_path,{'interpreter_provenance':{}})



def test_tv_public_group_and_sdk17_keep_exact_original14():
    assert q.GROUPS['public-TV']==('test_tv_chambolle_contracts.py','test_tv_chambolle_native_pipeline.py',
                                 'test_tv_chambolle_cursor_pipeline.py')
    assert len(q.TV_FAMILIES)==17 and q.TV_FAMILIES[:14]==q.ORIGINAL_FAMILIES
    assert len(q.FAMILIES)==18 and q.FAMILIES[:17]==q.TV_FAMILIES
    assert q.TV_FAMILIES[-3:]==('tv_chambolle','tv_reductions','tv_denoise')
    assert q.FAMILIES[-1]=='tv_bregman'
    old={'tests':[{'name':'installed_'+f+'_consumer'} for f in q.ORIGINAL_FAMILIES]}
    q.validate_sdk_listing(old,scope='rank14')
    with pytest.raises(q.QualificationError,match='SDK test discovery'):
        q.validate_sdk_listing(old)
    for family in q.FAMILIES:
        listing={'tests':[{'name':'installed_'+f+'_consumer'} for f in q.FAMILIES if f!=family]}
        with pytest.raises(q.QualificationError,match='SDK test discovery'):q.validate_sdk_listing(listing)


@pytest.mark.parametrize('missing',q.TV_SOURCE_PATHS)
def test_tv_source_and_registration_block_before_any_build(tmp_path,monkeypatch,missing):
    synthetic_complete_plan(tmp_path)
    names=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH})
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=names,image_count=191))
    assert not q.plan(tmp_path,'modern',191)['blockers']
    (tmp_path/missing).unlink()
    assert 'missing required TV source: '+missing in q.plan(tmp_path,'modern',191)['blockers']
    names.remove(q.TV_PATH)
    assert 'missing public TV registry: '+q.TV_PATH in q.plan(tmp_path,'modern')['blockers']


def synthetic_tv_admission(monkeypatch):
    # Explicit inert metadata stand-ins, never invoked and never native proof.
    core=types.SimpleNamespace(__name__='skimagexx._core',get_num_threads=lambda:1,
        tv_chambolle_interface='tv-chambolle-stages-v1',tv_chambolle_capabilities=63,
        tv_chambolle_finite_interface='tv-chambolle-finite-v1',tv_chambolle_finite_capabilities=3,
        tv_chambolle_finite_profiles=('1.24.4','2.5.4'))
    for name in ('tv_chambolle_divergence_into','tv_chambolle_gradient_into','tv_chambolle_norm_into',
                 'tv_chambolle_dual_into','tv_chambolle_finite_run'):
        def inert(*_,**__):raise AssertionError('metadata stub must never execute')
        inert.__name__=name;inert.__module__=core.__name__;inert.synthetic_builtin=True
        setattr(core,name,inert)
    monkeypatch.setattr(q.inspect,'isbuiltin',lambda value:getattr(value,'synthetic_builtin',False))
    def denoise_tv_chambolle(image,weight=0.1,eps=0.0002,max_num_iter=200,*,channel_axis=None):
        raise AssertionError('metadata stub must never execute')
    denoise_tv_chambolle.__module__='skimagexx._tv_chambolle'
    wrapper=types.SimpleNamespace(_core=core,denoise_tv_chambolle=denoise_tv_chambolle)
    restoration=types.SimpleNamespace(denoise_tv_chambolle=denoise_tv_chambolle)
    return core,restoration,wrapper,[q.TV_PATH]


def test_tv_preflight_metadata_does_not_call_any_image_operation(monkeypatch):
    q.tv_admission(*synthetic_tv_admission(monkeypatch))


@pytest.mark.parametrize('failure',['missing-registry','duplicate-registry','stage-only','boolean-capability',
    'wrong-profile','private-core','wrong-export','forwarding','python-entry','entry-owner','workers','signature'])
def test_tv_missing_or_forwarded_native_implementation_is_never_admitted(monkeypatch,failure):
    core,restoration,wrapper,registered=synthetic_tv_admission(monkeypatch)
    if failure=='missing-registry':registered=[]
    elif failure=='duplicate-registry':registered*=2
    elif failure=='stage-only':del core.tv_chambolle_finite_interface
    elif failure=='boolean-capability':core.tv_chambolle_finite_capabilities=True
    elif failure=='wrong-profile':core.tv_chambolle_finite_profiles=('2.5.4',)
    elif failure=='private-core':wrapper._core=types.SimpleNamespace()
    elif failure=='wrong-export':restoration.denoise_tv_chambolle=lambda x:x
    elif failure=='forwarding':wrapper.denoise_tv_chambolle.__module__='skimage.restoration'
    elif failure=='python-entry':core.tv_chambolle_finite_run.synthetic_builtin=False
    elif failure=='entry-owner':core.tv_chambolle_finite_run.__module__='private_tv_core'
    elif failure=='workers':core.get_num_threads=lambda:4
    else:
        def denoise_tv_chambolle(image):raise AssertionError('metadata only')
        denoise_tv_chambolle.__module__='skimagexx._tv_chambolle'
        wrapper.denoise_tv_chambolle=restoration.denoise_tv_chambolle=denoise_tv_chambolle
    with pytest.raises(q.QualificationError):q.tv_admission(core,restoration,wrapper,registered)


@pytest.mark.parametrize('change',['empty','missing','duplicate','substitute'])
def test_original_rank14_sdk_negative_listings_remain_rejected(change):
    listing={'tests':[{'name':'installed_'+f+'_consumer'} for f in q.ORIGINAL_FAMILIES]}
    if change=='empty':listing['tests']=[]
    elif change=='missing':listing['tests'].pop()
    elif change=='duplicate':listing['tests'].append(dict(listing['tests'][0]))
    else:listing['tests'][-1]['name']='installed_private_adapter_consumer'
    with pytest.raises(q.QualificationError,match='SDK test discovery'):
        q.validate_sdk_listing(listing,scope='rank14')


@pytest.fixture
def complete_bregman_fake_artifact(complete_tv_fake_artifact,tmp_path):
    """Complete110/63 inert census; no executable native bytes."""
    product,source,revision=complete_tv_fake_artifact
    additions={'include/skimagexx/tv_bregman.hpp':b'// synthetic Bregman header\n',
               'skimagexx/_tv_bregman.py':b'# synthetic Bregman helper\n'}
    with zipfile.ZipFile(product) as old:
        entries={n:old.read(n) for n in old.namelist() if not n.endswith('/RECORD')}
    entries.update(additions);new_product=tmp_path/'bregman-synthetic.whl';wheel(new_product,entries)
    new_source=tmp_path/'bregman-source.zip'
    with zipfile.ZipFile(source) as old,zipfile.ZipFile(new_source,'w') as archive:
        archive.comment=old.comment
        for name in old.namelist():archive.writestr(name,old.read(name))
        for name,value in additions.items():archive.writestr('python/'+name if name.startswith('skimagexx/') else name,value)
    return new_product,new_source,revision


def test_bregman110_63_is_exact_default_with_explicit_historical_scopes(complete_bregman_fake_artifact,complete_tv_fake_artifact,complete_fake_artifact):
    result=q.inspect_wheel(*complete_bregman_fake_artifact)
    assert result['product_scope']=='bregman18' and result['record_verified']
    assert len(result['members_sha256'])==110 and len(result['SDK_sha256'])==63
    assert q.product_census('rank14')==(104,59) and q.product_census('tv17')==(108,62)
    assert q.product_census()==(110,63)
    for artifact in (complete_tv_fake_artifact,complete_fake_artifact):
        with pytest.raises(q.QualificationError,match='complete110'):q.inspect_wheel(*artifact)
    assert q.inspect_wheel(*complete_tv_fake_artifact,scope='tv17')['product_scope']=='tv17'
    for scope in ('rank14','tv17'):
        with pytest.raises(q.QualificationError,match='complete'):q.inspect_wheel(*complete_bregman_fake_artifact,scope=scope)


@pytest.mark.parametrize('member',('include/skimagexx/tv_bregman.hpp','python/skimagexx/_tv_bregman.py'))
def test_complete110_wheel_reaches_bregman_source_guard(complete_bregman_fake_artifact,tmp_path,monkeypatch,member):
    product,source,revision=complete_bregman_fake_artifact;replacement=tmp_path/'missing-bregman.zip'
    with zipfile.ZipFile(source) as old,zipfile.ZipFile(replacement,'w') as archive:
        archive.comment=old.comment
        for name in old.namelist():
            if name!=member:archive.writestr(name,old.read(name))
    calls=[];original=q.wheel_members
    def witnessed(path):calls.append(str(path));return original(path)
    monkeypatch.setattr(q,'wheel_members',witnessed)
    with pytest.raises(q.QualificationError,match='Bregman product scope/source'):
        q.inspect_wheel(product,replacement,revision)
    assert calls==[str(product)],'complete110 metadata/RECORD must reach the new source guard'


@pytest.mark.parametrize('missing',q.BREGMAN_SOURCE_PATHS)
def test_bregman_sources_and_registry_block_before_build(tmp_path,monkeypatch,missing):
    synthetic_complete_plan(tmp_path)
    names=['filters.rank.'+name for name in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH})
    monkeypatch.setattr(q,'registry_expectation',lambda root:dict(functions=names,image_count=192,metadata_count=3,registered_count=195))
    assert not q.plan(tmp_path,'modern',192)['blockers']
    (tmp_path/missing).unlink()
    assert 'missing required Bregman source: '+missing in q.plan(tmp_path,'modern',192)['blockers']
    names.remove(q.BREGMAN_PATH)
    assert 'missing public Bregman registry: '+q.BREGMAN_PATH in q.plan(tmp_path,'modern')['blockers']


@pytest.mark.parametrize('scope,families',(('rank14',q.ORIGINAL_FAMILIES),('tv17',q.TV_FAMILIES),('bregman18',q.FAMILIES)))
def test_each_exact_sdk_census_keeps_missing_duplicate_substitute_guards(scope,families):
    assert len(families)=={'rank14':14,'tv17':17,'bregman18':18}[scope]
    listing={'tests':[{'name':'installed_'+name+'_consumer'} for name in families]}
    q.validate_sdk_listing(listing,scope=scope)
    for change in ('missing','duplicate','substitute'):
        altered=json.loads(json.dumps(listing))
        if change=='missing':altered['tests'].pop()
        elif change=='duplicate':altered['tests'].append(dict(altered['tests'][0]))
        else:altered['tests'][-1]['name']='installed_private_consumer'
        with pytest.raises(q.QualificationError,match='SDK test discovery'):q.validate_sdk_listing(altered,scope=scope)
    assert q.GROUPS['public-Bregman']==('test_tv_bregman_contracts.py','test_tv_bregman_public.py')
    assert not any('native_pipeline' in name for name in q.GROUPS['public-Bregman'])


def synthetic_bregman_admission(monkeypatch):
    """Admission routing metadata only; each callable raises if executed."""
    core=types.SimpleNamespace(__name__='skimagexx._core',get_num_threads=lambda:1,
        tv_bregman_interface='tv-bregman-prepared-v1',tv_bregman_capabilities=3,tv_bregman_parallel_sweep=False)
    def entry(*args,**kwargs):raise AssertionError('metadata-only entry must not execute')
    entry.__name__='tv_bregman_run';entry.__module__=core.__name__;entry.synthetic_builtin=True
    core.tv_bregman_run=entry
    monkeypatch.setattr(q.inspect,'isbuiltin',lambda value:getattr(value,'synthetic_builtin',False))
    def denoise_tv_bregman(image,weight=5.0,max_num_iter=100,eps=0.001,isotropic=True,*,channel_axis=None):
        raise AssertionError('metadata-only wrapper must not execute')
    denoise_tv_bregman.__module__='skimagexx._tv_bregman'
    wrapper=types.SimpleNamespace(_core=core,denoise_tv_bregman=denoise_tv_bregman)
    restoration=types.SimpleNamespace(denoise_tv_bregman=denoise_tv_bregman)
    return core,restoration,wrapper,[q.BREGMAN_PATH]


def test_bregman_admission_positive_metadata_never_calls_entry(monkeypatch):
    q.bregman_admission(*synthetic_bregman_admission(monkeypatch))


@pytest.mark.parametrize('fault',('missing-registry','duplicate-registry','interface','float-cap','bool-cap',
    'parallel-true','parallel-int','missing-entry','entry-name','entry-owner','python-entry','wrong-core',
    'wrong-export','forwarding','wrong-name','signature','default-worker4'))
def test_bregman_admission_rejects_invalid_metadata_before_provider(monkeypatch,fault):
    core,restoration,wrapper,registered=synthetic_bregman_admission(monkeypatch)
    if fault=='missing-registry':registered=[]
    elif fault=='duplicate-registry':registered*=2
    elif fault=='interface':core.tv_bregman_interface='stage-only'
    elif fault=='float-cap':core.tv_bregman_capabilities=3.
    elif fault=='bool-cap':core.tv_bregman_capabilities=True
    elif fault=='parallel-true':core.tv_bregman_parallel_sweep=True
    elif fault=='parallel-int':core.tv_bregman_parallel_sweep=0
    elif fault=='missing-entry':del core.tv_bregman_run
    elif fault=='entry-name':core.tv_bregman_run.__name__='borrowed_run'
    elif fault=='entry-owner':core.tv_bregman_run.__module__='private_core'
    elif fault=='python-entry':core.tv_bregman_run.synthetic_builtin=False
    elif fault=='wrong-core':wrapper._core=object()
    elif fault=='wrong-export':restoration.denoise_tv_bregman=lambda image:image
    elif fault=='forwarding':wrapper.denoise_tv_bregman.__module__='skimage.restoration'
    elif fault=='wrong-name':wrapper.denoise_tv_bregman.__name__='borrowed_name'
    elif fault=='signature':wrapper.denoise_tv_bregman.__signature__=__import__('inspect').Signature()
    else:core.get_num_threads=lambda:4
    with pytest.raises(q.QualificationError):q.bregman_admission(core,restoration,wrapper,registered)


def test_cursor_native_target_and_installed_pipeline_keep_sdk18_scope():
    cmake=(ROOT/'CMakeLists.txt').read_text()
    assert 'add_executable(test_tv_denoise_cursor_native tests/test_tv_denoise_cursor_native.cpp)' in cmake
    assert 'target_link_libraries(test_tv_denoise_cursor_native PRIVATE skimagexx Threads::Threads)' in cmake
    assert 'add_test(NAME tv_denoise_cursor_native COMMAND test_tv_denoise_cursor_native)' in cmake
    native=(ROOT/'tests/test_tv_denoise_cursor_native.cpp').read_text(encoding='utf-8-sig')
    assert '#include "skimagexx/tv_denoise.hpp"' in native and '#include "skimagexx/image.hpp"' in native
    assert '../src/' not in native and 'execution.hpp' not in native
    assert q.PRODUCT_MEMBERS==110 and q.SDK_MEMBERS==63 and len(q.FAMILIES)==18
    assert 'tv_denoise_cursor' not in q.FAMILIES
    assert {'tests/test_tv_denoise_cursor_native.cpp','tests/test_tv_chambolle_cursor_pipeline.py'}<=set(q.TV_SOURCE_PATHS)
    pipeline=(ROOT/'tests/test_tv_chambolle_cursor_pipeline.py').read_text()
    assert 'from skimagexx import _core as core' in pipeline
    assert "BACKEND=os.environ.get('SKIMAGEXX_TV_CHAMBOLLE_STAGE_CORE')" in pipeline


def test_cursor_private_selector_rejected_before_installed_pipeline_import(tmp_path,monkeypatch):
    # Inert package-envelope fixture exercises the real existing admission
    # boundary; no numerical stub or native acceptance is claimed.
    (tmp_path/'package.json').write_text(json.dumps({'members_sha256':{}}))
    monkeypatch.setattr(q,'verify_tree',lambda *_:None)
    monkeypatch.setattr(q,'validate_interpreter_provenance',lambda *_:None)
    monkeypatch.setenv('SKIMAGEXX_TV_CHAMBOLLE_STAGE_CORE','private_cursor_core')
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name in ('skimagexx','pytest'):raise AssertionError('installed package/fixtures reached before rejection')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(q.QualificationError,match='private selectors forbidden'):
        q.admission(tmp_path,{'interpreter_provenance':{}})
