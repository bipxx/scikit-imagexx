"""Tooling-only guards. Synthetic artifacts/modules never prove public parity."""
import base64
import ast
import builtins
import csv
import hashlib
import importlib.util
import inspect
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


@pytest.fixture
def complete_nmi_fake_artifact(complete_reconstruction_fake_artifact, tmp_path):
    """Inert file-envelope metadata; never a compiled or accepted product."""
    product, source, revision = complete_reconstruction_fake_artifact
    additions = {
        'include/skimagexx/normalized_mutual_information.hpp': b'// inert NMI header\n',
        'skimagexx/_normalized_mutual_information.py': b'# inert NMI helper\n'}
    with zipfile.ZipFile(product) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()
                   if not name.endswith('/RECORD')}
    entries.update(additions)
    target = tmp_path / 'nmi-synthetic.whl'
    wheel(target, entries)
    source_target = tmp_path / 'nmi-source.zip'
    with zipfile.ZipFile(source) as archive, zipfile.ZipFile(source_target, 'w') as output:
        output.comment = archive.comment
        for name in archive.namelist():
            output.writestr(name, archive.read(name))
        for name, raw in additions.items():
            output.writestr('python/' + name if name.startswith('skimagexx/') else name, raw)
    return target, source_target, revision


def test_nmi118_67_current_scope_preserves_historical116_66(complete_nmi_fake_artifact,
                                                       complete_reconstruction_fake_artifact):
    result = q.inspect_wheel(*complete_nmi_fake_artifact, scope='nmi22')
    assert result['product_scope'] == 'nmi22'
    assert len(result['members_sha256']) == 118 and len(result['SDK_sha256']) == 67
    assert q.product_census('nmi22') == (118, 67)
    assert q.product_census('reconstruction21') == (116, 66)
    assert len(q.RECONSTRUCTION_FAMILIES) == 21 and q.RECONSTRUCTION_FAMILIES[-1] == 'reconstruction'
    assert len(q.NMI_FAMILIES) == 22 and q.NMI_FAMILIES[:-1] == q.RECONSTRUCTION_FAMILIES
    assert q.NMI_FAMILIES[-1] == 'normalized_mutual_information'
    assert q.inspect_wheel(*complete_reconstruction_fake_artifact, scope='reconstruction21')['record_verified']
    with pytest.raises(q.QualificationError, match='complete118'):
        q.inspect_wheel(*complete_reconstruction_fake_artifact, scope='nmi22')
    with pytest.raises(q.QualificationError, match='complete116'):
        q.inspect_wheel(*complete_nmi_fake_artifact, scope='reconstruction21')


@pytest.fixture
def complete_h203_fake_artifact(complete_nmi_fake_artifact,tmp_path):
    product,source,revision=complete_nmi_fake_artifact
    additions={'include/skimagexx/h_extrema.hpp':b'// inert H header\n',
               'skimagexx/_h_extrema.py':b'# inert H helper\n'}
    with zipfile.ZipFile(product) as archive:
        entries={n:archive.read(n) for n in archive.namelist() if not n.endswith('/RECORD')}
    entries.update(additions);target=tmp_path/'h203-synthetic.whl';wheel(target,entries)
    src=tmp_path/'h203-source.zip'
    with zipfile.ZipFile(source) as archive,zipfile.ZipFile(src,'w') as output:
        output.comment=archive.comment
        for name in archive.namelist():output.writestr(name,archive.read(name))
        for name,raw in additions.items():output.writestr('python/'+name if name.startswith('skimagexx/') else name,raw)
    return target,src,revision


def test_h203_complete120_68_retains_nmi_historical_scope(complete_h203_fake_artifact,complete_nmi_fake_artifact):
    result=q.inspect_wheel(*complete_h203_fake_artifact)
    assert result['product_scope']=='h_extrema23'
    assert len(result['members_sha256'])==120 and len(result['SDK_sha256'])==68
    assert q.inspect_wheel(*complete_nmi_fake_artifact,scope='nmi22')['record_verified']
    with pytest.raises(q.QualificationError,match='complete120'):q.inspect_wheel(*complete_nmi_fake_artifact)
    with pytest.raises(q.QualificationError,match='complete118'):q.inspect_wheel(*complete_h203_fake_artifact,scope='nmi22')


@pytest.mark.parametrize('missing',('include/skimagexx/h_extrema.hpp','python/skimagexx/_h_extrema.py',
                                   'include/skimagexx/normalized_mutual_information.hpp','python/skimagexx/_normalized_mutual_information.py'))
def test_h203_requires_both_actual_family_source_members(complete_h203_fake_artifact,tmp_path,missing):
    product,source,revision=complete_h203_fake_artifact
    changed=tmp_path/'h203-missing-source.zip'
    with zipfile.ZipFile(source) as archive,zipfile.ZipFile(changed,'w') as output:
        output.comment=archive.comment
        for name in archive.namelist():
            if name!=missing:output.writestr(name,archive.read(name))
    with pytest.raises(q.QualificationError,match='H-extrema product|NMI product'):q.inspect_wheel(product,changed,revision)


@pytest.mark.parametrize('missing', ('include/skimagexx/normalized_mutual_information.hpp',
                                   'python/skimagexx/_normalized_mutual_information.py'))
def test_nmi_product_requires_actual_source_header_and_helper(complete_nmi_fake_artifact,
                                                            tmp_path, missing):
    product, source, revision = complete_nmi_fake_artifact
    changed = tmp_path / 'missing-nmi-source.zip'
    with zipfile.ZipFile(source) as archive, zipfile.ZipFile(changed, 'w') as output:
        output.comment = archive.comment
        for name in archive.namelist():
            if name != missing:
                output.writestr(name, archive.read(name))
    with pytest.raises(q.QualificationError, match='NMI product scope/source'):
        q.inspect_wheel(product, changed, revision, scope='nmi22')


@pytest.mark.parametrize('missing', q.NMI_SOURCE_PATHS)
def test_nmi_plan_requires_exact_sources_and_public_gate(tmp_path, monkeypatch, missing):
    synthetic_complete_plan(tmp_path)
    names = ['filters.rank.' + name for name in q.R5] + sorted(
        q.R6_PATHS | q.R7_PATHS | {q.TV_PATH, q.BREGMAN_PATH, q.MATRIX_PATH,
                                  q.RECONSTRUCTION_PATH, q.NMI_PATH} | set(q.MAX_TREE_PATHS) | set(q.H_EXTREMA_PATHS))
    monkeypatch.setattr(q, 'registry_expectation', lambda _: dict(
        functions=names, image_count=201, metadata_count=3, registered_count=204))
    assert not q.plan(tmp_path, 'modern', 201)['blockers']
    (tmp_path / missing).unlink()
    assert 'missing required NMI source: ' + missing in q.plan(tmp_path, 'modern', 201)['blockers']


@pytest.mark.parametrize('path', q.NMI_IMPORTED_SOURCE_PINS)
def test_nmi_frozen_kernel_policy_and_observer_hash_changes_block(tmp_path, monkeypatch, path):
    synthetic_complete_plan(tmp_path)
    names = ['filters.rank.' + name for name in q.R5] + sorted(
        q.R6_PATHS | q.R7_PATHS | {q.TV_PATH, q.BREGMAN_PATH, q.MATRIX_PATH,
                                  q.RECONSTRUCTION_PATH, q.NMI_PATH} | set(q.MAX_TREE_PATHS) | set(q.H_EXTREMA_PATHS))
    monkeypatch.setattr(q, 'registry_expectation', lambda _: dict(functions=names, image_count=201))
    target = tmp_path / path
    target.write_bytes(target.read_bytes() + b'\nchanged')
    assert 'frozen NMI source pin differs: ' + path in q.plan(tmp_path, 'modern', 201)['blockers']


def nmi_metadata_stub(monkeypatch):
    """Every inert numerical callable raises if the metadata guard calls it."""
    core = types.SimpleNamespace(__name__='unit_sole_core', get_num_threads=lambda: 1,
                                 has_openmp=True)
    for name in ('nmi_prepare', 'nmi_statistics', 'nmi_validate_inputs'):
        def forbidden(*args, **kwargs):
            raise AssertionError('NMI metadata stub executed')
        forbidden.__name__ = name
        forbidden.__module__ = core.__name__
        forbidden.unit_builtin = True
        setattr(core, name, forbidden)
    def normalized_mutual_information(image0, image1, *, bins=100):
        raise AssertionError('NMI wrapper metadata stub executed')
    normalized_mutual_information.__module__ = 'skimagexx._normalized_mutual_information'
    wrapper = types.SimpleNamespace(_core=core, normalized_mutual_information=normalized_mutual_information)
    metrics = types.SimpleNamespace(normalized_mutual_information=normalized_mutual_information,
                                    __all__=['normalized_mutual_information'])
    actual = q.inspect.isbuiltin
    monkeypatch.setattr(q.inspect, 'isbuiltin', lambda f: actual(f) or getattr(f, 'unit_builtin', False))
    return core, metrics, wrapper, [q.NMI_PATH]


def test_nmi_metadata_admission_calls_no_image_or_provider(monkeypatch):
    q.nmi_admission(*nmi_metadata_stub(monkeypatch))


@pytest.mark.parametrize('fault', ('missing', 'duplicate', 'worker4', 'no-OpenMP', 'entry-missing',
                                 'entry-name', 'entry-owner', 'python-entry', 'borrowed-builtin',
                                 'wrong-core', 'wrong-alias', 'forwarding', 'signature', 'duplicate-export'))
def test_nmi_metadata_admission_rejects_wrong_owner_before_any_fixture(monkeypatch, fault):
    core, metrics, wrapper, paths = nmi_metadata_stub(monkeypatch)
    if fault == 'missing': paths = []
    elif fault == 'duplicate': paths *= 2
    elif fault == 'worker4': core.get_num_threads = lambda: 4
    elif fault == 'no-OpenMP': core.has_openmp = False
    elif fault == 'entry-missing': del core.nmi_prepare
    elif fault == 'entry-name': core.nmi_prepare.__name__ = 'borrowed'
    elif fault == 'entry-owner': core.nmi_prepare.__module__ = 'upstream'
    elif fault == 'python-entry': core.nmi_prepare.unit_builtin = False
    elif fault == 'borrowed-builtin': core.nmi_prepare = core.nmi_statistics
    elif fault == 'wrong-core': wrapper._core = object()
    elif fault == 'wrong-alias': metrics.normalized_mutual_information = lambda *a: None
    elif fault == 'forwarding': wrapper.normalized_mutual_information.__module__ = 'skimage.metrics'
    elif fault == 'signature': wrapper.normalized_mutual_information.__signature__ = inspect.Signature()
    else: metrics.__all__ *= 2
    with pytest.raises(q.QualificationError):
        q.nmi_admission(core, metrics, wrapper, paths)


def test_nmi_exact_private_imports_and_original_reference_lineage_are_preserved():
    def canonical(node):
        if isinstance(node, ast.AST):
            return [type(node).__name__, [[key, canonical(value)]
                for key, value in ast.iter_fields(node) if not (key == 'type_params' and not value)]]
        if isinstance(node, list):
            return [canonical(value) for value in node]
        if node is Ellipsis:
            return ['literal', 'Ellipsis']
        if isinstance(node, bytes):
            return ['bytes', node.hex()]
        if isinstance(node, complex):
            return ['complex', node.real.hex(), node.imag.hex()]
        return node
    record = json.loads((ROOT / 'coverage/evidence/nmi-public-integration.json').read_bytes())
    assert record['private_source'] == 'fadef4cf9994506017ab9d54f4e91917c77ea770'
    assert record['original_source'] == '60c0bd03756d0577eecb08d8e51019faa1e0ba6a'
    assert len(record['exact_imports']) == 8
    for row in record['exact_imports']:
        raw = (ROOT / row['path']).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(raw).hexdigest() == row['git_sha256'] == q.NMI_IMPORTED_SOURCE_PINS[row['path']]
    for selector, digest in record['unchanged_original_function_AST_sha256'].items():
        path, name = selector.split('::')
        tree = ast.parse((ROOT / path).read_bytes())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        raw = json.dumps(canonical(function), sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(raw).hexdigest() == digest
    contract = json.loads((ROOT / 'coverage/contracts/normalized-mutual-information.json').read_bytes())
    for field, digest in record['unchanged_original_contract_fields_sha256'].items():
        raw = json.dumps(contract[field], sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(raw).hexdigest() == digest
    assert contract['comparison_policy']['floating']['rtol'] == 2e-12
    assert contract['comparison_policy']['floating']['atol'] == 2e-13
    assert 'probability<=0.99' in contract['native_scope']['limits']


def test_nmi_direct_sdk_and_ordinary_core_wiring_keep_release_assertions_live():
    cmake = (ROOT / 'CMakeLists.txt').read_text()
    consumer = (ROOT / 'tests/cmake_consumer/CMakeLists.txt').read_text()
    bindings = (ROOT / 'src/bindings.cpp').read_text()
    assert 'target_sources(skimagexx PRIVATE src/normalized_mutual_information.cpp)' in cmake
    assert 'target_sources(_core PRIVATE src/bind_normalized_mutual_information.cpp)' in cmake
    assert bindings.count('    bind_normalized_mutual_information(m);') == 1
    for name, text in (('test_normalized_mutual_information_native', cmake),
                       ('normalized_mutual_information_consumer', consumer)):
        assert 'target_compile_options(' + name + ' PRIVATE /UNDEBUG)' in text
        assert 'target_compile_options(' + name + ' PRIVATE -UNDEBUG)' in text
    assert 'target_link_libraries(normalized_mutual_information_consumer PRIVATE skimagexx::skimagexx)' in consumer
    native = (ROOT / 'tests/test_normalized_mutual_information_native.cpp').read_text()
    assert '#include "skimagexx/normalized_mutual_information.hpp"' in native
    assert 'nmi_prepare(' in native and 'nmi_statistics(' in native and '../src/' not in native
    public = (ROOT / 'tests/test_normalized_mutual_information_public.py').read_text()
    assert "import_module('skimagexx._normalized_mutual_information')" in public
    assert "core.__name__ == 'skimagexx._core'" in public
    assert "types.ModuleType" not in public and 'pytest.skip(' not in public
    assert q.GROUPS['public-NMI'] == ('test_normalized_mutual_information_public.py',)


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
    with pytest.raises(q.QualificationError,match='complete112'):
        q.inspect_wheel(*complete_fake_artifact,scope='max_tree19')
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
    monkeypatch.setattr(q,'registry_expectation',lambda _: dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS)),image_count=987))
    (tmp_path/'tests/cmake_consumer').mkdir(parents=True)
    (tmp_path/'tests/cmake_consumer/CMakeLists.txt').write_text('\n'.join('add_executable('+f+'_consumer x.cpp)' for f in q.FAMILIES))
    for names in q.GROUPS.values():
        for name in names:(tmp_path/'tests'/name.split('::')[0]).touch()
    (tmp_path/'tests/test_arraymap_dense_contracts.py').touch()
    for name in q.R6_SOURCE_PATHS+q.R7_SOURCE_PATHS+q.TV_SOURCE_PATHS+q.BREGMAN_SOURCE_PATHS+q.MAX_TREE_SOURCE_PATHS+q.MATRIX_SOURCE_PATHS+q.RECONSTRUCTION_SOURCE_PATHS+q.NMI_SOURCE_PATHS+q.H_EXTREMA_SOURCE_PATHS:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    for family in q.FAMILIES:(tmp_path/'tests'/('test_'+family+'_native.cpp')).touch()
    (tmp_path/'benchmarks').mkdir()
    for name,_ in q.CORPORA.values():(tmp_path/'benchmarks'/name).touch()
    populate_corrected_max_tree_metadata(tmp_path)
    # Source counts are still reported, but an unmapped future generation may
    # no longer get a green build plan merely because an explicit count agrees.
    future=q.plan(tmp_path,'modern',987)
    assert future['registry']['image_count']==987
    assert any('current qualification readiness:' in x for x in future['blockers'])
    assert 'explicit image count differs from source registry' in q.plan(tmp_path,'modern',180)['blockers']
    monkeypatch.setattr(q,'registry_expectation',lambda _: dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS))+['filters.rank.generic.percentile'],image_count=987))
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
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS)),image_count=186))
    (tmp_path/'tests/cmake_consumer').mkdir(parents=True)
    (tmp_path/'tests/cmake_consumer/CMakeLists.txt').write_text('\n'.join('add_executable('+f+'_consumer x.cpp)' for f in q.FAMILIES))
    for name in q.R6_SOURCE_PATHS:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    (tmp_path/missing).unlink()
    assert 'missing required R6 source: '+missing in q.plan(tmp_path,'modern')['blockers']


def test_bilateral_sdk_thirteenth_client_is_mandatory_in_plan(tmp_path,monkeypatch):
    assert len(q.ORIGINAL_FAMILIES)==14 and q.ORIGINAL_FAMILIES[-2:] == ('rank_bilateral','rank_decisions')
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS)),image_count=186))
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
    (path/'coverage.py').write_text('def native_functions(): return [str(i) for i in range(189)]\ndef execution_kind(n): return "native" if int(n)<186 else "metadata"\ndef native_dtypes(n): return ("uint8",)\n')
    synthetic_registry_inventory(tmp_path, [str(i) for i in range(189)], {str(i) for i in range(186,189)})
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
    for name in q.R6_SOURCE_PATHS+q.R7_SOURCE_PATHS+q.TV_SOURCE_PATHS+q.BREGMAN_SOURCE_PATHS+q.MAX_TREE_SOURCE_PATHS+q.MATRIX_SOURCE_PATHS+q.RECONSTRUCTION_SOURCE_PATHS+q.NMI_SOURCE_PATHS+q.H_EXTREMA_SOURCE_PATHS:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    for family in q.FAMILIES:(root/'tests'/('test_'+family+'_native.cpp')).touch()
    (root/'benchmarks').mkdir()
    for name,_ in q.CORPORA.values():(root/'benchmarks'/name).touch()
    populate_corrected_max_tree_metadata(root)


@pytest.mark.parametrize('missing',q.R7_SOURCE_PATHS)
def test_decisions_source_migration_and_public_paths_are_plan_blockers(tmp_path,monkeypatch,missing):
    synthetic_complete_plan(tmp_path)
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS)),image_count=190,registered_count=193,metadata_count=3))
    assert not q.plan(tmp_path,'modern',190)['blockers']
    (tmp_path/missing).unlink()
    assert 'missing required R7 source: '+missing in q.plan(tmp_path,'modern',190)['blockers']


@pytest.mark.parametrize('failure',['missing','duplicate','wrong','source'])
def test_sdk14_requires_exact_clients_and_existing_sources(tmp_path,monkeypatch,failure):
    synthetic_complete_plan(tmp_path)
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS)),image_count=190))
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
    (path/'coverage.py').write_text('def native_functions():return tuple("image"+str(i) for i in range(190))+tuple("metadata"+str(i) for i in range(3))\ndef execution_kind(name):return "native" if name.startswith("image") else "metadata"\ndef native_dtypes(name):return ("uint8",)\n')
    metadata={'metadata'+str(i) for i in range(3)}
    synthetic_registry_inventory(tmp_path, ['image'+str(i) for i in range(190)]+sorted(metadata), metadata)
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
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS)),image_count=190,registered_count=193,metadata_count=3))
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
    assert len(q.BREGMAN_FAMILIES)==18 and q.BREGMAN_FAMILIES[:17]==q.TV_FAMILIES
    assert len(q.MAX_TREE_FAMILIES)==19 and q.MAX_TREE_FAMILIES[:18]==q.BREGMAN_FAMILIES
    assert len(q.FAMILIES)==23 and q.FAMILIES[:19]==q.MAX_TREE_FAMILIES
    assert q.TV_FAMILIES[-3:]==('tv_chambolle','tv_reductions','tv_denoise')
    assert q.BREGMAN_FAMILIES[-1]=='tv_bregman' and q.MAX_TREE_FAMILIES[-1]=='max_tree'
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
    names=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS))
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


def test_bregman110_63_remains_an_exact_explicit_historical_scope(complete_bregman_fake_artifact,complete_tv_fake_artifact,complete_fake_artifact):
    result=q.inspect_wheel(*complete_bregman_fake_artifact,scope='bregman18')
    assert result['product_scope']=='bregman18' and result['record_verified']
    assert len(result['members_sha256'])==110 and len(result['SDK_sha256'])==63
    assert q.product_census('rank14')==(104,59) and q.product_census('tv17')==(108,62)
    assert q.product_census('bregman18')==(110,63) and q.product_census('max_tree19')==(112,64)
    for artifact in (complete_tv_fake_artifact,complete_fake_artifact):
        with pytest.raises(q.QualificationError,match='complete112'):q.inspect_wheel(*artifact,scope='max_tree19')
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
        q.inspect_wheel(product,replacement,revision,scope='bregman18')
    assert calls==[str(product)],'complete110 metadata/RECORD must reach the new source guard'


@pytest.mark.parametrize('missing',q.BREGMAN_SOURCE_PATHS)
def test_bregman_sources_and_registry_block_before_build(tmp_path,monkeypatch,missing):
    synthetic_complete_plan(tmp_path)
    names=['filters.rank.'+name for name in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS))
    monkeypatch.setattr(q,'registry_expectation',lambda root:dict(functions=names,image_count=192,metadata_count=3,registered_count=195))
    assert not q.plan(tmp_path,'modern',192)['blockers']
    (tmp_path/missing).unlink()
    assert 'missing required Bregman source: '+missing in q.plan(tmp_path,'modern',192)['blockers']
    names.remove(q.BREGMAN_PATH)
    assert 'missing public Bregman registry: '+q.BREGMAN_PATH in q.plan(tmp_path,'modern')['blockers']


@pytest.mark.parametrize('scope,families',(('rank14',q.ORIGINAL_FAMILIES),('tv17',q.TV_FAMILIES),('bregman18',q.BREGMAN_FAMILIES),('max_tree19',q.MAX_TREE_FAMILIES),('matrix20',q.MATRIX_FAMILIES),('reconstruction21',q.RECONSTRUCTION_FAMILIES),('nmi22',q.NMI_FAMILIES),('h_extrema22',q.H_EXTREMA_LEGACY_FAMILIES),('h_extrema23',q.FAMILIES)))
def test_each_exact_sdk_census_keeps_missing_duplicate_substitute_guards(scope,families):
    assert len(families)=={'rank14':14,'tv17':17,'bregman18':18,'max_tree19':19,'matrix20':20,'reconstruction21':21,'nmi22':22,'h_extrema22':22,'h_extrema23':23}[scope]
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


@pytest.fixture
def complete_max_tree_fake_artifact(complete_bregman_fake_artifact,tmp_path):
    """Inert metadata only: no native implementation is loaded or executed."""
    product,source,revision=complete_bregman_fake_artifact
    additions={'include/skimagexx/max_tree.hpp':b'// synthetic max-tree header\n',
               'skimagexx/_max_tree.py':b'# synthetic max-tree helper\n'}
    with zipfile.ZipFile(product) as old:
        entries={n:old.read(n) for n in old.namelist() if not n.endswith('/RECORD')}
    entries.update(additions);new_product=tmp_path/'max-tree-synthetic.whl';wheel(new_product,entries)
    new_source=tmp_path/'max-tree-source.zip'
    with zipfile.ZipFile(source) as old,zipfile.ZipFile(new_source,'w') as archive:
        archive.comment=old.comment
        for name in old.namelist():archive.writestr(name,old.read(name))
        for name,value in additions.items():archive.writestr('python/'+name if name.startswith('skimagexx/') else name,value)
    return new_product,new_source,revision


def test_max_tree112_64_is_exact_default_and_all_historical_products_stay_explicit(
        complete_max_tree_fake_artifact,complete_bregman_fake_artifact,complete_tv_fake_artifact,complete_fake_artifact):
    result=q.inspect_wheel(*complete_max_tree_fake_artifact,scope='max_tree19')
    assert result['product_scope']=='max_tree19' and result['record_verified']
    assert len(result['members_sha256'])==112 and len(result['SDK_sha256'])==64
    for scope,artifact,census in (('rank14',complete_fake_artifact,(104,59)),
            ('tv17',complete_tv_fake_artifact,(108,62)),('bregman18',complete_bregman_fake_artifact,(110,63))):
        assert q.product_census(scope)==census
        assert q.inspect_wheel(*artifact,scope=scope)['product_scope']==scope
        with pytest.raises(q.QualificationError,match='complete112'):q.inspect_wheel(*artifact,scope='max_tree19')
        with pytest.raises(q.QualificationError,match='complete'):q.inspect_wheel(*complete_max_tree_fake_artifact,scope=scope)


@pytest.mark.parametrize('member',('include/skimagexx/max_tree.hpp','python/skimagexx/_max_tree.py'))
def test_complete112_product_requires_max_tree_source_header_and_helper(complete_max_tree_fake_artifact,tmp_path,member):
    product,source,revision=complete_max_tree_fake_artifact;replacement=tmp_path/'missing-max-tree.zip'
    with zipfile.ZipFile(source) as old,zipfile.ZipFile(replacement,'w') as archive:
        archive.comment=old.comment
        for name in old.namelist():
            if name!=member:archive.writestr(name,old.read(name))
    with pytest.raises(q.QualificationError,match='max-tree product scope/source'):
        q.inspect_wheel(product,replacement,revision,scope='max_tree19')


@pytest.fixture
def complete_matrix_fake_artifact(complete_max_tree_fake_artifact,tmp_path):
    product,source,revision=complete_max_tree_fake_artifact
    additions={'include/skimagexx/matrix_transform.hpp':b'// inert matrix header\n',
               'skimagexx/_matrix_transform.py':b'# inert matrix helper\n'}
    with zipfile.ZipFile(product) as old:
        entries={n:old.read(n) for n in old.namelist() if not n.endswith('/RECORD')}
    entries.update(additions);new_product=tmp_path/'matrix-synthetic.whl';wheel(new_product,entries)
    new_source=tmp_path/'matrix-source.zip'
    with zipfile.ZipFile(source) as old,zipfile.ZipFile(new_source,'w') as archive:
        archive.comment=old.comment
        for name in old.namelist():archive.writestr(name,old.read(name))
        for name,value in additions.items():archive.writestr('python/'+name if name.startswith('skimagexx/') else name,value)
    return new_product,new_source,revision


def test_matrix114_65_exact_and_max_tree19_historical_scope_is_not_relaxed(
        complete_matrix_fake_artifact,complete_max_tree_fake_artifact):
    result=q.inspect_wheel(*complete_matrix_fake_artifact,scope='matrix20')
    assert result['product_scope']=='matrix20' and result['record_verified']
    assert len(result['members_sha256'])==114 and len(result['SDK_sha256'])==65
    assert len(q.MATRIX_FAMILIES)==20 and q.MATRIX_FAMILIES[-1]=='matrix_transform'
    assert q.inspect_wheel(*complete_max_tree_fake_artifact,scope='max_tree19')['record_verified']
    with pytest.raises(q.QualificationError,match='complete114'):q.inspect_wheel(*complete_max_tree_fake_artifact,scope='matrix20')
    with pytest.raises(q.QualificationError,match='complete112'):q.inspect_wheel(*complete_matrix_fake_artifact,scope='max_tree19')


@pytest.mark.parametrize('member',('include/skimagexx/matrix_transform.hpp','python/skimagexx/_matrix_transform.py'))
def test_matrix114_requires_both_installed_source_members(complete_matrix_fake_artifact,tmp_path,member):
    product,source,revision=complete_matrix_fake_artifact;replacement=tmp_path/'missing-matrix.zip'
    with zipfile.ZipFile(source) as old,zipfile.ZipFile(replacement,'w') as archive:
        archive.comment=old.comment
        for name in old.namelist():
            if name!=member:archive.writestr(name,old.read(name))
    with pytest.raises(q.QualificationError,match='matrix product scope/source'):
        q.inspect_wheel(product,replacement,revision,scope='matrix20')


@pytest.mark.parametrize('missing',q.MATRIX_SOURCE_PATHS)
def test_matrix_source_closure_is_required_before_build(tmp_path,monkeypatch,missing):
    synthetic_complete_plan(tmp_path)
    names=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS))
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=names,image_count=199,metadata_count=3,registered_count=202))
    assert not q.plan(tmp_path,'modern',199)['blockers']
    (tmp_path/missing).unlink()
    assert 'missing required matrix source: '+missing in q.plan(tmp_path,'modern',199)['blockers']


def synthetic_matrix_admission(monkeypatch):
    core=types.SimpleNamespace(__name__='skimagexx._core',get_num_threads=lambda:1,has_openmp=True,
        matrix_transform_interface='matrix-transform-math-v1',matrix_transform_capabilities=31,
        matrix_transform_v2_interface='matrix-transform-math-v2',matrix_transform_v2_capabilities=63)
    for name in ('matrix_transform_run','matrix_transform_run_v2','matrix_transform_completed','matrix_transform_v2_completed'):
        def entry(*a,**kw):raise AssertionError('metadata-only entry never executes')
        entry.__name__=name;entry.__module__=core.__name__;entry.synthetic_builtin=True;setattr(core,name,entry)
    monkeypatch.setattr(q.inspect,'isbuiltin',lambda value:getattr(value,'synthetic_builtin',False))
    def matrix_transform(coords,matrix):raise AssertionError('metadata-only wrapper never executes')
    matrix_transform.__module__='skimagexx._matrix_transform'
    # The workflow runs these metadata guards before numerical dependencies are
    # installed. This explicit inert module is fixture data, never admission of
    # a real runtime or a substitute for a numerical/image implementation.
    synthetic_numpy=types.ModuleType('numpy')
    synthetic_numpy.__version__='metadata-fixture-only'
    monkeypatch.setitem(sys.modules,'numpy',synthetic_numpy)
    wrapper=types.SimpleNamespace(_core=core,matrix_transform=matrix_transform,
                                  _runtime_profile=lambda:synthetic_numpy.__version__)
    transform=types.SimpleNamespace(matrix_transform=matrix_transform,__all__=['matrix_transform'])
    return core,transform,wrapper,[q.MATRIX_PATH]


def test_matrix_admission_metadata_does_not_execute_an_algorithm(monkeypatch):
    q.matrix_admission(*synthetic_matrix_admission(monkeypatch))


@pytest.mark.parametrize('fault',('registry','duplicate','core','workers','no-omp','v1','v2','float-cap','missing-entry',
    'entry-owner','entry-name','python-entry','forwarding','export','signature','star','runtime'))
def test_matrix_admission_refuses_substituted_or_unverified_paths(monkeypatch,fault):
    core,transform,wrapper,registered=synthetic_matrix_admission(monkeypatch)
    if fault=='registry':registered=[]
    elif fault=='duplicate':registered*=2
    elif fault=='core':wrapper._core=object()
    elif fault=='workers':core.get_num_threads=lambda:4
    elif fault=='no-omp':core.has_openmp=False
    elif fault=='v1':core.matrix_transform_capabilities=63
    elif fault=='v2':core.matrix_transform_v2_capabilities=31
    elif fault=='float-cap':core.matrix_transform_v2_capabilities=63.
    elif fault=='missing-entry':del core.matrix_transform_run_v2
    elif fault=='entry-owner':core.matrix_transform_run_v2.__module__='private_core'
    elif fault=='entry-name':core.matrix_transform_run_v2.__name__='borrowed'
    elif fault=='python-entry':core.matrix_transform_run_v2.synthetic_builtin=False
    elif fault=='forwarding':wrapper.matrix_transform.__module__='skimage.transform'
    elif fault=='export':transform.matrix_transform=lambda x:x
    elif fault=='signature':wrapper.matrix_transform.__signature__=inspect.Signature()
    elif fault=='star':transform.__all__=[]
    else:wrapper._runtime_profile=lambda:None
    with pytest.raises(q.QualificationError):q.matrix_admission(core,transform,wrapper,registered)


def require_matrix_projection_witness(row,projection,minor=None):
    """Authenticate the explicitly supported parser schema, never skip a hash."""
    minor=minor or '.'.join(map(str,sys.version_info[:2]))
    assert row['projection_ast_schema']=='python-ast-dump-attributes-off-v1'
    assert row['projection_authored_minor']=='3.13'
    by_minor=row['projection_sha256_by_minor']
    assert set(by_minor)=={'3.11','3.13'}
    assert row['unchanged_full_module_projection_sha256']==by_minor['3.13']
    assert minor in by_minor, 'unsupported matrix projection interpreter schema: '+minor
    assert hashlib.sha256(projection.encode()).hexdigest()==by_minor[minor]


def require_matrix_runtime_admission_successor(frozen, current, row):
    """Authenticate new admission metadata while preserving old math evidence."""
    assert row['schema'] == 'matrix-Windows-ed4-runtime-admission-successor-v1'
    assert row['original_revision'] == 'cfb5c4036a7b4d523ec1689454a75a5b23f489c2'
    assert row['current_revision'] == '4591b711583fb7a8d51c9af3199e82bc8696cc44'
    old = row['original_utf8'].encode()
    assert hashlib.sha256(old).hexdigest() == row['original_raw_git_sha256']
    assert hashlib.sha256(old.replace(b'\r\n', b'\n')).hexdigest() == row['original_canonical_lf_sha256'] == frozen
    assert hashlib.sha256(current).hexdigest() == row['current_canonical_lf_sha256']
    assert row['allowed_function_changes'] == ['_runtime_profile']
    assert row['allowed_function_additions'] == ['_skylakex_driver_fields', '_skylakex_table_readable', '_windows_skylakex_driver']
    assert row['allowed_assignment_additions'] == ['_RUNTIME_DRIVER']
    def project(raw):
        nodes = []
        for node in ast.parse(raw).body:
            if isinstance(node, ast.FunctionDef) and node.name in row['allowed_function_changes'] + row['allowed_function_additions']:
                continue
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in row['allowed_assignment_additions'] for t in node.targets):
                continue
            nodes.append(node)
        return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
    assert project(old) == project(current)


def test_matrix_migration_authenticates_full_module_ast_and_all_frozen_sources():
    migration=json.loads((ROOT/'coverage/evidence/matrix-transform-public-assertion-migration.json').read_bytes())
    for name,row in migration['modules'].items():
        old=row['original_utf8'].encode();new=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(old).hexdigest()==row['original_raw_sha256']
        assert hashlib.sha256(new).hexdigest()==row['current_canonical_lf_sha256']
        def project(raw):
            module=ast.parse(raw);nodes=[]
            for i,node in enumerate(module.body):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in row['allowed_function_changes']:
                    if node.name.startswith('test_native_'):nodes.append(ast.Pass())
                    continue
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in row['allowed_top_level_assignment_changes'] for t in node.targets):continue
                if i==0 and row['allowed_module_doc_change'] and isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):continue
                nodes.append(node)
            return ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)
        assert project(old)==project(new)
        require_matrix_projection_witness(row,project(old))
    updates = migration['qualified_implementation_updates']
    assert set(updates) == {'python/skimagexx/_matrix_transform.py'}
    for name,want in migration['frozen_implementation'].items():
        current = (ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        if name in updates:
            require_matrix_runtime_admission_successor(want, current, updates[name])
        else:
            assert hashlib.sha256(current).hexdigest()==want


@pytest.mark.parametrize('fault', (None, 'schema', 'old-source', 'old-pin', 'new-pin',
    'unbounded-functions', 'math-change', 'legacy-allowset-change'))
def test_matrix_runtime_successor_never_relabels_old_math_or_widens_the_projection(fault):
    migration = json.loads((ROOT/'coverage/evidence/matrix-transform-public-assertion-migration.json').read_bytes())
    name = 'python/skimagexx/_matrix_transform.py'
    row = migration['qualified_implementation_updates'][name]
    current = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
    if fault == 'schema':row['schema'] = 'unknown'
    elif fault == 'old-source':row['original_utf8'] += '\nchanged'
    elif fault == 'old-pin':row['original_canonical_lf_sha256'] = '0' * 64
    elif fault == 'new-pin':row['current_canonical_lf_sha256'] = '0' * 64
    elif fault == 'unbounded-functions':row['allowed_function_changes'].append('_materialized')
    elif fault in ('math-change', 'legacy-allowset-change'):
        if fault == 'math-change':
            current = current.replace(b'transformed = homogeneous @ matrix.T', b'transformed = matrix @ homogeneous.T')
        else:
            current = current.replace(b"frozenset(('Haswell',)),", b"frozenset(('Haswell', 'Unknown')),")
        row['current_canonical_lf_sha256'] = hashlib.sha256(current).hexdigest()
    if fault is None:
        require_matrix_runtime_admission_successor(migration['frozen_implementation'][name], current, row)
    else:
        with pytest.raises(AssertionError):
            require_matrix_runtime_admission_successor(migration['frozen_implementation'][name], current, row)


@pytest.mark.parametrize('fault',(None,'missing-version','wrong-version','unknown-interpreter','schema','authored-witness'))
def test_matrix_projection_requires_known_schema_and_matching_version_witness(fault):
    projection='inert AST projection';digest=hashlib.sha256(projection.encode()).hexdigest()
    row=dict(projection_ast_schema='python-ast-dump-attributes-off-v1',projection_authored_minor='3.13',
             projection_sha256_by_minor={'3.11':digest,'3.13':digest},unchanged_full_module_projection_sha256=digest)
    minor='3.11'
    if fault=='missing-version':del row['projection_sha256_by_minor']['3.11']
    elif fault=='wrong-version':row['projection_sha256_by_minor']['3.11']='0'*64
    elif fault=='unknown-interpreter':minor='3.12'
    elif fault=='schema':row['projection_ast_schema']='unknown-parser'
    elif fault=='authored-witness':row['unchanged_full_module_projection_sha256']='0'*64
    if fault is None:require_matrix_projection_witness(row,projection,minor)
    else:
        with pytest.raises(AssertionError):require_matrix_projection_witness(row,projection,minor)


@pytest.mark.parametrize('fault',(None,'private','registration','missing374','missing4000','provider','mismatch','ordinary','workers'))
def test_matrix_public_routes_require_all_real_completed_cohorts(tmp_path,fault):
    d=dict(public_registration=True,private_backend_adapter=False,operation_timings=0,
        rows=[dict(workers=w) for w in (1,4) for _ in range(374)],
        general_pipeline=[dict(workers=w,cpp_completed=1,expected_sha256='same',actual_sha256='same') for w in (1,4) for _ in range(4000)],
        general_cpp_required_pending=[dict(cpp_calls=1) for _ in range(8)])
    if fault=='private':d['private_backend_adapter']=True
    elif fault=='registration':d['public_registration']=False
    elif fault=='missing374':d['rows'].pop()
    elif fault=='missing4000':d['general_pipeline'].pop()
    elif fault=='provider':d['general_pipeline'][0]['cpp_completed']=0
    elif fault=='mismatch':d['general_pipeline'][0]['actual_sha256']='different'
    elif fault=='ordinary':d['general_cpp_required_pending'][0]['cpp_calls']=0
    elif fault=='workers':
        for row in d['general_pipeline']:row['workers']=1
    path=tmp_path/'routes.json';path.write_text(json.dumps(d))
    if fault is None:q.validate_matrix_rows(path)
    else:
        with pytest.raises(q.QualificationError):q.validate_matrix_rows(path)


def test_matrix_default_call_constructs_coordinate_metadata_without_numeric_execution(monkeypatch):
    """Inert providers expose missing imports without running any image math."""
    namespace={};exec((ROOT/'python/skimagexx/coverage.py').read_text(),namespace)
    events=[];flat=object();coords=object();matrix=object()
    image=types.SimpleNamespace(ravel=lambda:flat)
    def column_stack(values):
        assert values==(flat,flat);events.append('coordinate-columns');return coords
    def eye(size):
        assert size==3;events.append('matrix3');return matrix
    monkeypatch.setitem(sys.modules,'numpy',types.SimpleNamespace(column_stack=column_stack,eye=eye))
    args,kwargs=namespace['default_call']('transform.matrix_transform',image)
    assert args==(coords,matrix) and kwargs=={} and events==['coordinate-columns','matrix3']


def test_public_openmp_metadata_comes_from_actual_core_compilation():
    source=(ROOT/'src/bindings.cpp').read_text()
    assert '#ifdef _OPENMP\n    m.attr("has_openmp") = true;\n#else\n    m.attr("has_openmp") = false;\n#endif' in source
    assert 'target_link_libraries(skimagexx PUBLIC OpenMP::OpenMP_CXX)' in (ROOT/'CMakeLists.txt').read_text()


@pytest.mark.parametrize('fault',(None,'current-source','current-pin','matrix-raw','matrix-pin',
                                 'original-utf8','original-raw-pin','frozen-helper-pin'))
def test_actual_matrix_integrity_gate_checks_original_data_and_migrated_file(tmp_path,fault):
    """Execute the real identity function, with inert source-fact providers only.

    The actual cases/digest are metadata enumeration. No reference/native image
    function executes, and this synthetic fixture is never public acceptance.
    """
    name='tests/test_matrix_transform_contracts.py'
    source=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    matrix_raw=(ROOT/'coverage/contracts/matrix-transform.json').read_bytes()
    data=json.loads(matrix_raw)
    migration=json.loads((ROOT/'coverage/evidence/matrix-transform-public-assertion-migration.json').read_bytes())
    transition=migration['modules'][name]
    original_tree=ast.parse(transition['original_utf8'])
    tree=ast.parse(source)
    old=next(n for n in original_tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_signature_source_and_inventory_integrity')
    current=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==old.name)
    # Exactly the obsolete full-current-file hash is replaced. Every preceding
    # assertion and historical native0/pending assertion remains byte-for-AST.
    assert [ast.dump(n,include_attributes=False) for n in old.body[:-2]]==[ast.dump(n,include_attributes=False) for n in current.body[:len(old.body)-2]]
    assert ast.dump(old.body[-1],include_attributes=False)==ast.dump(current.body[-1],include_attributes=False)
    assert data['test_source_canonical_lf_sha256']==hashlib.sha256(transition['original_utf8'].encode()).hexdigest()
    assert migration['frozen_matrix_raw_sha256']==hashlib.sha256(matrix_raw).hexdigest()
    assert transition['current_canonical_lf_sha256']==hashlib.sha256(source).hexdigest()
    fixture=tmp_path/name;fixture.parent.mkdir(parents=True)
    if fault=='current-source':source+=b'\n# unapproved current source\n'
    elif fault=='current-pin':transition['current_canonical_lf_sha256']='0'*64
    elif fault=='matrix-raw':matrix_raw+=b'\n'
    elif fault=='matrix-pin':migration['frozen_matrix_raw_sha256']='0'*64
    elif fault=='original-utf8':transition['original_utf8']+='\n# unapproved original source\n'
    elif fault=='original-raw-pin':transition['original_raw_sha256']='0'*64
    elif fault=='frozen-helper-pin':data['test_source_canonical_lf_sha256']='0'*64
    fixture.write_bytes(source)
    matrix=tmp_path/'coverage/contracts/matrix-transform.json';matrix.parent.mkdir(parents=True);matrix.write_bytes(matrix_raw)
    receipt=tmp_path/'coverage/evidence/matrix-transform-public-assertion-migration.json';receipt.parent.mkdir(parents=True);receipt.write_text(json.dumps(migration))
    def matrix_transform(coords,matrix):raise AssertionError('metadata-only original image function must never execute')
    profile=next(iter(data['profiles']))
    namespace=dict(__file__=str(fixture),MATRIX=matrix,Path=Path,json=json,hashlib=hashlib,inspect=inspect,
        skimage=types.SimpleNamespace(__version__='0.26.0'),original=types.SimpleNamespace(matrix_transform=matrix_transform),
        matrix_data=lambda:data,profile=lambda:profile,source_facts=lambda:data['profiles'][profile]['source'])
    namespace['DTYPES']=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='DTYPES' for t in n.targets)))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('cases','digest',old.name)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(fixture),'exec'),namespace)
    if fault is None:namespace[old.name]()
    else:
        with pytest.raises(AssertionError):namespace[old.name]()


@pytest.mark.parametrize('missing',q.MAX_TREE_SOURCE_PATHS)
def test_max_tree_source_closure_blocks_before_build(tmp_path,monkeypatch,missing):
    synthetic_complete_plan(tmp_path)
    names=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS))
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=names,image_count=198,metadata_count=3,registered_count=201))
    assert not q.plan(tmp_path,'modern',198)['blockers']
    (tmp_path/missing).unlink()
    assert 'missing required max-tree source: '+missing in q.plan(tmp_path,'modern',198)['blockers']


@pytest.mark.parametrize('missing',q.MAX_TREE_PATHS)
def test_each_max_tree_public_registry_path_is_required_in_plan(tmp_path,monkeypatch,missing):
    synthetic_complete_plan(tmp_path)
    names=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS))
    names.remove(missing)
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=names,image_count=197,metadata_count=3,registered_count=200))
    assert 'missing public max-tree registry: '+missing in q.plan(tmp_path,'modern')['blockers']


def synthetic_max_tree_admission(monkeypatch):
    """Admission-only harness: all entries raise if called; never native proof."""
    core=types.SimpleNamespace(__name__='skimagexx._core',get_num_threads=lambda:1,
        max_tree_interface='max-tree-prepared-v1',max_tree_type_capabilities=1023,
        max_tree_operation_capabilities=63,max_tree_parallel_traversal=False,
        max_tree_stage_interface='max-tree-stages-v1',max_tree_stage_capabilities=31)
    def make_entry(name):
        def entry(*args,**kwargs):raise AssertionError('metadata-only entry must not execute')
        entry.__name__=name;entry.__module__=core.__name__;entry.synthetic_builtin=True
        return entry
    for name in ('max_tree_run','max_tree_counts','max_tree_build_stage','max_tree_area_stage',
                 'max_tree_extension_stage','max_tree_filter_stage','max_tree_maxima_stage'):
        setattr(core,name,make_entry(name))
    monkeypatch.setattr(q.inspect,'isbuiltin',lambda value:getattr(value,'synthetic_builtin',False))
    def max_tree(image,connectivity=1):raise AssertionError('metadata wrapper executed')
    def max_tree_local_maxima(image,connectivity=1,parent=None,tree_traverser=None):raise AssertionError('metadata wrapper executed')
    def area_opening(image,area_threshold=64,connectivity=1,parent=None,tree_traverser=None):raise AssertionError('metadata wrapper executed')
    def area_closing(image,area_threshold=64,connectivity=1,parent=None,tree_traverser=None):raise AssertionError('metadata wrapper executed')
    def diameter_opening(image,diameter_threshold=8,connectivity=1,parent=None,tree_traverser=None):raise AssertionError('metadata wrapper executed')
    def diameter_closing(image,diameter_threshold=8,connectivity=1,parent=None,tree_traverser=None):raise AssertionError('metadata wrapper executed')
    functions={f.__name__:f for f in (max_tree,max_tree_local_maxima,area_opening,area_closing,diameter_opening,diameter_closing)}
    for function in functions.values():function.__module__='skimagexx._max_tree'
    return core,types.SimpleNamespace(**functions),types.SimpleNamespace(_core=core,**functions),list(q.MAX_TREE_PATHS)


def test_max_tree_metadata_admission_never_calls_wrapper_or_entry(monkeypatch):
    q.max_tree_admission(*synthetic_max_tree_admission(monkeypatch))
    assert q.GROUPS['public-MaxTree']==('test_max_tree_contracts.py','test_max_tree_public.py')
    assert not any('native_pipeline' in n for n in q.GROUPS['public-MaxTree'])


@pytest.mark.parametrize('fault',('missing-registry','duplicate-registry','interface','float-types','bool-ops',
    'parallel-true','parallel-int','stage-version','float-stages','missing-entry','entry-name','entry-owner',
    'python-entry','wrong-core','wrong-export','forwarding','wrong-name','signature','default-worker4'))
def test_max_tree_invalid_metadata_rejects_before_any_provider(monkeypatch,fault):
    core,morphology,wrapper,registered=synthetic_max_tree_admission(monkeypatch)
    if fault=='missing-registry':registered.pop()
    elif fault=='duplicate-registry':registered.append(registered[0])
    elif fault=='interface':core.max_tree_interface='stage-only'
    elif fault=='float-types':core.max_tree_type_capabilities=1023.
    elif fault=='bool-ops':core.max_tree_operation_capabilities=True
    elif fault=='parallel-true':core.max_tree_parallel_traversal=True
    elif fault=='parallel-int':core.max_tree_parallel_traversal=0
    elif fault=='stage-version':core.max_tree_stage_interface='wrong'
    elif fault=='float-stages':core.max_tree_stage_capabilities=31.
    elif fault=='missing-entry':del core.max_tree_filter_stage
    elif fault=='entry-name':core.max_tree_area_stage.__name__='max_tree_extension_stage'
    elif fault=='entry-owner':core.max_tree_run.__module__='private_core'
    elif fault=='python-entry':core.max_tree_run.synthetic_builtin=False
    elif fault=='wrong-core':wrapper._core=object()
    elif fault=='wrong-export':morphology.area_opening=lambda image:image
    elif fault=='forwarding':wrapper.area_opening.__module__='skimage.morphology'
    elif fault=='wrong-name':wrapper.area_opening.__name__='borrowed'
    elif fault=='signature':wrapper.area_opening.__signature__=__import__('inspect').Signature()
    else:core.get_num_threads=lambda:4
    with pytest.raises(q.QualificationError):q.max_tree_admission(core,morphology,wrapper,registered)
def test_cursor_native_target_and_installed_pipeline_keep_closed_sdk_scopes():
    cmake=(ROOT/'CMakeLists.txt').read_text()
    assert 'add_executable(test_tv_denoise_cursor_native tests/test_tv_denoise_cursor_native.cpp)' in cmake
    assert 'target_link_libraries(test_tv_denoise_cursor_native PRIVATE skimagexx Threads::Threads)' in cmake
    assert 'add_test(NAME tv_denoise_cursor_native COMMAND test_tv_denoise_cursor_native)' in cmake
    native=(ROOT/'tests/test_tv_denoise_cursor_native.cpp').read_text(encoding='utf-8-sig')
    assert '#include "skimagexx/tv_denoise.hpp"' in native and '#include "skimagexx/image.hpp"' in native
    assert '../src/' not in native and 'execution.hpp' not in native
    assert q.PRODUCT_SCOPE=='h_extrema23' and q.PRODUCT_MEMBERS==120 and q.SDK_MEMBERS==68 and len(q.FAMILIES)==23
    assert q.product_census('bregman18')==(110,63) and len(q.BREGMAN_FAMILIES)==18
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


@pytest.mark.parametrize('missing',('src/tv_denoise_internal.hpp',
                                    'tests/test_tv_denoise_admission_native.cpp'))
def test_single_admission_private_source_is_a_prebuild_requirement(tmp_path,monkeypatch,missing):
    # Explicitly inert source envelope; no compilation or numerical calls.
    assert missing in q.TV_SOURCE_PATHS
    synthetic_complete_plan(tmp_path)
    names=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS))
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=names,image_count=199))
    complete=q.plan(tmp_path,'modern',199)
    assert not complete['blockers']
    assert complete['product_members']==120 and complete['SDK_members']==68
    assert complete['SDK_checks']==46 and len(complete['SDK_families'])==23
    (tmp_path/missing).unlink()
    assert 'missing required TV source: '+missing in q.plan(tmp_path,'modern',199)['blockers']
    assert q.product_census('bregman18') == (110, 63) and len(q.BREGMAN_FAMILIES) == 18


def test_single_admission_private_control_is_not_a_public_client_or_ctest():
    cmake=(ROOT/'CMakeLists.txt').read_text()
    assert 'test_tv_denoise_admission_native' not in cmake
    assert 'tv_denoise_admission' not in q.FAMILIES
    assert len(q.FAMILIES)==23 and q.PRODUCT_MEMBERS==120 and q.SDK_MEMBERS==68
    assert not (ROOT/'include/skimagexx/tv_denoise_internal.hpp').exists()
    assert q.product_census('bregman18') == (110, 63) and len(q.BREGMAN_FAMILIES) == 18
def populate_corrected_max_tree_metadata(root):
    """Only source bytes in inert plans: never imported or compiled."""
    for name in (*q.MAX_TREE_RUNTIME_SOURCE_PINS,q.MAX_TREE_ORACLE_PATH):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    populate_reconstruction_metadata(root)
    for name in (*q.H_EXTREMA_RUNTIME_SOURCE_PINS,*q.H_EXTREMA_FROZEN_RAW_PINS):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    for name in q.NMI_IMPORTED_SOURCE_PINS:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())


def test_corrected_max_tree_groups_keep_original_numerical_gate():
    assert q.GROUPS['public-MaxTree']==('test_max_tree_contracts.py','test_max_tree_public.py')
    assert q.GROUPS['public-MaxTree-fixes']==('test_max_tree_dtype_metadata.py','test_max_tree_sdk_oracle.py')
    assert not any('native_pipeline' in name for name in q.GROUPS['public-MaxTree-fixes'])
    assert q.MAX_TREE_RUNTIME_SOURCE_PINS['python/skimagexx/_max_tree.py']=='3519902c42187e3f79ebe1bf5e17bdf05b562518e773fee05d743f79160db3ae'
    migration=json.loads((ROOT/'coverage/evidence/max-tree-public-assertion-migration.json').read_bytes())
    assert migration['owned_wrapper_canonical_lf_sha256']==q.MAX_TREE_RUNTIME_SOURCE_PINS['python/skimagexx/_max_tree.py']
    # This ORIGINAL assertion is still selected naturally, never deselected.
    tree=ast.parse((ROOT/'tests/test_max_tree_public.py').read_bytes())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_signature_reference_and_frozen_kernel')
    assert "migration['owned_wrapper_canonical_lf_sha256']" in ast.unparse(function)


@pytest.mark.parametrize('name',(*q.MAX_TREE_RUNTIME_SOURCE_PINS,q.MAX_TREE_ORACLE_PATH))
def test_corrected_source_hash_mismatch_blocks_before_any_build(tmp_path,monkeypatch,name):
    synthetic_complete_plan(tmp_path)
    registry=dict(functions=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS)),image_count=199)
    monkeypatch.setattr(q,'registry_expectation',lambda _:registry)
    assert not q.plan(tmp_path,'windows',199)['blockers']
    path=tmp_path/name;path.write_bytes(path.read_bytes()+b'\r\nnot-authorized')
    assert 'corrected max-tree source pin differs: '+name in q.plan(tmp_path,'windows',199)['blockers']


def test_composed_skylakex_metadata_guard_is_required_and_selected(tmp_path, monkeypatch):
    name = 'tests/test_matrix_transform_skylakex_routing.py'
    assert name in q.MATRIX_SOURCE_PATHS
    assert q.GROUPS['public-Matrix'] == (
        'test_matrix_transform_contracts.py', 'test_matrix_transform_public.py',
        'test_matrix_transform_native_pipeline.py', 'test_matrix_transform_binding.py',
        'test_matrix_transform_skylakex_routing.py')
    synthetic_complete_plan(tmp_path)
    functions = ['filters.rank.' + n for n in q.R5] + sorted(
        q.R6_PATHS | q.R7_PATHS | {q.TV_PATH, q.BREGMAN_PATH, q.MATRIX_PATH, q.RECONSTRUCTION_PATH, q.NMI_PATH} | set(q.MAX_TREE_PATHS) | set(q.H_EXTREMA_PATHS))
    monkeypatch.setattr(q, 'registry_expectation', lambda _: dict(functions=functions, image_count=199))
    assert not q.plan(tmp_path, 'windows', 199)['blockers']
    (tmp_path / name).unlink()
    assert 'missing required matrix source: ' + name in q.plan(tmp_path, 'windows', 199)['blockers']



def populate_reconstruction_metadata(root):
    for name in (*q.RECONSTRUCTION_RUNTIME_SOURCE_PINS,*q.RECONSTRUCTION_TRACE_POLICY_PINS,q.RECONSTRUCTION_ORACLE_PATH):
        target=Path(root)/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())

@pytest.fixture
def complete_reconstruction_fake_artifact(complete_matrix_fake_artifact,tmp_path):
    """Inert metadata wheel only, never native/public evidence."""
    product,source,revision=complete_matrix_fake_artifact
    additions={'include/skimagexx/reconstruction.hpp':b'// synthetic reconstruction header\n',
               'skimagexx/_reconstruction.py':b'# synthetic reconstruction helper\n'}
    with zipfile.ZipFile(product) as z:entries={n:z.read(n) for n in z.namelist() if not n.endswith('/RECORD')}
    entries.update(additions);out=tmp_path/'reconstruction-synthetic.whl';wheel(out,entries)
    archive=tmp_path/'reconstruction-source.zip'
    with zipfile.ZipFile(source) as z,zipfile.ZipFile(archive,'w') as target:
        target.comment=z.comment
        for n in z.namelist():target.writestr(n,z.read(n))
        for n,b in additions.items():target.writestr('python/'+n if n.startswith('skimagexx/') else n,b)
    return out,archive,revision


def test_reconstruction116_66_historical_scope_remains_closed(complete_reconstruction_fake_artifact,complete_matrix_fake_artifact):
    result=q.inspect_wheel(*complete_reconstruction_fake_artifact,scope='reconstruction21')
    assert result['product_scope']=='reconstruction21' and len(result['members_sha256'])==116 and len(result['SDK_sha256'])==66
    assert q.inspect_wheel(*complete_matrix_fake_artifact,scope='matrix20')['record_verified']
    with pytest.raises(q.QualificationError,match='complete116'):q.inspect_wheel(*complete_matrix_fake_artifact,scope='reconstruction21')
    with pytest.raises(q.QualificationError,match='complete114'):q.inspect_wheel(*complete_reconstruction_fake_artifact,scope='matrix20')
    assert q.product_census('rank14')==(104,59) and q.product_census('tv17')==(108,62)
    assert q.product_census('bregman18')==(110,63) and q.product_census('max_tree19')==(112,64)
    assert q.product_census('matrix20')==(114,65) and q.product_census('reconstruction21')==(116,66)

@pytest.mark.parametrize('missing',('include/skimagexx/reconstruction.hpp','python/skimagexx/_reconstruction.py'))
def test_reconstruction_product_requires_source_header_and_helper(complete_reconstruction_fake_artifact,tmp_path,missing):
    product,source,revision=complete_reconstruction_fake_artifact;new=tmp_path/'missing-reconstruction-source.zip'
    with zipfile.ZipFile(source) as z,zipfile.ZipFile(new,'w') as target:
        target.comment=z.comment
        for n in z.namelist():
            if n!=missing:target.writestr(n,z.read(n))
    with pytest.raises(q.QualificationError,match='reconstruction product scope/source'):q.inspect_wheel(product,new,revision,scope='reconstruction21')

@pytest.mark.parametrize('missing',q.RECONSTRUCTION_SOURCE_PATHS)
def test_reconstruction_plan_requires_all_source_and_genuine_tests(tmp_path,monkeypatch,missing):
    synthetic_complete_plan(tmp_path)
    names=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS))
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=names,image_count=200,metadata_count=3,registered_count=203))
    assert not q.plan(tmp_path,'modern',200)['blockers']
    (tmp_path/missing).unlink()
    assert 'missing required reconstruction source: '+missing in q.plan(tmp_path,'modern',200)['blockers']

@pytest.mark.parametrize('path',tuple(q.RECONSTRUCTION_RUNTIME_SOURCE_PINS)+tuple(q.RECONSTRUCTION_TRACE_POLICY_PINS)+(q.RECONSTRUCTION_ORACLE_PATH,))
def test_reconstruction_plan_rejects_changed_compiled_oracle_source(tmp_path,monkeypatch,path):
    synthetic_complete_plan(tmp_path)
    names=['filters.rank.'+n for n in q.R5]+sorted(q.R6_PATHS|q.R7_PATHS|{q.TV_PATH,q.BREGMAN_PATH,q.MATRIX_PATH,q.RECONSTRUCTION_PATH,q.NMI_PATH}|set(q.MAX_TREE_PATHS)|set(q.H_EXTREMA_PATHS))
    monkeypatch.setattr(q,'registry_expectation',lambda _:dict(functions=names,image_count=200,metadata_count=3,registered_count=203))
    target=tmp_path/path;target.write_bytes(target.read_bytes()+b'\nchanged')
    assert 'reconstruction source pin differs: '+path in q.plan(tmp_path,'modern',200)['blockers']


def reconstruction_metadata_stub(monkeypatch):
    """Harness-only ownership metadata, no callable numerical implementation."""
    core=types.SimpleNamespace(__name__='unit_sole_core',get_num_threads=lambda:1,has_openmp=True,
        reconstruction_interface='reconstruction-v1',reconstruction_capabilities=3,reconstruction_parallel_flood=False)
    for name in ('reconstruction_run','reconstruction_ordered','reconstruction_negative_zero','reconstruction_counts'):
        def forbidden(*args,**kwargs):raise AssertionError('unit metadata stub executed as algorithm')
        forbidden.__name__=name;forbidden.__module__=core.__name__;forbidden.unit_builtin=True;setattr(core,name,forbidden)
    def reconstruction(seed,mask,method='dilation',footprint=None,offset=None):raise AssertionError('unit wrapper executed')
    reconstruction.__module__='skimagexx._reconstruction'
    wrapper=types.SimpleNamespace(_core=core,reconstruction=reconstruction)
    morphology=types.SimpleNamespace(reconstruction=reconstruction,__all__=['reconstruction'])
    real=q.inspect.isbuiltin;monkeypatch.setattr(q.inspect,'isbuiltin',lambda f:real(f) or getattr(f,'unit_builtin',False))
    return core,morphology,wrapper,[q.RECONSTRUCTION_PATH]


def test_reconstruction_metadata_admission_never_calls_fixture_or_algorithm(monkeypatch):
    q.reconstruction_admission(*reconstruction_metadata_stub(monkeypatch))

@pytest.mark.parametrize('fault',('missing','duplicate','version','float-cap','bool-cap','int-subclass-cap','parallel-int','parallel-true','worker4','no-OpenMP','entry-missing','entry-name','entry-owner','python-entry','borrowed-builtin','class-spoof','wrong-core','wrong-alias','forwarding','signature','duplicate-export'))
def test_reconstruction_metadata_rejected_before_fixture(monkeypatch,fault):
    core,morphology,wrapper,paths=reconstruction_metadata_stub(monkeypatch)
    if fault=='missing':paths=[]
    elif fault=='duplicate':paths*=2
    elif fault=='version':core.reconstruction_interface='stage-other'
    elif fault=='float-cap':core.reconstruction_capabilities=3.
    elif fault=='bool-cap':core.reconstruction_capabilities=True
    elif fault=='int-subclass-cap':
        class Cap(int):pass
        core.reconstruction_capabilities=Cap(3)
    elif fault=='parallel-int':core.reconstruction_parallel_flood=0
    elif fault=='parallel-true':core.reconstruction_parallel_flood=True
    elif fault=='worker4':core.get_num_threads=lambda:4
    elif fault=='no-OpenMP':core.has_openmp=False
    elif fault=='entry-missing':del core.reconstruction_run
    elif fault=='entry-name':core.reconstruction_run.__name__='borrowed'
    elif fault=='entry-owner':core.reconstruction_run.__module__='upstream'
    elif fault=='python-entry':core.reconstruction_run.unit_builtin=False
    elif fault=='borrowed-builtin':core.reconstruction_run=core.reconstruction_ordered
    elif fault=='class-spoof':
        class builtin_function_or_method:
            __module__='unit_sole_core';__name__='reconstruction_run'
        core.reconstruction_run=builtin_function_or_method()
    elif fault=='wrong-core':wrapper._core=object()
    elif fault=='wrong-alias':morphology.reconstruction=lambda *a:None
    elif fault=='forwarding':wrapper.reconstruction.__module__='skimage.morphology'
    elif fault=='signature':wrapper.reconstruction.__signature__=inspect.Signature()
    else:morphology.__all__*=2
    with pytest.raises(q.QualificationError):q.reconstruction_admission(core,morphology,wrapper,paths)


def test_reconstruction_library_binding_sdk_linkage_and_source_oracle_argument():
    text=(ROOT/'CMakeLists.txt').read_text();consumer=(ROOT/'tests/cmake_consumer/CMakeLists.txt').read_text()
    assert 'src/matrix_transform.cpp src/reconstruction.cpp)' in text
    assert 'src/bind_matrix_transform.cpp src/bind_reconstruction.cpp)' in text
    assert 'src/simd_reconstruction_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off;-fno-fast-math"' in text
    assert 'add_test(NAME reconstruction_native COMMAND test_reconstruction_native "${SKIMAGEXX_RECONSTRUCTION_SOURCE_ORACLE}")' in text
    assert 'matrix_transform reconstruction)' in consumer
    assert 'list(APPEND _family_test_args "${SKIMAGEXX_RECONSTRUCTION_SOURCE_ORACLE}")' in consumer
    assert 'COMMAND ${_family}_consumer ${_family_test_args}' in consumer
    assert hashlib.sha256((ROOT/q.RECONSTRUCTION_ORACLE_PATH).read_bytes()).hexdigest()==q.RECONSTRUCTION_ORACLE_SHA256
    source=(ROOT/'src/bindings.cpp').read_text();assert source.count('    bind_reconstruction(m);')==1
    assert q.GROUPS['public-Reconstruction']==('test_reconstruction_contracts.py','test_reconstruction_public.py')
    public=(ROOT/'tests/test_reconstruction_public.py').read_text()
    assert "import_module('skimagexx._reconstruction')" in public and "core.__name__=='skimagexx._core'" in public
    assert "package._core=" not in public and "types.ModuleType" not in public


def test_reconstruction_policy_preserves_the_historical_metadata_chain_and_declares_comparison_change():
    record=json.loads((ROOT/'coverage/evidence/reconstruction-public-assertion-migration.json').read_bytes())
    policy=__import__('runpy').run_path(str(ROOT/'tests/reconstruction_trace_policy.py'))
    successor=policy['authenticate_source'](ROOT,(ROOT/'tests/test_reconstruction_contracts.py').read_bytes())
    assert successor['classification'].startswith('Scientific validation-policy successor')
    old=record['original_source_UTF8'];current=successor['previous_source_UTF8'].encode().replace(b'\r\n',b'\n')
    assert hashlib.sha256(old.encode()).hexdigest()==record['original_raw_sha256']
    frozen=json.loads((ROOT/'coverage/contracts/reconstruction.json').read_bytes())
    assert hashlib.sha256(old.encode().replace(b'\r\n',b'\n')).hexdigest()==record['original_canonical_lf_sha256']==record['original_normalized_lf_sha256']==frozen['test_canonical_lf_sha256']
    assert hashlib.sha256(current).hexdigest()==record['current_canonical_lf_sha256']
    assert hashlib.sha256((ROOT/'coverage/contracts/reconstruction.json').read_bytes()).hexdigest()==record['frozen_matrix_raw_sha256']
    left=ast.parse(old);right=ast.parse(current);assert len(left.body)==len(right.body)
    allowed=set(record['allowed_metadata_body_changes']);assert len(allowed)==2
    for a,b in zip(left.body,right.body):
        if isinstance(a,ast.FunctionDef) and a.name in allowed:a.body=[];b.body=[]
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


SOURCE_INVENTORY_TEST_API = getattr(q, 'RECONSTRUCTION_PATH', q.MATRIX_PATH)


def synthetic_registry_inventory(root, names, metadata=()):
    """Internally closed metadata fixture; never a numerical/public proof."""
    entries = [dict(api=name, kind='callable',
                    implementation='metadata' if name in metadata else 'native-partial',
                    execution_kind='metadata' if name in metadata else 'native',
                    native_default_case_dtypes=['uint8'], parity_test='tests/synthetic.py',
                    benchmark='benchmarks/synthetic.py') for name in names]
    for relative in ('tests/synthetic.py', 'benchmarks/synthetic.py'):
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Synthetic metadata evidence path; no image implementation.\n')
    (root / 'coverage').mkdir(exist_ok=True)
    (root / 'coverage/api.json').write_text(json.dumps({'entries': entries}))


@pytest.fixture
def source_inventory_tree(tmp_path):
    """Real registry/inventory text with inert evidence paths; no package import."""
    for relative in ('python/skimagexx/coverage.py', 'coverage/api.json'):
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    entries = json.loads((tmp_path / 'coverage/api.json').read_bytes())['entries']
    for row in entries:
        if row['implementation'] not in ('native-partial', 'metadata'):
            continue
        for field in ('parity_test', 'benchmark'):
            path = tmp_path / row[field].split('::', 1)[0]
            path.parent.mkdir(parents=True, exist_ok=True); path.touch()
    return tmp_path


def test_actual_source_inventory_closure_has_no_image_package_import(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.', 1)[0] in ('numpy', 'scipy', 'skimage', 'skimagexx'):
            raise AssertionError('image package imported during source inventory guard')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    result = q.registry_expectation(ROOT)
    rows = json.loads((ROOT / 'coverage/api.json').read_bytes())['entries']
    assert result['image_count'] == sum(row['implementation'] == 'native-partial' for row in rows)
    assert result['metadata_count'] == sum(row['implementation'] == 'metadata' for row in rows)
    assert result['registered_count'] == len(result['functions']) == result['image_count'] + result['metadata_count']
    assert SOURCE_INVENTORY_TEST_API in result['functions']


@pytest.mark.parametrize('fault', (
    'stale-reconstruction', 'missing-native-row', 'extra-native-row', 'duplicate-api',
    'wrong-implementation', 'wrong-execution', 'wrong-dtypes', 'wrong-kind',
    'missing-parity', 'missing-benchmark', 'missing-evidence-file',
    'absolute-evidence', 'traversal-evidence', 'windows-evidence',
    'invalid-api', 'invalid-entries', 'invalid-json', 'missing-inventory',
))
def test_source_inventory_inconsistency_rejected_before_build(source_inventory_tree, fault):
    root = source_inventory_tree; path = root / 'coverage/api.json'
    inventory = json.loads(path.read_bytes()); rows = inventory['entries']
    row = next(row for row in rows if row['api'] == SOURCE_INVENTORY_TEST_API)
    if fault == 'stale-reconstruction':
        # Same stale-field pattern as frozen283. On the199-path main this uses
        # matrix_transform; composed sources also exercise reconstruction.
        row.update(implementation='upstream-only', native_default_case_dtypes=[],
                   execution_kind='upstream', parity_test=None, benchmark=None)
    elif fault == 'missing-native-row': rows.remove(row)
    elif fault == 'extra-native-row': rows.append(dict(row, api='morphology.unregistered_synthetic'))
    elif fault == 'duplicate-api': rows.append(dict(row))
    elif fault == 'wrong-implementation': row['implementation'] = 'metadata'
    elif fault == 'wrong-execution': row['execution_kind'] = 'upstream'
    elif fault == 'wrong-dtypes': row['native_default_case_dtypes'] = ['uint8']
    elif fault == 'wrong-kind': row['kind'] = 'class'
    elif fault == 'missing-parity': row['parity_test'] = None
    elif fault == 'missing-benchmark': row['benchmark'] = None
    elif fault == 'missing-evidence-file': (root / row['parity_test']).unlink()
    elif fault == 'absolute-evidence': row['parity_test'] = '/outside.py'
    elif fault == 'traversal-evidence': row['parity_test'] = '../outside.py'
    elif fault == 'windows-evidence': row['parity_test'] = 'C:/outside.py'
    elif fault == 'invalid-api': row['api'] = 7
    elif fault == 'invalid-entries': inventory['entries'] = {}
    if fault == 'missing-inventory': path.unlink()
    else: path.write_text('not JSON' if fault == 'invalid-json' else json.dumps(inventory))
    with pytest.raises(q.QualificationError, match='source inventory'):
        q.registry_expectation(root)


def test_inventory_extra_native_declaration_after_registry_removal_is_rejected(source_inventory_tree):
    path = source_inventory_tree / 'python/skimagexx/coverage.py'
    tree = ast.parse(path.read_text())
    declaration = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'NATIVE_FUNCTIONS' for target in node.targets))
    names = next(value for key, value in zip(declaration.keys, declaration.values) if key.value == SOURCE_INVENTORY_TEST_API.rsplit('.', 1)[0])
    assert sum(value.value == SOURCE_INVENTORY_TEST_API.rsplit('.', 1)[1] for value in names.elts) == 1
    names.elts = [value for value in names.elts if value.value != SOURCE_INVENTORY_TEST_API.rsplit('.', 1)[1]]
    path.write_text(ast.unparse(tree))
    with pytest.raises(q.QualificationError, match='extra native declarations.*' + SOURCE_INVENTORY_TEST_API):
        q.registry_expectation(source_inventory_tree)


def test_stale_declared_inventory_blocks_prepare_before_environment_or_download(source_inventory_tree, monkeypatch):
    root = source_inventory_tree; path = root / 'coverage/api.json'
    inventory = json.loads(path.read_bytes())
    row = next(row for row in inventory['entries'] if row['api'] == SOURCE_INVENTORY_TEST_API)
    row.update(implementation='upstream-only', native_default_case_dtypes=[],
               execution_kind='upstream', parity_test=None, benchmark=None)
    path.write_text(json.dumps(inventory))
    monkeypatch.setattr(q, 'ROOT', root)
    monkeypatch.setattr(q, 'source_state', lambda _: {'revision': 'a' * 40})
    def forbidden(*args, **kwargs):
        raise AssertionError('environment/download/build reached despite stale source inventory')
    monkeypatch.setattr(q, 'base_interpreter', forbidden)
    monkeypatch.setattr(q, 'Journal', forbidden)
    monkeypatch.setattr(q.subprocess, 'Popen', forbidden)
    with pytest.raises(q.QualificationError, match='source inventory/registry mismatch.*' + SOURCE_INVENTORY_TEST_API):
        q.prepare(root / 'fresh', 'modern', 200)
    assert not any((root / 'fresh' / name).exists() for name in ('build-env', 'runtime-env', 'build-wheels', 'runtime-wheels', 'native-build'))


@pytest.fixture
def current_metadata_artifact(tmp_path):
    """Inert independently enumerated package/install envelope; never imported."""
    import tomllib
    source = {name: (ROOT/name).read_bytes() for name in
              ('CMakeLists.txt','pyproject.toml',q.CURRENT_SCOPE_CONTRACT)}
    for name in ('image','tv_chambolle','tv_reductions','tv_denoise','tv_bregman',
                 'max_tree','matrix_transform','reconstruction','h_extrema','normalized_mutual_information'):
        source['include/skimagexx/'+name+'.hpp'] = b'// inert header\n'
    for name in ('__init__','_tv_chambolle','_tv_bregman','_max_tree','_matrix_transform',
                 '_reconstruction','_h_extrema','_normalized_mutual_information'):
        source['python/skimagexx/'+name+'.py'] = b'# inert module\n'
    licenses = tomllib.loads(source['pyproject.toml'].decode())['project']['license-files']
    for name in licenses: source[name] = b'inert license '+name.encode()
    entries = {n.removeprefix('python/'): raw for n,raw in source.items()
               if n.startswith(('python/skimagexx/','include/'))}
    for name in licenses:
        dest = 'pocketfft-LICENSE.md' if name=='third_party/pocketfft/LICENSE.md' else Path(name).name
        entries['share/licenses/scikit-imagexx/'+dest] = source[name]
        entries['scikit_imagexx-0.1.0.dist-info/licenses/'+name] = source[name]
    for name in ('skimagexxConfig.cmake','skimagexxConfigVersion.cmake','skimagexxTargets.cmake','skimagexxTargets-release.cmake'):
        entries['lib/cmake/skimagexx/'+name] = b'inert cmake'
    entries['lib/libskimagexx.a'] = b'NOT A STATIC LIBRARY'
    entries['skimagexx/_core.synthetic.so'] = b'NOT A NATIVE LIBRARY'
    entries['scikit_imagexx-0.1.0.dist-info/METADATA'] = b'Name: scikit-imagexx\nVersion: 0.1.0\n'
    entries['scikit_imagexx-0.1.0.dist-info/WHEEL'] = b'inert wheel metadata'
    product=tmp_path/'current-inert.whl';wheel(product,entries)
    archive=tmp_path/'current-inert-source.zip';revision='c'*40
    with zipfile.ZipFile(archive,'w') as output:
        output.comment=revision.encode()
        for name,raw in source.items():output.writestr(name,raw)
    return product,archive,revision


@pytest.mark.parametrize('profile',('windows','modern','minimum','debug'))
def test_current_source_plan_derives_full_closure_and_labels_sdk23(profile):
    result=q.plan(ROOT,profile,236)
    assert not result['blockers'] and result['readiness_only'] is True
    assert result['product_scope']=='current236' and result['registry']['image_count']==236
    assert result['product_members']==155 and result['SDK_members']==71
    assert result['SDK_checks']==46 and 'legacy23' in result['SDK_client_scope']
    assert len(result['public_groups']['public-post203'])==30
    names=[p.relative_to(ROOT).as_posix() for prefix in ('python/skimagexx','include')
           for p in (ROOT/prefix).rglob('*') if p.is_file()]
    layout=q.current_install_layout(lambda n:(ROOT/n).read_bytes(),names,profile)
    assert ('lib/skimagexx.lib' in layout['SDK_members']) == (profile=='windows')
    assert 'lib/cmake/skimagexx/skimagexxTargets-'+('debug' if profile=='debug' else 'release')+'.cmake' in layout['SDK_members']


@pytest.mark.parametrize('fault',('unknown-api','missing-group','missing-selector'))
def test_current_plan_blocks_unmapped_future_or_missing_contract_group(monkeypatch,fault):
    if fault=='unknown-api':
        registry=q.registry_expectation(ROOT)
        registry['functions'].append('util.unmapped_next_api');registry['image_count']+=1
        monkeypatch.setattr(q,'registry_expectation',lambda _:registry)
    else:
        groups=q.groups_for_source(ROOT)
        if fault=='missing-group':groups.pop('public-post203')
        else:groups['public-post203']=tuple(n for n in groups['public-post203'] if n!='test_compare_images_public.py')
        monkeypatch.setattr(q,'groups_for_source',lambda _:groups)
    assert any('current qualification readiness:' in b for b in q.plan(ROOT,'modern')['blockers'])


@pytest.mark.parametrize('fault',('missing-api-map','missing-required-selector','wrong-scope'))
def test_current_contract_rejects_missing_api_or_selector(fault):
    data=json.loads((ROOT/q.CURRENT_SCOPE_CONTRACT).read_text())
    if fault=='missing-api-map':data['post203_contracts'].pop('util.compare_images')
    elif fault=='missing-required-selector':data['required_selectors'].remove('tests/test_blur_effect_public.py')
    else:data['scope']='h_extrema23'
    with pytest.raises(q.QualificationError):q.current_contract(lambda _:json.dumps(data).encode())


@pytest.mark.parametrize('fault',('extra-package','changed-sdk-install','extra-install'))
def test_current_layout_rejects_unassessed_install_rules(fault):
    def read(name):
        raw=(ROOT/name).read_bytes()
        if fault=='extra-package' and name=='pyproject.toml':raw=raw.replace(b'wheel.packages = ["python/skimagexx"]',b'wheel.packages = ["python/skimagexx", "python/extra"]')
        if fault=='changed-sdk-install' and name=='CMakeLists.txt':raw=raw.replace(b'install(DIRECTORY include/',b'install(DIRECTORY other-include/')
        if fault=='extra-install' and name=='CMakeLists.txt':raw+=b'\ninstall(FILES unassessed.bin DESTINATION share/extra)\n'
        return raw
    with pytest.raises(q.QualificationError):q.current_install_layout(read,['python/skimagexx/__init__.py','include/skimagexx/image.hpp'],'modern')


@pytest.mark.parametrize('fault',(None,'same-count-substitution','extra-file','source-bytes','wrong-profile'))
def test_current_wheel_requires_source_closed_members_and_bytes(current_metadata_artifact,tmp_path,fault):
    product,source,revision=current_metadata_artifact
    if fault in ('same-count-substitution','extra-file','source-bytes'):
        with zipfile.ZipFile(product) as archive:entries={n:archive.read(n) for n in archive.namelist() if not n.endswith('/RECORD')}
        if fault=='same-count-substitution':entries['skimagexx/extra.py']=entries.pop('skimagexx/__init__.py')
        elif fault=='extra-file':entries['skimagexx/extra.py']=b'# extra'
        else:entries['skimagexx/__init__.py']=b'# different source'
        product=tmp_path/'changed.whl';wheel(product,entries)
    kwargs=dict(scope=q.CURRENT_PRODUCT_SCOPE,profile='windows' if fault=='wrong-profile' else 'modern')
    if fault:
        with pytest.raises(q.QualificationError):q.inspect_wheel(product,source,revision,**kwargs)
    else:
        result=q.inspect_wheel(product,source,revision,**kwargs)
        assert result['product_scope']=='current236' and result['record_verified']
        with pytest.raises(q.QualificationError,match='complete120'):q.inspect_wheel(product,source,revision)
        assert q.product_census('h_extrema23')==(120,68)
