"""Synthetic metadata/tooling guards only; no real native/public qualification."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


p = module('synthetic_products_stager', ROOT / 'tools/rank_statistics_products.py')
c = module('synthetic_products_wheel_validator', ROOT / 'benchmarks/rank_statistics_compare.py')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode())


def body_hash(value):
    return hashlib.sha256(value).hexdigest()


def wheel(path, name, members):
    info = name.replace('-', '_') + '-1.0.dist-info/'
    version = {'numpy': '2.5.4', 'scipy': '1.18.1', 'scikit-image': '0.26.0'}.get(name, '1.0')
    members = dict(members, **{info + 'METADATA': f'Name: {name}\nVersion: {version}\n'.encode()})
    record = io.StringIO()
    writer = csv.writer(record, lineterminator='\n')
    for key, data in members.items():
        writer.writerow((key, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(), len(data)))
    writer.writerow((info + 'RECORD', '', ''))
    members[info + 'RECORD'] = record.getvalue().encode()
    with zipfile.ZipFile(path, 'w') as archive:
        for key, value in members.items():
            archive.writestr(key, value)


@pytest.fixture
def synthetic(tmp_path, monkeypatch, request):
    """Tiny inert wheel/source metadata; identity stub is NOT native evidence."""
    q = tmp_path / 'qualification'
    q.mkdir()
    source_root = tmp_path / 'frozen-controller-stubs'
    source_root.mkdir()
    monkeypatch.setattr(p, 'ROOT', source_root)
    scope=getattr(request,'param','rank14')
    assert scope in ('rank14','tv17')
    families = ['synthetic-family-' + str(i) for i in range(14)]
    if scope=='tv17':families += ['tv_chambolle','tv_reductions','tv_denoise']
    expected_sdk=62 if scope=='tv17' else 59
    expected_members=108 if scope=='tv17' else 104
    expected_checks=2*len(families)
    source = {name: b'# Inert synthetic controller, never executed.\n' for name in p.CONTROLLERS}
    source['tools/public_qualification.py'] = ('FAMILIES = ' + repr(tuple(families)) + '\n'+("PRODUCT_SCOPE='tv17'\nPRODUCT_MEMBERS=108\nSDK_MEMBERS=62\n" if scope=='tv17' else '')).encode()
    cfg = {'source_text_sha256': {'tools/rank_statistics_products.py': body_hash(source['tools/rank_statistics_products.py'])},
           'case_contract_sha256': '1' * 64,
           'profiles': {'modern': {'numpy': '2.5.4', 'scipy': '1.18.1'}}}
    source['coverage/contracts/rank-statistics-comparison.json'] = json.dumps(cfg).encode()
    corpus = {'case_contract_sha256': '1' * 64, 'case_inventory': list(range(1379))}
    source['coverage/contracts/rank-statistics-benchmark.json'] = json.dumps(corpus).encode()
    source['python/skimagexx/_rank.py'] = b'# inert Python metadata\n'
    source['include/skimagexx/rank.hpp'] = b'// inert installed header\n'
    source['include/skimagexx/image.hpp'] = b'// inert installed header\n'
    if scope=='tv17':
        for name in ('tv_chambolle','tv_reductions','tv_denoise'):source['include/skimagexx/'+name+'.hpp']=b'// inert TVheader metadata\n'
        source['python/skimagexx/_tv_chambolle.py']=b'# inert TVhelper metadata\n'
    for name, data in source.items():
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    snapshot = {'revision': 'a' * 40, 'files_sha256': {n: body_hash(b) for n, b in source.items()}}
    with zipfile.ZipFile(q / 'source.zip', 'w') as archive:
        archive.comment = snapshot['revision'].encode()
        for name, data in source.items():
            archive.writestr(name, data)
    capsule = q / 'capsule-statistics'
    wheels = capsule / 'wheels'
    wheels.mkdir(parents=True)
    sdk = {n: b for n, b in source.items() if n.startswith('include/')}
    sdk['lib/libskimagexx.a'] = b'NOT A LIBRARY: synthetic metadata only'
    sdk['lib/cmake/skimagexx/skimagexxConfig.cmake'] = b'# inert synthetic metadata'
    sdk.update({'share/licenses/scikit-imagexx/unit-' + str(i): b'unit' for i in range(55)})
    assert len(sdk) == expected_sdk
    for name in ('numpy', 'scipy', 'scikit-image', 'scikit-imagexx'):
        members={'unit.txt': b'no executable code'}
        if name=='scikit-imagexx':
            members=dict(sdk)
            members.update({n.removeprefix('python/'):b for n,b in source.items() if n.startswith('python/skimagexx/')})
            members['skimagexx/_core.synthetic.so']=b'NOT A NATIVE LIBRARY: synthetic metadata only'
            for i in range(expected_members-2-len(members)):
                members['scikit_imagexx-1.0.dist-info/unit-padding-'+str(i)]=b'inert metadata'
        wheel(wheels / (name + '.whl'), name, members)
    records = {f.name: {k: v for k, v in c.verify_wheel(f).items() if k in ('name', 'version', 'sha256')} for f in wheels.iterdir()}
    (capsule / 'runtime-lock.txt').write_bytes(b'synthetic metadata lock\n')
    sdk_hashes = {n: body_hash(b) for n, b in sdk.items()}
    dump(capsule / 'rank-statistics-runtime.json', {'schema': 'ci-offline-measurement-v1', 'source': snapshot,
         'wheels': records, 'runtime': {'sdk_sha256': sdk_hashes}, 'lock_sha256': p.sha(capsule / 'runtime-lock.txt')})
    # Deliberately a different original ZIP stream from the qualification Git ZIP.
    with zipfile.ZipFile(capsule / 'source.zip', 'w') as archive:
        for name, data in source.items():
            archive.writestr(name, data)
    dump(capsule / 'rank-statistics-artifacts.json', {'schema': 'rank-statistics-exact-offline-v1',
         'runtime_manifest_sha256': p.sha(capsule / 'rank-statistics-runtime.json'),
         'corpus_manifest_sha256': body_hash(source['coverage/contracts/rank-statistics-benchmark.json']),
         'files_sha256': p.tree(capsule)})
    (q / 'product-wheels').mkdir()
    (q / 'product-wheels/scikit-imagexx.whl').write_bytes((wheels / 'scikit-imagexx.whl').read_bytes())
    dump(q / 'sdk.json', {'families': families, 'checks': expected_checks, 'SDK_sha256': sdk_hashes})
    dump(q / 'package.json', {'wheel_sha256': p.sha(wheels / 'scikit-imagexx.whl')})
    evidence = q / 'build-identity-evidence/unit/0-compiler.txt'
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b'inert tool metadata')
    dump(q / 'build-identity.json', {'schema': 'rank-statistics-build-identity-v2',
         'evidence_records': {'build-identity-evidence/unit/0-compiler.txt':
                              {'bytes': evidence.stat().st_size, 'sha256': p.sha(evidence), 'source_path': 'unit'}}})
    dump(q / 'build-identity-facts.json', {'schema': 'synthetic-only-no-native-claim'})
    binding = {'schema': 'rank-statistics-build-identity-v2', 'identity_sha256': p.sha(q / 'build-identity.json'),
               'facts_sha256': p.sha(q / 'build-identity-facts.json')}
    dump(q / 'state.json', {'source': snapshot, 'profile': 'modern', 'product_wheel': 'scikit-imagexx.whl', 'build_identity': binding})
    proof = {'schema': 'public-qualification-v2', 'passed': True, 'profile': 'modern', 'source': snapshot,
             'failures': [], 'no_timing': True, 'SDK_checks': expected_checks, 'build_identity': binding,
             'expected_registry': {'image_count': 191 if scope=='tv17' else 190, 'metadata_count': 3, 'registered_count': 194 if scope=='tv17' else 193,
                                   'functions': ['unit-only-path-' + str(i) for i in range(193)]+(['restoration.denoise_tv_chambolle'] if scope=='tv17' else [])}}
    receipt = tmp_path / 'verify.json'

    def reseal():
        proof['records'] = {f.relative_to(q).as_posix(): {'bytes': f.stat().st_size, 'sha256': p.sha(f)}
                            for f in q.rglob('*') if f.is_file() and f.name != 'manifest.json'}
        dump(q / 'manifest.json', proof)
        dump(receipt, {'passed': True, 'manifest_sha256': p.sha(q / 'manifest.json'), 'records_rehashed': len(proof['records'])})

    # The real build identity validator has its own independent guard suite.
    # This explicit in-test stub lets the stager metadata routing be exercised.
    def identity(path, profile, qualification, revision, wheel=None):
        assert qualification == proof and profile == 'modern' and revision == snapshot['revision']
        assert wheel['name'] == 'scikit-imagexx'
        return {'sha256': p.sha(path / 'build-identity.json')}

    monkeypatch.setattr(p, 'controller', lambda: SimpleNamespace(verify_wheel=c.verify_wheel, verify_build_identity=identity))
    reseal()
    return SimpleNamespace(q=q, receipt=receipt, output=tmp_path / 'product', proof=proof, reseal=reseal,
                           capsule=capsule, root=source_root, source=source)


def run(s):
    return p.stage(s.q, s.receipt, s.output, 'modern', '123456')


def test_synthetic_stage_preserves_raws_fulltree_and_outer_zip(synthetic):
    s = synthetic
    result = run(s)
    assert result['status'] == 'staged-unaccepted'
    product = p.load(s.output / 'product.json')
    assert product['files_sha256'] == {k: v for k, v in p.tree(s.output).items() if k != 'product.json'}
    for a, b in [('manifest.json', 'qualification.json'), ('source.zip', 'source.zip'), ('sdk.json', 'sdk.json'),
                 ('build-identity.json', 'build-identity.json'), ('build-identity-facts.json', 'build-identity-facts.json')]:
        assert (s.q / a).read_bytes() == (s.output / b).read_bytes()
    assert p.sha(s.output / 'source.zip') != p.sha(s.capsule / 'source.zip')
    assert p.tree(s.output / 'source') == {n: body_hash(b) for n, b in s.source.items()}
    assert not list(s.output.rglob('*.pyd')) and not (s.output / 'native-build').exists()
    assert not any(k in product for k in ('accepted_products', 'before', 'after'))


@pytest.mark.parametrize('field,value', [('passed', False), ('schema', 'fake'), ('no_timing', False),
                                        ('failures', ['failed']), ('profile', 'windows')])
def test_failed_or_wrong_qualification_blocks_before_output(synthetic, field, value):
    s = synthetic
    s.proof[field] = value
    s.reseal()
    with pytest.raises(RuntimeError, match='terminal passed'):
        run(s)
    assert not s.output.exists()


@pytest.mark.parametrize('kind', ('manifest', 'count', 'passed'))
def test_independent_verifier_receipt_must_bind_exact_manifest(synthetic, kind):
    s = synthetic
    value = p.load(s.receipt)
    value[{'manifest': 'manifest_sha256', 'count': 'records_rehashed', 'passed': 'passed'}[kind]] = 0
    dump(s.receipt, value)
    with pytest.raises(RuntimeError, match='verification receipt'):
        run(s)
    assert not s.output.exists()


@pytest.mark.parametrize('name', ['sdk.json', 'state.json', 'source.zip', 'build-identity-facts.json',
                                 'build-identity.json', 'build-identity-evidence/unit/0-compiler.txt',
                                 'capsule-statistics/runtime-lock.txt', 'capsule-statistics/wheels/numpy.whl'])
def test_original_selected_record_byte_tampering_rejected(synthetic, name):
    s = synthetic
    with (s.q / name).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises((RuntimeError, json.JSONDecodeError)):
        run(s)


def test_missing_identity_capture_cannot_use_old_qualified_product(synthetic):
    s = synthetic
    s.proof.pop('build_identity')
    s.reseal()
    with pytest.raises(RuntimeError, match='identity pins missing'):
        run(s)


def test_old_186_coverage_not_relabelled_190(synthetic):
    s = synthetic
    s.proof['expected_registry']['image_count'] = 186
    s.reseal()
    with pytest.raises(RuntimeError, match='190'):
        run(s)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a\\b', 'C:drive', './other', 'a//b'])
def test_unsafe_names_rejected(name):
    with pytest.raises(RuntimeError, match='unsafe'):
        p.safe(name)


def test_no_existing_or_nested_product_output(synthetic):
    s = synthetic
    with pytest.raises(RuntimeError, match='fresh disjoint'):
        p.stage(s.q, s.receipt, s.q / 'nested', 'modern', '123')
    s.output.mkdir()
    with pytest.raises(RuntimeError, match='fresh disjoint'):
        run(s)


def test_debug_not_a_release_comparison_product(synthetic):
    with pytest.raises(RuntimeError, match='Release'):
        p.stage(synthetic.q, synthetic.receipt, synthetic.output, 'debug', '123')


def test_missing_composed_identity_controller_fails_closed(synthetic, monkeypatch):
    # Exercise the actual controller loader in isolation against a file that
    # lacks the prospective verifier. It must not create invented accepted pins.
    pth = synthetic.root / 'benchmarks/rank_statistics_compare.py'
    pth.write_text('# absent future verifier\n')
    original = module('products_stager_original_loader', ROOT / 'tools/rank_statistics_products.py')
    monkeypatch.setattr(original, 'ROOT', synthetic.root)
    with pytest.raises(RuntimeError, match='verifier missing'):
        original.controller()


@pytest.mark.parametrize('field', ('runtime lock', 'runtime source', 'runtime closure', 'reference version',
                                  'SDK hash', 'SDK families', 'fixed corpus', 'controller', 'Python source',
                                  'header source', 'RECORD'))
def test_staged_payload_checks_are_independent_and_fail_closed(synthetic, field):
    s = synthetic
    run(s)
    output = s.output
    if field.startswith('runtime'):
        value = p.load(output / 'runtime.json')
        if field == 'runtime lock':
            value['lock_sha256'] = '0' * 64
        elif field == 'runtime source':
            value['source'] = {}
        else:
            value['wheels'].pop('numpy.whl')
        dump(output / 'runtime.json', value)
    elif field.startswith('SDK'):
        value = p.load(output / 'sdk.json')
        if field == 'SDK hash':
            value['SDK_sha256'] = {}
        else:
            value['families'][0] = 'wrong-family'
        dump(output / 'sdk.json', value)
    elif field == 'fixed corpus':
        value = p.load(output / 'source/coverage/contracts/rank-statistics-benchmark.json')
        value['case_inventory'].pop()
        dump(output / 'source/coverage/contracts/rank-statistics-benchmark.json', value)
    elif field in ('controller', 'Python source', 'header source'):
        name = {'controller': 'tools/rank_statistics_products.py', 'Python source': 'python/skimagexx/_rank.py',
                'header source': 'include/skimagexx/image.hpp'}[field]
        (output / 'source' / name).write_bytes(b'changed\n')
    elif field == 'reference version':
        value = p.load(output / 'source/coverage/contracts/rank-statistics-comparison.json')
        value['profiles']['modern']['numpy'] = '0.0'
        dump(output / 'source/coverage/contracts/rank-statistics-comparison.json', value)
        # Keep this isolated from the controller-byte guard.
        dump(s.root / 'coverage/contracts/rank-statistics-comparison.json', value)
    else:
        path = output / 'wheels/numpy.whl'
        # Replace a standalone synthetic dependency wheel, not any real package.
        # write_bytes would mutate its immutable hardlink source too, so unlink
        # only this test-owned staged link before constructing the bad archive.
        path.unlink()
        wheel(path, 'numpy', {'unit.txt': b'original'})
        with zipfile.ZipFile(path, 'a') as archive:
            archive.writestr('unrecorded.txt', b'not authenticated')
    with pytest.raises(RuntimeError):
        p.validate_payload(output, s.proof, 'modern', p.controller())


@pytest.mark.parametrize('mode', ('duplicate', 'symlink', 'unsafe', 'different set', 'wrong bytes'))
def test_source_archive_admission_rejects_unsafe_or_changed_members(tmp_path, mode):
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        if mode == 'duplicate':
            z.writestr('file', b'unit')
            with pytest.warns(UserWarning, match='Duplicate'):
                z.writestr('file', b'unit')
        elif mode == 'symlink':
            info = zipfile.ZipInfo('file')
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            z.writestr(info, b'unit')
        elif mode == 'unsafe':
            z.writestr('../file', b'unit')
        elif mode == 'different set':
            z.writestr('other', b'unit')
        else:
            z.writestr('file', b'different')
    expected = {'../file' if mode == 'unsafe' else 'file': body_hash(b'unit')}
    with pytest.raises(RuntimeError):
        p.extract_source(archive, tmp_path / 'extracted', expected)


def test_same_resolved_qualification_directory_cannot_be_product(synthetic):
    s = synthetic
    disguised = s.q.parent / 'unused/../qualification/new'
    with pytest.raises(RuntimeError, match='fresh disjoint'):
        p.stage(s.q, s.receipt, disguised, 'modern', '123')


def test_cli_output_boundary_before_any_staging(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError('staging must not start')
    monkeypatch.setattr(p, 'stage', forbidden)
    monkeypatch.setattr(p, 'ROOT', tmp_path / 'frozen-root')
    with pytest.raises(RuntimeError, match='ignored source build'):
        p.main(['--qualification', str(tmp_path / 'q'), '--verification-receipt', str(tmp_path / 'v'),
                '--output', str(tmp_path / 'outside'), '--profile', 'modern', '--artifact-run', '123'])


def test_workflow_stages_only_after_independent_verify_and_retains_raw_artifact():
    text = (ROOT / '.github/workflows/public-qualification.yml').read_text()
    guards = text.index('Tooling guards before preparation')
    prepare = text.index('Build and qualify one genuine')
    verify = text.index('Independently rehash')
    stage = text.index('Stage a bounded controlled product')
    upload = text.index('name: rank-statistics-product-')
    raw = text.index('name: public-qualification-')
    assert guards < prepare < verify < stage < upload < raw
    assert 'tests/test_build_identity_facts.py tests/test_rank_statistics_products.py' in text[:prepare]
    assert "if: success() && matrix.profile != 'debug'" in text[stage - 20:raw]
    assert "if: always()" in text[upload:raw]
    assert '| tee \'build/qualification-verify-${{ matrix.profile }}.json\'' in text
    assert 'include-hidden-files: true' in text[upload:raw]



@pytest.mark.parametrize('synthetic',['tv17'],indirect=True)
def test_tv_full108_62_sdk17_stage_remains_unaccepted(synthetic):
    s=synthetic;result=run(s)
    assert result['status']=='staged-unaccepted'
    sdk=p.load(s.output/'sdk.json')
    assert len(sdk['SDK_sha256'])==62 and len(sdk['families'])==17 and sdk['checks']==34
    product=c.verify_wheel(s.output/'wheels/scikit-imagexx.whl')
    assert len(product['members'])==108
    assert p.qualification_scope(s.output/'source')['name']=='tv17'
    assert not any(n in p.load(s.output/'product.json') for n in ('accepted_products','before','after'))


@pytest.mark.parametrize('failure',['missing-header','wrong-product-count','wrong-sdk-count','wrong-family',
                                   'duplicate-family','missing-scope','original-with-new-count'])
def test_authenticated_scope_literals_cannot_downgrade_or_omit_tv(tmp_path,failure):
    source=tmp_path/'source';(source/'tools').mkdir(parents=True)
    families=tuple('original-'+str(i) for i in range(14))+('tv_chambolle','tv_reductions','tv_denoise')
    values=dict(FAMILIES=families,PRODUCT_SCOPE='tv17',PRODUCT_MEMBERS=108,SDK_MEMBERS=62)
    if failure=='missing-header':values['SDK_MEMBERS']=61
    elif failure=='wrong-product-count':values['PRODUCT_MEMBERS']=104
    elif failure=='wrong-sdk-count':values['SDK_MEMBERS']=59
    elif failure=='wrong-family':values['FAMILIES']=families[:-1]+('not_tv_denoise',)
    elif failure=='duplicate-family':values['FAMILIES']=families[:-1]+('tv_reductions',)
    elif failure=='missing-scope':del values['PRODUCT_SCOPE']
    else:values['PRODUCT_SCOPE']='rank14'
    (source/'tools/public_qualification.py').write_text('\n'.join(k+' = '+repr(v) for k,v in values.items()))
    with pytest.raises(RuntimeError):p.qualification_scope(source)


def test_original_rank14_complete104_59_fixture_keeps_original_scope(synthetic):
    s=synthetic;run(s)
    scope=p.qualification_scope(s.output/'source')
    assert scope['name']=='rank14' and scope['members']==104 and scope['sdk']==59
    assert len(c.verify_wheel(s.output/'wheels/scikit-imagexx.whl')['members'])==104
    assert len(p.load(s.output/'sdk.json')['families'])==14


@pytest.mark.parametrize('synthetic',['tv17'],indirect=True)
@pytest.mark.parametrize('failure',['old190','missing-TV','duplicate-TV'])
def test_tv_registry_cannot_be_promoted_from_legacy_or_missing_function(synthetic,failure):
    s=synthetic;expected=s.proof['expected_registry']
    if failure=='old190':expected['image_count']=190
    elif failure=='missing-TV':expected['functions'][-1]='unit-only-replacement'
    else:expected['functions'][0]=expected['functions'][-1]
    s.reseal()
    with pytest.raises(RuntimeError):run(s)
    assert not s.output.exists()
