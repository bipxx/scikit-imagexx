"""Synthetic metadata/tooling guards only; no real native/public qualification."""
import ast
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
    assert scope in ('rank14','tv17','bregman18','max_tree19','matrix20','reconstruction21','nmi22','h_extrema23')
    families = ['synthetic-family-' + str(i) for i in range(14)]
    if scope!='rank14':families += ['tv_chambolle','tv_reductions','tv_denoise']
    if scope in ('bregman18','max_tree19','matrix20','reconstruction21','nmi22','h_extrema23'):families += ['tv_bregman']
    if scope in ('max_tree19','matrix20','reconstruction21','nmi22','h_extrema23'):families += ['max_tree']
    if scope in ('matrix20','reconstruction21','nmi22','h_extrema23'):families += ['matrix_transform']
    expected_sdk={'rank14':59,'tv17':62,'bregman18':63,'max_tree19':64,'matrix20':65,'reconstruction21':66,'nmi22':67,'h_extrema23':68}[scope]
    expected_members={'rank14':104,'tv17':108,'bregman18':110,'max_tree19':112,'matrix20':114,'reconstruction21':116,'nmi22':118,'h_extrema23':120}[scope]
    expected_checks=2*len(families)
    source = {name: b'# Inert synthetic controller, never executed.\n' for name in p.CONTROLLERS}
    source['tools/public_qualification.py'] = ('FAMILIES = ' + repr(tuple(families)) + '\n'+(f"PRODUCT_SCOPE={scope!r}\nPRODUCT_MEMBERS={expected_members}\nSDK_MEMBERS={expected_sdk}\n" if scope!='rank14' else '')).encode()
    cfg = {'source_text_sha256': {'tools/rank_statistics_products.py': body_hash(source['tools/rank_statistics_products.py'])},
           'case_contract_sha256': '1' * 64,
           'profiles': {'modern': {'numpy': '2.5.4', 'scipy': '1.18.1'}}}
    source['coverage/contracts/rank-statistics-comparison.json'] = json.dumps(cfg).encode()
    corpus = {'case_contract_sha256': '1' * 64, 'case_inventory': list(range(1379))}
    source['coverage/contracts/rank-statistics-benchmark.json'] = json.dumps(corpus).encode()
    source['python/skimagexx/_rank.py'] = b'# inert Python metadata\n'
    source['include/skimagexx/rank.hpp'] = b'// inert installed header\n'
    source['include/skimagexx/image.hpp'] = b'// inert installed header\n'
    if scope!='rank14':
        for name in ('tv_chambolle','tv_reductions','tv_denoise'):source['include/skimagexx/'+name+'.hpp']=b'// inert TVheader metadata\n'
        source['python/skimagexx/_tv_chambolle.py']=b'# inert TVhelper metadata\n'
    if scope in ('bregman18','max_tree19','matrix20','reconstruction21','nmi22','h_extrema23'):
        source['include/skimagexx/tv_bregman.hpp']=b'// inert Bregmanheader metadata\n'
        source['python/skimagexx/_tv_bregman.py']=b'# inert Bregmanhelper metadata\n'
    if scope in ('max_tree19','matrix20','reconstruction21','nmi22','h_extrema23'):
        source['include/skimagexx/max_tree.hpp']=b'// inert max-tree header metadata\n'
        # Real frozen source bytes in an INERT metadata wheel: never compiled/imported.
        for name in p.MAX_TREE_RUNTIME_SOURCE_PINS:
            source[name]=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        source[p.MAX_TREE_ORACLE_PATH]=(ROOT/p.MAX_TREE_ORACLE_PATH).read_bytes()
        source['tools/public_qualification.py'] += ('MAX_TREE_RUNTIME_SOURCE_PINS='+repr(p.MAX_TREE_RUNTIME_SOURCE_PINS)+'\nMAX_TREE_ORACLE_PATH='+repr(p.MAX_TREE_ORACLE_PATH)+'\nMAX_TREE_ORACLE_SHA256='+repr(p.MAX_TREE_ORACLE_SHA256)+'\n').encode()
    if scope in ('matrix20','reconstruction21','nmi22','h_extrema23'):
        source['include/skimagexx/matrix_transform.hpp']=b'// inert matrix header metadata\n'
        source['python/skimagexx/_matrix_transform.py']=b'# inert matrix helper metadata\n'
    if scope in ('reconstruction21','nmi22','h_extrema23'):
        families += ['reconstruction']
        source['tools/public_qualification.py']=source['tools/public_qualification.py'].replace(('FAMILIES = '+repr(tuple(families[:-1]))).encode(),('FAMILIES = '+repr(tuple(families))).encode())
        expected_checks=2*len(families)
        for name in {**p.RECONSTRUCTION_RUNTIME_SOURCE_PINS,**p.RECONSTRUCTION_TRACE_POLICY_PINS}:
            source[name]=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        source[p.RECONSTRUCTION_ORACLE_PATH]=(ROOT/p.RECONSTRUCTION_ORACLE_PATH).read_bytes()
        source['tools/public_qualification.py']+=('RECONSTRUCTION_RUNTIME_SOURCE_PINS='+repr(p.RECONSTRUCTION_RUNTIME_SOURCE_PINS)+'\nRECONSTRUCTION_TRACE_POLICY_PINS='+repr(p.RECONSTRUCTION_TRACE_POLICY_PINS)+'\nRECONSTRUCTION_ORACLE_PATH='+repr(p.RECONSTRUCTION_ORACLE_PATH)+'\nRECONSTRUCTION_ORACLE_SHA256='+repr(p.RECONSTRUCTION_ORACLE_SHA256)+'\n').encode()
    if scope in ('nmi22','h_extrema23'):
        families += ['normalized_mutual_information']
        source['tools/public_qualification.py']=source['tools/public_qualification.py'].replace(('FAMILIES = '+repr(tuple(families[:-1]))).encode(),('FAMILIES = '+repr(tuple(families))).encode())
        expected_checks=2*len(families)
        for name in p.NMI_IMPORTED_SOURCE_PINS:
            source[name]=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        source['tools/public_qualification.py']+=('NMI_IMPORTED_SOURCE_PINS='+repr(p.NMI_IMPORTED_SOURCE_PINS)+'\n').encode()
    if scope=='h_extrema23':
        families += ['h_extrema']
        source['tools/public_qualification.py']=source['tools/public_qualification.py'].replace(('FAMILIES = '+repr(tuple(families[:-1]))).encode(),('FAMILIES = '+repr(tuple(families))).encode())
        expected_checks=2*len(families)
        for name in (*p.H_EXTREMA_RUNTIME_SOURCE_PINS,*p.H_EXTREMA_FROZEN_RAW_PINS):
            source[name]=(ROOT/name).read_bytes()
        for key in ('H_EXTREMA_RUNTIME_SOURCE_PINS','H_EXTREMA_FROZEN_RAW_PINS','H_EXTREMA_HISTORICAL_RAW_PINS','H_EXTREMA_ORACLE_PATH','H_EXTREMA_ORACLE_SHA256'):
            source['tools/public_qualification.py']+=(key+'='+repr(getattr(p,key))+'\n').encode()
    for name, data in source.items():
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    snapshot = {'revision': 'a' * 40, 'files_sha256': {n: body_hash(b.replace(b'\r\n',b'\n')) for n, b in source.items()}}
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
             'expected_registry': {'image_count': {'rank14':190,'tv17':191,'bregman18':192,'max_tree19':198,'matrix20':199,'reconstruction21':200,'nmi22':201,'h_extrema23':203}[scope], 'metadata_count': 3, 'registered_count': {'rank14':193,'tv17':194,'bregman18':195,'max_tree19':201,'matrix20':202,'reconstruction21':203,'nmi22':204,'h_extrema23':206}[scope],
                                   'functions': ['unit-only-path-' + str(i) for i in range(193)]+(['restoration.denoise_tv_chambolle'] if scope!='rank14' else [])+(['restoration.denoise_tv_bregman'] if scope in ('bregman18','max_tree19','matrix20','reconstruction21','nmi22','h_extrema23') else [])+(['morphology.'+n for n in ('max_tree','max_tree_local_maxima','area_opening','area_closing','diameter_opening','diameter_closing')] if scope in ('max_tree19','matrix20','reconstruction21','nmi22','h_extrema23') else [])+(['transform.matrix_transform'] if scope in ('matrix20','reconstruction21','nmi22','h_extrema23') else [])+(['morphology.reconstruction'] if scope in ('reconstruction21','nmi22','h_extrema23') else [])+([p.NMI_PATH] if scope in ('nmi22','h_extrema23') else [])+(list(p.H_EXTREMA_PATHS) if scope=='h_extrema23' else [])}}
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


@pytest.mark.parametrize('synthetic',['bregman18'],indirect=True)
def test_bregman_full110_63_sdk18_stage_is_bounded_unaccepted(synthetic):
    s=synthetic;result=run(s)
    assert result['status']=='staged-unaccepted'
    sdk=p.load(s.output/'sdk.json');assert len(sdk['SDK_sha256'])==63 and len(sdk['families'])==18 and sdk['checks']==36
    assert len(c.verify_wheel(s.output/'wheels/scikit-imagexx.whl')['members'])==110
    assert p.qualification_scope(s.output/'source')['name']=='bregman18'
    assert p.load(s.output/'qualification.json')['expected_registry']['image_count']==192
    assert not any(name in p.load(s.output/'product.json') for name in ('accepted_products','before','after'))


@pytest.mark.parametrize('fault',('SDK62','wheel108','family17','wrong-family','duplicate-family','missing-scope','tv-scope-with-new-counts'))
def test_bregman_authenticated_source_literals_reject_census_downgrades(fault):
    families=tuple('tooling-original-'+str(i) for i in range(14))+('tv_chambolle','tv_reductions','tv_denoise','tv_bregman')
    values=dict(FAMILIES=families,PRODUCT_SCOPE='bregman18',PRODUCT_MEMBERS=110,SDK_MEMBERS=63)
    control='\n'.join(name+' = '+repr(value) for name,value in values.items())
    assert p.qualification_scope_text(control)['name']=='bregman18'
    if fault=='SDK62':values['SDK_MEMBERS']=62
    elif fault=='wheel108':values['PRODUCT_MEMBERS']=108
    elif fault=='family17':values['FAMILIES']=families[:-1]
    elif fault=='wrong-family':values['FAMILIES']=families[:-1]+('private_family',)
    elif fault=='duplicate-family':values['FAMILIES']=families[:-1]+('tv_denoise',)
    elif fault=='missing-scope':del values['PRODUCT_SCOPE']
    else:values['PRODUCT_SCOPE']='tv17'
    with pytest.raises(RuntimeError):p.qualification_scope_text('\n'.join(name+' = '+repr(value) for name,value in values.items()))


@pytest.mark.parametrize('synthetic',['bregman18'],indirect=True)
@pytest.mark.parametrize('fault',('image191','missing-Bregman','duplicate-Bregman','missing-TV'))
def test_bregman192_registry_requires_both_real_public_paths(synthetic,fault):
    s=synthetic;expected=s.proof['expected_registry']
    if fault=='image191':expected['image_count']=191
    elif fault=='missing-Bregman':expected['functions'][-1]='unit-only-replacement'
    elif fault=='duplicate-Bregman':expected['functions'][0]=expected['functions'][-1]
    else:expected['functions'][-2]='unit-only-replacement'
    s.reseal()
    match={'image191':'genuine192','missing-Bregman':'Bregman registry/source','duplicate-Bregman':'genuine192','missing-TV':'TV registry/source'}[fault]
    with pytest.raises(RuntimeError,match=match):run(s)
    assert not s.output.exists()


@pytest.mark.parametrize('scope,members,sdk,suffix',(('tv17',108,62,('tv_chambolle','tv_reductions','tv_denoise')),('bregman18',110,63,('tv_chambolle','tv_reductions','tv_denoise','tv_bregman')),('max_tree19',112,64,('tv_chambolle','tv_reductions','tv_denoise','tv_bregman','max_tree'))))
@pytest.mark.parametrize('field',('PRODUCT_MEMBERS','SDK_MEMBERS'))
def test_authenticated_census_counts_are_integer_literals_not_equal_floats(scope,members,sdk,suffix,field):
    values=dict(FAMILIES=tuple('tooling-'+str(i) for i in range(14))+suffix,PRODUCT_SCOPE=scope,PRODUCT_MEMBERS=members,SDK_MEMBERS=sdk)
    values[field]=float(values[field])
    with pytest.raises(RuntimeError,match='scope incomplete'):
        p.qualification_scope_text('\n'.join(name+' = '+repr(value) for name,value in values.items()))


@pytest.mark.parametrize('synthetic',['max_tree19'],indirect=True)
def test_max_tree_full112_64_sdk19_stage_stays_unaccepted(synthetic):
    s=synthetic;result=run(s)
    assert result['status']=='staged-unaccepted'
    sdk=p.load(s.output/'sdk.json')
    assert len(sdk['SDK_sha256'])==64 and len(sdk['families'])==19 and sdk['checks']==38
    assert len(c.verify_wheel(s.output/'wheels/scikit-imagexx.whl')['members'])==112
    assert p.qualification_scope(s.output/'source')['name']=='max_tree19'
    assert p.load(s.output/'qualification.json')['expected_registry']['image_count']==198
    assert not any(n in p.load(s.output/'product.json') for n in ('accepted_products','before','after'))


@pytest.mark.parametrize('fault',('SDK63','wheel110','family18','wrong-family','duplicate-family','missing-scope','bregman-scope-with-new-counts','unknown-scope'))
def test_max_tree_authenticated_literals_reject_census_downgrades(fault):
    families=tuple('tooling-original-'+str(i) for i in range(14))+('tv_chambolle','tv_reductions','tv_denoise','tv_bregman','max_tree')
    values=dict(FAMILIES=families,PRODUCT_SCOPE='max_tree19',PRODUCT_MEMBERS=112,SDK_MEMBERS=64)
    assert p.qualification_scope_text('\n'.join(k+' = '+repr(v) for k,v in values.items()))['name']=='max_tree19'
    if fault=='SDK63':values['SDK_MEMBERS']=63
    elif fault=='wheel110':values['PRODUCT_MEMBERS']=110
    elif fault=='family18':values['FAMILIES']=families[:-1]
    elif fault=='wrong-family':values['FAMILIES']=families[:-1]+('private_family',)
    elif fault=='duplicate-family':values['FAMILIES']=families[:-1]+('tv_denoise',)
    elif fault=='missing-scope':del values['PRODUCT_SCOPE']
    elif fault=='unknown-scope':values['PRODUCT_SCOPE']='max_tree20'
    else:values['PRODUCT_SCOPE']='bregman18'
    with pytest.raises(RuntimeError):p.qualification_scope_text('\n'.join(k+' = '+repr(v) for k,v in values.items()))


@pytest.mark.parametrize('synthetic',['max_tree19'],indirect=True)
@pytest.mark.parametrize('fault',('image192','missing-max-tree','duplicate-max-tree','missing-Bregman','missing-TV'))
def test_max_tree198_registry_requires_all_six_and_prior_public_paths(synthetic,fault):
    s=synthetic;expected=s.proof['expected_registry']
    if fault=='image192':expected['image_count']=192
    elif fault=='missing-max-tree':expected['functions'][-1]='unit-only-replacement'
    elif fault=='duplicate-max-tree':expected['functions'][0]=expected['functions'][-1]
    elif fault=='missing-Bregman':expected['functions'][-7]='unit-only-replacement'
    else:expected['functions'][-8]='unit-only-replacement'
    s.reseal()
    with pytest.raises(RuntimeError):run(s)
    assert not s.output.exists()


@pytest.mark.parametrize('synthetic',['matrix20'],indirect=True)
def test_matrix199_full114_65_sdk20_stage_remains_unaccepted(synthetic):
    s=synthetic;result=run(s)
    assert result['status']=='staged-unaccepted'
    sdk=p.load(s.output/'sdk.json')
    assert len(sdk['SDK_sha256'])==65 and len(sdk['families'])==20 and sdk['checks']==40
    assert len(c.verify_wheel(s.output/'wheels/scikit-imagexx.whl')['members'])==114
    assert p.qualification_scope(s.output/'source')['name']=='matrix20'
    assert p.load(s.output/'qualification.json')['expected_registry']['image_count']==199
    assert not any(n in p.load(s.output/'product.json') for n in ('accepted_products','before','after'))


@pytest.mark.parametrize('fault',('SDK64','wheel112','family19','wrong-family','duplicate-family','missing-scope','max-tree-scope','float-count'))
def test_matrix_source_literals_cannot_downgrade_closed_scope(fault):
    families=tuple('tooling-original-'+str(i) for i in range(14))+('tv_chambolle','tv_reductions','tv_denoise','tv_bregman','max_tree','matrix_transform')
    values=dict(FAMILIES=families,PRODUCT_SCOPE='matrix20',PRODUCT_MEMBERS=114,SDK_MEMBERS=65)
    assert p.qualification_scope_text('\n'.join(k+' = '+repr(v) for k,v in values.items()))['name']=='matrix20'
    if fault=='SDK64':values['SDK_MEMBERS']=64
    elif fault=='wheel112':values['PRODUCT_MEMBERS']=112
    elif fault=='family19':values['FAMILIES']=families[:-1]
    elif fault=='wrong-family':values['FAMILIES']=families[:-1]+('private_family',)
    elif fault=='duplicate-family':values['FAMILIES']=families[:-1]+('max_tree',)
    elif fault=='missing-scope':del values['PRODUCT_SCOPE']
    elif fault=='float-count':values['PRODUCT_MEMBERS']=114.
    else:values['PRODUCT_SCOPE']='max_tree19'
    with pytest.raises(RuntimeError):p.qualification_scope_text('\n'.join(k+' = '+repr(v) for k,v in values.items()))


@pytest.mark.parametrize('synthetic',['matrix20'],indirect=True)
@pytest.mark.parametrize('fault',('image198','missing-matrix','duplicate-matrix','missing-max-tree','missing-TV','missing-Bregman'))
def test_matrix199_requires_prior_paths_and_one_matrix_registration(synthetic,fault):
    s=synthetic;expected=s.proof['expected_registry']
    if fault=='image198':expected['image_count']=198
    elif fault=='missing-matrix':expected['functions'][-1]='unit-replacement'
    elif fault=='duplicate-matrix':expected['functions'][0]=expected['functions'][-1]
    else:
        target={'missing-max-tree':'morphology.max_tree','missing-TV':'restoration.denoise_tv_chambolle','missing-Bregman':'restoration.denoise_tv_bregman'}[fault]
        expected['functions'][expected['functions'].index(target)]='unit-replacement'
    s.reseal()
    with pytest.raises(RuntimeError):run(s)
    assert not s.output.exists()


@pytest.mark.parametrize('synthetic',['max_tree19','matrix20'],indirect=True)
@pytest.mark.parametrize('fault',('missing-declaration','old-wrapper-declaration','missing-wrapper','old-wrapper-bytes','changed-other-runtime','missing-oracle','binary-CRLF','missing-snapshot-pin','forged-snapshot-pin','old-header-declaration','old-header-bytes'))
def test_corrected_max_tree_source_archive_refuses_partial_or_stale(synthetic,fault):
    """Real source bytes in inert metadata only: no algorithm/native execution."""
    s=synthetic
    with zipfile.ZipFile(s.q/'source.zip') as z:source={name:z.read(name) for name in z.namelist()}
    snapshot={'files_sha256':{name:body_hash(raw) for name,raw in source.items()}}
    wrapper='python/skimagexx/_max_tree.py';header='include/skimagexx/max_tree.hpp'
    if fault=='missing-declaration':source['tools/public_qualification.py']+=b'\nMAX_TREE_RUNTIME_SOURCE_PINS={}\n'
    elif fault=='old-wrapper-declaration':
        source['tools/public_qualification.py']=source['tools/public_qualification.py'].replace(b'3519902c42187e3f79ebe1bf5e17bdf05b562518e773fee05d743f79160db3ae',b'faf537524b273dcdbb25a702590d139fdf94f8823b480bf7cef0f2cd3aa8eb53')
    elif fault=='missing-wrapper':del source[wrapper]
    elif fault=='old-wrapper-bytes':
        block=b"    elif operation in (3,5):\n        # The literal closing route allocates through subtract's canonical\n        # scalar dtype before filtering. Unlike opening's copy, it drops tags.\n        output=np.empty(image.shape,dtype=image.dtype.type)\n"
        assert block in source[wrapper]
        source[wrapper]=source[wrapper].replace(block,b'')
        assert body_hash(source[wrapper])=='faf537524b273dcdbb25a702590d139fdf94f8823b480bf7cef0f2cd3aa8eb53'
        snapshot['files_sha256'][wrapper]=body_hash(source[wrapper])
    elif fault=='changed-other-runtime':source[header]+=b'\n// caller resealed an unauthorized header';snapshot['files_sha256'][header]=body_hash(source[header])
    elif fault=='old-header-declaration':
        source['tools/public_qualification.py']=source['tools/public_qualification.py'].replace(b'60b33407a721e57b3629e671f0f822ab1dd344ccbaddfcf526a9e3f93ca9bd1c',b'7d66238905451f5ec2ecc7a1a8de6649dcb324c8f26bfa7c2e42dc00909be8e6')
    elif fault=='old-header-bytes':
        source[header]=private_header_from_public(source[header])
        snapshot['files_sha256'][header]=body_hash(source[header])
    elif fault=='missing-oracle':del source[p.MAX_TREE_ORACLE_PATH]
    elif fault=='binary-CRLF':
        raw=source[p.MAX_TREE_ORACLE_PATH];assert b'\n' in raw
        source[p.MAX_TREE_ORACLE_PATH]=raw.replace(b'\n',b'\r\n',1)
        assert hashlib.sha256(source[p.MAX_TREE_ORACLE_PATH].replace(b'\r\n',b'\n')).hexdigest()==snapshot['files_sha256'][p.MAX_TREE_ORACLE_PATH]
    elif fault=='missing-snapshot-pin':del snapshot['files_sha256'][wrapper]
    else:snapshot['files_sha256'][wrapper]='0'*64
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for name,raw in source.items():z.writestr(name,raw)
    stream.seek(0)
    with zipfile.ZipFile(stream) as z:
        with pytest.raises(RuntimeError,match='max-tree source'):p.validate_max_tree_source(z,snapshot,p.qualification_scope(s.root))


@pytest.mark.parametrize('scope',('rank14','tv17','bregman18'))
def test_prior_scopes_do_not_invent_max_tree_source_requirements(scope):
    class NotAnArchive:
        def read(self,*args):raise AssertionError('legacy scope touched new max-tree archive')
    p.validate_max_tree_source(NotAnArchive(),{},dict(name=scope))

def private_header_from_public(raw):
    raw=raw.replace(b'// six prepared operations',b'// private six-operation surface')
    raw=raw.replace(b'// Rechecks admission before writes. Stable typed ordering (U8 counting/U16\n// radix, otherwise merge; ties in original flat-index order), descending\n// union/find and ascending canonicalization; reverse',b'// Rechecks admission before writes. Stable typed mergesort (ties in original\n// flat-index order), descending union/find, ascending canonicalization; reverse')
    assert body_hash(raw)=='7d66238905451f5ec2ecc7a1a8de6649dcb324c8f26bfa7c2e42dc00909be8e6'
    return raw


@pytest.mark.parametrize('synthetic', ['matrix20'], indirect=True)
@pytest.mark.parametrize('line_endings', ('LF', 'CRLF'))
def test_corrected_text_fixture_models_canonical_archive_but_keeps_binary_oracle(synthetic, line_endings):
    with zipfile.ZipFile(synthetic.q / 'source.zip') as archive:
        source = {name: archive.read(name) for name in archive.namelist()}
    oracle = source[p.MAX_TREE_ORACLE_PATH]
    for name in p.MAX_TREE_RUNTIME_SOURCE_PINS:
        assert b'\r\n' not in source[name]
        if line_endings == 'CRLF':
            source[name] = source[name].replace(b'\n', b'\r\n')
    snapshot = {'files_sha256': {
        name: body_hash(data.replace(b'\r\n', b'\n') if name in p.MAX_TREE_RUNTIME_SOURCE_PINS else data)
        for name, data in source.items()}}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, data in source.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
        p.validate_max_tree_source(archive, snapshot, {'name': 'matrix20'})
    assert source[p.MAX_TREE_ORACLE_PATH] == oracle == (ROOT / p.MAX_TREE_ORACLE_PATH).read_bytes()


@pytest.mark.parametrize('synthetic',['reconstruction21'],indirect=True)
def test_reconstruction200_full116_66_sdk21_stage_is_unaccepted(synthetic):
    s=synthetic;result=run(s)
    assert result['status']=='staged-unaccepted'
    sdk=p.load(s.output/'sdk.json')
    assert len(sdk['SDK_sha256'])==66 and len(sdk['families'])==21 and sdk['checks']==42
    assert len(c.verify_wheel(s.output/'wheels/scikit-imagexx.whl')['members'])==116
    assert p.qualification_scope(s.output/'source')['name']=='reconstruction21'
    registry=p.load(s.output/'qualification.json')['expected_registry']
    assert registry['image_count']==200 and registry['registered_count']==203
    assert registry['functions'].count('morphology.reconstruction')==1
    assert not any(k in p.load(s.output/'product.json') for k in ('accepted_products','before','after'))

@pytest.mark.parametrize('synthetic',['reconstruction21'],indirect=True)
@pytest.mark.parametrize('fault',('image199','missing-reconstruction','duplicate-reconstruction','missing-matrix','missing-max-tree'))
def test_reconstruction200_registry_cannot_downgrade_or_omit_prior_paths(synthetic,fault):
    s=synthetic;e=s.proof['expected_registry']
    if fault=='image199':e['image_count']=199
    elif fault=='duplicate-reconstruction':e['functions'][0]='morphology.reconstruction'
    else:
        target={'missing-reconstruction':'morphology.reconstruction','missing-matrix':'transform.matrix_transform','missing-max-tree':'morphology.max_tree'}[fault]
        e['functions'][e['functions'].index(target)]='unit-replacement'
    s.reseal()
    with pytest.raises(RuntimeError):run(s)
    assert not s.output.exists()

@pytest.mark.parametrize('synthetic',['reconstruction21'],indirect=True)
@pytest.mark.parametrize('name',tuple(p.RECONSTRUCTION_RUNTIME_SOURCE_PINS)+tuple(p.RECONSTRUCTION_TRACE_POLICY_PINS)+(p.RECONSTRUCTION_ORACLE_PATH,))
@pytest.mark.parametrize('fault',('missing','mutated','forged-snapshot','missing-declaration'))
def test_reconstruction_source_closure_rejects_partial_or_resealed_bytes(synthetic,name,fault):
    with zipfile.ZipFile(synthetic.q/'source.zip') as z:source={n:z.read(n) for n in z.namelist()}
    snapshot={'files_sha256':{n:body_hash(raw) for n,raw in source.items()}}
    if fault=='missing':del source[name]
    elif fault=='mutated':source[name]+=b'changed';snapshot['files_sha256'][name]=body_hash(source[name])
    elif fault=='forged-snapshot':snapshot['files_sha256'][name]='0'*64
    else:source['tools/public_qualification.py']+=b'\nRECONSTRUCTION_RUNTIME_SOURCE_PINS={}\n'
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for n,raw in source.items():z.writestr(n,raw)
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as z:
        with pytest.raises(RuntimeError,match='reconstruction source'):p.validate_reconstruction_source(z,snapshot,{'name':'reconstruction21'})

@pytest.mark.parametrize('scope',('rank14','tv17','bregman18','max_tree19','matrix20'))
def test_legacy_scopes_never_invent_reconstruction_requirements(scope):
    class ForbiddenArchive:
        def read(self,*a):raise AssertionError('legacy touched new reconstruction source')
    p.validate_reconstruction_source(ForbiddenArchive(),{},dict(name=scope))


@pytest.mark.parametrize('fault', ('missing', 'duplicate', 'changed', 'extra-key'))
def test_reconstruction_trace_policy_declaration_cannot_be_omitted_or_resealed(fault):
    declared = dict(p.RECONSTRUCTION_TRACE_POLICY_PINS)
    if fault == 'changed':declared[next(iter(declared))] = '0' * 64
    elif fault == 'extra-key':declared['tests/unchecked_policy.py'] = '0' * 64
    statements = [
        'RECONSTRUCTION_RUNTIME_SOURCE_PINS=' + repr(p.RECONSTRUCTION_RUNTIME_SOURCE_PINS),
        'RECONSTRUCTION_ORACLE_PATH=' + repr(p.RECONSTRUCTION_ORACLE_PATH),
        'RECONSTRUCTION_ORACLE_SHA256=' + repr(p.RECONSTRUCTION_ORACLE_SHA256)]
    if fault != 'missing':statements.append('RECONSTRUCTION_TRACE_POLICY_PINS=' + repr(declared))
    if fault == 'duplicate':statements.append('RECONSTRUCTION_TRACE_POLICY_PINS=' + repr(declared))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('tools/public_qualification.py', '\n'.join(statements))
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
        with pytest.raises(RuntimeError, match='reconstruction source'):
            p.validate_reconstruction_source(archive, {}, {'name': 'reconstruction21'})


@pytest.mark.parametrize('synthetic', ['nmi22'], indirect=True)
def test_nmi201_complete118_67_sdk22_stage_remains_unaccepted(synthetic):
    s = synthetic
    assert run(s)['status'] == 'staged-unaccepted'
    sdk = p.load(s.output / 'sdk.json')
    assert len(sdk['SDK_sha256']) == 67 and len(sdk['families']) == 22 and sdk['checks'] == 44
    assert sdk['families'][-2:] == ['reconstruction', 'normalized_mutual_information']
    assert len(c.verify_wheel(s.output / 'wheels/scikit-imagexx.whl')['members']) == 118
    assert p.qualification_scope(s.output / 'source')['name'] == 'nmi22'
    registry = p.load(s.output / 'qualification.json')['expected_registry']
    assert registry['image_count'] == 201 and registry['registered_count'] == 204
    assert registry['functions'].count(p.NMI_PATH) == 1
    assert registry['functions'].count('morphology.reconstruction') == 1
    assert not any(k in p.load(s.output / 'product.json') for k in ('accepted_products', 'before', 'after'))
    assert (s.output / 'source.zip').read_bytes() == (s.q / 'source.zip').read_bytes()


@pytest.mark.parametrize('synthetic', ['nmi22'], indirect=True)
@pytest.mark.parametrize('fault', (
    'image200', 'registered203', 'metadata2', 'missing-NMI', 'duplicate-NMI',
    'missing-reconstruction', 'missing-matrix', 'missing-max-tree', 'missing-Bregman', 'missing-TV',
))
def test_nmi_registry_requires_unique_new_and_all_prior_paths(synthetic, fault):
    s = synthetic; expected = s.proof['expected_registry']
    if fault == 'image200': expected['image_count'] = 200
    elif fault == 'registered203': expected['registered_count'] = 203
    elif fault == 'metadata2': expected['metadata_count'] = 2
    elif fault == 'duplicate-NMI': expected['functions'][0] = p.NMI_PATH
    else:
        target = {'missing-NMI': p.NMI_PATH, 'missing-reconstruction': 'morphology.reconstruction',
                  'missing-matrix': 'transform.matrix_transform', 'missing-max-tree': 'morphology.max_tree',
                  'missing-Bregman': 'restoration.denoise_tv_bregman',
                  'missing-TV': 'restoration.denoise_tv_chambolle'}[fault]
        expected['functions'][expected['functions'].index(target)] = 'unit-replacement'
    s.reseal()
    with pytest.raises(RuntimeError): run(s)
    assert not s.output.exists()


@pytest.mark.parametrize('synthetic', ['rank14', 'reconstruction21'], indirect=True)
def test_legacy_registry_cannot_claim_nmi_without_nmi_scope(synthetic):
    synthetic.proof['expected_registry']['functions'][0] = p.NMI_PATH
    synthetic.reseal()
    with pytest.raises(RuntimeError, match='NMI registry/source'): run(synthetic)
    assert not synthetic.output.exists()


@pytest.mark.parametrize('fault', (
    'SDK66', 'wheel116', 'family21', 'wrong-family', 'duplicate-family', 'missing-scope',
    'reconstruction-scope', 'unknown-scope', 'float-wheel', 'float-SDK',
))
def test_nmi_source_census_cannot_downgrade_or_invent_a_scope(fault):
    suffix = ('tv_chambolle', 'tv_reductions', 'tv_denoise', 'tv_bregman', 'max_tree',
              'matrix_transform', 'reconstruction', 'normalized_mutual_information')
    values = dict(FAMILIES=tuple('synthetic-' + str(i) for i in range(14)) + suffix,
                  PRODUCT_SCOPE='nmi22', PRODUCT_MEMBERS=118, SDK_MEMBERS=67)
    assert p.qualification_scope_text('\n'.join(k + '=' + repr(v) for k, v in values.items()))['name'] == 'nmi22'
    if fault == 'SDK66': values['SDK_MEMBERS'] = 66
    elif fault == 'wheel116': values['PRODUCT_MEMBERS'] = 116
    elif fault == 'family21': values['FAMILIES'] = values['FAMILIES'][:-1]
    elif fault == 'wrong-family': values['FAMILIES'] = values['FAMILIES'][:-1] + ('other',)
    elif fault == 'duplicate-family': values['FAMILIES'] = values['FAMILIES'][:-1] + ('reconstruction',)
    elif fault == 'missing-scope': del values['PRODUCT_SCOPE']
    elif fault == 'reconstruction-scope': values['PRODUCT_SCOPE'] = 'reconstruction21'
    elif fault == 'unknown-scope': values['PRODUCT_SCOPE'] = 'nmi23'
    elif fault == 'float-wheel': values['PRODUCT_MEMBERS'] = 118.0
    else: values['SDK_MEMBERS'] = 67.0
    with pytest.raises(RuntimeError):
        p.qualification_scope_text('\n'.join(k + '=' + repr(v) for k, v in values.items()))


@pytest.fixture(scope='module')
def nmi_source_records():
    """Frozen source text only; never import/execute any scientific fixture."""
    return {name: (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
            for name in p.NMI_IMPORTED_SOURCE_PINS} | {
                'tools/public_qualification.py': ('NMI_IMPORTED_SOURCE_PINS=' + repr(p.NMI_IMPORTED_SOURCE_PINS) + '\n').encode()}


def nmi_source_archive(source):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, raw in source.items(): archive.writestr(name, raw)
    return zipfile.ZipFile(io.BytesIO(stream.getvalue()))


def test_actual_nmi_pins_and_closed_scope_match_qualification(nmi_source_records):
    text = (ROOT / 'tools/public_qualification.py').read_text(encoding='utf-8')
    declarations = [ast.literal_eval(node.value) for node in ast.parse(text).body
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                    and t.id == 'NMI_IMPORTED_SOURCE_PINS' for t in node.targets)]
    assert declarations == [p.NMI_IMPORTED_SOURCE_PINS] and len(declarations[0]) == 8
    scope = p.qualification_scope(ROOT)
    assert (scope['name'], scope['members'], scope['sdk'], len(scope['families'])) == ('current241', 159, 71, 25)
    # Historical literal envelopes remain independently closed.
    legacy=p.qualification_scope_text(text.replace("CURRENT_PRODUCT_SCOPE = 'current241'\n",''))
    assert (legacy['name'],legacy['members'],legacy['sdk'])==('h_extrema23',120,68)
    snapshot = {'files_sha256': {name: body_hash(raw) for name, raw in nmi_source_records.items()}}
    with nmi_source_archive(nmi_source_records) as archive:
        p.validate_nmi_source(archive, snapshot, scope)


@pytest.mark.parametrize('name', tuple(p.NMI_IMPORTED_SOURCE_PINS))
@pytest.mark.parametrize('fault', ('missing', 'mutated-resealed', 'forged-snapshot', 'missing-snapshot'))
def test_nmi_source_pins_reject_missing_or_resealed_imports(nmi_source_records, name, fault):
    source = dict(nmi_source_records)
    snapshot = {'files_sha256': {key: body_hash(raw) for key, raw in source.items()}}
    if fault == 'missing': del source[name]
    elif fault == 'mutated-resealed':
        source[name] += b'changed'; snapshot['files_sha256'][name] = body_hash(source[name])
    elif fault == 'forged-snapshot': snapshot['files_sha256'][name] = '0' * 64
    else: del snapshot['files_sha256'][name]
    with nmi_source_archive(source) as archive:
        with pytest.raises(RuntimeError, match='NMI source'):
            p.validate_nmi_source(archive, snapshot, {'name': 'nmi22'})


@pytest.mark.parametrize('fault', ('missing', 'empty', 'extra-key', 'wrong-hash', 'duplicate-declaration', 'duplicate-key', 'not-dict'))
def test_nmi_declaration_is_exact_and_unique(nmi_source_records, fault):
    source = dict(nmi_source_records); pins = dict(p.NMI_IMPORTED_SOURCE_PINS)
    name = next(iter(pins))
    if fault == 'empty': pins = {}
    elif fault == 'extra-key': pins['src/unqualified.cpp'] = '0' * 64
    elif fault == 'wrong-hash': pins[name] = '0' * 64
    text = 'NMI_IMPORTED_SOURCE_PINS=' + repr(pins) + '\n'
    if fault == 'missing': text = '# missing declaration\n'
    elif fault == 'duplicate-declaration': text += text
    elif fault == 'duplicate-key': text = 'NMI_IMPORTED_SOURCE_PINS={' + repr(name) + ':' + repr(pins[name]) + ',' + repr(pins)[1:] + '\n'
    elif fault == 'not-dict': text = 'NMI_IMPORTED_SOURCE_PINS=[]\n'
    source['tools/public_qualification.py'] = text.encode()
    snapshot = {'files_sha256': {key: body_hash(raw) for key, raw in source.items()}}
    with nmi_source_archive(source) as archive:
        with pytest.raises(RuntimeError, match='NMI source'):
            p.validate_nmi_source(archive, snapshot, {'name': 'nmi22'})


@pytest.mark.parametrize('newline', (b'\n', b'\r\n'))
def test_nmi_source_canonical_newlines_preserve_exact_pins(nmi_source_records, newline):
    source = {name: raw.replace(b'\n', newline) for name, raw in nmi_source_records.items()}
    snapshot = {'files_sha256': {name: body_hash(raw.replace(b'\r\n', b'\n')) for name, raw in source.items()}}
    with nmi_source_archive(source) as archive:
        p.validate_nmi_source(archive, snapshot, {'name': 'nmi22'})


def test_nmi_source_rejects_duplicate_zip_member(nmi_source_records):
    snapshot = {'files_sha256': {name: body_hash(raw) for name, raw in nmi_source_records.items()}}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, raw in nmi_source_records.items(): archive.writestr(name, raw)
        name = next(iter(p.NMI_IMPORTED_SOURCE_PINS))
        with pytest.warns(UserWarning, match='Duplicate name'):
            archive.writestr(name, nmi_source_records[name])
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
        with pytest.raises(RuntimeError, match='NMI source absent or duplicate'):
            p.validate_nmi_source(archive, snapshot, {'name': 'nmi22'})


@pytest.mark.parametrize('synthetic', ['nmi22'], indirect=True)
@pytest.mark.parametrize('name,expected_error', (
    ('src/max_tree.cpp', 'max-tree source'), ('src/reconstruction.cpp', 'reconstruction source'),
    ('src/normalized_mutual_information.cpp', 'NMI source'),
))
def test_nmi_stage_calls_new_and_inherited_source_guards_before_output(synthetic, name, expected_error):
    s = synthetic; source = dict(s.source); source[name] += b'changed'
    s.proof['source']['files_sha256'][name] = body_hash(source[name])
    with zipfile.ZipFile(s.q / 'source.zip', 'w') as archive:
        for key, raw in source.items(): archive.writestr(key, raw)
    s.reseal()
    with pytest.raises(RuntimeError, match=expected_error): run(s)
    assert not s.output.exists()


@pytest.mark.parametrize('scope', ('rank14', 'tv17', 'bregman18', 'max_tree19', 'matrix20', 'reconstruction21'))
def test_legacy_scopes_do_not_invent_nmi_requirements(scope):
    class ForbiddenArchive:
        def read(self, *args): raise AssertionError('legacy scope touched NMI source')
    p.validate_nmi_source(ForbiddenArchive(), {}, {'name': scope})


@pytest.mark.parametrize('synthetic',['h_extrema23'],indirect=True)
def test_h203_complete_combined_product_stages_without_acceptance(synthetic):
    s=synthetic
    assert run(s)['status']=='staged-unaccepted'
    sdk=json.loads((s.output/'sdk.json').read_bytes())
    assert len(sdk['SDK_sha256'])==68 and len(sdk['families'])==23 and sdk['checks']==46
    assert p.qualification_scope(s.output/'source')['name']=='h_extrema23'
    registry=json.loads((s.output/'qualification.json').read_bytes())['expected_registry']
    assert registry['image_count']==203 and registry['registered_count']==206
    assert all(registry['functions'].count(name)==1 for name in (*p.H_EXTREMA_PATHS,p.NMI_PATH))
    assert len(c.verify_wheel(s.output/'wheels/scikit-imagexx.whl')['members'])==120


@pytest.mark.parametrize('synthetic',['h_extrema23'],indirect=True)
@pytest.mark.parametrize('fault',('missing-H','duplicate-H','missing-NMI','image202','SDK44'))
def test_h203_stager_rejects_incomplete_combined_scope(synthetic,fault):
    s=synthetic;expected=s.proof['expected_registry']
    if fault=='missing-H':expected['functions'].remove(p.H_EXTREMA_PATHS[0])
    elif fault=='duplicate-H':expected['functions'][0]=p.H_EXTREMA_PATHS[0]
    elif fault=='missing-NMI':expected['functions'].remove(p.NMI_PATH)
    elif fault=='image202':expected['image_count']=202
    else:s.proof['SDK_checks']=44
    s.reseal()
    with pytest.raises(RuntimeError):run(s)


@pytest.mark.parametrize('profile',('windows','modern','minimum'))
def test_current_stager_scope_uses_source_closure_and_exact241_registry(profile):
    scope=p.qualification_scope(ROOT,profile)
    assert scope['name']=='current241' and scope['members']==159 and scope['sdk']==71
    assert scope['image_count']==241 and scope['metadata_count']==3 and len(scope['functions'])==244
    assert len(scope['families'])==25 and 'current25 required' in scope['SDK_client_scope']
    assert ('lib/skimagexx.lib' in scope['SDK_members']) == (profile=='windows')
    text=(ROOT/'tools/public_qualification.py').read_text()
    with pytest.raises(RuntimeError,match='authenticated install closure'):p.qualification_scope_text(text)


def test_current_stager_rejects_old_scope_claim_with_current241_source():
    text=(ROOT/'tools/public_qualification.py').read_text(encoding='utf-8')
    assert p.CURRENT_PRODUCT_SCOPE=='current241'
    stale=text.replace("CURRENT_PRODUCT_SCOPE = 'current241'", "CURRENT_PRODUCT_SCOPE = 'current240'")
    assert stale!=text
    names=[f.relative_to(ROOT).as_posix() for prefix in ('python/skimagexx','include')
           for f in (ROOT/prefix).rglob('*') if f.is_file()]
    with pytest.raises(RuntimeError,match='authenticated install closure'):
        p.qualification_scope_text(stale,read_bytes=lambda n:(ROOT/n).read_bytes(),names=names)


@pytest.mark.parametrize('guard',('validate_reconstruction_source','validate_max_tree_source','validate_h_extrema_source','validate_nmi_source'))
def test_current_stager_never_bypasses_historical_source_guards(guard):
    class NoSource:
        def read(self,*args):raise RuntimeError('inherited source guard reached')
    with pytest.raises((RuntimeError,KeyError)):
        getattr(p,guard)(NoSource(),{},dict(name='current241'))


@pytest.mark.parametrize('declaration',("FAMILIES + ('colocalization',)", "FAMILIES + ('colocalization','footprint_shapes','unmapped')", "FAMILIES + ['colocalization','footprint_shapes']", "('replacement',)"))
def test_current_stager_rejects_missing_or_replaced_client_declaration(declaration):
    text=(ROOT/'tools/public_qualification.py').read_text()
    text=text.replace("CURRENT_FAMILIES = FAMILIES + ('colocalization', 'footprint_shapes')",'CURRENT_FAMILIES = '+declaration)
    names=[f.relative_to(ROOT).as_posix() for prefix in ('python/skimagexx','include') for f in (ROOT/prefix).rglob('*') if f.is_file()]
    with pytest.raises(RuntimeError):p.qualification_scope_text(text,read_bytes=lambda n:(ROOT/n).read_bytes(),names=names)


# Historical whole-controller pins immediately before the reviewed 5b547cfa
# metadata-only edit; R4's value remains its unchanged comparison-contract pin.
CURRENT_CORPUS_PRE_METADATA_PINS = {
    'benchmarks/rank_statistics.py': '9131a5a249e1f015f810869a1d6d209e00821b865042b07db05ec2459e3c750a',
    'benchmarks/rank_percentile.py': 'f6ca251c1bab7f9b6fba6c48ef2ab157f103874dd555ecc635f0519e67fe4faf',
    'benchmarks/rank_bilateral.py': '02fcd2043dac50f71a809291b44bc65ea9cf284da1e7f4c06560c12ecea889d6',
    'benchmarks/rank_decisions.py': 'eb050df4f682a1b28ca64449e26c627f8ef7a0de51b439dad0b7f5bbc10f63c7',
}


@pytest.mark.parametrize('name,expected',CURRENT_CORPUS_PRE_METADATA_PINS.items())
def test_current_controller_exact_inverse_sdk_metadata_matches_historical_pin(name,expected):
    canonical=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    p.validate_frozen_controller(name,canonical,expected,p.CURRENT_PRODUCT_SCOPE)
    if name=='benchmarks/rank_statistics.py':
        contract=json.loads((ROOT/'coverage/contracts/rank-statistics-comparison.json').read_text())
        assert contract['source_text_sha256'][name]==expected


@pytest.mark.parametrize('name,expected',CURRENT_CORPUS_PRE_METADATA_PINS.items())
@pytest.mark.parametrize('fault',('extra-edit','missing-import','duplicate-import','missing-call','duplicate-call','changed-call'))
def test_current_controller_inverse_rejects_any_unreviewed_edit(name,expected,fault):
    canonical=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    added=b'from tools.rank_corpus_sdk_scope import validate_sdk_scope\n'
    current=b';validate_sdk_scope(ROOT,capsule,runtime)\n'
    if fault=='extra-edit':canonical+=b'# unreviewed controller edit\n'
    elif fault=='missing-import':canonical=canonical.replace(added,b'')
    elif fault=='duplicate-import':canonical=canonical.replace(added,added+added)
    elif fault=='missing-call':canonical=canonical.replace(current,b'\n')
    elif fault=='duplicate-call':canonical=canonical.replace(current,current+current)
    else:canonical=canonical.replace(current,b';validate_sdk_scope(ROOT,capsule,other)\n')
    with pytest.raises(RuntimeError):p.validate_frozen_controller(name,canonical,expected,p.CURRENT_PRODUCT_SCOPE)


@pytest.mark.parametrize('scope',('rank14','h_extrema23'))
def test_historical_controller_hash_validation_stays_literal(scope):
    name='benchmarks/rank_statistics.py'
    current=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    historical=current.replace(b'from tools.rank_corpus_sdk_scope import validate_sdk_scope\n',b'').replace(
        b';validate_sdk_scope(ROOT,capsule,runtime)\n',b";assert len(runtime['runtime']['sdk_sha256'])==SDK_MEMBERS\n")
    expected=CURRENT_CORPUS_PRE_METADATA_PINS[name]
    p.validate_frozen_controller(name,historical,expected,scope)
    with pytest.raises(RuntimeError):p.validate_frozen_controller(name,current,expected,scope)


def test_current_inverse_is_not_applied_to_other_frozen_sources():
    raw=b'from tools.rank_corpus_sdk_scope import validate_sdk_scope\n;validate_sdk_scope(ROOT,capsule,runtime)\n'
    p.validate_frozen_controller('benchmarks/run.py',raw,hashlib.sha256(raw).hexdigest(),p.CURRENT_PRODUCT_SCOPE)
    with pytest.raises(RuntimeError):p.validate_frozen_controller('benchmarks/run.py',raw+'#change'.encode(),hashlib.sha256(raw).hexdigest(),p.CURRENT_PRODUCT_SCOPE)
