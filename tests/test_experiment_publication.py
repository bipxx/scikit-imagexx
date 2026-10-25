"""Metadata transport/admission only; no image imports, builds, or timers."""
import base64
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publication_under_test', ROOT / 'tools/experiment_publication.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
# Synthetic certificates use the current compatible controller snapshot;
# previously published experiment bytes and controller revisions stay frozen.
REVISION = '9340b1cd4232d89676335c301a41817d34280db9'


@pytest.fixture(autouse=True)
def no_numerical_imports():
    attempts = []
    class Block:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('numpy', 'scipy', 'skimage', 'skimagexx'):
                attempts.append(fullname)
                raise AssertionError('Numerical import in metadata publication')
    blocker = Block()
    sys.meta_path.insert(0, blocker)
    yield
    sys.meta_path.remove(blocker)
    assert not attempts


def transport(raw):
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode(), hashlib.sha256(raw).hexdigest()


def encoded_document(value):
    raw = (json.dumps(value, indent=2) + '\n').encode()
    return raw, transport(raw)


@pytest.fixture
def rank_certificate():
    helper = p._helper(p.RANK_SOURCE)
    scope = dict(image_paths=192, metadata_paths=3, sdk_families=1,
                 sdk_members=63, sdk_family_names=['SYNTHETIC-metadata-only'])
    products = {}
    for role, revision in [('before', '1' * 40), ('after', '2' * 40)]:
        products[role] = {}
        for profile in ('windows', 'modern', 'minimum'):
            products[role][profile] = dict(
                revision=revision, qualification_sha256='3' * 64,
                product_sha256='4' * 64, wheel_sha256='5' * 64,
                source_sha256='6' * 64, scope=copy.deepcopy(scope),
                artifact_run='111', artifact_name='rank-statistics-product-' + profile)
    return dict(schema=p.RANK_SCHEMA, template_sha256=helper.text_sha(helper.CONTRACT),
                accepted_products=products)


@pytest.fixture
def tv_certificate(monkeypatch):
    # Existing validator with one synthetic bridge digest, never real acceptance.
    a = p._helper(p.TV_SOURCE)
    bridge = {'SYNTHETIC': 'metadata guard fixture'}
    monkeypatch.setattr(a, 'BEFORE_BRIDGE_DIGEST', a.digest(bridge))
    scope = dict(image_paths=192, metadata_paths=3, sdk_families=18, sdk_members=63,
                 sdk_family_names=['family' + str(i) for i in range(18)])
    products = {}
    for role, source in [('before', a.BEFORE_PRODUCT), ('after', a.AFTER_PRODUCT)]:
        products[role] = {}
        for profile in a.PROFILES:
            pin = dict(revision=source, qualification_sha256='1' * 64,
                       product_sha256='2' * 64, wheel_sha256='3' * 64,
                       source_sha256='4' * 64, scope=copy.deepcopy(scope),
                       artifact_run='111', artifact_name='rank-statistics-product-' + profile)
            products[role][profile] = dict(pin=pin, core_sha256='5' * 64,
                artifact=dict(id='123', digest='6' * 64, run='111', name=pin['artifact_name']))
    return dict(schema=p.TV_SCHEMA, status='root-reviewed', controller_revision=REVISION,
        before_product_source=a.BEFORE_PRODUCT, intended_before_source=a.BEFORE_ANCHOR,
        after_product_source=a.AFTER_PRODUCT, original_version='0.26.0', rounds=7,
        workers=1, case_count=400, timed_count=70, factory_sha256=a.FROZEN_FACTORY_SHA256,
        manifest_sha256=a.sha(ROOT / 'coverage/contracts/tv-chambolle-benchmark.json'),
        root_after_admission=dict(source=a.AFTER_PRODUCT, passed=True, sha256='7' * 64),
        products=products, approved_after_source_sha256={
            n: a.PRIVATE_HEADER_SHA256 if n == a.PRIVATE_HEADER else '8' * 64
            for n in a.ALLOWED_DELTA}, before_source_equivalence=bridge)


@pytest.mark.parametrize('fixture_name', ['rank_certificate', 'tv_certificate'])
def test_known_validators_publish_only_exact_bytes(tmp_path, request, fixture_name):
    value = request.getfixturevalue(fixture_name)
    raw, (encoded, digest) = encoded_document(value)
    result = p.publish(encoded, digest, REVISION, tmp_path / 'artifact')
    target = tmp_path / 'artifact/experiment.json'
    assert target.read_bytes() == raw
    assert result == dict(schema=value['schema'], sha256=digest, bytes=len(raw),
                         controller_revision=REVISION, path=str(target))
    assert [x.name for x in target.parent.iterdir()] == ['experiment.json']
    with pytest.raises(RuntimeError, match='already exists'):
        p.publish(encoded, digest, REVISION, target.parent)
    assert target.read_bytes() == raw


@pytest.mark.parametrize('mutation', ['draft', 'revision', 'workers', 'scope', 'profile', 'unknownfield'])
def test_tv_existing_semantic_gate_rejects_before_output(tmp_path, monkeypatch, tv_certificate, mutation):
    monkeypatch.setattr(p, 'verify_controller_sources', lambda *args: None)
    value = tv_certificate
    if mutation == 'draft': value['status'] = 'draft-root-review-pending'
    elif mutation == 'revision': value['controller_revision'] = 'a' * 40
    elif mutation == 'workers': value['workers'] = True
    elif mutation == 'scope': value['products']['before']['modern']['pin']['scope']['image_paths'] = 199
    elif mutation == 'profile': value['products']['after'].pop('minimum')
    else: value['validator_module'] = 'os'
    _, (encoded, digest) = encoded_document(value)
    with pytest.raises(RuntimeError): p.publish(encoded, digest, REVISION, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('mutation', ['template', 'roles', 'profile', 'scope', 'unknownfield'])
def test_rank_existing_semantic_gate_and_private_staging_cleanup(tmp_path, monkeypatch, rank_certificate, mutation):
    monkeypatch.setattr(p, 'verify_controller_sources', lambda *args: None)
    value = rank_certificate
    if mutation == 'template': value['template_sha256'] = '0' * 64
    elif mutation == 'roles': value['accepted_products']['other'] = {}
    elif mutation == 'profile': value['accepted_products']['after'].pop('minimum')
    elif mutation == 'scope': value['accepted_products']['after']['modern']['scope']['image_paths'] += 1
    else: value['validator_path'] = '../../arbitrary.py'
    _, (encoded, digest) = encoded_document(value)
    with pytest.raises(RuntimeError): p.publish(encoded, digest, REVISION, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('bad', [None, True, 1, b'eA==', '', 'x' * 40001, '%%%%', 'not base64\n'],
                         ids=['null', 'boolean', 'integer', 'bytes', 'empty', 'oversized', 'symbols', 'newline'])
def test_transport_rejects_encoded_types_and_bounds(bad):
    with pytest.raises(RuntimeError): p.authenticate_payload(bad, '0' * 64)


@pytest.mark.parametrize('bad', [None, True, 1, b'0' * 64, '', '0' * 63, '0' * 65, 'A' * 64, 'g' * 64])
def test_transport_rejects_sha_types_and_formats(bad):
    encoded, _ = transport(b'{}')
    with pytest.raises(RuntimeError): p.authenticate_payload(encoded, bad)


@pytest.mark.parametrize('mutation', ['truncate', 'concatenate', 'trailing', 'crc', 'wronghash', 'nongzip', 'compressed_over'])
def test_rejects_ambiguous_or_tampered_gzip(mutation):
    raw = b'{"schema":"metadata"}'
    packed = gzip.compress(raw, mtime=0)
    digest = hashlib.sha256(raw).hexdigest()
    if mutation == 'truncate': packed = packed[:-1]
    elif mutation == 'concatenate': packed += gzip.compress(b'{}', mtime=0)
    elif mutation == 'trailing': packed += b'\0'
    elif mutation == 'crc': packed = packed[:-8] + bytes([packed[-8] ^ 1]) + packed[-7:]
    elif mutation == 'wronghash': digest = '0' * 64
    elif mutation == 'nongzip': packed = b'not gzip'
    else: packed = b'0' * 30001
    with pytest.raises(RuntimeError):
        p.authenticate_payload(base64.b64encode(packed).decode(), digest)


def test_encoded_and_compressed_maximum_are_inclusive():
    raw = b'{}'
    packed = gzip.compress(raw, mtime=0)
    extra = 30000 - len(packed) - 2
    packed = packed[:3] + bytes([packed[3] | 4]) + packed[4:10] + extra.to_bytes(2, 'little') + b'x' * extra + packed[10:]
    encoded = base64.b64encode(packed).decode()
    assert len(packed) == p.MAX_COMPRESSED == 30000
    assert len(encoded) == p.MAX_ENCODED == 40000
    assert p.authenticate_payload(encoded, hashlib.sha256(raw).hexdigest()) == raw


def test_raw_maximum_is_inclusive_and_bomb_is_bounded():
    raw = b'x' * (256 * 1024)
    assert len(raw) == p.MAX_RAW
    assert p.authenticate_payload(*transport(raw)) == raw
    with pytest.raises(RuntimeError, match='oversized'):
        p.authenticate_payload(*transport(raw + b'x'))


def test_hash_is_checked_before_json_or_validator(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'validate_certificate', lambda *args: pytest.fail('validator reached before SHA'))
    encoded, _ = transport(b'not JSON or a certificate')
    with pytest.raises(RuntimeError, match='raw SHA differs'):
        p.publish(encoded, '0' * 64, REVISION, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('raw', [b'{}', b'[]', b'null', b'{"schema":false}',
    b'{"schema":"os"}', b'{"schema":"../../arbitrary.py"}',
    b'{"schema":"x","schema":"y"}', b'{"schema":"x","n":NaN}',
    b'{"schema":"x","n":Infinity}', b'{"schema":"x","n":1e999}', b'\xff'])
def test_unknown_schema_and_ambiguous_json_never_load_a_helper(tmp_path, monkeypatch, raw):
    monkeypatch.setattr(p, '_helper', lambda *args: pytest.fail('unregistered helper loaded'))
    with pytest.raises((RuntimeError, ValueError)):
        p.publish(*transport(raw), REVISION, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('revision', [None, True, 1, b'0' * 40, '', 'a' * 39, 'a' * 41, 'A' * 40, 'x;echo payload'])
def test_revision_type_and_hex_checked_before_git(monkeypatch, revision):
    monkeypatch.setattr(p.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Git reached'))
    with pytest.raises(RuntimeError, match='revision'):
        p.verify_controller_sources(p.RANK_SCHEMA, revision)


def test_controller_dependency_change_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p.subprocess, 'run', lambda *args, **kwargs: None)
    calls = []
    def show(argv, **kwargs):
        calls.append(argv)
        return b'expected\n'
    monkeypatch.setattr(p.subprocess, 'check_output', show)
    path = tmp_path / p.RANK_SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(b'changed\r\n')
    with pytest.raises(RuntimeError, match='dependency differs'):
        p.verify_controller_sources(p.RANK_SCHEMA, REVISION)
    assert calls[0][-1] == REVISION + ':' + p.RANK_SOURCE


def test_cli_reads_event_without_logging_payload_or_data_derived_errors(tmp_path, monkeypatch, capsys):
    inputs = dict(payload_gzip_base64='DO-NOT-LOG-TRANSPORT',
                  payload_sha256='0' * 64, controller_revision=REVISION)
    event_path = tmp_path / 'event.json'
    event_path.write_text(json.dumps(dict(inputs=inputs, repository={'name': 'test'})))
    monkeypatch.setenv('GITHUB_EVENT_PATH', str(event_path))
    calls = []
    def fail(*args):
        calls.append(args)
        raise RuntimeError('DO-NOT-LOG-VALIDATOR-DETAIL')
    monkeypatch.setattr(p, 'publish', fail)
    assert p.main() == 1
    assert calls == [('DO-NOT-LOG-TRANSPORT', '0' * 64, REVISION, ROOT / 'build/reviewed-experiment')]
    output = capsys.readouterr()
    assert output.out == ''
    assert output.err == 'Experiment publication failed validation.\n'


def test_workflow_uploads_one_file_with_readonly_permissions():
    source = (ROOT / '.github/workflows/experiment-publication.yml').read_text()
    assert 'contents: read' in source and 'actions: read' in source and ': write' not in source
    assert 'name: reviewed-experiment' in source
    assert 'path: build/reviewed-experiment/experiment.json' in source
    assert source.count('actions/upload-artifact@') == 1
    assert "['git', 'fetch', '--no-tags', '--depth=1', 'origin', revision]" in source
    assert "re.fullmatch('[0-9a-f]{40}', revision)" in source
    assert 'python tools/experiment_publication.py' in source
    assert 'echo ' not in source and '${{ inputs.payload' not in source
    assert 'PAYLOAD_GZIP_BASE64:' not in source


@pytest.fixture
def positive_zero_certificate():
    # Reuse the existing synthetic certificate; its fabricated AFTER frame
    # remains test-only and never becomes the real external experiment.
    spec = importlib.util.spec_from_file_location('positive_zero_publication_fixture',
        ROOT / 'tests/test_tv_positive_zero_compare.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    return fixtures.experiment.__wrapped__()


def test_positive_zero_fixed_registry_covers_its_frozen_dependencies():
    helper = p._helper(p.TV_POSITIVE_ZERO_SOURCE)
    expected = set(helper.PLAN['dependency_pins']) | {
        p.TV_POSITIVE_ZERO_SOURCE, 'benchmarks/tv_positive_zero_compare.py',
        'tests/test_tv_positive_zero_compare.py',
        '.github/workflows/tv-positive-zero-compare.yml', helper.PLAN_PATH}
    assert set(p.CONTROLLER_INPUTS[p.TV_POSITIVE_ZERO_SCHEMA]) == expected
    assert len(p.CONTROLLER_INPUTS[p.TV_POSITIVE_ZERO_SCHEMA]) == len(expected) == 16
    assert set(p.VALIDATORS) == {p.RANK_SCHEMA, p.TV_SCHEMA, p.TV_POSITIVE_ZERO_SCHEMA}


def test_positive_zero_publishes_exact_bytes_through_existing_validator(
        tmp_path, monkeypatch, positive_zero_certificate):
    calls = []
    monkeypatch.setattr(p, 'verify_controller_sources', lambda *args: calls.append(args))
    raw, (encoded, digest) = encoded_document(positive_zero_certificate)
    revision = positive_zero_certificate['controller_revision']
    result = p.publish(encoded, digest, revision, tmp_path)
    assert calls == [(p.TV_POSITIVE_ZERO_SCHEMA, revision)]
    assert result['schema'] == p.TV_POSITIVE_ZERO_SCHEMA
    assert (tmp_path / 'experiment.json').read_bytes() == raw
    assert [x.name for x in tmp_path.iterdir()] == ['experiment.json']


@pytest.mark.parametrize('mutation', ['draft', 'wrongcontroller', 'failed741', 'missingdebug'])
def test_positive_zero_semantic_rejection_precedes_publication(
        tmp_path, monkeypatch, positive_zero_certificate, mutation):
    monkeypatch.setattr(p, 'verify_controller_sources', lambda *args: None)
    value = positive_zero_certificate
    revision = value['controller_revision']
    if mutation == 'draft': value['status'] = 'draft-root-review-pending'
    elif mutation == 'wrongcontroller': value['controller_revision'] = 'd' * 40
    elif mutation == 'failed741': value['root_after_admission']['run'] = '40200919182'
    else: value['root_after_admission']['profiles'].pop('debug')
    raw, (encoded, digest) = encoded_document(value)
    with pytest.raises(RuntimeError):
        p.publish(encoded, digest, revision, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_positive_zero_workflow_keeps_complete_preparation_barrier_and_isolated_measurement():
    source = (ROOT / '.github/workflows/tv-positive-zero-compare.yml').read_text()
    assert 'default: reviewed-experiment' in source
    assert 'source_revision:' in source and "os.environ['GITHUB_SHA'] != expected" in source
    assert "['git', 'rev-parse', 'HEAD']" in source
    assert 'contents: read' in source and 'actions: read' in source and ': write' not in source
    preparation = source.split('  prepare:\n', 1)[1].split('  barrier:\n', 1)[0]
    barrier = source.split('  barrier:\n', 1)[1].split('  measure:\n', 1)[0]
    measurement = source.split('  measure:\n', 1)[1]
    for profile in ('windows', 'modern', 'minimum'):
        assert 'profile: ' + profile in preparation and 'profile: ' + profile in measurement
        assert 'build/receipts/tv-positive-zero-receipt-' + profile + '/preparation.json' in barrier
        assert 'build/receipts/tv-positive-zero-receipt-' + profile + '/preparation.json' in measurement
    assert 'needs: prepare' in barrier and 'needs: [prepare, barrier]' in measurement
    assert 'compare --check-only' in preparation and '--check-only' not in measurement
    assert 'Legacy400 plus fresh default70' in preparation
    assert 'one worker seven paired triads' in measurement
    assert 'pip install --no-index --no-deps --require-hashes --only-binary=:all: --no-compile' in measurement
    assert 'pytest' not in measurement.split('      - name: Fresh host', 1)[0]
    assert 'cmake' not in measurement and 'pip wheel' not in measurement
    assert 'if: always()' in preparation and 'if: always()' in measurement
    assert 'build/results/' in measurement and 'build/experiment/' in measurement
