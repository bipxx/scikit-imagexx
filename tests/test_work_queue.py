"""The executable plan must reject unsafe parallel ownership and invalid DAGs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools.work_queue import main, ready_packets, validate, verify_local_proofs


def packet(name, state='queued', owner=None, owns=(), depends=()):
    return dict(id=name, state=state, owner=owner, owns=list(owns),
                depends_on=list(depends), kind='private_kernel')


def ledger(*packets):
    return dict(max_active_agents=4, packets=list(packets))


def test_registration_waits_for_validated_implementation():
    doc = ledger(packet('contract', 'contracted'),
                 packet('kernel', owns=['src/new.cpp'], depends=['contract']),
                 packet('registration', owns=['python/skimagexx/measure.py'], depends=['kernel']))
    assert [p['id'] for p in ready_packets(doc)] == ['kernel']
    doc['packets'][1]['state'] = 'parity_passed'
    assert [p['id'] for p in ready_packets(doc)] == ['registration']


def proof(scope):
    return {scope: dict(status='verified', source_sha='a'*40,
                        manifest=dict(path='build/proof.json', sha256='b'*64),
                        passing_contract_checks=3, native_contract_checks=3, native_pending=0)}


def test_original_only_contract_cannot_unlock_public_registration():
    original = dict(packet('kernel', 'contracted'), proofs=proof('contract_reference'))
    registration = dict(packet('public', owns=['python/skimagexx/measure.py'], depends=['kernel']),
                        public_registration=True,
                        dependency_requirements={'kernel': {'evidence_scope': 'native_private'}})
    doc = ledger(original, registration)
    assert not ready_packets(doc)
    original['proofs'].update(proof('native_private'))
    assert [p['id'] for p in ready_packets(doc)] == ['public']
    original['proofs']['native_private']['native_pending'] = 1
    with pytest.raises(ValueError, match='zero pending'):
        validate(doc)


def test_registration_rejects_an_implicit_state_dependency():
    doc = ledger(packet('kernel', 'parity_passed'),
                 dict(packet('public', depends=['kernel']), public_registration=True))
    with pytest.raises(ValueError, match='explicit private native parity'):
        validate(doc)


def test_frozen_interface_releases_independent_work_before_kernel_completion():
    definition = 'dense_into(output, axis_costs, kind, limit, zero, one)'
    dense = dict(packet('dense', 'implementing', 'worker', ['src/dense.cpp']),
                 provided_interfaces={'v1': dict(state='frozen', definition=definition,
                    sha256=hashlib.sha256(definition.encode()).hexdigest())})
    decomposition = dict(packet('decomposition', owns=['src/decompose.cpp'], depends=['dense']),
                         dependency_requirements={'dense': {'interface': 'v1'}})
    doc = ledger(dense, decomposition)
    assert [p['id'] for p in ready_packets(doc)] == ['decomposition']
    dense['provided_interfaces']['v1']['definition'] += 'changed'
    with pytest.raises(ValueError, match='interface hash differs'):
        validate(doc)


def test_scoped_proof_can_pin_a_frozen_source_revision():
    provider = dict(packet('kernel', 'parity_passed'), proofs=proof('native_private'))
    dependent = dict(packet('consumer', owns=['src/consumer.cpp'], depends=['kernel']),
                     dependency_requirements={'kernel': {'evidence_scope': 'native_private', 'source_sha': 'c'*40}})
    doc = ledger(provider, dependent)
    assert not ready_packets(doc)
    dependent['dependency_requirements']['kernel']['source_sha'] = 'a'*40
    assert [p['id'] for p in ready_packets(doc)] == ['consumer']


def test_dependency_ready_packet_cannot_take_an_active_owned_file():
    doc = ledger(packet('active', 'implementing', 'worker', ['src/shared.cpp']),
                 packet('queued', owns=['SRC/shared.cpp']))
    assert not ready_packets(doc)


def test_verified_native_proof_requires_a_manifest_hash():
    provider = dict(packet('kernel'), proofs=proof('native_private'))
    provider['proofs']['native_private']['manifest'].pop('sha256')
    with pytest.raises(ValueError, match='hash-bearing manifest'):
        validate(ledger(provider))


def test_retained_manifest_bytes_must_match_the_recorded_proof(tmp_path):
    manifest = tmp_path/'evidence.json'
    manifest.write_bytes(b'{"native_checks":3}')
    provider = dict(packet('kernel'), proofs=proof('native_private'))
    provider['proofs']['native_private']['manifest'] = dict(
        path='evidence.json', sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    doc = ledger(provider)
    assert verify_local_proofs(doc, tmp_path) == 1
    manifest.write_bytes(b'{"native_checks":0}')
    with pytest.raises(ValueError, match='hash differs'):
        verify_local_proofs(doc, tmp_path)


@pytest.mark.parametrize('conflicting', [False, True])
def test_shared_manifest_is_read_once_but_each_digest_is_checked(tmp_path, monkeypatch, conflicting):
    manifest = tmp_path / 'evidence.json'
    manifest.write_bytes(b'{"native_checks":3}')
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    providers = [dict(packet(name), proofs=proof('native_private')) for name in ('a', 'b')]
    for provider in providers:
        provider['proofs']['native_private']['manifest'] = dict(path='evidence.json', sha256=expected)
    # Aliased spellings still identify one immutable input in this invocation.
    providers[1]['proofs']['native_private']['manifest']['path'] = './evidence.json'
    if conflicting:
        providers[1]['proofs']['native_private']['manifest']['sha256'] = '0' * 64
    original_read = Path.read_bytes
    reads = []
    def read(path):
        reads.append(path.resolve())
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    if conflicting:
        with pytest.raises(ValueError, match='hash differs'):
            verify_local_proofs(ledger(*providers), tmp_path)
    else:
        assert verify_local_proofs(ledger(*providers), tmp_path) == 1
    assert reads == [manifest.resolve()]


def test_measurement_can_require_completed_performance_dependencies():
    doc = ledger(packet('baseline', 'performance_pending'),
                 dict(packet('tune', depends=['baseline'], owns=['src/new.cpp']),
                      dependency_ready_states=['measured', 'accepted']))
    assert not ready_packets(doc)
    doc['packets'][0]['state'] = 'measured'
    assert [p['id'] for p in ready_packets(doc)] == ['tune']


def test_live_measurement_does_not_release_a_dependent_packet():
    doc = ledger(packet('measurement', 'measuring'),
                 packet('tune', owns=['src/new.cpp'], depends=['measurement']))
    assert not ready_packets(doc)


def test_inventory_names_cannot_be_dropped_or_double_assigned():
    doc = ledger(dict(packet('a'), apis=['measure.label']))
    doc.update(unassigned_inventoried_apis=['morphology.label'],
               inventory_snapshot={'public_names': 2})
    validate(doc)
    doc['unassigned_inventoried_apis'].append('measure.label')
    with pytest.raises(ValueError, match='both assigned and unassigned'):
        validate(doc)
    doc['unassigned_inventoried_apis'] = []
    with pytest.raises(ValueError, match='does not cover'):
        validate(doc)


def test_duplicate_missing_and_cyclic_dependencies_are_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        validate(ledger(packet('a'), packet('a')))
    with pytest.raises(ValueError, match='unknown dependency'):
        validate(ledger(packet('a', depends=['missing'])))
    with pytest.raises(ValueError, match='cycle'):
        validate(ledger(packet('a', depends=['b']), packet('b', depends=['a'])))


def test_windows_file_conflicts_are_rejected_even_for_same_agent():
    doc = ledger(packet('a', 'implementing', 'worker', ['src/kernel.cpp']),
                 packet('b', 'validating', 'worker', ['SRC\\KERNEL.cpp']))
    with pytest.raises(ValueError, match='active file conflict'):
        validate(doc)
    doc['packets'][0]['state'] = 'parity_passed'
    validate(doc)


def test_unassigned_or_oversubscribed_active_work_is_rejected():
    with pytest.raises(ValueError, match='needs an owner'):
        validate(ledger(packet('a', 'contracting')))
    doc = ledger(*(packet(str(i), 'implementing', f'worker{i}') for i in range(4)))
    with pytest.raises(ValueError, match='agent slots'):
        validate(doc)


@pytest.mark.parametrize('path', ['../source.cpp', '/tmp/source.cpp', 'C:\\source.cpp'])
def test_ownership_cannot_escape_workspace(path):
    with pytest.raises(ValueError, match='escapes workspace'):
        validate(ledger(packet('a', owns=[path])))


def test_inspection_never_mutates_the_ledger():
    doc = ledger(packet('a', owns=['src/a.cpp']))
    before = deepcopy(doc)
    ready_packets(doc)
    assert doc == before


@pytest.mark.parametrize('branch', [None, 'fix/recorded-source'])
def test_agents_reports_recorded_branch_or_explicit_unknown(tmp_path, capsys, branch):
    row = packet('legacy', 'preparing', 'worker')
    if branch is not None:
        row['branch'] = branch
    document = ledger(row)
    document['execution_policy'] = dict(
        implementation_goal_status='paused',
        current_three_agent_packets={'worker': 'legacy'},
    )
    path = tmp_path / 'ledger.json'
    path.write_text(json.dumps(document), encoding='utf-8')
    before = path.read_bytes()

    assert main(['--ledger', str(path), '--agents']) == 0

    report = json.loads(capsys.readouterr().out)
    assert report['live_process_query'] is False
    assert report['implementation_goal_status'] == 'paused'
    assert report['agents'] == [dict(agent='worker', packet='legacy',
                                    state='preparing', branch=branch,
                                    queue=['legacy'])]
    assert path.read_bytes() == before
    assert not ready_packets(document)


def test_actual_recorded_agents_match_optional_metadata_without_mutation(capsys):
    path = Path(__file__).resolve().parents[1] / 'coverage/work_items.json'
    before = path.read_bytes()
    document = json.loads(before)
    packets = {row['id']: row for row in document['packets']}

    assert main(['--ledger', str(path), '--agents']) == 0

    report = json.loads(capsys.readouterr().out)
    assignments = document['execution_policy']['current_three_agent_packets']
    assert {row['agent']: row['packet'] for row in report['agents']} == assignments
    for row in report['agents']:
        recorded = packets[row['packet']]
        assert row['branch'] == recorded.get('branch')
        assert row['state'] == recorded['state']
    assert report['live_process_query'] is False
    assert path.read_bytes() == before


def test_actual_superseded_matrix_binding_is_not_ready_and_evidence_is_unchanged():
    root = Path(__file__).resolve().parents[1]
    document = json.loads((root / 'coverage/work_items.json').read_text())
    before = deepcopy(document)
    by_id = validate(document)
    stale = by_id['matrix-transform-general-binding']
    assert stale['state'] == 'queued'
    assert stale['superseded_by_active_packet'] == 'matrix-transform-profile-binding'
    assert stale not in ready_packets(document)
    assert document == before


def test_superseded_packet_does_not_hide_an_ordinary_ready_packet():
    stale = dict(packet('old', owns=['src/old.cpp']), superseded_by_active_packet='replacement')
    document = ledger(stale, packet('replacement', 'parity_passed'),
                      packet('next', owns=['src/next.cpp'], depends=['replacement']))
    assert [p['id'] for p in ready_packets(document)] == ['next']


@pytest.mark.parametrize('target', ('missing', 'old', None))
def test_invalid_supersession_target_is_rejected(target):
    stale = dict(packet('old', owns=['src/old.cpp']), superseded_by_active_packet=target)
    with pytest.raises(ValueError, match='invalid supersession target'):
        validate(ledger(stale))
