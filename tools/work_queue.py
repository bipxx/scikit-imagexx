"""Inspect the parallel work ledger without editing source or launching jobs."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[1]
STATES = {
    'queued', 'contracting', 'contracted', 'implementing', 'validating',
    'parity_passed', 'performance_pending', 'preparing', 'measuring',
    'measured', 'accepted',
}
ACTIVE = {'contracting', 'implementing', 'validating'}
DEPENDENCY_READY = {
    'contracted', 'parity_passed', 'performance_pending', 'measured', 'accepted',
}
PROOF_SCOPES = {
    'contract_reference', 'native_private', 'public_native',
    'tooling', 'isolated_performance',
}


def dependency_ready(packet, dependency, by_id):
    requirement = packet.get('dependency_requirements', {}).get(dependency, {})
    provider = by_id[dependency]
    if 'interface' in requirement:
        interface = provider.get('provided_interfaces', {}).get(requirement['interface'], {})
        return interface.get('state') == 'frozen'
    if 'evidence_scope' in requirement:
        proof = provider.get('proofs', {}).get(requirement['evidence_scope'], {})
        return (proof.get('status') == 'verified'
                and (not requirement.get('source_sha')
                     or proof.get('source_sha') == requirement['source_sha']))
    return provider['state'] in packet.get('dependency_ready_states', DEPENDENCY_READY)


def validate(document):
    packets = document['packets']
    by_id = {}
    for packet in packets:
        name = packet['id']
        if not name or name in by_id:
            raise ValueError(f'duplicate or empty packet id: {name!r}')
        by_id[name] = packet
        if packet['state'] not in STATES:
            raise ValueError(f'{name}: unknown state {packet["state"]!r}')
        for owned in packet['owns']:
            path = PurePosixPath(owned.replace('\\', '/'))
            if path.is_absolute() or '..' in path.parts or ':' in str(path):
                raise ValueError(f'{name}: ownership path escapes workspace: {owned}')
        if packet['state'] in ACTIVE and not packet.get('owner'):
            raise ValueError(f'{name}: active packet needs an owner')
        requirements = packet.get('dependency_requirements', {})
        if set(requirements) - set(packet['depends_on']):
            raise ValueError(f'{name}: proof requirement is not a declared dependency')
        for dependency, requirement in requirements.items():
            if ('interface' in requirement) == ('evidence_scope' in requirement):
                raise ValueError(f'{name}: dependency {dependency} needs exactly one proof scope or interface')
            if 'evidence_scope' in requirement and requirement['evidence_scope'] not in PROOF_SCOPES:
                raise ValueError(f'{name}: unknown dependency proof scope')
        if packet.get('public_registration'):
            if not packet['depends_on'] or any(
                requirements.get(dep, {}).get('evidence_scope') != 'native_private'
                for dep in packet['depends_on']
            ):
                raise ValueError(f'{name}: public registration requires explicit private native parity proofs')
        if packet['kind'] == 'measurement' and packet['depends_on']:
            if not any(r.get('evidence_scope') == 'public_native' for r in requirements.values()):
                raise ValueError(f'{name}: measurement requires a public native proof')
        for version, interface in packet.get('provided_interfaces', {}).items():
            if interface.get('state') == 'frozen':
                digest = hashlib.sha256(interface['definition'].encode('utf-8')).hexdigest()
                if digest != interface.get('sha256'):
                    raise ValueError(f'{name}: frozen interface hash differs: {version}')
        for scope, proof in packet.get('proofs', {}).items():
            if scope not in PROOF_SCOPES or proof.get('status') not in ('pending', 'verified'):
                raise ValueError(f'{name}: invalid proof scope or status: {scope}')
            if proof['status'] != 'verified':
                continue
            manifest = proof.get('manifest', {})
            if (not re.fullmatch('[0-9a-f]{40}', proof.get('source_sha', ''))
                    or not manifest.get('path')
                    or not re.fullmatch('[0-9a-f]{64}', manifest.get('sha256', ''))):
                raise ValueError(f'{name}: verified proof needs a source SHA and hash-bearing manifest')
            if scope in ('native_private', 'public_native'):
                if proof.get('native_contract_checks', 0) <= 0 or proof.get('native_pending') != 0:
                    raise ValueError(f'{name}: native proof needs executed contract checks and zero pending cases in its scope')
            if scope == 'contract_reference' and proof.get('passing_contract_checks', 0) <= 0:
                raise ValueError(f'{name}: contract proof needs passing reference checks')
    visited, visiting = set(), set()

    def walk(name):
        if name in visiting:
            raise ValueError(f'dependency cycle at {name}')
        if name in visited:
            return
        visiting.add(name)
        for dependency in by_id[name]['depends_on']:
            if dependency not in by_id:
                raise ValueError(f'{name}: unknown dependency {dependency}')
            walk(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in by_id:
        walk(name)
    active = [p for p in packets if p['state'] in ACTIVE]
    owners = {p['owner'] for p in active if p['owner'] != 'integrator'}
    if len(owners) > document['max_active_agents'] - 1:
        raise ValueError('active worker owners exceed the available agent slots')
    paths = {}
    for packet in active:
        for owned in packet['owns']:
            key = owned.replace('\\', '/').casefold()
            if key in paths:
                raise ValueError(f'active file conflict: {owned}: {paths[key]} and {packet["id"]}')
            paths[key] = packet['id']
    if 'unassigned_inventoried_apis' in document:
        assigned = [name for p in packets for name in p.get('apis', [])]
        unassigned = document['unassigned_inventoried_apis']
        if len(set(assigned)) != len(assigned) or len(set(unassigned)) != len(unassigned):
            raise ValueError('duplicate API assignment in inventory ledger')
        if set(assigned) & set(unassigned):
            raise ValueError('API is both assigned and unassigned')
        expected = document.get('inventory_snapshot', {}).get('public_names')
        if expected is not None and len(assigned) + len(unassigned) != expected:
            raise ValueError('inventory accounting does not cover the public name count')
    return by_id


def ready_packets(document):
    by_id = validate(document)
    occupied = {path.replace('\\', '/').casefold()
                for p in document['packets'] if p['state'] in ACTIVE for path in p['owns']}
    return [p for p in document['packets']
            if p['state'] == 'queued'
            and (p['owns'] or p['kind'] == 'measurement')
            and not occupied.intersection(path.replace('\\', '/').casefold() for path in p['owns'])
            and all(dependency_ready(p, dep, by_id) for dep in p['depends_on'])]


def verify_local_proofs(document, root=ROOT):
    """Verify retained manifest bytes when running in the evidence workspace."""
    checked = set()
    for packet in document['packets']:
        for proof in packet.get('proofs', {}).values():
            if proof['status'] != 'verified':
                continue
            manifest = proof['manifest']
            path = Path(root)/manifest['path']
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as error:
                raise ValueError(f'retained proof is unavailable: {path}') from error
            if digest != manifest['sha256']:
                raise ValueError(f'retained proof hash differs: {path}')
            checked.add(manifest['path'])
    return len(checked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=ROOT/'coverage/work_items.json')
    parser.add_argument('--verify-local-proofs', action='store_true',
                        help='Also hash retained manifests; requires the original evidence workspace')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true')
    action.add_argument('--ready', action='store_true')
    action.add_argument('--agents', action='store_true', help='Print the recorded three-agent roster; not a live process query')
    action.add_argument('--packet', help='Print an agent handoff, without launching it')
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.ledger.read_text(encoding='utf-8-sig'))
        by_id = validate(document)
        if args.verify_local_proofs:
            verify_local_proofs(document)
        if args.check:
            print(f'Validated {len(by_id)} packets: no dependency cycles or active file conflicts.')
        elif args.agents:
            print(json.dumps(document.get('parallel_review', {}), indent=2))
        elif args.packet:
            if args.packet not in by_id:
                raise ValueError(f'unknown packet: {args.packet}')
            packet = by_id[args.packet]
            handoff = {k: packet.get(k) for k in (
                'id', 'state', 'owner', 'branch', 'base_sha', 'owns',
                'depends_on', 'dependency_requirements', 'provided_interfaces',
                'read_dependencies', 'build', 'launch', 'acceptance',
            )}
            handoff['implementation_goal_status'] = document.get('execution_policy', {}).get('implementation_goal_status', 'unspecified')
            print(json.dumps(handoff, indent=2))
        else:
            ready = {p['id'] for p in ready_packets(document)}
            selected = [p for p in document['packets'] if not args.ready or p['id'] in ready]
            print(json.dumps(dict(
                implementation_goal_status=document.get('execution_policy', {}).get('implementation_goal_status', 'unspecified'),
                note='Dependency readiness does not resume a paused goal or allocate an agent slot.',
                packets=[dict(id=p['id'], state=p['state'], owner=p['owner'],
                              dependency_ready=p['id'] in ready,
                              depends_on=p['depends_on'], owns=p['owns'])
                         for p in selected]), indent=2))
        return 0
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Invalid work ledger: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
