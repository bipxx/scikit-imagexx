"""Metadata-only guards for the declared original trace equivalence policy."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import struct

import pytest

ROOT = Path(__file__).resolve().parents[1]
P = runpy.run_path(str(ROOT / 'tests/reconstruction_trace_policy.py'))
MATRIX = ROOT / 'coverage/contracts/reconstruction.json'
M = json.loads(MATRIX.read_bytes())
EVIDENCE = ROOT / 'coverage/evidence/reconstruction-trace-policy-observations.json'
E = json.loads(EVIDENCE.read_bytes())
FROZEN = [(profile, case, row) for profile, value in M['profiles'].items()
          for case, row in value['observations'].items()]


def test_exact_provenance_and_complete_record_partitions():
    assert hashlib.sha256(MATRIX.read_bytes()).hexdigest() == '621981325f5e6939d3fb4d1c94e52910b369ef3ca798c326bdb93322a12b504c'
    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == '39b8a1b0e07018198550e624abd3955cf49928e87521b91a0936ca859312ed2e'
    assert len(FROZEN) == 825
    assert sum(trace['completed'] for _, _, row in FROZEN for trace in row['source_backend']) == 777
    assert len(E['observations']) == 34
    assert {r['mode'] for r in E['observations']} == {'normal', 'scalar'}
    assert len({r['case'] for r in E['observations']}) == 17
    assert E['run'] == 40189753345 and E['artifact_id'] == 12598358294
    assert E['new_calls'] == 0 and E['passed_qualification'] is False


@pytest.mark.parametrize('profile,case,row', FROZEN, ids=[p+'/'+c for p,c,_ in FROZEN])
def test_every_frozen_observation_and_completed_trace(profile, case, row):
    P['assert_observation_equivalent'](copy.deepcopy(row), row)


@pytest.mark.parametrize('record', E['observations'], ids=[r['mode']+'/'+r['case'] for r in E['observations']])
def test_actual_failed_run_observations(record):
    expected = M['profiles'][record['expected_profile']]['observations'][record['case']]
    assert record['actual'] != expected
    P['assert_observation_equivalent'](record['actual'], expected)


def actual_and_expected():
    row = E['observations'][0]
    return copy.deepcopy(row['actual']), copy.deepcopy(M['profiles']['modern']['observations'][row['case']])


def words(record):
    raw = bytes.fromhex(record['raw'])
    return list(struct.unpack('<' + str(len(raw)//4) + 'i', raw))


def put(record, values):
    record['raw'] = struct.pack('<' + str(len(values)) + 'i', *values).hex()


def traversal(trace):
    nxt = words(trace['next']); node = words(trace['start'])[0]; result = []
    while node != -1:
        assert node not in result
        result.append(node); node = nxt[node]
    return result


def relink(trace, order):
    prev = [-1] * len(order); nxt = [-1] * len(order)
    for a,b in zip(order,order[1:]):
        nxt[a] = b; prev[b] = a
    put(trace['prev'], prev); put(trace['next'], nxt); put(trace['start'], [order[0]])


@pytest.mark.parametrize('fault', [
    'out-of-range', 'negative-pointer', 'cycle', 'duplicate-node', 'second-head',
    'second-tail', 'wrong-head', 'broken-reciprocal', 'cross-rank', 'pointer-dtype',
    'start-dtype', 'bool-span', 'oversize-span', 'short-raw', 'long-raw', 'bad-hex',
    'rank-out-of-range', 'rank-after', 'strides', 'extra-trace-field', 'array-owner',
    'array-strides', 'array-shape', 'result', 'warning', 'error', 'event', 'input-state',
    'alias', 'bool-attempts', 'fake-completions', 'missing-trace', 'extra-trace',
    'incomplete', 'missing-backend',
])
def test_only_valid_equal_rank_lists_can_differ(fault):
    actual, expected = actual_and_expected(); trace = actual['source_backend'][0]
    order = traversal(trace); count = len(order)
    if fault == 'out-of-range':
        value=words(trace['next']); value[order[0]]=count; put(trace['next'],value)
    elif fault == 'negative-pointer':
        value=words(trace['next']); value[order[0]]=-2; put(trace['next'],value)
    elif fault == 'cycle':
        value=words(trace['next']); value[order[-1]]=order[0]; put(trace['next'],value)
    elif fault == 'duplicate-node':
        value=words(trace['next']); value[order[1]]=order[0]; put(trace['next'],value)
    elif fault == 'second-head':
        value=words(trace['prev']); value[order[1]]=-1; put(trace['prev'],value)
    elif fault == 'second-tail':
        value=words(trace['next']); value[order[1]]=-1; put(trace['next'],value)
    elif fault == 'wrong-head':put(trace['start'],[order[1]])
    elif fault == 'broken-reciprocal':
        value=words(trace['prev']); value[order[2]]=order[0]; put(trace['prev'],value)
    elif fault == 'cross-rank':
        rank=words(trace['ranks_before']); other=next(i for i,n in enumerate(order) if rank[n]!=rank[order[0]])
        order[0],order[other]=order[other],order[0];relink(trace,order)
    elif fault == 'pointer-dtype':trace['prev']['dtype']='>i4'
    elif fault == 'start-dtype':trace['start']['type']='uint32'
    elif fault == 'bool-span':trace['image_stride']=True
    elif fault == 'oversize-span':trace['image_stride']=50000
    elif fault == 'short-raw':trace['prev']['raw']=trace['prev']['raw'][:-8]
    elif fault == 'long-raw':trace['prev']['raw']+='00000000'
    elif fault == 'bad-hex':trace['prev']['raw']='z'+trace['prev']['raw'][1:]
    elif fault == 'rank-out-of-range':
        value=words(trace['ranks_before']);value[0]=count;put(trace['ranks_before'],value)
    elif fault == 'rank-after':trace['ranks_after']['raw']='ff'+trace['ranks_after']['raw'][2:]
    elif fault == 'strides':trace['strides']['raw']='00'+trace['strides']['raw'][2:]
    elif fault == 'extra-trace-field':trace['accepted']=True
    elif fault == 'array-owner':trace['prev']['own']=False
    elif fault == 'array-strides':trace['prev']['strides']=[-4]
    elif fault == 'array-shape':trace['prev']['shape']=[count,1]
    elif fault == 'result':actual['result']['raw']='ff'+actual['result']['raw'][2:]
    elif fault == 'warning':actual['warnings']=[]
    elif fault == 'error':actual['error']={'type':'RuntimeError','message':'wrong'}
    elif fault == 'event':actual['events']=['new']
    elif fault == 'input-state':actual['after'][0]['writeable']=False
    elif fault == 'alias':actual['aliases']['seed']=True
    elif fault == 'bool-attempts':actual['original_Cython_attempts']=True
    elif fault == 'fake-completions':actual['original_Cython_completed']=2
    elif fault == 'missing-trace':actual['source_backend']=[]
    elif fault == 'extra-trace':actual['source_backend'].append(copy.deepcopy(trace))
    elif fault == 'incomplete':trace['completed']=False
    else:actual.pop('source_backend')
    with pytest.raises(AssertionError):P['assert_observation_equivalent'](actual,expected)


@pytest.mark.parametrize('field', ['prev','next','start'])
def test_even_identically_malformed_lists_fail_independent_validation(field):
    actual, _ = actual_and_expected()
    actual['source_backend'][0][field]['raw']='00000000'
    with pytest.raises(AssertionError):P['assert_observation_equivalent'](actual,copy.deepcopy(actual))


def test_incomplete_or_absent_source_trace_remains_exact():
    actual, expected = actual_and_expected()
    for row in (actual,expected):
        row['source_backend'][0]['completed']=False
        row['original_Cython_completed']=0
    with pytest.raises(AssertionError):P['assert_observation_equivalent'](actual,expected)
    P['assert_observation_equivalent'](expected,copy.deepcopy(expected))


def test_actual_original_assertion_body_uses_policy_without_image_execution():
    path=ROOT/'tests/test_reconstruction_contracts.py'
    node=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='test_original_contract')
    node.decorator_list=[]
    row=E['observations'][0]
    namespace=dict(ROOT=ROOT,MATRIX=MATRIX,json=json,observe=lambda spec:copy.deepcopy(row['actual']),profile=lambda:'modern')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    namespace[node.name]({'id':row['case']})


def test_current_source_and_exact_two_body_inverse():
    record=P['authenticate_source'](ROOT,(ROOT/'tests/test_reconstruction_contracts.py').read_bytes())
    assert record['classification'].startswith('Scientific validation-policy successor')
    assert record['original_oracles_modified'] is False
    assert record['library_or_fixture_or_observer_modified'] is False


@pytest.mark.parametrize('fault', ['current-hash','old-hash','old-LF','unknown-policy','extra-body','missing-body','matrix','validator','observer-edit'])
def test_closed_source_migration_rejects_unknown_or_scientific_changes(tmp_path,fault):
    names=['tests/reconstruction_trace_policy.py','coverage/contracts/reconstruction.json',
           'coverage/evidence/reconstruction-public-assertion-migration.json',
           'coverage/evidence/reconstruction-trace-policy-migration.json']
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    path=tmp_path/names[-1];record=json.loads(path.read_bytes())
    current=(ROOT/'tests/test_reconstruction_contracts.py').read_bytes()
    if fault=='current-hash':record['current_raw_sha256']='0'*64
    elif fault=='old-hash':record['previous_raw_sha256']='0'*64
    elif fault=='old-LF':record['previous_canonical_LF_sha256']=record['previous_raw_sha256']
    elif fault=='unknown-policy':record['policy']='anything'
    elif fault=='extra-body':record['body_changes']['observe']=dict(previous_UTF8='x',current_UTF8='y')
    elif fault=='missing-body':record['body_changes'].pop('test_original_contract')
    elif fault=='matrix':(tmp_path/'coverage/contracts/reconstruction.json').write_bytes(b'{}')
    elif fault=='validator':(tmp_path/'tests/reconstruction_trace_policy.py').write_bytes(b'# changed')
    else:
        current=current.replace(b'def observe(s,function=None):',b'def observe(s,function=None): # unapproved')
        record['current_raw_sha256']=hashlib.sha256(current).hexdigest()
        record['current_canonical_LF_sha256']=hashlib.sha256(current.replace(b'\r\n',b'\n')).hexdigest()
    path.write_text(json.dumps(record),encoding='utf-8')
    with pytest.raises(AssertionError):P['authenticate_source'](tmp_path,current)
