"""Closed validation of original reconstruction's unspecified equal-rank order.

This changes no array operation, fixture, oracle or numerical tolerance. Only
the three raw linked-list words may differ after both recorded lists validate.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import struct

POLICY = 'reconstruction-original-equal-rank-list-v1'
MAX_SITES = 100000  # The unchanged original-entry safety observer requires <100000.
RAW_FIELDS = ('prev', 'next', 'start')
_ARRAY_FIELDS = {'type', 'dtype', 'metadata', 'shape', 'strides', 'raw', 'own',
                 'writeable', 'C', 'F', 'base'}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact(left, right):
    """JSON comparison with exact types: bool is never an integer substitute."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return type(left) in (str, int, float, bool, type(None)) and left == right


def _raw_words(raw, code, count):
    require(type(raw) is str and len(raw) == count * 8, 'typed raw byte length differs')
    require(all(c in '0123456789abcdef' for c in raw), 'noncanonical raw hex')
    return struct.unpack('<' + str(count) + code, bytes.fromhex(raw))


def _array_words(row, signed, count):
    require(type(row) is dict and set(row) == _ARRAY_FIELDS, 'integer array record fields differ')
    metadata = dict(type='ndarray', dtype='<i4' if signed else '<u4', metadata={},
                    shape=[count], strides=[4], own=True, writeable=True,
                    C=True, F=True, base=None)
    require(exact({k: v for k, v in row.items() if k != 'raw'}, metadata),
            'source integer array metadata differs')
    return _raw_words(row['raw'], 'i' if signed else 'I', count)


def validate_completed_trace(trace):
    """Return the source-sorted rank sequence after a full bounded-list proof."""
    require(type(trace) is dict and set(trace) == {
        'ranks_before', 'prev', 'next', 'strides', 'start', 'image_stride',
        'completed', 'ranks_after'}, 'completed source trace fields differ')
    require(trace['completed'] is True, 'only completed original traces may use tie equivalence')
    size = trace['image_stride']
    require(type(size) is int and 0 < size and 2 * size < MAX_SITES, 'source rank span outside safety scope')
    count = 2 * size
    ranks = _array_words(trace['ranks_before'], False, count)
    _array_words(trace['ranks_after'], False, count)
    require(all(r < count for r in ranks), 'rank outside source rank-order range')
    prev = _array_words(trace['prev'], True, count)
    nxt = _array_words(trace['next'], True, count)
    start = trace['start']
    require(type(start) is dict and set(start) == {'type', 'dtype', 'raw'} and
            start['type'] == 'int32' and start['dtype'] == '<i4', 'source start scalar metadata differs')
    node = _raw_words(start['raw'], 'i', 1)[0]
    require(0 <= node < count and all(-1 <= p < count for p in prev + nxt), 'pointer outside recorded span')
    require(prev.count(-1) == nxt.count(-1) == 1, 'list must have exactly one head and tail')
    seen = set()
    sequence = []
    previous = -1
    while node != -1:
        require(node not in seen and prev[node] == previous, 'cycle or nonreciprocal previous pointer')
        seen.add(node)
        sequence.append(ranks[node])
        previous, node = node, nxt[node]
    require(len(seen) == count, 'linked list is not a complete permutation')
    require(all(a >= b for a, b in zip(sequence, sequence[1:])), 'list crosses source rank order')
    return sequence


def _trace_without_order(trace):
    return {key: ({k: v for k, v in value.items() if k != 'raw'}
                  if key in RAW_FIELDS and type(value) is dict else value)
            for key, value in trace.items()}


def _check_observation(actual, expected):
    require(type(actual) is type(expected) is dict, 'observation must be a plain record')
    require(exact({k: v for k, v in actual.items() if k != 'source_backend'},
                  {k: v for k, v in expected.items() if k != 'source_backend'}),
            'public observation or source completion count differs')
    require('source_backend' in actual and 'source_backend' in expected, 'source backend record absent')
    left, right = actual['source_backend'], expected['source_backend']
    require(type(left) is type(right) is list and len(left) == len(right), 'source trace count differs')
    for observation, traces in ((actual, left), (expected, right)):
        require(type(observation['original_Cython_attempts']) is int and
                observation['original_Cython_attempts'] == len(traces), 'attempt count is not actual trace count')
        require(all(type(t) is dict and type(t.get('completed')) is bool for t in traces), 'untyped completion flag')
        require(type(observation['original_Cython_completed']) is int and
                observation['original_Cython_completed'] == sum(t['completed'] for t in traces),
                'completion count is not actual completed trace count')
    for a, b in zip(left, right):
        if a['completed'] is not True or b['completed'] is not True:
            require(exact(a, b), 'incomplete source trace must remain exact')
            continue
        require(exact(_trace_without_order(a), _trace_without_order(b)), 'non-order source field differs')
        require(validate_completed_trace(a) == validate_completed_trace(b), 'source tie-rank sequence differs')


def assert_observation_equivalent(actual, expected):
    try:
        _check_observation(actual, expected)
    except (AssertionError, KeyError, TypeError, ValueError, struct.error) as error:
        # Preserve full actual and expected raw records on failure. No fitted
        # or normalized record replaces either source observation.
        raise AssertionError(str(error) + '\nactual=' + repr(actual) +
                             '\nexpected=' + repr(expected)) from error


def authenticate_source(root, current):
    """Authenticate the explicit two-body policy successor and old metadata chain."""
    root = Path(root)
    path = root / 'coverage/evidence/reconstruction-trace-policy-migration.json'
    record = json.loads(path.read_bytes())
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    require(record['schema'] == 'reconstruction-original-trace-policy-migration-v1' and
            record['policy'] == POLICY, 'unknown trace policy migration')
    require(record['previous_revision'] == 'f82c0c26433fb60c233bfc888a5e8e1a0a925bb6' and
            record['previous_raw_sha256'] == 'b18f5ab2777a4dda3a1fe244cb7edc75627d6fd9ccb59fcc3c08343c8c4189cb' and
            record['previous_canonical_LF_sha256'] == '16a5009b2dee9dfb154cddcf5eadb805d300c1c532451c0718c4d0f0ee35be68' and
            record['frozen_matrix_raw_sha256'] == '621981325f5e6939d3fb4d1c94e52910b369ef3ca798c326bdb93322a12b504c',
            'policy predecessor or frozen matrix identity differs')
    old = record['previous_source_UTF8'].encode()
    require(digest(old) == record['previous_raw_sha256'] and
            digest(old.replace(b'\r\n', b'\n')) == record['previous_canonical_LF_sha256'], 'previous raw/LF mapping differs')
    require(digest(current) == record['current_raw_sha256'] and
            digest(current.replace(b'\r\n', b'\n')) == record['current_canonical_LF_sha256'], 'current raw/LF mapping differs')
    helper = (root / 'tests/reconstruction_trace_policy.py').read_bytes()
    require(digest(helper) == record['validator_raw_sha256'], 'closed validator source differs')
    require(digest((root / 'coverage/contracts/reconstruction.json').read_bytes()) == record['frozen_matrix_raw_sha256'], 'frozen matrix changed')
    allowed = {'test_original_contract', 'test_source_signature_and_fixed_scope'}
    require(set(record['body_changes']) == allowed, 'policy migration scope differs')
    restored = current
    for name, change in record['body_changes'].items():
        new_body, old_body = change['current_UTF8'].encode(), change['previous_UTF8'].encode()
        require(restored.count(new_body) == 1, 'current body ambiguous or changed: ' + name)
        restored = restored.replace(new_body, old_body, 1)
    require(restored == old, 'non-policy source byte changed')
    before, after = ast.parse(old), ast.parse(current)
    require(len(before.body) == len(after.body), 'source top-level shape changed')
    for a, b in zip(before.body, after.body):
        if isinstance(a, ast.FunctionDef) and a.name in allowed:
            require(isinstance(b, ast.FunctionDef) and a.name == b.name, 'policy function moved')
            a.body = []; b.body = []
        require(ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False), 'unapproved source AST changed')
    previous_record_raw = (root / 'coverage/evidence/reconstruction-public-assertion-migration.json').read_bytes()
    require(digest(previous_record_raw) == record['previous_migration_raw_sha256'], 'historical migration record changed')
    previous = json.loads(previous_record_raw)
    original = previous['original_source_UTF8'].encode()
    matrix = json.loads((root / 'coverage/contracts/reconstruction.json').read_bytes())
    require(digest(original) == previous['original_raw_sha256'] and
            digest(original.replace(b'\r\n', b'\n')) == matrix['test_canonical_lf_sha256'] ==
            previous['original_canonical_lf_sha256'] == previous['original_normalized_lf_sha256'], 'original source provenance differs')
    require(digest(old) == previous['current_raw_sha256'] and
            digest(old.replace(b'\r\n', b'\n')) == previous['current_canonical_lf_sha256'] and
            digest((root / 'coverage/contracts/reconstruction.json').read_bytes()) == previous['frozen_matrix_raw_sha256'], 'previous effective source/matrix differs')
    a, b = ast.parse(original), ast.parse(old)
    metadata = {'test_source_signature_and_fixed_scope', 'test_native_pending_is_not_upstream_forwarding'}
    require(set(previous['allowed_metadata_body_changes']) == metadata and len(a.body) == len(b.body), 'historical metadata scope differs')
    for left, right in zip(a.body, b.body):
        if isinstance(left, ast.FunctionDef) and left.name in metadata:
            require(isinstance(right, ast.FunctionDef) and left.name == right.name, 'historical metadata function moved')
            left.body = []; right.body = []
        require(ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False), 'historical nonmetadata AST changed')
    return record
