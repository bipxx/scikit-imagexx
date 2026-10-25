"""Closed H-only metadata source evolution; never authenticates image outputs.

Historical oracle/source pins remain authoritative. Only the five named
metadata bodies below may evolve, with exact raw inversion to their real Git
predecessors. Qualification must additionally pin this reader/evidence itself.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

RECORD = 'coverage/evidence/h-extrema-reference-assertion-migration.json'
SCHEMA = 'h-extrema-reference-assertion-migration-v1'
ALLOWED = {
    'tests/test_h_extrema_contracts.py': (
        '00ad418879bc5b4ffbeebca0f78497f03d06da9cc1b31f60cd8313aa80327151',
        ('test_source_scope_and_signatures', 'test_strict_native_pending')),
    'tests/h_extrema_reference.py': (
        'bc9efc45e338d6c447284df244185d8101d4615e6f72abfe500ced05f71bdfbf',
        ('authenticate',)),
    'tests/h_extrema_floating_reference.py': (
        '389e5cffe1b399c0deb3c8dce8424a414b9b2329e16907e6afa86415fab0dfb0',
        ('authenticate',)),
    'tests/test_h_extrema_floating_cases.py': (
        'f127526cd66c5043edd66d82518db3abb2a8956d3b29f1931057566ec090de1e',
        ('test_prospective_case_partition_and_source_pins',)),
}
LINEAGE = (
    '3939704203a9f097797f48bf5aead7646231e93e',
    '8503d1c54f7521e560e11ddab1a3357c2af5ed64',
    '71c237eb3b5e99ad7befc0bc3cd6cab6eb637a23',
    '3f4ed95e9fb322985374f2af71522398dbd8db7f',
    '14236c23cefdf814037e9eceac81c89317d06dcb',
    '8fb449efed4e0a378f181575f25cb55f24f9a0f8',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _body(raw, tree, name):
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, 'H metadata body absent or duplicated'
    function = found[0]
    lines = raw.splitlines(keepends=True)
    return b''.join(lines[function.body[0].lineno-1:function.body[-1].end_lineno])


def verify_record(record, current, reader_raw):
    """Verify ONLY this closed source mapping, including complete raw inverse."""
    assert set(record) == {'schema', 'historical_test_lineage', 'files', 'reader_sha256', 'scope'}
    assert record['schema'] == SCHEMA and record['reader_sha256'] == sha(reader_raw)
    expected_lineage = [dict(revision=n, bytes=10299,
                            raw_sha256=ALLOWED['tests/test_h_extrema_contracts.py'][0],
                            canonical_lf_sha256=ALLOWED['tests/test_h_extrema_contracts.py'][0]) for n in LINEAGE]
    assert record['historical_test_lineage'] == expected_lineage, 'different historical H lineage'
    assert set(record['files']) == set(current) == set(ALLOWED), 'unlisted H source mismatch'
    for name, (expected, names) in ALLOWED.items():
        entry = record['files'][name]
        assert set(entry) == {'original_source_utf8', 'original_raw_sha256', 'original_canonical_lf_sha256',
                              'current_raw_sha256', 'current_canonical_lf_sha256', 'body_replacements'}
        original = entry['original_source_utf8'].encode('utf-8')
        raw = current[name]
        assert sha(original) == entry['original_raw_sha256'] == expected, 'wrong H historical capture'
        assert sha(original.replace(b'\r\n', b'\n')) == entry['original_canonical_lf_sha256']
        assert sha(raw) == entry['current_raw_sha256'], 'H current source hash differs'
        assert sha(raw.replace(b'\r\n', b'\n')) == entry['current_canonical_lf_sha256']
        assert [r['name'] for r in entry['body_replacements']] == list(names), 'unlisted H metadata body'
        old_tree, new_tree = ast.parse(original), ast.parse(raw)
        inverse = raw
        for replacement in entry['body_replacements']:
            assert set(replacement) == {'name', 'original_body_utf8', 'current_body_utf8'}
            before = replacement['original_body_utf8'].encode('utf-8')
            after = replacement['current_body_utf8'].encode('utf-8')
            assert before == _body(original, old_tree, replacement['name'])
            assert after == _body(raw, new_tree, replacement['name'])
            assert original.count(before) == raw.count(after) == 1
            inverse = inverse.replace(after, before, 1)
        assert inverse == original, 'H change outside permitted metadata bodies'
        # Raw inverse protects every non-body byte and all source positions.
        # This second check retains every semantic AST field on the SAME runtime.
        restored = copy.deepcopy(new_tree)
        for node in restored.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                node.body = copy.deepcopy(next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == node.name).body)
        assert ast.dump(restored, include_attributes=False) == ast.dump(old_tree, include_attributes=False)
    return True


def authenticate_source(root, name, historical_sha256):
    """Return real current/predecessor pins; unknown paths never get an escape."""
    assert name in ALLOWED and historical_sha256 == ALLOWED[name][0], 'not an exact H historical source pin'
    root = Path(root)
    raw = (root / name).read_bytes()
    if sha(raw) == historical_sha256:
        return dict(path=name, current_raw_sha256=sha(raw), historical_raw_sha256=historical_sha256,
                    historical_canonical_lf_sha256=sha(raw.replace(b'\r\n', b'\n')), migrated=False)
    record_raw = (root / RECORD).read_bytes()
    record = json.loads(record_raw)
    current = {key: (root / key).read_bytes() for key in ALLOWED}
    verify_record(record, current, Path(__file__).read_bytes())
    return dict(path=name, current_raw_sha256=sha(raw), historical_raw_sha256=historical_sha256,
                historical_canonical_lf_sha256=record['files'][name]['original_canonical_lf_sha256'],
                migrated=True, record_sha256=sha(record_raw))
