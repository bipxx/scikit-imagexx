"""Exact current-reference test successor before immutable SDK/source-history gates."""
import hashlib
import json
from pathlib import Path

CONTEXT = 'coverage/contracts/qualification-capacity-source-v1.json'
SELF_PIN_PATH = 'tools/mark_lbp_fp_context.py'


def predecessor_bytes(root, name, raw, context_sha256, canonical=False):
    root = Path(root)
    path = root / CONTEXT
    if not path.is_file():
        return raw  # Explicit retained historical or synthetic source roots.
    encoded = path.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == context_sha256
    data = json.loads(encoded)
    assert data['schema'] == 'qualification-capacity-source-v1'
    assert data['predecessor_source'] == 'a378969be8e857e8ac904ec10e3cde472637af5d'
    assert hashlib.sha256((root / 'tools/qualification_capacity_context.py').read_bytes()).hexdigest() == data['helper_sha256']
    parent = data['predecessor_context']
    assert parent == {'path': 'coverage/contracts/vi-public-log2-reference-source-v2.json', 'sha256': '3f2e5cc3ad1330daa7455fff9d503b88493cc0a9a0314dcffe0d78c951de00f3'}
    assert hashlib.sha256((root / parent['path']).read_bytes()).hexdigest() == parent['sha256']
    assert hashlib.sha256((root / 'tools/vi_public_reference_context_v2.py').read_bytes()).hexdigest() == '905cc4a68b9475a672dfaf81cb432390420a497eaa572d5545d4261d1d8bceba'
    if name not in data['sources']:
        return raw
    row = data['sources'][name]
    if canonical:
        raw = raw.replace(b'\r\n', b'\n')
    if name == SELF_PIN_PATH:
        literal = ("QUALIFICATION_CAPACITY_CONTEXT_SHA256='" + context_sha256 + "'").encode()
        assert raw.count(literal) == 1
        raw = raw.replace(literal, b"QUALIFICATION_CAPACITY_CONTEXT_SHA256='" + b'0' * 64 + b"'", 1)
    domain = 'canonical' if canonical else 'raw'
    assert hashlib.sha256(raw).hexdigest() == row['current_' + domain + '_sha256'], name
    text = raw.decode('utf-8')
    for part in reversed(row['inverse']):
        assert text[part['start']:part['end']] == part['current'], name
        text = text[:part['start']] + part['predecessor'] + text[part['end']:]
    result = text.encode('utf-8')
    assert hashlib.sha256(result).hexdigest() == row['predecessor_' + domain + '_sha256'], name
    return result
