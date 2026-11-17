"""Exact current-reference test successor before immutable SDK/source-history gates."""
import hashlib
import json
from pathlib import Path

CONTEXT = 'coverage/contracts/current281-validation-source-v1.json'
SELF_PIN_PATH = 'tools/mark_lbp_fp_context.py'


def predecessor_bytes(root, name, raw, context_sha256, canonical=False):
    root = Path(root)
    path = root / CONTEXT
    if not path.is_file():
        return raw  # Explicit retained historical or synthetic source roots.
    encoded = path.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == context_sha256
    data = json.loads(encoded)
    assert data['schema'] == 'current281-validation-source-v1'
    assert data['predecessor_source'] == 'f10e04ae14f27055f5a09d540cbd3b616073d7ad'
    assert hashlib.sha256((root / 'tools/current_validation_context.py').read_bytes()).hexdigest() == data['helper_sha256']
    parent = data['predecessor_context']
    assert parent == {'path': 'coverage/contracts/qualification-capacity-source-v1.json', 'sha256': '88dfda9a5df6f7abc1f43e008e0a61511a03e5b661189de285a5e26d77596512'}
    assert hashlib.sha256((root / parent['path']).read_bytes()).hexdigest() == parent['sha256']
    assert hashlib.sha256((root / 'tools/qualification_capacity_context.py').read_bytes()).hexdigest() == '6d865e506d3f612d6218661894fa609eb2add2dcb31542d40859d505cba7eb71'
    retained = data['retained_raw_passthrough']
    assert retained == {'path': 'tests/test_mark_lbp_controller.py', 'bytes': 25301, 'sha256': '167edc6fa02c7b48d2b9d5598d9107828e163d542406d3718ddbe6b7e98b9dbd'}
    if name == retained['path'] and len(raw) == retained['bytes'] and hashlib.sha256(raw).hexdigest() == retained['sha256']:
        return raw  # Existing exact historical domain; all older gates still run.
    if name not in data['sources']:
        return raw
    row = data['sources'][name]
    if canonical:
        raw = raw.replace(b'\r\n', b'\n')
    if name == SELF_PIN_PATH:
        literal = ("CURRENT_VALIDATION_CONTEXT_SHA256='" + context_sha256 + "'").encode()
        assert raw.count(literal) == 1
        raw = raw.replace(literal, b"CURRENT_VALIDATION_CONTEXT_SHA256='" + b'0' * 64 + b"'", 1)
    domain = 'canonical' if canonical else 'raw'
    assert hashlib.sha256(raw).hexdigest() == row['current_' + domain + '_sha256'], name
    text = raw.decode('utf-8')
    for part in reversed(row['inverse']):
        assert text[part['start']:part['end']] == part['current'], name
        text = text[:part['start']] + part['predecessor'] + text[part['end']:]
    result = text.encode('utf-8')
    assert hashlib.sha256(result).hexdigest() == row['predecessor_' + domain + '_sha256'], name
    return result
