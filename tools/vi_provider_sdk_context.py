"""Exact SDK-test successor bytes before the retained VI source-history gates."""
import hashlib
import json
from pathlib import Path

CONTEXT = 'coverage/contracts/vi-provider-sdk-source-v1.json'
SELF_PIN_PATH = 'tools/mark_lbp_fp_context.py'


def predecessor_bytes(root, name, raw, context_sha256, canonical=False):
    root = Path(root)
    path = root / CONTEXT
    if not path.is_file():
        return raw  # Explicit retained historical or synthetic source roots.
    encoded = path.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == context_sha256
    data = json.loads(encoded)
    assert data['schema'] == 'vi-provider-sdk-source-v1'
    assert data['predecessor_source'] == '114ed9b3e0662982bfd82d217b59ee11acba9e25'
    assert hashlib.sha256((root / 'tools/vi_provider_sdk_context.py').read_bytes()).hexdigest() == data['helper_sha256']
    if name not in data['sources']:
        return raw
    row = data['sources'][name]
    if canonical:
        raw = raw.replace(b'\r\n', b'\n')
    if name == SELF_PIN_PATH:
        literal = ("VI_PROVIDER_SDK_CONTEXT_SHA256='" + context_sha256 + "'").encode()
        assert raw.count(literal) == 1
        raw = raw.replace(literal, b"VI_PROVIDER_SDK_CONTEXT_SHA256='" + b'0' * 64 + b"'", 1)
    domain = 'canonical' if canonical else 'raw'
    assert hashlib.sha256(raw).hexdigest() == row['current_' + domain + '_sha256'], name
    text = raw.decode('utf-8')
    for part in reversed(row['inverse']):
        assert text[part['start']:part['end']] == part['current'], name
        text = text[:part['start']] + part['predecessor'] + text[part['end']:]
    result = text.encode('utf-8')
    assert hashlib.sha256(result).hexdigest() == row['predecessor_' + domain + '_sha256'], name
    return result
