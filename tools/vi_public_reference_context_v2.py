"""Exact current-reference test successor before immutable SDK/source-history gates."""
import hashlib
import json
from pathlib import Path

CONTEXT = 'coverage/contracts/vi-public-log2-reference-source-v2.json'
SELF_PIN_PATH = 'tools/mark_lbp_fp_context.py'


def predecessor_bytes(root, name, raw, context_sha256, canonical=False):
    root = Path(root)
    path = root / CONTEXT
    if not path.is_file():
        return raw  # Explicit retained historical or synthetic source roots.
    encoded = path.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == context_sha256
    data = json.loads(encoded)
    assert data['schema'] == 'vi-public-log2-reference-source-v2'
    assert data['predecessor_source'] == '1237655d691740d20b35fe28109b4dc08642896a'
    assert hashlib.sha256((root / 'tools/vi_public_reference_context_v2.py').read_bytes()).hexdigest() == data['helper_sha256']
    prior = data['supersedes_context']
    assert prior == {'path': 'coverage/contracts/vi-public-log2-reference-source-v1.json', 'sha256': 'a4ec00255333080a3dcdba3c73a28049e6536ce73c4e607888f746dad9b93da7'}
    assert hashlib.sha256((root / prior['path']).read_bytes()).hexdigest() == prior['sha256']
    assert hashlib.sha256((root / 'tools/vi_public_reference_context.py').read_bytes()).hexdigest() == '52b3f9fd5913e9ebee0937ba78d5cea1f91e4a2707f7acfa50ff799e68cdd43a'
    parent = data['predecessor_context']
    assert parent == {'path': 'coverage/contracts/vi-provider-sdk-source-v1.json', 'sha256': 'a95edd6c736a2cbce6fb9530b1cfec000390b8368d6422640dfc1d6438e1714d'}
    assert hashlib.sha256((root / parent['path']).read_bytes()).hexdigest() == parent['sha256']
    if name not in data['sources']:
        return raw
    row = data['sources'][name]
    if canonical:
        raw = raw.replace(b'\r\n', b'\n')
    if name == SELF_PIN_PATH:
        literal = ("VI_PUBLIC_REFERENCE_CONTEXT_SHA256='" + context_sha256 + "'").encode()
        assert raw.count(literal) == 1
        raw = raw.replace(literal, b"VI_PUBLIC_REFERENCE_CONTEXT_SHA256='" + b'0' * 64 + b"'", 1)
    domain = 'canonical' if canonical else 'raw'
    assert hashlib.sha256(raw).hexdigest() == row['current_' + domain + '_sha256'], name
    text = raw.decode('utf-8')
    for part in reversed(row['inverse']):
        assert text[part['start']:part['end']] == part['current'], name
        text = text[:part['start']] + part['predecessor'] + text[part['end']:]
    result = text.encode('utf-8')
    assert hashlib.sha256(result).hexdigest() == row['predecessor_' + domain + '_sha256'], name
    return result
