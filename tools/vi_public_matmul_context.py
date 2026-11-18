"""Exact2270 source-history view for the explicit public matmul successor."""
import hashlib
import json
from pathlib import Path

CONTEXT = 'coverage/contracts/vi-public-matmul-source-history-v34.json'
SELF_PIN_PATH = 'tools/mark_lbp_fp_context.py'
PREDECESSOR = '2270b7fefc9a8e5cb3ba7fe51d2db5759c17634d'


def _context(root, context_sha256):
    root = Path(root)
    path = root / CONTEXT
    if not path.is_file():
        return None  # Explicit historical or synthetic source roots.
    encoded = path.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == context_sha256
    data = json.loads(encoded)
    assert data['schema'] == 'vi-public-matmul-source-history-v1'
    assert data['predecessor_source'] == PREDECESSOR
    assert data['execution_authority'] is None
    assert hashlib.sha256((root / 'tools/vi_public_matmul_context.py').read_bytes()).hexdigest() == data['helper_sha256']
    assert data['predecessor_context'] == {
        'path': 'coverage/contracts/current281-validation-source-v1.json',
        'sha256': '30e078cce3e012ce4cd5dbe29b3ef727fcc4aaf59413b0b7e22996b2b13303a5',
    }
    history_parent = data['predecessor_history_context']
    assert history_parent == {
        'path': 'coverage/contracts/vi-public-matmul-source-history-v14.json',
        'sha256': '16e3ab9c003881f18c072f88f4b2fd940d8915b74e11b59b87b5a80ee31e9715',
        'source': 'e27633bf9186bb7cab5c1d0b9fbf1584c94c6b81',
    }
    assert hashlib.sha256((root / history_parent['path']).read_bytes()).hexdigest() == history_parent['sha256']
    accepted = data['previous_candidate_history']
    assert accepted == {'path': 'coverage/contracts/vi-public-matmul-source-history-v33.json', 'sha256': '3c878318be342003c1d1281a5b5e080e47e6c95fd59cfcbc41a81619670846cc'}
    assert hashlib.sha256((root / accepted['path']).read_bytes()).hexdigest() == accepted['sha256']
    parent = data['predecessor_context']
    assert hashlib.sha256((root / parent['path']).read_bytes()).hexdigest() == parent['sha256']
    assert hashlib.sha256((root / 'tools/current_validation_context.py').read_bytes()).hexdigest() == '52d8eb80b2138d8d11fb8df7ef3b214c964faadc1bbf71419825d6ce0dc3d064'
    view = data['view_helper']
    assert view['path'] == 'tests/vi_public_matmul_source_view.py'
    assert hashlib.sha256((root / view['path']).read_bytes()).hexdigest() == view['sha256']
    return data


def _project(data, name, raw, context_sha256, canonical):
    if name not in data['sources']:
        return raw
    if canonical:
        raw = raw.replace(b'\r\n', b'\n')
    if name == SELF_PIN_PATH:
        literal = ("VI_PUBLIC_MATMUL_CONTEXT_SHA256='" + context_sha256 + "'").encode()
        assert raw.count(literal) == 1
        raw = raw.replace(literal, b"VI_PUBLIC_MATMUL_CONTEXT_SHA256='" + b'0' * 64 + b"'", 1)
    domain = 'canonical' if canonical else 'raw'
    row = data['sources'][name]
    assert hashlib.sha256(raw).hexdigest() == row['current_' + domain + '_sha256'], name
    text = raw.decode('utf-8')
    for part in reversed(row['inverse'][domain]):
        assert text[part['start']:part['end']] == part['current'], name
        text = text[:part['start']] + part['predecessor'] + text[part['end']:]
    result = text.encode('utf-8')
    assert hashlib.sha256(result).hexdigest() == row['predecessor_' + domain + '_sha256'], name
    return result


def predecessor_bytes(root, name, raw, context_sha256, canonical=False):
    data = _context(root, context_sha256)
    if data is None:
        return raw
    return _project(data, name, raw, context_sha256, canonical)


def immediate_predecessor_bytes(root, name, raw, context_sha256, canonical=False):
    """Authenticate current source bytes before the exact e276 source guards."""
    data = _context(root, context_sha256)
    assert data is not None and name in data['immediate_predecessors']
    return _project({'sources': data['immediate_predecessors']}, name, raw,
                    context_sha256, canonical)


def speed_predecessor_bytes(root, name, raw, context_sha256, canonical=False):
    """Authenticate current bytes before retained main1a23 metadata guards."""
    data = _context(root, context_sha256)
    assert data is not None and name in data['speed_predecessors']
    return _project({'sources': data['speed_predecessors']}, name, raw,
                    context_sha256, canonical)


def historical_source_root(root, destination, context_sha256):
    """Materialize authenticated source inputs for historical source guards only."""
    root, destination = Path(root), Path(destination)
    data = _context(root, context_sha256)
    assert data is not None
    snapshot = data['historical_source_inputs']
    assert snapshot['source'] == PREDECESSOR
    frozen = data['frozen_operational_inputs']
    assert set(frozen) == {'coverage/work_items.json'}
    receipt = destination.with_name(destination.name + '.receipt.json')
    if destination.exists():
        prior = json.loads(receipt.read_bytes())
        assert prior == {'context_sha256': context_sha256, 'source': PREDECESSOR,
                         'files': len(snapshot['files']), 'bytes': snapshot['bytes']}
        return destination
    destination.mkdir()
    for name, row in snapshot['files'].items():
        source = root / name
        assert not source.is_symlink() and not (getattr(source.lstat(), 'st_file_attributes', 0) & 0x400)
        # Operational state is mutable and has no current numerical authority.
        # The historical fixture uses its exact authenticated2270 payload only.
        raw = frozen[name].encode('utf-8') if name in frozen else _project(data, name, source.read_bytes(), context_sha256, False)
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        assert len(raw) == row['bytes'] and blob == row['git_blob_sha1'], name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(raw)
    # Detect a changed source contract during materialization, before reuse.
    assert hashlib.sha256((root / CONTEXT).read_bytes()).hexdigest() == context_sha256
    receipt.write_text(json.dumps({'context_sha256': context_sha256, 'source': PREDECESSOR,
                                  'files': len(snapshot['files']), 'bytes': snapshot['bytes']}) + '\n',
                       encoding='utf-8', newline='\n')
    return destination
