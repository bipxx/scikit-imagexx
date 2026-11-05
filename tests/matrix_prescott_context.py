"""Strict current-to-qualified wrapper projection for two historical guards."""
import hashlib,json
from pathlib import Path

def predecessor_bytes(root,source=None):
    root=Path(root);record=json.loads((root/'coverage/contracts/matrix-prescott-v1.json').read_text())['source_context']
    live=(root/'python/skimagexx/_matrix_transform.py').read_bytes().replace(b'\r\n',b'\n')if source is None else source.replace(b'\r\n',b'\n')
    assert hashlib.sha256(live).hexdigest()==record['live_wrapper_canonical_lf_sha256']
    for fragment in record['inverse_fragments']:
        current=fragment['current'].encode();assert current and live.count(current)==1
        live=live.replace(current,fragment['baseline'].encode())
    assert hashlib.sha256(live).hexdigest()==record['baseline_wrapper_canonical_lf_sha256']
    return live
