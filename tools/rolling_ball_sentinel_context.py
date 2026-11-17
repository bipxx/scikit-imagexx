"""Exact metadata-only predecessor for the rolling-ball default identity fix."""
from pathlib import Path
import hashlib
import json

CONTEXT='coverage/contracts/rolling-ball-sentinel-v1.json'

def predecessor_bytes(root,name,raw):
    path=Path(root)/CONTEXT
    if not path.is_file():return raw  # Historical synthetic roots predate this layer.
    context=json.loads(path.read_bytes())
    assert context['predecessor_source']=='3e2a0fd1a7112bed6171930f541eb5c93879c54e'
    if name not in context['sources']:return raw
    row=context['sources'][name]
    # Source is byte-preserved by .gitattributes. Canonical entry also accepts
    # CRLF transport only by normalization before the same strict current hash.
    raw=raw.replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==row['current_canonical_sha256'],name
    for fragment in row['inverse_fragments']:
        current,old=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and raw.count(current)==1,name
        raw=raw.replace(current,old,1)
    assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_sha256'],name
    return raw
