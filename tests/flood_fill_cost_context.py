"""Strict current-to-dd1a source projection for this CPP-only cost successor."""
import hashlib
import json
from pathlib import Path
from flood_fill_source_context import historical_source as integration_bytes
CONTEXT='coverage/contracts/flood-fill-native-v2.json'

def historical_bytes(root,name):
    root=Path(root);context=json.loads((root/CONTEXT).read_bytes())
    row=context['changed_sources'][name];raw=integration_bytes(root,name)
    assert hashlib.sha256(raw).hexdigest()==row['current_raw_sha256'],name
    for fragment in row['inverse']:
        current=fragment['current'].encode();assert current and raw.count(current)==1,name
        raw=raw.replace(current,fragment['predecessor'].encode(),1)
    assert hashlib.sha256(raw).hexdigest()==row['predecessor_raw_sha256'],name
    return raw
