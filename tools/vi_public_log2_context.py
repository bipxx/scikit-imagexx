"""One exact VI provider source successor; never a numerical qualification view."""
import hashlib
import json
from pathlib import Path
CONTEXT='coverage/contracts/vi-public-log2-provider-source-v1.json'
SELF_PIN_PATH='tools/mark_lbp_fp_context.py'


def predecessor_bytes(root,name,raw,context_sha256,canonical=False):
    root=Path(root);path=root/CONTEXT
    if not path.is_file():return raw  # Explicit historical/synthetic roots.
    encoded=path.read_bytes()
    assert hashlib.sha256(encoded).hexdigest()==context_sha256
    data=json.loads(encoded)
    assert data['schema']=='vi-public-log2-provider-source-v1'
    assert data['predecessor_source']=='32e44d69d3f74e5058671b14be978e25fb459750'
    assert hashlib.sha256((root/'tools/vi_public_log2_context.py').read_bytes()).hexdigest()==data['helper_sha256']
    if name not in data['sources']:return raw
    row=data['sources'][name]
    if canonical:raw=raw.replace(b'\r\n',b'\n')
    # Only this literal metadata self-pin is normalized to break its hash cycle.
    # Every other byte, the helper pin and all predecessor bytes stay bound.
    if name==SELF_PIN_PATH:
        current=("VI_PUBLIC_LOG2_CONTEXT_SHA256='"+context_sha256+"'").encode()
        assert raw.count(current)==1
        raw=raw.replace(current,b"VI_PUBLIC_LOG2_CONTEXT_SHA256='"+b'0'*64+b"'",1)
    domain='canonical'if canonical else'raw'
    assert hashlib.sha256(raw).hexdigest()==row['current_'+domain+'_sha256'],name
    text=raw.decode('utf-8')
    for part in reversed(row['inverse']):
        assert text[part['start']:part['end']]==part['current'],name
        text=text[:part['start']]+part['predecessor']+text[part['end']:]
    result=text.encode('utf-8')
    assert hashlib.sha256(result).hexdigest()==row['predecessor_'+domain+'_sha256'],name
    return result
