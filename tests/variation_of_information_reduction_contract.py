"""Exact historical source projection for private VI reduction successor."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/variation-of-information-reduction-v2.json'


def historical_vi_source(raw):
    context=json.loads(CONTEXT.read_bytes())
    assert hashlib.sha256(raw).hexdigest()==context['source_sha256']
    for part in reversed(context['source_inverse']):
        assert raw[part['start']:part['end']]==part['current'].encode()
        raw=raw[:part['start']]+part['baseline'].encode()+raw[part['end']:]
    assert hashlib.sha256(raw).hexdigest()==context['baseline_source_sha256']
    return raw
