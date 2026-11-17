"""Exact historical binder projection for family source evidence only."""
import hashlib,json
from pathlib import Path

def historical_bytes(root,name):
    root=Path(root);d=json.loads((root/'coverage/contracts/draw-multiblock-lbp-metadata-v1.json').read_bytes());assert name==d['source']
    raw=(root/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==d['current_sha256']
    current=d['inverse']['current'].encode();assert raw.count(current)==1
    old=raw.replace(current,d['inverse']['predecessor'].encode(),1)
    assert hashlib.sha256(old).hexdigest()==d['predecessor_sha256'];return old
