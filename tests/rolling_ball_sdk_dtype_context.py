"""Exact family source projection for root's later selective SDK intake."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/rolling-ball-sdk-dtypes-v1.json'
def historical_sdk_source(name,raw):
 data=json.loads((ROOT/CONTEXT).read_bytes());row=data['source_projection'][name]
 assert hashlib.sha256(raw).hexdigest()==row['raw_sha256']
 for part in reversed(row['inverse']):
  assert raw[part['start']:part['end']]==part['current'].encode()
  raw=raw[:part['start']]+part['predecessor'].encode()+raw[part['end']:]
 assert hashlib.sha256(raw).hexdigest()==row['predecessor_raw_sha256']
 return raw
