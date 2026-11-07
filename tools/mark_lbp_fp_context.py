"""Exact current MARK/LBP integration to frozen DDA metadata projection."""
import hashlib,json
from pathlib import Path
CONTEXT='coverage/contracts/mark-lbp-fp-integration-v1.json'
def predecessor_bytes(root,name,raw,canonical=False):
 path=Path(root)/CONTEXT
 if not path.is_file():return raw
 rows=json.loads(path.read_bytes())['predecessors']
 if name not in rows:return raw
 row=rows[name]
 if canonical:raw=raw.replace(b'\r\n',b'\n')
 suffix='canonical_sha256'if canonical else'raw_sha256'
 assert hashlib.sha256(raw).hexdigest()==row['current_'+suffix],name
 current=row['current'].encode();old=row['predecessor'].encode()
 if canonical:current=current.replace(b'\r\n',b'\n');old=old.replace(b'\r\n',b'\n')
 assert raw==current,name
 assert hashlib.sha256(old).hexdigest()==row['predecessor_'+suffix],name
 return old

def sentinel_predecessor_bytes(root,name,raw):
    from tools.rolling_ball_sentinel_context import predecessor_bytes as sentinel
    return sentinel(root,name,predecessor_bytes(root,name,raw,canonical=True))
