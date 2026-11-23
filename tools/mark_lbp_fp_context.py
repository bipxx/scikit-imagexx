"""Exact current MARK/LBP integration to frozen DDA metadata projection."""
import hashlib,json
from pathlib import Path
CONTEXT='coverage/contracts/mark-lbp-fp-integration-v2.json'
# Exact source-only successor before the immutable v2 historical guards.
SOURCE_SUCCESSOR='coverage/contracts/mark-lbp-controller-source-successor-v1.json'
VI_PUBLIC_MATMUL_CONTEXT_SHA256='9a4b21e0213e5073ae6c741bfd616cf9cf679e25385b82a946af361f64ced417'
VI_PUBLIC_MATMUL_HELPER_SHA256='50be82c6002b82ec15626eb2430f569cb48a7e3f21368a42512cec109c1a80b3'
CURRENT_VALIDATION_CONTEXT_SHA256='30e078cce3e012ce4cd5dbe29b3ef727fcc4aaf59413b0b7e22996b2b13303a5'
CURRENT_VALIDATION_HELPER_SHA256='52d8eb80b2138d8d11fb8df7ef3b214c964faadc1bbf71419825d6ce0dc3d064'
QUALIFICATION_CAPACITY_CONTEXT_SHA256='88dfda9a5df6f7abc1f43e008e0a61511a03e5b661189de285a5e26d77596512'
QUALIFICATION_CAPACITY_HELPER_SHA256='6d865e506d3f612d6218661894fa609eb2add2dcb31542d40859d505cba7eb71'
VI_PUBLIC_REFERENCE_CONTEXT_SHA256='3f2e5cc3ad1330daa7455fff9d503b88493cc0a9a0314dcffe0d78c951de00f3'
VI_PUBLIC_REFERENCE_HELPER_SHA256='905cc4a68b9475a672dfaf81cb432390420a497eaa572d5545d4261d1d8bceba'
VI_PROVIDER_SDK_CONTEXT_SHA256='a95edd6c736a2cbce6fb9530b1cfec000390b8368d6422640dfc1d6438e1714d'
VI_PROVIDER_SDK_HELPER_SHA256='77b8494697b9c52290be0705de7b8bab7bfeafafa3097637433b3baf37aa0c33'
VI_PUBLIC_LOG2_CONTEXT_SHA256='3b21af81541ffdc87104be4c6f22205778f3c60d55a561d431937c939833b143'
VI_PUBLIC_LOG2_HELPER_SHA256='83ad5f62c746b10fd29ab787fe2674710a16087265caa0a4bba059a7f769a352'
SOURCE_PREDECESSOR_SHA256={'tools/join_source_context.py': '3c5c566931632ad354bf5d25e850becd39157863cf3f1bb2aa8a603f2838b1bc', 'tools/mark_lbp_fp_context.py': '4ccccf650a774e17d917ae8fe1a71cdd2017609129f5198510e866c7370795f6', 'tools/public_qualification.py': '2f10933aa9c1b548e8e67693f8f24ab26f40830afea8ae1d864901fb80a7f2d1', 'tests/test_rolling_ball_enabled_source.py': '1b8ab178aa85e9693ee5bb540ba586959af4d6f166a61b2be0c843e5e0a0eaeb', 'tests/test_mark_lbp_fp_integration_source.py': '4a94cf41684e51c792db00e7e2422a0b6b026a7fde203518305ad2f56cbfd2a2', 'tests/test_mark_lbp_controller.py': '3286e1f57a8ee1625e2f8130150ed452593f3c99b5236fb00289fe234811f1e0', 'tests/test_vi_controller.py': '231e5db18a8ecbb598979bdfca73133ad55060c1ae0599a08171470bce91992c'}

def retained_mark_raw(name,raw):
 return (name=='tests/test_mark_lbp_controller.py'and len(raw)==25301
         and hashlib.sha256(raw).hexdigest()=='167edc6fa02c7b48d2b9d5598d9107828e163d542406d3718ddbe6b7e98b9dbd')


def source_successor_predecessor_bytes(root,name,raw,canonical=False):
 path=Path(root)/'coverage/contracts/vi-public-matmul-source-history-v1.json'
 if path.is_file():
  helper=Path(root)/'tools/vi_public_matmul_context.py'
  assert hashlib.sha256(helper.read_bytes()).hexdigest()==VI_PUBLIC_MATMUL_HELPER_SHA256
  from tools.vi_public_matmul_context import _context,predecessor_bytes as matmul_predecessor
  if retained_mark_raw(name,raw):
   # Authenticate the current context but leave the exact older raw caller
   # for the original downstream gates, which still execute below.
   _context(root,VI_PUBLIC_MATMUL_CONTEXT_SHA256)
  else:
   raw=matmul_predecessor(root,name,raw,VI_PUBLIC_MATMUL_CONTEXT_SHA256,canonical)
 path=Path(root)/'coverage/contracts/current281-validation-source-v1.json'
 if path.is_file():
  helper=Path(root)/'tools/current_validation_context.py'
  assert hashlib.sha256(helper.read_bytes()).hexdigest()==CURRENT_VALIDATION_HELPER_SHA256
  from tools.current_validation_context import predecessor_bytes as current_validation_predecessor
  raw=current_validation_predecessor(root,name,raw,CURRENT_VALIDATION_CONTEXT_SHA256,canonical)
 path=Path(root)/'coverage/contracts/qualification-capacity-source-v1.json'
 if path.is_file():
  helper=Path(root)/'tools/qualification_capacity_context.py'
  assert hashlib.sha256(helper.read_bytes()).hexdigest()==QUALIFICATION_CAPACITY_HELPER_SHA256
  from tools.qualification_capacity_context import predecessor_bytes as capacity_predecessor
  raw=capacity_predecessor(root,name,raw,QUALIFICATION_CAPACITY_CONTEXT_SHA256,canonical)
 path=Path(root)/'coverage/contracts/vi-public-log2-reference-source-v2.json'
 if path.is_file():
  helper=Path(root)/'tools/vi_public_reference_context_v2.py'
  assert hashlib.sha256(helper.read_bytes()).hexdigest()==VI_PUBLIC_REFERENCE_HELPER_SHA256
  from tools.vi_public_reference_context_v2 import predecessor_bytes as vi_reference_predecessor
  raw=vi_reference_predecessor(root,name,raw,VI_PUBLIC_REFERENCE_CONTEXT_SHA256,canonical)
 path=Path(root)/'coverage/contracts/vi-provider-sdk-source-v1.json'
 if path.is_file():
  helper=Path(root)/'tools/vi_provider_sdk_context.py'
  assert hashlib.sha256(helper.read_bytes()).hexdigest()==VI_PROVIDER_SDK_HELPER_SHA256
  from tools.vi_provider_sdk_context import predecessor_bytes as vi_sdk_predecessor
  raw=vi_sdk_predecessor(root,name,raw,VI_PROVIDER_SDK_CONTEXT_SHA256,canonical)
 path=Path(root)/'coverage/contracts/vi-public-log2-provider-source-v1.json'
 if path.is_file():
  helper=Path(root)/'tools/vi_public_log2_context.py'
  assert hashlib.sha256(helper.read_bytes()).hexdigest()==VI_PUBLIC_LOG2_HELPER_SHA256
  from tools.vi_public_log2_context import predecessor_bytes as vi_provider_predecessor
  raw=vi_provider_predecessor(root,name,raw,VI_PUBLIC_LOG2_CONTEXT_SHA256,canonical)
 if name not in SOURCE_PREDECESSOR_SHA256:return raw
 path=Path(root)/SOURCE_SUCCESSOR
 if not path.is_file():return raw
 context=json.loads(path.read_bytes())
 assert context['schema']=='mark-lbp-controller-source-successor-v1'
 assert context['predecessor_source']=='aaa13487d273b2a895e102cbe815daf8e008098d'
 parent=context['predecessor_context']
 assert parent=={'path': 'coverage/contracts/mark-lbp-fp-integration-v2.json', 'sha256': '8b6528ef6388cafbf8dd4e2c4081649d3193bb4ae4138d75df5cd85f33b374da'}
 assert hashlib.sha256((Path(root)/parent['path']).read_bytes()).hexdigest()==parent['sha256']
 assert set(context['sources'])==set(SOURCE_PREDECESSOR_SHA256)
 # Preserve only the exact already-authenticated raw MARK binding input,
 # before any canonical transport normalization; older gates still run.
 if retained_mark_raw(name,raw):return raw
 row=context['sources'][name]
 if canonical:raw=raw.replace(b'\r\n',b'\n')
 suffix='canonical_sha256'if canonical else'raw_sha256'
 assert hashlib.sha256(raw).hexdigest()==row['current_'+suffix],name
 for fragment in row['inverse_fragments']:
  current,previous=fragment['current'].encode(),fragment['predecessor'].encode()
  assert current and raw.count(current)==1,name
  raw=raw.replace(current,previous,1)
 assert hashlib.sha256(raw).hexdigest()==row['predecessor_'+suffix]==SOURCE_PREDECESSOR_SHA256[name],name
 return raw


def predecessor_bytes(root,name,raw,canonical=False):
 raw=source_successor_predecessor_bytes(root,name,raw,canonical)
 path=Path(root)/CONTEXT
 if not path.is_file():path=Path(root)/'coverage/contracts/mark-lbp-fp-integration-v1.json'  # Frozen historical/synthetic roots.
 if not path.is_file():return raw
 context=json.loads(path.read_bytes())
 if 'predecessor_context'in context:
  parent=context['predecessor_context'];assert hashlib.sha256((Path(root)/parent['path']).read_bytes()).hexdigest()==parent['sha256']
 rows=context['predecessors']
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
