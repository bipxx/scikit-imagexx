"""Exact current MARK/LBP integration to frozen DDA metadata projection."""
import hashlib,json
from pathlib import Path
CONTEXT='coverage/contracts/mark-lbp-fp-integration-v2.json'
# Exact source-only successor before the immutable v2 historical guards.
SOURCE_SUCCESSOR='coverage/contracts/mark-lbp-controller-source-successor-v1.json'
VI_PUBLIC_LOG2_CONTEXT_SHA256='3b21af81541ffdc87104be4c6f22205778f3c60d55a561d431937c939833b143'
VI_PUBLIC_LOG2_HELPER_SHA256='83ad5f62c746b10fd29ab787fe2674710a16087265caa0a4bba059a7f769a352'
SOURCE_PREDECESSOR_SHA256={'tools/join_source_context.py': '3c5c566931632ad354bf5d25e850becd39157863cf3f1bb2aa8a603f2838b1bc', 'tools/mark_lbp_fp_context.py': '4ccccf650a774e17d917ae8fe1a71cdd2017609129f5198510e866c7370795f6', 'tools/public_qualification.py': '2f10933aa9c1b548e8e67693f8f24ab26f40830afea8ae1d864901fb80a7f2d1', 'tests/test_rolling_ball_enabled_source.py': '1b8ab178aa85e9693ee5bb540ba586959af4d6f166a61b2be0c843e5e0a0eaeb', 'tests/test_mark_lbp_fp_integration_source.py': '4a94cf41684e51c792db00e7e2422a0b6b026a7fde203518305ad2f56cbfd2a2', 'tests/test_mark_lbp_controller.py': '3286e1f57a8ee1625e2f8130150ed452593f3c99b5236fb00289fe234811f1e0', 'tests/test_vi_controller.py': '231e5db18a8ecbb598979bdfca73133ad55060c1ae0599a08171470bce91992c'}

def source_successor_predecessor_bytes(root,name,raw,canonical=False):
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
 if name=='tests/test_mark_lbp_controller.py'and len(raw)==25301 and hashlib.sha256(raw).hexdigest()=='167edc6fa02c7b48d2b9d5598d9107828e163d542406d3718ddbe6b7e98b9dbd':return raw
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
