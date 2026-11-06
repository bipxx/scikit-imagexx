"""Exact JOIN additive source projection for immutable predecessor guards."""
from pathlib import Path
import hashlib
import json
CONTEXT_SHA256='15b413d6a5b3260497654349d79c20f52f0e5fdd16468cdb1011eb6e0ea55672'


def predecessor_bytes(root,name,raw):
    raw=raw.replace(b'\r\n',b'\n')
    successor=(Path(root)/'coverage/contracts/util-metadata-additive-context-v1.json').read_bytes()
    assert hashlib.sha256(successor).hexdigest()=='1ca834516792f2130f9e9dfb38ba73de76fbbfa2b0b33ea90d457d6f38d9acbd'
    successor=json.loads(successor)
    assert successor['schema']=='util-metadata-additive-context-v1'
    assert successor['predecessor_source']=='8c22dd6a729ac96e7b84fdd261294601a98748e3'
    assert set(successor['sources'])=={'CMakeLists.txt','src/bindings.cpp','python/skimagexx/coverage.py'}
    if name in successor['sources']:
        row=successor['sources'][name]
        assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256'],name
        for fragment in row['inverse_fragments']:
            current,previous=fragment['current'].encode(),fragment['predecessor'].encode()
            assert current and raw.count(current)==1,name
            raw=raw.replace(current,previous,1)
        assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256'],name
    path=Path(root)/'coverage/contracts/join-additive-source-context-v1.json'
    data=path.read_bytes()
    assert hashlib.sha256(data).hexdigest()==CONTEXT_SHA256
    context=json.loads(data)
    assert context['schema']=='join-additive-source-context-v1'
    assert context['predecessor_source']=='f13ae5ba01680c40f8ee0ae42153bd9480119ab1'
    assert set(context['sources'])=={'CMakeLists.txt','src/bindings.cpp','python/skimagexx/coverage.py','benchmarks/draw_polygon.py'}
    if name not in context['sources']:return raw
    row=context['sources'][name]
    assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256'],name
    for fragment in row['inverse_fragments']:
        current,previous=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and raw.count(current)==1,name
        raw=raw.replace(current,previous,1)
    assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256'],name
    return raw
