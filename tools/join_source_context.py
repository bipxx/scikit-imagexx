"""Exact JOIN additive source projection for immutable predecessor guards."""
from pathlib import Path
import hashlib
import json
CONTEXT_SHA256='15b413d6a5b3260497654349d79c20f52f0e5fdd16468cdb1011eb6e0ea55672'


def sdk_family_assertion_predecessor_bytes(root,name,raw):
    """Strict assertion-only source successor, before every shared historical entry."""
    path=Path(root)/'coverage/contracts/current281-sdk-family-source-v2.json'
    if not path.is_file():return raw
    context=json.loads(path.read_bytes())
    assert context['predecessor_source']=='cd703fbda0442944b150c9529e09e0a40ab37c59'
    if name not in context['sources']:return raw
    row=context['sources'][name];raw=raw.replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256'],name
    for fragment in row['inverse_fragments']:
        current,old=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and raw.count(current)==1,name
        raw=raw.replace(current,old,1)
    assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256'],name
    return raw


def integration_fix_predecessor_bytes(root,name,raw):
    """Exact three integration fixes before frozen188; no numerical projection."""
    raw=sdk_family_assertion_predecessor_bytes(root,name,raw)
    successor=Path(root)/'coverage/contracts/current281-projection-v2.json'
    if successor.is_file():
        layer=json.loads(successor.read_bytes())
        assert layer['predecessor_source']=='cbb3d2cd993f71fff08bb4a548d9c456c7d2de60'
        if name in layer['sources']:
            raw=raw.replace(b'\r\n',b'\n');row=layer['sources'][name]
            assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256'],name
            for fragment in row['inverse_fragments']:
                current,old=fragment['current'].encode(),fragment['predecessor'].encode()
                assert current and raw.count(current)==1,name
                raw=raw.replace(current,old,1)
            assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256'],name
    path=Path(root)/'coverage/contracts/current281-integration-fixes-v1.json'
    if not path.is_file():return raw
    context=json.loads(path.read_bytes())
    assert context['schema']=='current281-integration-fixes-v1'
    assert context['predecessor_source']=='188a91ef31609c796d4d83e8957c772544f31428'
    if name not in context['sources']:return raw
    raw=raw.replace(b'\r\n',b'\n');row=context['sources'][name]
    assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256'],name
    for fragment in row['inverse_fragments']:
        current,old=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and raw.count(current)==1,name
        raw=raw.replace(current,old,1)
    assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256'],name
    return raw


def rolling_predecessor_bytes(root,name,raw):
    """Exact enabled rolling-ball additions before the admitted a5 source."""
    return _rolling188_predecessor_bytes(root,name,integration_fix_predecessor_bytes(root,name,raw))


def _rolling188_predecessor_bytes(root,name,raw):
    raw=raw.replace(b'\r\n',b'\n')
    path=Path(root)/'coverage/contracts/rolling-ball-enabled-context-v1.json'
    if not path.is_file():return raw
    context=json.loads(path.read_bytes())
    assert context['predecessor_source']=='a5e019c6a4b0fff3a18635ab0e2f552986abba3e'
    if name not in context['sources']:return raw
    row=context['sources'][name]
    assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256'],name
    for fragment in row['inverse_fragments']:
        current,old=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and raw.count(current)==1,name
        raw=raw.replace(current,old,1)
    assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256'],name
    return raw


def rolling_predecessor_raw_bytes(root,name,raw):
    """Raw domain for unchanged historical controller files; edited shared files are LF."""
    raw=integration_fix_predecessor_bytes(root,name,raw)
    context=json.loads((Path(root)/'coverage/contracts/rolling-ball-enabled-context-v1.json').read_bytes())
    if name not in context['sources']:return raw
    row=context['sources'][name]
    assert row['predecessor_raw_sha256']==row['predecessor_canonical_lf_sha256']
    assert hashlib.sha256(raw).hexdigest()==row['current_raw_sha256'],name
    return _rolling188_predecessor_bytes(root,name,raw)


def correction_predecessor_bytes(root,name,raw):
    """Authenticate the correction, then reconstruct the frozen ab78 input."""
    root=Path(root)
    raw=rolling_predecessor_bytes(root,name,raw)
    if name=='src/variation_of_information.cpp':
        import importlib.util
        path=root/'tests/variation_of_information_reduction_contract.py'
        spec=importlib.util.spec_from_file_location('_vi_reduction_projection',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        raw=module.historical_vi_source(raw)
    raw=raw.replace(b'\r\n',b'\n')
    path=root/'coverage/contracts/vi-reduction-integration-v2.json'
    if not path.is_file():return raw
    context=json.loads(path.read_bytes())
    assert context['predecessor_source']=='ab7872a5483a5d5e3660ba20fdd873d78463a1b5'
    if name not in context['sources']:return raw
    row=context['sources'][name]
    assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256'],name
    for fragment in row['inverse_fragments']:
        current,old=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and raw.count(current)==1,name
        raw=raw.replace(current,old,1)
    assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256'],name
    return raw


def enabled_predecessor_bytes(root,name,raw):
    """One exact admitted enabled-family layer before historical shared guards."""
    raw=correction_predecessor_bytes(root,name,raw)
    raw=raw.replace(b'\r\n',b'\n')
    current=(Path(root)/'coverage/contracts/abi-vi-cumulative-integration-v1.json').read_bytes()
    assert hashlib.sha256(current).hexdigest()=='beeb170a40af8b8710ad0d6c7d8fd5d6d571933b20beac65b4515a018fdfe16d'
    layer=json.loads(current)
    assert layer['predecessor_source']=='8c4fb4307746a0eb5fac5d125b6384fa80e88621'
    if name in layer['sources']:
        entry=layer['sources'][name]
        assert hashlib.sha256(raw).hexdigest()==entry['current_canonical_lf_sha256'],name
        for fragment in entry['inverse_fragments']:
            now,previous=fragment['current'].encode(),fragment['predecessor'].encode()
            assert now and raw.count(now)==1,name
            raw=raw.replace(now,previous,1)
        assert hashlib.sha256(raw).hexdigest()==entry['predecessor_canonical_lf_sha256'],name
    active=(Path(root)/'coverage/contracts/mark-lbp-radon-util-additive-context-v1.json').read_bytes()
    assert hashlib.sha256(active).hexdigest()=='47cee57ec955fbbe7cab192d6e948a1db0c67ceb0b71342a945f49e7fde1dd7a'
    successor=json.loads(active)
    assert successor['predecessor_source']=='2991881411972e6266f7e132c9dca8b7226a8567'
    if name in successor['sources']:
        row=successor['sources'][name]
        assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256'],name
        for f in row['inverse_fragments']:
            current,old=f['current'].encode(),f['predecessor'].encode()
            assert current and raw.count(current)==1,name
            raw=raw.replace(current,old,1)
        assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256'],name
    body=(Path(root)/'coverage/contracts/mark-lbp-enabled-additive-context-v1.json').read_bytes()
    assert hashlib.sha256(body).hexdigest()=='79de3c7ef15cf213d5639b915956a077b16cc897c356640a9ce91d7bbcdafb5e'
    context=json.loads(body)
    assert context['predecessor_source']=='a6bae9ac86dbba362537758784edf06c973e9c53'
    assert set(context['sources'])=={'CMakeLists.txt','src/bindings.cpp','python/skimagexx/coverage.py'}
    if name not in context['sources']:return raw
    row=context['sources'][name]
    assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256'],name
    for f in row['inverse_fragments']:
        current,old=f['current'].encode(),f['predecessor'].encode()
        assert current and raw.count(current)==1,name;raw=raw.replace(current,old,1)
    assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256'],name
    return raw


def predecessor_bytes(root,name,raw):
    raw=enabled_predecessor_bytes(root,name,raw)
    raw=raw.replace(b'\r\n',b'\n')
    current_raw=(Path(root)/'coverage/contracts/util-views-density-additive-context-v1.json').read_bytes()
    assert hashlib.sha256(current_raw).hexdigest()=='608865130cc3ff264c695ead40eb10f9d102c5cca4f6440d696b22aefa754c1a'
    current_context=json.loads(current_raw)
    assert current_context['predecessor_source']=='277ffc3ca858eaeae1d08fd9c22b217aa1c8b459'
    if name in current_context['sources']:
        row=current_context['sources'][name]
        assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256'],name
        for fragment in row['inverse_fragments']:
            current,previous=fragment['current'].encode(),fragment['predecessor'].encode()
            assert current and raw.count(current)==1,name
            raw=raw.replace(current,previous,1)
        assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256'],name

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
