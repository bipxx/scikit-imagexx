"""Authenticate the fixed metrics closure through exact predecessor source bytes."""
from pathlib import Path
import hashlib
import json


def validate_source_closure(root, contract):
    import importlib.util
    helper_spec=importlib.util.spec_from_file_location('join_metric_source_context',Path(root)/'tools/join_source_context.py')
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    root=Path(root)
    context_raw=(root/'coverage/contracts/label-pair-metrics-benchmark-context-v1.json').read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(context_raw).hexdigest()=='d380c13f49e7c59c6fdad0aae9bb5dddbf88a6f338a657fee463aba6b3d229a1'
    context=json.loads(context_raw)
    assert context['schema']=='label-pair-metrics-benchmark-context-v1'
    contract_raw=(root/context['contract']).read_bytes()
    assert hashlib.sha256(contract_raw).hexdigest()==context['contract_raw_sha256']=='5b4f3935966aa048eb2d06666a1ea6f548a43d08edd7d43ff68ccd8a211bdfa6'
    assert json.loads(contract_raw)==contract
    assert set(context['sources'])=={'python/skimagexx/coverage.py','benchmarks/label_pair_metrics.py'}
    predecessors={}
    for path,row in context['sources'].items():
        raw=(root/path).read_bytes().replace(b'\r\n',b'\n')
        raw=helper.predecessor_bytes(root,path,raw)
        assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256'],path
        for fragment in row['inverse_fragments']:
            current,previous=fragment['current'].encode(),fragment['predecessor'].encode()
            assert current and raw.count(current)==1
            raw=raw.replace(current,previous,1)
        assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256'],path
        predecessors[path]=raw
    for path,pin in contract['source_closure'].items():
        raw=predecessors[path]if path in predecessors else(root/path).read_bytes().replace(b'\r\n',b'\n')
        if path=='src/label_pair_metrics.cpp':raw=helper.predecessor_bytes(root,path,raw)
        assert hashlib.sha256(raw).hexdigest()==pin,path
