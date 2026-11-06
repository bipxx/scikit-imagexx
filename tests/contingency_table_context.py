"""Exact current-to-admitted-ARE source inverse for five concrete consumers."""
from pathlib import Path
import hashlib
import json

CONTEXT_SHA256 = "8c65b7b4cd42bd8659a5538a985cb97284a1941e9d3e25c1877a3dc43611a9fc"
PATHS = (
    'include/skimagexx/label_pair_metrics.hpp', 'src/label_pair_metrics.cpp',
    'src/bind_label_pair_metrics.cpp', 'tests/cmake_consumer/label_pair_metrics_consumer.cpp',
    'tests/test_label_pair_metrics_reference.py',
)

def predecessor_bytes(root, path, *, source=None):
    raw = (Path(root)/path).read_bytes().replace(b'\r\n', b'\n') if source is None else source
    if path in ('python/skimagexx/_label_pair_metrics.py','tests/test_label_pair_rand_public.py'):
        context_raw=(Path(root)/'coverage/contracts/label-pair-ARE-metadata-context-v1.json').read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(context_raw).hexdigest()=='db0271f27b041cf17fb7d3e2a5f5843f3af8f2b2c813d5be4bb68c009a095773'
        context=json.loads(context_raw)
        assert context['schema']=='label-pair-ARE-metadata-context-v1' and context['predecessor']=='a53b62d5ad755742206e8f32ed103d2e5968032c'
        row=context['sources'][path]
        assert hashlib.sha256(raw).hexdigest()==row['current_canonical_lf_sha256']
        for fragment in row['inverse_fragments']:
            current,previous=fragment['current'].encode(),fragment['predecessor'].encode()
            assert current and raw.count(current)==1
            raw=raw.replace(current,previous,1)
        assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_lf_sha256']
        return raw
    if path not in PATHS:
        return raw  # The caller's unchanged historical pin still applies.
    context_raw = (Path(root)/'coverage/contracts/contingency-table-source-context-v1.json').read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(context_raw).hexdigest() == CONTEXT_SHA256
    context = json.loads(context_raw)
    assert context['schema'] == 'contingency-table-source-context-v1'
    assert context['predecessor'] == 'a53b62d5ad755742206e8f32ed103d2e5968032c'
    assert set(context['sources']) == set(PATHS)
    row = context['sources'][path]
    assert hashlib.sha256(raw).hexdigest() == row['current_canonical_lf_sha256']
    for fragment in row['inverse_fragments']:
        current, previous = fragment['current'].encode(), fragment['predecessor'].encode()
        assert current and raw.count(current) == 1
        raw = raw.replace(current, previous, 1)
    assert hashlib.sha256(raw).hexdigest() == row['predecessor_canonical_lf_sha256']
    return raw
