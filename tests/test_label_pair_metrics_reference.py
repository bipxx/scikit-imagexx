"""Separate dependency-free manifest guards from installed fixture intake."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib
import pytest
from label_pair_metrics_contract import APIS,MAX_PIXELS,MAX_LABEL,cases,fixture
ROOT=Path(__file__).parents[1]
MATRIX=ROOT/'coverage/contracts/label-pair-metrics-v1.json'


def test_label_pair_manifest_and_source_order_without_science():
    data=json.loads(MATRIX.read_bytes());recipes=cases()
    assert data['cases']==recipes and len(recipes)==len({s['id']for s in recipes})==24
    assert [sum(s['api']==api for s in recipes)for api in APIS]==[8,8,8]
    for spec in recipes:
        assert len(spec['truth'])==len(spec['test'])<=MAX_PIXELS
        assert all(-1<=v<=MAX_LABEL for v in spec['truth']+spec['test'])
        assert spec.get('alpha',.5)in(-.1,0.,.25,.5,1.)
        assert spec.get('table')in(None,'counts-array','counts-matrix','normalized-array')
    for path,pin in data['source_freeze'].items():
        assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    assert data['source_sensitive_order']['normalization']=='per-pixel divide before duplicate coalescing; no count/N shortcut'
    assert data['source_sensitive_order']['VI_supplied_table']=='check_shape_equality is unconditional; None images are not silently accepted'


def test_label_pair_header_scope_without_science():
    text=(ROOT/'include/skimagexx/label_pair_metrics.hpp').read_text()
    assert 'bool label_pair_counts(const LabelPairInput&,LabelPairCsr&,int workers=0);'in text
    assert 'namespace skimagexx::detail'in text
    assert 'explicit zero'in text and 'integer_count/N'in text and 'unclosed ownership gates'in text
    assert data_status()in('PROSPECTIVE_PRECAPTURE_NOT_NATIVE_PARITY','ORIGINAL_CAPTURED_PENDING_NATIVE')


def data_status():return json.loads(MATRIX.read_bytes())['status']


def test_label_pair_saved_rows_without_replay():
    data=json.loads(MATRIX.read_bytes())
    if not data['original_observations']:
        assert data['status']=='PROSPECTIVE_PRECAPTURE_NOT_NATIVE_PARITY';return
    for version in ('1.24.4','2.5.4'):
        rows=data['original_observations'][version];assert [r['id']for r in rows]==[s['id']for s in cases()]
        pending=list(rows)
        while pending:
            value=pending.pop()
            if isinstance(value,dict):
                if 'raw_zlib_base64'in value:
                    raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']));assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
                pending.extend(value.values())
            elif isinstance(value,list):pending.extend(value)


@pytest.fixture(scope='module')
def reference_context():
    from label_pair_metrics_observer import reference_runtime
    return reference_runtime()


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_label_pair_fixture_without_metric_activity(spec,reference_context):
    from label_pair_metrics_observer import observation,without_reference_activity
    np,functions,_=reference_context
    with without_reference_activity(functions):
        args,options,buffers=fixture(spec,np)
        assert all(value.size<=MAX_PIXELS for value in buffers.values())
        assert all(value.dtype.kind in 'iuf'for value in buffers.values())
        if 'table'in options:
            table=options['table'];assert table.shape==(2,3)and len(table.data)==4
            assert table.indptr.tolist()==[0,2,4]and all(0<=v<3 for v in table.indices)
        if spec.get('images_none'):assert args==(None,None)
        else:assert args[0].size==args[1].size==len(spec['truth'])
        json.dumps(observation((args,options,buffers),np),allow_nan=False)
