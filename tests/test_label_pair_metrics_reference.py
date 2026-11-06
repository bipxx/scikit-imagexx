"""Separate dependency-free manifest guards from installed fixture intake."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib
import pytest
from label_pair_metrics_contract import APIS,MAX_PIXELS,MAX_LABEL,cases,fixture
from contingency_table_context import predecessor_bytes
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
        assert hashlib.sha256(predecessor_bytes(ROOT,path)).hexdigest()==pin
    assert data['source_sensitive_order']['normalization']=='per-pixel divide before duplicate coalescing; no count/N shortcut'
    assert data['source_sensitive_order']['VI_supplied_table']=='check_shape_equality is unconditional; None images are not silently accepted'


def test_label_pair_header_scope_without_science():
    text=(ROOT/'include/skimagexx/label_pair_metrics.hpp').read_text()
    assert 'bool label_pair_counts(const LabelPairInput&,LabelPairCsr&,int workers=0);'in text
    assert 'namespace skimagexx::detail'in text
    assert 'explicit zero'in text and 'integer_count/N'in text
    assert 'bool label_pair_rand_terms(const LabelPairInput&,LabelPairRandTerms&,int workers=0);'in text
    assert 'No generic sparse engine or VI reduction/log claim'in text
    rand=json.loads((ROOT/'coverage/contracts/label-pair-rand-native-v1.json').read_bytes())
    assert rand['accuracy'].startswith('Exact0ULP')and rand['partition'].startswith('table=None')
    for path,pin in rand.get('current_source_pins',{}).items():
        assert hashlib.sha256(predecessor_bytes(ROOT,path)).hexdigest()==pin
    tree=ast.parse((ROOT/'python/skimagexx/_label_pair_metrics.py').read_bytes())
    public=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='adapted_rand_error')
    assert [a.arg for a in public.args.args]==['image_true','image_test']
    assert [a.arg for a in public.args.kwonlyargs]==['table','ignore_labels','alpha']
    assert public.args.vararg is None and public.args.kwarg is None
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


@pytest.mark.parametrize('api',APIS)
def test_label_pair_per_function_saved_contract_without_science(api):
    data=json.loads(MATRIX.read_bytes())
    for version in ('1.24.4','2.5.4'):
        rows=[r for r in data['original_observations'][version]if r['id'].startswith(api.split('.')[1]+'/')]
        assert len(rows)==8 and all(all(r['checks'].values())for r in rows)
        returned=[r['result']for r in rows if r['error']is None]
        if api=='metrics.contingency_table':
            assert len(returned)==5
            for value in returned:
                assert value['format']=='csr'and value['dtype']=='<f8'
                assert value['data']['shape']==value['indices']['shape']==[value['nnz']]
                assert value['data']['base_shape']==value['indices']['base_shape']==[6]
                assert value['indices']['dtype']==value['indptr']['dtype']=='<i4'
        elif api=='metrics.adapted_rand_error':
            assert len(returned)==6
            assert all(v['container']=='tuple'and len(v['items'])==3 for v in returned)
            assert all(x['type']=='float64'and x['dtype']=='<f8'and len(x['raw'])==16 for v in returned for x in v['items'])
        else:
            assert len(returned)==7
            assert all(v['type']=='ndarray'and v['shape']==[2]and v['dtype']=='<f8'and v['own']and v['base_type']=='NoneType'for v in returned)


def test_label_pair_profile_specific_supplied_table_mutation_without_science():
    data=json.loads(MATRIX.read_bytes());old=data['original_observations']['1.24.4'];new=data['original_observations']['2.5.4']
    changed=('adapted_rand_error/table-array','adapted_rand_error/table-matrix')
    assert [a['id']for a,b in zip(old,new)if a!=b]==list(changed)
    for a,b in zip(old,new):
        assert a['result']==b['result']and a['error']==b['error']and a['warnings']==b['warnings']and a['events']==b['events']
        assert not a['input_state_changed']
        assert b['input_state_changed']==(b['id']in changed)
        if b['id']in changed:
            before=b['before']['items'][1]['table'];after=b['after']['items'][1]['table']
            assert before['cached_sorted']is None and before['cached_canonical']is None
            assert after['cached_sorted']is True and after['cached_canonical']is True
            assert before['data']['raw_sha256']!=after['data']['raw_sha256']
            assert before['indices']['raw_sha256']!=after['indices']['raw_sha256']


def test_label_pair_saved_FP_and_error_prefix_without_science():
    data=json.loads(MATRIX.read_bytes())
    for rows in data['original_observations'].values():
        assert sum(r['error']is not None for r in rows)==6
        assert sum(len(r['events'])for r in rows)==5 and sum(len(r['warnings'])for r in rows)==4
        singleton=next(r for r in rows if r['id']=='adapted_rand_error/singleton')
        assert singleton['events']==[dict(event='FP-handler',message='invalid value',flags=8)]*3
        table=next(r for r in rows if r['id']=='adapted_rand_error/table-matrix')
        assert table['error']['type']=='ValueError'and 'matmul:'in table['error']['message']
        none=next(r for r in rows if r['id']=='variation_of_information/table-with-None-images')
        assert none['error']==dict(type='AttributeError',module='builtins',message="'NoneType' object has no attribute 'shape'")


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


@pytest.mark.parametrize('path',('include/skimagexx/label_pair_metrics.hpp',
    'src/label_pair_metrics.cpp','src/bind_label_pair_metrics.cpp',
    'tests/cmake_consumer/label_pair_metrics_consumer.cpp',
    'tests/test_label_pair_metrics_reference.py'))
def test_contingency_context_rejects_tampered_source(path):
    current=(ROOT/path).read_bytes().replace(b'\r\n',b'\n')
    with pytest.raises(AssertionError):predecessor_bytes(ROOT,path,source=current+b'\nchanged')
    if path=='src/label_pair_metrics.cpp':
        changed=current.replace(b'scalar_result(1.-value,output)',b'scalar_result(2.-value,output)')
        assert changed!=current
        with pytest.raises(AssertionError):predecessor_bytes(ROOT,path,source=changed)
