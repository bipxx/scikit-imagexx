"""Saved-data/source checks only; no metric/helper/provider computations."""
import ast
import builtins
import importlib.util
import sys
import types
import base64
import hashlib
import json
from pathlib import Path
import struct
import zlib
import pytest
from variation_of_information_source_context import historical_source
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/variation-of-information-native-v1.json'


@pytest.mark.parametrize('name',('include/skimagexx/variation_of_information.hpp',
    'src/variation_of_information.cpp','tests/fixtures/variation_of_information_saved_v1.hpp',
    'tests/test_variation_of_information_native.cpp'))
def test_exact_prototype_source_words(name):
    data=json.loads(CONTRACT.read_bytes())
    assert hashlib.sha256(historical_source(ROOT,name)).hexdigest()==data['source_raw_sha256'][name]


def test_all_eight_saved_rows_both_profiles_unchanged():
    contract=json.loads(CONTRACT.read_bytes());raw=(ROOT/contract['original_contract']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==contract['original_contract_raw_sha256']
    data=json.loads(raw);rows={v:[r for r in data['original_observations'][v]if r['id'].startswith('variation_of_information/')]for v in contract['retained_profiles']}
    assert len(rows['1.24.4'])==8 and rows['1.24.4']==rows['2.5.4']
    fixture=(ROOT/'tests/fixtures/variation_of_information_saved_v1.hpp').read_text()
    for row in rows['1.24.4']:
        if row['error'] is not None:
            assert row['id']=='variation_of_information/table-with-None-images'
            assert row['error']==dict(type='AttributeError',module='builtins',message="'NoneType' object has no attribute 'shape'")
            continue
        value=row['result'];raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
        assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
        assert fixture.count('"'+row['id']+'"')==1
        for word in struct.unpack('<QQ',raw):assert 'UINT64_C(0x%016x)'%word in fixture
        if row['id'].endswith('/allignored'):
            assert struct.unpack('<QQ',raw)==(0xfff8000000000000,)*2
            assert row['events']==[dict(event='FP-handler',message='invalid value',flags=8)]


def test_strict_failure_history_preserved_without_tolerance_or_completion():
    data=json.loads(CONTRACT.read_bytes());assert data['status'].startswith('PRIVATE_SAVED8')
    assert data['API_completion']is False and data['advertised_in_registry']is False
    assert data['actual_native_viability']['strict_pass']==7 and data['actual_native_viability']['strict_fail']==0
    assert '6rawPASS1FAIL'in data['failure_history']['native-v2']
    assert data['original_calls']==data['timed_batches']==0
    assert data['accuracy'].startswith('Exact0ULP')
    client=(ROOT/'tests/test_variation_of_information_native.cpp').read_text()
    assert 'word==test.words[i]'in client and 'return failures?1:0;'in client
    source=(ROOT/'src/variation_of_information.cpp').read_text()
    assert 'std::log2(value)'in source and 'label_pair_counts(normalized,table,snapshot)'in source
    assert 'nan('not in source and 'copysign'not in source and '0xfff8000000000000'not in source
    tree=ast.parse((ROOT/'python/skimagexx/_variation_of_information.py').read_bytes())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='variation_of_information')
    assert [a.arg for a in fn.args.args]==['image0','image1']
    assert [a.arg for a in fn.args.kwonlyargs]==['table','ignore_labels']
    assert fn.args.vararg is None and fn.args.kwarg is None


def test_operation_reuses_count_types_and_keeps_ordered_core():
    header=(ROOT/'include/skimagexx/variation_of_information.hpp').read_text()
    assert '#include <skimagexx/label_pair_metrics.hpp>'in header
    assert 'VariationTableView' in header and 'detail::LabelPairInput' in header
    assert 'std::vector<double> data'not in header
    source=(ROOT/'src/variation_of_information.cpp').read_text()
    assert 'get_num_threads():workers'in source and 'normalized.normalize=true'in source
    assert 'omp parallel'not in source and 'omp simd'not in source


def test_all_family_source_and_structural_providers_are_explicit():
    data=json.loads(CONTRACT.read_bytes())
    for name,pin in data['source_raw_sha256'].items():assert hashlib.sha256(historical_source(ROOT,name)).hexdigest()==pin,name
    text=(ROOT/'python/skimagexx/_variation_of_information.py').read_text()
    assert 'contingency_table(image0,image1,ignore_labels=ignore_labels,normalize=True)'in text
    assert 'table.data' in text and 'table.indices'in text and 'table.indptr'in text
    assert 'np.log'not in text and '@'not in text and '.sum('not in text
    assert 'if image0.shape!=image1.shape' in text
    body=(ROOT/'src/bind_variation_of_information.cpp').read_text()
    assert 'variation_of_information_csr' in body and 'py::array_t<double> out(2)'in body
    assert 'noconvert()'in body and 'check_array_storage' in body


def benchmark_module():
    path=ROOT/'benchmarks/variation_of_information.py'
    spec=importlib.util.spec_from_file_location('vi_prospective_benchmark',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_prospective_benchmark_case_pool_and_plan(monkeypatch,capsys):
    module=benchmark_module();data=json.loads(module.CONTRACT.read_bytes())
    assert len(module.case_specs())==4 and len({s['id']for s in module.case_specs()})==4
    assert all(v is None for v in data['binding'].values())
    assert data['method']['threads']==1 and data['method']['repeat']==7
    assert data['method']['target_ms']==100 and data['method']['warmup_min_ms']==250 and data['method']['retain_all_losses']
    old=data['shared_fixture'];assert hashlib.sha256((ROOT/old['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==old['canonical_sha256']
    for s in module.case_specs():assert module.source_safe(s)['pixels']<=262144
    monkeypatch.setattr(sys,'argv',['vi-benchmark','--plan']);module.main();plan=json.loads(capsys.readouterr().out)
    assert plan['cases']==data['cases']and len(plan['bounds'])==4


@pytest.mark.parametrize('cached',(False,True))
def test_actual_benchmark_unbound_refuses_cold_cached_science(monkeypatch,cached):
    module=benchmark_module();seen=[];previous=builtins.__import__
    if cached:monkeypatch.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
    def guard(name,*a,**k):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):seen.append(name);raise AssertionError('Forbidden scientific REQUEST')
        return previous(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guard);monkeypatch.setattr(sys,'argv',['vi-benchmark'])
    with pytest.raises(SystemExit,match='VI_UNBOUND'):module.main()
    assert not seen
