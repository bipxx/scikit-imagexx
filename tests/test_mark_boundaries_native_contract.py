"""Source/transport/benchmark guards; normal conftest/plugins, no science."""
import ast,base64,hashlib,json,runpy,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_all249_saved_formats_and_consumed_gap_preserved():
    d=json.loads((ROOT/'tests/fixtures/mark_boundaries_originals_v1.json').read_bytes());counts={'minimum':0,'modern':0}
    for name,payload in d['blobs'].items():
        raw=zlib.decompress(base64.b64decode(payload['zlib_base64']));assert len(raw)==payload['bytes']and hashlib.sha256(raw).hexdigest()==payload['sha256']
        if name.endswith('journal.jsonl'):
            rows=[json.loads(line)for line in raw.splitlines()];counts['minimum']+=sum(r['event']=='completed'for r in rows)
            assert len([r for r in rows if r['event']=='started'])==105
        else:counts['minimum'if'/minimum/'in name else'modern']+=len(json.loads(raw))
    assert counts==dict(minimum=124,modern=125)and 'gap'in d


def test_complete_CPP_operation_and_owned_stage_no_original_forwarding():
    header=(ROOT/'include/skimagexx/mark_boundaries.hpp').read_text();leaf=(ROOT/'src/mark_boundaries.cpp').read_text()
    assert 'MarkImage mark_boundaries('in header and 'MarkImage mark_boundaries('in leaf
    assert 'get_num_threads()'in leaf and 'workers=budget(workers);auto out=mark_boundaries_prepare'in leaf
    for forbidden in ('feholdexcept','fesetenv','feclearexcept','scipy.ndimage','PyImport_Import'):assert forbidden not in leaf
    source=(ROOT/'python/skimagexx/_mark_boundaries.py').read_text();tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='mark_boundaries')
    getters=[n for n in ast.walk(function)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='get_num_threads'];assert len(getters)==1
    assert 'fallback('not in source and 'skimage.segmentation'not in source


def test_frozen_benchmark_cases_method_and_pending_execution():
    data=runpy.run_path(str(ROOT/'benchmarks/mark_boundaries_consumer.py'))['plan']()
    original=runpy.run_path(str(ROOT/'benchmarks/mark_boundaries.py'))['case_specs']()
    assert data['cases']==original and len(original)==8 and data['root_GO']is None
    assert data['threads']==1 and data['repeat']==7 and data['target_ms']==100 and data['warmup_min_ms']==250 and data['retain_all_losses']
    assert data['status']=='UNBOUND_NO_EXECUTION'


def test_native_successor_honest_scope_and_historical_failure_retained():
    d=json.loads((ROOT/'coverage/contracts/mark-boundaries-native-v2.json').read_bytes())
    assert d['ready_for_shared_integration']is False and d['target_FP_correction']['case']=='mark_boundaries/color-tiny'
    assert d['original_calls_authorized']==0 and d['complete_saved_rows']==250
    assert hashlib.sha256((ROOT/'coverage/contracts/mark-boundaries-native-v1.json').read_bytes()).hexdigest()==d['historical_contract_raw_sha256']
    adapter=runpy.run_path(str(ROOT/'benchmarks/mark_boundaries_execution.py'))
    context=json.loads((ROOT/adapter['CONTEXT']).read_bytes())
    for name,pin in d['source_files_canonical_LF_sha256'].items():
        raw=adapter['historical_bytes'](ROOT,name)if name in context['historical_sources']else(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(raw).hexdigest()==pin
    for name,pin in d['donor_header_canonical_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


def test_fresh_witness_closes_125th_case_without_changing_oldraw():
    d=json.loads((ROOT/'tests/fixtures/mark_boundaries_originals_v2.json').read_bytes())
    assert hashlib.sha256((ROOT/d['predecessor']['path']).read_bytes()).hexdigest()==d['predecessor']['sha256']
    admission_raw=zlib.decompress(base64.b64decode(d['admission']['raw_zlib_base64']))
    assert hashlib.sha256(admission_raw).hexdigest()==d['admission']['raw_sha256']=='3eb0cdd022e24bda9a863219f2d2b7df7d67e1677ace896538f64b5cd1b0984f'
    witness_raw=zlib.decompress(base64.b64decode(d['witness']['raw_zlib_base64']));assert hashlib.sha256(witness_raw).hexdigest()==d['witness']['raw_sha256']
    admission=json.loads(admission_raw);assert admission['raw_files'][d['witness']['path']]['sha256']==d['witness']['raw_sha256']
    row,=json.loads(witness_raw);assert row['id']=='mark_boundaries/background-none'
    error=row['error'];assert error['type']=='UFuncTypeError'and error['qualname']=='_UFuncOutputCastingError'
    args=error['args']['items'];assert args[0]['name']=='bitwise_and'and args[1]=='same_kind'and args[2]['str']=='<i8'and args[3]['str']=='|b1'and args[4]==2
    assert d['complete_rows_per_profile']==125 and d['consumed_calls']==251 and d['old_failed_record_unchanged']


def test_palette_provider_is_consumed_not_a_flag_mask_or_image_algorithm():
    text=(ROOT/'python/skimagexx/_mark_boundaries.py').read_text();tree=ast.parse(text)
    paint=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_paint')
    stores=[n for n in ast.walk(paint)if isinstance(n,ast.Subscript)and isinstance(n.ctx,ast.Store)]
    assert len(stores)==1 and ast.unparse(stores[0].value)=='palette'
    source=ast.unparse(paint);assert source.index('palette = np.empty')<source.index('palette[np.ones')<source.index('values = palette[0].reshape')<source.index('_core.mark_boundaries_paint')
    assert 'feclearexcept'not in text and 'seterr('not in text and 'errstate('not in text and 'ctypes'not in text
    assert 'np.bitwise_and('not in text and "exceptions._UFuncOutputCastingError"in text
    assert 'if legacy_string and selected!=2:'in text
