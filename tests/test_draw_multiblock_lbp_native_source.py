"""Pure guards for saved-reference LBP successor; genuine delivery stays open."""
import ast,hashlib,json,runpy
from pathlib import Path
import pytest
from draw_multiblock_lbp_saved_parity import portable_record,run_saved_comparison
ROOT=Path(__file__).resolve().parents[1]


def test_native_source_and_historical_contract_closure():
    d=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-native-v2.json').read_bytes())
    assert not d['API_completion']and not d['registration_ready']and d['original_budget']==0
    assert hashlib.sha256((ROOT/'coverage/contracts/draw-multiblock-lbp-native-v1.json').read_bytes()).hexdigest()==d['historical_contract_sha256']
    adapter=runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp_execution.py'))
    context=json.loads((ROOT/adapter['CONTEXT']).read_bytes())
    for name,pin in d['source_raw_sha256'].items():
        if name=='src/bind_draw_multiblock_lbp.cpp':
            from draw_multiblock_lbp_metadata_context import historical_bytes
            raw=historical_bytes(ROOT,name)
        else:raw=adapter['historical_bytes'](ROOT,name)if name in context['historical_sources']else(ROOT/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==pin,name
    for name,pin in d['frozen_reference_raw_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name


def test_one_snapshot_complete_payload_and_ordered_stages():
    s=(ROOT/'src/draw_multiblock_lbp.cpp').read_text()
    assert s.count('get_num_threads()')==1 and 'chunk=detail::parallel_min-1'in s
    assert 'crop_copy_into('in s and 'convert_into('in s
    assert 'for(int block=0;block<8;++block)'in s and 'first_product_float32'in s
    assert s.index('const double complement')<s.index('std::vector<double> colors')<s.index('sum64(values.data()')
    assert 'SKIMAGEXX_LBP_SCALAR'in s and '_mm_mul_pd'in s and '_mm_add_pd'in s
    assert 'std::fma'not in s and 'fesetenv'not in s
    assert 'multiblock_lbp_binary('in s and 'multiblock_lbp_assign('in s
    assert 'detail::decode_half'in s and 'detail::encode_half'in s
    assert 'std::feclearexcept'in s and 'std::fetestexcept'in s


def test_frontend_signature_and_no_whole_original_forward():
    tree=ast.parse((ROOT/'python/skimagexx/_draw_multiblock_lbp.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='draw_multiblock_lbp')
    assert [a.arg for a in fn.args.args]==['image','r','c','width','height','lbp_code','color_greater_block','color_less_block','alpha']
    assert 'skimage.feature'not in ast.unparse(fn)and 'fallback('not in ast.unparse(fn)
    assert len(fn.args.defaults)==4
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)]
    numpy_calls={ast.unparse(n.func)for n in calls if isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='np'}
    assert numpy_calls<={'np.geterr','np.geterrcall','np.asarray','np.result_type','np.copy','np.empty_like','np.asanyarray','np.dtype'}
    assert {'_core._lbp_prepare','_core._lbp_assign','_core._lbp_binary'}<={ast.unparse(n.func)for n in calls}


def test_saved_parity_fails_closed_without_gold():
    with pytest.raises(AssertionError,match='UNBOUND'):run_saved_comparison(None,None,None,None,None)
    original={'pointer_delta':123,'pointer_delta_role':'nonsharing-diagnostic-only','control':1}
    assert portable_record(original)=={'pointer_delta_role':'nonsharing-diagnostic-only','control':1}and original['pointer_delta']==123


def test_four_benchmark_cases_and_method_unchanged():
    m=runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp_native.py'))
    old=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-v1.json').read_bytes())
    assert m['case_specs']()==old['benchmark_cases']and len(m['case_specs']())==4
    assert old['benchmark_method']['threads']==1 and old['benchmark_method']['repeat']==7
    text=(ROOT/'tools/montage_sustained.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    assert hashlib.sha256(''.join(text.splitlines(True)[node.lineno-1:node.end_lineno]).encode()).hexdigest()==old['benchmark_method']['method_sha256']


def test_saved_sdk_generator_is_raw_bound_and_not_target_fitted():
    source=(ROOT/'tests/draw_multiblock_lbp_sdk_saved.py').read_text()
    assert '1bcfff2be1b241b989a6142a756131474e89e756dd30047b0e653dde3d822726'in source
    assert "packed(row['result'])"in source and "packed(image)"in source
    assert 'np.result_type('in source and 'np.empty((0,)'in source
    assert 'import skimage'not in source and 'np.multiply('not in source
    assert "if row['error']is not None:continue"in source
    d=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-native-v2.json').read_bytes())
    assert d['SDK_saved_scope']['return_rows_each']==71 and d['SDK_saved_scope']['public_error_parity_claim']is False


def test_frontend_policy_and_source_order_are_explicit():
    tree=ast.parse((ROOT/'python/skimagexx/_draw_multiblock_lbp.py').read_text())
    text=ast.unparse(tree)
    assert text.index('greater = np.asarray')<text.index('less = np.asarray')<text.index('copied = np.copy')<text.index('gray = len(image.shape)')
    assert text.index('first = _binary')<text.index('second = alpha * selected')<text.index('combined = _binary')
    assert 'exceptions._UFuncOutputCastingError' in text
    assert 'np.seterr('not in text and 'errstate('not in text
    assert 'first_product_half'in(ROOT/'include/skimagexx/draw_multiblock_lbp.hpp').read_text()
