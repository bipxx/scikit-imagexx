"""Source/inert safeguards only; no NumPy/helper/native/original execution."""
import ast,hashlib,json,re
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTRACT='coverage/contracts/mark-wide-outer-source-v1.json'
DATA=json.loads((ROOT/CONTRACT).read_bytes())
H=lambda raw:hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('name',tuple(DATA['inverse']))
def test_exact_new_and_frozen_predecessor_source(name):
    row=DATA['inverse'][name];current=(ROOT/name).read_bytes()
    if name=='src/find_boundaries.cpp':
        phase=json.loads((ROOT/'coverage/contracts/mark-wide-comparison-fp-v1.json').read_bytes())
        assert H(current)==phase['kernel_current_raw_sha256']
        for fragment in phase['inverse_fragments']:
            a,b=fragment['current'].encode(),fragment['predecessor'].encode()
            assert current.count(a)==1;current=current.replace(a,b,1)
    assert current.decode()==row['current']and H(current)==row['current_raw_sha256']
    assert H(row['predecessor'].encode())==row['predecessor_raw_sha256']


def test_new_kernel_delta_preserves_every_previous_algorithm_body():
    row=DATA['inverse']['src/find_boundaries.cpp'];old=row['predecessor'].replace('\r\n','\n');new=row['current']
    start=new.index('// The source gray morphology stages typed data through binary64')
    stop=new.index('// Independent 2D interior pixels:',start)
    inverse=new[:start]+new[stop:]
    inverse=inverse.replace('#include <cfenv>\n#if defined(__GNUC__) && defined(__x86_64__)\n#include <emmintrin.h>\n#endif\n','')
    current='            if(parameters.mode==FindBoundaryMode::outer)\n                return wide_source_outer<T>(view,parameters,output,strides,input_count);'
    assert inverse.count(current)==1;inverse=inverse.replace(current,'            if(parameters.mode==FindBoundaryMode::outer)return false;')
    assert inverse==old
    assert new.count('get_num_threads(')==old.count('get_num_threads(')


def test_frontend_body_and_SDK_layout_unchanged():
    row=DATA['inverse']['python/skimagexx/_find_boundaries.py']
    def body(text):
        tree=ast.parse(text);function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
        assert isinstance(function.body[0],ast.Expr)and isinstance(function.body[0].value,ast.Constant)
        function.body=function.body[1:]
        return ast.dump(tree,include_attributes=False)
    assert body(row['current'])==body(row['predecessor'])
    row=DATA['inverse']['include/skimagexx/find_boundaries.hpp']
    remove=lambda text:re.sub(r'//[^\n]*','',text).replace('\r\n','\n')
    assert remove(row['current']).split()==remove(row['predecessor']).split()


def test_full_64bit_words_and_typed_axis_stores_no_case_fitting():
    source=DATA['inverse']['src/find_boundaries.cpp']['current']
    start=source.index('// The source gray morphology stages typed data through binary64');stop=source.index('// Independent 2D interior pixels:',start);block=source[start:stop]
    assert 'original[i]=load<T>(view,point)'in block
    assert 'std::numeric_limits<T>::max()'in block
    assert 'wide_source_store<T>(result)'in block and 'current.swap(next)'in block
    assert 'parameters.connectivity==rank'in block and 'axis<passes'in block
    assert '(high[i]!=low[i])&&(bg||adjacent)'in block
    assert 'std::vector<T> original'in block and 'auto inverted=original'in block
    assert not any(s in block for s in('int32_t','uint32_t','gray-u8-outer','%7','512','513','skimage','fallback'))
    assert 'INT64_MIN..INT64_MAX'in DATA['domain']['input_value_ranges']
    assert '0..UINT64_MAX'in DATA['domain']['input_value_ranges']


def test_FP_is_source_operation_not_a_normalized_acceptance_mask():
    source=DATA['inverse']['src/find_boundaries.cpp']['current']
    start=source.index('// The source gray morphology stages typed data through binary64');stop=source.index('// Independent 2D interior pixels:',start);block=source[start:stop]
    assert '_mm_cvtsi64_sd'in block and '_mm_cvttsd_si64'in block
    assert 'std::fegetround()!=FE_TONEAREST'in block and '(control&0x1f80u)!=0x1f80u'in block
    assert not any(s in block for s in('feclearexcept','fesetenv','fesetround','_mm_setcsr','np.seterr','expected','gold'))
    assert 'source-only'in DATA['domain']['FP']or 'pending'in DATA['domain']['FP']
    assert 'GNU x86_64'in DATA['domain']['platform']


def test_work_and_memory_bounds_remain_explicit_source_metadata():
    source=DATA['inverse']['src/find_boundaries.cpp']['current']
    assert 'size>find_boundaries_max_neighbor_visits/visits'in source
    assert 'parameters.connectivity==rank?6*rank:2*near.size()'in source
    case=DATA['resource_bounds']['benchmark_first_case']
    assert case['n']==512*513 and case['named_bytes_upper']==57*case['n']
    assert case['extrema_visits']==22*case['n']<64*1024*1024
    assert 'serial'in DATA['domain']['workers']


def test_actual_pinned_primary_sources_and_per_axis_rule():
    # Source text only; these retained locations are authenticated by author
    # evidence. Shipping source contract retains their identities, not imports.
    for row in DATA['source_prerequisites']['SciPy_source']:
        assert len(row['sha256'])==64
    assert len(DATA['source_prerequisites']['SciPy_source'])==6
    assert 'typed store after every axis'in ' '.join(DATA['algorithm'])
    assert DATA['source_prerequisites']['boundaries']['canonical_LF_sha256']=='fbdbbf36d8813c09d00583d235063acb0911d842cd94628d482d5c5685818131'


def test_benchmark_arguments_and_case_pool_not_changed_to_evade_failure():
    from tools.mark_lbp_fp_context import predecessor_bytes
    for name,pin in DATA['benchmark_immutable'].items():assert H(predecessor_bytes(ROOT,name,(ROOT/name).read_bytes()))==pin
    source=(ROOT/'benchmarks/mark_boundaries_execution.py').read_text()
    assert 'labels=np.arange(h*w,dtype=np.int64).reshape(h,w)%7'in source
    assert DATA['base']=='dda9cc83d9434a10616f84cf1e1125d93f5f36fd'
    assert DATA['status']=='SOURCE_IMPLEMENTED_INERT_ONLY_UNREADY_NO_RUNTIME_ORIGINAL_AUTHORITY'
    assert any('No compile/native/scientific/helper/original/timing execution'in s for s in DATA['pending'])



def test_new_wide_background_boundary_retains_unrepresented_metadata_frontier():
    row=DATA['inverse']['python/skimagexx/_mark_boundaries.py'];source=row['current']
    a=source.index('    # A wide outer comparison against a floating/out-of-range background can')
    b=source.index('    try:mask=_core.mark_boundaries_mask',a)
    assert source[:a]+source[b:]==row['predecessor'].replace('\r\n','\n')
    assert 'bits==64 and selected==2'in source[a:b]and 'type(background_label)not in(int,bool)or not representable'in source[a:b]
    kernel=DATA['inverse']['src/find_boundaries.cpp']['current']
    assert 'if(!parameters.background_representable)return false;'in kernel


def test_current_comparison_phase_matches_pinned_NumPy_conditional_clear():
    phase=json.loads((ROOT/'coverage/contracts/mark-wide-comparison-fp-v1.json').read_bytes())
    raw=(ROOT/'src/find_boundaries.cpp').read_bytes();assert H(raw)==phase['kernel_current_raw_sha256']
    source=raw.decode();start=source.index('template<class T>bool wide_source_outer(');stop=source.index('// Independent 2D interior pixels:',start)
    block=source[start:stop]
    after=block.index('const auto full_low=wide_source_extrema(')
    clear=block.index('if(std::fetestexcept(comparison_errors))std::feclearexcept(comparison_errors);')
    loop=block.index('for(std::size_t i=0;i<size;++i) {',clear)
    assert after<clear<loop<block.index('return true;',loop)
    assert 'comparison_errors=FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID;'in block
    assert 'FE_INEXACT'not in block and source.count('std::feclearexcept(')==1
    assert all(s not in block for s in ('fesetenv','fesetround','_mm_setcsr','expected','gold'))
    assert phase['primitive_calls']=={'minimum':4,'modern':4}and phase['new_original_calls']==0


def test_deferred_native_unit_uses_whole_operation_and_authentic_two_masks():
    phase=json.loads((ROOT/'coverage/contracts/mark-wide-comparison-fp-v1.json').read_bytes())
    source=(ROOT/'tests/test_find_boundaries_wide_comparison_fp.cpp').read_bytes()
    assert H(source)==phase['native_test']['raw_sha256']
    text=source.decode();assert 'find_boundaries_into('in text and 'std::memcmp(output.data()+1,expected,12)'in text
    assert 'actual_after' in text and 'semantic(actual_after,expected_after)'in text
    assert 'env.restore(caller)'in text and 'env.finish()'in text
    assert 'reference_fenv_restore'in text and 'std::fesetenv'not in text
    assert all(n in text for n in ('budget:{1,4}','UINT64_C(18446744073709551615)'))
    assert phase['native_test']['execution']=='PENDING_ROOT_COMPILE_NATIVE_GO'
