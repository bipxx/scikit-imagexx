"""Source/storage/inert SDK-client controls; no compiler/helper/native/FP calls."""
import ast,copy,hashlib,importlib.util,json,struct,zlib,base64
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('mark_wide_sdk_words',ROOT/'tests/mark_wide_outer_sdk.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
C=json.loads((ROOT/'coverage/contracts/mark-wide-outer-sdk-v1.json').read_bytes())
CPP=ROOT/C['client']
H=lambda raw:hashlib.sha256(raw).hexdigest()


def test_generated_header_is_exact_transport_of_admitted_raw_words():
    rows=M.observations();assert (ROOT/C['generated_fixture']).read_text()==M.render()
    assert {r['profile']for r in rows}=={'minimum','modern'}
    assert sum(len(r['result_bytes'])for r in rows)==912
    for row in rows:
        assert H(row['result_bytes'])==row['result']['raw_sha256']
        assert len(row['label_bytes'])==96
        if row['mark']:assert len(row['result_bytes'])in(144,288)and len(row['image_bytes'])in(12,144)
        else:assert len(row['result_bytes'])==12 and set(row['result_bytes'])<={0,1}
        assert set(row['fp']['before'])==set(row['fp']['after'])=={'control','status','mxcsr'}
    # The original rawtransport remains byteexact, not an invented C++ oracle.
    raw=json.loads((ROOT/C['raw_transport']).read_bytes())
    assert C['raw_admission']==raw['root_admission']and raw['original_budget_remaining']==0


@pytest.mark.parametrize('fault',('words','size','type'))
def test_literal_payload_tamper_is_rejected(fault):
    value=copy.deepcopy(M.observations()[0]['labels'])
    if fault=='words':value['raw_zlib_base64']=base64.b64encode(zlib.compress(b'X'*96)).decode()
    if fault=='size':value['shape']=[3,5]
    if fault=='type':value['dtype']='O'
    with pytest.raises(AssertionError):M.payload(value)


@pytest.mark.parametrize('group',('dependencies_raw_sha256','source_raw_sha256'))
def test_old_sdk_native_and_current_literal_sources_are_pinned(group):
    for name,pin in C[group].items():
        raw=(ROOT/name).read_bytes()
        if name in ('src/find_boundaries.cpp','src/mark_boundaries.cpp'):
            from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
            from tools.vi_public_matmul_context import speed_predecessor_bytes
            raw=speed_predecessor_bytes(ROOT,name,raw,VI_PUBLIC_MATMUL_CONTEXT_SHA256)
        if name=='src/find_boundaries.cpp':
            phase=json.loads((ROOT/'coverage/contracts/mark-wide-comparison-fp-v1.json').read_bytes())
            assert H(raw)==phase['kernel_current_raw_sha256']
            for f in phase['inverse_fragments']:
                assert raw.count(f['current'].encode())==1;raw=raw.replace(f['current'].encode(),f['predecessor'].encode(),1)
        assert H(raw)==pin,name


def test_client_uses_named_complete_operations_not_saved_mask_overlay():
    text=CPP.read_text()
    assert 'result=skimagexx::mark_boundaries(image.view,labels.view,c.parameters,workers)'in text
    assert 'detail::find_boundaries_into(labels.view,c.parameters.boundary,output.data()+8,c.result_strides,workers)'in text
    assert not any(token in text for token in('mark_boundaries_prepare(', 'mark_boundaries_mask(', 'mark_boundaries_paint(', 'mark_boundaries_finish(', 'mark_saved_cases(', 'mark_boundaries_consumer('))
    calls=[line for line in text.splitlines()if 'mark_boundaries('in line or 'find_boundaries_into('in line]
    assert len(calls)==4 and all('expected'not in line for line in calls)
    assert '#include "../fixtures/mark_wide_outer_sdk_v1.hpp"'in text
    assert 'mark_compare(result,c);auto copied=result;'in text


def test_owning_lifetime_capacity_input_and_exact_boolean_guards_are_real():
    text=CPP.read_text()
    for token in('image.words==a&&labels.words==b','result.data()!=image.view.data','result.data()!=labels.view.data',
                  'result.f32.size()==n&&result.f64.empty()&&result.f32.capacity()>=n',
                  'result.f64.size()==n&&result.f32.empty()&&result.f64.capacity()>=n',
                  'copied.data()!=result.data()','auto moved=std::move(copied);mark_compare(moved,c)',
                  'std::array<std::uint8_t,28> output;output.fill(0xad)',
                  'std::equal(c.expected.begin(),c.expected.end(),output.begin()+8)',
                  'output.begin()+20','[](auto x){return x<=1;}'):
        assert token in text


def test_admitted_FP_transport_pattern_is_reused_without_after_normalization():
    text=CPP.read_text();previous=(ROOT/'tests/test_rolling_ball_sdk_dtype_fp.cpp').read_text()
    for prefix in('WideFP state(','bool equal(','void set('):
        current=next(line for line in text.splitlines()if line.startswith(prefix))
        old_prefix=prefix.replace('WideFP','FPState')
        old=next(line for line in previous.splitlines()if line.startswith(old_prefix))
        assert current==old.replace('FPState','WideFP')
    target=text[text.index('template<class Operation>'):text.index('void run_case')]
    assert target.index('incoming=caller;set(incoming,row.before)')<target.index('operation();check(std::fegetenv(&after)')<target.index('const bool restored=restore_exact(caller)')<target.index('equal(state(after),row.after)')
    assert 'std::memcmp(&caller,&restored,32)==0'in text
    assert 'std::rethrow_exception(primary)'in target
    assert not any(t in target for t in('fesetround','feclearexcept','_mm_setcsr','set(after','memcpy(&after'))
    # Beforetemplates contain only3semanticwords, never capturedIPs/dataaddresses.
    assert 'w.before={895,'in (ROOT/C['generated_fixture']).read_text()


def test_worker_frontier_and_platform_controls_are_explicit_deferred():
    text=CPP.read_text()
    assert 'all_budgets[3][2]={{1,0},{4,0},{1,4}}'in text
    assert 'check(snapshots==std::size_t(workers==0)'in text
    assert 'get_num_threads()==budgets[0]'in text
    assert 'parameters.boundary.background_representable=false'in text
    assert 'std::fesetround(FE_DOWNWARD)'in text
    assert 'catch(const std::invalid_argument&)'in text
    assert '#if defined(__GNUC__) && defined(__x86_64__) && defined(__linux__)'in text
    assert 'not a parity PASS' in text and 'return 2;'in text
    assert C['actual_authority']==dict(compile=0,helper=0,FP=0,native=0,original=0,timing=0,hosted=0)
    assert 'UNKNOWN untilexecution'in C['FP']


def test_renderer_is_stdlib_word_transport_not_image_computation():
    tree=ast.parse((ROOT/'tests/mark_wide_outer_sdk.py').read_bytes())
    names=[]
    for node in tree.body:
        if isinstance(node,ast.Import):names.extend(n.name.split('.')[0]for n in node.names)
        if isinstance(node,ast.ImportFrom):names.append(node.module.split('.')[0])
    assert not set(names)&{'numpy','scipy','skimage','skimagexx','ctypes','subprocess'}
    assert C['deferred_per_binary']['total_named_calls']==28 and C['deferred_all_three_modes_named_calls']==84
