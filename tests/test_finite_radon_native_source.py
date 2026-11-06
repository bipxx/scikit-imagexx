"""Pure owned-native successor guards, not original/parity execution."""
import ast,base64,hashlib,json,runpy,zlib
import pytest
import finite_radon_contract as c
from finite_radon_saved_parity import run_saved,portable_record

def test_native_source_pins_and_historical_packet():
    d=json.loads((c.ROOT/'coverage/contracts/finite-radon-native-v3.json').read_bytes())
    assert not d['API_completion']and not d['ready_for_integration']and d['original_calls']==0
    assert hashlib.sha256(c.CONTRACT.read_bytes()).hexdigest()==d['source_contract_sha256']
    for n,p in d['source_files'].items():assert hashlib.sha256((c.ROOT/n).read_bytes()).hexdigest()==p,n

def test_native_modular_spans_and_budget_not_floating_claim():
    s=(c.ROOT/'src/finite_radon.cpp').read_text();assert s.count('get_num_threads()')==1
    assert 'accumulate(dst,src+start,n-start,simd)'in s and 'accumulate(dst+n-start,src,start,simd)'in s
    assert '_mm_add_epi32'in s and 'SKIMAGEXX_FINITE_RADON_SCALAR'in s
    assert 'plan.weak_scalar'in s and 'plan.accumulator_bits'in s and 'plan.signed_sum'in s
    assert 'std::fma'not in s and 'fesetenv'not in s
    assert s.count('std::feclearexcept')==1 and 'if(std::fetestexcept(cast_errors))std::feclearexcept(cast_errors)'in s

def test_public_no_sum_roll_forwarding_and_signature():
    s=(c.ROOT/'python/skimagexx/_finite_radon.py').read_text();tree=ast.parse(s)
    for fn in [n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('frt2','ifrt2')]:assert [a.arg for a in fn.args.args]==['a']
    assert 'np.roll('not in s and '.sum('not in s and 'skimage.transform'not in s
    assert 'NotImplementedError'in s and 'source'not in [n.name for n in tree.body if isinstance(n,ast.FunctionDef)]

def test_no_saved_gold_and_no_weakened_target_flags():
    with pytest.raises(AssertionError,match='FINITE_RADON_UNBOUND'):run_saved(None,None,None,None,None)
    assert portable_record(dict(raw='diagnostic',control=1,status=2,mxcsr=3))==dict(control=1,status=2,mxcsr=3)

def test_prospective_benchmark_manifest_unbound_real_method():
    d=runpy.run_path(str(c.ROOT/'benchmarks/finite_radon_native_v2.py'))['plan']()
    assert len(d['cases'])==8 and d['root_GO']is None and d['producer']is None
    assert d['threads']==1 and d['repeat']==7 and d['target_ms']==100 and d['warmup_min_ms']==250 and d['retain_all_losses']
    assert d['status']=='UNBOUND_NO_EXECUTION'and len(d['sampler_source_sha256'])==64
    assert len({row['id']for row in d['cases']})==8
    original=runpy.run_path(str(c.ROOT/'benchmarks/finite_radon_native.py'))['plan']()['cases']
    projected=[{k:(r['source_case_id']if k=='id'else v)for k,v in r.items()if k!='source_case_id'}for r in d['cases']]
    assert projected==original

def test_successor_observer_reuses_exact_typed_codec_without_old_mutation():
    d=json.loads((c.ROOT/'coverage/contracts/finite-radon-observer-v2.json').read_bytes())
    for n,p in d['source_files'].items():assert hashlib.sha256((c.ROOT/n).read_bytes()).hexdigest()==p
    assert d['original_budget']==0 and d['historical_source_packet_unchanged']
    source=(c.ROOT/'tests/finite_radon_observer_v2.py').read_text()
    assert 'from finite_radon_exception_codec import encode_exception_args as error_value'in source

def test_all216_original_rows_are_exact_transport_not_fitted_expected_values():
    d=json.loads((c.ROOT/'tests/fixtures/finite_radon_originals_v1.json').read_bytes())
    assert d['raw_admission_sha256']=='ae58afd446546b0005edf6051a5f9745be2dd7e466416f8364b0547a363cb2ec'
    for profile,blob in d['profiles'].items():
        raw=zlib.decompress(base64.b64decode(blob['zlib_base64']));assert len(raw)==blob['bytes']and hashlib.sha256(raw).hexdigest()==blob['sha256']
        rows=json.loads(raw);assert len(rows)==len({r['id']for r in rows})==108
        assert sum(r['error']is None for r in rows)==61 and sum(r['error']is not None for r in rows)==47

def test_native_projection_cast_warning_phase_and_policy_provider():
    s=(c.ROOT/'src/finite_radon.cpp').read_text();binder=(c.ROOT/'src/bind_finite_radon.cpp').read_text()
    assert 'pair_sum('in s and 'finite_radon_inverse_prefix('in s and 'detail::decode_half'in s
    assert '_mm_cvttsd_si64'in s and '_mm_cvttss_si64'in s
    assert s.index('std::fetestexcept(cast_errors)')<s.index('out.values[m*n+c]=wide?contiguous_double_word')
    assert 'PyUFunc_GiveFloatingpointErrors'in binder and 'PyUFunc_handlefperr'in binder
    assert 'PyArray_Cast'not in binder and 'PyArray_CopyInto'not in binder



def test_actual_wide_cast_source_policy_and_no_case_or_after_FP_mask():
    s=(c.ROOT/'src/finite_radon.cpp').read_text()
    helper=s[s.index('std::uint32_t contiguous_double_word'):s.index('template<class T>FiniteRadonTransform project_float')]
    assert '_mm_cmple_sd'in helper and '_mm_cvttsd_si32'in helper
    assert 'if(std::fetestexcept(FE_INVALID))out.cast_status[m]|=2;'in s
    assert 'n-n%4'in s and 'wide?contiguous_double_word(double(value)):cast_uint32(value)'in s
    assert 'exceptional-n5'not in s and 'isnan'not in helper and 'isinf'not in helper
    assert 'fesetenv'not in s and 'setcsr'not in s
    assert s.count('get_num_threads()')==1


def test_all8_new_raw_rows_verbatim_and_observer_fixture_inverse():
    root=c.ROOT;d=json.loads((root/'tests/fixtures/finite_radon_exceptional_originals_v1.json').read_bytes())
    assert d['admission_sha256']=='9311433dc1e06532d4ebe9c38e562214074b4c33bb733ca28d05c263b2ceaa5f'
    for profile,blob in d['profiles'].items():
        raw=zlib.decompress(base64.b64decode(blob['zlib_base64']));assert hashlib.sha256(raw).hexdigest()==blob['sha256']and len(raw)==blob['bytes']
        rows=json.loads(raw);assert len(rows)==4 and sum(row['error']is None for row in rows)==2
    observer=(root/'tests/finite_radon_exceptional_observer_v1.py').read_text()
    assert observer.replace('from finite_radon_exceptional_contract import fixture','from finite_radon_contract import fixture')==(root/'tests/finite_radon_observer_v2.py').read_text()
