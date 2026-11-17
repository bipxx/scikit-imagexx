"""Strict source-only compatibility inverse; no NumPy, native or FP execution."""
import ast
import hashlib
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/mark-wide-compatibility-v1.json'
PIN='31f15650dbbe6d7448040f856b9253034ec69635eab412d65461033822a1e6ac'
H=lambda raw:hashlib.sha256(raw).hexdigest()


def context():
    raw=CONTEXT.read_bytes();assert H(raw)==PIN
    return json.loads(raw)


def predecessor(name,raw,canonical=False):
    row=context()['sources'][name]
    if canonical:raw=raw.replace(b'\r\n',b'\n')
    suffix='canonical_lf_sha256'if canonical else'raw_sha256'
    assert H(raw)==row['current_'+suffix]
    for fragment in row['inverse']:
        current=fragment['current'].encode();old=fragment['predecessor'].encode()
        assert current and raw.count(current)==1;raw=raw.replace(current,old,1)
    assert H(raw)==row['predecessor_'+suffix]
    return raw


@pytest.mark.parametrize('name',tuple(context()['sources']))
@pytest.mark.parametrize('canonical',(False,True))
def test_exact_once_current_to_frozen_projection(name,canonical):
    raw=(ROOT/name).read_bytes();old=predecessor(name,raw,canonical)
    assert H(old)==context()['sources'][name]['predecessor_raw_sha256']
    with pytest.raises(AssertionError):predecessor(name,old,canonical)
    if canonical:assert predecessor(name,raw.replace(b'\n',b'\r\n'),True)==old
    else:
        with pytest.raises(AssertionError):predecessor(name,raw.replace(b'\n',b'\r\n'))


@pytest.mark.parametrize('name',tuple(context()['sources']))
@pytest.mark.parametrize('canonical',(False,True))
def test_changed_input_is_not_accepted_as_current_or_historical(name,canonical):
    with pytest.raises(AssertionError):predecessor(name,(ROOT/name).read_bytes()+b'\n# tamper',canonical)


def test_only_find_boundaries_per_source_flags_are_added():
    raw=(ROOT/'CMakeLists.txt').read_bytes();old=predecessor('CMakeLists.txt',raw)
    new_lines=[line for line in raw.decode().splitlines()if'set_source_files_properties(src/find_boundaries.cpp PROPERTIES'in line]
    assert new_lines==[
        '    set_source_files_properties(src/find_boundaries.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")',
        '    set_source_files_properties(src/find_boundaries.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math;-frounding-math")']
    assert raw.decode().index('if(MSVC)')<raw.decode().index(new_lines[0])<raw.decode().index(new_lines[1])
    for line in new_lines:raw=raw.replace((line+'\n').encode(),b'',1)
    assert raw==old


def test_public_refusal_keeps_eight_variants_and_explicit_unsupported_background():
    raw=(ROOT/'tests/test_find_boundaries_public.py').read_bytes();old=predecessor('tests/test_find_boundaries_public.py',raw)
    tree=ast.parse(raw);oldtree=ast.parse(old)
    name='test_find_boundaries_private_refusal_before_write'
    current=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    previous=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    variants=ast.literal_eval(current.decorator_list[0].args[1]);old_variants=ast.literal_eval(previous.decorator_list[0].args[1])
    assert len(variants)==len(old_variants)==8
    assert variants==[v if v!='outer64'else'outer64-unrepresentable-background'for v in old_variants]
    text=ast.get_source_segment(raw.decode(),current)
    assert "kind=='outer64-unrepresentable-background':a=a.astype(np.int64);mode=2;background_representable=False"in text
    assert 'before=out.tobytes();assert owned._core.find_boundaries_into(a,out,1,mode,0,background_representable,1)is None'in text
    assert 'assert out.tobytes()==before'in text
    # All other complete public parity/state/storage/protocol test bodies exact.
    def rest(t):return [ast.dump(n)for n in t.body if not(isinstance(n,ast.FunctionDef)and n.name==name)]
    assert rest(tree)==rest(oldtree)


def test_native_refusal_preserves_nonexact_thick_and_every_error_guard():
    raw=(ROOT/'tests/test_find_boundaries_native.cpp').read_bytes();old=predecessor('tests/test_find_boundaries_native.cpp',raw)
    text=raw.decode()
    assert 'refuse();a[1]=1;p.mode=FindBoundaryMode::outer;'in text
    assert 'p.background_representable=false;refuse();p.background_representable=true;'in text
    assert 'std::array<std::uint64_t,2> a{{0,(std::uint64_t{1}<<53)+1}}'in text
    assert 'std::all_of(out.begin(),out.end(),[](auto x){return x==0xad;})'in text
    # No expected masks, source words, algorithm bodies or extra target calls.
    assert text.count('find_boundaries_into(')==old.decode().count('find_boundaries_into(')
    assert context()['sources']['tests/test_find_boundaries_native.cpp']['inverse'][0]['predecessor']==' refuse();a[1]=1;p.mode=FindBoundaryMode::outer;refuse();\n'


def test_unchanged_kernel_refuses_other_platform_rounding_masks_background():
    raw=(ROOT/'src/find_boundaries.cpp').read_bytes()
    phase=json.loads((ROOT/'coverage/contracts/mark-wide-comparison-fp-v1.json').read_bytes())
    assert H(raw)==phase['kernel_current_raw_sha256']
    for f in phase['inverse_fragments']:
        assert raw.count(f['current'].encode())==1;raw=raw.replace(f['current'].encode(),f['predecessor'].encode(),1)
    text=raw.decode()
    start=text.index('template<class T>bool wide_source_outer');stop=text.index('// Independent 2D interior pixels:',start)
    block=text[start:stop]
    for token in('#if defined(__GNUC__) && defined(__x86_64__)',
                  'if(!parameters.background_representable)return false;',
                  'std::fegetround()!=FE_TONEAREST','(control&(3u<<13))',
                  '(control&0x1f80u)!=0x1f80u'):
        assert token in block
    assert 'parameters.mode==FindBoundaryMode::outer' in text
    assert 'value<-(std::int64_t{1}<<53) || value>(std::int64_t{1}<<53)'in text
    assert 'else if(value>(std::uint64_t{1}<<53))return false;'in text
    assert not any(token in block for token in('fesetround','feclearexcept','_mm_setcsr','fesetenv'))


@pytest.mark.parametrize('name',tuple(context()['dependencies']))
def test_frozen_native_headers_frontend_and_original_context_unchanged(name):
    raw=(ROOT/name).read_bytes()
    if name=='src/find_boundaries.cpp':
        phase=json.loads((ROOT/'coverage/contracts/mark-wide-comparison-fp-v1.json').read_bytes())
        assert H(raw)==phase['kernel_current_raw_sha256']
        for f in phase['inverse_fragments']:
            assert raw.count(f['current'].encode())==1;raw=raw.replace(f['current'].encode(),f['predecessor'].encode(),1)
    assert H(raw)==context()['dependencies'][name]
    assert context()['actual_execution']==dict(original=0,native=0,helper=0,FP=0,compile=0,timing=0)
