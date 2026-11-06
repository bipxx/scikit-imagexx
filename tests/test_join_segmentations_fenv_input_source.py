"""Captured INPUT byte checks only; no helper or NumPy/target bodies."""
import ast,hashlib,json
from pathlib import Path
import pytest
from join_segmentations_fenv_input import compatible,recorded_input,FIELDS
ROOT=Path(__file__).resolve().parents[1]


def test_all100_authentic_inputs_supported_and_diagnostics_remain_current():
    gold=json.loads((ROOT/'tests/fixtures/join_segmentations_originals_v1.json').read_bytes())
    count=0
    for profile,data in gold['profiles'].items():
        assert data['runtime']['numpy']==('1.24.4'if profile=='minimum'else'2.5.4')
        for row in data['rows']:
            record=bytes.fromhex(row['controller_fenv']['before']);compatible(record)
            assert row['controller_fenv']['before']==row['controller_fenv']['after']
            current=bytearray(record);current[12:28]=bytes(range(16));current[4:6]=(0).to_bytes(2,'little');current[28:32]=(0x1fa3).to_bytes(4,'little');current=bytes(current)
            seeded=recorded_input(current,row['controller_fenv']['before'])
            selected={i for offset,width in FIELDS for i in range(offset,offset+width)}
            assert all(seeded[i]==(record[i]if i in selected else current[i])for i in range(32))
            assert current[12:28]==seeded[12:28]and record.hex()==row['controller_fenv']['before'];count+=1
    assert count==100


@pytest.mark.parametrize('bad',('length','stack','x87-trap','SSE-trap','reserved'))
def test_unsupported_inputs_refused_before_restoration(bad):
    fixture=json.loads((ROOT/'tests/fixtures/join_segmentations_originals_v1.json').read_bytes())
    b=bytearray.fromhex(fixture['profiles']['modern']['rows'][0]['controller_fenv']['before']);assert len(b)==32
    if bad=='length':b.pop()
    elif bad=='stack':b[8]=0
    elif bad=='x87-trap':b[0]&=~1
    elif bad=='SSE-trap':b[29]&=~1
    else:b[31]=1
    with pytest.raises(AssertionError):recorded_input(bytes(b),bytes(b).hex())


def test_only_before_fixture_and_outer_finally_change_no_after_mask_or_assertion_weakening():
    source=(ROOT/'tests/test_join_segmentations_public.py').read_text();tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='test_saved_original')
    text=ast.unparse(fn)
    assert text.index('recorded_input(')<text.index('actual = observe(')
    assert 'bytes(before) == bytes(after)'in text and 'bytes(restored) == bytes(outer)'in text
    assert 'feclearexcept'not in text and 'fesetenv'not in text
    assert text.count('helper.reference_fenv_restore(')==2
    assert 'after=recorded_input'not in source and 'after = recorded_input'not in source


def test_authentic_context_and_complete_post_observe_assertion_tail_unchanged():
    context=json.loads((ROOT/'coverage/contracts/join-segmentations-captured-fenv-input-v1.json').read_bytes())
    raw=(ROOT/context['public_test']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==context['current_raw_sha256']
    previous=raw.decode()
    for before,after in reversed(context['inverse_replacements']):
        assert previous.count(after)==1
        previous=previous.replace(after,before,1)
    assert hashlib.sha256(previous.encode()).hexdigest()==context['predecessor_raw_sha256']
    marker="    actual['controller_fenv']=dict("
    assert raw.decode().split(marker,1)[1]==previous.split(marker,1)[1]
    for name,key in [('tests/util_views_fenv_restore.c','helper_source_raw_sha256'),('tests/util_views_fenv.py','backend_raw_sha256'),(context['fixture'],'fixture_raw_sha256')]:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==context[key]
    assert context['original_calls']==0 and context['numerical_fields']==[[0,2],[4,2],[28,4]]
