"""Family source/immutable raw transport gates; no scientific imports."""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/rolling-ball-native-v2.json'
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def test_current_native_closure_and_frozen_source_are_exact():
    context=json.loads(CONTEXT.read_bytes())
    for group in('source_files_raw_sha256','frozen_source_files_raw_sha256'):
        for name,pin in context[group].items():assert H(ROOT/name)==pin,name
    assert context['reference_calls_remaining']==0 and context['actual_original_authority']is context['actual_performance_authority']is None
    assert context['state']=='PRIVATE_UNREADY_NO_PUBLIC_REGISTRATION'


def test_complete_saved_domain_typed_bytes_and_no_row_rewriting():
    from rolling_ball_contract import recipes
    saved=json.loads((ROOT/'tests/fixtures/rolling_ball_original_v1.json').read_bytes())
    def payload(value):
        if isinstance(value,list):
            for item in value:payload(item)
        elif isinstance(value,dict):
            if value.get('type')=='ndarray':
                raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64'],validate=True))
                assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
                assert len(raw)==math.prod(value['shape'])*int(value['dtype'][2:])
            for item in value.values():payload(item)
    for profile in('minimum','modern'):
        rows=saved['profiles'][profile]['rows'];assert [r['id']for r in rows]==[s['id']for s in recipes()]and len(rows)==75
        assert sum(r['error']is None for r in rows)==63 and sum(r['error']is not None for r in rows)==12
        assert sum(len(r['warnings'])for r in rows)==8
        for row in rows:
            assert row['caller_restore']['exact32']and row['caller_restore']['saved']==row['caller_restore']['actual']
            assert row['supplied_call_attempts']==1 and row['before']==row['after']and row['numpy_before']==row['numpy_after']
            payload(row)


def test_installed_SDK_gold_is_exact_original_float_word_projection():
    saved=json.loads((ROOT/'tests/fixtures/rolling_ball_original_v1.json').read_bytes())['profiles']
    assert [r['result']for r in saved['minimum']['rows']]==[r['result']for r in saved['modern']['rows']]
    expected=[r for r in saved['minimum']['rows']if r['error']is None and r['result']['dtype']in('<f4','<f8')]
    raw=(ROOT/'tests/fixtures/rolling_ball_sdk_v1.bin').read_bytes();assert raw[:8]==b'ROLLSDK1'
    count,=struct.unpack_from('<Q',raw,8);assert count==len(expected)==52;offset=16
    for row in expected:
        length,item,rank,backing,origin,kernel,radius,safe=struct.unpack_from('<QQQQQQdQ',raw,offset);offset+=64
        name=raw[offset:offset+length].decode();offset+=length;assert name==row['id']
        shape=struct.unpack_from('<'+'q'*rank,raw,offset);offset+=rank*8
        strides=struct.unpack_from('<'+'q'*rank,raw,offset);offset+=rank*8
        kernel_shape=struct.unpack_from('<'+'q'*rank,raw,offset);offset+=rank*8
        assert 1<=rank<=8 and backing<=64 and kernel<=201 and item in(4,8)
        offset+=(backing+kernel)*item
        output=raw[offset:offset+math.prod(shape)*item];offset+=len(output)
        assert hashlib.sha256(output).hexdigest()==row['result']['raw_sha256']
        assert output==zlib.decompress(base64.b64decode(row['result']['raw_zlib_base64']))
    assert offset==len(raw)


def test_captured_FP_input_is_seeded_before_call_and_outer_state_exact():
    raw=(ROOT/'tests/test_rolling_ball_public.py').read_text()
    assert "for key in('control','status','mxcsr')"in raw
    assert "expected['target_immediate']['before'][key]"in raw
    assert "restore(incoming);actual_state['before']=numerical(state())"in raw
    assert "assert actual=={k:expected[k]for k in actual}"in raw
    assert "restore(outer);assert bytes(state())==bytes(outer)"in raw
    assert not any(s in raw for s in('nan_to_num','allclose','seterr(','skip','xfail'))


def test_frontend_has_no_original_or_whole_kernel_provider_fallback():
    raw=(ROOT/'python/skimagexx/_rolling_ball.py').read_text();tree=ast.parse(raw)
    imports=[n.module for n in ast.walk(tree)if isinstance(n,ast.ImportFrom)]
    assert not any(n and n.startswith('skimage.')for n in imports)
    assert 'np.meshgrid'not in raw and 'np.sqrt'not in raw and 'np.pad'not in raw
    assert '_core.ball_kernel_apply('in raw and '_core.rolling_ball_apply('in raw
    assert "raise NotImplementedError('rolling_ball owned kernel refused"in raw
    assert 'rolling_ball_original'not in raw and 'recipes('not in raw
