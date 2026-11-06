"""Saved-only originals authentication. No NumPy or target import/execution."""
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import zlib
import pytest
ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures/join_segmentations_originals_v1.json'
MAX_ARRAY_BYTES=384*8


def decode_array(value):
    match=re.fullmatch(r'[<>=|]([biuf])(1|2|4|8)',value['dtype']);assert match
    elements=math.prod(value['shape']);assert all(type(n)is int and n>=0 for n in value['shape'])
    size=elements*int(match[2]);assert size<=MAX_ARRAY_BYTES
    packed=base64.b64decode(value['raw_zlib_base64'],validate=True);decoder=zlib.decompressobj()
    raw=decoder.decompress(packed,MAX_ARRAY_BYTES+1)
    assert decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data
    assert len(raw)==size and hashlib.sha256(raw).hexdigest()==value['raw_sha256']
    assert len(value['strides'])==len(value['shape'])
    return raw


def validate_codecs(value):
    arrays=0
    if isinstance(value,dict):
        if 'raw_zlib_base64'in value:decode_array(value);arrays+=1
        elif 'dtype'in value and 'raw'in value:
            match=re.fullmatch(r'[<>=|][biuf](1|2|4|8)',value['dtype']);assert match
            assert len(bytes.fromhex(value['raw']))==int(match[1])
        for item in value.values():arrays+=validate_codecs(item)
    elif isinstance(value,list):
        for item in value:arrays+=validate_codecs(item)
    return arrays


def validate_row(row):
    assert row['api']=='segmentation.join_segmentations'and all(row['checks'].values())
    assert row['input_before']==row['input_after']
    assert row['original_python_helpers']['_join.py:join_segmentations']==1
    if row['error']is None:assert row['result']is not None
    else:assert len(row['error'])==3 and row['result']is None
    for warning in row['warnings']:assert len(warning)==3 and all(type(v)is str for v in warning)
    mapped=row['returned_map_aliases']is not None
    if mapped:
        assert row['returned_map_aliases']==dict(same_object=False,shared_in_values=True,shared_out_values=False)
        assert row['result']['container']=='tuple'and len(row['result']['maps'])==2
        for m in row['result']['maps']:
            assert m['type']==['skimage.util._map_array','ArrayMap']and m['cache']is None and m['max_str_lines']==4
        assert row['output_aliases'][1*5+3][:2]==[True,True]
    if row['map_protocols']:
        assert row['id']=='join_segmentations/int16-maps'and mapped
        assert [a['action']for a in row['map_protocols']]==['getitem','call','scalar','slice','bool','dense','repr','str','setitem-zero-to-one']
        assert all(a.get('error')is None for a in row['map_protocols'])
        assert row['map_protocols'][-1]['shared_in_values_after']is False
    assert all(not identity and not shared for identity,shared in row['output_input_aliases'])
    fp=row['controller_fenv'];assert len(bytes.fromhex(fp['before']))==len(bytes.fromhex(fp['after']))==32


def validate_saved(data):
    assert data['schema']=='join-segmentations-originals-v1'and data['status']=='CAPTURE_COMPLETE_PENDING_INDEPENDENT_RAW_ADMISSION'
    assert data['source_revision']=='1cfd37713062ee63e464c8abbe06ebd8518d01fc'
    assert data['root_GO_sha256']=='d5dbfc9392c0bf302577ca8a65f54b440f75387709f38fcb06e8f7f2d4d84f70'
    assert data['safety_peer_sha256']=='f381d6b76189eeaf5acc123f01205918713d3d9ccfa672e64b12f1bf038ca841'
    assert data['actual_public_topcalls']==100 and data['native_calls']==data['timed_batches']==0
    contract=json.loads((ROOT/'coverage/contracts/join-segmentations-v1.json').read_bytes());ids=[s['id']for s in contract['cases']]
    assert len(data['frozen_source_raw_sha256'])==6
    for name,pin in data['frozen_source_canonical_lf_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    assert set(data['profiles'])=={'minimum','modern'}
    arrays=0
    for name,profile in data['profiles'].items():
        rows=profile['rows'];assert [r['id']for r in rows]==ids and len(rows)==50
        assert profile['terminal']['counts']==dict(top=50,refused=0)and profile['terminal']['status']=='COMPLETE_PENDING_INDEPENDENT_RAW_REVIEW'
        assert profile['runtime']['numpy']=={'minimum':'1.24.4','modern':'2.5.4'}[name]
        expected=dict(rows=50,returns=sum(r['error']is None for r in rows),errors=sum(r['error']is not None for r in rows),mapped_returns=sum(r['returned_map_aliases']is not None for r in rows),warning_rows=sum(bool(r['warnings'])for r in rows),warning_count=sum(len(r['warnings'])for r in rows))
        assert profile['summary']==expected
        assert sum(bool(row['map_protocols'])for row in rows)==1
        for row in rows:validate_row(row)
        arrays+=validate_codecs(rows)
    return arrays


@pytest.fixture(scope='module')
def saved():
    data=json.loads(FIXTURE.read_bytes());count=validate_saved(data);return data,count


def test_join_saved_originals_only(saved):
    data,count=saved
    assert count>0
    assert data['profiles']['minimum']['summary']==dict(rows=50,returns=37,errors=13,mapped_returns=25,warning_rows=0,warning_count=0)
    assert data['profiles']['modern']['summary']==dict(rows=50,returns=37,errors=13,mapped_returns=25,warning_rows=44,warning_count=176)


@pytest.mark.parametrize('mutation',['hash','shape','payload'])
def test_join_saved_codec_tamper_refused(saved,mutation):
    value=saved[0]['profiles']['minimum']['rows'][0]['result'];bad=dict(value)
    if mutation=='hash':bad['raw_sha256']='0'*64
    elif mutation=='shape':bad['shape']=[999999999]
    else:bad['raw_zlib_base64']=base64.b64encode(zlib.compress(b'bad')).decode()
    with pytest.raises(AssertionError):decode_array(bad)


def test_join_saved_shared_map_alias_tamper_refused(saved):
    row=next(r for r in saved[0]['profiles']['minimum']['rows']if r['returned_map_aliases'])
    bad=dict(row,returned_map_aliases=dict(row['returned_map_aliases'],shared_in_values=False))
    with pytest.raises(AssertionError):validate_row(bad)
