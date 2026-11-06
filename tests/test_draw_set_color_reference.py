import json,hashlib,zlib,base64,math
from pathlib import Path
from draw_set_color_contract import CASES
ROOT=Path(__file__).resolve().parents[1]
def test_manifest():
    contract=json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_text())
    assert contract['cases']==CASES and len(CASES)==26 and len({x['id']for x in CASES})==26

def test_saved():
    contract=json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_text())
    if 'observations'not in contract:return
    for profile,rows in contract['observations'].items():
        assert [x['id']for x in rows]==[x['id']for x in CASES]
        assert all(all(x['checks'].values())for x in rows)
        assert sum(x['error']is None for x in rows)==20
        def validate(value):
            if isinstance(value,dict):
                if value.get('type')=='ndarray':
                    raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
                    assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
                    assert len(raw)==math.prod(value['shape'])*int(value['dtype'][2:])
                for item in value.values():validate(item)
            elif isinstance(value,list):
                for item in value:validate(item)
        validate(rows)

def test_integer_stage_original_supplement():
    contract=json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_text())
    assert len(contract['observations']['1.24.4'])==len(contract['observations']['2.5.4'])==26
    for profile,witness in contract['integer_stage_supplement'].items():
        assert witness['counts']=={'original_public':1,'native_public':1,'native_C_return':1,'worker_C_return':1}
        assert [row['image_words']for row in witness['records']]==['0000000000000000','0000000000000080']
        assert witness['original_stage_facts']['integer_color_times_alpha']['dtype']=='<i8'
    for profile,witness in contract['integer_stage_operation_witnesses'].items():
        assert witness['original_algorithms']==witness['native_calls']==0 and len(witness['rows'])==18
        for row in witness['rows']:
            integer=row['alpha_kind']=='builtin-int'and row['color_dtype']not in('uint64','float64')
            assert row['weighted_dtype']==('<i8'if integer else'<f8')

def test_float32_saved_partition():
    contract=json.loads((ROOT/'coverage/contracts/draw-set-color.json').read_text());ids=[x['id']for x in contract['float32_cases']]
    assert len(ids)==len(set(ids))==12
    for profile,rows in contract['float32_original_observations'].items():
        assert [r['id']for r in rows]==ids
        assert all(r['error']is None and r['result']is None and r['warnings']==[]and r['FP_events']==[]and all(r['checks'].values())for r in rows)
        for row in rows:
            image=row['after']['items'][0]['items'][0]
            assert image['dtype']=='<f4'and math.prod(image['shape'])<=48
            raw=zlib.decompress(base64.b64decode(image['raw_zlib_base64']));assert hashlib.sha256(raw).hexdigest()==image['raw_sha256']
    for profile,facts in contract['float32_operation_witnesses'].items():
        assert len(facts['operation_facts'])==9 and all(x['values_dtype']==x['sum_dtype']=='<f8'for x in facts['operation_facts'])
        assert next(x for x in facts['cast_facts']if x['id']=='tiny-inexact')['events']==[['underflow',4]]
