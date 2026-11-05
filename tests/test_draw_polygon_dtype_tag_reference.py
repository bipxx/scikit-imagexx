"""Tagged-f64 supplemental saved reference; no new originals in tests."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
ROOT=Path(__file__).parents[1]

def tag_fixture(spec,np):
    assert spec['id'] in ('draw.polygon/tagged-f64','draw.polygon2mask/tagged-C-f64')
    assert spec['shape']==[3,3]
    dtype=np.dtype(np.float64,metadata={'tag':7})
    if spec['api']=='draw.polygon':
        r=np.array([0,0,2],dtype=dtype);c=np.array([0,2,0],dtype=dtype)
        return (r,c),{'shape':(3,3)},{'r':r,'c':c},{'max_output_pairs':9,'max_mask_elements':9}
    assert spec['api']=='draw.polygon2mask'
    polygon=np.array([[0,0],[0,2],[2,0]],dtype=dtype)
    return ((3,3),polygon),{},{'polygon':polygon},{'max_output_pairs':9,'max_mask_elements':9}

def test_draw_polygon_tagged_saved_reference_complete():
    data=json.loads((ROOT/'coverage/contracts/draw-polygon-dtype-tag-v1.json').read_bytes())
    assert set(data['original_observations'])=={'1.24.4','2.5.4'}
    for rows in data['original_observations'].values():
        assert [r['id']for r in rows]==[s['id']for s in data['cases']]
        for row in rows:
            assert row['error']is None and row['warnings']==[]and row['FP_events']==[]
            assert row['inputs_before']==row['inputs_after']and all(row['checks'].values())
            assert row['call_accounting']['top_level_calls']==1

def test_draw_polygon_tagged_saved_payloads():
    data=json.loads((ROOT/'coverage/contracts/draw-polygon-dtype-tag-v1.json').read_bytes())
    def check(value):
        if isinstance(value,dict):
            if 'raw_zlib_base64'in value:
                raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
                assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
            for child in value.values():check(child)
        elif isinstance(value,list):
            for child in value:check(child)
    check(data['original_observations'])
