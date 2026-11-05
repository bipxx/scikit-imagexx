"""One supplemental source predicate holdout; saved validation never replays draw."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
import pytest
if __package__:
    from .line_nd_contract import source_safe
    from .line_nd_observer import observe, scientific_profile
    from .test_draw_coordinates_public import without_draw_activity
else:
    from line_nd_contract import source_safe
    from line_nd_observer import observe, scientific_profile
    from test_draw_coordinates_public import without_draw_activity
ROOT = Path(__file__).parents[1]
MATRIX = ROOT/'coverage/contracts/line-nd-rounding-promotion-v1.json'


def test_line_nd_rounding_promotion_actual_controller_original_refused():
    matrix=json.loads(MATRIX.read_bytes())
    assert len(matrix['cases'])==1 and matrix['numerical_policy']==dict(max_ulp=0,rtol=0,atol=0)
    for name,pin in matrix['source_pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    spec=matrix['cases'][0]
    assert source_safe(spec)['max_output_elements']==512
    np=scientific_profile()
    calls=[]
    def refused(*args,**options):
        calls.append((args,options))
        assert args[0].tobytes().hex()=='ffffffbe'
        assert options==dict(endpoint=True,integer=True)
        raise RuntimeError('line-nd-rounding-original-refused')
    with without_draw_activity():
        row=observe(refused,spec,np)
    assert len(calls)==1 and row['error']==['builtins','RuntimeError','line-nd-rounding-original-refused']
    assert all(row['checks'].values()) and not row['warnings'] and not row['events']['items']


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_line_nd_rounding_promotion_saved_original_only(profile):
    matrix=json.loads(MATRIX.read_bytes())
    if not matrix['original_observations']:
        pytest.skip('Prospective supplement; original capture not executed yet')
    row=matrix['original_observations'][profile][0]
    assert row['id']==matrix['cases'][0]['id'] and row['error'] is None
    assert all(row['checks'].values()) and not row['warnings'] and not row['events']['items']
    assert row['result']['container']=='tuple' and len(row['result']['items'])==1
    value=row['result']['items'][0]
    raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
    assert hashlib.sha256(raw).hexdigest()==value['raw_sha256'] and len(raw)==48
    assert value['dtype']=='<i8' and value['shape']==[6] and not value['own']



def test_line_nd_rounding_promotion_source_safety_without_science():
    matrix = json.loads(MATRIX.read_bytes())
    assert len(matrix['cases']) == 1
    assert matrix['numerical_policy'] == dict(max_ulp=0, rtol=0, atol=0)
    for name, pin in matrix['source_pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pin
    assert source_safe(matrix['cases'][0])['max_output_elements'] == 512
