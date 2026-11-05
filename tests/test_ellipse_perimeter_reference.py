"""Prospective original-refused intake; numerical capture is separately gated."""
import base64,copy,hashlib,json,math,zlib
from pathlib import Path
import pytest
if __package__:
    from .ellipse_perimeter_contract import API,UNASSESSED,cases,source_safe
    from .ellipse_perimeter_observer import SIGNATURE,SOURCE_SHA,CYTHON_SHA,observe,scientific_profile
    from .test_draw_coordinates_public import without_draw_activity
else:
    from ellipse_perimeter_contract import API,UNASSESSED,cases,source_safe
    from ellipse_perimeter_observer import SIGNATURE,SOURCE_SHA,CYTHON_SHA,observe,scientific_profile
    from test_draw_coordinates_public import without_draw_activity
ROOT=Path(__file__).parents[1];CASES=cases();MATRIX=ROOT/'coverage/contracts/ellipse-perimeter.json'

def test_ellipse_perimeter_prospective_source_safety():
    matrix=json.loads(MATRIX.read_bytes())
    assert matrix['api']==API and matrix['signature']==SIGNATURE and matrix['cases']==CASES
    assert len(CASES)==len({s['id'] for s in CASES}) and matrix['unassessed']==list(UNASSESSED)
    assert matrix['source_sha256']==SOURCE_SHA and matrix['cython_sha256']==CYTHON_SHA
    assert matrix['native_calls']==0 and matrix['timings'] is False
    assert matrix['original_calls']==(58 if matrix['original_observations'] else 0)
    for name,pin in matrix['source_freeze'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    for spec in CASES:assert source_safe(spec)['max_output_elements']<=1024

@pytest.mark.parametrize('fault',('shape0','shape1','nan-angle','unequal-rotated','big-center','big-radius','negative-radius','nonzero-single-zero'))
def test_ellipse_perimeter_unsafe_partitions_refused_before_factory(fault):
    spec=copy.deepcopy(CASES[0])
    if fault.startswith('shape'):spec['kwargs']['shape']=[8]*int(fault[-1])
    elif fault=='nan-angle':spec['kwargs']['orientation']=float('nan')
    elif fault=='unequal-rotated':spec['safe_args']=[0,0,2,4];spec['kwargs']['orientation']=.3
    elif fault=='big-center':spec['safe_args'][0]=2**40
    elif fault=='big-radius':spec['safe_args'][2]=2**40
    elif fault=='negative-radius':spec['safe_args'][2]=-1
    else:spec['safe_args'][2]=0;spec['kwargs']['orientation']=.3
    with pytest.raises(AssertionError):source_safe(spec)

@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
def test_ellipse_perimeter_controller_original_refused(spec):
    np=scientific_profile();calls=[]
    def refused(*args,**options):calls.append((args,options));raise RuntimeError('ellipse-perimeter-original-refused-before-capture')
    with without_draw_activity():row=observe(refused,spec,np)
    assert len(calls)==1 and row['result'] is None
    assert row['error']==['builtins','RuntimeError','ellipse-perimeter-original-refused-before-capture']
    assert not row['warnings'] and row['events']=={'container':'list','items':[]} and all(row['checks'].values())


@pytest.mark.parametrize('profile',('1.24.4','2.5.4'))
def test_ellipse_perimeter_saved_observations_without_replay(profile):
    matrix=json.loads(MATRIX.read_bytes());rows=matrix['original_observations'][profile]
    assert [r['id'] for r in rows]==[s['id'] for s in CASES]
    assert len(rows)==29 and sum(r['error'] is None for r in rows)==23
    for spec,row in zip(CASES,rows):
        assert row['api']==API and all(row['checks'].values()) and row['input_before']==row['input_after']
        if row['error'] is not None:
            assert row['result'] is None
            continue
        assert row['result']['container']=='tuple' and len(row['result']['items'])==2
        bound=source_safe(spec);size=0
        for value in row['result']['items']:
            assert value['type']=='ndarray' and value['dtype']=='<i8' and value['dtype_metadata'] is None
            assert len(value['shape'])==(0 if bound.get('early_0D_before_shape') else 1)
            raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64']))
            assert hashlib.sha256(raw).hexdigest()==value['raw_sha256']
            assert len(raw)==8*math.prod(value['shape']);size+=math.prod(value['shape'])
        assert size<=bound['max_output_elements']
