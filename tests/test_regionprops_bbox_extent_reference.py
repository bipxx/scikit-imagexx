"""No original constructors/properties: refused intake and synthetic schema probe."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import warnings
import zlib
import pytest
if __package__:
    from .regionprops_bbox_extent_contract import API,MEMBERS,ALIASES,UNASSESSED,cases,source_safe
    from .regionprops_bbox_extent_observer import observe,scientific_profile,SOURCE_SHA,UTILS_SHA,SIGNATURE
else:
    from regionprops_bbox_extent_contract import API,MEMBERS,ALIASES,UNASSESSED,cases,source_safe
    from regionprops_bbox_extent_observer import observe,scientific_profile,SOURCE_SHA,UTILS_SHA,SIGNATURE
ROOT=Path(__file__).parents[1];CASES=cases();MATRIX=ROOT/'coverage/contracts/regionprops-bbox-extent.json'


def test_bbox_extent_prospective_source_and_scope():
    matrix=json.loads(MATRIX.read_bytes())
    assert matrix['api']==API and matrix['members']==list(MEMBERS) and matrix['aliases']==ALIASES
    assert matrix['signature']==SIGNATURE and matrix['cases']==CASES and len(CASES)==14
    assert len({s['id'] for s in CASES})==14 and set(matrix['original_observations'])=={'1.24.4','2.5.4'} and matrix['original_calls']==28
    assert matrix['source_sha256']==SOURCE_SHA and matrix['utils_sha256']==UTILS_SHA and matrix['unassessed']==list(UNASSESSED)
    for name,pin in matrix['source_freeze'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    discovery=json.loads((ROOT/'coverage/contracts/regionprops-runtime-surface.json').read_bytes())
    code=discovery['source_witnesses']['RegionProperties']['source'];cls=ast.parse(code).body[0]
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    assert ast.unparse(methods['area_bbox'].body[0].value)=='self.image.size * self._pixel_area'
    assert ast.unparse(methods['extent'].body[0].value)=='self.area / self.area_bbox'
    assert [ast.unparse(n) for n in methods['area_bbox'].decorator_list]==['property']
    assert [ast.unparse(n) for n in methods['extent'].decorator_list]==['property']
    for spec in CASES:assert source_safe(spec)['input_elements']<=64


@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
def test_bbox_extent_actual_factory_original_refused(spec):
    np=scientific_profile();calls=[]
    def refused(image,**options):calls.append((image.shape,image.dtype.name,options));raise RuntimeError('bbox-extent-original-refused-before-capture')
    row=observe(refused,spec,np)
    assert len(calls)==1 and row['constructor_error']==['builtins','RuntimeError','bbox-extent-original-refused-before-capture']
    assert row['returned_count'] is None and row['steps']==[] and not row['FP_events'] and not row['constructor_warnings']
    assert all(row['checks'].values()) and row['input_before']==row['input_after']


@pytest.mark.parametrize('fault',('shape','labeldtype','spacing-nan','spacing-negative','offset','slice','pixel'))
def test_bbox_extent_unsafe_partitions_rejected(fault):
    spec=copy.deepcopy(CASES[0])
    if fault=='shape':spec['shape']=[4096,4096]
    elif fault=='labeldtype':spec['dtype']='object'
    elif fault=='spacing-nan':spec['spacing']=[float('nan'),1.]
    elif fault=='spacing-negative':spec['spacing']=[-1.,1.]
    elif fault=='offset':spec['offset']=[2**50,0]
    elif fault=='slice':spec['sequence']=[['set-slice',[[0,999],[0,2]]]]
    else:spec['sequence']=[['set-pixel',[[999,0],1]]]
    with pytest.raises(AssertionError):source_safe(spec)


@pytest.mark.parametrize('index',(2,10))
def test_bbox_extent_observer_action_schema_without_original(index):
    np=scientific_profile();spec=CASES[index];calls=[]
    class Standin:
        slice=None
        area=np.float64(3.)
        image=np.ones((2,2),dtype=bool)
        area_bbox=np.float64(4.)
        extent=np.float64(.75)
        bbox_area=np.float64(4.)
        def __getitem__(self,key):
            if key in ALIASES:warnings.warn('synthetic alias warning',FutureWarning)
            return getattr(self,ALIASES.get(key,key))
    def synthetic(image,**options):calls.append(True);return [Standin()]
    row=observe(synthetic,spec,np)
    assert calls==[True] and row['constructor_error'] is None and all(row['checks'].values())
    assert len(row['steps'])==len(spec['sequence'])
    assert any(s['warnings'] for s in row['steps'] if s['action']=='item')
    assert any(s['error'] and s['argument']=='BoundingBoxArea' for s in row['steps'])
    assert row['region_type'][1].endswith('Standin')


def test_bbox_extent_saved_rows_and_declared_pixel_mutation():
    matrix=json.loads(MATRIX.read_bytes())
    assert matrix['original_observations']['1.24.4']==matrix['original_observations']['2.5.4']
    for rows in matrix['original_observations'].values():
        # Hash the exact original final-JSON formatting, not normalized values.
        assert hashlib.sha256((json.dumps(rows,indent=2)+'\n').encode()).hexdigest()=='776dde11a2e0e2881fd7215013c72c4e90bd5429637f037b54cc2dcccfbc137e'
        assert [row['id'] for row in rows]==[s['id'] for s in CASES]
        changed=0
        for row,spec in zip(rows,CASES):
            assert all(row['checks'].values()) and not row['constructor_warnings']
            before=row['input_before']['items'][0];after=row['input_after']['items'][0]
            assert row['input_before']['items'][1]==row['input_after']['items'][1]
            assert {k:v for k,v in before.items() if not k.startswith('raw_')}=={k:v for k,v in after.items() if not k.startswith('raw_')}
            raw_before=zlib.decompress(base64.b64decode(before['raw_zlib_base64']))
            raw_after=zlib.decompress(base64.b64decode(after['raw_zlib_base64']))
            assert hashlib.sha256(raw_before).hexdigest()==before['raw_sha256']
            assert hashlib.sha256(raw_after).hexdigest()==after['raw_sha256']
            expected=bytearray(raw_before);width=int(before['dtype'][-1])
            for action,argument in spec['sequence']:
                if action=='set-pixel':
                    position,pixel=argument;flat=0
                    for index,size in zip(position,spec['shape']):flat=flat*size+index
                    expected[flat*width:(flat+1)*width]=pixel.to_bytes(width,'little',signed='i' in before['dtype'])
            assert bytes(expected)==raw_after
            changed+=row['input_before']!=row['input_after']
        assert changed==6


def test_bbox_extent_saved_scalar_types_errors_and_FP_phases():
    matrix=json.loads(MATRIX.read_bytes())
    for rows in matrix['original_observations'].values():
        assert len(rows)==14 and sum(len(r['steps']) for r in rows)==161
        assert sum(len(r['FP_events']) for r in rows)==4
        scalars=attribute_errors=floating_errors=0
        for row,spec in zip(rows,CASES):
            if row['constructor_error']:
                assert spec['id'].endswith('bad-spacing-length')
                assert row['constructor_error']==['builtins','ValueError',"spacing isn't a scalar nor a sequence of shape (2,), got [1. 2. 3.]."]
                assert row['steps']==[] and row['returned_count'] is None and row['region_type'] is None
                continue
            assert row['returned_count']==2 and row['region_type']==['skimage.measure._regionprops','RegionProperties']
            assert [(s['action'],s['argument']) for s in row['steps']]==[(a,b) for a,b in spec['sequence']]
            for step in row['steps']:
                result=step['result']
                if isinstance(result,dict) and 'raw' in result:
                    assert result['type']=='float64' and result['dtype']=='<f8' and len(bytes.fromhex(result['raw']))==8
                    scalars+=1
                if step['error']:
                    if step['argument']=='BoundingBoxArea':
                        assert step['action']=='attr' and step['error']==['builtins','AttributeError',"'RegionProperties' object has no attribute 'BoundingBoxArea'"]
                        attribute_errors+=1
                    else:
                        assert spec['id'].endswith('zero-spacing') and step['argument'] in ('extent','Extent')
                        assert step['error']==['builtins','FloatingPointError','invalid value encountered in scalar divide']
                        floating_errors+=1
                for warning in step['warnings']:assert warning[:2]==['builtins','FutureWarning']
            for event in row['FP_events']:
                assert spec['id']=='regionprops-bbox-extent/empty-slice2d-cache-False'
                assert event['message']=='invalid value' and event['flags']==8
                assert row['steps'][event['phase']]['argument'] in ('extent','Extent')
            if row['FP_events']:assert [e['phase'] for e in row['FP_events']]==[3,5,8,12]
        assert (scalars,attribute_errors,floating_errors)==(126,13,4)
