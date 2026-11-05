"""No original constructors/properties: refused intake and synthetic schema probe."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import warnings
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
    assert len({s['id'] for s in CASES})==14 and matrix['original_observations']=={} and matrix['original_calls']==0
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
