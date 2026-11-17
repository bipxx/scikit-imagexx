"""Affected additive source/index guards only; no scientific import."""
import hashlib
import json
import pytest
import draw_multiblock_lbp_overlap_contract as c


def test_exact_predecessor_and_successor_projection():
    d=json.loads(c.CONTRACT.read_bytes());old=(c.ROOT/d['predecessor']['contract']).read_bytes()
    assert hashlib.sha256(old).hexdigest()==d['predecessor']['contract_sha256']
    historical=json.loads(old)
    assert c.legacy.cases()==historical['cases'] and len(c.legacy.cases())==93
    assert c.cases()==historical['cases']+[d['added_case']] and len(c.cases())==94
    assert d['bounds']==c.source_safe(c.added_case())
    assert d['indices']==c.source_indices(c.added_case())
    for name,pin in d['source_freeze'].items():
        assert hashlib.sha256((c.ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    for name,pin in d['predecessor']['raw_source_sha256'].items():
        assert hashlib.sha256((c.ROOT/name).read_bytes()).hexdigest()==pin,name
    assert d['original_calls']==d['helper_calls']==d['native_calls']==d['timed_batches']==0
    assert d['root_GO'] is None


def test_repeated_physical_center_is_not_logical_source_center():
    index=c.source_indices(c.added_case());patches=index['patches']
    assert [i for i,p in enumerate(patches) if [1,1] in p]==[0,2,4,6]
    assert index['normalized_source_center']==[]
    assert patches==[[[0,0],[0,1],[1,0],[1,1]],[],[[0,1],[0,2],[1,1],[1,2]],[],[[1,1],[1,2],[2,1],[2,2]],[],[[1,0],[1,1],[2,0],[2,1]],[]]
    old=next(r for r in c.legacy.cases() if r['id'].endswith('/wrapped-overlap'))
    sets=[set(map(tuple,p)) for p in c.source_indices(old)['patches']]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    assert [bool(c.added_case()['code'] & (1<<(7-i))) for i in range(8)]==[True,False]*4


@pytest.mark.parametrize('fault',({'shape':[30,30,3]},{'alpha':0.4},{'r':-2},{'code':171}))
def test_unreviewed_added_recipe_refused(fault):
    with pytest.raises(AssertionError):c.source_safe(dict(c.added_case(),**fault))
