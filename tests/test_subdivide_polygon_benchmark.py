"""Prospective spline cohorts and scalar oracle; no image-library calls."""
import ast
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prospective_subdivide',ROOT/'benchmarks/subdivide_polygon.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_frozen_cohorts_cover_all_degrees_and_bounded_layouts():
    contract=json.loads(module.CONTRACT.read_bytes());rows=module.case_specs()
    assert len(rows)==len({r['case_id']for r in rows})==14
    assert (contract['threads'],contract['repeat'],contract['target_ms'],contract['raw_samples'])==(1,7,19,196)
    for degree in range(1,8):assert {r['closed']for r in rows if r['degree']==degree}=={False,True}
    assert {r['layout']for r in rows}=={'C','F','reverse-rows','step-rows','step-columns'}
    assert all(256<=r['points']<=contract['max_points']and r['points']*4<=contract['max_backing_elements']for r in rows)
    for r in rows:
        count=r['base_points']if r['closed']else r['base_points']-(r['degree']+1)//2
        assert r['expected_output_shape']==[2*count+(1 if r['closed']else 2 if r['preserve_ends']else 0),2]


def test_pascal_derivation_matches_all_fourteen_original_mask_vectors():
    frozen=json.loads((ROOT/'coverage/contracts/subdivide-polygon.json').read_bytes())['subdivision_masks']
    for degree in range(1,8):
        even,odd=module.integer_masks(degree)
        assert [even,odd]==frozen[str(degree)]
        assert sum(even)==sum(odd)==1<<degree
    # Odd degrees align their zero at the source's beginning, never at the end.
    assert all(module.integer_masks(d)[1][0]==0 for d in(3,5,7))


@pytest.mark.parametrize('preserve,expected',[(False,[(1.,1.),(3.,3.),(5.,3.),(7.,1.)]),(True,[(0.,0.),(1.,1.),(3.,3.),(5.,3.),(7.,1.),(8.,0.)])])
def test_independent_degree_two_oracle_and_endpoint_placement(preserve,expected):
    assert module.subdivision_oracle([(0.,0.),(4.,4.),(8.,0.)],2,preserve)==expected


def test_independent_degree_one_special_pair():
    assert module.subdivision_oracle([(0.,1.),(2.,3.)],1,False)==[(1.,2.),(1.,2.)]


def test_independent_circular_wrap_closes_computed_point():
    assert module.subdivision_oracle([(2.,3.),(2.,3.)],7,True)==[(2.,3.)]*3


def test_runner_times_whole_registered_calls_after_ownership_and_oracle():
    tree=ast.parse((ROOT/'benchmarks/subdivide_polygon.py').read_bytes())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    timer=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure')
    assert [ast.unparse(n.body)for n in timer.args[:2]]==['reference(coordinates, degree, preserve)','native(coordinates, degree, preserve)']
    checks=[n for n in ast.walk(main)if isinstance(n,ast.Assert)]
    assert any("calls == ['subdivide_polygon_apply']"in ast.unparse(n)and n.lineno<timer.lineno for n in checks)
    assert any('a == b == c'in ast.unparse(n)and n.lineno<timer.lineno for n in checks)
    assert any('backing_before == snapshot(owner)'in ast.unparse(n)and n.lineno>timer.lineno for n in checks)
