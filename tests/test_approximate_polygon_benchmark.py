"""Bounded prospective RDP cohort: no image libraries or actual benchmark calls."""
import importlib.util
import ast
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prospective_rdp',ROOT/'benchmarks/approximate_polygon.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_frozen_cases_have_meaningful_geometry_and_bounded_storage():
    contract=json.loads(module.CONTRACT.read_bytes());rows=module.case_specs()
    assert len(rows)==len({r['case_id']for r in rows})==9
    assert (contract['threads'],contract['repeat'],contract['target_ms'],contract['raw_samples'])==(1,7,19,126)
    assert {r['geometry']for r in rows}=={'collinear','zigzag','smooth'}
    assert {r['layout']for r in rows}=={'C','F','reverse-rows','step-rows','step-columns'}
    assert all(3<=r['points']<=contract['max_points']and r['tolerance']>0 for r in rows)
    assert max(r['points']*4 for r in rows)<=contract['max_backing_elements']
    assert 'NumPy scalar atan2/sin/cos' in contract['provider_frontier']


@pytest.mark.parametrize('tolerance,expected',[(0.9999999999999999,[0,1,2]),(1.,[0,2]),(1.0000000000000002,[0,2])])
def test_independent_oracle_strict_threshold(tolerance,expected):
    assert module.selection_oracle([(0.,0.),(1.,1.),(0.,2.)],tolerance)==expected


def test_independent_oracle_uses_endpoint_distance_outside_segment():
    assert module.selection_oracle([(0.,0.),(0.,-2.),(0.,2.),(0.,1.)],0.5)==[0,1,2,3]


def test_independent_oracle_handles_closed_and_repeated_endpoints():
    assert module.selection_oracle([(0.,0.),(1.,1.),(0.,0.),(1.,-1.),(0.,0.)],0.1)==[0,1,2,3,4]


def test_runner_times_whole_public_calls_after_exact_ownership_checks():
    tree=ast.parse((ROOT/'benchmarks/approximate_polygon.py').read_bytes())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    timer=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure')
    assert [ast.unparse(n.body)for n in timer.args[:2]]==['reference(coordinates, tolerance)','native(coordinates, tolerance)']
    checks=[n for n in ast.walk(main)if isinstance(n,ast.Assert)]
    assert any("calls == ['approximate_polygon_apply']"in ast.unparse(n)and n.lineno<timer.lineno for n in checks)
    assert any('a == b == c'in ast.unparse(n)and n.lineno<timer.lineno for n in checks)
    assert any('backing_before == snapshot(owner)'in ast.unparse(n)and n.lineno>timer.lineno for n in checks)


def test_oracle_elementary_math_provider_is_explicit_and_shared_selection_is_not_used():
    tree=ast.parse((ROOT/'benchmarks/approximate_polygon.py').read_bytes())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    oracle=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='selection_oracle')
    assert ast.unparse(oracle.args[-1])=='implementation._scalar_trig'
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='selection_oracle')
    assert not any(isinstance(n,ast.Name)and n.id in('xx','skimage','reference','_core')for n in ast.walk(helper))
