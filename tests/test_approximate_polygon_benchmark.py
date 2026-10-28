"""Bounded prospective RDP cohort: no image libraries or actual benchmark calls."""
import importlib.util
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
