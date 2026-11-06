"""Installed-only fixture/observer intake, after revised C-source safety peer.
Never run in pure prebuild; ordinary int64/uint64 outer recipes are excluded.
"""
import pytest
from find_boundaries_contract import cases
from find_boundaries_observer import reference_runtime,without_original_activity,observe


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_find_boundaries_fixture_observer_originals_refused(spec):
    np,original,_=reference_runtime()
    def refused(*args,**kwargs):raise RuntimeError('ORIGINAL_REFUSED_AT_FIND_BOUNDARIES_INTAKE')
    with without_original_activity(original):row=observe(refused,spec,np)
    assert row['error']==dict(type='RuntimeError',module='builtins',message='ORIGINAL_REFUSED_AT_FIND_BOUNDARIES_INTAKE')
    assert all(row['checks'].values())and row['result']is None
