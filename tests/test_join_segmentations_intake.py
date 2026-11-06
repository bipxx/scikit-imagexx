"""Installed NumPy fixture/observer intake; ORIGINAL_REFUSED for all50recipes."""
import numpy as np
import pytest
from join_segmentations_contract import cases
from join_segmentations_observer import observe


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_join_fixture_observer_with_original_refused(spec):
    def refused(*args,**kwargs):raise RuntimeError('ORIGINAL_REFUSED_AT_JOIN_INTAKE')
    row=observe(refused,spec,np)
    assert row['error']==['builtins','RuntimeError','ORIGINAL_REFUSED_AT_JOIN_INTAKE']
    assert row['result']is None and row['original_python_helpers']=={}and row['source_frame']=={}
    assert row['map_protocols']==[]and all(row['checks'].values())
