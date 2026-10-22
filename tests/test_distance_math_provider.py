"""Exercise SVML compatibility even on AVX2-only and scalar CI machines."""
import json
from pathlib import Path
import numpy as np
import pytest
import skimagexx as xx
from skimagexx import _core

DATA=Path(__file__).parent/'data'

@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('repeat',[1,19,1025])
@pytest.mark.parametrize('layout',['contiguous','reversed','planar','broadcast'])
def test_saved_numpy124_svml_original_outputs(workers,repeat,layout):
    pairs=np.loadtxt(DATA/'ciede2000-sharma.txt')
    expected=np.tile(json.loads((DATA/'ciede2000-numpy124-svml.json').read_text())['values'],repeat)
    a=np.tile(pairs[:,:3],(repeat,1));b=np.tile(pairs[:,3:6],(repeat,1))
    if layout=='reversed':a=a[::-1];b=b[::-1];expected=expected[::-1]
    if layout=='planar':a=np.ascontiguousarray(a.T).T;b=np.ascontiguousarray(b.T).T
    if layout=='broadcast':
        a=np.broadcast_to(a[9],a.shape);b=np.broadcast_to(b[9],b.shape)
        expected=np.full(expected.shape,expected[9])
    old=xx.get_num_threads();xx.set_num_threads(workers)
    try:actual=_core.distance_metric(a,b,1,(1.,1.,1.,.045,.015),True,False,svml_angles=True)
    finally:xx.set_num_threads(old)
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)

def test_non_ciede_svml_profile_rejected():
    with pytest.raises(ValueError,match='SVML angles require CIEDE2000'):
        _core.distance_metric(np.ones((1,3)),np.ones((1,3)),0,(1.,1.,1.,.045,.015),True,False,svml_angles=True)
