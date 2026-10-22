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

@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['contiguous','reversed','planar'])
def test_antipodal_ulp_neighborhoods(workers,dtype,layout):
    from skimage.color import deltaE_ciede2000
    h=np.array([0,164,345,180,360],dtype=dtype)*dtype(np.pi/180)
    h=np.concatenate((h,np.nextafter(h,dtype(-np.inf)),np.nextafter(h,dtype(np.inf))))
    lab=np.stack((np.resize(np.array([0,15.999,16,50,100],dtype=dtype),h.shape),20*np.cos(h),20*np.sin(h)),axis=-1)
    variants=[]
    for item in lab:
        for dx in range(-3,4):
            for dy in range(-3,4):
                value=item.copy()
                for dim,n in [(1,dx),(2,dy)]:
                    for _ in range(abs(n)):value[dim]=np.nextafter(value[dim],dtype(np.inf if n>0 else -np.inf))
                variants.append(value)
    a=np.array(variants);b=a.copy();b[:,1:]*=-1
    if layout=='reversed':a=a[::-1];b=b[::-1]
    if layout=='planar':a=np.ascontiguousarray(a.T).T;b=np.ascontiguousarray(b.T).T
    previous=xx.get_num_threads();xx.set_num_threads(workers)
    fallback=xx.set_fallback_enabled(False)
    try:actual=xx.color.deltaE_ciede2000(a,b)
    finally:
        xx.set_num_threads(previous)
        xx.set_fallback_enabled(fallback)
    tol=(2e-6,2.5e-4) if dtype==np.float32 else (2e-13,2e-13)
    np.testing.assert_allclose(actual,deltaE_ciede2000(a,b),rtol=tol[0],atol=tol[1])
