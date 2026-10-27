"""Boundary refusals only: original unchecked inputs are never executed."""
import pytest


@pytest.mark.parametrize('api',('points','grid'))
def test_polygon_empty_vertices_refused(api):
    import numpy as np
    import skimagexx as xx
    empty=np.empty(0,dtype=np.float64)
    assert xx.get_num_threads()==1
    with pytest.raises(ValueError,match='vertices must be nonempty'):
        if api=='points':xx._core.points_in_poly(np.zeros(1),np.zeros(1),empty,empty,1)
        else:xx._core.grid_points_in_poly(empty,empty,1,1,1)
    assert xx.get_num_threads()==1


@pytest.mark.parametrize('shape',((3,),[3]))
def test_polygon_short_shape_refused(shape):
    import numpy as np
    import skimagexx as xx
    with pytest.raises(ValueError,match='requires at least two dimensions'):
        xx.measure.grid_points_in_poly(shape,np.array([[0.,0.],[1.,0.],[0.,1.]]))
    assert xx.get_num_threads()==1
