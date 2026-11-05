"""Direct C/F binder destinations and real capacity boundary; no originals."""
import numpy as np
import pytest


@pytest.mark.parametrize('shape', ((3,4),(2,3,4)))
@pytest.mark.parametrize('order', ('C','F'))
@pytest.mark.parametrize('workers', (1,4))
def test_regionprops_direct_strides_current_mask_and_capacity(shape,order,workers):
    import skimagexx as xx
    old=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        labels=np.ones(shape,dtype=np.int32,order=order)
        prepared,geometry=xx._core.regionprops_state_prepare(labels)
        assert len(geometry)==1
        mask=np.empty(shape,dtype=bool,order=order)
        assert xx._core.regionprops_state_image(prepared,0,labels,mask) is True
        assert mask.all()
        mask[:]=False
        first=(0,)*len(shape);last=tuple(n-1 for n in shape)
        mask[first]=mask[last]=True
        # Cached public-mask edits govern coords even after underlying labels
        # change. Source geometry remains constructor-frozen.
        labels[first]=0
        expected=np.array([first,last],dtype=np.intp)
        output=np.full((2,len(shape)),12345,dtype=np.intp,order=order)
        assert xx._core.regionprops_state_coords(prepared,0,labels,mask,output) is True
        np.testing.assert_array_equal(output,expected)
        wrong=np.full((1,len(shape)),12345,dtype=np.intp,order=order)
        assert xx._core.regionprops_state_coords(prepared,0,labels,mask,wrong) is None
        assert (wrong==12345).all()
        fresh=np.empty_like(mask)
        assert xx._core.regionprops_state_image(prepared,0,labels,fresh) is True
        assert not fresh[first] and fresh[last] and int(fresh.sum())==labels.size-1
        assert xx._core.regionprops_state_population(mask)==2
        assert xx.get_num_threads()==workers
    finally:
        xx.set_num_threads(old)
