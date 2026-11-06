"""Platform-neutral installed ownership/dtype/shape consumers; Linux saved FP is separate."""
import importlib,inspect,sys
import numpy as np
import pytest
import skimagexx as xx
DTYPES=('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','float32','float64')

@pytest.fixture(scope='module')
def modules():
    mark=importlib.import_module('skimagexx._mark_boundaries');lbp=importlib.import_module('skimagexx._draw_multiblock_lbp')
    assert mark._core is lbp._core is xx._core
    assert xx.segmentation.mark_boundaries is mark.mark_boundaries and xx.feature.draw_multiblock_lbp is lbp.draw_multiblock_lbp
    assert xx.get_num_threads()==1;xx.set_fallback_enabled(False)
    for name in ('mark_boundaries_prepare','mark_boundaries_mask','mark_boundaries_outline','mark_boundaries_paint','_lbp_prepare','_lbp_binary','_lbp_assign'):
        fn=getattr(xx._core,name);assert inspect.isbuiltin(fn)and fn.__module__=='skimagexx._core'
    return mark,lbp

@pytest.mark.parametrize('dtype',DTYPES)
def test_lbp_real_public_numeric_dtype_copy_shape_ownership(dtype,modules):
    _,lbp=modules;image=np.zeros((2,3),dtype=dtype);before=image.tobytes();image.flags.writeable=False
    actual=lbp.draw_multiblock_lbp(image,0,0,0,0)
    expected_dtype=image.dtype if image.dtype.kind=='f'else np.dtype('float64')
    assert actual.dtype==expected_dtype and actual.shape==(2,3,3)and actual.flags.owndata and actual.flags.writeable
    assert not np.shares_memory(actual,image)and not np.any(actual)and image.tobytes()==before

@pytest.mark.parametrize('dtype',DTYPES)
def test_mark_real_public_numeric_dtype_binary_label_ownership(dtype,modules):
    mark,_=modules;image=np.zeros((2,2),dtype=dtype);labels=np.array([[0,1],[1,0]],dtype=np.uint8);labels.flags.writeable=False
    actual=mark.mark_boundaries(image,labels,mode='thick')
    assert actual.shape==(2,2,3)and actual.dtype==np.dtype('float32'if image.dtype.kind=='f'and image.dtype.itemsize<=4 else'float64')
    assert actual.tolist()==[[[1,1,0],[1,1,0]],[[1,1,0],[1,1,0]]]and actual.flags.owndata
    assert not np.shares_memory(actual,image)and not np.shares_memory(actual,labels)

@pytest.mark.parametrize('partition',('rank5','swapped-labels','nonexact-wide-labels'))
def test_mark_explicit_platform_neutral_refused_partitions(partition,modules):
    mark,_=modules;image=np.zeros((2,2),dtype=np.uint8);labels=np.array([[0,1],[1,0]],dtype=np.uint8)
    if partition=='rank5':image=np.zeros((1,1,1,1,3),dtype=np.float32)
    elif partition=='swapped-labels':labels=labels.astype('>u2')
    else:labels=np.array([[2**54,1],[1,0]],dtype=np.int64)
    with pytest.raises(NotImplementedError):mark.mark_boundaries(image,labels,mode='thick')

def test_complete_arity_and_single_global_worker_state(modules):
    mark,lbp=modules
    for fn in (mark.mark_boundaries,lbp.draw_multiblock_lbp):
        with pytest.raises(TypeError):fn(np.zeros((2,2),dtype=np.uint8))
    old=xx.get_num_threads()
    try:
        for workers in (1,4):
            xx.set_num_threads(workers)
            assert mark._core.get_num_threads()==lbp._core.get_num_threads()==xx.get_num_threads()==workers
    finally:xx.set_num_threads(old)
    assert old==1
