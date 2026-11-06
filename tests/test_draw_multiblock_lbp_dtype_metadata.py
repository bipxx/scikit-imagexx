"""Observed gray metadata regression; candidate-only representation/FP checks."""
import ctypes
import inspect
import os
import sys
from pathlib import Path
import numpy as np
import pytest
import skimagexx._draw_multiblock_lbp as owned
META={'units':'linear','source':'parity'}


@pytest.fixture(scope='module')
def fenv():
    from util_views_fenv import backend
    libc,helper=backend()
    def save():
        state=(ctypes.c_ubyte*32)();assert libc.fegetenv(ctypes.byref(state))==0;return state
    def restore(state):
        assert helper.reference_fenv_restore(ctypes.byref(state),32)==0
        assert bytes(save())==bytes(state)
    def hardware(state):
        raw=bytes(state);return raw[0:2],raw[4:6],raw[28:32]
    return save,restore,hardware


@pytest.mark.parametrize('dtype',('float16','float32','float64'))
@pytest.mark.parametrize('rgb',(False,True))
@pytest.mark.parametrize('swapped',(False,True))
@pytest.mark.parametrize('workers',(1,4))
def test_gray_stack_descriptor_and_rgb_copy_descriptor(dtype,rgb,swapped,workers,fenv):
    # Root's retained original failure supplies the grayscale metadata=None
    # contract; this test never resolves/calls an original or color helper.
    descriptor=np.dtype(dtype).newbyteorder('S'if swapped else'=')
    tagged=np.dtype(descriptor,metadata=META)
    shape=(9,10,3)if rgb else(9,10)
    values=[float(k%251)/255 for k in range(270 if rgb else 90)]
    image=np.array(values,dtype=tagged).reshape(shape);image.flags.writeable=False
    plain=image.view(descriptor);before=image.tobytes();input_dtype=image.dtype
    expected_dtype=tagged if rgb else np.dtype(image.dtype.type)
    # Safe descriptor-only cross-check of NumPy's stacked dtype resolution;
    # no stack/concatenate/conversion algorithm body is called.
    if not rgb:assert np.result_type(descriptor,descriptor,descriptor)==expected_dtype
    core=owned._core;assert inspect.isbuiltin(core._lbp_prepare)
    save,restore,hardware=fenv;outer=save();old=core.get_num_threads();policy=np.geterr(),np.geterrcall(),np.getbufsize();prior=sys.getprofile();calls=[]
    def guard(frame,event,value):
        if event=='call'and frame.f_globals.get('__name__','').startswith(('skimage.','scipy.ndimage')):raise AssertionError('ORIGINAL_PROVIDER_BODY_FORBIDDEN')
        if event=='c_return'and value is core._lbp_prepare:calls.append('_lbp_prepare')
    try:
        core.set_num_threads(workers);restore(outer)
        sys.setprofile(guard)
        try:actual=owned.draw_multiblock_lbp(image,1,1,2,2,170,alpha=.3);actual_FP=save()
        finally:sys.setprofile(prior)
        restore(outer);baseline=owned.draw_multiblock_lbp(plain,1,1,2,2,170,alpha=.3);baseline_FP=save()
        assert hardware(actual_FP)==hardware(baseline_FP)
        assert actual.dtype.str==expected_dtype.str and actual.dtype.metadata==(META if rgb else None)
        assert actual.dtype.isnative==(not swapped if rgb else True)
        assert actual.tobytes()==baseline.tobytes()and actual.shape==baseline.shape
        assert actual.flags.owndata and actual.flags.writeable and not np.shares_memory(actual,image)
        assert image.dtype is input_dtype and image.dtype.metadata==META and image.tobytes()==before and not image.flags.writeable
        assert calls==['_lbp_prepare']and core.get_num_threads()==workers
        assert (np.geterr(),np.geterrcall(),np.getbufsize())==policy
    finally:
        sys.setprofile(prior);core.set_num_threads(old);np.seterr(**policy[0]);np.seterrcall(policy[1]);np.setbufsize(policy[2]);restore(outer)
        assert bytes(save())==bytes(outer)
