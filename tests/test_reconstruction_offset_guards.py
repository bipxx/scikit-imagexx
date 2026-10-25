"""Safe offset owner/edge checks, no stale or out-of-bounds ndarray reads."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('reconstruction_offset_pipeline',Path(__file__).with_name('test_reconstruction_native_pipeline.py'));M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
backend=M.backend

def test_valid_offset_span_ends_exactly_at_root_end(backend):
    core,_=backend;a=np.zeros((4,5),'f4');b=np.ones_like(a);fp=np.ones((3,3),bool);root=np.full(8,1,np.uintp);offset=root[-2:]
    assert offset.ctypes.data+offset.nbytes==root.ctypes.data+root.nbytes
    result,_=core.reconstruction_run(a,b,fp,offset,0)
    assert result.shape==(6,7) and np.all(result==0)

def test_empty_offset_scalar_span_is_defined(backend):
    core,_=backend;a=np.array(1,'f4');b=np.array(2,'f4');fp=np.array(False);offset=np.array([],np.uintp)
    result,_=core.reconstruction_run(a,b,fp,offset,0)
    assert result.shape==() and result[()]==np.float32(1)

def test_wrong_length_with_valid_extent_rejects_before_offset_index(backend):
    core,_=backend;a=np.zeros((4,5),'f4');b=np.ones_like(a);fp=np.ones((3,3),bool);offset=np.array([1],np.uintp)
    core.reconstruction_counts(True)
    with pytest.raises(ValueError,match='rank/footprint mismatch'):core.reconstruction_run(a,b,fp,offset,0)
    assert core.reconstruction_counts(False)['run']==dict(attempted=1,completed=0)

def test_inbounds_dummy_owner_is_conservatively_rejected(backend):
    core,_=backend;a=np.zeros((4,5),'f4');b=np.ones_like(a);fp=np.ones((3,3),bool);root=np.array([1,1],np.uintp)
    # This view is IN BOUNDS. It exercises exotic-owner rejection without ever
    # constructing a stale pointer or inviting a read beyond valid storage.
    offset=np.lib.stride_tricks.as_strided(root,shape=(2,),strides=root.strides)
    assert offset.nbytes==root.nbytes and offset.ctypes.data==root.ctypes.data
    with pytest.raises(ValueError,match='ndarray-backed storage'):core.reconstruction_run(a,b,fp,offset,0)
