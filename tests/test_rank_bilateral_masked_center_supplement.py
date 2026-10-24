"""Defined masked-high bilateral centers, separate from frozen708/96/680.

Only permanently masked pixels exceed the captured histogram capacity. The
original forms i-s0/i+s1 for i0..bins-1; uint16 center g need not be a bin index.
No undefined histogram-read, alias or signed arithmetic probe is sent upstream.
"""
import importlib.util
import os
import sys
from pathlib import Path
import warnings
import numpy as np
import pytest
from skimage.filters import rank as reference

spec=importlib.util.spec_from_file_location('rank_bilateral_masked_private_shell',Path(__file__).with_name('test_rank_bilateral_pipeline.py'))
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
core=pipeline.core
NAMES=pipeline.contracts.NAMES
RADII=((10,10),(10,65536),(1,2001),(1,65536),(-1999,10),(-65534,10),(0,0),(sys.maxsize,10),(3-sys.maxsize,10),(10,-sys.maxsize-1),(10,sys.maxsize-3))
CASES=tuple((name,high,field,radius,dtype) for name in NAMES for high in (2000,65535) for field in ('s0','s1') for radius in RADII for dtype in ('u1','u2','f8'))
assert len(CASES)==396


def observe(function,name,high,field,radius,dtype):
    image=(np.arange(9,dtype=np.uint16)%4).reshape(3,3);fp=np.ones((3,3),np.uint8)
    mask=np.ones(image.shape,np.uint8);mask[1,1]=0;out=np.full(image.shape,99,np.dtype(dtype));events=[]
    assert image.max()==3 and mask[1,1]==0 and high<=65535
    assert not np.shares_memory(image,out) and not np.shares_memory(mask,out)
    # This pre-call source safety certificate uses CAPTURED bins4, not guint16
    # capacity. It includes defined edges that a65536-bin guard would reject.
    assert all(-sys.maxsize-1<=i-radius[0]<=sys.maxsize and -sys.maxsize-1<=i+radius[1]<=sys.maxsize for i in range(4))
    class Radius:
        def __init__(self,label,value):self.label,self.value=label,value
        def __index__(self):
            events.append(self.label+'.__index__')
            if self.label==field:image[1,1]=high
            return self.value
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result=function(image,fp,mask=mask,out=out,s0=Radius('s0',radius[0]),s1=Radius('s1',radius[1]));error=None
        except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
    return dict(error=error,events=events,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],out=pipeline.contracts.array(out),image=pipeline.contracts.array(image),mask=pipeline.contracts.array(mask),footprint=pipeline.contracts.array(fp),result=None if result is None else pipeline.contracts.array(result))


@pytest.fixture(autouse=True,params=(1,4))
def workers(request):
    previous=core.get_num_threads()
    if request.param==4 and os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')=='1':
        with pytest.raises(ValueError,match='this build has no OpenMP support'):core.set_num_threads(4)
        assert core.get_num_threads()==previous==1
        pytest.skip('OpenMP-off rejects worker4; not parallel parity')
    core.set_num_threads(request.param)
    yield
    core.set_num_threads(previous)


@pytest.mark.parametrize('case',CASES,ids=['/'.join(map(str,c)) for c in CASES])
def test_every_defined_masked_center_observation(case):
    name,high,field,radius,dtype=case
    expected=observe(getattr(reference,name),*case)
    assert expected['error'] is None and expected['events']==['s0.__index__','s1.__index__']
    assert observe(getattr(pipeline.native,name),*case)==expected


@pytest.mark.parametrize('name',NAMES)
def test_mask_does_not_blanket_zero_center(name):
    expected={'mean_bilateral':1,'pop_bilateral':8,'sum_bilateral':12}[name]
    observed=observe(getattr(pipeline.native,name),name,65535,'s0',(10,65536),'f8')
    assert observed['error'] is None and observed['out']['values'][1][1]==expected


def mask_feedback_observe(function,offset):
    image=(np.arange(25,dtype=np.uint16)%4).reshape(5,5);fp=np.ones((3,3),np.uint8)
    owner=np.ones(26,np.uint8);owner[::4]=0;mask=owner[:25].reshape(5,5);out=owner[offset:offset+25].reshape(5,5)
    # Every possible future histogram read is a valid bin, regardless of mask
    # feedback. These already-defined aliases remain controls, not new scope.
    assert image.max()==3
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');result=function(image,fp,mask=mask,out=out)
    return dict(owner=pipeline.contracts.array(owner),mask=pipeline.contracts.array(mask),out=pipeline.contracts.array(out),result=pipeline.contracts.array(result),warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices])


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('offset',(0,1))
def test_existing_narrow_feedback_with_all_values_valid(name,offset):
    assert mask_feedback_observe(getattr(pipeline.native,name),offset)==mask_feedback_observe(getattr(reference,name),offset)


@pytest.mark.parametrize('operation',(26,27,28))
def test_uncertified_feedback_may_reenable_high_bin_native_only(operation):
    # Never send this potential histogram escape to bounds-disabled Cython.
    # The offset output can re-enable a masked high source during traversal.
    image=(np.arange(25,dtype=np.uint16)%4).reshape(5,5);image[2,2]=2000
    fp=np.ones((3,3),np.uint8);owner=np.ones(26,np.uint8);owner[12]=0
    mask=owner[:25].reshape(5,5);out=owner[1:].reshape(5,5,1);before=owner.copy()
    with pytest.raises(ValueError,match='rank image exceeded histogram bounds'):
        core.rank_bilateral_run(image,fp,mask,out,[0,0],4,operation,10,65536)
    assert owner.tobytes()==before.tobytes()


@pytest.mark.parametrize('name',NAMES)
def test_masked_center_never_forwards(name,monkeypatch):
    expected=observe(getattr(reference,name),name,65535,'s0',(10,65536),'f8')
    def forbidden(*a,**k):raise AssertionError('original bilateral forwarding')
    for target in (reference,pipeline.contracts.bilateral_module):
        for operation in NAMES:monkeypatch.setattr(target,operation,forbidden)
    assert observe(getattr(pipeline.native,name),name,65535,'s0',(10,65536),'f8')==expected


@pytest.mark.parametrize('operation',tuple(range(9,16))+tuple(range(17,26)))
def test_old_prepared_route_guards_are_unchanged_native_only(operation):
    # Preserve existing0..25 admission, not a new original-parity assertion.
    # Never send these uncertified prepared buffers to upstream functions.
    image=(np.arange(9,dtype=np.uint16)%4).reshape(3,3);image[1,1]=2000
    fp=np.ones((3,3),np.uint8);mask=np.ones(image.shape,np.uint8);mask[1,1]=0
    out=np.full((3,3,1),99,np.float64)
    message='rank center exceeded histogram bounds' if operation<13 else 'rank image exceeded histogram bounds'
    with pytest.raises(ValueError,match=message):
        if operation<13:core.rank_run(image,fp,mask,out,[0,0],4,operation)
        elif operation<16:core.rank_statistics_run(image,fp,mask,out,[0,0],4,operation)
        else:core.rank_percentile_run(image,fp,mask,out,[0,0],4,operation,.25,.75)
