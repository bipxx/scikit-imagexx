"""Closing allocation metadata parity on the genuine compiled max-tree core.

This focused supplement preserves all frozen 351 observations and kernels.
Modified-package test stages are implementation evidence, not qualification.
"""
from contextlib import contextmanager
import importlib
import sys

import numpy as np
import pytest
import skimagexx as xx

original=importlib.import_module('skimage.morphology.max_tree')
native=importlib.import_module('skimagexx._max_tree')
NAMES=('max_tree','max_tree_local_maxima','area_opening','diameter_opening','area_closing','diameter_closing')
DTYPES=('u1','u2','u4','u8','i1','i2','i4','i8','f4','f8')
TAGS={'units':'linear','source':'closing-metadata-regression'}


@pytest.fixture(params=(1,4),autouse=True)
def workers(request):
    previous=xx.set_fallback_enabled(False);old=xx.get_num_threads()
    assert native._core is xx._core and native._require_core() is xx._core
    assert xx.set_num_threads is xx._core.set_num_threads
    xx.set_num_threads(request.param)
    try:yield request.param
    finally:xx.set_num_threads(old);xx.set_fallback_enabled(previous)


@contextmanager
def forbid_upstream():
    codes={getattr(original,name).__code__ for name in NAMES}
    old=sys.getprofile()
    def reject(frame,event,arg):
        if event=='call' and frame.f_code in codes:
            raise AssertionError('upstream max-tree algorithm forwarding')
    sys.setprofile(reject)
    try:yield
    finally:sys.setprofile(old)


def fixture(dtype,layout):
    dtype=np.dtype(dtype,metadata=TAGS)
    a=(np.arange(20).reshape(4,5)%7).astype(dtype)
    if dtype.kind=='f':a/=7
    if layout in ('F','readonly-F'):a=np.asfortranarray(a)
    elif layout=='reverse':a=a[::-1,::-1]
    elif layout=='step':
        owner=np.zeros((8,10),dtype=dtype);owner[::2,::2]=a;a=owner[::2,::2]
    if layout=='readonly-F':a.flags.writeable=False
    return a


def compare(a,e):
    if isinstance(e,tuple):
        assert type(a) is tuple and len(a)==len(e)
        for actual,expected in zip(a,e):compare(actual,expected)
        return
    assert type(a) is type(e) is np.ndarray
    assert a.shape==e.shape and a.strides==e.strides and a.dtype==e.dtype
    assert a.dtype.metadata==e.dtype.metadata
    assert a.flags.owndata==e.flags.owndata and a.flags.writeable==e.flags.writeable
    assert a.flags.c_contiguous==e.flags.c_contiguous and a.flags.f_contiguous==e.flags.f_contiguous
    assert a.tobytes()==e.tobytes()


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('layout',('C','F','reverse','step','readonly-F'))
def test_tagged_all_functions_preserve_original_allocation_metadata(name,dtype,layout):
    image=fixture(dtype,layout);before=image.tobytes();descriptor=image.dtype
    expected=getattr(original,name)(image)
    xx._core.max_tree_counts(reset=True)
    with forbid_upstream():actual=getattr(xx.morphology,name)(image)
    compare(actual,expected)
    assert image.dtype is descriptor and image.dtype.metadata==TAGS and image.tobytes()==before
    counts=xx._core.max_tree_counts()
    if layout!='readonly-F':assert counts['aggregate']['completed']==1
    elif name in ('area_closing','diameter_closing'):
        # Literal invert produces a writable temporary, so its nested tree
        # build is pristine even though the outer closing uses staged filtering.
        assert counts['aggregate']['completed']==1 and counts['filter']['completed']==1
    else:
        assert counts['aggregate']['completed']==0
        assert counts['build']['completed']==1
    if name in ('area_opening','diameter_opening'):assert actual.dtype.metadata==TAGS
    elif name in ('area_closing','diameter_closing'):assert actual.dtype.metadata is None


@pytest.mark.parametrize('name',('area_closing','diameter_closing'))
@pytest.mark.parametrize('dtype',('u1','f4','f8'))
def test_inversion_provider_retains_literal_stage_order(monkeypatch,name,dtype):
    image=fixture(dtype,'C');subtract=np.subtract;events=[]
    def recording(*args,**kwargs):
        result=subtract(*args,**kwargs)
        events.append((args[1].dtype.metadata,result.dtype.metadata,kwargs['dtype'].metadata))
        return result
    monkeypatch.setattr(np,'subtract',recording)
    expected=getattr(original,name)(image);reference_events=list(events);events.clear()
    xx._core.max_tree_counts(reset=True)
    with forbid_upstream():actual=getattr(xx.morphology,name)(image)
    assert events==reference_events==[(TAGS,None,TAGS),(None,None,None)]
    compare(actual,expected)
    counts=xx._core.max_tree_counts()
    assert counts['aggregate']['completed']==0 and counts['filter']['completed']==1


@pytest.mark.parametrize('name',('area_closing','diameter_closing'))
def test_replaced_empty_provider_excludes_pristine_allocation(monkeypatch,name):
    image=fixture('f4','C');empty=np.empty;events=[]
    def recording(*args,**kwargs):
        events.append((args,kwargs));return empty(*args,**kwargs)
    monkeypatch.setattr(np,'empty',recording)
    expected=getattr(original,name)(image);events.clear()
    xx._core.max_tree_counts(reset=True)
    with forbid_upstream():actual=getattr(xx.morphology,name)(image)
    compare(actual,expected)
    assert events==[] # neither literal closing route requests np.empty
    counts=xx._core.max_tree_counts()
    assert counts['aggregate']['completed']==0 and counts['filter']['completed']==1
