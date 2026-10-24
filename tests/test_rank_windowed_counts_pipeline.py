"""Private opcode16 counts/pop policy proof; never whole/public admission.

A real separately compiled bind_rank module is required explicitly. The source
wrapper calls that core, without replacing public namespaces or registry.
"""
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import warnings
import numpy as np
import pytest
from skimage.filters import rank as original

ROOT=Path(__file__).parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

core=importlib.import_module(os.environ.get('SKIMAGEXX_WINDOWED_COUNTS_CORE','_rank_windowed_core'))
shell=load('skimagexx._rank_windowed_counts_candidate',ROOT/'python/skimagexx/_rank.py');shell._core=core
contracts=load('windowed_counts_frozen_statistics',ROOT/'tests/test_rank_statistics_contracts.py')
FROZEN=json.loads(contracts.MATRIX.read_text())
OLD_NAMES=('minimum','maximum','mean','sum','pop','gradient','median','modal','entropy',
           'autolevel','equalize','enhance_contrast','threshold','geometric_mean','noise_filter','subtract_mean',
           'autolevel_percentile','enhance_contrast_percentile','gradient_percentile','mean_percentile',
           'pop_percentile','subtract_mean_percentile','threshold_percentile','sum_percentile','percentile',
           'mean_bilateral','pop_bilateral','sum_bilateral')

@pytest.fixture(params=(1,4))
def active(request,monkeypatch):
    previous=core.get_num_threads();core.set_num_threads(request.param)
    def forbidden(*a,**kw):raise AssertionError('private counts policy forwarded upstream/fallback')
    monkeypatch.setattr(shell,'fallback',forbidden)
    try:yield shell
    finally:core.set_num_threads(previous)


def test_private_interface_and_initial_worker_policy():
    assert core.get_num_threads()==1
    assert core.rank_vector_interface=='rank-vector-v1' and core.rank_vector_capabilities==0x10000
    assert core.rank_histogram_capabilities==8191 and core.rank_alias_capabilities==480
    assert core.rank_statistics_capabilities==0xe000 and core.rank_percentile_capabilities==0x3fe0000 and core.rank_bilateral_capabilities==0x1c000000
    assert inspect.isbuiltin(core.rank_windowed_histogram_run) and core.rank_windowed_histogram_run.__module__==core.__name__
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert core.gil_assertions is True


@pytest.mark.parametrize('case',contracts.CASES,ids=lambda case:case[0])
def test_all995_frozen_observations_private_actual_binding(case,active):
    key,name,parameters=case
    expected=FROZEN['observations'][FROZEN['snapshots'][contracts.profile()][key]]
    assert contracts.observe(getattr(active,name),name,parameters)==expected,key


@pytest.mark.parametrize('dtype',(np.uint8,np.uint16))
@pytest.mark.parametrize('width',(15,16,17,31,32,33,63,64,65,67))
@pytest.mark.parametrize('depth',(3,4,7,31,32,33,63,64,65))
@pytest.mark.parametrize('uniform',(False,True))
def test_uniform_chunk_thresholds_and_normalizer_tails(dtype,width,depth,uniform,active):
    image=np.full((3,width),127,dtype) if uniform else ((np.arange(3*width)*17)%71).astype(dtype).reshape(3,width)
    if not uniform:image.flat[-1]=127
    # Cython reads output depth without checking it against prepared bins.
    # Pin the observed maximum before EVERY original call, including u16.
    assert depth<=256 if dtype is np.uint8 else depth<=int(image.max())+1
    fp=np.ones((1,65),np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected=original.windowed_histogram(image,fp,n_bins=depth)
        actual=active.windowed_histogram(image,fp,n_bins=depth)
    assert actual.dtype==expected.dtype and actual.strides==expected.strides and actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('name',OLD_NAMES)
@pytest.mark.parametrize('dtype',(np.uint8,np.uint16))
@pytest.mark.parametrize('mask_kind',('none','sparse'))
def test_unchanged_old0_to28_routes(name,dtype,mask_kind,active,monkeypatch):
    image=((np.arange(7*11)*7)%23).astype(dtype).reshape(7,11);fp=np.ones((3,5),np.uint8);fp.flat[::3]=0
    mask=None if mask_kind=='none' else ((np.arange(image.size)%4)!=0).astype(np.uint8).reshape(image.shape)
    left=np.full(image.shape,99.,np.float64);right=left.copy();options=dict(mask=mask,shift_x=1,shift_y=-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reference_return=getattr(original,name)(image,fp,out=left,**options)
        expected=reference_return.copy()
        def returned_metadata(result,out):
            return (result is out,np.shares_memory(result,out),result.base is out,result.dtype.str,
                    result.shape,result.strides,result.flags.owndata,result.flags.writeable)
        expected_return=returned_metadata(reference_return,left)
        def forbidden(*a,**k):raise AssertionError('old algorithm forwarded upstream')
        monkeypatch.setattr(original,name,forbidden)
        actual=getattr(active,name)(image,fp,out=right,**options)
    assert returned_metadata(actual,right)==expected_return,name
    assert actual.tobytes()==expected.tobytes() and right.tobytes()==left.tobytes(),name


@pytest.mark.parametrize('kind',('masked-zero','masked255','footprint-zero','stepped','narrow-high-outdepth'))
@pytest.mark.parametrize('out_dtype',(np.uint8,np.uint16,np.float32,np.float64))
def test_count_population_distributions_and_output_owner(kind,out_dtype,active):
    image=(np.arange(9*17)%7).astype(np.uint16).reshape(9,17);fp=np.ones((7,7),np.uint8);fp.flat[::3]=0;mask=None
    if kind=='masked-zero':mask=np.zeros(image.shape,np.uint8)
    elif kind=='masked255':mask=np.where(np.arange(image.size).reshape(image.shape)%3,255,0).astype(np.uint8)
    elif kind=='footprint-zero':fp.fill(0)
    elif kind=='stepped':image=image[:,::2]
    else:image[0,0]=65535
    depth=5;assert depth<=int(image.max())+1
    left=np.full(image.shape+(depth,),99,out_dtype);right=left.copy();before=image.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected=original.windowed_histogram(image,fp,out=left,mask=mask,n_bins=depth)
        actual=active.windowed_histogram(image,fp,out=right,mask=mask,n_bins=depth)
    assert actual is right and expected is left
    assert actual.tobytes()==expected.tobytes() and np.array_equal(image,before)
