"""Defined zero-population u16 radius-hook parity, separate from frozen708.

Original bounds safety: every pixel changed above captured bins is protected
by an entirely zero prepared footprint or mask. No aliases/resizes/dtype hooks.
The nonzero-population histogram-escape guard is native-only, never upstream.
"""
import importlib.util
import os
from pathlib import Path
import warnings
import numpy as np
import pytest
from skimage.filters import rank as reference

spec=importlib.util.spec_from_file_location('rank_bilateral_mutation_private_shell',Path(__file__).with_name('test_rank_bilateral_pipeline.py'))
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
core=pipeline.core
NAMES=pipeline.contracts.NAMES
STATES=('zero-fp-high-all','zero-fp-high-tail','zero-mask-high-all','zero-mask-high-tail','zero-fp-enable-mask-high','full-fp-disable-mask-high','full-fp-enable-mask-valid','full-fp-rewrite-valid','full-fp-empty-selected')
CASES=tuple((name,state,field,output,radius) for name in NAMES for state in STATES for field in ('s0','s1') for output in ('u1','f8') for radius in ((10,10),(0,0),(-3,7)))


def observe(function,name,state,field,output,radius):
    assert state in STATES
    image=(np.arange(12,dtype=np.uint16)%4).reshape(3,4)
    fp=np.zeros((3,3),np.uint8) if state.startswith('zero-fp') else np.ones((3,3),np.uint8)
    mask=np.zeros(image.shape,np.uint8) if state.startswith('zero-mask') or state in ('zero-fp-enable-mask-high','full-fp-enable-mask-valid') else np.ones(image.shape,np.uint8)
    out=np.full(image.shape,99,np.dtype(output));events=[]
    assert not np.shares_memory(image,out) and not np.shares_memory(mask,out)
    # This certificate applies BEFORE any bounds-disabled original call. A
    # high source is sampled only as the valid center value, not histo[index].
    if 'high' in state:assert state.startswith(('zero-fp','zero-mask')) or state=='full-fp-disable-mask-high'
    pointers={key:value.ctypes.data for key,value in dict(image=image,mask=mask,out=out,fp=fp).items()}
    class Radius:
        def __init__(self,label,value):self.label,self.value=label,value
        def __index__(self):
            events.append(self.label+'.__index__')
            if self.label==field:
                if 'high-tail' in state:image[-1,-1]=2000
                elif 'high' in state:image.fill(2000)
                elif state=='full-fp-rewrite-valid':image.fill(2)
                if 'enable-mask' in state:mask.fill(1)
                if 'disable-mask' in state:mask.fill(0)
            return 0 if state=='full-fp-empty-selected' else self.value
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            result=function(image,fp,mask=mask,out=out,s0=Radius('s0',radius[0]),s1=Radius('s1',radius[1]));error=None
        except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
    tracked=dict(image=image,mask=mask,out=out,fp=fp)
    return dict(error=error,events=events,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],after={key:pipeline.contracts.array(value) for key,value in tracked.items()},pointer_stable={key:value.ctypes.data==pointers[key] for key,value in tracked.items()},result=None if result is None else dict(pipeline.contracts.array(result),is_out=result is out,shares_out=bool(np.shares_memory(result,out))))


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
def test_defined_radius_mutation_against_original(case):
    name,state,field,output,radius=case
    expected=observe(getattr(reference,name),*case)
    assert expected['error'] is None and expected['events']==['s0.__index__','s1.__index__']
    if 'high' in state:assert np.count_nonzero(expected['after']['out']['values'])==0
    actual=observe(getattr(pipeline.native,name),*case)
    assert actual==expected


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('field',('s0','s1'))
@pytest.mark.parametrize('radius',((10,10),(0,0)))
def test_nonzero_population_histogram_escape_is_native_only_prewrite(name,field,radius):
    # Source-undefined histo[2000] with captured bins4: NEVER invoke original.
    image=(np.arange(12,dtype=np.uint16)%4).reshape(3,4);fp=np.ones((3,3),np.uint8)
    out=np.full(image.shape,99,np.uint8)
    class Radius:
        def __init__(self,label,value):self.label,self.value=label,value
        def __index__(self):
            if self.label==field:image.fill(2000)
            return self.value
    with pytest.raises(ValueError,match='rank image exceeded histogram bounds'):
        getattr(pipeline.native,name)(image,fp,out=out,s0=Radius('s0',radius[0]),s1=Radius('s1',radius[1]))
    assert np.all(out==99) and np.all(image==2000)


@pytest.mark.parametrize('name',NAMES)
def test_ordinary_and_zero_population_do_not_forward_original(name,monkeypatch):
    expected=observe(getattr(reference,name),name,'zero-fp-high-all','s0','f8',(10,10))
    def forbidden(*a,**k):raise AssertionError('upstream bilateral forwarding')
    for target in (reference,pipeline.contracts.bilateral_module):
        for operation in NAMES:monkeypatch.setattr(target,operation,forbidden)
    assert observe(getattr(pipeline.native,name),name,'zero-fp-high-all','s0','f8',(10,10))==expected


def test_linked_interface_and_debug_assertions():
    assert core.rank_bilateral_interface=='rank-bilateral-v1' and core.rank_bilateral_capabilities==0x1c000000
    assert (core.rank_histogram_capabilities,core.rank_alias_capabilities,core.rank_percentile_capabilities)==(8191,480,0x3fe0000)
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert core.gil_assertions is True
