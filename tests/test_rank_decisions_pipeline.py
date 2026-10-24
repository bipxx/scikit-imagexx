"""Private R7 source shell/real compiled decisions, never public admission."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import BuiltinFunctionType,SimpleNamespace
import warnings
import numpy as np
import pytest
from skimage.filters import rank as reference


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


if 'SKIMAGEXX_RANK_DECISIONS_BACKEND' not in os.environ and importlib.util.find_spec('_rank_decisions_core') is None:
    pytest.skip('Private compiled fixture _rank_decisions_core unavailable; supplementary private tests are not public parity',allow_module_level=True)

core=importlib.import_module(os.environ.get('SKIMAGEXX_RANK_DECISIONS_BACKEND','_rank_decisions_core'))
assert core.rank_decisions_interface=='rank-decisions-v1' and core.rank_decisions_capabilities==0x60000000
assert isinstance(core.rank_decisions_run,BuiltinFunctionType)
INITIAL_WORKERS=core.get_num_threads()
contracts=load('rank_decisions_private_contracts',Path(__file__).with_name('test_rank_decisions_contracts.py'))
shell=load('skimagexx._rank_decisions_private_shell',Path(__file__).parents[1]/'python/skimagexx/_rank.py')


class PreparedAdapter:
    """Route only prepared R7 buffers; existing native dtype providers stay owned."""
    def __getattr__(self,name):return getattr(core,name)
    def rank_run(self,image,fp,mask,out,shifts,bins,operation):
        if operation in (29,30):return core.rank_decisions_run(image,fp,mask,out,shifts,bins,operation)
        return core.rank_run(image,fp,mask,out,shifts,bins,operation)


shell._core=PreparedAdapter()


def otsu(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    return shell._apply(29,image,footprint,out,mask,shift_x,shift_y,shift_z)


def majority(image,footprint,*,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    return shell._apply(30,image,footprint,out,mask,shift_x,shift_y,shift_z)


native=SimpleNamespace(otsu=otsu,majority=majority)


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


@pytest.fixture(scope='module')
def frozen():return json.loads(contracts.MATRIX.read_text())


@pytest.mark.parametrize('case',contracts.CASES,ids=[c[0] for c in contracts.CASES])
def test_every_frozen_original_observation(case,frozen):
    key,name,parameters=case
    assert contracts.observe(getattr(native,name),name,parameters)==frozen['observations'][frozen['snapshots'][contracts.profile()][key]],key


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('ndim',(2,3))
@pytest.mark.parametrize('failure',contracts.PROVIDERS)
def test_all_provider_order_and_prewrite_traces(name,ndim,failure,frozen,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    assert contracts.provider_observe(name,ndim,failure)==frozen['provider_traces'][contracts.profile()][f'{name}/{ndim}/{failure}']


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('ndim',(2,3))
@pytest.mark.parametrize('dtype',('u1','u2'))
@pytest.mark.parametrize('kind',('dense','even','disconnected','empty','masked','shifted','tied','full-bin'))
@pytest.mark.parametrize('output',('same','u1','f4','f8'))
def test_independent_literal_neighborhood_decisions(name,ndim,dtype,kind,output,monkeypatch):
    monkeypatch.setattr(contracts,'reference',native)
    contracts.test_independent_neighborhood_decision_replay(name,ndim,dtype,kind,output)


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('values',((0,1,2,3,32767,32768,65533,65534,65535),(0,0,1,1,4,4,8,8,65535),(1,2,4,8,16,32,64,128,65535),(65535,)*9))
@pytest.mark.parametrize('dtype',('u1','u2','f4','f8','bool'))
def test_full_capacity_occupied_batches_raw_output(name,values,dtype):
    image=np.array([values],np.uint16);fp=np.ones((1,9),np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected=getattr(reference,name)(image,fp,out=np.full(image.shape,99,np.dtype(dtype)))
        actual=getattr(native,name)(image,fp,out=np.full(image.shape,99,np.dtype(dtype)))
    assert actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('ndim',(2,3))
@pytest.mark.parametrize('dtype',(np.uint8,np.uint16))
def test_large_disjoint_global_worker_identity(name,ndim,dtype):
    shape=(513,519) if ndim==2 else (33,31,257)
    image=((np.arange(np.prod(shape))*37)%61).astype(dtype).reshape(shape)
    fp=np.ones((3,)*ndim,np.uint8);fp.flat[::4]=0
    mask=(np.arange(image.size).reshape(shape)%4!=0).astype(np.uint8)
    expected=getattr(reference,name)(image,fp,mask=mask,out=np.empty(shape,np.float64))
    first=getattr(native,name)(image,fp,mask=mask,out=np.empty(shape,np.float64))
    assert first.tobytes()==expected.tobytes()
    if os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')!='1':
        previous=core.get_num_threads();core.set_num_threads(4 if previous==1 else 1)
        try:assert getattr(native,name)(image,fp,mask=mask,out=np.empty(shape,np.float64)).tobytes()==first.tobytes()
        finally:core.set_num_threads(previous)


@pytest.mark.parametrize('operation',(29,30))
@pytest.mark.parametrize('unsafe',('output-short','zero-depth','mask-short','image-alias','footprint-alias','escaping-bin'))
def test_unsupported_native_buffers_prewrite_without_original(operation,unsafe):
    # No bounds-disabled original is called with these unsafe/uncertified views.
    image=np.arange(12,dtype=np.uint8).reshape(3,4);fp=np.ones((3,3),np.uint8);mask=None
    out=np.full(image.shape+(1,),99,np.uint8);bins=256
    if unsafe=='output-short':out=np.full((1,4,1),99,np.uint8)
    elif unsafe=='zero-depth':out=np.zeros(image.shape+(0,),np.uint8)
    elif unsafe=='mask-short':mask=np.ones((1,4),np.uint8)
    elif unsafe=='image-alias':out=image.reshape(image.shape+(1,))
    elif unsafe=='footprint-alias':fp=np.ones(image.shape,np.uint8);out=fp.reshape(image.shape+(1,))
    elif unsafe=='escaping-bin':bins=4
    before=out.copy()
    with pytest.raises(ValueError):core.rank_decisions_run(image,fp,mask,out,[0,0],bins,operation)
    assert out.tobytes()==before.tobytes()


@pytest.mark.parametrize('name',contracts.NAMES)
def test_no_original_rank_forwarding(name,monkeypatch):
    image=np.arange(35,dtype=np.uint8).reshape(5,7);fp=np.ones((3,3),np.uint8)
    expected=getattr(reference,name)(image,fp)
    def forbidden(*a,**k):raise AssertionError('original decision forwarding')
    generic=importlib.import_module('skimage.filters.rank.generic')
    for target in (reference,generic):
        for operation in contracts.NAMES:monkeypatch.setattr(target,operation,forbidden)
    assert getattr(native,name)(image,fp).tobytes()==expected.tobytes()


def shift_mutation_observe(function,ndim,action,field):
    shape=(3,4) if ndim==2 else (2,3,4)
    image=(np.arange(np.prod(shape))%4).astype(np.uint16).reshape(shape)
    fp=np.ones((3,)*ndim,np.uint8);mask=np.ones(shape,np.uint8);out=np.full(shape,99,np.float64);events=[]
    class Shift:
        def __int__(self):
            events.append(field+'.__int__')
            if action=='image-fill':image.fill(0)
            elif action=='mask-fill':mask.fill(0)
            elif action=='footprint-fill':fp.fill(0)
            elif action=='image-dtype':image.dtype=np.dtype('i1')
            elif action=='out-dtype':out.dtype=np.dtype('i1')
            return 0
    options=dict(out=out,mask=mask);options[field]=Shift()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result=function(image,fp,**options);error=None
        except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
    return dict(events=events,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],result=None if result is None else contracts.array(result),after={key:contracts.array(value) for key,value in dict(image=image,fp=fp,mask=mask,out=out).items()})


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('ndim',(2,3))
@pytest.mark.parametrize('action',('image-fill','mask-fill','footprint-fill','image-dtype','out-dtype'))
@pytest.mark.parametrize('field',('shift_x','shift_y'))
def test_acquired_buffers_precede_defined_shift_mutations(name,ndim,action,field):
    # Same-allocation dtype reinterpretations only; no resize, overlap or
    # histogram escape is introduced into the bounds-disabled reference.
    expected=shift_mutation_observe(getattr(reference,name),ndim,action,field)
    assert expected['error'] is None and expected['events']==[field+'.__int__']
    assert shift_mutation_observe(getattr(native,name),ndim,action,field)==expected


SHIFT_PROTOCOLS=('index-only','int-only','float-only','both','int-subclass','float-subclass','trunc-only','int','float','nan','inf','none','numeric-string','np-i64','np-f64','array0','array1','np-bool')


def shift_protocol_observe(function,token):
    events=[]
    class IndexOnly:
        def __index__(self):events.append('__index__');return 0
    class IntOnly:
        def __int__(self):events.append('__int__');return 0
    class FloatOnly:
        def __float__(self):events.append('__float__');return 0.
    class Both:
        def __int__(self):events.append('__int__');return 0
        def __index__(self):events.append('__index__');raise ValueError('unexpected-index')
    class Integer(int):
        def __int__(self):events.append('subclass.__int__');return 1
        def __index__(self):events.append('subclass.__index__');return 1
    class Floating(float):
        def __int__(self):events.append('subclass.__int__');return 0
    class TruncOnly:
        def __trunc__(self):events.append('__trunc__');return 0
    classes={'index-only':IndexOnly,'int-only':IntOnly,'float-only':FloatOnly,'both':Both,'int-subclass':lambda:Integer(0),'float-subclass':lambda:Floating(.5),'trunc-only':TruncOnly}
    value=classes[token]() if token in classes else {'int':0,'float':.5,'nan':float('nan'),'inf':float('inf'),'none':None,'numeric-string':'0','np-i64':np.int64(0),'np-f64':np.float64(.5),'array0':np.array(0),'array1':np.array([0]),'np-bool':np.bool_(False)}[token]
    image=np.arange(6,dtype=np.uint8).reshape(2,3);fp=np.zeros((3,3),np.uint8)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result=function(image,fp,shift_x=value);error=None
        except Exception as caught:result=None;error=dict(type=type(caught).__name__,message=str(caught))
    return dict(events=events,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],result=None if result is None else contracts.array(result))


@pytest.mark.parametrize('name',contracts.NAMES)
@pytest.mark.parametrize('token',SHIFT_PROTOCOLS)
def test_source_specific_signed_char_protocol(name,token):
    # Zero footprint and bounded geometry make every accepted provider safe.
    # Exact unsupported error/hook behavior is retained, never normalized away.
    assert shift_protocol_observe(getattr(native,name),token)==shift_protocol_observe(getattr(reference,name),token)


OLD_NAMES=shell._GENERIC_FUNCTIONS+shell._PERCENTILE_FUNCTIONS+('mean_bilateral','pop_bilateral','sum_bilateral')


@pytest.mark.parametrize('operation',range(29))
@pytest.mark.parametrize('empty_mask',(False,True))
def test_previous_operation_families_retain_prepared_results(operation,empty_mask):
    # Compact fresh numerical routing guard, supplementing inherited complete
    # old-family proofs. Source-safe ordinary values/disjoint buffers only.
    name=OLD_NAMES[operation];image=(np.arange(12)%7).astype(np.uint8).reshape(3,4)
    fp=np.ones((3,3),np.uint8);mask=np.full(image.shape,not empty_mask,np.uint8)
    shape=image.shape+(7,) if operation==16 else image.shape
    options=dict(mask=mask,out=np.full(shape,99,np.float64))
    if operation==16:options['n_bins']=7
    if 17<=operation<=25:
        options['p0']=.25
        if operation not in (23,25):options['p1']=.75
    expected=getattr(reference,name)(image,fp,**options).copy()
    options['out']=np.full(shape,99,np.float64)
    if operation<26:actual=getattr(shell,name)(image,fp,**options)
    else:
        # Only a prepared-binding adapter for already-owned R6; no prototype
        # public wrapper, upstream preprocessing or forwarding is counted.
        core.rank_bilateral_run(image,fp,mask,options['out'].reshape(image.shape+(1,)),[0,0],256,operation,10,10)
        actual=options['out']
    assert actual.dtype==expected.dtype and actual.shape==expected.shape and actual.tobytes()==expected.tobytes()


def test_separate_interface_old_routes_and_default_policy():
    assert INITIAL_WORKERS==1
    assert (core.rank_histogram_capabilities,core.rank_alias_capabilities,core.rank_statistics_capabilities,core.rank_vector_capabilities,core.rank_percentile_capabilities,core.rank_bilateral_capabilities)==(8191,480,0xe000,0x10000,0x3fe0000,0x1c000000)
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':assert core.gil_assertions is True
    a=np.ones((2,3),np.uint8);f=np.ones((1,1),np.uint8);out=np.full((2,3,1),99,np.uint8)
    for operation in (-1,0,28,31,32):
        with pytest.raises(ValueError,match='invalid rank decisions operation'):core.rank_decisions_run(a,f,None,out,[0,0],256,operation)
    for operation in (29,30):
        with pytest.raises(ValueError,match='invalid rank operation'):core.rank_run(a,f,None,out,[0,0],256,operation)
        with pytest.raises(ValueError,match='invalid rank bilateral operation'):core.rank_bilateral_run(a,f,None,out,[0,0],256,operation,10,10)
    assert np.all(out==99)
