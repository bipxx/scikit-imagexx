"""Actual public R4 contract. No alternate backend or namespace facade."""
import importlib.util
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
import skimagexx as xx
from skimagexx import _rank
from skimagexx.filters import rank
from skimagexx.filters.rank import generic
from skimage.filters import rank as reference
from skimage.filters.rank import generic as reference_generic
from skimagexx.coverage import native_functions

spec=importlib.util.spec_from_file_location('rank_statistics_public_contracts',Path(__file__).with_name('test_rank_statistics_contracts.py'))
contracts=importlib.util.module_from_spec(spec);spec.loader.exec_module(contracts)
MATRIX=json.loads(contracts.MATRIX.read_text())
NAMES=contracts.NAMES


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    count=xx.get_num_threads();previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield request.param
    xx.set_num_threads(count);xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('name',NAMES)
def test_actual_public_identity_signature_and_separate_interfaces(name):
    assert _rank._core is xx._core
    assert xx.set_num_threads is xx._core.set_num_threads and xx.get_num_threads is xx._core.get_num_threads
    assert xx._core.rank_histogram_capabilities==8191
    assert xx._core.rank_alias_interface=='rank-alias-v1' and xx._core.rank_alias_capabilities==480
    assert xx._core.rank_statistics_interface=='rank-statistics-v1' and xx._core.rank_statistics_capabilities==0xe000
    assert xx._core.rank_vector_interface=='rank-vector-v1' and xx._core.rank_vector_capabilities==0x10000
    assert getattr(rank,name) is getattr(generic,name) is getattr(_rank,name)
    assert getattr(rank,name) is not getattr(reference,name)
    assert inspect.signature(getattr(rank,name))==inspect.signature(getattr(reference,name))
    assert getattr(reference,name) is getattr(reference_generic,name)
    assert name in rank.__all__ and (name in generic.__all__)==(name!='windowed_histogram')
    for namespace in ('filters.rank','filters.rank.generic'):assert namespace+'.'+name in native_functions()


@pytest.mark.parametrize('namespace',[rank,generic],ids=['canonical','generic'])
@pytest.mark.parametrize('case',contracts.CASES,ids=[row[0] for row in contracts.CASES])
def test_public_frozen_values_errors_warnings_layout_and_owners(namespace,case,monkeypatch):
    key,name,parameters=case
    expected=MATRIX['observations'][MATRIX['snapshots'][contracts.profile()][key]]
    def forbidden(*a,**k):raise AssertionError('upstream rank forwarding')
    monkeypatch.setattr(_rank,'fallback',forbidden)
    for operation in NAMES:monkeypatch.setattr(reference,operation,forbidden)
    assert contracts.observe(getattr(namespace,name),name,parameters)==expected,key


@pytest.mark.parametrize('name',contracts.SCALAR)
@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('dtype',['u1','u2'])
@pytest.mark.parametrize('kind',['dense','even','disconnected','zero','masked','shift'])
@pytest.mark.parametrize('out_type',['same','f4','f8'])
def test_public_independent_scalar_definition(name,ndim,dtype,kind,out_type,monkeypatch):
    # Select the actual public target for the unchanged independent math oracle.
    monkeypatch.setattr(contracts,'reference',rank)
    contracts.test_independent_arithmetic_and_ordered_replay(name,ndim,dtype,kind,out_type)


@pytest.mark.parametrize('dtype',['u1','u2'])
@pytest.mark.parametrize('kind',['dense','even','zero','masked'])
@pytest.mark.parametrize('depth',[4,32])
@pytest.mark.parametrize('out_type',['u1','u2','f4','f8'])
def test_public_independent_vector_definition(dtype,kind,depth,out_type,monkeypatch):
    monkeypatch.setattr(contracts,'reference',rank)
    contracts.test_histogram_independent_counts_and_exact_reciprocal_replay(dtype,kind,depth,out_type)


def test_public_reciprocal_rounding_and_singleton_capacity(monkeypatch):
    monkeypatch.setattr(contracts,'reference',rank)
    contracts.test_histogram_integer_output_exposes_reciprocal_rounding()
    contracts.test_source_safe_singleton_full_uint16_histogram()


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('mode',['missing-interface','wrong-interface','missing-bit','old-mask','missing-entry','non-callable-entry'])
def test_legacy_mask_cannot_admit_statistics_or_vector(name,mode,monkeypatch):
    vector=name=='windowed_histogram';operation=16 if vector else 13+contracts.SCALAR.index(name)
    interface='rank_vector_interface' if vector else 'rank_statistics_interface'
    capabilities='rank_vector_capabilities' if vector else 'rank_statistics_capabilities'
    entry='rank_windowed_histogram_run' if vector else 'rank_statistics_run'
    if mode=='missing-interface':monkeypatch.delattr(xx._core,interface)
    elif mode=='wrong-interface':monkeypatch.setattr(xx._core,interface,'unknown-version')
    elif mode=='missing-bit':monkeypatch.setattr(xx._core,capabilities,getattr(xx._core,capabilities)&~(1<<operation))
    elif mode=='old-mask':monkeypatch.setattr(xx._core,capabilities,8191)
    elif mode=='missing-entry':monkeypatch.delattr(xx._core,entry)
    else:monkeypatch.setattr(xx._core,entry,0)
    class Untouched:
        def __getattribute__(self,name):raise AssertionError('input inspected before backend guard')
    with pytest.raises(NotImplementedError,match='backend required'):getattr(rank,name)(Untouched(),Untouched())


def test_noise_metadata_entry_is_required_before_inputs(monkeypatch):
    monkeypatch.delattr(xx._core,'rank_noise_footprint')
    class Untouched:
        def __getattribute__(self,name):raise AssertionError('input inspected before metadata backend guard')
    with pytest.raises(NotImplementedError,match='backend required'):rank.noise_filter(Untouched(),Untouched())


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('kind',['image','mask'])
def test_overlap_is_explicit_prewrite_rejection_without_forwarding(name,kind,monkeypatch):
    image=np.arange(35,dtype=np.uint8).reshape(5,7);before=image.copy()
    mask=np.ones(image.shape,np.uint8) if kind=='mask' else None
    owner=image if mask is None else mask;owner_before=owner.copy()
    output=owner.reshape(5,7,1) if name=='windowed_histogram' else owner.view()
    def forbidden(*a,**k):raise AssertionError('unsafe overlap forwarding')
    monkeypatch.setattr(_rank,'fallback',forbidden);monkeypatch.setattr(reference,name,forbidden)
    with pytest.raises(NotImplementedError,match='Outside qualified'):
        getattr(rank,name)(image,np.ones((3,3),np.uint8),out=output,mask=mask)
    np.testing.assert_array_equal(image,before)
    np.testing.assert_array_equal(owner,owner_before)


@pytest.mark.parametrize('depth',[0,257])
def test_vector_unsafe_depth_rejects_before_stores(depth):
    image=np.arange(35,dtype=np.uint8).reshape(5,7)
    output=np.full(image.shape+(depth,),99,np.float64);before=output.copy()
    with pytest.raises(NotImplementedError,match='Outside qualified'):
        rank.windowed_histogram(image,np.ones((3,3),np.uint8),out=output)
    np.testing.assert_array_equal(output,before)


def test_subtract_undefined_unsigned_conversion_is_not_clamped():
    image=np.array([[0,70]],np.uint16);fp=np.array([[0,0,1]],np.uint8);out=np.full(image.shape,99,np.uint16)
    with pytest.raises(NotImplementedError,match='unsigned conversion outside defined coverage'):
        rank.subtract_mean(image,fp,out=out)
    np.testing.assert_array_equal(out,np.full_like(out,99))


def noise_hook_observation(function,ndim,failure):
    events=[];state={'active':True,'reads':0}
    class Footprint(np.ndarray):
        @property
        def shape(self):
            if state['active']:
                state['reads']+=1;events.append('shape')
                if state['reads']==failure:raise RuntimeError('shape hook stopped')
            return np.ndarray.shape.__get__(self)
        def copy(self,*a,**k):
            if state['active']:events.append('copy')
            return super().copy(*a,**k)
        def __setitem__(self,key,value):
            if state['active']:events.append(('clear',repr(key)));state['active']=False
            return super().__setitem__(key,value)
    fp=np.ones((3,)*ndim,np.uint8).view(Footprint);image=np.arange(35 if ndim==2 else 60,dtype=np.uint8).reshape((5,7) if ndim==2 else (3,4,5))
    before=fp.tobytes()
    try:result=function(image,fp);error=None
    except Exception as exc:result=None;error=(type(exc).__name__,str(exc))
    return events,error,None if result is None else result.tobytes(),fp.tobytes()==before


@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('failure',[None,1,2])
def test_noise_shape_copy_clear_hooks_match_original(ndim,failure):
    assert noise_hook_observation(rank.noise_filter,ndim,failure)==noise_hook_observation(reference.noise_filter,ndim,failure)


def noise_mutating_shift_observation(function):
    image=np.arange(35,dtype=np.uint8).reshape(5,7);fp=np.ones((3,3),np.uint8);events=[]
    class Shift:
        def __radd__(self,value):events.append(('add',value));fp.shape=(1,9);return 0
        def __int__(self):events.append('int');return 0
        def __index__(self):events.append('index');return 0
    try:result=function(image,fp,shift_y=Shift());error=None
    except Exception as exc:result=None;error=(type(exc).__name__,str(exc))
    return events,error,None if result is None else result.tobytes(),fp.shape,fp.tobytes()


def test_noise_shift_callback_changes_later_shape_read(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('protocol-bearing shift admitted to cached-shape helper')
    monkeypatch.setattr(xx._core,'rank_noise_footprint',forbidden)
    assert noise_mutating_shift_observation(rank.noise_filter)==noise_mutating_shift_observation(reference.noise_filter)


def histogram_max_observation(function,mode):
    events=[]
    class Image(np.ndarray):
        def max(self,*a,**k):
            events.append('raw-max')
            if mode=='raise':raise RuntimeError('raw max stopped before footprint')
            value=super().max(*a,**k);self.flat[0]=7;return value
    image=np.arange(35,dtype=np.uint8).reshape(5,7).view(Image)
    supplied=np.full(image.shape+(4,),99,np.float64) if mode in ('out','explicit') else None
    fp=None if mode=='raise' else np.ones((3,3),np.uint8)
    try:
        result=function(image,fp,out=supplied,n_bins='ignored' if mode=='explicit' else None)
        observed=(result.dtype.str,result.shape,result.strides,result.tobytes(),result is supplied);error=None
    except Exception as exc:observed=None;error=(type(exc).__name__,str(exc))
    return events,error,observed,image.tobytes(),None if supplied is None else supplied.tobytes()


@pytest.mark.parametrize('mode',['default','out','explicit','raise'])
def test_histogram_raw_max_callback_and_actual_output_depth(mode):
    assert histogram_max_observation(rank.windowed_histogram,mode)==histogram_max_observation(reference.windowed_histogram,mode)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
def test_public_large_masked_pipeline(name,dtype,monkeypatch):
    image=((np.arange(513*519)*37)%71).astype(dtype).reshape(513,519)
    fp=np.ones((3,5),np.uint8);fp.flat[::4]=0
    mask=(np.arange(image.size).reshape(image.shape)%4!=0).astype(np.uint8)
    options=dict(mask=mask)
    if name=='subtract_mean':options['out']=np.empty(image.shape,np.float64)
    if name=='windowed_histogram':options['n_bins']=7
    expected=getattr(reference,name)(image,fp,**options).copy()
    def forbidden(*a,**k):raise AssertionError('upstream pipeline forwarding')
    monkeypatch.setattr(reference,name,forbidden);monkeypatch.setattr(_rank,'fallback',forbidden)
    actual=getattr(rank,name)(image,fp,**options)
    np.testing.assert_array_equal(actual,expected)
