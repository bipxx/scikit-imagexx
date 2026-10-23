"""Pinned consumer and returned-ArrayMap contracts; native cases stay pending.

Only bounded label values are used: no enormous bincount/dense map allocation,
unsafe striding, or upstream out-of-bounds map_array input is exercised.
"""
import gc
import inspect
import json
from pathlib import Path
import warnings
import weakref
import numpy as np
import pytest
from skimage import morphology,segmentation
import skimagexx as xx
from skimagexx.coverage import native_functions

APIS=('morphology.remove_small_objects','morphology.remove_small_holes',
      'segmentation.clear_border','segmentation.relabel_sequential')
TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,
       np.uint64,np.int64,np.float16,np.float32,np.float64)
MATRIX=Path(__file__).parents[1]/'coverage/contracts/component-consumers.json'


def reference(api):
    namespace,name=api.split('.')
    return getattr(morphology if namespace=='morphology' else segmentation,name)


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    assert xx.get_num_threads()==request.param
    yield
    xx.set_num_threads(1);xx.set_fallback_enabled(previous)


@pytest.fixture(params=APIS)
def api(request):return request.param


def native_or_reference(api,kind):
    if kind=='reference':return reference(api)
    namespace,name=api.split('.');candidate=getattr(getattr(xx,namespace),name)
    if api not in native_functions():
        image=np.array([0,1,2],np.int16);before=image.copy()
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            with pytest.raises(NotImplementedError,match='fallback is disabled'):candidate(image)
        assert not notices;np.testing.assert_array_equal(image,before)
        pytest.xfail(f'native {api} consumer implementation pending')
    assert candidate is not reference(api),'forwarded original is not native implementation'
    return candidate


@pytest.fixture(params=['reference','native'])
def function(request,api):return native_or_reference(api,request.param)


@pytest.fixture(params=['reference','native'])
def relabel_function(request):return native_or_reference('segmentation.relabel_sequential',request.param)


def state(value):
    if isinstance(value,np.ndarray):
        return (type(value).__name__,value.dtype.str,repr(value.dtype.metadata),value.shape,value.strides,
                value.flags.c_contiguous,value.flags.f_contiguous,value.flags.owndata,value.flags.writeable,
                value.flags.aligned,value.tobytes(order='C'))
    return (type(value).__name__,repr(value))


def method_observation(call):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:value,error=call(),None
        except Exception as caught:value,error=None,caught
    result=('error',type(error).__name__,str(error)) if error is not None else ('returned',state(value))
    return result,[(type(w.message).__name__,str(w.message)) for w in notices]


def describe(value,source,options):
    if isinstance(value,np.ndarray):
        arrays=[source]+[v for v in options.values() if isinstance(v,np.ndarray)]
        return ('array',state(value),[value is a for a in arrays],[np.shares_memory(value,a) for a in arrays])
    if isinstance(value,tuple):return ('tuple',tuple(describe(v,source,options) for v in value))
    if type(value).__name__=='ArrayMap':
        return ('ArrayMap',state(value.in_values),state(value.out_values),value.dtype.str,
                method_observation(lambda:len(value)),repr(value),str(value),value._max_str_lines,value._array is None)
    return state(value)


def observe(function,factory):
    image,args,options=factory();arrays=[image]+[v for v in options.values() if isinstance(v,np.ndarray)]
    before=[state(a) for a in arrays]
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result,error=function(image,*args,**options),None
        except Exception as caught:result,error=None,caught
    after=[state(a) for a in arrays]
    return result,error,[(type(w.message),str(w.message)) for w in notices],before,after,image,options


def check(function,api,factory):
    expected,error,notices,before,after,image,options=observe(reference(api),factory)
    actual,actual_error,actual_notices,actual_before,actual_after,source,opts=observe(function,factory)
    assert actual_notices==notices and actual_before==before and actual_after==after
    if error is not None:
        assert type(actual_error) is type(error) and str(actual_error)==str(error)
        return None
    if actual_error is not None:raise actual_error
    assert describe(actual,source,opts)==describe(expected,image,options)
    if api.endswith('relabel_sequential') and function is not reference(api):
        assert not type(actual[1]).__module__.startswith('skimage.'),'returned ArrayMap class port is pending'
    return actual


def image_for(dtype,shape=(5,7),layout='C'):
    image=np.resize(np.array([0,1,1,0,2,2,2,3,0,3,4,4],np.int16),shape).astype(dtype)
    if layout=='F':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[tuple(slice(None,None,-1) for _ in shape)]
    elif layout=='step':
        image=np.resize(image,tuple(2*n for n in shape))[tuple(slice(None,None,2) for _ in shape)]
    elif layout=='endian':image=image.astype(image.dtype.newbyteorder('>'))
    elif layout=='broadcast':image=np.broadcast_to(image[:1],image.shape)
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='unaligned':
        original=image;image=np.ndarray(shape,image.dtype,buffer=bytearray(image.nbytes+1),offset=1);image[:]=original
    elif layout=='tagged':image=image.view(np.dtype(image.dtype,metadata={'unit':'labels'}))
    return image


def test_signatures_and_explicit_inventory(function,api):
    record=json.loads(MATRIX.read_text())
    assert str(inspect.signature(function))==record['apis'][api]['signature']
    assert record['returned_classes']['ArrayMap']['native_implemented'] is False


@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('layout',['C','F','reverse','step','endian','broadcast','readonly','unaligned','tagged'])
def test_real_dtype_and_layout_contract(function,api,dtype,layout):
    check(function,api,lambda:(image_for(dtype,layout=layout),(),{}))


@pytest.mark.parametrize('dtype',[np.bool_,np.int16,np.float32])
@pytest.mark.parametrize('shape',[(),(0,),(0,3),(2,0,3),(1,),(7,),(2,3,4),(2,1,3),(2,2,2,2)])
def test_rank_empty_scalar_and_singleton(function,api,dtype,shape):
    check(function,api,lambda:(image_for(dtype,shape),(),{}))


@pytest.mark.parametrize('case',range(12))
def test_threshold_offset_connectivity_and_validation_order(function,api,case):
    def factory():
        image=image_for(np.int16)
        if api.startswith('morphology.'):
            old='min_size' if api.endswith('objects') else 'area_threshold'
            options=[dict(max_size=0),dict(max_size=-1),dict(max_size=2),dict(max_size=2.5),
                     {old:2},{old:2,'max_size':2},dict(connectivity=0),dict(connectivity=9),
                     dict(connectivity=1.5),dict(max_size=None),dict(max_size='bad'),dict(max_size=0)][case]
            if case in [6,7,8]:image=image.astype(bool)
            if case in [5,11]:image=np.array([-1,0,2],np.int16)
        elif api.endswith('clear_border'):
            options=[dict(buffer_size=0),dict(buffer_size=1),dict(buffer_size=-1),dict(buffer_size=5),
                     dict(buffer_size=.5),dict(buffer_size='bad'),dict(bgval=7),dict(bgval=.5),
                     dict(mask=np.ones(image.shape,bool)),dict(mask=np.ones((2,3),bool)),
                     dict(mask=np.ones(image.shape,np.uint8)),dict(buffer_size=999,mask=np.ones(image.shape,bool))][case]
        else:
            options=dict(offset=[0,-1,1,2,127,128,255,256,.5,1.5,None,'1'][case])
            if case in [0,1]:image=np.array([-1,2],np.int16)
        return image,(),options
    check(function,api,factory)


@pytest.mark.parametrize('out_kind',['self','copy','F','strided','bool','float','different-shape','readonly'])
@pytest.mark.parametrize('api',APIS[:3])
def test_out_identity_casting_partial_mutation_and_errors(function,api,out_kind):
    def factory():
        image=image_for(bool if api.endswith('holes') else np.int16)
        if out_kind=='self':out=image
        elif out_kind=='copy':out=np.full(image.shape,9,image.dtype)
        elif out_kind=='F':out=np.asfortranarray(np.full(image.shape,9,image.dtype))
        elif out_kind=='strided':out=np.full((10,14),9,image.dtype)[::2,::2]
        elif out_kind=='bool':out=np.ones(image.shape,bool)
        elif out_kind=='float':out=np.full(image.shape,9,np.float32)
        elif out_kind=='different-shape':out=np.full((3,4),9,image.dtype)
        else:out=image.copy();out.flags.writeable=False
        return image,(),dict(out=out)
    check(function,api,factory)


@pytest.mark.parametrize('case',range(8))
def test_negative_nonfinite_and_joint_error_precedence(function,api,case):
    def factory():
        images=[np.array([-1,0,2],np.int16),np.array([0,np.nan,np.inf],np.float64),
                np.array(np.nan),np.array([],np.float32),np.array([0,1],np.float32),
                np.array([0,1],np.int16),np.array([0,1],np.int16),np.array([0,1],np.int16)]
        image=images[case]
        if api.startswith('morphology.'):
            old='min_size' if api.endswith('objects') else 'area_threshold'
            options=[{},dict(connectivity='bad'),dict(max_size=0),{},
                     {old:2,'max_size':2},{old:2},dict(out=np.zeros(2,np.float32)),dict(max_size=0)][case]
        elif api.endswith('clear_border'):
            options=[{},dict(buffer_size='bad'),dict(mask=np.ones((),np.uint8)),{},
                     dict(buffer_size=999,mask=np.ones((1,),np.uint8)),dict(mask=np.ones((2,),np.uint8)),
                     dict(mask=np.ones((1,),bool)),dict(buffer_size=-1)][case]
        else:options=[{},dict(offset=0),dict(offset=1),dict(offset=0),dict(offset=.5),dict(offset=np.nan),dict(offset=np.inf),dict(offset=True)][case]
        return image,(),options
    check(function,api,factory)


@pytest.mark.parametrize('api',APIS[:2])
def test_threshold_public_deprecation_is_inclusive(function,api):
    old='min_size' if api.endswith('objects') else 'area_threshold'
    image=(np.array([0,1,1,0,2,2,2],np.int16) if api.endswith('objects') else
           np.array([1,0,0,1,0,0,0,1],bool))
    wanted=([0,0,0,0,2,2,2] if api.endswith('objects') else [True,True,True,True,False,False,False,True])
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');actual=function(image,**{old:2})
    assert [(type(w.message).__name__,str(w.message)) for w in notices]==[
        ('FutureWarning',json.loads(MATRIX.read_text())['apis'][api]['deprecation_warning'])]
    np.testing.assert_array_equal(actual,wanted)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');modern=function(image,max_size=2)
    np.testing.assert_array_equal(modern,actual)


@pytest.mark.parametrize('api',[APIS[2]])
def test_clear_border_uses_existing_out_contents(function,api):
    image=np.zeros((5,5),np.int16);out=np.zeros_like(image);out[2,2]=2;out[0,0]=2
    actual=function(image,out=out)
    assert actual is out and actual[2,2]==2 and actual[0,0]==0
    assert not image.any()


def test_positional_parameters(function,api):
    args=(2,2) if api.startswith('morphology.') else (1,7) if api.endswith('clear_border') else (2,)
    check(function,api,lambda:(image_for(np.int16),args,{}))


@pytest.mark.parametrize('api',APIS[:3])
@pytest.mark.parametrize('zero',[False,True])
def test_scalar_explicit_out_partial_mutation(function,api,zero):
    def factory():
        image=np.array(True)
        options=dict(out=image)
        if api.startswith('morphology.') and zero:options['max_size']=0
        return image,(),options
    check(function,api,factory)


@pytest.mark.parametrize('dtype',[np.bool_,np.int16])
@pytest.mark.parametrize('pattern',['stripes','checkerboard'])
def test_large_inputs_cross_worker_threshold(function,api,dtype,pattern):
    def factory():
        values=np.indices((513,513)).sum(axis=0)%2 if pattern=='checkerboard' else np.broadcast_to(np.arange(513)[:,None]%3==1,(513,513))
        return values.astype(dtype),(),{}
    check(function,api,factory)


def test_relabel_rejects_unsupported_out_keyword(relabel_function):
    check(relabel_function,'segmentation.relabel_sequential',
          lambda:(image_for(np.int16),(),dict(out=np.zeros((5,7),np.int16))))


@pytest.mark.parametrize('api',APIS[:2])
def test_binary_and_integer_label_meaning(function,api):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');integer=function(np.array([1,0,1],np.int16),max_size=1)
    boolean=function(np.array([True,False,True]),max_size=1)
    assert len(notices)==1 and type(notices[0].message) is UserWarning
    if api.endswith('objects'):
        np.testing.assert_array_equal(integer,[1,0,1]);assert not boolean.any()
    else:
        assert integer.dtype==bool and integer.all() and boolean.all()


def test_relabel_sorted_values_and_map_aliases(relabel_function):
    image=np.array([0,8,8,3,11,3],np.int16)
    labels,forward,inverse=relabel_function(image,offset=2)
    np.testing.assert_array_equal(labels,[0,3,3,2,4,2])
    assert forward.in_values is inverse.out_values and forward.out_values is inverse.in_values
    assert forward._array is None and inverse._array is None
    assert forward._max_str_lines==inverse._max_str_lines==4
    assert forward.dtype==labels.dtype and inverse.dtype==image.dtype
    np.testing.assert_array_equal(forward(image),labels)
    np.testing.assert_array_equal(inverse[labels],image)
    assert len(forward)==12 and len(inverse)==5
    assert not np.shares_memory(labels,image)
    assert not np.shares_memory(forward.in_values,image)


@pytest.mark.parametrize('access',['getitem','call'])
@pytest.mark.parametrize('index',[0,3,11,99,-1,np.array([0,3,11,99,-1]),np.array([[0,8],[11,3]]),
                                  np.array([False,True,False,True]),slice(None),slice(0,12,2),[1,2],1.5,np.array([1.5]),slice(None,None,0)])
def test_returned_arraymap_indexing_and_errors(relabel_function,index,access):
    _,expected,_=reference('segmentation.relabel_sequential')(np.array([0,3,8,11],np.int16),offset=2)
    _,actual,_=relabel_function(np.array([0,3,8,11],np.int16),offset=2)
    def observe_index(mapping):
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:result,error=(mapping[index] if access=='getitem' else mapping(index)),None
            except Exception as caught:result,error=None,caught
        return result,error,[(type(w.message),str(w.message)) for w in notices]
    want,error,notices=observe_index(expected);result,actual_error,actual_notices=observe_index(actual)
    assert actual_notices==notices
    if error is not None:assert type(actual_error) is type(error) and str(actual_error)==str(error)
    else:
        assert type(result) is type(want)
        if isinstance(result,np.ndarray):assert state(result)==state(want)
        else:assert result==want and result.dtype==want.dtype
    assert actual._array is None


def test_returned_arraymap_materialization_is_fresh(relabel_function):
    _,forward,_=relabel_function(np.array([0,3,8,11],np.int16),offset=2)
    first=np.asarray(forward);second=np.asarray(forward)
    assert first is not second and not np.shares_memory(first,second)
    assert first.flags.owndata and first.dtype==forward.dtype and forward._array is None
    np.testing.assert_array_equal(first,[0,0,0,2,0,0,0,0,3,0,0,4])
    converted=forward.__array__(dtype=np.float32,copy=False)
    assert converted.dtype==np.float32 and converted.flags.owndata
    first[3]=99;assert forward[3]==2


def test_returned_arraymap_mutation_cache_and_lifetime(relabel_function):
    image=np.array([0,3,8,11],np.int16);image_ref=weakref.ref(image)
    labels,forward,inverse=relabel_function(image,offset=2)
    original_inverse_keys=inverse.in_values
    forward[3]=7
    assert forward._array is not None and forward[3]==7 and inverse[2]==3
    assert inverse.in_values is original_inverse_keys
    assert forward.out_values is not inverse.in_values
    assert 0 not in forward.in_values
    forward[3]=0;assert forward[3]==0 and 3 not in forward.in_values
    del image,labels;gc.collect();assert image_ref() is None
    assert inverse[4]==11


def test_returned_arraymap_repr_state_and_public_attributes(relabel_function):
    _,forward,inverse=relabel_function(np.arange(9,dtype=np.int16))
    _,expected,_=reference('segmentation.relabel_sequential')(np.arange(9,dtype=np.int16))
    assert type(forward.__len__()) is type(expected.__len__()) and forward.__len__()==expected.__len__()
    assert forward.dtype is forward.out_values.dtype
    assert set(vars(forward))=={'in_values','out_values','_max_str_lines','_array'}
    assert repr(forward)==f'ArrayMap({forward.in_values!r}, {forward.out_values!r})'
    assert str(forward).startswith('ArrayMap:\n') and '\n  ...\n' in str(forward)
    forward._max_str_lines=20;assert '\n  ...\n' not in str(forward)
    # Returned vectors initially alias their inverse partners; mutation is visible.
    forward.out_values[3]=7;assert inverse.in_values[3]==7 and forward[3]==7


def test_returned_arraymap_constructor_and_failed_mutation_state(relabel_function):
    _,forward,inverse=relabel_function(np.array([0,3,8,11],np.int16),offset=2)
    keys=np.array([1,4],np.int16);values=np.array([2.5,-1.],np.float32)
    mapping=type(forward)(keys,values)
    assert mapping.in_values is keys and mapping.out_values is values
    assert mapping.dtype==np.float32 and len(mapping)==5 and mapping._array is None
    np.testing.assert_array_equal(mapping[np.array([0,1,4,9],np.int16)],[0.,2.5,-1.,0.])
    original_keys=forward.in_values;original_values=forward.out_values
    with pytest.raises(IndexError):forward[99]=1
    assert forward._array is not None
    assert forward.in_values is original_keys and forward.out_values is original_values
    assert inverse.in_values is original_values


def test_returned_arraymap_unsigned_runtime_size_quirks(relabel_function):
    image=np.array([0,3,8],np.uint64)
    _,forward,_=relabel_function(image)
    _,expected,_=reference('segmentation.relabel_sequential')(image)
    for method in [len,lambda mapping:mapping.__len__(),lambda mapping:np.asarray(mapping)]:
        assert method_observation(lambda:method(forward))==method_observation(lambda:method(expected))
    # Functional lookup still works even when NumPy1 uint64 length/dense fail.
    np.testing.assert_array_equal(forward[np.array([0,3,8],np.int64)],[0,1,2])


@pytest.mark.parametrize('offset,dtype',[(125,np.int8),(126,np.uint8),(255,np.uint16)])
def test_relabel_signed_boundary_dtype_selection(relabel_function,offset,dtype):
    labels,forward,inverse=relabel_function(np.array([0,2,3],np.int8),offset=offset)
    assert labels.dtype==np.dtype(dtype) and forward.dtype==np.dtype(dtype)
    assert inverse.dtype==np.dtype(np.int8)
    np.testing.assert_array_equal(labels,[0,offset,offset+1])


def test_returned_class_scope_is_not_public_constructor_port():
    record=json.loads(MATRIX.read_text())['returned_classes']['ArrayMap']
    assert record['origin']=='skimage.util._map_array.ArrayMap'
    assert record['public_util_export'] is False and record['native_implemented'] is False
