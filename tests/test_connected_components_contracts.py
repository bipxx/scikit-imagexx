"""Test-first contracts for measure.label and its morphology.label alias.

Reference behavior and a small independent flood-fill oracle are tested now.
Native cases explicitly remain pending behind strict fallback rejection until
the corresponding API is registered; forwarding never counts as native work.
"""
from collections import deque
import inspect
from itertools import product
import json
from pathlib import Path
import warnings
import numpy as np
import pytest
from skimage import measure,morphology
import skimagexx as xx
from skimagexx.coverage import native_functions


APIS=('measure.label','morphology.label')
TYPES=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,
       np.uint64,np.int64,np.float16,np.float32,np.float64)
MATRIX=Path(__file__).parents[1]/'coverage/contracts/connected-components.json'


def original(api):return getattr(measure if api.startswith('measure.') else morphology,'label')


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


@pytest.fixture(params=APIS)
def api(request):return request.param


@pytest.fixture(params=['reference','native'])
def label_function(request,api):
    if request.param=='reference':return original(api)
    namespace,name=api.split('.')
    candidate=getattr(getattr(xx,namespace),name)
    if api not in native_functions():
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            with pytest.raises(NotImplementedError,match='fallback is disabled'):
                candidate(np.array([0,1],np.int8))
        assert not notices
        pytest.xfail(f'native {api} connected-component contract is pending')
    assert candidate is not original(api),'registered native API leaked the upstream function'
    return candidate


def observe(function,image,options):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:result,error=function(image,**options),None
        except Exception as caught:result,error=None,caught
    return result,error,[(type(w.message),str(w.message)) for w in notices]


def flood_fill(image,background=None,connectivity=None):
    """Independent definition-level oracle, not union-find or reference code."""
    if image.dtype==bool:
        values=np.logical_not(image) if background==1 else np.asarray(image,dtype=bool)
        background=0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            values=image.astype(np.intp)
        background=0 if background is None else int(background)
    connectivity=image.ndim if connectivity is None else connectivity
    offsets=[offset for offset in product((-1,0,1),repeat=image.ndim)
             if 1<=sum(abs(step) for step in offset)<=connectivity]
    labels=np.zeros(image.shape,np.intp);count=0
    # np.ndindex is logical C order, which fixes the canonical component IDs.
    for start in np.ndindex(image.shape):
        if labels[start] or values[start]==background:continue
        count+=1;labels[start]=count;pending=deque([start]);value=values[start]
        while pending:
            point=pending.popleft()
            for offset in offsets:
                neighbor=tuple(index+step for index,step in zip(point,offset))
                if any(index<0 or index>=length for index,length in zip(neighbor,image.shape)):continue
                if not labels[neighbor] and values[neighbor]==value:
                    labels[neighbor]=count;pending.append(neighbor)
    return labels,count


def check(function,api,image,*,oracle=True,**options):
    before=image.tobytes();metadata=(image.dtype,image.shape,image.strides,image.flags.writeable,image.ctypes.data)
    expected,error,notices=observe(original(api),image,options)
    actual,actual_error,actual_notices=observe(function,image,options)
    assert actual_notices==notices
    assert image.tobytes()==before
    assert (image.dtype,image.shape,image.strides,image.flags.writeable,image.ctypes.data)==metadata
    if error is not None:
        assert type(actual_error) is type(error)
        assert str(actual_error)==str(error)
        return None
    if actual_error is not None:raise actual_error
    if isinstance(expected,tuple):
        assert type(actual) is tuple and len(actual)==2
        expected_labels,expected_count=expected;labels,count=actual
        assert type(count) is type(expected_count) is int and count==expected_count
    else:
        labels=actual;expected_labels=expected;count=None
    assert type(labels) is type(expected_labels) is np.ndarray
    assert labels.dtype==expected_labels.dtype and labels.dtype.metadata==expected_labels.dtype.metadata
    assert labels.shape==expected_labels.shape and labels.strides==expected_labels.strides
    assert labels.flags.c_contiguous and labels.flags.writeable
    assert labels.flags.owndata==expected_labels.flags.owndata
    assert not np.shares_memory(labels,image)
    np.testing.assert_array_equal(labels,expected_labels)
    if oracle:
        model,number=flood_fill(image,options.get('background'),options.get('connectivity'))
        np.testing.assert_array_equal(labels,model)
        if count is not None:assert count==number
    return labels


def sample(dtype,shape):
    values=np.resize(np.array([0,1,1,2,0,2,3,1,0,3,3,0],np.int64),shape)
    return values.astype(dtype)


def test_signature_alias_and_declared_status(label_function,api):
    record=json.loads(MATRIX.read_text(encoding='utf-8'))
    assert str(inspect.signature(label_function))==record['signature']
    assert measure.label is morphology.label
    if all(name in native_functions() for name in APIS):
        assert xx.measure.label is xx.morphology.label


@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('shape',[(9,),(3,5),(2,3,4),(2,2,2,2)])
@pytest.mark.parametrize('connectivity',[None,1,2,3])
def test_dtypes_dimensions_and_connectivity(label_function,api,dtype,shape,connectivity):
    check(label_function,api,sample(dtype,shape),connectivity=connectivity,return_num=True)


@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('layout',['fortran','reverse','step','broadcast','readonly','unaligned','endian','tagged'])
def test_layout_dtype_and_source_contract(label_function,api,dtype,layout):
    image=sample(dtype,(5,7))
    if layout=='fortran':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='step':image=sample(dtype,(10,14))[::2,::2]
    elif layout=='broadcast':image=np.broadcast_to(image[:1],image.shape)
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='unaligned':
        source=image;image=np.ndarray(image.shape,image.dtype,buffer=bytearray(image.nbytes+1),offset=1);image[...]=source
    elif layout=='endian':image=image.astype(image.dtype.newbyteorder('>'))
    else:image=image.view(np.dtype(image.dtype,metadata={'unit':'labels'}))
    check(label_function,api,image,return_num=True)


@pytest.mark.parametrize('dtype',[np.bool_,np.int64])
@pytest.mark.parametrize('background',[None,-1,0,1,2,1.5,'1',np.nan,np.inf])
def test_background_conversion_and_boolean_special_case(label_function,api,dtype,background):
    check(label_function,api,np.array([[0,1,0],[1,2,2]],dtype),background=background,return_num=True)


@pytest.mark.parametrize('dtype',[np.bool_,np.int64])
@pytest.mark.parametrize('connectivity',[-1,0,1.5,2.5,'1',np.nan,np.inf])
def test_connectivity_scalar_validation(label_function,api,dtype,connectivity):
    check(label_function,api,np.eye(3,dtype=dtype),connectivity=connectivity,return_num=True)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('pattern',[[0.,.9,1.1,1.9,-.9,-1.2],
    [np.nan,np.inf,-np.inf,0.,1.],[0.,-0.,1.,-1.],[1e30,-1e30,1.,0.]])
def test_float_to_index_conversion(label_function,api,dtype,pattern):
    with np.errstate(over='ignore'):image=np.asarray(pattern,dtype)
    check(label_function,api,image,return_num=True)


@pytest.mark.parametrize('invalid',['ignore','warn','raise'])
def test_nonfinite_cast_error_policy(label_function,api,invalid):
    image=np.array([np.nan,np.inf,-np.inf,1.],np.float64)
    with np.errstate(invalid=invalid):check(label_function,api,image,return_num=True)


@pytest.mark.parametrize('case',range(6))
def test_joint_validation_and_warning_precedence(label_function,api,case):
    cases=[(np.array(np.nan),dict(connectivity=0,background='bad')),
           (np.full((1,1,1,1),np.nan),dict(connectivity=0,background='bad')),
           (np.array([np.nan]),dict(connectivity=0,background='bad')),
           (np.array([0,1]),dict(connectivity=0,background='bad')),
           (np.array(True),dict(connectivity=0,background='bad')),
           (np.ones((1,1,1,1),bool),dict(connectivity=0,background=np.array([1,2])))]
    image,options=cases[case]
    check(label_function,api,image,**options)


@pytest.mark.parametrize('dtype',[np.uint64,np.int64,np.longlong,np.ulonglong])
@pytest.mark.parametrize('background',[None,-1,2**63-1,2**64-1])
def test_full_width_values_and_background(label_function,api,dtype,background):
    words=np.array([0,1,2**63,2**63+1,2**64-1,2**64-1,0],np.uint64)
    image=words.astype(dtype)
    check(label_function,api,image,background=background,return_num=True)


@pytest.mark.parametrize('dtype',[np.bool_,np.int64,np.float64])
@pytest.mark.parametrize('shape',[(),(0,),(0,3),(2,0,3),(2,2,2,2,2)])
def test_scalar_empty_and_higher_ranks(label_function,api,dtype,shape):
    check(label_function,api,np.zeros(shape,dtype),return_num=True)


@pytest.mark.parametrize('dtype',[np.bool_,np.int64])
@pytest.mark.parametrize('return_num',[False,True,None,0,1,'yes',np.bool_(True),np.array(False),np.array([True]),np.array([True,False])])
def test_return_number_truth_and_shape(label_function,api,dtype,return_num):
    check(label_function,api,sample(dtype,(3,5)),return_num=return_num)


def test_fixed_canonical_component_numbering(label_function,api):
    image=np.array([[1,0,1],[0,1,0],[2,0,2]],np.int64)
    for connectivity,wanted in [(1,[[1,0,2],[0,3,0],[4,0,5]]),(2,[[1,0,1],[0,1,0],[2,0,3]])]:
        labels=check(label_function,api,image,connectivity=connectivity,return_num=True)
        np.testing.assert_array_equal(labels,np.asarray(wanted))


@pytest.mark.parametrize('dtype',[np.bool_,np.int64])
@pytest.mark.parametrize('pattern',['stripes','checkerboard'])
def test_large_canonical_numbering_at_worker_limits(label_function,api,dtype,pattern):
    if pattern=='stripes':
        image=np.zeros((257,1025),dtype);image[::2]=1
        expected=np.zeros(image.shape,np.intp);expected[::2]=np.arange(1,130)[:,None];number=129
    else:
        image=(np.indices((513,513)).sum(axis=0)%2).astype(dtype)
        expected=np.zeros(image.shape,np.intp);number=int(np.count_nonzero(image))
        expected[image!=0]=np.arange(1,number+1)
    actual=label_function(image,connectivity=1,return_num=True)
    assert type(actual) is tuple and actual[1]==number
    np.testing.assert_array_equal(actual[0],expected)
    assert actual[0].dtype==(np.dtype(np.int32) if dtype is np.bool_ else np.dtype(np.intp))


def test_noncanonical_boolean_storage(label_function,api):
    image=np.resize(np.array([0,1,2,128,255,0],np.uint8),(5,7)).view(np.bool_)
    check(label_function,api,image,return_num=True)
    check(label_function,api,image,background=1,return_num=True)


@pytest.mark.parametrize('name',APIS)
def test_unregistered_label_rejects_strict_fallback(name):
    if name in native_functions():return
    namespace,attribute=name.split('.')
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        with pytest.raises(NotImplementedError,match='fallback is disabled'):
            getattr(getattr(xx,namespace),attribute)(np.array([0,1],np.int8))
    assert not notices
