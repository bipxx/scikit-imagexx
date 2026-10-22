"""Label-color contracts captured before the native implementation."""
import inspect
import warnings

import numpy as np
import pytest
from skimage.color import label2rgb as reference
import skimagexx as xx


def observed(function,*args,**kwargs):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            value=function(*args,**kwargs)
            result=('value',value)
        except Exception as error:
            result=('error',type(error),str(error))
    return result,[(type(n.message),str(n.message)) for n in notices]


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1)
    xx.set_fallback_enabled(previous)


def check(labels,**options):
    saved=labels.copy()
    expected,expected_notices=observed(reference,labels,**options)
    actual,notices=observed(xx.color.label2rgb,labels,**options)
    assert notices==expected_notices
    assert actual[0]==expected[0]
    if actual[0]=='error':assert actual[1:]==expected[1:]
    else:
        assert actual[1].dtype==expected[1].dtype
        assert actual[1].dtype.metadata==expected[1].dtype.metadata
        assert actual[1].shape==expected[1].shape
        underlay=options.get('image')
        if underlay is not None and options.get('kind','overlay')=='overlay' and underlay.ndim==labels.ndim+1:
            # RGB desaturation inherits the separately tested HSV roundoff
            # budget. Palette-only calls and average colors stay exact.
            single=underlay.dtype==np.float32
            np.testing.assert_allclose(actual[1],expected[1],rtol=2e-6 if single else 2e-13,atol=2e-6 if single else 2e-13)
        else:np.testing.assert_array_equal(actual[1],expected[1])
        np.testing.assert_array_equal(np.signbit(actual[1]),np.signbit(expected[1]))
    np.testing.assert_array_equal(labels,saved)


def test_signature():
    assert inspect.signature(xx.color.label2rgb)==inspect.signature(reference)


@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float32,np.float64])
@pytest.mark.parametrize('layout',['scalar','flat','volume','fortran','reversed','stepped','broadcast','readonly'])
@pytest.mark.parametrize('axis',[-1,0,1])
def test_labels_without_underlay(dtype,layout,axis):
    labels=np.random.default_rng(8227).integers(0,19,(3,11,17)).astype(dtype)
    if layout=='scalar':labels=np.array(7,dtype=dtype)
    elif layout=='flat':labels=labels.ravel()
    elif layout=='fortran':labels=np.asfortranarray(labels)
    elif layout=='reversed':labels=labels[::-1,:,::-1]
    elif layout=='stepped':labels=labels[:,::2,::3]
    elif layout=='broadcast':labels=np.broadcast_to(labels[:1],(7,11,17))
    elif layout=='readonly':labels.flags.writeable=False
    check(labels,channel_axis=axis)


@pytest.mark.parametrize('colors',[None,['red','blue','yellow'],[(.2,.3,.4)],[(1,0,0,9),(0,1,0,8)],np.array([[.1,.2,.3],[.4,.5,.6]],dtype=np.float32),[(1,0,0,object())]])
@pytest.mark.parametrize('background',[0,2,9,256])
@pytest.mark.parametrize('background_color',[None,'cyan',(0,0,0),(.2,.4,.6),(0,0,0,object())])
def test_palette_and_background(colors,background,background_color):
    check(np.array([[0,1,2,9],[9,2,1,0]],dtype=np.uint8),colors=colors,bg_label=background,bg_color=background_color)


@pytest.mark.parametrize('alpha',[None,object(),np.array([1,2]),'ignored',-7,np.nan])
@pytest.mark.parametrize('image_alpha',[None,object(),np.array([1,2]),'ignored'])
def test_opacity_ignored_without_underlay(alpha,image_alpha):
    check(np.array([[0,1,2],[3,2,1]],dtype=np.uint8),alpha=alpha,image_alpha=image_alpha)


@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.int32,np.float32,np.float64])
@pytest.mark.parametrize('shape',[(0,),(3,0,2)])
def test_empty_labels(dtype,shape):
    check(np.empty(shape,dtype=dtype))


@pytest.mark.parametrize('background',[0,256])
def test_dense_byte_rank_overflow(background):
    check(np.arange(256,dtype=np.uint8).reshape(16,16),bg_label=background)


@pytest.mark.parametrize('dtype',[np.uint16,np.uint32,np.uint64,np.int64])
def test_sparse_labels(dtype):
    check(np.array([0,7,33,1024,4095,7,0],dtype=dtype))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_fractional_labels_are_cast_before_ranking(dtype):
    check(np.array([0,.1,.9,1.,1.9,2.,2.9,4.1],dtype=dtype))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('background',[2,9,256])
def test_floating_labels_with_zero_and_explicit_background(dtype,background):
    check(np.array([0,.1,.9,1.,1.9,2.,2.9,4.1],dtype=dtype),bg_label=background,bg_color=None)


@pytest.mark.parametrize('size',[15,16,17,31,32,33,63,64,65,127,128,129,513])
def test_packet_tails(size):
    check(np.random.default_rng(size).integers(0,256,size,dtype=np.uint8))


def test_large_parallel_labels():
    check(np.random.default_rng(17).integers(0,256,(513,517),dtype=np.uint8))


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('rgb',[False,True])
@pytest.mark.parametrize('axis',[-1,0,1])
@pytest.mark.parametrize('saturation',[0,.5,1])
@pytest.mark.parametrize('alpha',[0,.3,1])
def test_overlay_underlay(dtype,rgb,axis,saturation,alpha):
    rng=np.random.default_rng(911)
    labels=rng.integers(0,19,(13,17),dtype=np.uint8)
    image=rng.integers(0,256,(*labels.shape,3) if rgb else labels.shape,dtype=np.uint8)
    if dtype!=np.uint8:image=(image/255).astype(dtype)
    image=np.moveaxis(image,-1,axis)
    saved=image.copy()
    check(labels,image=image,channel_axis=axis,saturation=saturation,alpha=alpha,image_alpha=.7)
    np.testing.assert_array_equal(image,saved)


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('rgb',[False,True])
@pytest.mark.parametrize('axis',[-1,0,1])
@pytest.mark.parametrize('background',[0,2,400])
def test_average_underlay(dtype,rgb,axis,background):
    rng=np.random.default_rng(701)
    labels=rng.integers(0,19,(13,17),dtype=np.uint8)
    image=rng.integers(0,256,(*labels.shape,3) if rgb else labels.shape,dtype=np.uint8)
    if dtype!=np.uint8:image=(image/255).astype(dtype)
    image=np.moveaxis(image,-1,axis)
    saved=image.copy()
    check(labels,image=image,kind='avg',channel_axis=axis,bg_label=background)
    np.testing.assert_array_equal(image,saved)


def test_average_ignores_overlay_controls():
    labels=np.array([[0,1,2],[1,2,3]],dtype=np.uint8)
    image=np.arange(18,dtype=np.float32).reshape(2,3,3)/18
    check(labels,image=image,kind='avg',colors=object(),alpha=object(),image_alpha=object(),saturation=object())


@pytest.mark.parametrize('value',[0,1])
def test_scalar_average_compatibility(value):
    labels=np.array(value,dtype=np.uint8);image=np.array([.2,.4,.6])
    expected,expected_notices=observed(reference,labels,image=image,kind='avg')
    xx.set_fallback_enabled(True)
    actual,notices=observed(xx.color.label2rgb,labels,image=image,kind='avg')
    assert [n for n in notices if n[0] is xx.FallbackWarning]==[
        (xx.FallbackWarning,'skimage.color.label2rgb executed by scikit-image fallback')]
    assert [n for n in notices if n[0] is not xx.FallbackWarning]==expected_notices
    assert actual[0]==expected[0]
    if actual[0]=='error':assert actual[1:]==expected[1:]
    else:np.testing.assert_array_equal(actual[1],expected[1])


@pytest.mark.parametrize('labels,options',[
    ([0,1,2],{}),
    (np.array([0,1,2],dtype=np.float16),{}),
    (np.array([-1,0,2],dtype=np.int16),{}),
    (np.array([0,np.nan,2]),{}),
    (np.array([0,1,2],dtype=np.uint8),{'bg_label':-1}),
    (np.array([0,1,2],dtype=np.uint8),{'bg_label':np.int64(0)}),
    (np.array([0,1,2],dtype=np.uint8),{'image':np.ones(3,dtype=np.uint16)}),
    (np.array([0,1,2],dtype=np.uint8),{'image':np.array([0.,2.,1.])}),
    (np.array([0,1,2],dtype=np.uint8),{'image':np.array([0.,np.inf,1.]),'kind':'avg'}),
])
def test_unsupported_domains_use_explicit_whole_call_fallback(labels,options):
    expected,expected_notices=observed(reference,labels,**options)
    with pytest.raises(NotImplementedError):xx.color.label2rgb(labels,**options)
    xx.set_fallback_enabled(True)
    actual,notices=observed(xx.color.label2rgb,labels,**options)
    assert [n for n in notices if n[0] is xx.FallbackWarning]==[
        (xx.FallbackWarning,'skimage.color.label2rgb executed by scikit-image fallback')]
    assert [n for n in notices if n[0] is not xx.FallbackWarning]==expected_notices
    assert actual[0]==expected[0]
    if actual[0]=='error':assert actual[1:]==expected[1:]
    else:
        assert actual[1].dtype==expected[1].dtype
        np.testing.assert_array_equal(actual[1],expected[1])


@pytest.mark.parametrize('kind',['overlay','avg'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_underlay_dtype_metadata(kind,dtype):
    labels=np.array([[0,1,2],[1,2,3]],dtype=np.uint8)
    image=(np.arange(18,dtype=dtype).reshape(2,3,3)/18).view(np.dtype(dtype,metadata={'unit':'intensity'}))
    check(labels,image=image,kind=kind)


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('rgb',[False,True])
@pytest.mark.parametrize('seed',[19,311])
def test_large_grouped_average(dtype,rgb,seed):
    rng=np.random.default_rng(seed)
    labels=rng.integers(0,19,(513,517),dtype=np.uint8)
    image=rng.integers(0,256,(*labels.shape,3) if rgb else labels.shape,dtype=np.uint8)
    if dtype!=np.uint8:image=(image/255).astype(dtype)
    image.flags.writeable=False
    check(labels,image=image,kind='avg')


@pytest.mark.parametrize('buffer_size',[16,1024,8192,16384,32768])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_grayscale_average_reduction_buffer(dtype,buffer_size):
    rng=np.random.default_rng(19)
    labels=rng.integers(0,3,(129,131),dtype=np.uint8)
    image=(rng.integers(0,256,labels.shape,dtype=np.uint8)/255).astype(dtype)
    previous=np.setbufsize(buffer_size)
    try:check(labels,image=image,kind='avg')
    finally:np.setbufsize(previous)


@pytest.mark.parametrize('groups',[31,512,513])
@pytest.mark.parametrize('dtype',[np.uint64,np.int64,np.float64])
@pytest.mark.parametrize('kind',['overlay','avg'])
def test_large_sparse_rank_groups(groups,dtype,kind):
    rng=np.random.default_rng(411)
    raw=rng.integers(0,groups,(129,131),dtype=np.uint64)
    labels=np.where(raw==0,0,raw*10000000019+2**50).astype(dtype)
    image=rng.random((*labels.shape,3))
    check(labels,image=image,kind=kind)


@pytest.mark.parametrize('kind',['','invalid','Overlay'])
def test_invalid_kind(kind):
    check(np.array([[0,1,2]],dtype=np.uint8),kind=kind)


def test_reference_contract_ignores_both_opacities():
    labels=np.array([0,1,2],dtype=np.uint8)
    np.testing.assert_array_equal(reference(labels,alpha=object(),image_alpha=object()),
                                  np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,1.]]))


def test_reference_contract_dense_rank_wrap():
    labels=np.arange(256,dtype=np.uint8)
    result=reference(labels,bg_label=256)
    # The upstream inverse ranks are assigned back to its minimal uint8 copy.
    # Rank256 wraps to0 even though background256 is absent from the labels.
    np.testing.assert_array_equal(result[-1],[0.,0.,0.])
    np.testing.assert_array_equal(result[0],[1.,0.,0.])


def test_reference_contract_background_removal_uses_remapped_labels():
    labels=np.arange(4,dtype=np.uint8)
    result=reference(labels,bg_label=2,bg_color=None)
    # Removal compares against remapped labels, so rank2 is also set to black.
    np.testing.assert_array_equal(result[1],[0.,0.,0.])
    np.testing.assert_array_equal(result[2],[0.,0.,0.])


def test_reference_contract_does_not_mutate_labels():
    labels=np.arange(12,dtype=np.uint8).reshape(3,4)
    before=labels.copy()
    reference(labels,bg_label=2,bg_color=None)
    np.testing.assert_array_equal(labels,before)


def test_reference_contract_desaturation_uses_hsv_value():
    labels=np.ones((1,1),dtype=np.uint8)
    image=np.array([[[.2,.8,.4]]],dtype=np.float64)
    np.testing.assert_array_equal(reference(labels,image=image,alpha=0,saturation=0),
                                  np.full((1,1,3),.8))


def test_reference_contract_average_keeps_raw_integer_scale():
    labels=np.ones((1,2),dtype=np.uint8)
    image=np.array([[[0,2,4],[1,3,5]]],dtype=np.uint8)
    result=reference(labels,image=image,kind='avg')
    assert result.dtype==np.uint8
    np.testing.assert_array_equal(result,np.array([[[0,2,4],[0,2,4]]],dtype=np.uint8))
