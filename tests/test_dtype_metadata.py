"""Dtype tags are observable even though NumPy dtype equality ignores them."""
import importlib
import re
import warnings
import numpy as np
import pytest
import skimagexx as xx
from skimagexx.coverage import native_functions,native_dtypes,needs_color_input,default_call,parity_tolerance

CASES=[(api,dtype) for api in native_functions() for dtype in native_dtypes(api)]


def calls(api,dtype):
    rng=np.random.default_rng(533)
    shape=(17,19,3) if needs_color_input(api) else (17,19)
    data=rng.integers(0,256,shape,dtype=np.uint8)
    if dtype!='uint8':data=(data/255).astype(dtype)
    tagged=np.dtype(dtype,metadata={'units':'linear','source':'parity'})
    data=data.astype(tagged)
    arguments,options=default_call(api,data)
    if arguments and isinstance(arguments[0],np.ndarray):
        first=arguments[0].view(np.dtype(arguments[0].dtype,metadata={'units':'linear','source':'parity'}))
        arguments=(first,*arguments[1:])
    module,name=api.rsplit('.',1)
    reference=getattr(importlib.import_module('skimage.'+module),name)
    native=getattr(importlib.import_module('skimagexx.'+module),name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected=reference(*arguments,**options)
        actual=native(*arguments,**options)
        if arguments and expected is arguments[0]:assert actual is arguments[0]
    return actual,expected,data


def compare(actual,expected,api,dtype,path='result',image=None):
    if isinstance(expected,(tuple,list)):
        assert type(actual) is type(expected),path
        assert len(actual)==len(expected),path
        for index,(a,e) in enumerate(zip(actual,expected)):compare(a,e,api,dtype,f'{path}[{index}]',image=image)
        return
    if api=='segmentation.relabel_sequential' and type(expected).__name__=='ArrayMap':
        # These results store sparse mapping vectors. Array conversion is a
        # separate protocol, exercised by the dedicated returned-map contracts.
        assert type(actual).__name__==type(expected).__name__,path
        for field in ('in_values','out_values'):
            a,e=getattr(actual,field),getattr(expected,field)
            assert a.shape==e.shape and a.dtype==e.dtype,f'{path}.{field}'
            np.testing.assert_array_equal(a,e)
            assert a.dtype.metadata==e.dtype.metadata,f'{path}.{field}: dtype metadata'
        return
    a,e=np.asarray(actual),np.asarray(expected)
    assert a.shape==e.shape and a.dtype==e.dtype,path
    rtol,atol=parity_tolerance(api,np.dtype(dtype),image=image,expected=expected)
    np.testing.assert_allclose(a,e,rtol=rtol,atol=atol)
    assert a.dtype.metadata==e.dtype.metadata,f'{path}: native metadata={a.dtype.metadata!r}; reference={e.dtype.metadata!r}'


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


@pytest.mark.parametrize('api,dtype',CASES)
def test_every_native_entry_dtype_metadata(api,dtype):
    actual,expected,image=calls(api,dtype)
    compare(actual,expected,api,dtype,image=image)


@pytest.mark.parametrize('name',['deltaE_cie76','deltaE_ciede94','deltaE_ciede2000','deltaE_cmc'])
@pytest.mark.parametrize('first_dtype',['uint8','float32','float64'])
@pytest.mark.parametrize('second_dtype',['uint8','float32','float64'])
@pytest.mark.parametrize('tags',['none','first','second','both'])
@pytest.mark.parametrize('layout',['arrays','first-scalar','second-scalar','scalars'])
def test_distance_metadata_operand_order_and_promotion(name,first_dtype,second_dtype,tags,layout):
    first=np.dtype(first_dtype,metadata={'operand':'first'}) if tags in ('first','both') else np.dtype(first_dtype)
    second=np.dtype(second_dtype,metadata={'operand':'second'}) if tags in ('second','both') else np.dtype(second_dtype)
    a=np.full((2,5,3),[20,10,5],dtype=first)
    b=np.full((2,5,3),[25,15,10],dtype=second)
    if layout in ('first-scalar','scalars'):a=a[0,0]
    if layout in ('second-scalar','scalars'):b=b[0,0]
    reference=getattr(importlib.import_module('skimage.color'),name)
    native=getattr(xx.color,name)
    try:expected=reference(a,b)
    except Exception as error:
        with pytest.raises(type(error),match=re.escape(str(error))):native(a,b)
        return
    actual=native(a,b)
    compare(actual,expected,'color.'+name,'float32' if first_dtype==second_dtype=='float32' else 'float64')


@pytest.mark.parametrize('name',['gray2rgb','gray2rgba'])
@pytest.mark.parametrize('dtype',['bool','uint8','int16','uint64','float16','float32','float64',
                                   'complex64','complex128','S5','U4','V7','datetime64[ns]',
                                   'timedelta64[ns]',np.dtype([('x','i2'),('y','f4')])])
@pytest.mark.parametrize('endian',[False,True])
def test_grayscale_expansion_metadata_across_fixed_storage(name,dtype,endian):
    dtype=np.dtype(dtype,metadata={'units':'linear','source':'parity'})
    if endian:dtype=dtype.newbyteorder('>')
    image=np.zeros((5,7),dtype=dtype)
    options={} if name=='gray2rgb' else {'alpha':np.zeros_like(image)}
    expected=getattr(importlib.import_module('skimage.color'),name)(image,**options)
    actual=getattr(xx.color,name)(image,**options)
    assert actual.dtype==expected.dtype
    assert actual.dtype.metadata==expected.dtype.metadata
    assert actual.shape==expected.shape
    np.testing.assert_array_equal(actual,expected)


RECORD_TYPES=[np.dtype([('x','>i2'),('y','>f4')]),
              np.dtype([('x','>u4'),('y','<f8')]),
              np.dtype([('x','>i2'),('y','>f8')],align=True),
              np.dtype([('record',[('a','>i2'),('b','>u4')]),('vec','>f4',(3,))]),
              np.dtype([('label','>U2'),('z','>c8')]),
              np.dtype([('z','>c16'),('when','>M8[ns]'),('dt','>m8[ms]')])]


def record_values(dtype,shape,seed):
    data=np.zeros(shape,dtype=dtype);rng=np.random.default_rng(seed)
    def fill(view):
        if view.dtype.fields:
            for name in view.dtype.names:fill(view[name])
        elif view.dtype.kind in 'iu':view[:]=rng.integers(1,100,view.shape)
        elif view.dtype.kind in 'fc':
            view[:]=rng.normal(size=view.shape)
            if view.dtype.kind=='c':view[:]+=1j*rng.normal(size=view.shape)
        elif view.dtype.kind=='U':view[:]='AZ'
        elif view.dtype.kind in 'mM':view[:]=np.arange(view.size).reshape(view.shape).astype(view.dtype)
    fill(data)
    return data


def check_record(name,image,alpha):
    options={} if name=='gray2rgb' else {'alpha':alpha}
    with warnings.catch_warnings(record=True) as reference_notices:
        warnings.simplefilter('always');expected=getattr(importlib.import_module('skimage.color'),name)(image,**options)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');actual=getattr(xx.color,name)(image,**options)
    assert [(type(w.message),str(w.message)) for w in notices]==[(type(w.message),str(w.message)) for w in reference_notices]
    assert actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
    def equal(a,e):
        if e.dtype.fields:
            for field in e.dtype.names:equal(a[field],e[field])
        else:np.testing.assert_array_equal(a,e)
    equal(actual,expected)
    assert not np.shares_memory(actual,image)


@pytest.mark.parametrize('name',['gray2rgb','gray2rgba'])
@pytest.mark.parametrize('dtype',RECORD_TYPES)
@pytest.mark.parametrize('layout',['contiguous','fortran','reversed','stepped','unaligned','readonly'])
def test_structured_endian_nonzero_values(name,dtype,layout):
    image=record_values(dtype,(17,19),534);alpha=record_values(dtype,image.shape,535)
    if layout=='fortran':image=np.asfortranarray(image);alpha=np.asfortranarray(alpha)
    elif layout=='reversed':image=image[::-1,::-1];alpha=alpha[::-1,::-1]
    elif layout=='stepped':image=np.tile(image,(2,2))[::2,::2];alpha=np.tile(alpha,(2,2))[::2,::2]
    elif layout=='readonly':image.flags.writeable=False;alpha.flags.writeable=False
    elif layout=='unaligned':
        storage=np.empty(image.nbytes+1,dtype=np.uint8)
        view=np.ndarray(image.shape,dtype=dtype,buffer=storage,offset=1);view[:]=image;image=view
    check_record(name,image,alpha)


@pytest.mark.parametrize('name',['gray2rgb','gray2rgba'])
@pytest.mark.parametrize('dtype',[RECORD_TYPES[0],RECORD_TYPES[3]])
def test_parallel_structured_endian(name,dtype):
    image=record_values(dtype,(513,517),536);alpha=record_values(dtype,image.shape,537)
    check_record(name,image,alpha)


@pytest.mark.parametrize('dtype',RECORD_TYPES)
def test_broadcast_structured_alpha(dtype):
    image=record_values(dtype,(17,19),538)
    alpha=record_values(dtype,(1,1),539).reshape(())
    check_record('gray2rgba',image,alpha)


@pytest.mark.parametrize('field,value',[('a',np.nan),('when',np.datetime64('NaT','ns'))])
def test_structured_alpha_comparison_warnings(field,value):
    dtype=np.dtype([('a','>f4'),('when','>M8[ns]')])
    image=record_values(dtype,(17,19),540);alpha=record_values(dtype,image.shape,541)
    alpha[field][5,7]=value
    check_record('gray2rgba',image,alpha)
