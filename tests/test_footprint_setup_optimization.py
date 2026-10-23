"""Untimed rectangle-setup and mirror-metadata regression contracts."""
import warnings
import numpy as np
import pytest
from skimage import morphology as original
import skimagexx as xx
from skimagexx import _footprint_dense as dense


@pytest.fixture(params=[1,4],autouse=True)
def configuration(request):
    previous=xx.set_fallback_enabled(False);workers=xx.get_num_threads();xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(workers);xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('dtype',[np.dtype(prefix+name,metadata={'unit':'test'}) for prefix in ['<','>'] for name in ['?','u1','i1','u2','i2','u4','i4','u8','i8','f2','f4','f8','c8','c16']])
@pytest.mark.parametrize('shape',[(),(0,3),(3,5),(3,1,5)])
def test_numeric_rectangle_bytes_metadata_and_ownership(dtype,shape):
    actual=xx.morphology.footprint_rectangle(shape,dtype=dtype)
    expected=original.footprint_rectangle(shape,dtype=dtype)
    assert actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
    assert actual.shape==expected.shape and actual.strides==expected.strides
    assert actual.flags.owndata and actual.flags.writeable
    assert actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('decomposition',[None,'sequence','separable'])
def test_custom_dtype_provider_sideeffect_order_is_preserved(decomposition):
    def observe(namespace):
        events=[]
        class Provider:
            @property
            def dtype(self):events.append('dtype');return np.dtype('>i2')
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');result=namespace.footprint_rectangle((3,5),dtype=Provider(),decomposition=decomposition)
        return events,result,[(type(w.message),str(w.message)) for w in notices]
    events,actual,notices=observe(xx.morphology);expected_events,expected,wanted=observe(original)
    assert notices==wanted
    assert events==expected_events
    a=actual if isinstance(actual,tuple) else ((actual,1),);e=expected if isinstance(expected,tuple) else ((expected,1),)
    for (got,n),(want,m) in zip(a,e):assert n==m and got.tobytes()==want.tobytes()


def test_custom_metaclass_hash_is_never_consulted_by_admission():
    events=[]
    class Meta(type):
        def __hash__(cls):events.append('hash');return super().__hash__()
    class Custom(metaclass=Meta):pass
    assert not dense._inert_numeric_dtype(Custom)
    assert not events


@pytest.mark.parametrize('decomposition',[None,'sequence','separable'])
def test_rectangle_shape_validation_has_original_single_pass(decomposition):
    def observe(namespace):
        events=[]
        class Width(int):
            def __mod__(self,other):events.append(('mod',int(self)));return super().__mod__(other)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');result=namespace.footprint_rectangle((Width(3),Width(5)),decomposition=decomposition)
        return events,result,[(type(w.message),str(w.message)) for w in notices]
    events,actual,notices=observe(xx.morphology);expected_events,expected,wanted=observe(original)
    assert events==expected_events and notices==wanted
    a=actual if isinstance(actual,tuple) else ((actual,1),);e=expected if isinstance(expected,tuple) else ((expected,1),)
    for (got,n),(want,m) in zip(a,e):assert n==m and got.tobytes()==want.tobytes()


@pytest.mark.parametrize('dtype',[object,'U3','S3','M8[D]','m8[D]',np.dtype([('value','i2'),('flag','?')],align=True),np.dtype({'names':['x'],'formats':['i2'],'offsets':[2],'itemsize':8})])
def test_unusual_dtype_rectangle_retains_general_provider(dtype):
    assert not dense._inert_numeric_dtype(dtype)
    with warnings.catch_warnings(record=True) as a:
        warnings.simplefilter('always')
        try:actual,error=xx.morphology.footprint_rectangle((3,5),dtype=dtype),None
        except Exception as caught:actual,error=None,caught
    with warnings.catch_warnings(record=True) as e:
        warnings.simplefilter('always')
        try:expected,wanted=original.footprint_rectangle((3,5),dtype=dtype),None
        except Exception as caught:expected,wanted=None,caught
    assert [(type(w.message),str(w.message)) for w in a]==[(type(w.message),str(w.message)) for w in e]
    if wanted is not None:assert type(error) is type(wanted) and str(error)==str(wanted)
    else:
        assert error is None;np.testing.assert_array_equal(actual,expected)
        assert actual.dtype==expected.dtype and actual.dtype.metadata==expected.dtype.metadata
        # Padding is deliberately not interpreted or normalized by the typed
        # path; original NumPy scalar providers remain responsible for it.
        assert actual.flags.owndata and actual.strides==expected.strides


@pytest.mark.parametrize('shape',[(),(0,),(3,),(2,3),(2,1,4)])
@pytest.mark.parametrize('layout',['C','F','reverse','readonly','subclass'])
def test_mirror_view_metadata_scalar_and_protocols(shape,layout):
    image=np.arange(np.prod(shape,dtype=int),dtype=np.int16).reshape(shape)
    if layout=='F':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[(slice(None,None,-1),)*image.ndim]
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='subclass':
        class Subclass(np.ndarray):pass
        image=image.view(Subclass)
    actual=xx.morphology.mirror_footprint(image);expected=original.mirror_footprint(image)
    assert type(actual) is type(expected) and actual.dtype==expected.dtype
    np.testing.assert_array_equal(actual,expected)
    if isinstance(expected,np.ndarray):
        assert actual.strides==expected.strides and actual.flags.owndata==expected.flags.owndata
        assert actual.flags.writeable==expected.flags.writeable
        assert np.shares_memory(actual,image)==np.shares_memory(expected,image)


def test_mirror_sequence_count_identity_and_array_conversion_sideeffects():
    image=np.arange(6,dtype=np.int16).reshape(2,3);count=np.int64(2)
    sequence=((image,count),(image[::-1],3))
    actual=xx.morphology.mirror_footprint(sequence);expected=original.mirror_footprint(sequence)
    assert actual[0][1] is count and actual[1][1] is sequence[1][1]
    for (got,n),(want,m) in zip(actual,expected):assert n is m and got.strides==want.strides and np.shares_memory(got,image)
    def observe(namespace):
        events=[]
        class Provider:
            @property
            def __array_interface__(self):events.append('interface');return image.__array_interface__
        result=namespace.mirror_footprint(Provider());return events,result
    a,got=observe(xx.morphology);e,want=observe(original)
    assert a==e and got.strides==want.strides and np.shares_memory(got,image)
