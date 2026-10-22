"""Multi-Otsu thresholds, including rounding and first-maximum ties."""
import itertools
import warnings
import numpy as np
import pytest
from skimage.filters import threshold_multiotsu as original
import skimagexx as xx


@pytest.fixture(autouse=True,params=[1,4])
def strict(request):
    previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


def compare(image=None,**options):
    def call(function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:result=function(image,**options)
            except Exception as error:result=error
        return result,[(w.category,str(w.message)) for w in caught]
    expected,expected_warnings=call(original)
    actual,actual_warnings=call(xx.filters.threshold_multiotsu)
    assert actual_warnings==expected_warnings
    if isinstance(expected,Exception):
        assert type(actual) is type(expected)
        assert str(actual)==str(expected)
    else:
        assert type(actual) is type(expected) and actual.dtype==expected.dtype
        np.testing.assert_array_equal(actual,expected)
        assert actual.dtype.metadata==expected.dtype.metadata
        if expected.dtype.kind=='f':np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('classes',[2,3])
@pytest.mark.parametrize('bins',[7,32,256])
@pytest.mark.parametrize('layout',['flat','volume','reversed','fortran','constant','rgb','empty'])
def test_image(dtype,classes,bins,layout):
    if layout=='empty' and dtype!=np.uint8:
        pytest.skip('Reference empty floating histogram has undefined threshold indices')
    image=np.random.default_rng(73).integers(0,256,(9,17,21),dtype=np.uint8).astype(dtype)
    if dtype!=np.uint8:image=image/dtype(31)-dtype(3)
    if layout=='flat':image=image.ravel()
    elif layout=='reversed':image=image[::-1,::2,::-1]
    elif layout=='fortran':image=np.asfortranarray(image)
    elif layout=='constant':image[:]=13
    elif layout=='rgb':image=image[...,:3]
    elif layout=='empty':image=image[:0]
    compare(image,classes=classes,nbins=bins)


@pytest.mark.parametrize('classes',[2,3,4,5])
@pytest.mark.parametrize('bins',[7,17,33,65])
@pytest.mark.parametrize('kind',['random','gaps','flat','fractional','tiny','scaled'])
@pytest.mark.parametrize('center_dtype',[np.int64,np.float32,np.float64])
def test_histograms(classes,bins,kind,center_dtype):
    counts=np.random.default_rng(997).integers(1,100,bins).astype(np.float32)
    if kind=='gaps':counts[::3]=0
    elif kind=='flat':counts[:]=1
    elif kind=='fractional':counts/=np.float32(7.3)
    elif kind=='tiny':counts*=np.float32(1e-20)
    elif kind=='scaled':counts*=np.float32(1e7)
    centers=np.arange(bins,dtype=center_dtype)
    if center_dtype!=np.int64:centers=centers/center_dtype(13)-center_dtype(7)
    compare(hist=(counts,centers),classes=classes)


def test_exhaustive_small_histograms():
    for values in itertools.product(range(3),repeat=5):
        if sum(v!=0 for v in values)<3:continue
        compare(hist=np.array(values,dtype=np.int64),classes=3)


@pytest.mark.parametrize('classes',[2,3,4])
def test_large_parallel_and_unaligned(classes):
    # 1024 bins activates optional histogram-table and candidate workers.
    # Four classes uses fewer bins to bound the reference exhaustive search.
    bins=1024 if classes<4 else 65
    buffer=np.empty(4*bins+1,dtype=np.uint8)
    counts=np.ndarray((bins,),dtype=np.float32,buffer=buffer,offset=1)
    counts[:]=np.random.default_rng(29).integers(1,100,bins)
    compare(hist=counts,classes=classes)


def test_hot_swap():
    import skimage
    image=np.random.default_rng(3).integers(0,256,(31,37),dtype=np.uint8)
    def pipeline(backend):
        return np.digitize(image,backend.filters.threshold_multiotsu(image))
    np.testing.assert_array_equal(pipeline(xx),pipeline(skimage))


def test_equal_value_count_and_ignored_image():
    compare(np.arange(3,dtype=np.uint8),classes=3)
    compare(hist=(np.array([0,2,0,3,0,4,0]),np.arange(7,dtype=np.float32)),classes=3)
    compare(np.full((3,3),np.nan),hist=np.array([1,2,3,4]),classes=3)
    compare(classes=3)


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int64])
@pytest.mark.parametrize('layout',['readonly','strided','reversed','strided-readonly'])
@pytest.mark.parametrize('classes',[2,3,7])
def test_histogram_buffer_contract(dtype,layout,classes):
    counts=np.arange(1,15,dtype=dtype)
    if layout=='reversed':counts=counts[::-2]
    elif 'strided' in layout:counts=counts[::2]
    if 'readonly' in layout:counts.flags.writeable=False
    compare(hist=counts,classes=classes)


@pytest.mark.parametrize('classes',range(6,17))
def test_more_classes(classes):
    counts=np.random.default_rng(classes).integers(1,99,classes+3).astype(np.float32)
    compare(hist=counts,classes=classes)


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('supplied',[False,True])
def test_dtype_metadata(dtype,supplied):
    tagged=np.dtype(dtype,metadata={'units':'linear','source':'parity'})
    image=np.random.default_rng(533).integers(0,256,(17,19),dtype=np.uint8).astype(tagged)
    if supplied:
        centers=np.arange(17,dtype=dtype).astype(tagged)
        compare(hist=(np.arange(1,18,dtype=np.float32),centers))
    else:compare(image)


@pytest.mark.parametrize('dtype',['bool','uint16','float16','complex64','S5','U5','datetime64[D]','timedelta64[D]',object])
@pytest.mark.parametrize('gaps',[False,True])
def test_supplied_center_indexing(dtype,gaps):
    centers=np.arange(17).astype(dtype)
    counts=np.arange(1,18,dtype=np.float32)
    if gaps:counts[::3]=0
    compare(hist=(counts,centers[::-1]))


@pytest.mark.parametrize('classes',[2,3,4])
def test_unwritten_upstream_objective_is_explicit_fallback(classes,monkeypatch):
    # Never execute upstream's uninitialized-index path in this test.
    counts=np.full(7,1e-40,dtype=np.float32)
    with pytest.raises(NotImplementedError):xx.filters.threshold_multiotsu(hist=counts,classes=classes)
    import skimage.filters as upstream
    sentinel=np.array([-1,1],dtype=np.int64)
    def delegated(image=None,classes=3,nbins=256,*,hist=None):
        assert hist is counts
        return sentinel
    monkeypatch.setattr(upstream,'threshold_multiotsu',delegated)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        assert xx.filters.threshold_multiotsu(hist=counts,classes=classes) is sentinel
