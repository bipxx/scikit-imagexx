"""Minimum-threshold public contract, including unsuccessful convergence."""
import inspect
import warnings

import numpy as np
import pytest
import skimage
from skimage import filters as reference
import skimagexx as xx


@pytest.fixture(autouse=True, params=[1, 4])
def strict(request):
    previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1)
    xx.set_fallback_enabled(previous)


def outcome(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            value=function(*args, **kwargs)
            result=('value', type(value), np.asarray(value).dtype, value)
        except Exception as error:
            result=('error', type(error), str(error))
    return result, [(type(w.message), str(w.message)) for w in notices]


def check(*args, _fallback_warning=False, **kwargs):
    expected, expected_notices=outcome(reference.threshold_minimum,*args,**kwargs)
    actual, actual_notices=outcome(xx.filters.threshold_minimum,*args,**kwargs)
    diagnostics=[notice for notice in actual_notices if notice[0] is xx.FallbackWarning]
    assert diagnostics==([(xx.FallbackWarning,'skimage.filters.threshold_minimum executed by scikit-image fallback')]
                         if _fallback_warning else [])
    actual_notices=[notice for notice in actual_notices if notice[0] is not xx.FallbackWarning]
    assert actual_notices==expected_notices
    assert actual[:3]==expected[:3]
    if expected[0]=='value':np.testing.assert_array_equal(actual[3],expected[3])
    return actual


def test_signature():
    assert inspect.signature(xx.filters.threshold_minimum)==inspect.signature(reference.threshold_minimum)


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('nbins',[1,2,3,7,32,256,513])
@pytest.mark.parametrize('layout',['flat','volume','fortran','reversed','stepped','broadcast','constant'])
def test_image(dtype,nbins,layout):
    rng=np.random.default_rng(9371)
    image=rng.integers(0,64,(3,17,29),dtype=np.uint8)
    image[1:]+=np.uint8(160)
    if dtype!=np.uint8:image=(image.astype(dtype)/dtype(255)-dtype(.3))
    if layout=='flat':image=image.ravel()
    elif layout=='fortran':image=np.asfortranarray(image)
    elif layout=='reversed':image=image[::-1,:,::-1]
    elif layout=='stepped':image=image[:,::2,::3]
    elif layout=='broadcast':image=np.broadcast_to(image[0],(7,*image.shape[1:]))
    elif layout=='constant':image=np.full((3,17),13,dtype=dtype)
    before=image.copy()
    check(image,nbins)
    np.testing.assert_array_equal(image,before)


@pytest.mark.parametrize('counts_dtype',[np.int64,np.float32,np.float64])
@pytest.mark.parametrize('centers_dtype',[np.int64,np.float32,np.float64])
@pytest.mark.parametrize('kind',['random','bimodal','plateau','trimmed','fractional','large','tiny','constant','zeros'])
def test_histogram(counts_dtype,centers_dtype,kind):
    rng=np.random.default_rng(310)
    counts=rng.integers(1,1000,67).astype(counts_dtype)
    if kind=='bimodal':counts[17:49]=0
    elif kind=='plateau':counts[8:19]=2000;counts[38:49]=2000
    elif kind=='trimmed':counts[:7]=0;counts[-9:]=0
    elif kind=='constant':counts[:]=13
    elif kind=='zeros':counts[:]=0
    elif kind=='fractional':counts=counts.astype(np.float64)/7.3
    elif kind=='large':counts=counts.astype(np.float64)*1e20
    elif kind=='tiny':counts=counts.astype(np.float64)*1e-30
    centers=np.arange(counts.size,dtype=centers_dtype)
    if centers_dtype!=np.int64:centers=centers/centers_dtype(13)-centers_dtype(7)
    before=counts.copy()
    check(hist=(counts,centers))
    np.testing.assert_array_equal(counts,before)


@pytest.mark.parametrize('limit',[1,2,3,7,25,100,10000,np.int64(10000),True])
@pytest.mark.parametrize('counts',[[0,2,5,2,0,0,1,5,2,0],[1,2,3],[2,0,3,0,4,0,3],[7],[0,0,0]])
def test_iteration_boundary(limit,counts):
    check(hist=np.array(counts),max_num_iter=limit)


@pytest.mark.parametrize('bins',[3,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,257])
@pytest.mark.parametrize('seed',[2,3,19])
def test_smoothing_packet_edges(bins,seed):
    counts=np.random.default_rng(seed).integers(1,10000,bins).astype(np.float32)
    check(hist=counts)


@pytest.mark.parametrize('layout',['reversed','stepped','endian','unaligned','readonly','tagged'])
@pytest.mark.parametrize('dtype',[np.uint64,np.int64,np.float32,np.float64])
def test_histogram_storage(dtype,layout):
    counts=np.random.default_rng(711).integers(1,10000,129).astype(dtype)
    centers=np.arange(counts.size,dtype=np.float64)/13-7
    if layout=='reversed':counts=counts[::-1];centers=centers[::-1]
    elif layout=='stepped':counts=counts[::2];centers=centers[::2]
    elif layout=='endian':counts=counts.astype(counts.dtype.newbyteorder('>'))
    elif layout=='unaligned':
        storage=np.full(counts.nbytes+2,0xa5,dtype=np.uint8)
        view=np.ndarray(counts.shape,dtype=counts.dtype,buffer=storage,offset=1)
        view[:]=counts;counts=view
    elif layout=='readonly':counts.flags.writeable=False;centers.flags.writeable=False
    elif layout=='tagged':
        counts=counts.view(np.dtype(counts.dtype,metadata={'source':'histogram'}))
        centers=centers.view(np.dtype(centers.dtype,metadata={'unit':'intensity'}))
    saved=counts.copy()
    check(hist=(counts,centers))
    np.testing.assert_array_equal(counts,saved)
    if layout=='unaligned':assert storage[0]==storage[-1]==0xa5


@pytest.mark.parametrize('seed',[3,17,59])
@pytest.mark.parametrize('scale',[1.,1e-30,1e30])
def test_histogram_dynamic_range(seed,scale):
    rng=np.random.default_rng(seed)
    # Wide count ratios and plateaus stress the ordered rolling accumulator;
    # a reassociated three-term sum can change a selected plateau endpoint.
    counts=np.ldexp(rng.uniform(.5,1.,257),rng.integers(-24,25,257))*scale
    check(hist=counts)


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_parallel_image_and_pipeline(dtype):
    image=np.random.default_rng(771).integers(0,48,(513,517),dtype=np.uint8)
    image[::2]+=np.uint8(200)
    if dtype!=np.uint8:image=image.astype(dtype)/dtype(255)
    image.flags.writeable=False
    check(image)
    def pipeline(backend):return image>backend.filters.threshold_minimum(image)
    np.testing.assert_array_equal(pipeline(xx),pipeline(skimage))


def test_histogram_precedence():
    check(object(),nbins=object(),hist=np.array([9,5,1,0,1,5,9,5,1]))


def test_missing_input():
    check()


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('nbins',[2,7,256])
def test_edges_neighbors(dtype,nbins):
    edges=np.linspace(-1.3,2.7,nbins+1,dtype=dtype)
    image=np.concatenate([edges,np.nextafter(edges,dtype(-np.inf)),np.nextafter(edges,dtype(np.inf))])
    check(np.clip(image,edges[0],edges[-1]),nbins)


@pytest.mark.parametrize('kwargs,delegated',[
    ({'image':np.arange(9,dtype=np.int16)},True),
    ({'image':np.array([np.nan,1.,2.])},False),
    ({'hist':np.array([1.,np.inf,2.])},True),
    ({'hist':np.array([2,0,3,0,2]),'max_num_iter':0},True),
    ({'hist':np.array([2,0,3,0,2]),'max_num_iter':-1},True),
    ({'hist':np.array([2,0,3,0,2]),'max_num_iter':1.5},True),
])
def test_explicit_compatibility(kwargs,delegated):
    # Unsupported domains stay visibly delegated until their native contract is
    # implemented. This test is not counted as native coverage.
    xx.set_fallback_enabled(True)
    check(**kwargs,_fallback_warning=delegated)
