"""Threshold selectors and histogram boundaries versus the pinned original."""
import numpy as np
import pytest
from skimage import filters as original
import skimagexx as xx

NAMES = ('threshold_yen', 'threshold_triangle', 'threshold_isodata')

@pytest.fixture(autouse=True, params=[1,4])
def strict(request):
    previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
@pytest.mark.parametrize('nbins',[2,7,32,256,513])
@pytest.mark.parametrize('layout',['flat','volume','reversed','fortran','constant','skewed'])
def test_image_reference(name,dtype,nbins,layout):
    rng=np.random.default_rng(59)
    a=rng.integers(0,256,(7,19,23),dtype=np.uint8)
    if dtype!=np.uint8:
        a=(a.astype(dtype)/dtype(255)*dtype(7)-dtype(3))
    if layout=='flat': a=a.ravel()
    if layout=='reversed': a=a[::-1,::2,::-1]
    if layout=='fortran': a=np.asfortranarray(a)
    if layout=='constant': a=np.full((3,17),dtype(13),dtype=dtype)
    if layout=='skewed': a.ravel()[:a.size*9//10]=dtype(0)
    try:expected=getattr(original,name)(a,nbins)
    except (IndexError,ValueError) as error:
        with pytest.raises(type(error)):getattr(xx.filters,name)(a,nbins)
        return
    actual=getattr(xx.filters,name)(a,nbins)
    assert np.asarray(actual).dtype==np.asarray(expected).dtype
    np.testing.assert_array_equal(actual,expected)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('nbins',[2,7,256])
def test_edges_neighbors(name,dtype,nbins):
    edges=np.linspace(dtype(-1.3),dtype(2.7),nbins+1,dtype=dtype)
    a=np.concatenate([edges,np.nextafter(edges,dtype(-np.inf)),np.nextafter(edges,dtype(np.inf))])
    a=np.clip(a,edges[0],edges[-1])
    try:expected=getattr(original,name)(a,nbins)
    except (IndexError,ValueError) as error:
        with pytest.raises(type(error)):getattr(xx.filters,name)(a,nbins)
        return
    actual=getattr(xx.filters,name)(a,nbins)
    np.testing.assert_array_equal(actual,expected)

@pytest.mark.parametrize('name',['threshold_yen','threshold_isodata'])
@pytest.mark.parametrize('dtype',[np.int64,np.float32,np.float64])
@pytest.mark.parametrize('counts_kind',['random','zeros','constant','large','fractional'])
@pytest.mark.parametrize('return_all',[False,True])
def test_histogram_reference(name,dtype,counts_kind,return_all):
    counts=np.random.default_rng(33).integers(1,1000,257).astype(np.float32)
    if counts_kind=='zeros': counts[:31]=0;counts[-9:]=0;counts[45:100]=0
    if counts_kind=='constant': counts[:]=13
    if counts_kind=='large': counts*=np.float32(1e6)
    if counts_kind=='fractional': counts/=np.float32(7.3)
    centers=np.arange(257,dtype=dtype)
    if dtype!=np.int64: centers=centers/dtype(13)-dtype(7)
    options={'hist':(counts,centers)}
    if name=='threshold_isodata':options['return_all']=return_all
    actual=getattr(xx.filters,name)(**options)
    expected=getattr(original,name)(**options)
    assert np.asarray(actual).dtype==np.asarray(expected).dtype
    np.testing.assert_array_equal(actual,expected)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_large_parallel_and_readonly(name,dtype):
    a=np.random.default_rng(9).integers(0,256,(513,517),dtype=np.uint8)
    if dtype!=np.uint8:a=(a/255).astype(dtype)
    a.flags.writeable=False
    np.testing.assert_array_equal(getattr(xx.filters,name)(a),getattr(original,name)(a))

@pytest.mark.parametrize('name',NAMES)
def test_pipeline_swap(name):
    a=np.random.default_rng(61).integers(0,256,(37,43),dtype=np.uint8)
    def pipeline(backend):return a>getattr(backend.filters,name)(a)
    import skimage
    np.testing.assert_array_equal(pipeline(xx),pipeline(skimage))


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('kind',['constant','two-valued','tiny-counts','large-counts'])
def test_warning_contract(name,kind):
    import warnings
    a=np.full((7,11),13,dtype=np.float32)
    options={}
    if kind=='two-valued':a.flat[-1]=14
    if kind in ('tiny-counts','large-counts'):
        if name=='threshold_triangle':return
        options={'hist':np.array([1e30,1,1,1],dtype=np.float32) if kind=='large-counts' else np.array([1,1e-30,1e-30,1e-30],dtype=np.float32)}
    def execute(module):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:result=getattr(module,name)(a,**options);error=None
            except (IndexError,ValueError,FloatingPointError) as e:result=None;error=type(e)
        return result,error,[(type(w.message),str(w.message)) for w in caught]
    with np.errstate(under='ignore',over='ignore'):
        actual,error,notices=execute(xx.filters)
        expected,expected_error,expected_notices=execute(original)
    assert error==expected_error
    assert notices==expected_notices
    if error is None:np.testing.assert_array_equal(actual,expected)

@pytest.mark.parametrize('name',['threshold_yen','threshold_isodata'])
def test_errstate_raise(name):
    a=np.full((7,11),13,dtype=np.float32)
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError):getattr(original,name)(a)
        with pytest.raises(FloatingPointError):getattr(xx.filters,name)(a)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_empty_errors_or_result(name,dtype):
    a=np.empty((0,3),dtype=dtype)
    try:expected=getattr(original,name)(a)
    except (IndexError,ValueError) as error:
        with pytest.raises(type(error)):getattr(xx.filters,name)(a)
        return
    np.testing.assert_array_equal(getattr(xx.filters,name)(a),expected)

@pytest.mark.parametrize('name',['threshold_yen','threshold_isodata'])
@pytest.mark.parametrize('bins',[1,7,257])
def test_zero_histogram(name,bins):
    h=np.zeros(bins,dtype=np.float32)
    try:expected=getattr(original,name)(hist=h)
    except (IndexError,ValueError) as error:
        with pytest.raises(type(error)):getattr(xx.filters,name)(hist=h)
        return
    np.testing.assert_array_equal(getattr(xx.filters,name)(hist=h),expected)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_unaligned_views_and_scalar(name,dtype):
    a=np.ndarray((31,37),dtype=dtype,buffer=bytearray(31*37*np.dtype(dtype).itemsize+1),offset=1)
    a[:]=np.random.default_rng(20).random(a.shape)
    assert not a.flags.aligned
    np.testing.assert_array_equal(getattr(xx.filters,name)(a),getattr(original,name)(a))

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_histogram_counts_match_exactly(dtype):
    from skimage.exposure import histogram
    from skimagexx.filters import _threshold_histogram
    rng=np.random.default_rng(18)
    for bins in (2,7,256,513):
        a=(rng.normal(size=(129,133))*7+20).astype(dtype)
        actual,centers,_=_threshold_histogram(a,None,bins)
        expected,reference_centers=histogram(a,nbins=bins)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(centers,reference_centers)


def test_yen_log_rounding_changes_threshold():
    from pathlib import Path
    counts=np.loadtxt(Path(__file__).parent/'data/yen-log-rounding-counts.txt',dtype=np.float32)
    expected=original.threshold_yen(hist=counts)
    assert xx.filters.threshold_yen(hist=counts)==expected


@pytest.mark.parametrize('seed',range(40))
def test_random_histogram_dynamic_ranges(seed):
    rng=np.random.default_rng(seed)
    n=int(rng.integers(2,600))
    counts=rng.integers(1,1000,n).astype(np.float32)*np.exp2(rng.integers(-15,35,n)).astype(np.float32)
    for dtype in (np.int64,np.float32,np.float64):
        centers=np.arange(n,dtype=dtype)
        for name in ('threshold_yen','threshold_isodata'):
            options={'hist':(counts,centers)}
            if name=='threshold_isodata':options['return_all']=True
            np.testing.assert_array_equal(getattr(xx.filters,name)(**options),getattr(original,name)(**options))

@pytest.mark.parametrize('name',['threshold_yen','threshold_isodata'])
def test_unaligned_histogram_metadata(name):
    counts=np.ndarray(31,dtype=np.float64,buffer=bytearray(31*8+1),offset=1)
    centers=np.ndarray(31,dtype=np.float64,buffer=bytearray(31*8+1),offset=1)
    counts[:]=np.arange(31)+1;centers[:]=np.arange(31)/13
    options={'hist':(counts,centers)}
    np.testing.assert_array_equal(getattr(xx.filters,name)(**options),getattr(original,name)(**options))


@pytest.mark.parametrize('dtype',[np.uint64,np.int64])
def test_wide_count_conversion_matches_upstream(dtype):
    rng=np.random.default_rng(84)
    counts=(rng.integers(1,1000,257,dtype=np.uint64)*np.uint64(1<<50)+np.uint64((1<<26)+1)).astype(dtype)
    for name in ('threshold_yen','threshold_isodata'):
        options={'hist':counts}
        if name=='threshold_isodata':options['return_all']=True
        np.testing.assert_array_equal(getattr(xx.filters,name)(**options),getattr(original,name)(**options))

@pytest.mark.parametrize('name',['threshold_yen','threshold_isodata'])
def test_histogram_cast_underflow_error(name):
    counts=np.array([1e-46,2e-46,3e-46],dtype=np.float64)
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError,match='underflow'):getattr(original,name)(hist=counts)
        with pytest.raises(FloatingPointError,match='underflow'):getattr(xx.filters,name)(hist=counts)


@pytest.mark.parametrize('kind',['large-offset','overflow-width'])
def test_isodata_integer_center_width(kind):
    import warnings
    centers=np.arange(7,dtype=np.int64)+(1<<60) if kind=='large-offset' else np.array([np.iinfo(np.int64).min,np.iinfo(np.int64).max])
    counts=np.ones(centers.size,dtype=np.float32)
    def evaluate(module):
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            result=module.threshold_isodata(hist=(counts,centers),return_all=True)
        return result,[(type(w.message),str(w.message)) for w in notices]
    actual,notices=evaluate(xx.filters);expected,reference_notices=evaluate(original)
    np.testing.assert_array_equal(actual,expected)
    assert notices==reference_notices


def test_yen_log_provider_rounding_across_exponents():
    from skimagexx.filters import _yen_log_mode
    from skimagexx import _core
    bits=np.random.default_rng(60).integers(1,0x7f800000,4096,dtype=np.uint32)
    values=bits.view(np.float32)
    mode=_yen_log_mode()
    assert mode is not None
    np.testing.assert_array_equal(_core.threshold_logarithms(values,mode),np.log(values))
