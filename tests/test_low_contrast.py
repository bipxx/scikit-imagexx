"""Test-first native low-contrast contract against original scikit-image."""
import inspect
import re
import warnings
import numpy as np
import pytest
import skimage
from skimage import exposure as original
from skimage.util import dtype_limits
import skimagexx as xx
from test_integral import REAL, sample


@pytest.fixture(autouse=True, params=[1, 4])
def configuration(request):
    previous=xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


def check(image, allow_fallback=False, **options):
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always')
        try:
            expected=original.is_low_contrast(image, **options)
        except Exception as error:
            with pytest.raises(type(error), match=re.escape(str(error))):
                xx.exposure.is_low_contrast(image, **options)
            return
    snapshot=np.array(image, copy=True)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        actual=xx.exposure.is_low_contrast(image, **options)
    if allow_fallback:
        fallback=[w for w in notices if isinstance(w.message, xx.FallbackWarning)]
        assert len(fallback)<=1
        notices=[w for w in notices if w not in fallback]
    assert [(type(w.message), str(w.message)) for w in notices]==[
        (type(w.message), str(w.message)) for w in expected_notices]
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, snapshot)


def test_signature_and_backend_swap():
    assert inspect.signature(xx.exposure.is_low_contrast)==inspect.signature(original.is_low_contrast)
    image=np.random.default_rng(9785).random((43, 29, 3))
    def pipeline(backend):
        gray=backend.color.rgb2gray(image)
        scaled=backend.exposure.rescale_intensity(gray, out_range=(0., .04))
        return backend.exposure.is_low_contrast(scaled)
    assert pipeline(xx)==pipeline(skimage)


@pytest.mark.parametrize('dtype', REAL)
@pytest.mark.parametrize('layout', ['C', 'F', 'reverse', 'step', 'endian', 'unaligned', 'broadcast'])
@pytest.mark.parametrize('percentiles', [(1, 99), (0, 100), (99, 1), (49.25, 50.75)])
def test_real_layouts(dtype, layout, percentiles):
    image=sample(dtype, (19, 23))
    if layout=='F': image=np.asfortranarray(image)
    elif layout=='reverse': image=image[::-1, ::-1]
    elif layout=='step': image=image[::2, 1::3]
    elif layout=='endian': image=image.astype(image.dtype.newbyteorder('>'))
    elif layout=='unaligned':
        target=np.ndarray(image.shape, dtype=image.dtype, buffer=bytearray(image.nbytes+1), offset=1)
        target[:]=image
        image=target
    elif layout=='broadcast': image=np.broadcast_to(image[:1], image.shape)
    check(image, lower_percentile=percentiles[0], upper_percentile=percentiles[1])


@pytest.mark.parametrize('dtype', REAL)
@pytest.mark.parametrize('shape', [(), (1,), (0,), (0, 3), (1, 3, 1, 2), (2, 1, 3, 5)])
def test_shapes(dtype, shape):
    check(sample(dtype, shape))


@pytest.mark.parametrize('dtype', REAL)
def test_metadata_and_memmap(dtype, tmp_path):
    image=sample(dtype, (13, 17))
    image=image.astype(np.dtype(dtype, metadata={'tag': ['low-contrast', 1]}))
    check(image)
    mapped=np.memmap(tmp_path/'contrast.dat', mode='w+', dtype=image.dtype, shape=image.shape)
    mapped[:]=image
    try: check(mapped[::-1])
    finally: del mapped


@pytest.mark.parametrize('dtype', REAL)
@pytest.mark.parametrize('percentiles', [(1, 99), (0, 100), (25, 75), (13.1, 87.7)])
def test_decision_boundary_requires_exact_interpolation(dtype, percentiles):
    if np.dtype(dtype).kind=='b':
        check(np.array([False, True]), fraction_threshold=object(),
              lower_percentile=object(), upper_percentile=object(), method=object())
        return
    image=sample(dtype, (37,))
    if np.dtype(dtype).kind in 'iu':
        limits=np.iinfo(dtype)
        image=np.array([limits.min, limits.min+1, 0, limits.max-1, limits.max], dtype=dtype)
    p0, p1=percentiles
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        low, high=np.percentile(image, [p0, p1])
        lower, upper=dtype_limits(image, clip_negative=False)
        ratio=(high-low)/(upper-lower)
    for threshold in (ratio, np.nextafter(ratio, -np.inf), np.nextafter(ratio, np.inf)):
        check(image, fraction_threshold=threshold, lower_percentile=p0, upper_percentile=p1)


@pytest.mark.parametrize('dtype', ['uint8', 'float32', 'float64'])
@pytest.mark.parametrize('channels', [3, 4])
@pytest.mark.parametrize('layout', ['C', 'F', 'reverse'])
def test_rgb_and_alpha(dtype, channels, layout):
    image=np.random.default_rng(15).random((17, 19, channels))
    image=(image*255).astype(dtype) if dtype=='uint8' else image.astype(dtype)
    if layout=='F': image=np.asfortranarray(image)
    elif layout=='reverse': image=image[::-1, ::-1]
    check(image)


@pytest.mark.parametrize('image', [True, False, [True, False], np.array([[True]]), np.zeros((2, 4), bool)])
def test_boolean_ignores_other_arguments(image):
    check(image, fraction_threshold=object(), lower_percentile=-1000,
          upper_percentile=object(), method=object())


@pytest.mark.parametrize('options', [
    {'lower_percentile': -1}, {'upper_percentile': 101},
    {'lower_percentile': np.nan}, {'upper_percentile': np.inf},
    {'method': 'not-a-method'}, {'method': object()},
    {'fraction_threshold': np.nan}, {'fraction_threshold': np.inf},
    {'fraction_threshold': -.5}, {'fraction_threshold': np.float32(.05)},
])
def test_validation_and_ignored_method(options):
    check(np.arange(33, dtype=np.uint8), **options)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('value', [0., -0., 1., -1., 1e-5])
def test_constant_and_subnormal(dtype, value):
    check(np.full((7, 13), value, dtype=dtype))


@pytest.mark.parametrize('dtype', ['uint8', 'float32', 'float64'])
def test_parallel_sized_image(dtype):
    image=sample(dtype, (517, 521))
    check(image)
    check(image[::-1, ::-1], lower_percentile=2.5, upper_percentile=97.5)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_exceptional_ranges_have_explicit_fallback(dtype, value):
    image=np.array([0, 1, value], dtype=dtype)
    with pytest.raises(NotImplementedError, match='outside native coverage'):
        xx.exposure.is_low_contrast(image)
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always')
        expected=original.is_low_contrast(image)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        actual=xx.exposure.is_low_contrast(image)
    assert type(actual) is type(expected) and actual==expected
    fallback=[w for w in notices if isinstance(w.message, xx.FallbackWarning)]
    assert len(fallback)==1
    assert [(type(w.message), str(w.message)) for w in notices if w not in fallback]==[
        (type(w.message), str(w.message)) for w in expected_notices]


def test_readonly_input_and_array_thresholds():
    image=np.linspace(0, .04, 131, dtype=np.float32)
    image.flags.writeable=False
    check(image)
    check(image, fraction_threshold=np.array([0., .01, .1, 1.]))


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
def test_quantile_scalar_precision(dtype):
    image=np.random.default_rng(99).random(131).astype(dtype)
    for quantile_type in (np.float16, np.float32, np.float64):
        check(image, lower_percentile=quantile_type(1.1), upper_percentile=quantile_type(97.3))


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('position', [0, 7, 8, 15, 16, 31, 32, 64, 65])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_finite_scan_packet_and_tail(dtype, position, value):
    image=np.linspace(0, 1, 67, dtype=dtype)
    image[position]=value
    with pytest.raises(NotImplementedError, match='outside native coverage'):
        xx.exposure.is_low_contrast(image)


@pytest.mark.parametrize('count', [1, 7, 8, 15, 16, 31, 32, 33, 64, 65])
def test_noncanonical_boolean_storage(count):
    raw=np.full(count, 255, dtype=np.uint8)
    check(raw.view(np.bool_))
    raw[-1]=0
    check(raw.view(np.bool_))
    raw[:]=0
    raw[0]=2
    check(raw.view(np.bool_))


@pytest.mark.parametrize('word,dtype,bits', [
    ('uint16', 'float16', 0x7c01), ('uint16', 'float16', 0x7e01),
    ('uint32', 'float32', 0x7f800001), ('uint32', 'float32', 0x7fc00001),
    ('uint64', 'float64', 0x7ff0000000000001), ('uint64', 'float64', 0x7ff8000000000001),
])
def test_nan_payload_fallback_warning_parity(word, dtype, bits):
    special=np.array([bits], dtype=word).view(dtype)[0]
    image=np.linspace(0, 1, 67, dtype=dtype)
    image[-1]=special
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings(record=True) as expected_notices:
        warnings.simplefilter('always')
        expected=original.is_low_contrast(image)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        actual=xx.exposure.is_low_contrast(image)
    assert type(actual) is type(expected) and actual==expected
    fallback=[w for w in notices if isinstance(w.message, xx.FallbackWarning)]
    assert len(fallback)==1
    assert [(type(w.message), str(w.message)) for w in notices if w not in fallback]==[
        (type(w.message), str(w.message)) for w in expected_notices]


def test_callback_policy_is_delegated_before_private_arithmetic():
    image=np.array([-np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
    previous=np.geterrcall()
    expected_calls=[]
    actual_calls=[]
    xx.set_fallback_enabled(True)
    try:
        np.seterrcall(lambda *args: expected_calls.append(args))
        with np.errstate(all='call'):
            expected=original.is_low_contrast(image)
        np.seterrcall(lambda *args: actual_calls.append(args))
        with np.errstate(all='call'), pytest.warns(xx.FallbackWarning):
            actual=xx.exposure.is_low_contrast(image)
    finally:
        np.seterrcall(previous)
    assert expected_calls and actual_calls==expected_calls
    assert type(actual) is type(expected) and actual==expected


@pytest.mark.parametrize('length', [65503, 65504, 65505, 65536, 70001])
@pytest.mark.parametrize('upper', [97.3,100])
def test_low_precision_percentile_index_overflow(length,upper):
    image=np.linspace(0, .04, length, dtype=np.float32)
    xx.set_fallback_enabled(True)
    check(image, allow_fallback=True, lower_percentile=np.float16(1.1),
          upper_percentile=np.float16(upper))


@pytest.mark.parametrize('dtype', REAL)
@pytest.mark.parametrize('length', [63,64,65,127,128,129,255,256,257,511,513])
@pytest.mark.parametrize('percentiles', [(1,99),(49.25,50.75),(13.1,87.7)])
def test_selection_packets_have_exact_decision_boundaries(dtype,length,percentiles):
    image=sample(dtype,(length,),seed=139)
    if image.dtype.kind=='b':
        check(image)
        return
    limits=dtype_limits(image,clip_negative=False)
    low,high=np.percentile(image,percentiles)
    ratio=(high-low)/(limits[1]-limits[0])
    for threshold in (ratio,np.nextafter(ratio,-np.inf),np.nextafter(ratio,np.inf)):
        check(image,lower_percentile=percentiles[0],upper_percentile=percentiles[1],
              fraction_threshold=threshold)


@pytest.mark.parametrize('dtype', ['uint16','int16','uint32','int32','uint64','int64','float16','float32','float64'])
@pytest.mark.parametrize('pattern', ['constant','ascending','descending','organ-pipe','alternating','one-outlier'])
def test_duplicate_and_ordered_selection(dtype,pattern):
    descriptor=np.dtype(dtype)
    if pattern=='constant': image=np.full(513,7,dtype=dtype)
    elif pattern=='ascending': image=(np.arange(513)%255).astype(dtype)
    elif pattern=='descending': image=(np.arange(513)%255).astype(dtype)[::-1]
    elif pattern=='organ-pipe': image=np.minimum(np.arange(513),np.arange(513)[::-1]).astype(dtype)
    elif pattern=='alternating':
        if descriptor.kind in 'iu':
            info=np.iinfo(dtype);image=np.array([info.min]*257+[info.max]*256,dtype=dtype)
        else:image=np.array([-7.]*257+[7.]*256,dtype=dtype)
    else:
        image=np.zeros(513,dtype=dtype);image[-1]=9
    bounds=dtype_limits(image,clip_negative=False)
    low,high=np.percentile(image,[49.75,50.25])
    ratio=(high-low)/(bounds[1]-bounds[0])
    for threshold in (ratio,np.nextafter(ratio,np.inf)):
        check(image,lower_percentile=49.75,upper_percentile=50.25,
              fraction_threshold=threshold)


@pytest.mark.parametrize('dtype',REAL)
@pytest.mark.parametrize('percentiles',[(1.1,98.9),(49.9,50.1)])
def test_parallel_copy_has_exact_decision_boundaries(dtype,percentiles):
    image=sample(dtype,(262145,),seed=891)
    if image.dtype.kind=='b':
        check(image)
        return
    low,high=np.percentile(image,percentiles)
    lower,upper=dtype_limits(image,clip_negative=False)
    ratio=(high-low)/(upper-lower)
    for threshold in (ratio,np.nextafter(ratio,-np.inf),np.nextafter(ratio,np.inf)):
        check(image,lower_percentile=percentiles[0],upper_percentile=percentiles[1],
              fraction_threshold=threshold)
