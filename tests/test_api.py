"""Public interchangeability, native dispatch, and per-function reference tests."""
import inspect
import numpy as np
import pytest
import skimage
from skimage import filters as reference_filters, color as reference_color, util as reference_util
import skimagexx as xx
from skimagexx.coverage import native_functions, native_dtypes, needs_color_input, default_call, assert_default_parity, DERIVED_MOMENTS_FUNCTIONS,COORDS_MOMENTS_FUNCTIONS, native_classes, class_metadata_providers
from skimage import exposure as reference_exposure
from skimage import morphology as reference_morphology
from skimage import transform as reference_transform
from skimage import feature as reference_feature
from skimage import metrics as reference_metrics, measure as reference_measure
from skimage import segmentation as reference_segmentation
from skimage import restoration as reference_restoration
from skimage.filters import rank as reference_rank
from skimage.filters.rank import generic as reference_rank_generic
from skimage.filters.rank import bilateral as reference_rank_bilateral
from skimagexx.filters import rank as native_rank
from skimagexx.filters.rank import generic as native_rank_generic
from skimagexx.filters.rank import bilateral as native_rank_bilateral

MODULES = [(xx.filters, reference_filters), (xx.color, reference_color), (xx.util, reference_util),
           (xx.exposure, reference_exposure), (xx.morphology,reference_morphology), (xx.transform,reference_transform),
           (xx.feature,reference_feature), (xx.metrics,reference_metrics), (xx.measure,reference_measure),
           (xx.segmentation,reference_segmentation), (xx.restoration,reference_restoration),
           (native_rank,reference_rank),(native_rank_generic,reference_rank_generic),
           (native_rank_bilateral,reference_rank_bilateral)]
FUNCTIONS = [(native, original, name) for native, original in MODULES for name in native.__all__
             if not inspect.isclass(getattr(native,name))]
# Pinned generic exposes these callable identity aliases outside star exports.
# Exercise signatures and values independently of the advertised __all__ list.
FUNCTIONS += [(native_rank_generic, reference_rank_generic, name)
              for name in ('sum', 'windowed_histogram', 'majority') if name not in native_rank_generic.__all__]
CLASSES = [(native,original,name) for native,original in MODULES for name in native.__all__
           if inspect.isclass(getattr(native,name))]

# The original Bregman Cython helper has unchecked halo reads for small axes.
# Use separately reviewed geometry for Bregman and max-tree APIs; their contracts retain
# unsafe geometry as explicitly blocked intake metadata, never original calls.
VALUE_CASES = [
    (native, original, name, dtype, shape)
    for native, original, name in FUNCTIONS
    for dtype in (np.uint8, np.float32, np.float64)
    for shape in (((4, 5), (7, 19), (519, 521))
                  if ((native is xx.restoration and name == 'denoise_tv_bregman')
                      or (native is xx.morphology and name in ('max_tree','max_tree_local_maxima',
                          'area_opening','area_closing','diameter_opening','diameter_closing')))
                  else ((1, 1), (3, 19), (519, 521)))
]


def test_registry_matches_per_function_tests():
    tested = {module.__name__.split('.', 1)[1]+'.'+name for module, _, name in FUNCTIONS}
    assert tested == set(native_functions())
    tested_classes={module.__name__.split('.',1)[1]+'.'+name for module,_,name in CLASSES}
    assert tested_classes==set(native_classes())
    assert set(class_metadata_providers())=={'util.FailedEstimationAccessError'}
    assert inspect.isclass(xx.util.FailedEstimationAccessError)


def test_unported_namespace_interchangeability():
    a = np.zeros((9, 11), dtype=bool)
    a[3:6, 4:7] = True
    with pytest.raises(NotImplementedError):
        xx.morphology.convex_hull_image(a)
    xx.set_fallback_enabled(True)
    from skimage import morphology
    with pytest.warns(xx.FallbackWarning):
        actual = xx.morphology.convex_hull_image(a)
    np.testing.assert_array_equal(actual, morphology.convex_hull_image(a))


@pytest.fixture(autouse=True, params=[1, 4])
def configuration(request):
    previous = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


@pytest.mark.parametrize('native,original,name', FUNCTIONS, ids=[x[2] for x in FUNCTIONS])
def test_signatures(native, original, name):
    assert inspect.signature(getattr(native, name)) == inspect.signature(getattr(original, name))


@pytest.mark.parametrize('native,original,name,dtype,shape', VALUE_CASES,
                         ids=[f'{n.__name__}.{name}-{np.dtype(dtype).name}-{shape}'
                              for n, _, name, dtype, shape in VALUE_CASES])
def test_every_native_function(native, original, name, dtype, shape):
    rng = np.random.default_rng(15)
    api = native.__name__.split('.', 1)[1]+'.'+name
    a = rng.integers(0, 256, (*shape, 3) if needs_color_input(api) else shape, dtype=np.uint8)
    if dtype != np.uint8:
        a = (a/255).astype(dtype)
    if np.dtype(dtype).name not in native_dtypes(api):
        if api == 'measure.moments_normalized':
            # Integer output cannot store the source's required low-order NaN.
            # Safe 4x4 geometry reaches that error; it is not numeric credit.
            args, options = default_call(api, a)
            with pytest.raises(ValueError) as expected:getattr(original, name)(*args, **options)
            with pytest.raises(ValueError) as actual:getattr(native, name)(*args, **options)
            assert str(actual.value) == str(expected.value)
            return
        if api in ('morphology.remove_small_objects','morphology.remove_small_holes','segmentation.relabel_sequential'):
            # These integer-only algorithms preserve original defined dtype
            # rejection; this is separate from successful default-case dtypes.
            with pytest.raises(TypeError) as expected:getattr(original,name)(a)
            with pytest.raises(TypeError) as actual:getattr(native,name)(a)
            assert str(actual.value)==str(expected.value)
            return
        with pytest.raises(NotImplementedError):
            getattr(native, name)(a)
        return
    args, options = default_call(api,a)
    try:
        expected = getattr(original, name)(*args,**options)
    except (IndexError,ValueError,RuntimeError) as error:
        # Some valid images have no converged threshold. Match the reference's
        # algorithm failure just as we match its successful numerical output.
        with pytest.raises(type(error)) as actual_error:
            getattr(native, name)(*args,**options)
        assert str(actual_error.value)==str(error)
        return
    actual = getattr(native, name)(*args,**options)
    if api=='segmentation.relabel_sequential':
        # Compare sparse mapping vectors without dense ArrayMap conversion.
        assert type(actual) is type(expected) is tuple and len(actual)==3
        np.testing.assert_array_equal(actual[0],expected[0])
        assert actual[0].dtype==expected[0].dtype
        for value,reference in zip(actual[1:],expected[1:]):
            assert type(value).__name__==type(reference).__name__=='ArrayMap'
            assert not type(value).__module__.startswith('skimage.')
            for field in ('in_values','out_values'):
                np.testing.assert_array_equal(getattr(value,field),getattr(reference,field))
                assert getattr(value,field).dtype==getattr(reference,field).dtype
        return
    if name=='rank_order' or api=='morphology.max_tree':
        assert type(actual) is type(expected) is tuple and len(actual)==len(expected)==2
        for value,reference in zip(actual,expected):
            assert value.dtype==reference.dtype and value.shape==reference.shape
            np.testing.assert_array_equal(value,reference)
        return
    assert np.asarray(actual).dtype == np.asarray(expected).dtype
    assert_default_parity(api,np.dtype(dtype),actual,expected,
                          image=args if api == 'measure.mesh_surface_area' else args[0] if api in DERIVED_MOMENTS_FUNCTIONS or api in COORDS_MOMENTS_FUNCTIONS or api == 'measure.perimeter' or api == 'measure.euler_number' else a)


@pytest.mark.parametrize('name', ['sobel', 'prewitt', 'scharr'])
@pytest.mark.parametrize('axis', [0, 1])
def test_gradient_axes(name, axis):
    a = np.random.default_rng(0).random((17, 23), dtype=np.float32)
    np.testing.assert_allclose(getattr(xx.filters, name)(a, axis=axis),
                               getattr(reference_filters, name)(a, axis=axis), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('axis', [-3, -2, -1, 0, 1, 2])
def test_channel_axes(axis):
    a = np.random.default_rng(0).random((13, 17))
    rgb = xx.color.gray2rgb(a, channel_axis=axis)
    np.testing.assert_array_equal(rgb, reference_color.gray2rgb(a, channel_axis=axis))
    np.testing.assert_allclose(xx.color.rgb2gray(rgb, channel_axis=axis),
                               reference_color.rgb2gray(rgb, channel_axis=axis), atol=2e-13)


@pytest.mark.parametrize('signed', [False, True])
def test_invert_signed_and_nonfinite(signed):
    a = np.array([[0.0, -0.0, np.inf, -np.inf, np.nan, -1, 2]])
    expected = reference_util.invert(a, signed)
    actual = xx.util.invert(a, signed)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_pipeline_can_swap_backend():
    a = np.random.default_rng(0).integers(0, 256, (37, 41, 3), dtype=np.uint8)
    def pipeline(backend):
        gray = backend.color.rgb2gray(a)
        smooth = backend.filters.gaussian(gray, 1.2, mode='reflect')
        edges = backend.filters.scharr(smooth, axis=0)
        return backend.morphology.opening(edges > backend.filters.threshold_mean(edges))
    np.testing.assert_array_equal(pipeline(xx), pipeline(skimage))


def test_explicit_fallback_for_unsupported_options():
    a = np.arange(81, dtype=np.uint16).reshape(9, 9)
    actual = xx.filters.gaussian(a, mode='wrap')
    np.testing.assert_allclose(actual, reference_filters.gaussian(a, mode='wrap'),rtol=2e-13,atol=2e-13)
    out = np.empty(a.shape)
    result = xx.filters.gaussian(a, out=out)
    assert result is out
    np.testing.assert_allclose(out, reference_filters.gaussian(a),rtol=2e-13,atol=2e-13)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.threshold_otsu(a, nbins=32)
    assert actual == reference_filters.threshold_otsu(a, nbins=32)


def test_upstream_adapter_obeys_fallback_policy():
    from skimagexx._compat import module_attribute
    a = np.arange(9.0).reshape(3, 3)
    # Exercise the upstream adapter directly so adding native algorithms never
    # turns this policy check into an assertion that a function must stay unported.
    upstream = module_attribute('filters', 'median')
    with pytest.raises(NotImplementedError):
        upstream(a)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        actual = upstream(a)
    np.testing.assert_array_equal(actual, reference_filters.median(a))


def test_mask_sequence_axis_and_histogram_fallback():
    xx.set_fallback_enabled(True)
    a = np.arange(81.0).reshape(9, 9)
    mask = a > 10
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.sobel(a, mask, axis=[0, 1], mode='nearest')
    np.testing.assert_array_equal(actual, reference_filters.sobel(a, mask, axis=[0, 1], mode='nearest'))
    hist = np.histogram(a, bins=16)
    hist = (hist[0], (hist[1][:-1]+hist[1][1:])/2)
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.threshold_otsu(hist=hist)
    assert actual == reference_filters.threshold_otsu(hist=hist)
    # Preserve 0.26's handling of negative axes instead of normalizing it.
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.sobel(a, axis=-1)
    np.testing.assert_array_equal(actual, reference_filters.sobel(a, axis=-1))


def test_integral_pipeline_can_swap_backend():
    image=np.random.default_rng(545).integers(0,256,(37,41,3),dtype=np.uint8)
    def pipeline(backend):
        gray=backend.color.rgb2gray(image)
        smooth=backend.filters.gaussian(gray,.75)
        return backend.transform.integral_image(smooth)
    actual,expected=pipeline(xx),pipeline(skimage)
    assert actual.dtype==expected.dtype and actual.strides==expected.strides
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
