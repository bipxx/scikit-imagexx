"""Public label aliases execute the tested shared-core pipeline."""
import inspect
import warnings
import numpy as np
import pytest
from skimage import measure as reference
import skimage as original
import skimagexx as xx
from skimagexx import _components as implementation
from skimagexx.coverage import native_functions


def test_public_label_alias_signature_and_shared_core():
    assert {'measure.label','morphology.label'} <= set(native_functions())
    assert xx.measure.label is xx.morphology.label is implementation.label
    assert xx.measure.label is not reference.label
    assert inspect.signature(xx.measure.label)==inspect.signature(reference.label)
    assert implementation._core is xx._core


@pytest.mark.parametrize('workers',[1,4])
def test_backend_swapped_labeling_pipeline(workers):
    image=(np.random.default_rng(551).integers(0,7,(23,29))*5).astype(np.uint8)
    before=image.tobytes();previous=xx.set_fallback_enabled(False);threads=xx.get_num_threads()
    try:
        xx.set_num_threads(workers)
        def pipeline(backend):
            footprint=backend.morphology.footprint_rectangle((3,3))
            mask=backend.morphology.opening(image,footprint)>10
            return backend.measure.label(mask,connectivity=2,return_num=True)
        wanted=pipeline(original);actual=pipeline(xx)
        np.testing.assert_array_equal(actual[0],wanted[0])
        assert actual[0].dtype==wanted[0].dtype and type(actual[1]) is int and actual[1]==wanted[1]
        assert image.tobytes()==before and xx.get_num_threads()==workers
    finally:
        xx.set_fallback_enabled(previous);xx.set_num_threads(threads)


@pytest.mark.parametrize('dtype',[np.complex64,np.complex128,np.longdouble,object,'U3','S3'])
def test_label_unported_dtype_uses_explicit_compatibility(dtype):
    image=np.array([[0,1],[1,0]],dtype=dtype)
    previous=xx.set_fallback_enabled(True)
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');expected=reference.label(image)
        expected_notices=[(type(w.message),str(w.message)) for w in notices]
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');actual=xx.measure.label(image)
        fallback=[w for w in notices if isinstance(w.message,xx.FallbackWarning)]
        assert len(fallback)==1
        assert [(type(w.message),str(w.message)) for w in notices if not isinstance(w.message,xx.FallbackWarning)]==expected_notices
        np.testing.assert_array_equal(actual,expected)
        xx.set_fallback_enabled(False)
        with pytest.raises(NotImplementedError,match='fallback is disabled'):xx.measure.label(image)
    finally:
        xx.set_fallback_enabled(previous)
