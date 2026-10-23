"""Real registered footprint functions compose with the native image API."""
import inspect
import numpy as np
import pytest
from skimage import morphology as reference
import skimagexx as xx
from skimagexx import _footprint_dense as dense
from skimagexx import _footprint_decomposition as decomposition
from skimagexx import _footprint_util as utility
from skimagexx.coverage import native_functions


@pytest.mark.parametrize('name',['footprint_rectangle','diamond','disk','ball','mirror_footprint','pad_footprint'])
def test_public_footprint_identity_and_shared_core(name):
    implementation=utility if name in ('mirror_footprint','pad_footprint') else decomposition
    assert 'morphology.'+name in native_functions()
    assert getattr(xx.morphology,name) is getattr(implementation,name)
    assert getattr(xx.morphology,name) is not getattr(reference,name)
    assert inspect.signature(getattr(xx.morphology,name))==inspect.signature(getattr(reference,name))
    assert dense._core is decomposition._core is utility._core is xx._core


@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_backend_swapped_footprints_and_morphology(workers,dtype):
    image=(np.arange(17*19).reshape(17,19)%23).astype(dtype)
    old_fallback=xx.set_fallback_enabled(False);old_workers=xx.get_num_threads()
    try:
        xx.set_num_threads(workers)
        def pipeline(backend):
            rectangle=backend.footprint_rectangle((3,5),dtype=np.uint8,decomposition='sequence')
            reflected=backend.mirror_footprint(rectangle)
            padded=backend.pad_footprint(backend.disk(2,dtype=np.uint8))
            return backend.dilation(backend.erosion(image,reflected),padded)
        np.testing.assert_array_equal(pipeline(xx.morphology),pipeline(reference))
        assert xx.get_num_threads()==workers
    finally:
        xx.set_num_threads(old_workers);xx.set_fallback_enabled(old_fallback)
