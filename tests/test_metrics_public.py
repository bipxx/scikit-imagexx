"""Registered public metrics execute the exact tested native implementation."""
import inspect

import numpy as np
import pytest
from skimage import metrics as reference

import skimagexx as xx
from skimagexx import _metrics as implementation
from skimagexx.coverage import native_functions


NAMES=('mean_squared_error','normalized_root_mse','peak_signal_noise_ratio')


@pytest.mark.parametrize('name',NAMES)
def test_public_metric_identity_signature_and_shared_workers(name):
    assert 'metrics.'+name in native_functions()
    assert getattr(xx.metrics,name) is getattr(implementation,name)
    assert getattr(xx.metrics,name) is not getattr(reference,name)
    assert inspect.signature(getattr(xx.metrics,name))==inspect.signature(getattr(reference,name))
    assert implementation._core is xx._core
    assert implementation._core.set_num_threads is xx.set_num_threads
    assert implementation._core.get_num_threads is xx.get_num_threads
    before=xx.get_num_threads()
    try:
        xx.set_num_threads(4)
        assert implementation._core.get_num_threads()==4
    finally:
        xx.set_num_threads(before)


@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_public_backend_swap_metrics_pipeline_and_memmap(tmp_path,workers,dtype):
    image=np.memmap(tmp_path/'image.bin',mode='w+',dtype=dtype,shape=(17,19))
    values=np.arange(image.size).reshape(image.shape)%17
    image[:]=values if dtype is np.uint8 else values/17
    image.flush();image.flags.writeable=False
    second=image[::-1,::-1]
    before=image.tobytes()
    old_fallback=xx.set_fallback_enabled(False);old_workers=xx.get_num_threads()
    try:
        xx.set_num_threads(workers)
        def pipeline(backend):
            return (backend.mean_squared_error(image,second),
                    backend.normalized_root_mse(image,second),
                    backend.peak_signal_noise_ratio(image,second,data_range=255 if dtype is np.uint8 else 1))
        expected=pipeline(reference);actual=pipeline(xx.metrics)
        assert all(type(x) is type(y) is np.float64 for x,y in zip(actual,expected))
        np.testing.assert_allclose(actual,expected,rtol=3e-6 if dtype is np.float32 else 3e-13,atol=0)
        assert image.tobytes()==before and not image.flags.writeable
    finally:
        xx.set_num_threads(old_workers);xx.set_fallback_enabled(old_fallback)
