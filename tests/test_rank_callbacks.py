import importlib
import os

import numpy as np
import pytest


@pytest.mark.parametrize('dtype', [np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, 'datetime64[ns]', 'timedelta64[ns]', np.float16, np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize('workers', [1, 4])
def test_positive_dispatch_owns_python_metadata_under_gil(dtype, workers):
    backend = importlib.import_module(os.environ.get('SKIMAGEXX_RANK_BACKEND', 'skimagexx._core'))
    previous = backend.get_num_threads()
    try:
        backend.set_num_threads(workers)
        image = np.arange(35).reshape(5, 7).astype(dtype)
        if image.dtype.kind in 'mM':
            expected = np.where(image.view(np.int64) > 0, 255, 0).astype(np.uint8)
        else:
            expected = np.where(image > 0, 255, 0).astype(np.uint8)
        actual, invalid = backend.rank_positive(image)
        np.testing.assert_array_equal(actual, expected)
        assert invalid is False
    finally:
        backend.set_num_threads(previous)
