"""Private byte-domain admission; no original algorithm or numeric tolerance."""
import numpy as np
import pytest
import skimagexx as xx


@pytest.mark.parametrize('volume', (False, True), ids=('2d', '3d'))
@pytest.mark.parametrize('value', (2, 255))
def test_euler_private_counts_refuse_nonbinary_bytes(volume, value):
    image = np.zeros((4, 4, 4) if volume else (4, 4), dtype=np.uint8)
    image[(1, 1, 1) if volume else (1, 1)] = value
    before = image.copy()
    policy = np.geterr().copy()
    workers = xx.get_num_threads()
    assert workers == 1
    with pytest.raises(ValueError, match='^Euler counts require binary byte values$'):
        xx._core.euler_counts(image, volume, workers)
    np.testing.assert_array_equal(image, before)
    assert np.geterr() == policy and xx.get_num_threads() == workers
