import warnings

import numpy as np
import pytest

from hough_ellipse_reference import frozen_rows, restore, result_record, snapshot


@pytest.mark.parametrize('reference', [row for row in frozen_rows() if row['scope'] not in ('error', 'compatibility')],
                         ids=lambda row: row['id'])
def test_hough_ellipse_matches_exact_public_reference(reference):
    from skimagexx.transform import hough_ellipse

    image = restore(np, reference)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        result = hough_ellipse(image, **reference['options'])
    actual = result_record(np, result)
    assert actual == reference['result']
    assert not emitted
    assert not np.shares_memory(result, image)
    assert snapshot(image, content=True) == reference['image']
