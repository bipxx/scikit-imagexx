import sys

import numpy as np
import pytest

from hough_circle_peaks_reference import cases, circle_fields, observe, reference


PROFILE = 'windows' if sys.platform == 'win32' else 'linux-minimum'


@pytest.fixture(scope='module')
def prepared_circle_fields():
    from skimage.transform import hough_circle

    return dict(circle_fields(np, hough_circle))


@pytest.mark.parametrize('case_id', [row['id'] for row in reference()['profiles'][PROFILE]])
def test_hough_circle_peaks_exact_public(case_id, prepared_circle_fields):
    from skimagexx import set_fallback_enabled
    from skimagexx.transform import hough_circle_peaks

    args, row = next((args, row) for args, row in cases(np, prepared_circle_fields, PROFILE) if row['id'] == case_id)
    compat = row['scope'] in ('error', 'compatibility')
    previous = set_fallback_enabled(compat)
    try:
        observe(hough_circle_peaks, np, args, row, fallback_warning=compat)
    finally:
        set_fallback_enabled(previous)
