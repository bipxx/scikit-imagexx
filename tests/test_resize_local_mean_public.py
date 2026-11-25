import sys

import numpy as np
import pytest

from resize_local_mean_reference import COUNTEREXAMPLES, image_for, observe, reference


PROFILE = 'windows' if sys.platform == 'win32' else 'linux-minimum'
ROWS = reference()[PROFILE]['observations']


@pytest.mark.parametrize('case_id', [name for name, info in ROWS.items() if info['native'] or name in COUNTEREXAMPLES])
def test_resize_local_mean_exact_public(case_id):
    import skimagexx as xx
    from skimagexx.transform import resize_local_mean

    row = ROWS[case_id]
    image = image_for(np, row['case'])
    compat = not row['native']
    previous = xx.set_fallback_enabled(compat)
    try:
        if row['native']:
            observe(np, resize_local_mean, image, row, compat=False)
            return
        import warnings
        from skimage.transform import resize_local_mean as stock_resize_local_mean
        from resize_local_mean_reference import array_details, digest

        case = row['case']
        with warnings.catch_warnings(record=True) as stock_warnings:
            warnings.simplefilter('always')
            expected = stock_resize_local_mean(image, tuple(case['output']), preserve_range=case['preserve_range'])
        live_row = dict(
            row,
            stock_output=array_details(expected, np),
            stock_input_share=bool(np.shares_memory(image, expected)),
            warnings=[{'type': w.category.__name__, 'message': str(w.message)} for w in stock_warnings],
        )
        observe(np, resize_local_mean, image, live_row, compat=True)
    finally:
        xx.set_fallback_enabled(previous)
