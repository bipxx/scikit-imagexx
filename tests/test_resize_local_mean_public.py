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
        observe(np, resize_local_mean, image, row, compat=compat)
    finally:
        xx.set_fallback_enabled(previous)
