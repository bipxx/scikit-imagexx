import numpy as np
import pytest

from probabilistic_hough_line_reference import cases, frozen_rows, observe


@pytest.mark.parametrize('case_id', [row['id'] for row in frozen_rows() if row['scope'] != 'compatibility'])
def test_probabilistic_hough_line_exact_public(case_id):
    import skimagexx as xx
    from skimagexx.transform import probabilistic_hough_line

    case = {args[0]: args for args in cases(np)}[case_id]
    row = {item['id']: item for item in frozen_rows()}[case_id]
    previous = xx.set_fallback_enabled(False)
    try:
        observe(probabilistic_hough_line, np, case, row)
    finally:
        xx.set_fallback_enabled(previous)
