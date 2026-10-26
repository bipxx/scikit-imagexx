"""Exact float32 lane/buffer controls against the genuine NumPy reduction."""
import warnings

import pytest

CASES = ("cancellation", "overflow", "buffer_split")


def _reference(case):
    import numpy as np
    assert case in CASES and np.__version__ in ("1.24.4", "2.5.4")
    maximum = np.finfo(np.float32).max
    values = np.full(32 if case == "buffer_split" else 16, maximum, dtype=np.float32)
    if case == "cancellation":
        values[8:] = -maximum
    elif case == "buffer_split":
        values[16:] = -maximum
    previous = np.getbufsize()
    try:
        np.setbufsize(16)
        with np.errstate(all="warn"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected = np.sum(values)
    finally:
        np.setbufsize(previous)
    messages = [(warning.category.__name__, str(warning.message)) for warning in caught]
    invalid = case == "buffer_split" and np.__version__ == "1.24.4"
    required = [] if case == "cancellation" else [("RuntimeWarning", "overflow encountered in reduce")]
    if invalid:
        required.append(("RuntimeWarning", "invalid value encountered in reduce"))
    assert messages == required
    if case == "cancellation":
        assert expected.tobytes() == b"\0" * 4
    elif invalid:
        assert np.isnan(expected)
    else:
        assert np.isposinf(expected)
    return values, expected, (0 if case == "cancellation" else 2) | (8 if invalid else 0), messages


@pytest.mark.parametrize("case", CASES)
def test_float32_reduction_lanes(case):
    import numpy as np
    values, expected, expected_status, _ = _reference(case)
    from skimagexx import _core
    assert _core.get_num_threads() == 1
    guards = _core.ColocalizationGuards()
    guards.add(values)
    native = _core.coloc_sum(guards, values, 1 if np.__version__ == "1.24.4" else 2, 16)
    assert native is not None
    actual, status = native
    assert type(actual) is type(expected) and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes() and status == expected_status
    assert _core.get_num_threads() == 1
