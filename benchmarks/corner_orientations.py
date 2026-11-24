"""Fixed paired public benchmarks for feature.corner_orientations."""

from functools import partial
import importlib.util
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def fixed_public_calls(original, candidate, native_entry):
    path = ROOT / 'tests/fixtures/corner_orientations_v1.py'
    spec = importlib.util.spec_from_file_location('_corner_orientations_cases', path)
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    assert str(inspect.signature(original)) == str(inspect.signature(candidate)) == cases.CONTRACT['signature']
    assert inspect.isbuiltin(native_entry)
    for name in cases.FIXED:
        image, corners, mask, row = cases.make_case(name)
        stock_out = original(image, corners, mask)
        cases.assert_memoryview_response(stock_out, len(corners), stock=True)
        entries = []

        def profile(frame, event, value):
            if event == 'call' and frame.f_code is original.__code__:
                raise AssertionError('native corner_orientations called stock API')
            if event == 'c_return' and value is native_entry:
                entries.append(value)

        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            actual = candidate(image, corners, mask)
        finally:
            sys.setprofile(previous)
        assert entries == [native_entry]
        cases.assert_memoryview_response(actual, len(corners))
        assert cases.sha(actual) == cases.expected_sha256(row)
        del actual, stock_out
        yield name, partial(original, image, corners, mask), partial(candidate, image, corners, mask)
