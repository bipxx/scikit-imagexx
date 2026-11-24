"""Four frozen public calls for a separately authorized private comparison."""

from functools import partial
import importlib.util
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def fixed_public_calls(original, candidate, native_entry):
    path = ROOT / 'tests/fixtures/corner_moravec_v1.py'
    spec = importlib.util.spec_from_file_location('_corner_moravec_cases', path)
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    assert str(inspect.signature(original)) == str(inspect.signature(candidate)) == cases.CONTRACT['signature']
    assert inspect.isbuiltin(native_entry)
    for name in cases.FIXED:
        image, kwargs = cases.make_input(name)
        cases.assert_frozen_response(name, image, original(image, **kwargs), stock=True)
        entries = []

        def profile(frame, event, value):
            if event == 'call' and frame.f_code is original.__code__:
                raise AssertionError('private Moravec called the stock API')
            if event == 'c_return' and value is native_entry:
                entries.append(value)

        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            actual = candidate(image, **kwargs)
        finally:
            sys.setprofile(previous)
        assert entries == [native_entry]
        cases.assert_frozen_response(name, image, actual)
        del actual
        yield name, partial(original, image, **kwargs), partial(candidate, image, **kwargs)
