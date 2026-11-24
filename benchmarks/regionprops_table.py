"""Four fixed public-call pairs for the shared one-worker benchmark sampler.

The qualification caller supplies the authenticated installed public functions
and native entry. This adapter performs the frozen prechecks before it yields
a pair of complete allocation-inclusive calls.
"""

from functools import partial
import importlib.util
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def fixed_public_calls(original, candidate, native_entry):
    path = ROOT / 'tests/fixtures/regionprops_table_v1.py'
    spec = importlib.util.spec_from_file_location('_regionprops_table_cases', path)
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    signature = cases.CONTRACT['signature']
    assert str(inspect.signature(original)) == str(inspect.signature(candidate)) == signature
    assert inspect.isbuiltin(native_entry)

    for name in cases.FIXED_CASES:
        image = cases.make_input(name)
        cases.assert_frozen_table(name, image, original(image))
        returned = []

        def profile(frame, event, value):
            if event == 'call' and frame.f_code is original.__code__:
                raise AssertionError('regionprops_table called the original')
            if event == 'c_return' and value is native_entry:
                returned.append(value)

        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            actual = candidate(image)
        finally:
            sys.setprofile(previous)
        assert returned == [native_entry]
        cases.assert_frozen_table(name, image, actual)
        del actual
        yield name, partial(original, image), partial(candidate, image)
