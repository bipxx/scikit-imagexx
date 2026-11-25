"""Fixed allocation-inclusive FAST workloads; source-only plan."""

from functools import partial
import inspect
from pathlib import Path
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def public_cases(np):
    recipe = runpy.run_path(str(ROOT / 'tests/corner_fast_cases.py'))
    cases = {name: (image, options) for name, image, options, _ in recipe['cases'](np)}
    return [(name, *cases[name]) for name in recipe['TIMED']]


def public_call(function, image, options):
    return function(image, **options)


def fixed_public_calls(original, candidate, native_entry):
    import numpy as np

    assert inspect.signature(original) == inspect.signature(candidate)
    assert inspect.isbuiltin(native_entry)
    for name, image, options in public_cases(np):
        reference = partial(public_call, original, image, options)
        owned = partial(public_call, candidate, image, options)
        entries = []

        def profile(frame, event, value):
            if event == 'c_return' and value is native_entry:
                entries.append(value)

        expected = reference()
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            actual = owned()
        finally:
            sys.setprofile(previous)
        assert entries == [native_entry]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
        yield name, reference, owned


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', required=True)
    parser.parse_args()
    contract = json.loads((ROOT / 'coverage/contracts/corner-fast-v1.json').read_bytes())
    print(json.dumps(contract['timing'], indent=2))
