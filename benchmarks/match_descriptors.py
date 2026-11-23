"""Fixed public call adapter for the existing sustained one-worker sampler."""

from functools import partial
import importlib.util
import inspect
from pathlib import Path
import sys
import warnings


def public_call_pairs(original, candidate, native_entry):
    """Check and yield the four frozen calls before entering timed batches."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "match_descriptors_reference_v1", root / "tests/fixtures/match_descriptors_reference_v1.py",
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    contract = fixture.contract()
    assert str(inspect.signature(original)) == str(inspect.signature(candidate)) == contract["signature"]
    assert inspect.isbuiltin(native_entry) and native_entry.__module__ == "skimagexx._core"
    original_code = inspect.unwrap(original).__code__
    for name, left, right, kwargs in fixture.fixed_cases():
        case = contract["observations"][name]
        fixture.assert_fixed_inputs(case, left, right)
        calls = []

        def profile(frame, event, value):
            if event == "call" and frame.f_code is original_code:
                raise AssertionError(f"{name} entered the original public function")
            if event == "c_return" and value is native_entry:
                calls.append(value)

        stock_call = partial(original, left, right, **kwargs)
        native_call = partial(candidate, left, right, **kwargs)
        fixture.assert_result(stock_call(), case, left, right)
        before = sys.getprofile()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sys.setprofile(profile)
            try:
                value = native_call()
            finally:
                sys.setprofile(before)
        assert calls == [native_entry]
        fixture.assert_result(value, case, left, right)
        fixture.assert_fixed_inputs(case, left, right)
        yield name, stock_call, native_call
