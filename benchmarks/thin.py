"""Bounded full-public thin comparisons for the retained sustained sampler."""

from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import inspect
from pathlib import Path
import sys
import warnings


def public_call_pairs(original, candidate, native_entry):
    root = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location("thin_reference_v1", root / "tests/fixtures/thin_reference_v1.py")
    fixture = module_from_spec(spec)
    spec.loader.exec_module(fixture)
    contract = fixture.contract()
    assert str(inspect.signature(original)) == str(inspect.signature(candidate)) == contract["signature"]
    assert inspect.isbuiltin(native_entry) and native_entry.__module__ == "skimagexx._core"
    original_code = inspect.unwrap(original).__code__
    for name, image, max_num_iter in fixture.fixed_cases():
        expected = contract["observations"][name]
        assert fixture.snapshot(image) == expected["input"]
        reference = partial(original, image, max_num_iter=max_num_iter)
        native = partial(candidate, image, max_num_iter=max_num_iter)
        calls = []

        def profile(frame, event, value):
            if event == "call" and frame.f_code is original_code:
                raise AssertionError(f"{name} entered the original public function")
            if event == "c_return" and value is native_entry:
                calls.append(value)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fixture.snapshot(reference()) == expected["output"]
            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                assert fixture.snapshot(native()) == expected["output"]
            finally:
                sys.setprofile(previous)
        assert calls == [native_entry] and fixture.snapshot(image) == expected["input"]
        yield name, reference, native
