"""Frozen full-call endpoints for the existing sustained sampler."""

from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import inspect
from pathlib import Path
import sys
import warnings


def public_call_pairs(original, candidate, native_entry):
    root = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location(
        "convex_hull_image_v1", root / "tests/fixtures/convex_hull_image_v1.py",
    )
    fixture = module_from_spec(spec)
    spec.loader.exec_module(fixture)
    contract = fixture.contract()
    assert str(inspect.signature(original)) == str(inspect.signature(candidate)) == contract["signature"]
    assert inspect.isbuiltin(native_entry) and native_entry.__module__ == "skimagexx._core"
    stock_code = inspect.unwrap(original).__code__
    for name, image, options in fixture.fixed_cases():
        frozen = contract["observations"][name]
        assert fixture.snapshot(image) == frozen["input"]
        stock = partial(original, image, **options)
        native = partial(candidate, image, **options)
        calls = []

        def profile(frame, event, arg):
            if event == "call" and frame.f_code is stock_code:
                raise AssertionError(f"{name} entered the original through the candidate")
            if event == "c_return" and arg is native_entry:
                calls.append(arg)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fixture.snapshot(stock()) == frozen["output"]
            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                assert fixture.snapshot(native()) == frozen["output"]
            finally:
                sys.setprofile(previous)
        assert calls == [native_entry] and fixture.snapshot(image) == frozen["input"]
        yield name, stock, native
