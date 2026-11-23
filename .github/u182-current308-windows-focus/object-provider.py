"""Run the installed convex-hull-object provider witness after product admission."""

import argparse
import inspect
import json
import os
from pathlib import Path
from runpy import run_path
from unittest.mock import patch
import warnings


def verify(source_root, candidate_site, stock_site, mode):
    import numpy as np
    import skimage
    from skimage import morphology as stock_namespace
    from skimage.morphology import convex_hull as stock_module
    import skimagexx as candidate
    from skimagexx import _convex_hull_object as route
    from skimagexx import _core, morphology
    from skimagexx._compat import FallbackWarning

    assert os.environ.get("SKIMAGEXX_DISABLE_SIMD", "0") == ("1" if mode == "scalar" else "0")
    for module in (candidate, route, _core, morphology):
        assert Path(module.__file__).resolve().is_relative_to(candidate_site)
    for module in (skimage, stock_namespace, stock_module):
        assert Path(module.__file__).resolve().is_relative_to(stock_site)
    assert skimage.__version__ == "0.26.0"

    public = morphology.convex_hull_object
    stock = stock_namespace.convex_hull_object
    entry = _core.convex_hull_object_binary
    assert public is route.convex_hull_object
    assert stock is stock_module.convex_hull_object
    assert inspect.isbuiltin(entry) and entry.__module__ == "skimagexx._core"
    assert str(inspect.signature(public)) == str(inspect.signature(stock)) == "(image, *, connectivity=2)"

    saved = run_path(str(source_root / "tests/test_morphology_convex_hull_object.py"))
    names = ("tiny-diagonal-1", "strided")
    cases = {name: (image, options) for name, image, options in saved["cases"]() if name in names}
    assert set(cases) == set(names)
    fixture = saved["FIXTURE"]
    assert names[0] not in fixture["compatibility"] and names[1] in fixture["compatibility"]
    signature = saved["signature"]

    native_calls = []
    provider_calls = []

    def trace_native(image, connectivity):
        result = entry(image, connectivity)
        native_calls.append((image, connectivity, result))
        return result

    def trace_provider(image, **options):
        result = stock(image, **options)
        provider_calls.append((image, options, result))
        return result

    def result_signature(result, image, expected):
        assert signature(result) == expected["output"]
        assert not np.shares_memory(result, image)

    previous = candidate.set_fallback_enabled(False)
    try:
        with patch.object(_core, "convex_hull_object_binary", trace_native), patch.object(
            stock_namespace, "convex_hull_object", trace_provider
        ):
            image, options = cases[names[0]]
            expected = fixture["rows"][names[0]]
            assert signature(image) == expected["input"]
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                result = public(image, **options)
            assert len(native_calls) == 1 and not provider_calls
            passed_image, passed_connectivity, native_result = native_calls.pop()
            assert passed_image is image and passed_connectivity == options["connectivity"] == 1
            assert result is native_result and not notices
            result_signature(result, image, expected)
            assert signature(image) == expected["input"]

            image, options = cases[names[1]]
            expected = fixture["rows"][names[1]]
            assert options == {} and signature(image) == expected["input"]
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                try:
                    public(image, **options)
                except NotImplementedError as error:
                    assert "fallback is disabled" in str(error)
                else:
                    raise AssertionError("strict public call did not refuse the strided fixture")
            assert len(native_calls) == 1 and not provider_calls and not notices
            passed_image, passed_connectivity, native_result = native_calls.pop()
            assert passed_image is image and passed_connectivity == 2 and native_result is None
            assert signature(image) == expected["input"]

            candidate.set_fallback_enabled(True)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                result = public(image, **options)
            assert len(native_calls) == len(provider_calls) == 1
            passed_image, passed_connectivity, native_result = native_calls.pop()
            provider_image, provider_options, provider_result = provider_calls.pop()
            assert passed_image is provider_image is image
            assert passed_connectivity == 2 and native_result is None
            assert provider_options == {"connectivity": 2} and result is provider_result
            assert expected["warnings"] == [] and len(notices) == 1
            assert notices[0].category is FallbackWarning
            assert str(notices[0].message) == "skimage.morphology.convex_hull_object executed by scikit-image fallback"
            result_signature(result, image, expected)
            assert signature(image) == expected["input"]
    finally:
        candidate.set_fallback_enabled(previous)

    return {
        "mode": mode,
        "candidate_core": str(Path(_core.__file__).resolve()),
        "stock_module": str(Path(stock_module.__file__).resolve()),
        "native": names[0],
        "strict_refusal_and_replaced_provider": names[1],
        "calls": {"builtin": 3, "stock": 1},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-site", type=Path, required=True)
    parser.add_argument("--stock-site", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "scalar"), required=True)
    args = parser.parse_args()
    roots = [path.resolve(strict=True) for path in (args.source_root, args.candidate_site, args.stock_site)]
    assert all(root.is_dir() for root in roots)
    print(json.dumps(verify(*roots, args.mode), sort_keys=True))


if __name__ == "__main__":
    main()
