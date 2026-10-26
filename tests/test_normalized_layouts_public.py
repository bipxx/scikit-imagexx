"""Saved image-derived moment tensors; prospective C/F batch contracts."""
import argparse
import json
from pathlib import Path
import sys

import pytest

if __package__:
    from .moments_derived_cases import error_limits
    from .test_moments_derived_public import runtime, observe, compare, result_record
else:
    from moments_derived_cases import error_limits
    from test_moments_derived_public import runtime, observe, compare, result_record

DATA_PATH = Path(__file__).resolve().parents[1] / "benchmarks/normalized_layouts_cases.json"
DATA = json.loads(DATA_PATH.read_text(encoding="utf-8"))
TENSORS = {row["id"]: row for row in DATA["cases"]}
CASES = [dict(id=name + "-" + layout, tensor=name, layout=layout, control=None)
         for name in TENSORS for layout in ("C", "F")]
CASES += [dict(id="F-" + control, tensor="u8-auto", layout="F", control=control)
          for control in ("zero-mass", "nan-entry", "inf-entry")]
POLICY = dict(api="moments_normalized", exact=False, order=3, spacing=None)


def fixture(spec, np):
    saved = TENSORS[spec["tensor"]]
    raw = bytes.fromhex("".join(saved["logical_c_words_hex"]))
    values = np.frombuffer(raw, dtype=saved["dtype"]).reshape(saved["shape"])
    image = np.array(values, copy=True, order=spec["layout"])
    if spec["control"] == "zero-mass":
        image.fill(0)
    elif spec["control"] == "nan-entry":
        image[3, 3] = np.nan
    elif spec["control"] == "inf-entry":
        image[3, 3] = np.inf
    assert image.flags.owndata and image.flags.writeable and image.flags.aligned
    assert image.flags.c_contiguous == (spec["layout"] == "C")
    assert image.flags.f_contiguous == (spec["layout"] == "F")
    return image


@pytest.fixture(scope="module")
def reference_runtime():
    return runtime()


@pytest.fixture(scope="module")
def worker_results():
    return {}


@pytest.mark.parametrize("spec", CASES, ids=[row["id"] for row in CASES])
@pytest.mark.parametrize("workers", (1, 4))
def test_saved_tensor_layouts(reference_runtime, spec, workers, worker_results, monkeypatch, tmp_path):
    np, original, _ = reference_runtime
    image = fixture(spec, np)
    expected, error, messages, checks = observe(original.moments_normalized, image, {}, {}, np)
    original_row = dict(id=spec["id"], input=result_record(image), result=result_record(expected),
                        error=error, warnings=messages, checks=checks)
    (tmp_path / "original.json").write_text(json.dumps(original_row) + "\n")
    assert error is None and all(checks.values())
    with np.errstate(all="ignore"):
        bound = error_limits(POLICY, image, expected, np)

    image = fixture(spec, np)
    import skimage.measure as public_original
    import skimagexx as xx
    from skimagexx import _compat, _moments_derived as implementation
    from skimagexx.coverage import native_functions
    assert "measure.moments_normalized" in native_functions()
    assert xx.get_num_threads() == 1
    previous = xx.set_fallback_enabled(False)
    old_profile = sys.getprofile()
    returns = dict(batch_probe=0, scalar_divide=0)
    batch, forbidden = {}, []

    def deny(*args, **kwargs):
        forbidden.append(True)
        raise AssertionError("upstream moments forwarding")

    def trace(frame, event, arg):
        if (event == "c_return" and getattr(arg, "__module__", None) == xx._core.__name__
                and getattr(arg, "__name__", None) == "moments_normalize"):
            if frame.f_code is implementation.moments_normalized.__code__:
                returns["batch_probe"] += 1
            elif frame.f_code is implementation._divide.__code__:
                returns["scalar_divide"] += 1
            else:
                raise AssertionError("unclassified normalization return")
        if event == "return" and frame.f_code is implementation.moments_normalized.__code__:
            probe = frame.f_locals.get("batch")
            batch.update(attempted="batch" in frame.f_locals,
                         accepted=isinstance(probe, np.ndarray), returned_probe=arg is probe)

    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            for name in ("moments", "moments_central", "centroid", "moments_normalized", "moments_hu"):
                patch.setattr(public_original, name, deny)
                patch.setattr(original, name, deny)
            sys.setprofile(trace)
            try:
                actual, actual_error, actual_messages, actual_checks = observe(
                    xx.measure.moments_normalized, image, {}, {}, np)
            finally:
                sys.setprofile(old_profile)
        row = dict(id=spec["id"], workers=workers, input=result_record(image), result=result_record(actual),
                   error=actual_error, warnings=actual_messages, checks=actual_checks,
                   batch=batch, actual_C_returns=returns, forbidden=forbidden)
        (tmp_path / "candidate.json").write_text(json.dumps(row) + "\n")
        assert not forbidden and actual_error == error and actual_messages == messages
        assert all(actual_checks.values())
        compare(actual, expected, bound, dict(POLICY, id=spec["id"]), np)
        words = (actual.tobytes(order="C"), actual_error, actual_messages)
        assert worker_results.setdefault(spec["id"], words) == words
        accepted = spec["control"] is None
        assert batch == dict(attempted=True, accepted=accepted, returned_probe=accepted)
        assert returns == dict(batch_probe=1, scalar_divide=0 if accepted else 13)
        assert xx.get_num_threads() == workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(old_profile)
        xx.set_num_threads(1)
        xx.set_fallback_enabled(previous)


def reference_only(output):
    assert not any(name == "skimagexx" or name.startswith("skimagexx.") for name in sys.modules)
    output.mkdir()
    np, original, record = runtime()
    (output / "runtime.json").write_text(json.dumps(record, indent=2) + "\n")
    with (output / "journal.jsonl").open("x") as stream:
        for spec in CASES:
            image = fixture(spec, np)
            result, error, messages, checks = observe(original.moments_normalized, image, {}, {}, np)
            row = dict(id=spec["id"], input=result_record(image), result=result_record(result),
                       error=error, warnings=messages, checks=checks)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            assert error is None and all(checks.values())
            with np.errstate(all="ignore"):
                bound = error_limits(POLICY, image, result, np)
            compare(result, result, bound, dict(POLICY, id=spec["id"]), np)
    (output / "summary.json").write_text(json.dumps(dict(
        status="ORIGINAL_ONLY", cases=len(CASES), returns=len(CASES), errors=0,
        native_calls=0, native_coverage=0), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-only", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_only(args.output)
