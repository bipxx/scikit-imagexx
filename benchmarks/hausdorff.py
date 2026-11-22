"""Sustained public-call speed checks on the frozen Hausdorff workloads."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LIMITS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
PATHS = (
    "benchmarks/hausdorff.py", "coverage/contracts/hausdorff-family-v1.json",
    "include/skimagexx/hausdorff.hpp", "src/hausdorff.cpp", "src/bind_hausdorff.cpp",
    "src/bind_array_storage.hpp", "src/bind_fft.cpp", "python/skimagexx/_hausdorff.py",
    "tests/fixtures/hausdorff_reference_v1.py", "tests/fixtures/hausdorff_original_windows_v1.json",
    "tools/montage_sustained.py",
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threadpoolctl", type=Path, required=True)
    parser.add_argument("--private-overlay", type=Path)
    args = parser.parse_args()
    journal = args.output.with_suffix(".batches.jsonl")
    assert not args.output.exists() and not journal.exists()
    go = json.loads(args.go.read_bytes())
    assert go["authority"] == "hausdorff-fifteen-one-worker-cases"
    assert datetime.now(timezone.utc) < datetime.fromisoformat(go["deadline_utc"])
    assert {os.environ[key] for key in LIMITS} == {"1"}
    assert git("rev-parse", "HEAD") == go["source"] and not git("status", "--porcelain")
    pins = {path: digest(ROOT / path) for path in PATHS}
    assert pins == go["source_raw_sha256"]

    import numpy as np
    import scipy
    import skimage
    from skimage import metrics as original

    if args.private_overlay:
        xx = load("_hausdorff_private_overlay", args.private_overlay).overlay()
        from skimagexx import _hausdorff
        backend = _hausdorff
        scope = "private leaf and wrapper"
    else:
        import skimagexx as xx
        from skimagexx import _hausdorff
        backend = xx.metrics
        assert backend.hausdorff_pair is _hausdorff.hausdorff_pair
        assert backend.hausdorff_distance is _hausdorff.hausdorff_distance
        scope = "installed public metrics routes"
    assert (skimage.__version__, np.__version__, scipy.__version__) == tuple(go["versions"])
    assert digest(Path(inspect.getsourcefile(original.hausdorff_pair))) == go["original_sha256"]
    binary = Path(xx._core.__file__)
    assert digest(binary) == go["core_sha256"]
    entry = xx._core.hausdorff_mask
    assert inspect.isbuiltin(entry) and entry.__module__ == "skimagexx._core"
    xx.set_num_threads(1)
    xx.set_fallback_enabled(False)

    pool = load("threadpoolctl", args.threadpoolctl)
    sampler = load("_hausdorff_sampler", ROOT / "tools/montage_sustained.py")
    factory = load("_hausdorff_cases", ROOT / "tests/fixtures/hausdorff_reference_v1.py")
    contract = json.loads((ROOT / "coverage/contracts/hausdorff-family-v1.json").read_bytes())
    fixture = ROOT / contract["fixture"]
    assert digest(fixture) == contract["fixture_sha256"]
    rows = {row["id"]: row for row in json.loads(fixture.read_bytes())["rows"]}
    results = []
    with pool.threadpool_limits(limits=1):
        pools = pool.threadpool_info()
        assert pools and all(runtime["num_threads"] == 1 for runtime in pools)
        for case_name, a, b in factory.benchmark_cases():
            inputs = [factory.image_signature(a), factory.image_signature(b)]
            assert json.loads(json.dumps(inputs)) == rows[case_name]["inputs"]
            for operation in contract["performance"]["operations"]:
                case = f"{case_name}:{operation}"
                assert case == go["cases"][len(results)]
                if operation == "pair":
                    reference, native = original.hausdorff_pair, backend.hausdorff_pair
                    options = {}
                else:
                    reference, native = original.hausdorff_distance, backend.hausdorff_distance
                    options = {"method": operation}
                assert inspect.signature(reference) == inspect.signature(native)
                expected = reference(a, b, **options)
                returned = []

                def profile(frame, event, value):
                    if event == "call" and frame.f_code is reference.__code__:
                        raise AssertionError("native Hausdorff entered the original callable")
                    if event == "c_return" and value is entry:
                        returned.append(entry.__name__)

                previous = sys.getprofile()
                sys.setprofile(profile)
                try:
                    actual = native(a, b, **options)
                finally:
                    sys.setprofile(previous)
                assert returned == ["hausdorff_mask"]
                assert type(actual) is type(expected)
                if operation == "pair":
                    assert factory.description(actual) == factory.description(expected)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0)
                timing = sampler.measure(lambda: reference(a, b, **options),
                    lambda: native(a, b, **options), 7, 100, journal=journal, case_index=len(results),
                    case_id=case, pass_index=1, deadline=go["deadline_utc"])
                assert [factory.image_signature(a), factory.image_signature(b)] == inputs
                assert xx.get_num_threads() == 1
                results.append(dict(case_id=case, core_returns=returned, inputs=inputs,
                                    output=factory.description(actual), **timing))
                print(f"{case}: {timing['speedup']:.6f}x", flush=True)
        assert pool.threadpool_info() == pools
    assert len(results) == 15 and all(list(map(len, row["samples_ms"])) == [7, 7] for row in results)
    assert git("rev-parse", "HEAD") == go["source"] and not git("status", "--porcelain")
    assert {path: digest(ROOT / path) for path in PATHS} == pins and digest(binary) == go["core_sha256"]
    output = dict(scope=scope, source=go["source"], core_sha256=go["core_sha256"], source_raw_sha256=pins,
                  versions=go["versions"], python=sys.version, runtime_pools=pools,
                  accepted_samples=len(results) * 14, cases=results)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
