"""Allocation-inclusive one-worker comparison of the eight frozen Pearson cases."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import inspect
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LIMITS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
PATHS = (
    "benchmarks/pearson.py", "coverage/contracts/pearson-reference-v1.json",
    "include/skimagexx/pearson.hpp", "python/skimagexx/_pearson.py", "src/bind_array_storage.hpp",
    "src/bind_fft.cpp", "src/bind_pearson.cpp", "src/pearson.cpp",
    "tests/fixtures/pearson_reference_v1.py", "tests/fixtures/pearson_original_windows_v1.json",
    "tools/montage_sustained.py",
)


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
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
    assert not args.output.exists()
    journal = args.output.with_suffix(".batches.jsonl")
    assert not journal.exists()
    go = json.loads(args.go.read_bytes())
    assert go["authority"] == "pearson-eight-one-worker-cases"
    deadline = go["deadline_utc"]
    assert datetime.now(timezone.utc) < datetime.fromisoformat(deadline)
    assert {os.environ[key] for key in LIMITS} == {"1"}
    assert not git("status", "--porcelain")
    assert git("rev-parse", "HEAD") == go["source"]
    pins = {path: file_hash(ROOT / path) for path in PATHS}
    assert pins == go["source_raw_sha256"]

    import numpy as np
    import scipy
    import skimage
    from skimage.measure import pearson_corr_coeff as reference

    if args.private_overlay:
        overlay = load("_pearson_private_overlay", args.private_overlay)
        xx = overlay.overlay()
        from skimagexx import _pearson
        native = _pearson.pearson_corr_coeff
        scope = "private leaf and wrapper"
    else:
        import skimagexx as xx
        from skimagexx import _pearson
        native = xx.measure.pearson_corr_coeff
        assert native is _pearson.pearson_corr_coeff
        scope = "installed public measure route"
    assert str(inspect.signature(native)) == str(inspect.signature(reference))
    assert skimage.__version__ == "0.26.0"
    assert [np.__version__, scipy.__version__] == go["runtime_versions"]
    original_pin = file_hash(Path(inspect.getsourcefile(reference)))
    assert original_pin == go["original_raw_sha256"]
    core = Path(xx._core.__file__)
    assert file_hash(core) == go["core_sha256"]
    entry = xx._core.pearson_f64
    assert inspect.isbuiltin(entry) and entry.__module__ == "skimagexx._core"
    xx.set_num_threads(1)
    assert xx.get_num_threads() == 1
    xx.set_fallback_enabled(False)

    pool = load("threadpoolctl", args.threadpoolctl)
    method = load("_pearson_sampler", ROOT / "tools/montage_sustained.py")
    factory = load("_pearson_benchmark_cases", ROOT / "tests/fixtures/pearson_reference_v1.py")
    contract = json.loads((ROOT / "coverage/contracts/pearson-reference-v1.json").read_bytes())
    fixture = ROOT / contract["reference"]["fixture"]
    assert file_hash(fixture) == contract["reference"]["sha256"]
    frozen = {row["id"]: row for row in json.loads(fixture.read_bytes())["rows"]}
    cases = tuple(factory.benchmark_cases())
    assert [case[0] for case in cases] == go["cases"]
    rows = []
    policy = np.geterr(), np.geterrcall(), np.getbufsize()
    with pool.threadpool_limits(limits=1):
        pools = pool.threadpool_info()
        assert pools and all(runtime["num_threads"] == 1 for runtime in pools)
        for index, (case, spec) in enumerate(zip(cases, contract["benchmark"]["cases"], strict=True)):
            name, left, right, mask, count, _ = case
            operands = (left, right) if mask is None else (left, right, mask)
            before = [factory.descriptor(array) for array in operands]
            assert name == spec["id"] and count == spec["selected_count"]
            assert before == frozen[name]["inputs"]
            expected = reference(*operands)
            native_returns = []

            def profile(frame, event, value):
                if event == "call" and frame.f_code is reference.__code__:
                    raise AssertionError("Pearson native path called the upstream API")
                if event == "c_return" and value is entry:
                    native_returns.append(entry.__name__)

            prior = sys.getprofile()
            sys.setprofile(profile)
            try:
                actual = native(*operands)
            finally:
                sys.setprofile(prior)
            assert native_returns == ["pearson_f64"]
            assert type(actual) is tuple and tuple(map(type, actual)) == (float, float)
            np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=2e-14)
            np.testing.assert_allclose(actual[1], expected[1], rtol=2e-9, atol=0)

            timing = method.measure(
                lambda: reference(*operands), lambda: native(*operands), 7, 100,
                journal=journal, case_index=index, case_id=name, pass_index=1, deadline=deadline,
            )
            assert [factory.descriptor(array) for array in operands] == before
            assert xx.get_num_threads() == 1
            assert policy == (np.geterr(), np.geterrcall(), np.getbufsize())
            rows.append(dict(**spec, native_returns=native_returns, actual=actual,
                             original=expected, **timing))
            print(f"{name}: {timing['speedup']:.6f}x", flush=True)
        assert pool.threadpool_info() == pools
    assert not git("status", "--porcelain") and git("rev-parse", "HEAD") == go["source"]
    assert {path: file_hash(ROOT / path) for path in PATHS} == pins
    assert file_hash(core) == go["core_sha256"]
    assert all(list(map(len, row["samples_ms"])) == [7, 7] for row in rows)
    result = dict(scope=scope, source=go["source"], source_raw_sha256=pins,
                  core_sha256=file_hash(core), python=sys.version, numpy=np.__version__,
                  scipy=scipy.__version__, original_source_sha256=original_pin,
                  runtime_pools=pools, cases=rows, accepted_samples=len(rows) * 14)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
