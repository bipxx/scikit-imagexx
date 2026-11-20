"""Measure bounded native SSIM public calls against scikit-image 0.26.0."""

# The shared timing entry point sets reference thread limits before imports.
from run import measure

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import skimage
from skimage.metrics import structural_similarity as reference_score
import skimagexx as xx
from skimagexx.metrics import structural_similarity as native_score

from provenance import source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", default=7, type=int)
    parser.add_argument("--target-ms", default=100, type=float)
    args = parser.parse_args()
    if args.repeat < 1 or not 0 < args.target_ms < float("inf"):
        parser.error("expected positive repeat and finite positive target-ms")
    assert skimage.__version__ == "0.26.0"
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    owner = importlib.import_module(xx._core.structural_similarity_score.__module__)
    binaries = {Path(module.__file__) for module in (xx._core, owner)}
    binary_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in binaries}
    assert xx.get_num_threads() == owner.get_num_threads() == 1
    xx.set_fallback_enabled(False)
    rows = []
    rng = np.random.default_rng(787854)
    for dtype, atol in ((np.float32, 2e-7), (np.float64, 1e-13)):
        for shape, window, sample in (
            ((16, 21), None, True),
            ((256, 257), 7, True),
            ((513, 641), 11, False),
            ((1024, 1031), 7, True),
        ):
            left = rng.uniform(0.01, 0.95, size=shape).astype(dtype)
            right = (left * 0.75 + rng.uniform(0.01, 0.20, size=shape)).astype(dtype)
            options = {"win_size": window, "data_range": 1.0, "use_sample_covariance": sample}
            def reference():
                return reference_score(left, right, **options)
            def native():
                return native_score(left, right, **options)

            expected, actual = reference(), native()
            assert type(expected) is type(actual) is np.float64
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)
            timing = measure(reference, native, args.repeat, args.target_ms)
            rows.append({
                "api": "metrics.structural_similarity", "dtype": left.dtype.name,
                "shape": list(shape), "options": options, "threads": 1, "rtol": 0, "atol": atol,
                "expected_hex": float(expected).hex(), "current_hex": float(actual).hex(),
                **timing,
            })
    assert fingerprint == source_fingerprint(root)
    assert binary_hashes == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in binaries}
    report = {"metadata": {
        "time_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "cpu": platform.processor(),
        "scikit_image": skimage.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
        "avx2": xx.has_avx2(), "reference_thread_pools": 1,
        "native_workers": 1, "fallback_enabled": False, "repeat": args.repeat,
        "target_ms": args.target_ms, "native_owner": owner.__name__,
        "binary_sha256": binary_hashes, **fingerprint,
    }, "results": rows}
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
