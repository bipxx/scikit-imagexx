"""Measure the affected public API in the exact authenticated ordinary CI install."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

source = Path.cwd() / "source"
sys.path.insert(0, str(source / "benchmarks"))

import numpy as np
from run import measure
from skimage import exposure as original
import scipy
import skimage
import skimagexx as xx

OPTIONS = (
    ("default", {}),
    ("observed_reversed_range", {"out_range": (0.7, 0.0)}),
    ("nondefault_range", {"in_range": (12, 190), "out_range": (-2, 4)}),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("normal", "disabled"))
    mode = parser.parse_args().mode
    assert os.environ.get("SKIMAGEXX_DISABLE_SIMD") == ("1" if mode == "disabled" else None)
    assert not {"NPY_ENABLE_CPU_FEATURES", "NPY_DISABLE_CPU_FEATURES"} & os.environ.keys()
    assert (np.__version__, scipy.__version__, skimage.__version__) == ("2.5.4", "1.18.1", "0.26.0")
    assert ctypes.CDLL(None).fegetround() == 0
    assert all(Path(module.__file__).is_relative_to(Path(sys.base_prefix))
               for module in (np, scipy, skimage, xx, xx._core))
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    default_workers = xx.get_num_threads()
    assert default_workers == 1
    xx.set_fallback_enabled(False)
    xx.set_num_threads(1)
    has_avx2 = xx.has_avx2()
    assert has_avx2 == (mode == "normal"), (mode, has_avx2)
    records = []
    rng = np.random.default_rng(2026)
    current = xx.exposure.rescale_intensity
    reference = original.rescale_intensity
    for size in (64, 1024):
        pixels = rng.integers(0, 256, (size, size), dtype=np.uint8)
        for dtype in (np.uint8, np.float32, np.float64):
            image = pixels if dtype == np.uint8 else (pixels / 255).astype(dtype)
            for name, options in OPTIONS:
                actual = current(image, **options)
                expected = reference(image, **options)
                assert actual.dtype == expected.dtype
                if dtype == np.uint8:
                    np.testing.assert_array_equal(actual, expected)
                else:
                    tolerance = 2e-6 if dtype == np.float32 else 2e-13
                    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
                timing = measure(lambda: reference(image, **options),
                                 lambda: current(image, **options), 7, 15)
                records.append({"case": name, "options": options, "dtype": np.dtype(dtype).name,
                                "size": size, "threads": 1, **timing})
                print(mode, size, np.dtype(dtype).name, name, timing["speedup"], flush=True)
    artifact = Path("prepared/ci-artifacts/measurement.json")
    core = Path(xx._core.__file__)
    report = {
        "source_revision": os.environ["PRODUCT_SOURCE"], "mode": mode, "affinity": [cpu],
        "cpu": platform.processor(), "platform": platform.platform(), "python": sys.version,
        "cpuinfo": Path("/proc/cpuinfo").read_text(encoding="utf-8"),
        "numpy_cpu_features": np._core._multiarray_umath.__cpu_features__,
        "default_workers": default_workers, "active_workers": xx.get_num_threads(),
        "runtime": [np.__version__, scipy.__version__, skimage.__version__],
        "core": str(core), "core_sha256": hashlib.file_digest(core.open("rb"), "sha256").hexdigest(),
        "artifact_manifest_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "rounding_mode": 0, "has_avx2": has_avx2, "fallback": False, "results": records,
        "other_processes": subprocess.check_output(["ps", "-eo", "stat,comm"], text=True),
    }
    with Path("proof", f"{mode}.json").open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
