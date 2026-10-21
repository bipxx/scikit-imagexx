"""Allocation-inclusive, warmed Python API comparison against scikit-image.

Run from the repository root: python benchmarks/run.py --output benchmarks/results.json
"""
import os

# Control the reference's BLAS/OpenMP pools before importing numerical packages.
for variable in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"]:
    os.environ[variable] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import argparse
import hashlib
import json
import platform
from pathlib import Path
import random
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy
import skimage
from skimage import color, filters
import skimagexx as xx


def batch(fn, loops):
    # Outside timing: reduce interference from the previous call's worker pool.
    time.sleep(0.02)
    start = time.perf_counter_ns()
    for _ in range(loops):
        result = fn()
        # Include output destruction in timing for each call, including the last.
        del result
    return (time.perf_counter_ns()-start)/loops/1e6


def measure(reference, native, repeat, target_ms, baseline=None):
    functions = [reference, native] + ([baseline] if baseline else [])
    for fn in functions:
        fn()  # warm imports, caches, SIMD dispatch, and thread pool
    first = [batch(fn, 1) for fn in functions]
    loops = [max(1, min(1000, int(target_ms/max(t, 0.001)))) for t in first]
    samples = [[] for _ in functions]
    rng = random.Random(81)
    for _ in range(repeat):
        order = list(range(len(functions)))
        rng.shuffle(order)
        for index in order:
            samples[index].append(batch(functions[index], loops[index]))
    medians = [statistics.median(v) for v in samples]
    result = {"reference_ms": medians[0], "cpp_ms": medians[1],
              "speedup": medians[0]/medians[1], "samples_ms": samples, "loops": loops,
              "sample_labels": ["scikit-image", "current"] + (["baseline"] if baseline else [])}
    if baseline:
        result.update(baseline_ms=medians[2], improvement=medians[2]/medians[1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 1024, 2048])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--target-ms", type=float, default=15)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument("--baseline-dir", type=Path, help="Directory containing a preserved _baseline package")
    args = parser.parse_args()
    if min(args.sizes+args.threads+[args.repeat]) < 1 or args.target_ms <= 0:
        parser.error("sizes, threads, repeat, and target-ms must be positive")
    root = Path(__file__).resolve().parents[1]
    baseline = None
    if args.baseline_dir:
        sys.path.insert(0,str(args.baseline_dir.resolve()))
        import _baseline as baseline
    digest = hashlib.sha256()
    for p in sorted(list((root/"src").glob("*"))+list((root/"include/skimagexx").glob("*"))):
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    report = {"metadata": {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "cpu": platform.processor(),
        "cpu_identifier": os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "logical_cpus": os.cpu_count(), "python": sys.version,
        "numpy": np.__version__, "scipy": scipy.__version__, "scikit_image": skimage.__version__,
        "skimagexx": xx.__version__, "avx2": xx.has_avx2(),
        "source_sha256": digest.hexdigest(), "command": sys.argv,
        "reference_thread_pools": 1, "omp_wait_policy": "PASSIVE (runtime hint)",
        "cooldown_ms": 20, "repeat": args.repeat, "target_ms": args.target_ms,
        "scope": "Warmed, allocation-inclusive Python calls; C++ input conversions included; seeded synthetic inputs; no IO",
    }, "results": []}
    package = Path(xx.__file__).parent
    report["metadata"]["current_binary_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(list(package.glob("*.pyd"))+list(package.glob("*.so")))}
    if baseline:
        package = Path(baseline.__file__).parent
        binaries = sorted(list(package.glob("*.pyd"))+list(package.glob("*.so")))
        report["metadata"]["baseline_binary_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries}
    for size in args.sizes:
        for dtype in [np.float32, np.float64, np.uint8]:
            rng = np.random.default_rng(2026)
            if dtype == np.uint8:
                a = rng.integers(0,256,(size,size),dtype=dtype)
                rgb = rng.integers(0,256,(size,size,3),dtype=dtype)
            else:
                a = rng.random((size,size)).astype(dtype)
                rgb = rng.random((size,size,3)).astype(dtype)
            cases = [
                ("rgb2gray", lambda: color.rgb2gray(rgb), lambda: xx.rgb2gray(rgb)),
                ("gaussian_sigma1", lambda: filters.gaussian(a,1), lambda: xx.gaussian(a,1)),
                ("sobel", lambda: filters.sobel(a), lambda: xx.sobel(a)),
            ]
            if dtype == np.uint8:
                cases.append(("threshold_otsu", lambda: filters.threshold_otsu(a), lambda: xx.threshold_otsu(a)))
            for name, reference, native in cases:
                expected = reference()
                for threads in args.threads:
                    xx.set_num_threads(threads)
                    actual = native()
                    old = None
                    if baseline:
                        baseline.set_num_threads(threads)
                        old_fn = getattr(baseline,name.removesuffix("_sigma1"))
                        data = rgb if name == "rgb2gray" else a
                        old = lambda: old_fn(data)
                        np.testing.assert_allclose(old(),expected,rtol=2e-6,atol=2e-6)
                    tol = 2e-6 if dtype == np.float32 else 2e-13
                    if name == "threshold_otsu":
                        assert actual == expected
                    else:
                        assert actual.dtype == expected.dtype
                        np.testing.assert_allclose(actual,expected,rtol=tol,atol=tol)
                    error = float(np.max(np.abs(np.asarray(actual,dtype=float)-np.asarray(expected,dtype=float))))
                    result = {"algorithm": name, "size": size, "dtype": np.dtype(dtype).name,
                              "threads": threads, "max_abs_error": error,
                              **measure(reference,native,args.repeat,args.target_ms,old)}
                    report["results"].append(result)
                    print(f"{name:17} {size:4} {np.dtype(dtype).name:7} t={threads}: "
                          f"{result['reference_ms']:.3f} / {result['cpp_ms']:.3f} ms = {result['speedup']:.2f}x"
                          + (f"; old/new {result['improvement']:.2f}x" if old else ""),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    lines = ["# Measured performance", "", f"Run: {report['metadata']['timestamp_utc']}", "",
             f"CPU: {platform.processor()} ({os.cpu_count()} logical CPUs). AVX2: {xx.has_avx2()}.", "",
             f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, scikit-image {skimage.__version__}.", "",
             "Median milliseconds for warmed, allocation-inclusive Python calls. Reference thread pools are limited to one; C++ worker counts are explicit. Small inputs use a serial path. Speedup = reference / C++; values below 1 are regressions. Input conversion is included. These synthetic workloads do not establish full-library performance.", "",
             "| Algorithm | Size | Dtype | C++ workers | Reference ms | C++ ms | Speedup | Max error |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in report["results"]:
        lines.append(f"| {r['algorithm']} | {r['size']} | {r['dtype']} | {r['threads']} | {r['reference_ms']:.4f} | {r['cpp_ms']:.4f} | {r['speedup']:.2f}x | {r['max_abs_error']:.2g} |")
    if baseline:
        lines += ["", "## Same-run comparison with the previous binary", "",
                  "Previous and current implementations run in the same process, with shuffled timing order and identical inputs. Values below 1x are regressions.", "",
                  "| Algorithm | Size | Dtype | Workers | Previous ms | Current ms | Improvement |",
                  "|---|---:|---|---:|---:|---:|---:|"]
        for r in report["results"]:
            lines.append(f"| {r['algorithm']} | {r['size']} | {r['dtype']} | {r['threads']} | {r['baseline_ms']:.4f} | {r['cpp_ms']:.4f} | {r['improvement']:.2f}x |")
    lines += ["", "Raw samples, repetitions, loop counts, source hash, and environment are recorded in the companion JSON file.", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    main()
