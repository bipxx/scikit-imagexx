"""Compare Otsu input distributions against the preserved previous binary."""
from run import measure  # also configures reference thread pools before NumPy
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from skimage.filters import threshold_otsu
import skimagexx as xx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/histogram-results.json"))
    args = parser.parse_args()
    if args.size < 1:
        parser.error("size must be positive")
    sys.path.insert(0,str(args.baseline_dir.resolve()))
    import _baseline as old
    xx.set_num_threads(1)
    old.set_num_threads(1)
    rng = np.random.default_rng(2026)
    shape = (args.size,args.size)
    cases = {
        "random": rng.integers(0,256,shape,dtype=np.uint8),
        "constant": np.full(shape,99,np.uint8),
        "binary": (rng.integers(0,2,shape,dtype=np.uint8)*255),
        "skewed": np.where(rng.random(shape)<0.001,255,99).astype(np.uint8),
        "late_outlier": np.full(shape,99,np.uint8),
    }
    cases["late_outlier"][-1,-1] = 255
    source = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for p in sorted(list((root/"src").glob("*"))+list((root/"include/skimagexx").glob("*"))):
        source.update(p.name.encode()); source.update(p.read_bytes())
    results = []
    for name,a in cases.items():
        expected = threshold_otsu(a)
        assert xx.threshold_otsu(a) == old.threshold_otsu(a) == expected
        result = {"pattern": name,"size": args.size,"threads": 1,
                  **measure(lambda: threshold_otsu(a),lambda: xx.threshold_otsu(a),7,15,
                            lambda: old.threshold_otsu(a))}
        results.append(result)
        print(f"{name}: previous {result['baseline_ms']:.4f}, current {result['cpp_ms']:.4f} ms; "
              f"{result['improvement']:.2f}x faster",flush=True)
    report = {"source_sha256": source.hexdigest(),"avx2": xx.has_avx2(),"results": results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    lines = ["# Otsu input distributions", "", f"{args.size}² uint8 pixels, one worker, AVX2, seven samples. Same process and shuffled timing order; allocation-inclusive Python calls.", "",
             "| Input | Previous ms | Current ms | Old/current | scikit-image/current |",
             "|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['pattern']} | {r['baseline_ms']:.4f} | {r['cpp_ms']:.4f} | {r['improvement']:.2f}x | {r['speedup']:.2f}x |")
    args.output.with_suffix(".md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
