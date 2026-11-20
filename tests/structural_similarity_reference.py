"""Original-only fixtures for the bounded uniform-window SSIM score."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import skimage
from skimage.metrics import structural_similarity

# The uniform passes, products and moment/map arithmetic keep the upstream
# working dtype. The final mean sums in float64 in a different order. Allow
# two float32 ulps near one and 1e-13 float64 absolute error for filter drift,
# arbitrary-size pairwise summation and the squared stabilization constants.
ATOL = {"float32": 2e-7, "float64": 1e-13}


def cases():
    for dtype in ("float32", "float64"):
        for name, shape, window, sample, data_range in (
            ("single-center", (3, 3), 3, True, 0.5),
            ("signed", (17, 31), 5, False, 2.5),
            ("default-window", (33, 24), None, True, 1.0),
            ("wide-window", (255, 256), 255, True, 4.0),
            ("constant", (11, 21), 7, True, 1.0),
            ("near-constant", (21, 13), 9, False, 1.0),
            ("cutover", (513, 513), 11, True, 1.0),
        ):
            index = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
            left = ((index * 19 % 127 - 63) * (data_range / 64)).astype(dtype)
            right = ((index * 29 % 151 - 75) * (data_range / 76)).astype(dtype)
            if name == "constant":
                left.fill(0.125)
                right.fill(-0.125)
            elif name == "near-constant":
                left = (0.5 + left * 2**-15).astype(dtype)
                right = (0.5 + right * 2**-15).astype(dtype)
            yield f"{dtype}-{name}", left, right, {
                "data_range": data_range, "win_size": window, "use_sample_covariance": sample,
            }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert skimage.__version__ == "0.26.0"
    result = {
        "reference": skimage.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "python": platform.python_version(),
        "upstream_sha256": sha256(inspect.getsourcefile(structural_similarity)),
        "observer_sha256": sha256(__file__), "atol": ATOL, "cases": {},
    }
    for label, left, right, options in cases():
        score = structural_similarity(left, right, **options)
        assert type(score) is np.float64
        result["cases"][label] = {
            "input_sha256": [hashlib.sha256(image.tobytes()).hexdigest() for image in (left, right)],
            "shape": list(left.shape), "options": options, "score_hex": float(score).hex(),
        }
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
