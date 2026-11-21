"""Capture scoped Pearson behavior from the pinned Windows originals."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import struct
import sys
import warnings

import numpy as np
import scipy
import skimage
from scipy.stats import pearsonr
from skimage.measure import pearson_corr_coeff

VERSIONS = {"numpy": "2.5.4", "scipy": "1.18.1", "scikit_image": "0.26.0"}
SOURCE_HASHES = {
    "skimage": "22dbf52758f0aaa1c0b1e5175058472efb7efd3c0e239d0da7326bdc5e6d488d",
    "scipy": "2f10dc94cbecdc7fe119c35955509b468c9fcf36ed8e98e1f57f1acc35cf6023",
}
POOLS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def type_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def descriptor(array):
    value = np.asarray(array)
    return {
        "type": type_name(array), "dtype": value.dtype.str, "shape": list(value.shape),
        "strides": list(value.strides), "c_order": bool(value.flags.c_contiguous),
        "f_order": bool(value.flags.f_contiguous), "aligned": bool(value.flags.aligned),
        "writeable": bool(value.flags.writeable),
        "logical_sha256": None if value.dtype.hasobject else sha256(value.tobytes(order="C")),
        "object_items": (
            [repr(item) for item in value.flat] if value.dtype.hasobject else None
        ),
    }


def semantic_cases():
    x = np.array([1, 4, 2, 8, 5, 0, 7, 3], dtype=np.float64).reshape(2, 4)
    y = np.array([3, 1, 4, 7, 2, 5, 0, 6], dtype=np.float64).reshape(2, 4)
    selected = np.array([[False, True, False, True], [True, False, False, True]])
    n = np.arange(8, dtype=np.float64)
    tail_x = np.arange(17, dtype=np.float64)
    tail_delta = np.where(tail_x % 2 == 0, 1.0, -1.0)
    constant = np.full(8, 3.0)
    near = 1.0 + (n - 3.0) * 2**-45
    near32 = (1.0 + (n - 3.0) * 2**-20).astype("float32")
    reversed_x = x[:, ::-1]
    readonly = x.copy()
    readonly.flags.writeable = False

    cases = [
        ("plain-f64", x, y, None, 8),
        ("plain-f32", x.astype("float32"), y.astype("float32"), None, 8),
        ("mixed-f32-f64", x.astype("float32"), y, None, 8),
        ("integer-uint8-int16", x.astype("uint8"), y.astype("int16"), None, 8),
        ("bool-images", x > 3, y < 3, None, 8),
        ("float16", x.astype("float16"), y.astype("float16"), None, 8),
        ("wide-int64", x.astype("int64") + 2**60, y.astype("int64"), None, 8),
        ("big-endian-float64", x.astype(">f8"), y.astype(">f8"), None, 8),
        ("complex", x.astype("complex128") + 1j, y, None, 8),
        ("object", x.astype(object), y.astype(object), None, 8),
        ("list-coercion", x.ravel().tolist(), y.ravel().tolist(), None, 8),
        ("reversed", reversed_x, y[:, ::-1], None, 8),
        ("fortran", np.asfortranarray(x), np.asfortranarray(y), None, 8),
        ("strided", x[:, ::2], y[:, ::2], None, 4),
        ("broadcast", np.broadcast_to(x[0], x.shape), y, None, 8),
        ("readonly", readonly, y, None, 8),
        ("shape-3d", x.reshape(2, 2, 2), y.reshape(2, 2, 2), None, 8),
        ("shape-mismatch", x, y.reshape(4, 2), None, None),
        ("shape-broadcast-is-error", x, y[:1], None, None),
        ("scalar", np.array(1.0), np.array(2.0), None, 1),
        ("scalar-masked-true", np.array(1.0), np.array(2.0), np.array(True), 1),
        ("scalar-masked-false", np.array(1.0), np.array(2.0), np.array(False), 0),
        ("empty", x[:0], y[:0], None, 0),
        ("single", x.ravel()[:1], y.ravel()[:1], None, 1),
        ("two-positive", np.array([2., 5.]), np.array([-3., 4.]), None, 2),
        ("two-negative", np.array([2., 5.]), np.array([4., -3.]), None, 2),
        ("two-constant", np.array([2., 2.]), np.array([4., -3.]), None, 2),
        ("three", np.array([0., 1., 3.]), np.array([3., 0., 1.]), None, 3),
        ("zero-correlation", np.array([-1., -1., 1., 1.]), np.array([-1., 1., -1., 1.]), None, 4),
        ("mask-bool", x, y, selected, 4),
        ("mask-all", x, y, np.ones(x.shape, dtype=bool), 8),
        ("mask-empty", x, y, np.zeros(x.shape, dtype=bool), 0),
        ("mask-single", x, y, (n.reshape(x.shape) == 1), 1),
        ("mask-two", x, y, (n.reshape(x.shape) < 2), 2),
        ("mask-three", x, y, (n.reshape(x.shape) < 3), 3),
        ("mask-bool-strided", x, y, np.ascontiguousarray(selected[:, ::-1])[:, ::-1], 4),
        ("mask-binary-u8", x, y, selected.astype("uint8"), 4),
        ("mask-binary-f64", x, y, selected.astype("float64"), 4),
        ("mask-binary-complex", x, y, selected.astype("complex128"), 4),
        ("mask-invalid-value", x, y, selected.astype("int8") - 1, None),
        ("mask-invalid-fraction", x, y, selected.astype("float64") * 0.5, None),
        ("mask-invalid-nan", x, y, np.where(selected, 1.0, np.nan), None),
        ("mask-wrong-shape", x, y, selected.T, None),
        ("mask-invalid-before-shape", x, y.T, np.full((1,), 2, dtype="uint8"), None),
        ("constant-left", constant, y.ravel(), None, 8),
        ("constant-both", constant, -constant, None, 8),
        ("near-constant-f64", near, y.ravel(), None, 8),
        ("near-constant-f32", near32, y.ravel().astype("float32"), None, 8),
        ("combined-warning-semantics", constant, near, None, 8),
        ("near-constant-passed", 1.0 + (n - 3.0) * 2**-29, y.ravel(), None, 8),
        ("perfect-plus", n, n, None, 8),
        ("perfect-minus", n, -n, None, 8),
        ("near-perfect-plus", tail_x, tail_x + tail_delta * 2**-7, None, 17),
        ("near-perfect-minus", tail_x, -tail_x + tail_delta * 2**-7, None, 17),
        ("rounded-beta-tail", tail_x, tail_x + tail_delta * 2**-22, None, 17),
        ("nan-selected", np.where(x == 1, np.nan, x), y, None, 8),
        ("nan-excluded", np.where(selected, x, np.nan), y, selected, 4),
        ("infinity-selected", np.where(x == 1, np.inf, x), y, None, 8),
        ("large-amplitude", (n - 3) * 1e200, y.ravel(), None, 8),
        ("very-small-amplitude", (n - 3) * 1e-200, y.ravel(), None, 8),
    ]
    for name, a, b, mask, count in cases:
        yield name, a, b, mask, count, "always"
    yield "constant-warning-as-error", constant, y.ravel(), None, 8, "error"
    yield "near-warning-as-error", near, y.ravel(), None, 8, "error"


BENCHMARK_SPECS = (
    ("tiny-unmasked", (2, 4), None),
    ("tiny-mask-half", (2, 4), 50),
    ("mid-unmasked", (64, 67), None),
    ("mid-mask-quarter", (64, 67), 25),
    ("large-unmasked", (513, 641), None),
    ("large-mask-half", (513, 641), 50),
    ("large-mask-sparse", (513, 641), 2),
    ("large-mask-all", (513, 641), 100),
)


def benchmark_cases():
    for index, (name, shape, density) in enumerate(BENCHMARK_SPECS):
        rng = np.random.Generator(np.random.PCG64(616633 + index))
        left = rng.integers(-32768, 32768, size=shape, dtype=np.int32).astype("float64") / 32768
        right = rng.integers(-32768, 32768, size=shape, dtype=np.int32).astype("float64") / 32768
        mask = None if density is None else rng.integers(0, 100, size=shape, dtype=np.int32) < density
        count = left.size if mask is None else int(np.count_nonzero(mask))
        yield name, left, right, mask, count, "always"


def observation(case, *, group):
    name, left, right, mask, selected_count, warning_action = case
    args = (left, right) if mask is None else (left, right, mask)
    before = [descriptor(arg) for arg in args]
    policy = np.geterr(), np.geterrcall(), np.getbufsize()
    result = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(warning_action)
        try:
            pair = pearson_corr_coeff(*args)
        except Exception as error:
            result["error"] = {"class": type_name(error), "message": str(error)}
        else:
            if type(pair) is not tuple or len(pair) != 2 or any(type(v) is not float for v in pair):
                raise AssertionError(f"unexpected Pearson return type: {name}")
            result["type"] = type_name(pair)
            result["components"] = [
                {"name": label, "type": type_name(v), "hex": v.hex(), "ieee754_be": struct.pack(">d", v).hex()}
                for label, v in zip(("r", "p"), pair)
            ]
    result["warnings"] = [{"class": type_name(w.message), "message": str(w.message)} for w in caught]
    assert before == [descriptor(arg) for arg in args], name
    assert policy == (np.geterr(), np.geterrcall(), np.getbufsize()), name
    return {
        "id": name, "group": group, "selected_count": selected_count,
        "warning_action": warning_action, "inputs": before, **result,
    }


def main():
    from numpy._core import _multiarray_umath
    from scipy.special import _ufuncs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    versions = dict(numpy=np.__version__, scipy=scipy.__version__, scikit_image=skimage.__version__)
    assert versions == VERSIONS and os.name == "nt"
    sources = {
        name: Path(inspect.getsourcefile(fn))
        for name, fn in (("skimage", pearson_corr_coeff), ("scipy", pearsonr))
    }
    hashes = {name: sha256(path.read_bytes()) for name, path in sources.items()}
    assert hashes == SOURCE_HASHES
    pool_settings = {name: os.environ.get(name) for name in POOLS}
    assert set(pool_settings.values()) == {"1"}
    binaries = {"numpy": Path(_multiarray_umath.__file__), "scipy_special": Path(_ufuncs.__file__)}
    metadata = {
        "api": "skimage.measure.pearson_corr_coeff", "versions": versions,
        "python": sys.version, "platform": platform.platform(), "executable": sys.executable,
        "source_files": {name: str(path) for name, path in sources.items()}, "source_sha256": hashes,
        "observer_sha256": sha256(Path(__file__).read_bytes()), "pool_limits": pool_settings,
        "numpy_error_policy": np.geterr(), "argv": sys.argv, "cwd": str(Path.cwd()),
        "binary_sha256": {name: sha256(path.read_bytes()) for name, path in binaries.items()},
    }
    rows = [observation(case, group=group)
            for group, cases in (("semantics", semantic_cases()), ("benchmark-input", benchmark_cases()))
            for case in cases]
    assert len({row["id"] for row in rows}) == len(rows)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump({"metadata": metadata, "rows": rows}, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"captured {len(rows)} original Pearson cases at {args.output}")


if __name__ == "__main__":
    main()
