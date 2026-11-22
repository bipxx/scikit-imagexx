"""Frozen case recipes and original-only observer for the Hausdorff family."""

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
import warnings

import numpy as np


def masks(shape_a, shape_b, points_a, points_b, dtype=bool):
    a = np.zeros(shape_a, dtype=dtype)
    b = np.zeros(shape_b, dtype=dtype)
    for point in points_a:
        a[point] = 1
    for point in points_b:
        b[point] = 1
    return a, b


def semantic_cases():
    tiny = masks((17, 19), (17, 19), [(1, 2), (8, 4), (14, 18), (6, 6)],
                 [(3, 2), (7, 6), (10, 12), (11, 4)])
    yield "ordinary", *tiny, True
    yield "readonly", tiny[0].view(), tiny[1], True
    a, b = tiny[0].view(), tiny[1]
    a.flags.writeable = False
    yield "readonly-flag", a, b, True
    yield "both-empty", np.zeros((3, 4), bool), np.zeros((6, 7), bool), True
    yield "left-empty", np.zeros_like(tiny[0]), tiny[1], True
    yield "right-empty", tiny[0], np.zeros_like(tiny[1]), True
    yield "rank-zero-extent", np.zeros((0, 3, 1), bool), np.zeros((2, 1, 0), bool), True
    yield "shape-mismatch", *masks((4, 8), (7, 5), [(2, 7), (3, 3)],
                                   [(1, 2), (5, 0), (6, 4)]), True
    yield "dimension-mismatch", *masks((12,), (4, 4), [(2,), (5,)], [(1, 2)]), False
    yield "dimension-mismatch-empty", np.zeros((4,), bool), tiny[0], False
    yield "three-dimensional", *masks((3, 4, 5), (3, 4, 5),
                                      [(0, 2, 4), (2, 3, 1)],
                                      [(2, 0, 1), (0, 2, 3), (0, 0, 0)]), True
    yield "four-dimensional", *masks((4, 4, 5, 3), (4, 4, 5, 3),
                                      [(0, 2, 4, 2), (1, 3, 1, 1)],
                                      [(2, 0, 1, 1), (0, 2, 3, 0)]), False
    yield "nonzero-float", np.array([0., -2., np.nan, np.inf]), np.array([2., 0., -0., 0.]), False
    yield "nonzero-uint8", tiny[0].astype("uint8") * 7, tiny[1].astype("uint8") * 13, True
    yield "strided", tiny[0][:, ::-1], tiny[1][:, ::-1], False
    yield "list-inputs", [1, 0, 1], [0, 0, 1], False
    yield "scalar", np.array(1), np.array(0), False
    # First directed maximum among equally attaining query points wins.
    yield "attainer-query-tie", *masks((13,), (13,), [(6,)], [(2,), (10,)]), True
    # Equal forward/backward maxima choose a query from the second input.
    yield "direction-maximum-tie", *masks((12,), (12,), [(1,), (6,)], [(3,), (8,)]), True
    yield "nearest-tie-leaf", *masks((18,), (18,), [(2,), (10,)], [(2,), (6,), (10,)]), False
    yield "nearest-tie-tree", *masks((60,), (60,), [(i,) for i in (*range(19), 23, 27)],
                                     [(i,) for i in (*range(19), 23, 25, 27)]), False
    yield "identical", tiny[0], tiny[0], True


def benchmark_cases():
    yield "tiny", *masks((17, 19), (17, 19), [(1, 2), (8, 4), (14, 18), (6, 6)],
                         [(3, 2), (7, 6), (10, 12), (11, 4)])
    for name, shape, first_count, second_count, seed in (
        ("rank1-sparse", (1024,), 18, 24, 17),
        ("rank2-sparse", (128, 128), 48, 72, 84),
        ("rank3-sparse", (18, 19, 20), 36, 52, 438),
        ("rank2-wide", (256, 320), 96, 128, 1221),
    ):
        rng = np.random.Generator(np.random.PCG64(seed))
        a = np.zeros(shape, dtype=bool)
        b = np.zeros(shape, dtype=bool)
        a.ravel()[rng.choice(a.size, size=first_count, replace=False)] = True
        b.ravel()[rng.choice(b.size, size=second_count, replace=False)] = True
        yield name, a, b


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def type_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def image_signature(image):
    image = np.asarray(image)
    return dict(shape=image.shape, strides=image.strides, dtype=image.dtype.str,
                data=sha256(image.tobytes(order="C")).hexdigest())


def description(value):
    if isinstance(value, tuple):
        return dict(type=type_name(value), items=[description(item) for item in value])
    if isinstance(value, np.ndarray):
        base = value.base
        return dict(type=type_name(value), values=value.tolist(), dtype=value.dtype.str,
                    shape=value.shape, strides=value.strides, owns=bool(value.flags.owndata),
                    writeable=bool(value.flags.writeable), base_shape=None if base is None else base.shape,
                    base_strides=None if base is None else base.strides)
    return dict(type=type_name(value), hex=float(value).hex())


def capture(function, left, right, kwargs):
    before = [image_signature(left), image_signature(right)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = function(left, right, **kwargs)
        except Exception as error:
            output = dict(error=dict(type=type_name(error), message=str(error)))
        else:
            output = dict(result=description(value))
    assert before == [image_signature(left), image_signature(right)]
    return dict(**output, warnings=[dict(type=type_name(w.message), message=str(w.message)) for w in caught])


def main():
    import scipy
    import skimage
    from skimage.metrics import hausdorff_distance, hausdorff_pair

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == "win32" and sys.version_info[:3] == (3, 13, 14)
    assert (skimage.__version__, scipy.__version__, np.__version__) == ("0.26.0", "1.18.1", "2.5.4")
    source = Path(inspect.getsourcefile(hausdorff_pair))
    assert source == Path(inspect.getsourcefile(hausdorff_distance))
    rows = []
    for name, left, right, _ in (*semantic_cases(), *(row + (True,) for row in benchmark_cases())):
        rows.append(dict(id=name, inputs=[image_signature(left), image_signature(right)],
            standard=capture(hausdorff_distance, left, right, {}),
            modified=capture(hausdorff_distance, left, right, {"method": "modified"}),
            pair=capture(hausdorff_pair, left, right, {})))
    method = next(semantic_cases())
    rows.append(dict(id="method-unknown", result=capture(
        hausdorff_distance, method[1], method[2], {"method": "other"})))
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(dict(source_path=str(source), source_sha256=digest(source),
            observer_sha256=digest(__file__), python=sys.version, scipy=scipy.__version__,
            numpy=np.__version__, scikit_image=skimage.__version__, rows=rows), output, indent=2)
        output.write("\n")
    print(f"captured {len(rows)} Hausdorff observations")


if __name__ == "__main__":
    main()
