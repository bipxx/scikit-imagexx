"""Exact original-first Hough line case recipes."""

from hashlib import sha256


def sha(data):
    return sha256(data).hexdigest()


def snapshot(array):
    payload = array.tobytes(order="C")
    row = dict(type=type(array).__name__, dtype=array.dtype.str, metadata=array.dtype.metadata is not None,
               shape=list(array.shape), strides=list(array.strides), c_order=bool(array.flags.c_contiguous),
               f_order=bool(array.flags.f_contiguous), aligned=bool(array.flags.aligned),
               owns=bool(array.flags.owndata), writable=bool(array.flags.writeable),
               base=type(array.base).__name__, sha256=sha(payload))
    return row


def stock_cases(np):
    base = np.zeros((5, 8), dtype=np.bool_)
    base[2, 1:8] = True
    base[0:4, 4] = True
    diag = np.eye(5, 8, dtype=np.uint8)
    corner = np.zeros((3, 4), dtype=np.uint8)
    corner[2, 3] = 1
    single = np.zeros((2, 3), dtype=np.bool_)
    single[1, 0] = True
    a = float.fromhex("0x1.0c152382d7366p-1")
    around = np.array([np.nextafter(a, 0.0), a, np.nextafter(a, np.inf)], dtype=np.float64)
    signed_half = np.concatenate([around, -around])
    b = float.fromhex("0x1.0284ba3743ce5p+1")
    cross = np.array([np.nextafter(b, -np.inf), b, np.nextafter(b, np.inf)], dtype=np.float64)
    read_only = np.array([0.0, 0.8, -0.8, -0.0, 0.8, 0.1], dtype=np.float64)[::-2]
    read_only.flags.writeable = False
    axis = np.array([-np.pi / 2, 0.0, np.pi / 2, np.pi], dtype=np.float64)
    order = np.array([0.3, -0.0, 0.3, -1.0, 4.25, 0.0], dtype=np.float64)
    rows = (
        ("cross-default", base, None, "bounded"),
        ("diagonal-default", diag, None, "bounded"),
        ("no-votes-default", np.zeros((3, 6), dtype=np.bool_), None, "bounded"),
        ("single-origin-angles", np.array([[1]], dtype=np.uint8), order, "bounded"),
        ("signed-corner-extent", corner, axis, "bounded"),
        ("arbitrary-order-duplicates", diag, order, "bounded"),
        ("signed-half-and-neighbors", single, signed_half, "bounded"),
        ("mixed-coordinate-half-neighbors", corner, cross, "bounded"),
        ("zero-angles", base, np.empty(0, dtype=np.float64), "control"),
        ("zero-height", np.zeros((0, 7), dtype=np.bool_), axis, "control"),
        ("zero-both", np.zeros((0, 0), dtype=np.bool_), axis, "control"),
        ("read-only-reversed-theta", diag, read_only, "control"),
        ("fortran-image", np.asfortranarray(diag), axis, "control"),
        ("reversed-image", base[:, ::-1], axis, "control"),
        ("signed-non-binary-nonzero", np.array([[0, -2, 0], [7, 0, -1]], dtype=np.int16), axis, "control"),
        ("rank-one-image", np.zeros((3,), dtype=np.bool_), axis, "control"),
        ("rank-three-image", np.zeros((2, 2, 2), dtype=np.bool_), axis, "control"),
        ("float32-theta", base, np.array([0.1, 0.25], dtype=np.float32), "control"),
        ("list-theta", base, [0.1, 0.25], "control"),
    )
    return rows
