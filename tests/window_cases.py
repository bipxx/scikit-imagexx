"""Prospective filters.window cases; no scientific imports at collection.

Native sampling scope is orders0/1; SciPy get_window and NumPy radial-coordinate
and clipping stages are explicit providers. Orders2--5 retain original outcomes
but initially require strict refusal, with no numerical parity credit.
Before candidate outputs: relative8*eps64 plus absolute32*eps64*max(1, finite
reference-output peak). Metadata, nonfinite/zero masks, warnings, errors and
caller state stay exact.
This is an engineering envelope, not an exhaustive API-domain error theorem.
"""
import copy

RELATIVE_EPS_FACTOR = 8
PEAK_ABSOLUTE_EPS_FACTOR = 32

# These are NOT executed original cases or passing native tests. Mixed-empty
# shapes reach scipy.ndimage with a zero-size output after a nonempty provider
# window, unlike all-zero shapes rejected by warp's Python empty-image guard.
BLOCKED_INTAKE = ((0, 5), (5, 0), (2, 0, 3))


def cases():
    rows = []

    def add(ident, window="hann", shape=(5, 7), *, shape_kind="plain", kwargs=None,
            outcome="success", route="owned", output_shape=None):
        if output_shape is None and outcome == "success":
            output_shape = (int(round(shape)),) if isinstance(shape, (int, float)) else tuple(int(round(n)) for n in shape)
        rows.append(dict(id=ident, window=window, shape=shape, shape_kind=shape_kind,
                         kwargs=kwargs, outcome=outcome, route=route,
                         output_shape=output_shape))

    for name in ("boxcar", "triang", "blackman", "hamming", "hann", "bartlett",
                 "flattop", "parzen", "bohman", "blackmanharris", "nuttall",
                 "barthann", "cosine", "lanczos"):
        add("named-"+name, name, (2, 2) if name == "hamming" else (5, 7),
            kwargs={"clip": True} if name == "hamming" else None)
    for name, window in (("kaiser-float", 8.6), ("kaiser-tuple", ("kaiser", 8.6)),
            ("tukey", ("tukey", .25)), ("gaussian", ("gaussian", 1.5)),
            ("general-gaussian", ("general_gaussian", 1.5, 2.)),
            ("dpss", ("dpss", 2.)), ("chebwin", ("chebwin", 80)),
            ("chebwin-warning", ("chebwin", 20)),
            ("exponential", ("exponential", None, 2.)),
            ("general-hamming", ("general_hamming", .54)),
            ("general-cosine", ("general_cosine", [.5, .5])),
            ("taylor", ("taylor", 4, 30))):
        add("parameter-"+name, window)
    for name, shape, kind, outcome in (("scalar", 7, "plain", "success"),
            ("numpy-int", 7, "numpy-int", "success"),
            ("numpy-float", 7., "numpy-float", "success"),
            ("near-integer", 7.0004, "plain", "success"),
            ("list", (5, 7), "list", "success"),
            ("array", (6, 8), "array", "success"),
            ("array-step", (3, 8), "array-step", "success"),
            ("array-reverse", (8, 3), "array-reverse", "success"),
            ("array-endian", (5, 7), "array-endian", "success"),
            ("zero-dimensional", 7, "array-zero-d", "error")):
        add("shape-"+name, shape=shape, shape_kind=kind, outcome=outcome)
    for index, shape in enumerate(((1,), (2,), (5,), (6,), (3, 8), (8, 3),
                                   (3, 4, 5), (1, 3, 7), (3, 1, 5), (1, 1, 1))):
        add("geometry-"+str(index), "boxcar" if index in (0, 1, 9) else "hann", shape)
    for name, order, outcome in (("nearest", 0, "success"), ("linear", 1, "success"),
            ("numpy-int", ("numpy", "int64", 1), "success"),
            ("bool", True, "success"), ("float", 1., "error")):
        add("order-"+name, kwargs={"order": order}, outcome=outcome)
    for order in (2, 3, 4, 5):
        add("order-"+str(order)+"-deferred", kwargs={"order": order}, route="strict-refusal")
    add("boxcar-small-clip-false", "boxcar", (2, 2), kwargs={"clip": False})
    add("hamming-small-clip-false", "hamming", (2, 2), kwargs={"clip": False})
    add("boxcar-small-clip-true", "boxcar", (2, 2), kwargs={"clip": True})
    add("boxcar-large-actual-zero", "boxcar", (7, 7))
    add("preserve-range", kwargs={"preserve_range": True})
    add("output-shape-ignored-for-coordinates", kwargs={"output_shape": (2, 3)})
    add("map-args-unused-for-coordinates", kwargs={"map_args": {"unused": 7}})
    for name, kwargs in (("mode-constant", {"mode": "constant"}),
            ("mode-reflect", {"mode": "reflect"}), ("cval-zero", {"cval": 0.}),
            ("unknown-option", {"not_a_warp_option": 1}), ("not-mapping", ["order", 1])):
        add("options-"+name, kwargs=kwargs, outcome="error")
    for name, window in (("unknown", "unknown-window"), ("missing-parameter", "kaiser"),
            ("empty-tuple", ()), ("list", ["hann"]), ("none", None),
            ("extra-parameter", ("hann", 1))):
        add("window-error-"+name, window, outcome="error")
    for name, window in (("gaussian-zero", ("gaussian", 0.)),
            ("gaussian-nan", ("gaussian", float("nan"))),
            ("kaiser-inf", ("kaiser", float("inf")))):
        add("nonfinite-"+name, window)
    for name, shape in (("negative", (-1, 5)), ("empty-tuple", ()),
            ("fractional", (3.1, 5)), ("nan", float("nan")),
            ("inf", float("inf")), ("zero-scalar", 0), ("all-zero", (0, 0))):
        add("shape-error-"+name, shape=shape, outcome="error")
    for name, order in (("negative", -1), ("too-large", 6), ("nan", float("nan"))):
        add("order-error-"+name, kwargs={"order": order}, outcome="error")
    add("precedence-shape-before-window", "unknown-window", (-1, 5), outcome="error")
    add("precedence-window-before-mode", "unknown-window", kwargs={"mode": "reflect"}, outcome="error")
    add("parallel-sampling-boundary", "hann", (512, 513))
    assert len(rows) == 89 and len({r["id"] for r in rows}) == 89
    return rows


def fixture(spec, np):
    shape = copy.deepcopy(spec["shape"])
    kind = spec["shape_kind"]
    if kind == "numpy-int":
        shape = np.int64(shape)
    elif kind == "numpy-float":
        shape = np.float32(shape)
    elif kind == "list":
        shape = list(shape)
    elif kind in ("array", "array-zero-d"):
        shape = np.asarray(shape, dtype=np.int64)
    elif kind == "array-step":
        backing = np.full(len(shape) * 2, 99, dtype=np.int64)
        backing[::2] = shape
        shape = backing[::2]
    elif kind == "array-reverse":
        shape = np.asarray(shape[::-1], dtype=np.int64)[::-1]
    elif kind == "array-endian":
        shape = np.asarray(shape, dtype=np.dtype("int64").newbyteorder())
    kwargs = copy.deepcopy(spec["kwargs"])
    if isinstance(kwargs, dict):
        for key, value in kwargs.items():
            if isinstance(value, tuple) and value and value[0] == "numpy":
                kwargs[key] = getattr(np, value[1])(value[2])
    return copy.deepcopy(spec["window"]), shape, kwargs


def error_limits(expected, np):
    finite = np.abs(expected[np.isfinite(expected)])
    peak = max(1., float(finite.max()) if finite.size else 0.)
    eps = np.finfo(np.float64).eps
    return RELATIVE_EPS_FACTOR * eps, PEAK_ABSOLUTE_EPS_FACTOR * eps * peak
