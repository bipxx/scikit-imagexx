"""Prospective LPI contracts; collection imports no scientific packages.

Before candidate results: L=ceil(log2(max(2, product(2*shape-1)))), with
relative32*eps*L and absolute64*eps*L*max(finite input/output peak). Output
precision determines eps. Cache comparisons use cache peak only. These are
cross-FFT-backend engineering limits, not error theorems. Nonfinite masks,
metadata, callbacks, errors, warnings and state are separate exact contracts.
Callbacks and predefined duck filters remain explicit Python providers; this
cohort requires owned FFT execution, not fully native callback arithmetic.
"""
import math

RELATIVE_EPS_FACTOR = 32
PEAK_ABSOLUTE_EPS_FACTOR = 64
OPERATIONS = ("class", "filter_forward", "filter_inverse", "wiener")


def cases():
    rows = []

    def add(ident, operation="filter_forward", *, shape=(5, 7), dtype="float32",
            layout="C", response="gaussian", scenario="once", outcome="success",
            pattern="ramp", **options):
        rows.append(dict(id=ident, operation=operation, shape=shape, dtype=dtype,
                         layout=layout, response=response, scenario=scenario,
                         outcome=outcome, pattern=pattern, options=options))

    for operation in OPERATIONS:
        for dtype in ("uint8", "float16", "float32", "float64"):
            add(f"dtype-{operation}-{dtype}", operation, dtype=dtype)
    for index, shape in enumerate(((3, 3), (4, 4), (3, 6), (6, 3), (1, 1), (1, 4))):
        add(f"geometry-{shape[0]}x{shape[1]}", OPERATIONS[index % 4], shape=shape)
    for index, layout in enumerate(("F", "step", "reverse", "endian", "readonly", "unaligned", "metadata")):
        add("layout-"+layout, OPERATIONS[index % 4], dtype="float64" if index % 2 else "float32", layout=layout)
    for index, response in enumerate(("delta", "asymmetric", "box", "complex", "float64", "uint8")):
        add("response-"+response, OPERATIONS[index % 4], response=response)
    for scenario in ("cache-hit", "shape-change", "dtype-change", "inverse-forward", "wiener-forward", "params-change"):
        add("state-"+scenario, "class", scenario=scenario,
            response="nearzero" if scenario in ("inverse-forward", "wiener-forward") else "gaussian")
    for operation in OPERATIONS[1:]:
        add("predefined-precedence-"+operation, operation, scenario="predefined")
    add("constructor-noncallable", "class", response="noncallable", outcome="error")
    for response in ("raise", "scalar", "list", "wrong-size"):
        add("callback-"+response, response=response, outcome="error")
    add("params-not-mapping", scenario="bad-params", outcome="error")
    add("input-rank-one", shape=(7,), outcome="error")
    add("input-rank-three", "filter_inverse", shape=(2, 3, 4), outcome="error")
    # Public check_nD rejects size0 before _prepare, padding or FFT; no unchecked
    # private numerical helper is called for this source-defined error case.
    add("input-empty", "wiener", shape=(0, 3), outcome="error")
    add("input-complex", "class", dtype="complex64", outcome="error")
    for name, value in (("zero", 0), ("quarter", .25), ("large", 1e6),
            ("negative", -1), ("numpy-f32", ("numpy", "float32", .25)),
            ("nan", float("nan")), ("inf", float("inf")), ("string", "two")):
        add("gain-"+name, "filter_inverse", max_gain=value,
            outcome="error" if name == "string" else "success")
    for name, value, outcome in (("zero", 0., "success"), ("negative", -.25, "success"),
            ("nan", float("nan"), "success"), ("inf", float("inf"), "success"),
            ("numpy-f32", ("numpy", "float32", .25), "error"),
            ("numpy-f64", ("numpy", "float64", .25), "success"),
            ("integer", 1, "error"), ("array-zero-d", ("array", "scalar"), "error"),
            ("array-broadcast", ("array", "broadcast"), "success"),
            ("array-padded", ("array", "padded"), "success"),
            ("array-input-shape", ("array", "input"), "error")):
        add("K-"+name, "wiener", K=value, outcome=outcome)
    add("input-nan", pattern="nan")
    add("input-inf", "filter_inverse", pattern="inf")
    add("zero-kernel-inverse", "filter_inverse", response="zero")
    add("zero-kernel-wiener", "wiener", response="zero")
    add("nearzero-inverse", "filter_inverse", response="nearzero")
    add("predefined-duck-provider", scenario="duck")
    add("predefined-invalid-inverse", "filter_inverse", scenario="invalid-predefined", outcome="error")
    assert len(rows) == 80 and len({row["id"] for row in rows}) == 80
    return rows


def fixture(spec, np, *, shape=None, dtype=None):
    shape = spec["shape"] if shape is None else shape
    dtype = np.dtype(spec["dtype"] if dtype is None else dtype)
    indices = np.arange(math.prod(shape), dtype=np.float64).reshape(shape)
    values = (indices * 7 + 3) % 251 if dtype.kind in "iu" else ((indices * 7) % 23 - 11) / 8
    image = values.astype(dtype)
    if spec["pattern"] != "ramp":
        image.flat[0] = np.nan if spec["pattern"] == "nan" else np.inf
    layout = spec["layout"]
    if layout == "F":
        image = np.asfortranarray(image)
    elif layout == "step":
        backing = np.full(shape[:-1] + (2 * shape[-1],), 99, dtype=dtype)
        backing[..., ::2] = image
        image = backing[..., ::2]
    elif layout == "reverse":
        image = image[::-1, ::-1]
    elif layout == "endian":
        image = image.astype(dtype.newbyteorder())
    elif layout == "readonly":
        image.flags.writeable = False
    elif layout == "unaligned":
        backing = np.empty(image.nbytes + 1, dtype=np.uint8)
        view = np.ndarray(shape, dtype=dtype, buffer=backing, offset=1)
        view[...] = image
        image = view
    elif layout == "metadata":
        image = image.view(np.dtype(dtype, metadata={"lpi": "retained"}))
    options = {}
    for name, value in spec["options"].items():
        if isinstance(value, tuple) and value[0] == "numpy":
            value = getattr(np, value[1])(value[2])
        elif isinstance(value, tuple) and value[0] == "array":
            array_shape = {"scalar": (), "broadcast": (1, 1), "input": shape,
                           "padded": tuple(2 * n - 1 for n in shape)}[value[1]]
            value = np.full(array_shape, .25, dtype=np.float32)
        options[name] = value
    return image, options


def finite_peak(array, np):
    finite = np.abs(array[np.isfinite(array)])
    return float(finite.max()) if finite.size else 0.


def error_limits(shape, expected, np, *, input_peak=0.):
    levels = max(1, (max(2, math.prod(2 * n - 1 for n in shape)) - 1).bit_length())
    eps = np.finfo(expected.real.dtype).eps
    return (RELATIVE_EPS_FACTOR * eps * levels,
            PEAK_ABSOLUTE_EPS_FACTOR * eps * levels * max(input_peak, finite_peak(expected, np)))
