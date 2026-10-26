"""Prospective Butterworth cases; no scientific imports at collection time.

Frozen before any candidate: finite values use eps of the *reference output*
real dtype, L=ceil(log2(max(2, padded spatial element count))), relative8*eps*L
plus absolute32*eps*L*peak(abs(input)). This is an engineering acceptance
limit, not a worst-case theorem or cross-backend bitwise claim. NaN/Inf masks,
metadata, warnings, errors, input bytes and state are compared exactly first.
Exceptional cutoff/order rows retain their actual source outcomes separately.
The small deterministic cohort is representative, not the full API domain.
"""
import math

RELATIVE_EPS_FACTOR = 8
PEAK_ABSOLUTE_EPS_FACTOR = 32


def cases():
    rows = []

    def add(ident, *, shape=(5, 7), dtype="float32", layout="C", pattern="ramp",
            outcome="success", **kwargs):
        rows.append(dict(id=ident, shape=shape, dtype=dtype, layout=layout,
                         pattern=pattern, outcome=outcome, kwargs=kwargs))

    for dtype in ("float16", "float32", "float64", "uint8", "complex64", "complex128"):
        for high in (False, True):
            add(f"dtype-{dtype}-{'high' if high else 'low'}", dtype=dtype,
                cutoff_frequency_ratio=.18, high_pass=high)
    for index, shape in enumerate(((7,), (8,), (6, 8), (6, 7), (3, 4, 5), (4, 3, 6), (1, 1))):
        add(f"geometry-{index}", shape=shape, dtype=("float32", "float64", "complex64")[index % 3],
            cutoff_frequency_ratio=.2)
    for index, (shape, axis, dtype) in enumerate((((3, 5, 7), 0, "float32"),
            ((5, 7, 3), -1, "float32"), ((2, 3, 5, 7), 0, "complex64"),
            ((3, 5, 7, 2), -1, "float64"))):
        add(f"channels-{index}", shape=shape, dtype=dtype, channel_axis=axis,
            cutoff_frequency_ratio=.2, high_pass=False)
    for layout, dtype in (("F", "float32"), ("F", "complex64"),
            ("step", "float64"), ("step", "complex128"),
            ("reverse", "float32"), ("reverse", "complex64"),
            ("endian", "float64"), ("endian", "complex128"),
            ("readonly", "float32"), ("unaligned", "float64")):
        add(f"layout-{layout}-{dtype}", dtype=dtype, layout=layout, cutoff_frequency_ratio=.23)
    for high in (False, True):
        for squared in (False, True):
            add(f"response-{high}-{squared}", shape=(5, 6), dtype="float64",
                cutoff_frequency_ratio=.25, order=3.5, high_pass=high,
                squared_butterworth=squared)
    for name, value in (("integer", 3), ("builtin-float", .75),
            ("numpy-float16", ("numpy", "float16", .75)),
            ("numpy-float32", ("numpy", "float32", .75)),
            ("numpy-float64", ("numpy", "float64", .75)),
            ("zero", 0), ("negative", -1), ("negative-fraction", -.5),
            ("nan", float("nan")), ("inf", float("inf")),
            ("complex", 1+.25j), ("string", "two")):
        add(f"order-{name}", order=value, cutoff_frequency_ratio=.2,
            outcome="error" if name == "string" else "success")
    for name, value, high, outcome in (("zero-high", 0, True, "success"),
            ("zero-low", 0, False, "success"), ("nyquist-high", .5, True, "success"),
            ("nyquist-low", .5, False, "success"), ("tiny", 1e-6, True, "success"),
            ("nan", float("nan"), True, "success"), ("inf", float("inf"), True, "error"),
            ("negative", -.01, True, "error"), ("above-nyquist", .51, True, "error"),
            ("string", "quarter", True, "error")):
        add(f"cutoff-{name}", cutoff_frequency_ratio=value, high_pass=high, outcome=outcome)
    add("pad-one", dtype="float64", npad=1, cutoff_frequency_ratio=.2)
    add("pad-two-complex", dtype="complex64", npad=2, cutoff_frequency_ratio=.2)
    add("pad-channel-zero", shape=(2, 5, 7), channel_axis=0, npad=1, cutoff_frequency_ratio=.2)
    add("pad-channel-last", shape=(5, 7, 2), channel_axis=-1, npad=2, cutoff_frequency_ratio=.2)
    add("pad-numpy-int", npad=("numpy", "int64", 2), cutoff_frequency_ratio=.2)
    add("pad-negative", npad=-1, outcome="error")
    add("pad-fraction", npad=1.5, outcome="error")
    for name, shape, kwargs in (("empty-1d", (0,), {}), ("empty-2d", (3, 0), {}),
            ("empty-3d", (0, 3, 4), {}), ("empty-padding", (0, 3), {"npad": 1}),
            ("scalar", (), {}), ("channel-oob", (3, 4), {"channel_axis": 2}),
            ("channel-fraction", (3, 4), {"channel_axis": .5})):
        add(name, shape=shape, outcome="error", **kwargs)
    for pattern, dtype in (("nan", "float32"), ("positive-inf", "float64"),
            ("negative-inf", "float32"), ("complex-nan", "complex64")):
        add("nonfinite-"+pattern, pattern=pattern, dtype=dtype, cutoff_frequency_ratio=.2)
    add("analytic-zero", shape=(4, 8), dtype="float64", pattern="zero", cutoff_frequency_ratio=.2)
    add("analytic-constant-low", shape=(4, 8), dtype="float64", pattern="constant",
        high_pass=False, cutoff_frequency_ratio=.2)
    add("analytic-constant-high", shape=(4, 8), dtype="float64", pattern="constant",
        high_pass=True, cutoff_frequency_ratio=.2)
    add("analytic-impulse-7", shape=(7,), dtype="float64", pattern="impulse",
        high_pass=False, cutoff_frequency_ratio=.2, order=2)
    add("cutoff-numpy-float32", cutoff_frequency_ratio=("numpy", "float32", .25))
    add("all-defaults")
    add("dtype-bool", dtype="bool", cutoff_frequency_ratio=.2)
    add("dtype-int64", dtype="int64", cutoff_frequency_ratio=.2)
    add("pad-float-zero", npad=0.0, cutoff_frequency_ratio=.2)
    add("order-numpy-int64", order=("numpy", "int64", 2), cutoff_frequency_ratio=.2)
    assert len(rows) == 87 and len({row["id"] for row in rows}) == len(rows)
    return rows


def fixture(spec, np):
    shape = spec["shape"]
    indices = np.arange(math.prod(shape), dtype=np.float64).reshape(shape)
    dtype = np.dtype(spec["dtype"])
    if dtype.kind in "biu":
        image = ((indices * 7 + 3) % (2 if dtype.kind == "b" else 251)).astype(dtype)
    else:
        values = ((indices * 7) % 23 - 11) / 8
        if dtype.kind == "c":
            values = values + 1j * (((indices * 3) % 17 - 8) / 9)
        image = values.astype(dtype)
    pattern = spec["pattern"]
    if pattern in ("zero", "constant", "impulse"):
        image.fill(3 if pattern == "constant" else 0)
        if pattern == "impulse":
            image.flat[0] = 1
    elif pattern != "ramp":
        image.flat[0] = {"nan": np.nan, "positive-inf": np.inf,
                         "negative-inf": -np.inf, "complex-nan": complex(1, float("nan"))}[pattern]
    layout = spec["layout"]
    if layout == "F":
        image = np.asfortranarray(image)
    elif layout == "step":
        backing = np.empty(shape[:-1] + (shape[-1] * 2,), dtype=dtype)
        backing.fill(99)
        backing[..., ::2] = image
        image = backing[..., ::2]
    elif layout == "reverse":
        image = image[..., ::-1]
    elif layout == "endian":
        image = image.astype(dtype.newbyteorder())
    elif layout == "readonly":
        image.flags.writeable = False
    elif layout == "unaligned":
        backing = np.empty(image.nbytes + 1, dtype=np.uint8)
        view = np.ndarray(shape, dtype=dtype, buffer=backing, offset=1)
        view[...] = image
        image = view
    kwargs = {name: getattr(np, value[1])(value[2])
              if isinstance(value, tuple) and value[0] == "numpy" else value
              for name, value in spec["kwargs"].items()}
    return image, kwargs


def error_limits(spec, image, kwargs, result, np):
    """Prospective envelope, evaluated only after successful source execution."""
    pad = int(kwargs.get("npad", 0)) if kwargs.get("npad", 0) > 0 else 0
    spatial = [length + 2 * pad for length in image.shape]
    axis = kwargs.get("channel_axis")
    if axis is not None:
        del spatial[int(axis) % image.ndim]
    levels = max(1, (max(2, math.prod(spatial)) - 1).bit_length())
    eps = np.finfo(result.real.dtype).eps
    finite = np.abs(image[np.isfinite(image)])
    peak = float(finite.max()) if finite.size else 0.0
    return RELATIVE_EPS_FACTOR * eps * levels, PEAK_ABSOLUTE_EPS_FACTOR * eps * levels * peak
