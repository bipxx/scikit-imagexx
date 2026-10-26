"""Prospective moments-family cases; collection imports no image package.

Finite policy, frozen before candidate outputs: eps is the larger supported
input/output component epsilon, L=max(1,sum(shape)), and g=8*eps*L. For output
multi-index a, raw/supplied-center S_a=sum|I|*product|x-c|**a and the allowance
is g*(|reference|+S_a). Automatic-center uses c=sum(I*x)/sum(I), k=sum|I|/|sumI|,
T_a=sum|I|*product(|x|+|c|)**a, and g*|reference|+g*(1+2*sum(a)*k)*T_a.
Centroid axis d uses g*|reference_d|+2*g*k*(max|x_d|+|c_d|)/(1-g*k).
Finite signed fixtures require k<=8 and g*k<1/4. These engineering envelopes
cover differing typed contraction associations, not arbitrary conditioning.
Zero-mass/nonfinite inputs retain exact masks/warnings and remaining finite
values. Source-initialized analytical zeros stay exact; incidental cancellation
zeros use the finite envelope. Coordinate powers and owned analytical Python
stages may remain providers; each ordinary contraction must enter C++.
"""
import copy
import itertools
import math

APIS = ("moments", "moments_central", "centroid")
EPS_FACTOR = 8
MAX_MASS_CONDITION = 8
UNASSESSED = ("forged or stale ndarray storage", "complex nonfinite parameters",
              "finite mass condition above 8", "arbitrary array/protocol subclasses")


def cases():
    rows = []

    def add(ident, api="moments", shape=(4, 5), dtype="float64", *, order=3,
            center=None, spacing=None, layout="C", pattern="positive",
            outcome="success", errstate=None, extra=None, exact=False):
        rows.append(dict(id=ident, api=api, shape=shape, dtype=dtype, order=order,
            center=center, spacing=spacing, layout=layout, pattern=pattern,
            outcome=outcome, errstate=errstate or {}, extra=extra or {}, exact=exact))

    for api in APIS:
        for shape in ((7,), (4, 5), (3, 4, 2)):
            for dtype in ("uint8", "float16", "float32", "float64"):
                add(f"{api}-{len(shape)}d-{dtype}", api, shape, dtype)
    for order in (0, 1, 2, 4):
        add(f"raw-order-{order}", order=order, dtype="float32")
        add(f"auto-order-{order}", "moments_central", order=order,
            dtype="float32", outcome="error" if order == 0 else "success")
    for order in (0, 4):
        add(f"supplied-order-{order}", "moments_central", center=(0., 0.), order=order)
    for layout in ("F", "step", "reverse", "endian", "metadata"):
        add("raw-layout-" + layout, dtype="float32", layout=layout)
    for layout in ("F", "step"):
        add("central-layout-" + layout, "moments_central", center=(1.25, 2.5), layout=layout)
    add("centroid-layout-reverse", "centroid", dtype="float32", layout="reverse")
    for name, spacing in (
            ("tuple", (.5, 2.)),
            ("array-f64", {"kind": "array", "dtype": "float64", "values": (.5, 2.)}),
            ("scalars-f32", {"kind": "scalars", "dtype": "float32", "values": (.5, 2.)}),
            ("negative-zero", (-.5, 0.)),
            ("complex", (.5 + .25j, 2. - .5j))):
        add("raw-spacing-" + name, dtype="float32", spacing=spacing)
    add("supplied-fractional-anisotropic", "moments_central", dtype="float32",
        center=(.25, 1.5), spacing=(.5, 2.))
    add("auto-anisotropic", "moments_central", dtype="float32", spacing=(.5, 2.))
    add("auto-complex-spacing-cast", "moments_central", order=2,
        spacing=(.5 + .25j, 2. - .5j))
    add("supplied-complex-center", "moments_central", dtype="float32",
        center=(.25 + .5j, 1.5 - .25j))
    add("centroid-spacing-array-f32", "centroid", dtype="float32",
        spacing={"kind": "array", "dtype": "float32", "values": (.5, 2.)})
    for api in APIS:
        for pattern in ("zero", "signed", "nan", "inf"):
            add(api + "-" + pattern, api, pattern=pattern)
    add("raw-spacing-protocol", spacing={"kind": "tracked", "values": (.5, 2.)})
    add("central-parameter-protocol", "moments_central",
        center={"kind": "tracked", "values": (.25, 1.5)},
        spacing={"kind": "tracked", "values": (.5, 2.)})
    add("central-ignored-kwargs", "moments_central", extra={"unused": ["unchanged"]})
    add("raw-numpy-order", order={"kind": "scalar", "dtype": "int32", "value": 2})
    add("raw-fractional-order", order=2.5)
    add("centroid-zero-sum", "centroid", pattern="zero-sum")
    for api in APIS:
        add(api + "-complex-image-error", api, dtype="complex64", outcome="error")
    add("raw-list-image-error", layout="list", outcome="error")
    add("raw-scalar-spacing-error", spacing=1., outcome="error")
    add("raw-short-spacing-error", spacing=(1.,), outcome="error")
    add("central-scalar-center-error", "moments_central", center=0., outcome="error")
    add("central-short-center-error", "moments_central", center=(0.,), outcome="error")
    for api in APIS:
        add(api + "-empty", api, shape=(0, 4))
    add("raw-four-dimensional", shape=(2, 2, 3, 2), order=2)
    add("central-four-dimensional", "moments_central", shape=(2, 2, 3, 2), order=3)
    add("raw-large-worker-boundary", shape=(512, 513), dtype="uint8", order=1,
        pattern="ones", exact=True)
    add("raw-overflow-default", shape=(64,), dtype="float32", order=0,
        pattern="overflow", outcome="fp")
    add("raw-overflow-raise", shape=(64,), dtype="float32", order=0,
        pattern="overflow", outcome="fp", errstate={"over": "raise", "invalid": "raise"})
    add("raw-underflow-warn", shape=(4, 5), dtype="float32", order=2,
        pattern="underflow", spacing=(.1, .2), errstate={"under": "warn"})
    add("raw-power-overflow-raise", dtype="float32", order=4,
        spacing={"kind": "scalars", "dtype": "float32", "values": (1.e20, 1.)},
        outcome="fp", errstate={"over": "raise", "invalid": "raise"})
    assert len(rows) == 100 and len({r["id"] for r in rows}) == 100
    return rows


class TrackedSequence:
    def __init__(self, values, role, events):
        self.values, self.role, self.events = tuple(values), role, events

    def __getitem__(self, index):
        self.events.append((self.role, index))
        return self.values[index]


def parameter(value, np, events, role):
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    kind = value["kind"]
    if kind == "tracked":
        return TrackedSequence(value["values"], role, events)
    if kind == "array":
        return np.asarray(value["values"], dtype=value["dtype"])
    if kind == "scalars":
        return tuple(getattr(np, value["dtype"])(x) for x in value["values"])
    if kind == "scalar":
        return getattr(np, value["dtype"])(value["value"])
    raise AssertionError(kind)


def fixture(spec, np):
    shape, dtype = spec["shape"], np.dtype(spec["dtype"])
    image = (np.arange(math.prod(shape)).reshape(shape) % 9 + 1).astype(dtype)
    if dtype.kind == "f":
        image /= 4
    pattern = spec["pattern"]
    if pattern == "zero":
        image.fill(0)
    elif pattern == "ones":
        image.fill(1)
    elif pattern == "signed":
        image = ((np.arange(image.size).reshape(shape) % 7) - 2).astype(dtype)
    elif pattern == "zero-sum":
        image = (1 - 2 * (np.arange(image.size).reshape(shape) % 2)).astype(dtype)
    elif pattern in ("nan", "inf"):
        image.flat[0] = getattr(np, pattern)
    elif pattern == "overflow":
        image.fill(np.finfo(dtype).max / 8)
    elif pattern == "underflow":
        image.fill(np.finfo(dtype).tiny / 4)
    layout = spec["layout"]
    if layout == "F":
        image = np.asfortranarray(image)
    elif layout == "step":
        backing = np.full(tuple(2 * n for n in shape), -99, dtype=dtype)
        index = (slice(None, None, 2),) * len(shape)
        backing[index] = image
        image = backing[index]
    elif layout == "reverse":
        image = image[(slice(None, None, -1),) * len(shape)]
    elif layout == "endian":
        image = image.astype(dtype.newbyteorder())
    elif layout == "metadata":
        image = image.view(np.dtype(dtype, metadata={"moments": "preserved"}))
    elif layout == "list":
        image = image.tolist()
    events = []
    kwargs = copy.deepcopy(spec["extra"])
    if spec["api"] != "centroid":
        kwargs["order"] = parameter(spec["order"], np, events, "order")
    if spec["api"] == "moments_central":
        kwargs["center"] = parameter(spec["center"], np, events, "center")
    kwargs["spacing"] = parameter(spec["spacing"], np, events, "spacing")
    return image, kwargs, events


def values(value):
    return value["values"] if isinstance(value, dict) else value


def error_limits(spec, image, expected, np):
    """Input-derived cell bounds, evaluated before native traps or outputs."""
    weights = image.astype(np.float64)
    if spec["exact"] or not np.all(np.isfinite(weights)):
        return np.zeros(expected.shape, dtype=np.float64)
    eps_in = float(np.finfo(np.float32 if image.dtype.kind == "f" and image.dtype.itemsize <= 4 else np.float64).eps)
    eps_out = float(np.finfo(expected.real.dtype).eps)
    g = EPS_FACTOR * max(eps_in, eps_out) * max(1, sum(image.shape))
    reference_abs = np.abs(expected.astype(np.complex128 if expected.dtype.kind == "c" else np.float64))
    spacing = values(spec["spacing"])
    if spacing is None:
        spacing = (1.,) * image.ndim
    coordinates = [np.arange(n, dtype=np.float64) * spacing[d] for d, n in enumerate(image.shape)]
    if any(not np.all(np.isfinite(x)) for x in coordinates):
        return np.zeros(expected.shape, dtype=np.float64)
    absolute = np.abs(weights)
    mass, total = float(np.sum(weights, dtype=np.float64)), float(np.sum(absolute, dtype=np.float64))
    auto = spec["api"] == "moments_central" and spec["center"] is None
    if auto or spec["api"] == "centroid":
        if mass == 0 or not math.isfinite(mass) or not math.isfinite(total):
            return np.zeros(expected.shape, dtype=np.float64)
        condition = total / abs(mass)
        assert condition <= MAX_MASS_CONDITION and g * condition < .25
        center = []
        for d, x in enumerate(coordinates):
            shape = [1] * image.ndim
            shape[d] = image.shape[d]
            center.append(np.sum(weights * x.reshape(shape)) / mass)
        if spec["api"] == "centroid":
            radius = np.asarray([max(np.abs(x), default=0.) for x in coordinates])
            return g * reference_abs + 2 * g * condition * (radius + np.abs(center)) / (1 - g * condition)
    else:
        center = (0.,) * image.ndim if spec["api"] == "moments" else values(spec["center"])
        condition = 0.
    bound = np.empty(expected.shape, dtype=np.float64)
    for alpha in itertools.product(*(range(n) for n in expected.shape)):
        terms = absolute.copy()
        for d, degree in enumerate(alpha):
            distances = np.abs(coordinates[d]) + abs(center[d]) if auto else np.abs(coordinates[d] - center[d])
            shape = [1] * image.ndim
            shape[d] = image.shape[d]
            terms *= (distances ** degree).reshape(shape)
        scale = float(np.sum(terms, dtype=np.float64))
        bound[alpha] = g * float(reference_abs[alpha]) + g * (1 + 2 * sum(alpha) * condition) * scale
    return bound


def initialized_zeros(spec, expected, np):
    mask = np.zeros(expected.shape, dtype=bool)
    if spec["api"] == "moments_central" and spec["center"] is None:
        order = expected.shape[0] - 1
        fast = expected.ndim in (2, 3) and order < 4
        for alpha in itertools.product(*(range(n) for n in expected.shape)):
            mask[alpha] = sum(alpha) > order or (fast and sum(alpha) == 1)
    return mask
