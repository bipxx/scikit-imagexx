"""Prospective inertia-tensor cases and operation-derived cell allowances.

For automatic moments, E is the unchanged central-moment allowance from
moments_cases.error_limits, evaluated on the passively captured original mu.
For supplied mu, E=0. Let m=mu[0,...,0], b=E[0,...,0], and q_k=mu[2*e_k].
For diagonal i: A=sum(abs(q_k))+abs(q_i), B=sum(E[q_k])+E[q_i].
For off-diagonal (i,j): A=abs(mu[e_i+e_j]), B=E[e_i+e_j].
Set u=eps(output component)/2 and g=2*p*u/(1-2*p*u), where p=d+3 for
diagonals (sum, subtraction, division, storage) and p=3 off-diagonal.
The per-cell allowance is
  (B + abs(reference)*b + g*(A+B+abs(reference)*abs(m))) / (abs(m)-b).
Require b<abs(m)/4; inherited signed-mass condition<=8 remains unchanged.
This bounded engineering envelope includes both arithmetic paths, preserves
units and cancellation scale, and is not a universal conditioning guarantee.
Provider-only/exceptional cells and the finite 1D diagonal remain exact.
"""
import itertools
import math

if __package__:
    from .moments_cases import error_limits as central_limits, fixture as image_fixture
else:
    from moments_cases import error_limits as central_limits, fixture as image_fixture

API = "measure.inertia_tensor"
STAGE = "inertia_tensor"
UNASSESSED = ("empty or rank-zero images", "forged/stale storage", "extended dtypes",
              "arbitrary object/subclass protocols", "signed-mass condition above8")


def cases():
    rows = []

    def add(ident, shape=(4, 5), dtype="float64", layout="C", pattern="positive",
            spacing=None, mu_dtype=None, mu_layout="C", mu_pattern="positive",
            route="tensor", outcome="return", policy=None):
        rows.append(dict(id=ident, shape=shape, dtype=dtype, layout=layout, pattern=pattern,
                         spacing=spacing, mu_dtype=mu_dtype, mu_layout=mu_layout,
                         mu_pattern=mu_pattern, route=route, outcome=outcome, policy=policy or {}))

    for shape in ((7,), (4, 5), (3, 4, 2)):
        for dtype in ("uint8", "float16", "float32", "float64"):
            add(f"auto-{len(shape)}d-{dtype}", shape, dtype)
    for layout in ("F", "step", "reverse", "endian", "metadata"):
        add("auto-layout-" + layout, dtype="float32", layout=layout)
    add("auto-spacing-anisotropic", dtype="float32", spacing=(.5, 2.))
    add("auto-spacing-array-f32", dtype="float32",
        spacing={"kind": "array", "dtype": "float32", "values": (.5, 2.)})
    add("auto-spacing-negative-zero", spacing=(-.5, 0.))
    add("auto-spacing-short", spacing=(1.,), route="error", outcome="error")
    add("auto-spacing-complex", spacing=(.5+.25j, 2.-.5j), route="provider")
    for pattern in ("zero", "signed", "nan", "inf", "single"):
        add("auto-" + pattern, pattern=pattern,
            route="provider" if pattern in ("zero", "nan", "inf") else "tensor")
    add("auto-bool", dtype="bool")
    add("auto-complex-image-error", dtype="complex64", route="error", outcome="error")
    add("auto-zero-raise", pattern="zero", route="error", outcome="error", policy={"all": "raise"})
    for shape in ((7,), (4, 5), (3, 4, 2)):
        for dtype in ("float32", "float64"):
            add(f"supplied-{len(shape)}d-{dtype}", shape, mu_dtype=dtype)
    for layout in ("F", "step", "reverse", "endian", "metadata", "readonly", "unaligned"):
        add("supplied-layout-" + layout, mu_dtype="float64", mu_layout=layout)
    for dtype in ("float16", "int8", "uint8", "bool", "complex64"):
        add("supplied-dtype-" + dtype, mu_dtype=dtype, route="provider",
            outcome="error" if dtype == "bool" else "return")
    add("supplied-ignores-image-and-spacing", dtype="complex64", pattern="nan",
        spacing="ignored", mu_dtype="float64")
    for pattern in ("zero-mass", "negative-mass", "nan", "inf"):
        add("supplied-" + pattern, mu_dtype="float64", mu_pattern=pattern,
            route="tensor" if pattern == "negative-mass" else "provider")
    add("supplied-zero-raise", mu_dtype="float64", mu_pattern="zero-mass",
        route="error", outcome="error", policy={"all": "raise"})
    add("supplied-1d-infinite-second", shape=(7,), mu_dtype="float64",
        mu_pattern="inf", route="provider")
    add("supplied-short-mu", mu_dtype="float64", mu_pattern="short", route="error", outcome="error")
    add("supplied-wrong-rank", mu_dtype="float64", mu_pattern="wrong-rank", route="error", outcome="error")
    assert len(rows) == 57 and len({r["id"] for r in rows}) == 57
    return rows


def moment_spec(spec):
    return dict(api="moments_central", shape=spec["shape"], dtype=spec["dtype"],
                layout=spec["layout"], pattern=spec["pattern"], center=None,
                order=2, spacing=spec["spacing"], extra={}, exact=False)


def fixture(spec, np):
    image, kwargs, _ = image_fixture(moment_spec(spec), np)
    if spec["pattern"] == "single":
        image.fill(0)
        image[tuple(n//2 for n in image.shape)] = 1
    options = dict(spacing=kwargs["spacing"])
    if spec["mu_dtype"] is None:
        return image, options
    dtype = np.dtype(spec["mu_dtype"])
    shape = (3,) * image.ndim
    mu = ((np.arange(math.prod(shape), dtype=np.float64).reshape(shape) + 1) / 16).astype(dtype)
    mu[(0,) * image.ndim] = 3.5 if dtype.kind in "fc" else 3
    for axis in range(image.ndim):
        index = [0] * image.ndim
        index[axis] = 2
        mu[tuple(index)] = 1.25 + axis
    for i, j in itertools.combinations(range(image.ndim), 2):
        index = [0] * image.ndim
        index[i] = index[j] = 1
        mu[tuple(index)] = .375
    if dtype.kind == "c":
        mu.flat[0] += .5j
        mu.flat[-1] += .25j
    pattern = spec["mu_pattern"]
    if pattern == "zero-mass":
        mu.flat[0] = 0
    elif pattern == "negative-mass":
        mu.flat[0] = -3.5
    elif pattern in ("nan", "inf"):
        index = [0] * image.ndim
        index[0] = 2
        mu[tuple(index)] = getattr(np, pattern)
    elif pattern == "short":
        mu = mu[:2, :]
    elif pattern == "wrong-rank":
        mu = mu.ravel()
    layout = spec["mu_layout"]
    if layout == "F":
        mu = np.asfortranarray(mu)
    elif layout == "step":
        backing = np.zeros(tuple(2*n for n in shape), dtype=dtype)
        view = backing[(slice(None, None, 2),) * len(shape)]
        view[...] = mu
        mu = view
    elif layout == "reverse":
        mu = mu[(slice(None, None, -1),) * len(shape)]
    elif layout == "endian":
        mu = mu.astype(dtype.newbyteorder("S"))
    elif layout == "metadata":
        mu = mu.view(np.dtype(dtype, metadata={"inertia-contract": True}))
    elif layout == "readonly":
        mu.flags.writeable = False
    elif layout == "unaligned":
        backing = np.empty(mu.nbytes+1, dtype=np.uint8)
        view = np.ndarray(shape, dtype=dtype, buffer=backing, offset=1)
        view[...] = mu
        mu = view
    options["mu"] = mu
    return image, options


def error_limits(spec, image, mu, expected, np):
    """Propagate the existing moment bound through the source's tiny quotient."""
    bound = np.zeros(expected.shape, dtype=np.float64)
    if spec["route"] != "tensor" or spec["pattern"] == "single":
        return bound
    d = image.ndim
    if spec["mu_dtype"] is None:
        uncertainty = central_limits(moment_spec(spec), image, mu, np)
    else:
        uncertainty = np.zeros(mu.shape, dtype=np.float64)
    m = float(mu[(0,) * d])
    b = float(uncertainty[(0,) * d])
    assert math.isfinite(m) and b >= 0 and b < abs(m)/4
    corners = [tuple(2 if k == axis else 0 for k in range(d)) for axis in range(d)]
    q = [float(mu[index]) for index in corners]
    e = [float(uncertainty[index]) for index in corners]
    u = float(np.finfo(expected.real.dtype).eps) / 2
    for i, j in itertools.product(range(d), repeat=2):
        if i == j:
            a, delta, operations = sum(map(abs, q)) + abs(q[i]), sum(e) + e[i], d+3
        else:
            index = tuple(1 if k in (i, j) else 0 for k in range(d))
            a, delta, operations = abs(float(mu[index])), float(uncertainty[index]), 3
        g = 2*operations*u / (1-2*operations*u)
        reference = abs(float(expected[i, j]))
        bound[i, j] = (delta + reference*b + g*(a+delta+reference*abs(m))) / (abs(m)-b)
    if d == 1:
        # The source sum of its single finite second moment minus itself is exact.
        bound[0, 0] = 0
    finite = np.isfinite(expected)
    assert np.all(np.isfinite(bound[finite]) & (bound[finite] >= 0))
    return bound
