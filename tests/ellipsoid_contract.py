"""Finite first ellipsoid partition; factories never generate spatial grids."""
import math

API = 'draw.ellipsoid'
MAX_AXIS = 8
MAX_GRID_AXIS = 80
MAX_OUTPUT_ELEMENTS = MAX_GRID_AXIS ** 3
UNASSESSED = (
    'Nonfinite/nonpositive spacing, unbounded extents and allocation failures',
    'NumPy scalar/array geometry, arbitrary scalar/sequence protocols and subclasses',
    'Nonbool levelset values and custom truth/conversion callbacks',
    'Exceptional floating environments and successful nonfinite outputs',
    'Profiles/platforms outside the eventual retained observations',
)


def cases():
    rows = []
    for name, axes, spacing in (
        ('unit', [1, 1, 1], [1., 1., 1.]),
        ('unequal', [2, 3, 4], [1., 1., 1.]),
        ('tiny', [.125, .25, .5], [1., 1., 1.]),
        ('fractional', [1.5, 2.25, 3.75], [1., 1., 1.]),
        ('anisotropic', [2, 3, 4], [.5, 1., 2.]),
        ('nonbinary-spacing', [1.25, 2.5, 3.75], [.75, 1.25, .5]),
        ('step-not-exact', [1.1, 2.2, 3.3], [.3, .6, 1.2]),
        ('permuted', [4, 2, 3], [2., .5, 1.]),
        ('bounded-large', [8, 7, 6], [.25, .5, 1.]),
    ):
        for levelset in (False, True):
            rows.append(dict(id=f'ellipsoid/{name}-levelset-{levelset}', api=API,
                             axes=axes, spacing=spacing, levelset=levelset,
                             kind='bounded', holdout=name in ('permuted', 'step-not-exact')))
    rows.append(dict(id='ellipsoid/defaults', api=API, axes=[2, 3, 4],
                     spacing=[1., 1., 1.], levelset=False, kind='defaults', holdout=False))
    for axis in range(3):
        for value in (0, -1):
            axes = [2, 3, 4]
            axes[axis] = value
            rows.append(dict(id=f'ellipsoid/error-axis-{axis}-{value}', api=API,
                             axes=axes, spacing=[1., 1., 1.], levelset=False,
                             kind='nonpositive-axis', holdout=False))
    for length in (1, 2, 4):
        rows.append(dict(id=f'ellipsoid/error-spacing-length-{length}', api=API,
                         axes=[2, 3, 4], spacing=[1.] * length, levelset=False,
                         kind='spacing-length', holdout=False))
    return rows


def source_safe(spec):
    """Bound the pinned source before any original call; not a numeric oracle."""
    assert spec['api'] == API and type(spec['levelset']) is bool
    axes, spacing = spec['axes'], spec['spacing']
    assert len(axes) == 3
    assert all(type(v) in (int, float) and math.isfinite(v) and -1 <= v <= MAX_AXIS for v in axes)
    assert all(type(v) is float and .25 <= v <= 2 for v in spacing)
    if spec['kind'] == 'nonpositive-axis':
        assert any(v <= 0 for v in axes)
        return dict(pregrid_rejection=True, max_output_elements=0)
    assert all(v > 0 for v in axes)
    if spec['kind'] == 'spacing-length':
        assert len(spacing) in (1, 2, 4) and all(v == 1 for v in spacing)
        # Length2/4 fails bounded broadcasting. Length1 broadcasts the offset,
        # finishes the small first-axis arange, then rejects spacing[1].
        return dict(pregrid_rejection=True, max_output_elements=0)
    assert spec['kind'] in ('bounded', 'defaults') and len(spacing) == 3
    bounds = []
    for axis, step in zip(axes, spacing):
        low, high = math.ceil(-axis-step), math.floor(axis+step+1)
        if (high-low) % 2 == 0:
            low -= 1
        # Source may translate low toward zero by less than one positive step.
        # Padding two points covers this adjustment and ceil arithmetic. This
        # is deliberately a conservative allocation bound, not expected shape.
        bound = math.ceil((high-low)/step)+2
        assert 0 < bound <= MAX_GRID_AXIS
        bounds.append(bound)
    return dict(pregrid_rejection=False, max_axis_elements=bounds,
                max_output_elements=math.prod(bounds))


def fixture(spec, np):
    bound = source_safe(spec)
    args = tuple(spec['axes'])
    options = {} if spec['kind'] == 'defaults' else dict(spacing=tuple(spec['spacing']), levelset=spec['levelset'])
    return args, options, bound
