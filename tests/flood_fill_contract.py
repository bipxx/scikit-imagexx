"""Prospective bounded recipes. Importing this module never requests science."""
from math import prod

DTYPES = ('uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16',
          'int32', 'int64', 'float32', 'float64')


def recipes():
    base = []
    for dtype in DTYPES:
        base.append(dict(name='dtype-' + dtype, dtype=dtype))
    for layout in ('F', 'reverse', 'step', 'transpose', 'readonly', 'metadata', 'swapped'):
        base.append(dict(name='layout-' + layout, layout=layout))
    for name, changes in (
        ('one-dimensional', dict(shape=[5], seed=[2])),
        ('three-dimensional', dict(shape=[2, 3, 4], seed=[0, 1, 1])),
        ('singleton', dict(shape=[1, 1], seed=[0, 0])),
        ('empty', dict(shape=[0, 3], seed=[999, 999])),
        ('negative-seed', dict(seed=[-1, -1])),
        ('bad-seed', dict(seed=[3, 0])),
        ('scalar-seed-1d', dict(shape=[5], seed=2)),
        ('connectivity-one', dict(connectivity=1)),
        ('connectivity-zero', dict(connectivity=0)),
        ('connectivity-high', dict(connectivity=9)),
        ('footprint-cross', dict(footprint='cross')),
        ('footprint-no-center', dict(footprint='no-center')),
        ('footprint-center-only', dict(footprint='center-only')),
        ('footprint-large', dict(footprint='large')),
        ('footprint-empty', dict(footprint='empty')),
        ('footprint-even', dict(footprint='even')),
        ('footprint-rank', dict(footprint='rank')),
        ('tolerance-zero', dict(tolerance=0)),
        ('tolerance-positive', dict(tolerance=1)),
        ('tolerance-negative', dict(tolerance=-1)),
        ('tolerance-numpy', dict(tolerance='np-int8-one')),
        ('tolerance-float', dict(dtype='float32', tolerance=0.5)),
        ('tolerance-inf', dict(dtype='float64', tolerance='inf')),
        ('seed-nan', dict(dtype='float64', pattern='nan')),
        ('seed-inf', dict(dtype='float64', pattern='inf')),
        ('float16', dict(dtype='float16')),
        ('bool', dict(dtype='bool')),
        ('complex', dict(dtype='complex64')),
    ):
        base.append(dict(name=name, **changes))
    out = []
    for api in ('flood', 'flood_fill'):
        for changes in base:
            row = dict(api=api, dtype='int16', shape=[3, 4], seed=[1, 1],
                       layout='C', pattern='plateau', footprint=None,
                       connectivity=None, tolerance=None, in_place=False,
                       new_value=7)
            row.update(changes)
            row['id'] = api + ':' + row.pop('name')
            out.append(row)
    for name, changes in (
        ('in-place', dict(in_place=True)),
        ('in-place-readonly', dict(in_place=True, layout='readonly')),
        ('new-value-fraction', dict(new_value=1.5)),
        ('new-value-out-of-range', dict(dtype='uint8', new_value=300)),
        ('new-value-complex', dict(new_value='complex-one')),
        ('new-value-array', dict(new_value='array-two')),
    ):
        row = dict(out[len(base)])
        row.update(changes, id='flood_fill:' + name)
        out.append(row)
    return out


def bounds(row):
    """Conservative source-derived ceilings, no simulated algorithm/gold."""
    shape = row['shape']
    assert 1 <= len(shape) <= 3 and all(type(n) is int and 0 <= n <= 5 for n in shape)
    logical = prod(shape)
    radius = 2 if row['footprint'] == 'large' else 1
    padded = prod(n + 2 * radius for n in shape)
    neighbors = (2 * radius + 1) ** len(shape)
    # Queue marks before push: each interior sample is enqueued at most once.
    # Offset-address reads stay in the padded box; border2 never enters queue.
    return dict(logical=logical, physical_input=logical * 2,
                padded=padded, queue_entries=logical,
                neighbor_checks=logical * neighbors,
                storage_bytes=logical * 2 * 16 + padded * 17 +
                max(64, logical * 2) * 8 + neighbors * 40)


def factory(row):
    """Future safe NumPy-only setup; NOT executed by the source-first barrier."""
    import numpy as np
    bounds(row)
    shape = tuple(row['shape'])
    a = np.array([(i // 2) % 3 for i in range(prod(shape))], dtype=row['dtype']).reshape(shape)
    if row['pattern'] in ('nan', 'inf'):
        a[tuple(row['seed'])] = float(row['pattern'])
    layout = row['layout']
    if layout == 'F':
        a = np.array(a, order='F')
    elif layout == 'transpose':
        a = np.array(a.T, order='C').T
    elif layout == 'reverse':
        a = a[::-1]
    elif layout == 'step':
        backing = np.zeros((shape[0] * 2,) + shape[1:], dtype=a.dtype)
        backing[::2] = a
        a = backing[::2]
    elif layout == 'readonly':
        a.flags.writeable = False
    elif layout == 'metadata':
        a = a.view(np.dtype(a.dtype, metadata={'source': 'flood-reference'}))
    elif layout == 'swapped':
        a = a.astype(a.dtype.newbyteorder('S'))
    kwargs = {}
    footprint = row['footprint']
    if footprint is not None:
        side = 5 if footprint == 'large' else 2 if footprint == 'even' else 3
        f = np.ones((side, side), dtype=bool)
        if footprint in ('cross', 'no-center', 'center-only', 'empty'):
            f[:] = False
            if footprint == 'cross':
                f[1, :] = True
                f[:, 1] = True
            elif footprint == 'no-center':
                f[0, 1] = f[1, 2] = True
            elif footprint == 'center-only':
                f[1, 1] = True
        if footprint == 'rank':
            f = np.ones(3, dtype=bool)
        kwargs['footprint'] = f
    if row['connectivity'] is not None:
        kwargs['connectivity'] = row['connectivity']
    tolerance = row['tolerance']
    if tolerance is not None:
        kwargs['tolerance'] = np.int8(1) if tolerance == 'np-int8-one' else float('inf') if tolerance == 'inf' else tolerance
    seed = tuple(row['seed']) if isinstance(row['seed'], list) else row['seed']
    args = [a, seed]
    if row['api'] == 'flood_fill':
        value = row['new_value']
        if value == 'complex-one':
            value = complex(1, 1)
        elif value == 'array-two':
            value = np.array([1, 2], dtype='int16')
        args.append(value)
        kwargs['in_place'] = row['in_place']
    return args, kwargs


def require_reference_authority(binding=None):
    # No capture/controller has been admitted. A truthy envelope is NOT a GO.
    raise RuntimeError('UNBOUND: root source-safety, actual prefix/controller and explicit capture GO required')


def invoke_reference(recipe_id, binding=None):
    require_reference_authority(binding)
    # Scientific imports and target resolution intentionally absent in v1.
