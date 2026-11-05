"""Finite original-first Bezier cases; factories never call draw algorithms."""
API = 'draw.bezier_curve'
MAX_OUTPUT_ELEMENTS = 4096
UNASSESSED = (
    'shape tuples shorter than2: original unchecked helper lowering is unsafe/unassessed; no capture',
    'nonfinite, subnormal or exceptional weights and unbounded geometry',
    'signed-size/int32 intermediate overflow and allocation failure',
    'arbitrary protocols beyond the explicitly journalled scalar providers',
    'concurrent mutation and malformed external storage',
)


def cases():
    recipes = [
        ('point', [2,3,2,3,2,3,1.], None),
        ('horizontal', [0,0,0,3,0,6,1.], None),
        ('vertical', [0,1,3,1,6,1,1.], None),
        ('diagonal', [0,0,3,3,6,6,1.], None),
        ('reverse-diagonal', [6,6,3,3,0,0,1.], None),
        ('zero-weight', [0,0,2,4,6,6,0.], None),
        ('monotone', [0,0,1,4,6,6,1.], None),
        ('row-mirror', [6,0,5,4,0,6,1.], None),
        ('reverse-monotone', [6,6,1,4,0,0,1.], None),
        ('clipped', [-2,1,0,4,4,6,1.], [4,5]),
        ('shape-extra', [0,0,2,5,6,6,1.], [5,5,99]),
        ('numpy-int64', [0,0,1,4,6,6,1.], None),
        ('column-extremum-unit', [1,1,11,11,21,1,1.], None),
        ('column-extremum-weighted', [1,1,11,11,21,1,2.], None),
        ('row-extremum-unit', [1,1,11,11,1,21,1.], None),
        ('row-extremum-weighted', [1,1,11,11,1,21,2.], None),
        ('both-extrema-unit', [0,0,20,20,10,5,1.], None),
        ('both-extrema-weighted', [0,0,20,20,10,5,2.], None),
        ('flat-quarter-weight', [0,0,0,10,2,10,.25], None),
        ('half-weight', [0,0,1,4,6,6,.5], None),
        ('heavy-weight', [0,0,1,4,6,6,2.], None),
        ('reverse-weighted', [6,6,1,4,0,0,2.], None),
        ('negative-translated', [-3,-3,-2,1,3,3,.5], None),
        ('empty-shape', [1,1,11,11,21,1,2.], [0,8]),
        ('endpoint-provider', [0,0,1,4,6,6,1.], None),
        ('weight-provider', [0,0,1,4,6,6,1.], None),
        ('float32-weight', [0,0,1,4,6,6,.5], None),
        ('shape-readonly-view', [0,0,1,4,6,6,1.], [5,6]),
        ('endpoint-type', ['bad',0,1,4,6,6,1.], None),
        ('endpoint-overflow', [2**80,0,1,4,6,6,1.], None),
        ('weight-type', [0,0,1,4,6,6,'bad'], None),
    ]
    return [dict(id='bezier/'+name, api=API, kind=name, args=args,
                 shape=shape, policy={}, max_output_elements=MAX_OUTPUT_ELEMENTS)
            for name,args,shape in recipes]


def fixture(spec, np):
    # Only the entry validates the closed manifest. Internal consumers get the
    # frozen literal recipe; no call to the original or native algorithm.
    assert spec in cases()
    args = list(spec['args']); options = {}; events = []; buffers = {}
    if spec['shape'] is not None:
        options['shape'] = tuple(spec['shape'])
    if spec['kind'] == 'numpy-int64':
        args[:6] = [np.int64(value) for value in args[:6]]
        args[6] = np.float64(args[6])
    elif spec['kind'] in ('endpoint-provider', 'weight-provider'):
        if __package__:
            from .draw_coordinates_contract import ScalarProvider
        else:
            from draw_coordinates_contract import ScalarProvider
        index = 0 if spec['kind'] == 'endpoint-provider' else 6
        args[index] = ScalarProvider(args[index], events, 'endpoint' if index == 0 else 'weight')
    elif spec['kind'] == 'float32-weight':
        args[6] = np.float32(args[6])
    elif spec['kind'] == 'shape-readonly-view':
        base = np.array([6,5], dtype=np.int64)
        shape = base[::-1]; shape.flags.writeable = False
        options['shape'] = shape; buffers.update(shape_base=base, shape=shape)
    return tuple(args), options, events, buffers
