"""Prospective perimeter policy, frozen before any original/candidate output.

The cast, erosion, uint8 subtraction/stencil and histogram are exact. For the
50-term binary64 dot use 2*gamma_100*S, S=sum(abs(w[i])*h[i]), with exact
rational coefficients/counts and one upward binary64 rounding. This covers
both multiply/add trees (50 products and at most49 additions per dot).
All assessed histogram counts are <=2**53, so integer-to-double conversion is
exact; dtype, zero sign, warnings/errors/state and histogram shape stay exact.
"""
from fractions import Fraction
import math
import warnings

API = 'measure.perimeter'
STAGES = ('perimeter_counts', 'perimeter_weighted')
UNASSESSED = ('forged/stale storage', 'arbitrary ndarray subclass/astype protocols',
              'arbitrary object conversions', 'counts above2**53', 'huge unsafe shapes')


class NeighborhoodProbe:
    def __init__(self, mode, events):
        self.mode, self.events = mode, events

    def __repr__(self):
        return 'NeighborhoodProbe(' + repr(self.mode) + ')'

    def __eq__(self, other):
        self.events.append(('neighborhood.eq', other))
        if self.mode == 'raise':
            raise RuntimeError('perimeter neighborhood probe')
        if self.mode == 'warn':
            warnings.warn('perimeter neighborhood probe', UserWarning)
        return self.mode != 'warn'


def cases():
    rows = []
    def add(id, pattern='binary', dtype='uint8', layout='C', neighborhood=4, outcome='return', policy=None):
        rows.append(dict(id=id, pattern=pattern, dtype=dtype, layout=layout,
                         neighborhood=neighborhood, outcome=outcome, policy=policy or {}))
    for pattern in ('empty00', 'empty05', 'empty40', 'single0', 'single1',
                    'row', 'column', 'rectangle', 'ring', 'diagonal'):
        add('geometry-' + pattern, pattern)
    for dtype in ('bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
                  'int64', 'uint64', 'float16', 'float32', 'float64'):
        add('dtype-' + dtype, dtype=dtype)
    for layout in ('F', 'step', 'reverse', 'endian', 'readonly', 'unaligned', 'metadata', 'broadcast'):
        add('layout-' + layout, dtype='float64', layout=layout)
    for name, value in (('eight', 8), ('numpy-int', 'np-int4'), ('numpy-float', 'np-float4'),
                        ('three', 3), ('zero', 0), ('none', None), ('text', '4'),
                        ('array-scalar', 'array-scalar'), ('array-many', 'array-many'),
                        ('eq-true', 'probe-true'), ('eq-warn', 'probe-warn'), ('eq-raise', 'probe-raise')):
        add('neighborhood-' + name, neighborhood=value,
            outcome='error' if name in ('array-many', 'eq-raise') else 'return')
    add('nonbinary-single2', 'single2')
    add('nonbinary-single255', 'single255', outcome='error')
    add('nonbinary-full2', 'full2', outcome='source')
    add('nonbinary-castwrap', 'castwrap', dtype='uint16')
    add('nonfinite-warn', 'nonfinite', dtype='float64', outcome='source')
    add('nonfinite-raise', 'nonfinite', dtype='float64', policy={'invalid': 'raise'}, outcome='error')
    add('neighborhood-before-cast-error', 'nonfinite', dtype='float64', neighborhood='probe-raise', outcome='error')
    for pattern in ('scalar', 'rank1', 'rank3', 'list'):
        add('intake-' + pattern, pattern, outcome='error')
    add('complex-cast-provider', 'complex', dtype='complex128')
    add('object-cast-provider', 'object', dtype='object')
    assert len(rows) == 55 and len({row['id'] for row in rows}) == 55
    return rows


def fixture(spec, np):
    pattern, dtype = spec['pattern'], spec['dtype']
    shapes = {'empty00': (0, 0), 'empty05': (0, 5), 'empty40': (4, 0),
              'single0': (1, 1), 'single1': (1, 1), 'single2': (1, 1),
              'single255': (1, 1), 'row': (1, 7), 'column': (6, 1),
              'rectangle': (5, 7), 'ring': (7, 9), 'diagonal': (6, 6)}
    shape = shapes.get(pattern, (6, 8))
    if pattern in ('scalar', 'rank1', 'rank3'):
        image = np.ones({'scalar': (), 'rank1': (7,), 'rank3': (2, 3, 4)}[pattern], dtype=dtype)
    elif pattern == 'list':
        image = [[0, 1], [1, 0]]
    elif pattern == 'nonfinite':
        image = np.array([[0., -0., math.nan, math.inf], [-math.inf, -2., 3., math.nan]], dtype=dtype)
    elif pattern == 'complex':
        image = np.array([[0j, 1+1j], [1-2j, 0j]], dtype=dtype)
    elif pattern == 'object':
        image = np.array([[0, 1, 0], [1, 0, 1]], dtype=object)
    elif pattern in ('single1', 'row', 'column', 'rectangle'):
        image = np.ones(shape, dtype=dtype)
    elif pattern in ('single2', 'single255', 'full2'):
        image = np.full(shape, 255 if pattern == 'single255' else 2, dtype=dtype)
    elif pattern == 'castwrap':
        image = np.array([[256, 257, 512], [513, 256, 257]], dtype=dtype)
    elif pattern.startswith('empty') or pattern == 'single0':
        image = np.zeros(shape, dtype=dtype)
    elif pattern == 'ring':
        image = np.ones(shape, dtype=dtype); image[2:-2, 2:-2] = 0
    elif pattern == 'diagonal':
        image = np.eye(6, dtype=dtype)
    else:
        values = (0, 2**63, 2**63+1, 1) if dtype == 'uint64' else (
            (-2**63, -2**63+1, 0, 1) if dtype == 'int64' else (0, 1, 1, 0))
        image = np.array([[values[(row*5+col*3) % len(values)] for col in range(shape[1])]
                          for row in range(shape[0])], dtype=dtype)
    layout = spec['layout']
    if layout == 'F': image = np.asfortranarray(image)
    elif layout == 'step':
        owner = np.zeros(tuple(2*n for n in image.shape), dtype=image.dtype)
        owner[::2, ::2] = image; image = owner[::2, ::2]
    elif layout == 'reverse': image = image[::-1, ::-1]
    elif layout == 'endian': image = image.astype(image.dtype.newbyteorder('>'))
    elif layout == 'readonly': image.flags.writeable = False
    elif layout == 'unaligned':
        owner = np.zeros(image.nbytes+1, dtype=np.uint8)
        view = np.ndarray(image.shape, dtype=image.dtype, buffer=owner, offset=1)
        view[...] = image; image = view
    elif layout == 'metadata': image = image.astype(np.dtype(image.dtype, metadata={'perimeter': 'source'}))
    elif layout == 'broadcast': image = np.broadcast_to(image[:1].copy(), image.shape)
    events = []; neighborhood = spec['neighborhood']
    if neighborhood == 'np-int4': neighborhood = np.int64(4)
    elif neighborhood == 'np-float4': neighborhood = np.float32(4)
    elif neighborhood == 'array-scalar': neighborhood = np.array(4)
    elif neighborhood == 'array-many': neighborhood = np.array([4, 8])
    elif isinstance(neighborhood, str) and neighborhood.startswith('probe-'):
        neighborhood = NeighborhoodProbe(neighborhood[6:], events)
    return image, {'neighborhood': neighborhood}, events


def error_limit(histogram, coefficients):
    assert len(histogram) == len(coefficients) == 50
    assert all(0 <= int(n) <= 2**53 for n in histogram)
    scale = sum((abs(Fraction.from_float(float(c))) * int(n)
                 for c, n in zip(coefficients, histogram)), Fraction(0))
    if scale == 0: return 0.
    allowance = float(2 * Fraction(100, 2**53-100) * scale)
    assert math.isfinite(allowance) and allowance >= 0
    return math.nextafter(allowance, math.inf)
