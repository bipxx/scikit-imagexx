"""Additive pre-candidate H-extrema boundaries; old 468 cases are unchanged."""
import itertools


def cases():
    result = []
    integers = ('u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'u8', 'i8')
    def add(function, family, dtype, tag, **options):
        result.append(dict(function=function, id=function+'/supplement/'+family+'/'+dtype+'/'+tag,
                           family=family, dtype=dtype, **options))
    for function in ('h_maxima', 'h_minima'):
        for dtype, base in itertools.product(('f2', 'f4', 'f8'), (1, 1000)):
            add(function, 'resolution-reached', dtype, str(base), base=base, required_cython=True)
        for dtype, relation, height_type in itertools.product(('f2', 'f4', 'f8'), (-1, 0, 1), ('python', 'typed')):
            add(function, 'floating-cutoff', dtype, str(relation)+'/'+height_type,
                relation=relation, height_type=height_type, required_cython=True)
        for dtype, limit in itertools.product(('f2', 'f4', 'f8'), ('tiny-normal', 'subnormal', 'large-positive', 'large-negative')):
            add(function, 'floating-real-limit', dtype, limit, limit=limit, required_cython=True)
        for dtype, edge, height in itertools.product(integers, ('low', 'high'), (1, 2, 4)):
            add(function, 'integer-saturation', dtype, edge+'/'+str(height), edge=edge, height=height)
        for dtype, height in itertools.product(integers, ('zero', 'one', 'range', 'above-range')):
            add(function, 'full-integer-span', dtype, height, height=height)
        for dtype, center, height in itertools.product(('u8', 'i8'), ('below53', 'at53', 'above53', 'large'), (1, 2, 4)):
            add(function, 'wide-integer-rounding', dtype, center+'/'+str(height), center=center, height=height)
        for dtype, relation in itertools.product(integers, (-1, 0, 1)):
            add(function, 'integer-cutoff', dtype, str(relation), relation=relation)
        for dtype, height in itertools.product(('u8', 'i8'), ('python-fraction', 'float32-fraction', 'float64-fraction', 'float64-integral', 'uint64-integral')):
            add(function, 'wide-height-promotion', dtype, height, height=height)
    assert len(result) == 348 and len({row['id'] for row in result}) == 348
    return result


def fixture(spec, events):
    import numpy as np
    dtype = np.dtype(spec['dtype'])
    family = spec['family']
    if family == 'resolution-reached':
        base = spec['base']
        step = dtype.type(4.0 * float(np.finfo(dtype).resolution) * base)
        # This spacing makes the total range exceed h for every pinned dtype.
        # The recorder separately requires an actual certified Cython call.
        image = np.asarray([base + float(step) * ((i*7) % 13) for i in range(35)], dtype=dtype).reshape(5, 7)
        height = dtype.type(float(step) / 2.0)
    elif family == 'floating-cutoff':
        values = [0, 4, 0, 8, 0] if spec['function'] == 'h_maxima' else [8, 4, 8, 0, 8]
        image = np.asarray(values, dtype=dtype)
        height = dtype.type(4)
        if spec['relation']:
            height = np.nextafter(height, dtype.type(np.inf if spec['relation'] > 0 else -np.inf))
        if spec['height_type'] == 'python':
            height = float(height)
    elif family == 'floating-real-limit':
        limit = spec['limit']
        if limit == 'tiny-normal':
            unit = dtype.type(np.finfo(dtype).tiny)
        elif limit == 'subnormal':
            unit = np.nextafter(dtype.type(0), dtype.type(1))
        else:
            unit = dtype.type(float(np.finfo(dtype).max) / 32.0)
        values = [0, 4, 0, 8, 0] if spec['function'] == 'h_maxima' else [8, 4, 8, 0, 8]
        if limit == 'large-negative':
            values = [value - 8 for value in values]
        image = np.asarray([float(unit) * value for value in values], dtype=dtype)
        height = unit
    elif family == 'integer-saturation':
        info = np.iinfo(dtype)
        minimum, maximum = int(info.min), int(info.max)
        values = ([minimum, minimum+1, minimum+4, minimum+2, minimum+8] if spec['edge'] == 'low'
                  else [maximum-8, maximum-2, maximum-4, maximum-1, maximum])
        image, height = np.asarray(values, dtype=dtype), spec['height']
    elif family == 'full-integer-span':
        info = np.iinfo(dtype)
        minimum, maximum = int(info.min), int(info.max)
        image = np.asarray([minimum, 0, maximum, minimum+1, maximum-1], dtype=dtype)
        height = {'zero': 0, 'one': 1, 'range': maximum-minimum, 'above-range': maximum-minimum+1}[spec['height']]
    elif family == 'wide-integer-rounding':
        center = {'below53': 2**53-4, 'at53': 2**53, 'above53': 2**53+8,
                  'large': 2**63+16 if dtype.kind == 'u' else -2**53-8}[spec['center']]
        image = np.asarray([center, center+1, center+2, center+4, center+8], dtype=dtype)
        height = spec['height']
    elif family == 'integer-cutoff':
        values = [0, 3, 0, 6, 0] if spec['function'] == 'h_maxima' else [6, 3, 6, 0, 6]
        image, height = np.asarray(values, dtype=dtype), 3 + spec['relation']
    elif family == 'wide-height-promotion':
        image = np.asarray([2**53, 2**53+1, 2**53+2, 2**53+4, 2**53+8], dtype=dtype)
        height = {'python-fraction': 1.5, 'float32-fraction': np.float32(1.5),
                  'float64-fraction': np.float64(1.5), 'float64-integral': np.float64(2),
                  'uint64-integral': np.uint64(2)}[spec['height']]
    else:
        raise AssertionError('unknown frozen boundary family')
    return image, height, {}


def observe(spec, legacy, function=None):
    previous = legacy.fixture
    legacy.fixture = fixture
    try:
        return legacy.observe(spec, function)
    finally:
        legacy.fixture = previous
