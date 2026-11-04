"""Finite, source-audited first returned-object partition; no image calls."""
API = 'measure.regionprops'
PROPERTIES = ('label', 'slice', 'bbox', 'image', 'coords', 'num_pixels', 'area')
ALIASES = {'Label': 'label', 'Slice': 'slice', 'BoundingBox': 'bbox',
           'Image': 'image', 'Coordinates': 'coords', 'Area': 'area'}
UNASSESSED = (
    'Other canonical properties and their aliases, intensity/channel behavior and tables',
    'extra_properties callbacks, arbitrary array/protocol subclasses and concurrent mutation',
    'pickle/equality/repr, object assignment/lifecycle, descriptor and private cache fields',
    'huge/sparse maximum labels, allocation failure and nonordinary spacing/offset domains',
    'native implementation, numerical parity or performance credit',
)


def cases():
    two = dict(shape=[3, 4], values=[0, 3, 3, 0, 1, 0, 3, 0, 1, 1, 0, 0])
    three = dict(shape=[2, 3, 4], values=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0])
    rows = []
    for ndim, geometry in ((2, two), (3, three)):
        for cache in (True, False):
            rows.append(dict(id=f'regionprops-state/{ndim}d-cache-{cache}', **geometry, cache=cache))
    rows += [dict(id='regionprops-state/empty-2d', shape=[0, 4], values=[], rejection='empty'),
             dict(id='regionprops-state/empty-3d', shape=[2, 0, 3], values=[], rejection='empty'),
             dict(id='regionprops-state/background', shape=[2, 3], values=[0]*6),
             dict(id='regionprops-state/one-label', shape=[2, 3], values=[1]*6),
             dict(id='regionprops-state/negative-background', shape=[2, 3], values=[-1, 2, 0, 2, -1, 0])]
    for dtype in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'uint64'):
        rows.append(dict(id='regionprops-state/dtype-'+dtype, **two, dtype=dtype))
    for layout in ('fortran', 'transpose', 'reverse'):
        rows.append(dict(id='regionprops-state/'+layout, **two, layout=layout))
    rows += [dict(id='regionprops-state/spacing-offset-2d', **two, spacing=[2., 0.5], offset=[-2, 3], offset_kind='tuple'),
             dict(id='regionprops-state/spacing-offset-3d', **three, spacing=[0.5, 2., 3.], offset=[1, -2, 4], offset_kind='array'),
             dict(id='regionprops-state/float-offset', **two, offset=[0.5, -1.25]),
             dict(id='regionprops-state/input-mutation-cache-on', **two, mutation=True),
             dict(id='regionprops-state/input-mutation-cache-off', **two, mutation=True, cache=False),
             dict(id='regionprops-state/error-1d', shape=[3], values=[0, 1, 0], rejection='ndim'),
             dict(id='regionprops-state/error-float-label', **two, dtype='float64', rejection='dtype'),
             dict(id='regionprops-state/error-bool-label', shape=[2, 2], values=[0, 1, 0, 1], dtype='bool', rejection='dtype'),
             dict(id='regionprops-state/error-offset', **two, offset=[1], rejection='offset'),
             dict(id='regionprops-state/error-spacing', **two, spacing=[1., 2., 3.], rejection='spacing')]
    result = []
    for row in rows:
        row = dict(dtype='int64', cache=True, layout='c', spacing=None, offset=None,
                   offset_kind='list', mutation=False, rejection=None) | row
        row['access_sequence'] = ([['attr', name] for name in PROPERTIES]
                                  + [['attr', 'image'], ['attr', 'coords']]
                                  + [['item', name] for name in PROPERTIES]
                                  + [['item', name] for name in ALIASES]
                                  + [['iter', None], ['item', 'not_a_property'], ['attr', 'Area']])
        if row['mutation']:
            row['access_sequence'] += [['mutate_input', None]] + [['attr', name] for name in ('image', 'num_pixels', 'area', 'coords')]
        result.append(row)
    return result


def source_safe(spec):
    shape = spec['shape']; count = 1
    assert len(shape) in (1, 2, 3) and all(type(v) is int and 0 <= v <= 4 for v in shape)
    for v in shape:
        count *= v
    assert count == len(spec['values']) and count <= 64
    assert all(type(v) is int and -1 <= v <= 3 for v in spec['values'])
    assert spec['dtype'] in ('bool', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64')
    assert spec['layout'] in ('c', 'fortran', 'transpose', 'reverse')
    assert type(spec['cache']) is bool
    assert spec['rejection'] in (None, 'ndim', 'dtype', 'offset', 'spacing', 'empty')
    if spec['rejection'] is None:
        assert len(shape) in (2, 3) and spec['dtype'] not in ('bool', 'float64')
    for name in ('spacing', 'offset'):
        values = spec[name]
        if values is not None:
            assert all(type(v) in (int, float) and abs(v) <= 4 for v in values)
            if spec['rejection'] != name:
                assert len(values) == len(shape)
    if spec['spacing'] is not None:
        assert all(v > 0 for v in spec['spacing'])
    if spec['mutation']:
        assert spec['layout'] == 'c' and len(shape) == 2 and 1 in spec['values']
    return dict(input_elements=count, maximum_label=max(spec['values'], default=0),
                region_count_upper_bound=3, coords_elements_upper_bound=3*count,
                preentry_rejection=spec['rejection'] in ('ndim', 'dtype', 'offset'))


def fixture(spec, np):
    source_safe(spec)
    image = np.array(spec['values'], dtype=spec['dtype']).reshape(spec['shape'])
    if spec['layout'] == 'fortran':
        image = np.asfortranarray(image)
    elif spec['layout'] == 'transpose':
        image = image.T
    elif spec['layout'] == 'reverse':
        image = image[::-1]
    options = dict(cache=spec['cache'])
    if spec['spacing'] is not None:
        options['spacing'] = tuple(spec['spacing'])
    if spec['offset'] is not None:
        value = spec['offset'][:]
        if spec['offset_kind'] == 'tuple':
            value = tuple(value)
        elif spec['offset_kind'] == 'array':
            value = np.array(value, dtype=np.int64)
        options['offset'] = value
    return image, options
