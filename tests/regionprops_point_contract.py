"""Four ordinary single-pixel geometries; no algorithm calls."""
if __package__:
    from .regionprops_state_contract import ALIASES, PROPERTIES, source_safe
else:
    from regionprops_state_contract import ALIASES, PROPERTIES, source_safe


def cases():
    embedded_two = [0] * 12
    embedded_two[6] = 1
    embedded_three = [0] * 24
    embedded_three[18] = 1
    rows = [
        dict(id='regionprops-point/unit-2d-cache-on', shape=[1, 1], values=[1], cache=True),
        dict(id='regionprops-point/unit-3d-cache-off', shape=[1, 1, 1], values=[1], cache=False),
        dict(id='regionprops-point/embedded-2d-offset-cache-off', shape=[3, 4],
             values=embedded_two, cache=False, offset=[-2, 3]),
        dict(id='regionprops-point/embedded-3d-cache-on', shape=[2, 3, 4],
             values=embedded_three, cache=True),
    ]
    result = []
    for row in rows:
        spec = dict(dtype='int64', layout='c', spacing=None, offset=None,
                    offset_kind='list', mutation=False, rejection=None) | row
        spec['access_sequence'] = ([['attr', name] for name in PROPERTIES]
                                   + [['attr', 'image'], ['attr', 'coords']]
                                   + [['item', name] for name in PROPERTIES]
                                   + [['item', name] for name in ALIASES]
                                   + [['iter', None], ['item', 'not_a_property'], ['attr', 'Area']])
        bounds = source_safe(spec)
        assert bounds['maximum_label'] == 1 and spec['values'].count(1) == 1
        result.append(spec)
    return result
