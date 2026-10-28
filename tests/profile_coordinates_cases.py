"""Additive coordinate-only holdouts; no image or SciPy C sampler is called.

These do not recapture or replace the earlier46 public Profile references.
"""


class EndpointProbe:
    def __init__(self, values, events, role, np, fail=False):
        self.values, self.events, self.role, self.np, self.fail = values, events, role, np, fail

    def __array__(self, dtype=None, copy='not-passed'):
        self.events.append(dict(event='endpoint.array', role=self.role, dtype=str(dtype), copy=copy))
        if self.fail:
            raise TypeError('coordinate endpoint probe')
        return self.np.asarray(self.values, dtype=dtype)


class WidthProbe:
    def __init__(self, value, events):
        self.value, self.events = value, events

    def __sub__(self, other):
        self.events.append(dict(event='width.sub', other=other))
        return self.value - other

    def __index__(self):
        self.events.append(dict(event='width.index'))
        return self.value


def cases():
    rows = []
    for side in (256, 1024):
        for width in (1, 9, 33):
            rows.append(dict(id=f'benchmark-{side}-width{width}', kind='benchmark', side=side, width=width))
    for name, src, dst in (
        ('horizontal', (2., 1.), (2., 8.)),
        ('reverse-vertical', (7., 2.), (1., 2.)),
        ('negative-border', (-2.25, -1.5), (3.5, 4.75)),
        ('subpixel', (.125, .375), (2.875, 4.125)),
        ('degenerate', (3., 4.), (3., 4.)),
        ('signed-zero', (-0., -0.), (0., -0.)),
    ):
        for width in (1, 2, 9, 33):
            rows.append(dict(id=f'{name}-width{width}', kind='ordinary', src=src, dst=dst, width=width))
    for name in ('tiny-endpoint', 'tiny-angle', 'tiny-angle-raise', 'tiny-angle-warn', 'tiny-angle-call', 'large-offset',
                 'largest-finite', 'overflow-endpoints', 'nan-endpoint', 'inf-endpoint',
                 'width-zero', 'width-negative', 'width-float', 'width-bool',
                 'width-numpy-int', 'width-zero-dimensional', 'width-protocol',
                 'endpoint-protocol', 'endpoint-protocol-error', 'short-source', 'long-source'):
        rows.append(dict(id=name, kind=name))
    assert len(rows) == 51 and len({r['id'] for r in rows}) == 51
    return rows


def fixture(spec, np):
    events = []; src, dst = (.25, .5), (2.25, 3.5); width = 3; policy = {}
    kind = spec['kind']
    if kind == 'benchmark':
        side = spec['side']; src, dst, width = (.25, 1.5), (side-1.25, side-1.75), spec['width']
    elif kind == 'ordinary': src, dst, width = spec['src'], spec['dst'], spec['width']
    elif kind == 'tiny-endpoint': src, dst, width = (0., 0.), (np.nextafter(0., 1.), 0.), 33
    elif kind.startswith('tiny-angle'):
        src, dst, width = (0., 0.), (np.nextafter(0., 1.), 1.), 2
        if kind.rsplit('-', 1)[-1] in ('raise', 'warn', 'call'):
            policy = {'under': kind.rsplit('-', 1)[-1]}
    elif kind == 'large-offset': src, dst, width = (1e15, -1e15), (1e15+3, -1e15+4), 33
    elif kind == 'largest-finite': src = dst = (np.finfo(float).max, 0.); width = 33
    elif kind == 'overflow-endpoints': src, dst = (-np.finfo(float).max, 0.), (np.finfo(float).max, 0.)
    elif kind == 'nan-endpoint': dst = (np.nan, 3.)
    elif kind == 'inf-endpoint': dst = (np.inf, 3.)
    elif kind == 'width-zero': width = 0
    elif kind == 'width-negative': width = -1
    elif kind == 'width-float': width = 1.5
    elif kind == 'width-bool': width = True
    elif kind == 'width-numpy-int': width = np.int64(3)
    elif kind == 'width-zero-dimensional': width = np.array(3)
    elif kind == 'width-protocol': width = WidthProbe(3, events)
    elif kind.startswith('endpoint-protocol'):
        src = EndpointProbe(src, events, 'src', np, fail=kind.endswith('error'))
        dst = EndpointProbe(dst, events, 'dst', np)
    elif kind == 'short-source': src = (1.,)
    elif kind == 'long-source': src = (1., 2., 3.)
    return (src, dst, width), events, policy
