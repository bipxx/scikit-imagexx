"""Prospective deterministic colocalization inputs; importing never imports NumPy.

No function here calls an image algorithm or a timer. fixture() receives the
already authenticated original NumPy module explicitly, after authorisation.
"""
APIS = ('intersection_coeff', 'manders_coloc_coeff', 'manders_overlap_coeff')
DTYPES = ('bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16',
          'int32', 'int64', 'float16', 'float32', 'float64')
LAYOUTS = ('C', 'F', 'step', 'reverse', 'transpose', 'readonly', 'unaligned', 'endian')
SHAPES = {'scalar': [], 'vector': [17], 'volume': [3, 4, 5], 'empty': [0],
          'zero_axis': [2, 0, 3], 'singleton': [1, 1]}
MAX_ELEMENTS = 1024 * 1024
MAX_WORKSPACE_BYTES = 128 * 1024 * 1024


def cases():
    rows = []
    def add(api, group, name, **options):
        rows.append(dict(id=api + '/' + group + '/' + name, api=api,
                         group=group, name=name, dtype='float64', shape=[5, 7],
                         layout='C', policy='warn', trace=True, **options))
    # Use update, rather than duplicate keyword defaults, for explicit overrides.
    def item(api, group, name, **options):
        add(api, group, name)
        rows[-1].update(options)
    for api in APIS:
        for dtype in DTYPES: item(api, 'dtype', dtype, dtype=dtype)
        for dtype in DTYPES[1:9]:
            for bound in ('min','max'):item(api,'limit',dtype+'-'+bound,dtype=dtype,bound=bound)
        for dtype in DTYPES[9:]:
            for bound in ('smallest_subnormal','max'):item(api,'limit',dtype+'-'+bound,dtype=dtype,bound=bound)
        for layout in LAYOUTS: item(api, 'layout', layout, dtype='float32', layout=layout)
        for name, shape in SHAPES.items(): item(api, 'shape', name, shape=shape)
        for name in ('all_true', 'all_false', 'sparse', 'negative_hidden', 'wrong_shape', 'invalid'):
            item(api, 'mask', name)
        for name in ('zero', 'negative0', 'negative1', 'nan0', 'inf0', 'tiny',
                     'large', 'overflow0', 'overflow1', 'mismatch'):
            item(api, 'special', name, dtype='uint8' if name.startswith('overflow') else 'float32')
        for name in ('nan0', 'inf0', 'overflow0', 'tiny'):
            for policy in ('ignore', 'raise', 'call'):
                item(api, 'fp', name + '-' + policy, special=name,
                     dtype='uint8' if name == 'overflow0' else 'float32', policy=policy)
        for name in ('list', 'array_provider0', 'array_provider1', 'array_provider_roi',
                     'array_raises0', 'array_raises1', 'tagged', 'object_numeric',
                     'object_raises', 'warning_mutation'):
            item(api, 'protocol', name, dtype='float32', policy='call' if name=='warning_mutation' else 'warn')
        for name in ('invalid0', 'invalid1', 'invalid_roi', 'complex_binary'):
            item(api, 'binary_control', name)
    for api in APIS[1:]:
        for size in (7, 8, 127, 128, 129, 8191, 8192, 8193):
            item(api, 'reduction', str(size), shape=[size], dtype='float32')
    assert len(rows) == 286 and len({r['id'] for r in rows}) == 286
    return rows


def benchmark_cases():
    """Prospective only; no timing or candidate admission exists in this packet."""
    layouts = (('tiny', [32, 32], 'C'), ('cache', [256, 256], 'C'),
               ('memory', [1024, 1024], 'C'), ('volume', [64, 64, 64], 'C'),
               ('stepped', [256, 256], 'step'), ('masked', [256, 256], 'C'))
    return [dict(id=api + '/future/' + name, api=api, group='future', name=name,
                 shape=shape, layout=layout, dtype='uint8' if api == APIS[0] else 'float64' if name in ('memory','volume') else 'float32',
                 policy='default', trace=False, roi=name == 'masked')
            for api in APIS for name, shape, layout in layouts]


def certificate(spec):
    """Worst live byte accounting includes owner padding, operands and temporaries."""
    n = 1
    for dim in spec['shape']:
        assert type(dim) is int and dim >= 0
        n *= dim
    assert n <= MAX_ELEMENTS
    # At most 16-byte complex controls; stepped owner <=2N; unaligned owner
    # +1; three conversion/mask copies, two gathered inputs, three typed
    # products, min/sum iterator workspace and input snapshots. 112N+64KiB
    # dominates those simultaneous allocations; zero/scalar use N>=1 below.
    bound = 112 * max(n, 1) + 65536
    assert bound <= MAX_WORKSPACE_BYTES
    return dict(elements=n, byte_bound=bound, limit=MAX_WORKSPACE_BYTES,
                safe_domain='Owned bounded NumPy arrays/Python scalars only; no Cython/image kernel',
                unsafe_descriptor_probes=False)


def fixture(spec, np, events):
    """Return (args, kwargs, retained arrays) without executing any API."""
    certificate(spec)
    shape = tuple(spec['shape']);n = 1
    for d in shape: n *= d
    dtype = np.dtype(spec['dtype'])
    index = np.arange(n, dtype=np.int64).reshape(shape)
    a = np.asarray((index * 3 + 1) % 13, dtype=dtype)
    b = np.asarray((index * 5 + 2) % 11, dtype=dtype)
    if dtype.kind == 'f':
        a = a / dtype.type(8); b = b / dtype.type(8)
    if spec['api'] == APIS[0]:
        a = np.asarray(index % 3 != 0, dtype=dtype)
        b = np.asarray(index % 5 != 0, dtype=dtype)
    elif spec['api'] == APIS[1]:
        b = index % 3 != 0
    name=spec['name'];group=spec['group'];special=spec.get('special',name)
    roi = None
    if group=='limit':
        info=np.finfo(dtype) if dtype.kind=='f' else np.iinfo(dtype)
        bound=spec['bound']
        exact=(info.smallest_subnormal if bound=='smallest_subnormal' else getattr(info,bound))
        a[...] = exact
        if spec['api']==APIS[2]:b[...] = exact
    if group == 'mask' or spec.get('roi'):
        roi = np.ones(shape, dtype=bool)
        if name == 'all_false': roi[...] = False
        elif name in ('sparse','masked'): roi = index % 3 != 0
        elif name == 'negative_hidden' and n:
            roi = index != 0
            if spec['api'] != APIS[0]: a.flat[0] = -1
        elif name == 'wrong_shape': roi = np.ones((2,), dtype=bool)
        elif name == 'invalid': roi = np.full(shape, 2, dtype=np.uint8)
    if group in ('special','fp'):
        if special == 'zero': a[...] = 0; b[...] = 0
        elif special == 'negative0' and n: a.flat[0] = -1
        elif special == 'negative1' and n: b.flat[0] = -1
        elif special in ('nan0','inf0','tiny','large') and n:
            value = {'nan0':np.nan,'inf0':np.inf,'tiny':2.0**-120,'large':2.0**120}[special]
            if special in ('tiny','large'): a[...] = value; b[...] = value
            else: a.flat[0] = value
        elif special == 'overflow0': a[...] = 255; b[...] = 2
        elif special == 'overflow1': a[...] = 16; b[...] = 16
        elif special == 'mismatch': b = np.ones((2,), dtype=dtype)
    if group == 'binary_control':
        if name == 'invalid0': a.flat[0] = 2
        elif name == 'invalid1': b = np.array(b, dtype=np.float64); b.flat[0] = 2
        elif name == 'invalid_roi': roi = np.full(shape, 2, dtype=np.uint8)
        elif name == 'complex_binary':
            a = np.asarray(a != 0, dtype=np.complex128); b = np.asarray(b != 0, dtype=np.complex128)
    layout=spec['layout']
    def storage(value):
        if layout == 'F': return np.array(value, order='F')
        if layout == 'step' and value.ndim:
            owner=np.zeros(value.shape[:-1]+(value.shape[-1]*2,),dtype=value.dtype)
            view=owner[...,::2];view[...] = value;return view
        if layout == 'reverse' and value.ndim: return value[::-1]
        if layout == 'transpose' and value.ndim == 2: return value.T
        if layout == 'readonly': value.setflags(write=False);return value
        if layout == 'unaligned':
            owner=bytearray(value.nbytes+1)
            result=np.ndarray(value.shape,dtype=value.dtype,buffer=owner,offset=1)
            result[...] = value;return result
        if layout == 'endian' and value.dtype.itemsize > 1:
            return value.byteswap().view(value.dtype.newbyteorder())
        return value
    a,b=storage(a),storage(b)
    if roi is not None and layout == 'transpose' and roi.ndim == 2:roi=roi.T
    retained=[a,b]+([roi] if isinstance(roi,np.ndarray) else [])
    if group == 'protocol':
        class Provider:
            def __init__(self,value,label,raises=False):self.value=value;self.label=label;self.raises=raises
            def __array__(self,dtype=None,copy=None):
                events.append(dict(event='__array__',label=self.label,dtype=None if dtype is None else str(dtype),copy=copy))
                if self.raises:raise TypeError('colocalization array provider rejected '+self.label)
                result=self.value if dtype is None else self.value.astype(dtype,copy=False)
                return result.copy() if copy is True else result
        if name == 'list':a=a.tolist();b=b.tolist()
        elif name.startswith('array_'):
            if name.endswith('roi'):roi=Provider(np.ones(shape,dtype=bool),'roi')
            elif name.endswith('0'):a=Provider(a,'0',name=='array_raises0')
            else:b=Provider(b,'1',name=='array_raises1')
        elif name == 'tagged':
            class Tagged(np.ndarray):pass
            a=a.view(Tagged);b=b.view(Tagged)
        elif name == 'object_numeric':a=a.astype(object);b=b.astype(object);retained += [a,b]
        elif name == 'object_raises':
            class Token:
                def __lt__(self,other):events.append(dict(event='object-lt'));raise TypeError('colocalization object comparison rejected')
                def __eq__(self,other):events.append(dict(event='object-eq'));raise TypeError('colocalization object equality rejected')
            a=np.full(shape,Token(),dtype=object);retained += [a]
        elif name == 'warning_mutation':
            # Bounded finite mutation happens only in an actual FP call handler;
            # source computation/handler order determines whether it is observed.
            a[...] = 0;b[...] = 0
            if a.size:a.flat[0]=1
    return (a,b),({} if roi is None else {'mask':roi}),retained
