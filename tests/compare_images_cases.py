"""Original-first compare_images contracts; exact words, metadata and protocols."""
API = 'util.compare_images'
STAGES = ('compare_images',)
UNASSESSED = ('forged storage', 'allocation exhaustion', 'arbitrary ndarray subclasses and mutated shape protocols')


def cases():
    rows = []
    def add(name, **options):
        row = dict(id=name, shape=(7, 11), dtype='float64', dtype1=None, layout='C',
                   method='diff', n_tiles=(8, 8), pattern='texture', route='native',
                   outcome='return', invocation='normal', policy={})
        row.update(options); rows.append(row)
    for method in ('diff', 'blend', 'checkerboard'):
        for dtype in ('bool', 'uint8', 'int16', 'float32', 'float64'):
            add(method+'-'+dtype, method=method, dtype=dtype)
    for method in ('diff', 'blend', 'checkerboard'):
        add('mixed-'+method, method=method, dtype='float32', dtype1='float64')
    add('mixed-integer-float', method='blend', dtype='uint8', dtype1='float32')
    for layout in ('F', 'step', 'reverse', 'endian', 'unaligned', 'readonly', 'metadata', 'broadcast'):
        add('layout-'+layout, method='checkerboard' if layout in ('F', 'reverse', 'metadata') else 'diff', layout=layout, dtype='float32')
    add('scalar-diff', shape=())
    add('rank3-blend', shape=(3, 5, 4), method='blend')
    add('channel-checker-error', shape=(3, 5, 4), method='checkerboard', route='intake', outcome='error')
    add('empty-diff', shape=(0, 7), route='intake')
    add('empty-checker', shape=(0, 7), method='checkerboard', route='intake')
    add('shape-error', invocation='shape', route='intake', outcome='error')
    for name, tiles, outcome in (('uneven', (3, 4), 'return'), ('oversized', (20, 20), 'return'),
        ('zero', (0, 2), 'error'), ('negative', (-2, 3), 'return'), ('float', (2., 3), 'error'),
        ('short', (2,), 'error'), ('none', None, 'error'), ('protocol', 'tracked', 'return')):
        add('tiles-'+name, method='checkerboard', n_tiles=tiles, outcome=outcome,
            route='intake' if outcome == 'error' else 'native')
    add('method-error', method='wrong', route='intake', outcome='error')
    add('shape-before-method', method='wrong', invocation='shape', route='intake', outcome='error')
    add('complex-error', dtype='complex64', method='wrong', route='intake', outcome='error')
    add('list-error', invocation='list', route='intake', outcome='error')
    add('overflow-blend', method='blend', pattern='overflow', route='provider')
    add('overflow-raise', method='blend', pattern='overflow', policy={'over': 'raise'}, route='provider', outcome='error')
    add('nonfinite-diff', pattern='nonfinite', route='provider')
    add('signed-zero', pattern='signedzero')
    add('shape-array-protocol', invocation='proxy', route='provider')
    add('method-protocol', method='tracked', route='provider')
    add('positional-method', method='blend', invocation='positional')
    add('duplicate-argument', invocation='duplicate', route='intake', outcome='error')
    add('surplus-args-ignored', method='blend', invocation='surplus')
    assert len(rows) == 54 and len({r['id'] for r in rows}) == 54
    return rows


def fixture(spec, np):
    shape = tuple(spec['shape']); count = __import__('math').prod(shape)
    raw = ((np.arange(count, dtype=np.int64)*37+11)%251).reshape(shape)
    images = []
    for index, name in enumerate((spec['dtype'], spec['dtype1'] or spec['dtype'])):
        dtype = np.dtype(name)
        values = raw if index == 0 else (raw*3+7)%251
        if dtype.kind == 'b': image = values%3 == 0
        elif dtype.kind in 'ui': image = values.astype(dtype)
        else: image = (values/256).astype(dtype)
        if spec['pattern'] == 'overflow': image[...] = np.finfo(dtype).max
        elif spec['pattern'] == 'nonfinite': image.flat[:4] = [np.nan, np.inf, -np.inf, -0.0]
        elif spec['pattern'] == 'signedzero': image[...] = -0.0 if index == 0 else 0.0
        layout = spec['layout']
        if layout == 'F': image = np.asfortranarray(image)
        elif layout == 'step':
            storage = np.empty(tuple(n*2 for n in shape), dtype=image.dtype)
            view = storage[tuple(slice(None,None,2) for n in shape)]; view[...] = image; image = view
        elif layout == 'reverse': image = image[::-1,::-1]
        elif layout == 'endian': image = image.astype(image.dtype.newbyteorder('>'))
        elif layout == 'unaligned':
            view = np.ndarray(shape,dtype=image.dtype,buffer=bytearray(image.nbytes+1),offset=1)
            view[...] = image; image = view
        elif layout == 'readonly': image.flags.writeable = False
        elif layout == 'metadata': image = image.view(np.dtype(image.dtype,metadata={'unit':'linear'}))
        elif layout == 'broadcast': image = np.broadcast_to(image[:1],shape)
        images.append(image)
    events = []
    options = dict(method=spec['method'],n_tiles=spec['n_tiles'])
    if spec['n_tiles'] == 'tracked':
        class Tile:
            def __init__(self, axis, value): self.axis,self.value=axis,value
            def __rtruediv__(self, other): events.append(['divide',self.axis,other]); return other/self.value
            def __index__(self): events.append(['index',self.axis]); return self.value
            def __repr__(self): return 'Tile('+str(self.axis)+')'
        class Tiles:
            def __getitem__(self,index): events.append(['tile',index]); return Tile(index,(3,4)[index])
            def __repr__(self): return 'Tiles(3,4)'
        options['n_tiles'] = Tiles()
    if spec['method'] == 'tracked':
        class Method:
            def __eq__(self, other): events.append(['method-eq',other]); return other == 'blend'
            def __repr__(self): return 'Method(blend)'
        options['method'] = Method()
    invocation = spec['invocation']
    if invocation == 'shape': images[1] = images[1][:-1]
    elif invocation == 'list': images[0] = images[0].tolist()
    elif invocation == 'proxy':
        class Image:
            def __init__(self, image, role): self.image,self.role=image,role
            @property
            def shape(self): events.append(['shape',self.role]); return self.image.shape
            def __array__(self,dtype=None,copy=None):
                events.append(['array',self.role,None if dtype is None else str(dtype),copy])
                return np.asarray(self.image,dtype=dtype)
            def __repr__(self): return 'Image('+self.role+')'
        images = [Image(image,str(index)) for index,image in enumerate(images)]
    args = list(images)
    if invocation in ('positional','surplus'):
        args += [options.pop('method'),options.pop('n_tiles')]
        if invocation == 'surplus': args += ['ignored']
    elif invocation == 'duplicate': options['image0'] = images[0]
    return args,options,events
