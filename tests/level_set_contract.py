"""Prospective level-set inputs, frozen before original observations.

No native admission is asserted here. Every original call is bounded to at
most 1,048,576 output elements; huge/forged storage and allocation failure are
unassessed. Outputs require exact int8 words and complete storage metadata.
"""
APIS = ('segmentation.checkerboard_level_set', 'segmentation.disk_level_set')
UNASSESSED = ('huge or forged shapes/storage', 'allocation failure',
              'concurrent mutation', 'extended/object arithmetic native admission')


def cases():
    rows = []
    def add(api, name, shape=(7, 11), *, variant='tuple', options=None,
            prospective='ordinary', policy=None, benchmark=False):
        rows.append(dict(id=api.split('.')[-1]+'/'+name, api=api, shape=list(shape),
                         variant=variant, options=options or {}, prospective=prospective,
                         policy=policy or {}, benchmark=benchmark))
    checker, disk = APIS
    for api in APIS:
        for name, shape in (('one-axis',(13,)), ('default',(7,11)),
                            ('volume',(3,5,7)), ('four-axis',(2,3,2,5)),
                            ('singleton',(1,9)), ('empty-first',(0,7)),
                            ('empty-last',(7,0)), ('empty-shape',())):
            add(api,name,shape)
        for variant in ('list','array-int16','array-uint8','array-float64',
                        'array-readonly','array-strided','shape-probe','generator','scalar'):
            add(api,'shape-'+variant,variant=variant,
                prospective='ordinary' if variant in ('list','array-int16','array-uint8','array-readonly','array-strided') else 'provider-protocol')
        add(api,'negative-shape',(-2,3),prospective='source-error')
        for name, shape in (('bench-256',(256,256)),('bench-1024',(1024,1024)),
                            ('bench-volume',(64,64,64)),('bench-thin',(128,64,32))):
            add(api,name,shape,benchmark=True)
    for name,value in (('unit',1),('two',2),('oversized',100),('negative',-3),
                       ('zero',0),('bool',True),('false',False),('wide',2**70)):
        add(checker,'square-'+name,options=dict(square_size=value),
            prospective='ordinary' if name in ('unit','two','oversized','negative','bool') else 'provider-exceptional')
    for name,kind in (('float','float'),('np-float32','np-float32'),
                      ('np-int16','np-int16'),('index-probe','index-probe')):
        add(checker,'square-'+name,options=dict(square_kind=kind),prospective='provider-protocol')
    for policy in ('raise','call'):
        add(checker,'zero-'+policy,options=dict(square_size=0),policy=dict(divide=policy),prospective='provider-exceptional')
    for name,value in (('zero',0),('negative',-1),('fraction',2.5),
                       ('exact-boundary',5),('nan','nan'),('inf','inf')):
        add(disk,'radius-'+name,options=dict(radius=value),
            prospective='ordinary' if name not in ('nan','inf') else 'provider-exceptional')
    for kind in ('np-float16','np-float32','longdouble','complex'):
        add(disk,'radius-'+kind,options=dict(radius_kind=kind),prospective='provider-protocol')
    for name,value in (('integer',[2,4]),('fractional',[2.25,4.75]),
                       ('outside',[-5,17]),('broadcast',[2]),('bad-length',[1,2,3])):
        add(disk,'center-'+name,options=dict(center=value),
            prospective='source-error' if name=='bad-length' else 'ordinary')
    for kind in ('array-float32','array-readonly','array-probe'):
        add(disk,'center-'+kind,options=dict(center_kind=kind),prospective='provider-protocol')
    for side in ('below','equal','above'):
        add(disk,'strict-boundary-'+side,(5,5),options=dict(center=[0,0],radius_kind='sqrt2-'+side))
    ids=[row['id'] for row in rows]
    assert len(ids)==len(set(ids))
    return rows


class ShapeProbe:
    def __init__(self, values, events): self.values,self.events=values,events
    def __iter__(self):
        self.events.append('shape.__iter__')
        return iter(self.values)
    def __repr__(self): return 'ShapeProbe'


class IndexProbe:
    def __init__(self, events): self.events=events
    def __index__(self):
        self.events.append('square.__index__')
        return 3
    def __repr__(self): return 'IndexProbe'


class CenterProbe:
    def __init__(self, np, events): self.np,self.events=np,events
    def __array__(self, dtype=None, copy=None):
        self.events.append(dict(event='center.__array__',dtype=str(dtype),copy=copy))
        return self.np.array([2.25,4.75],dtype=dtype)
    def __repr__(self): return 'CenterProbe'


def fixture(spec, np):
    events=[]; shape=tuple(spec['shape']);variant=spec['variant']
    if variant=='list':shape=list(shape)
    elif variant.startswith('array-'):
        dtype={'array-int16':np.int16,'array-uint8':np.uint8,'array-float64':np.float64}.get(variant,np.int64)
        if variant=='array-strided':
            array=np.zeros(len(shape)*2,dtype=dtype);array[::2]=shape;shape=array[::2]
        else:shape=np.array(shape,dtype=dtype)
        if variant=='array-readonly':shape.flags.writeable=False
    elif variant=='shape-probe':shape=ShapeProbe(shape,events)
    elif variant=='generator':shape=(value for value in shape)
    elif variant=='scalar':shape=7
    options=dict(spec['options'])
    if 'square_kind' in options:
        kind=options.pop('square_kind')
        options['square_size']={'float':lambda:3.0,'np-float32':lambda:np.float32(3),
                                'np-int16':lambda:np.int16(3),'index-probe':lambda:IndexProbe(events)}[kind]()
    if 'radius' in options and isinstance(options['radius'],str):options['radius']=float(options['radius'])
    if 'radius_kind' in options:
        kind=options.pop('radius_kind')
        if kind.startswith('sqrt2-'):
            value=np.sqrt(np.float64(2));side=kind.split('-')[1]
            options['radius']=np.nextafter(value,-np.inf) if side=='below' else np.nextafter(value,np.inf) if side=='above' else value
        else:options['radius']={'np-float16':np.float16,'np-float32':np.float32,'longdouble':np.longdouble,'complex':complex}[kind](2.5)
    if 'center_kind' in options:
        kind=options.pop('center_kind')
        if kind=='array-probe':options['center']=CenterProbe(np,events)
        else:
            options['center']=np.array([2.25,4.75],dtype=np.float32 if kind=='array-float32' else np.float64)
            if kind=='array-readonly':options['center'].flags.writeable=False
    return shape,options,events
