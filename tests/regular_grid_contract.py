"""Prospective metadata-only grid cases; no original imported or called."""
API='util.regular_grid'
SOURCE_SHA256='64fcce3556f0cbf8cafd05a54ea3545b65b18bdcabbee25b953a21ba06875581'
UNASSESSED=('NaN/sNaN shape or point conversion to integer','product overflow or float-to-int overflow',
    'arbitrary user callbacks/subclasses','forged or concurrently mutated storage','unbounded rank/containers')


def cases():
    rows=[
        ('ordinary1d',[17],6),('doc2d',[20,40],8),('dense2d',[20,40],32),
        ('doc3d',[3,20,40],8),('axis-permutation',[40,3,20],8),
        ('thin-axes',[1,1,100],10),('singleton',[1],1),('equal-axes',[3,3,8],7),
        ('rank8',[1,2,1,3,1,4,1,5],12),('half-even-down',[25],10),('half-even-up',[35],10),
        ('half-even-2d',[25,25],100),('one-point',[13,17],1),('full-grid',[5,7],35),
        ('excess-points',[5,7],100),('fractional-points',[5,7],6.5),
        ('small-fraction-error',[5],.5),('zero-count',[5,7],0),
        ('negative-count',[5],-2),('negative-zero-step',[5],-20),
        ('empty-shape',[],1),('empty-shape-zero',[],0),('zero-axis',[0,7],0),
        ('negative-dimension',[-1,7],6),('float-dimensions',[5.,7.],6),
        ('bool-dimensions',[True,True],1),('bool-count',[5,7],True),
        ('large-safe-early',[2**53],2**53),('huge-object-early',[2**80],2**100),
        ('positive-inf-early',[5,7],{'float':'inf'}),('negative-inf-zero-step',[5],{'float':'-inf'}),
        ('none-count',[5,7],None),
    ]
    result=[dict(id=name,api=API,shape=shape,points=points,carrier='tuple',policy={})for name,shape,points in rows]
    for name,carrier,dtype in [('list','list',None),('array-i16','array','int16'),
        ('array-u64','array','uint64'),('array-f32','array','float32'),
        ('array-reversed-readonly','reversed-readonly','int64'),('array-swapped','array','>i2'),
        ('array-metadata','metadata','int64')]:
        result.append(dict(id=name,api=API,shape=[5,7],points=6,carrier=carrier,dtype=dtype,policy={}))
    result += [dict(id='scalar-shape',api=API,shape=5,points=6,carrier='scalar',policy={}),
        dict(id='rank2-shape-prefix',api=API,shape=[[3,4],[5,6]],points=8,carrier='rank2',policy={}),
        dict(id='numpy-count',api=API,shape=[5,7],points={'scalar':'int64','value':6},carrier='tuple',policy={}),
        dict(id='numpy-zero-warn',api=API,shape=[5,7],points={'scalar':'float64','value':0},carrier='tuple',policy={'divide':'warn'}),
        dict(id='numpy-zero-raise',api=API,shape=[5,7],points={'scalar':'float64','value':0},carrier='tuple',policy={'divide':'raise'}),
        dict(id='numpy-zero-call',api=API,shape=[5,7],points={'scalar':'float64','value':0},carrier='tuple',policy={'divide':'call'})]
    return result


def fixture(spec,np):
    shape=spec['shape'];carrier=spec['carrier']
    if carrier=='tuple':shape=tuple(shape)
    elif carrier=='list':shape=list(shape)
    elif carrier=='scalar':pass
    else:
        dtype=np.dtype(spec.get('dtype','int64'))
        if carrier=='metadata':dtype=np.dtype(dtype,metadata={'contract':'grid-shape'})
        shape=np.array(shape,dtype=dtype)
        if carrier=='reversed-readonly':shape=shape[::-1];shape.flags.writeable=False
    points=spec['points']
    if type(points)is dict:
        if 'float'in points:points=float(points['float'])
        else:points=getattr(np,points['scalar'])(points['value'])
    return (shape,points),{},[]
