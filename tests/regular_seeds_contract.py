"""Prospective regular_seeds cases; regular_grid remains an explicit provider."""
API='util.regular_seeds'
UNASSESSED=('huge allocation failures','forged storage','concurrent mutation','longdouble or padded structured output bytes')


def cases():
    names=('ordinary2d','ordinary1d','ordinary3d','ordinary4d','thin-axis','reordered-axes',
           'sparse-large','one-seed','full-grid','excess-points','approximate-count','shape-list','shape-array',
           'shape-index-protocol','empty-axis','scalar-shape','negative-dimension','float-dimension',
           'zero-count','negative-count','negative-zero-step','numpy-count','fractional-count',
           'none-count','grid-error-before-dtype','dtype-error-before-zero-step','shape-float-bad-dtype',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64','bool','float16',
           'float32','float64','complex64','complex128','signed-wrap','unsigned-wrap',
           'half-overflow-warning','half-overflow-raise','big-endian','dtype-metadata',
           'object-dtype','bytes-dtype','unicode-dtype','structured-dtype','datetime-dtype','invalid-dtype')
    return [dict(id=n,api=API,kind=n,policy={'over':'raise'}if n=='half-overflow-raise'else{},prospective='Original outcomes and native admission pending')for n in names]


class IndexDimension:
    def __init__(self,value,events):self.value,self.events=value,events
    def __index__(self):self.events.append('dimension.__index__');return self.value
    def __float__(self):self.events.append('dimension.__float__');return float(self.value)
    def __lt__(self,other):self.events.append('dimension.__lt__');return self.value<other
    def __mul__(self,other):self.events.append('dimension.__mul__');return self.value*other
    def __rmul__(self,other):self.events.append('dimension.__rmul__');return other*self.value
    def __repr__(self):return 'IndexDimension('+str(self.value)+')'


def fixture(spec,np):
    kind=spec['kind'];events=[];shape=(5,7);points=6;options={}
    shapes={'ordinary1d':(17,),'ordinary3d':(3,7,9),'ordinary4d':(2,3,4,5),
            'thin-axis':(1,3,31),'reordered-axes':(31,1,3),'empty-axis':(0,7),
            'scalar-shape':(),'negative-dimension':(-1,7),'float-dimension':(5.,7)}
    if kind in shapes:shape=shapes[kind]
    elif kind=='one-seed':points=1
    elif kind=='full-grid':points=35
    elif kind=='excess-points':points=100
    elif kind=='approximate-count':shape=(13,17);points=11
    elif kind=='shape-list':shape=[5,7]
    elif kind=='shape-array':shape=np.array([5,7],np.int64)
    elif kind=='shape-index-protocol':shape=(IndexDimension(7,events),);points=3
    elif kind=='zero-count':points=0
    elif kind=='negative-count':shape=(5,);points=-2
    elif kind=='negative-zero-step':shape=(5,);points=-20
    elif kind=='numpy-count':points=np.int64(6)
    elif kind=='fractional-count':points=6.5
    elif kind=='none-count':points=None
    elif kind=='grid-error-before-dtype':points=0;options['dtype']='not-a-dtype'
    elif kind=='dtype-error-before-zero-step':shape=(5,);points=-20;options['dtype']='not-a-dtype'
    elif kind=='shape-float-bad-dtype':shape=(5.,7);options['dtype']='not-a-dtype'
    elif kind in ('int8','uint8','int16','uint16','int32','uint32','int64','uint64','bool','float16','float32','float64','complex64','complex128'):options['dtype']=np.dtype(kind)
    elif kind in ('signed-wrap','unsigned-wrap'):shape=(300,);points=300;options['dtype']=np.dtype('int8'if kind=='signed-wrap'else'uint8')
    elif kind in ('half-overflow-warning','half-overflow-raise'):shape=(65520,);points=65520;options['dtype']=np.dtype('float16')
    elif kind=='big-endian':options['dtype']=np.dtype('>u2')
    elif kind=='dtype-metadata':options['dtype']=np.dtype('int32',metadata={'contract':'seed-labels'})
    elif kind=='object-dtype':options['dtype']=np.dtype(object)
    elif kind=='bytes-dtype':options['dtype']=np.dtype('S4')
    elif kind=='unicode-dtype':options['dtype']=np.dtype('U4')
    elif kind=='structured-dtype':options['dtype']=np.dtype([('left','i2'),('right','u1')])
    elif kind=='datetime-dtype':options['dtype']=np.dtype('datetime64[D]')
    elif kind=='invalid-dtype':options['dtype']='not-a-dtype'
    if kind=='sparse-large':shape=(257,263);points=16
    return (shape,points),options,events,{}
