"""Prospective bounded RDP cases, frozen before original calls."""
API='measure.approximate_polygon'
MAX_POINTS=257
UNASSESSED=('huge/adversarial chains or allocation failure','forged storage or concurrent mutation',
            'longdouble and padded/structured coordinate representations')


def cases():
    names=('open-zigzag','closed-square','closed-square-coarse','collinear','outside-segment',
           'repeated-endpoints','all-identical','duplicate-interior','empty-positive','singleton','two-points',
           'threshold-below','threshold-equal','threshold-above','first-max-tie',
           'float16','float32','int8','int16','int32','int64','uint8','uint64','complex64-error','object-numbers',
           'fortran','reverse-rows','reverse-columns','step-rows','step-columns','readonly',
           'big-endian-float','signed-zero','dtype-metadata','nan-coordinates','inf-coordinates','nan-tolerance',
           'zero-tolerance','negative-tolerance','negative-zero-tolerance','negative-inf-tolerance','list-zero','none-zero',
           'coords-provider-positive','coords-provider-zero','tolerance-provider-positive','tolerance-mutates-coordinates',
           'list-positive','vector-error','three-columns-error','scalar-error','none-tolerance',
           'array0d-tolerance','array2-tolerance','float32-tolerance','moderate-257','int8-overflow','invalid-raise',
           'subclass-positive','inf-tolerance')
    return [dict(id=name,api=API,kind=name,policy={'invalid':'raise'}if name=='invalid-raise'else{},max_points=MAX_POINTS,
                 prospective='Original observations and native admission pending')for name in names]


def key_event(value):
    if isinstance(value,tuple):return [key_event(item)for item in value]
    if isinstance(value,slice):return dict(slice=[value.start,value.stop,value.step])
    if hasattr(value,'tolist'):return value.tolist()
    return value


class CoordinatesProvider:
    def __init__(self,array,events):self.array,self.events=array,events
    @property
    def shape(self):self.events.append('coords.shape');return self.array.shape
    def __getitem__(self,key):
        self.events.append(dict(event='coords.__getitem__',key=key_event(key)))
        return self.array[key]


class ToleranceProvider:
    def __init__(self,value,events,array=None):self.value,self.events,self.array=value,events,array
    def __le__(self,other):
        self.events.append(dict(event='tolerance.__le__',other=other))
        if self.array is not None:
            self.array[1]=[3.,1.];self.events.append('tolerance.mutated-coordinate-row1')
        return self.value<=other
    def __lt__(self,other):
        self.events.append(dict(event='tolerance.__lt__',other=other))
        return self.value<other


def fixture(spec,np):
    kind=spec['kind'];events=[];buffers={};tolerance=.6
    coords=np.array([[0.,0.],[.25,1.],[2.,2.],[.5,3.],[0.,4.]],np.float64)
    if kind in('closed-square','closed-square-coarse'):
        coords=np.array([[0.,0.],[0.,1.],[0.,2.],[1.,2.],[2.,2.],[2.,1.],[2.,0.],[1.,0.],[0.,0.]])
        if kind.endswith('coarse'):tolerance=2.
    elif kind=='collinear':coords=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.],[4.,4.]]);tolerance=.1
    elif kind=='outside-segment':coords=np.array([[0.,0.],[0.,-2.],[0.,2.],[0.,1.]]);tolerance=.5
    elif kind=='repeated-endpoints':coords=np.array([[0.,0.],[1.,1.],[0.,0.],[1.,-1.],[0.,0.]])
    elif kind=='all-identical':coords=np.full((7,2),2.)
    elif kind=='duplicate-interior':coords=np.array([[0.,0.],[1.,1.],[1.,1.],[2.,2.],[2.,2.],[0.,4.]])
    elif kind=='empty-positive':coords=np.empty((0,2),np.float64)
    elif kind=='singleton':coords=np.array([[2.,3.]])
    elif kind=='two-points':coords=np.array([[2.,3.],[5.,1.]])
    elif kind.startswith('threshold-'):
        coords=np.array([[0.,0.],[1.,1.],[0.,2.]])
        tolerance={'threshold-below':np.nextafter(1.,0.),'threshold-equal':1.,'threshold-above':np.nextafter(1.,np.inf)}[kind]
    elif kind=='first-max-tie':coords=np.array([[0.,0.],[1.,1.],[1.,3.],[0.,4.]]);tolerance=.5
    elif kind in('float16','float32','int8','int16','int32','int64','uint8','uint64'):coords=coords.astype(kind)
    elif kind=='complex64-error':coords=coords.astype(np.complex64)
    elif kind=='object-numbers':coords=coords.astype(object)
    elif kind=='fortran':coords=np.asfortranarray(coords)
    elif kind=='reverse-rows':coords=coords[::-1]
    elif kind=='reverse-columns':coords=coords[:,::-1]
    elif kind=='step-rows':
        owner=np.repeat(coords,2,axis=0);coords=owner[::2];buffers['owner']=owner
    elif kind=='step-columns':
        owner=np.repeat(coords,2,axis=1);coords=owner[:,::2];buffers['owner']=owner
    elif kind=='readonly':coords.flags.writeable=False
    elif kind=='big-endian-float':coords=coords.astype('>f8')
    elif kind=='signed-zero':coords=np.array([[0.,-0.],[-0.,0.],[1.,-0.],[0.,0.]])
    elif kind=='dtype-metadata':coords=coords.astype(np.dtype('float64',metadata={'contract':'RDP-input-words'}))
    elif kind=='nan-coordinates':coords[1,0]=np.nan
    elif kind=='invalid-raise':coords[0,0]=np.inf
    elif kind=='inf-coordinates':coords[0,0]=np.inf
    elif kind=='nan-tolerance':tolerance=float('nan')
    elif kind in('zero-tolerance','negative-tolerance','negative-zero-tolerance','negative-inf-tolerance'):
        tolerance={'zero-tolerance':0.,'negative-tolerance':-1.,'negative-zero-tolerance':-0.,'negative-inf-tolerance':float('-inf')}[kind]
    elif kind in('list-zero','list-positive'):coords=coords.tolist();tolerance=0 if kind=='list-zero'else .6
    elif kind=='none-zero':coords=None;tolerance=0
    elif kind in('coords-provider-positive','coords-provider-zero'):
        buffers['coords']=coords;coords=CoordinatesProvider(coords,events);tolerance=0 if kind.endswith('zero')else .6
    elif kind in('tolerance-provider-positive','tolerance-mutates-coordinates'):
        tolerance=ToleranceProvider(.6,events,coords if kind=='tolerance-mutates-coordinates'else None)
    elif kind=='vector-error':coords=np.array([1.,2.,3.])
    elif kind=='three-columns-error':coords=np.zeros((3,3),np.float64)
    elif kind=='scalar-error':coords=np.array(3.)
    elif kind=='none-tolerance':tolerance=None
    elif kind=='array0d-tolerance':tolerance=np.array(.6)
    elif kind=='array2-tolerance':tolerance=np.array([.5,.6])
    elif kind=='float32-tolerance':tolerance=np.float32(.6)
    elif kind=='moderate-257':
        axis=np.arange(257,dtype=np.int64);coords=np.column_stack((axis,((axis*17)%13)-6)).astype(np.float64);tolerance=1.25
    elif kind=='int8-overflow':coords=np.array([[-120,-120],[120,120],[-110,100],[110,-100]],np.int8)
    elif kind=='subclass-positive':
        class TraceCoordinates(np.ndarray):
            def __array_finalize__(self,parent):self.events=getattr(parent,'events',events)
            def __getitem__(self,key):
                self.events.append(dict(event='subclass.__getitem__',key=key_event(key)))
                return np.ndarray.__getitem__(self,key)
        coords=coords.view(TraceCoordinates)
    elif kind=='inf-tolerance':tolerance=float('inf')
    if isinstance(coords,np.ndarray):buffers['coords']=coords
    if isinstance(tolerance,np.ndarray):buffers['tolerance']=tolerance
    return (coords,tolerance),{},events,buffers
