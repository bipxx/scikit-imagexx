"""Prospective bounded label_points inputs; freeze before any original calls."""
API='util.label_points'
UNASSESSED=('allocation failure or huge output shapes','forged or unaligned buffers',
            'concurrent mutation','platform-dependent longdouble padding',
            'integer label overflow requiring more than 2**64 points')


def cases():
    names=('ordinary2d','ordinary1d','ordinary3d','ordinary4d','float16','float32','float64',
           'int8','int16','int32','int64','uint8','uint16','uint32','uint64','bool',
           'big-endian','fortran','reverse','step','readonly','ties-even','duplicate-exact',
           'duplicate-rounded','negative','negative-fraction','negative-zero','out-of-bounds',
           'rounded-out-of-bounds','nan','infinity','invalid-raise','empty-points',
           'zero-axis-empty','zero-axis-point','rank0-onepoint','rank0-twopoints',
           'coords-vector','coords-scalar','coords-list','dimension-mismatch-negative',
           'shape-list','shape-numpy-integers','shape-float','shape-negative',
           'object-coordinates','complex-coordinates','protocol-order')
    return [dict(id=n,api=API,kind=n,policy={'invalid':'raise'} if n=='invalid-raise' else {},
                 prospective='original-first; native admission remains pending') for n in names]


def fixture(spec,np):
    kind=spec['kind'];events=[];buffers={}
    coords=np.array([[0,1],[2,3],[1,2]],np.float64);shape=(4,5)
    if kind=='ordinary1d':coords=np.array([[0],[3],[1]],np.int64);shape=(5,)
    elif kind=='ordinary3d':coords=np.array([[0,1,2],[2,1,0]],np.float64);shape=(3,4,5)
    elif kind=='ordinary4d':coords=np.array([[0,1,0,2],[1,0,1,1]],np.int32);shape=(2,2,2,3)
    elif kind in ('float16','float32','float64','int8','int16','int32','int64','uint8','uint16','uint32','uint64','bool'):
        coords=coords.astype(kind)
    elif kind=='big-endian':coords=coords.astype('>f8')
    elif kind=='fortran':coords=np.asfortranarray(coords)
    elif kind=='reverse':coords=coords[::-1]
    elif kind=='step':
        storage=np.repeat(coords,2,axis=0);coords=storage[::2];buffers['storage']=storage
    elif kind=='readonly':coords.flags.writeable=False
    elif kind=='ties-even':coords=np.array([[.5,1.5],[1.5,2.5],[2.5,3.5],[3.49,4.49]])
    elif kind=='duplicate-exact':coords=np.array([[1,2],[0,0],[1,2],[1,2]],np.int64)
    elif kind=='duplicate-rounded':coords=np.array([[1.1,2.1],[1.4,2.4],[.6,1.6]])
    elif kind=='negative':coords[1,0]=-1
    elif kind=='negative-fraction':coords[1,0]=-.1
    elif kind=='negative-zero':coords[0,0]=-0.
    elif kind=='out-of-bounds':coords[1,0]=4
    elif kind=='rounded-out-of-bounds':coords[1,1]=4.6
    elif kind in ('nan','infinity','invalid-raise'):coords[1,0]=np.nan if kind!='infinity' else np.inf
    elif kind=='empty-points':coords=np.empty((0,2),np.float64)
    elif kind=='zero-axis-empty':coords=np.empty((0,2),np.int64);shape=(0,5)
    elif kind=='zero-axis-point':coords=np.array([[0,1]],np.int64);shape=(0,5)
    elif kind in ('rank0-onepoint','rank0-twopoints'):
        coords=np.empty((1 if kind=='rank0-onepoint' else 2,0),np.float64);shape=()
    elif kind=='coords-vector':coords=np.array([0,1],np.int64)
    elif kind=='coords-scalar':coords=np.array(0,np.int64)
    elif kind=='coords-list':coords=[[0,1],[2,3]]
    elif kind=='dimension-mismatch-negative':coords=np.array([[-1,0,0]],np.int64)
    elif kind=='shape-list':shape=[4,5]
    elif kind=='shape-numpy-integers':shape=(np.int64(4),np.int32(5))
    elif kind=='shape-float':shape=(4.,5)
    elif kind=='shape-negative':shape=(-1,5)
    elif kind=='object-coordinates':coords=coords.astype(object)
    elif kind=='complex-coordinates':coords=coords.astype(np.complex128)+.25j
    elif kind=='protocol-order':
        class TraceArray(np.ndarray):
            def __array_finalize__(self,parent):self.events=getattr(parent,'events',events)
            def __lt__(self,other):
                self.events.append(dict(event='coords.__lt__',other=other))
                return np.ndarray.__lt__(self,other)
            def round(self,*args,**kwargs):
                self.events.append(dict(event='coords.round'))
                return np.ndarray.round(self,*args,**kwargs)
            def astype(self,dtype,*args,**kwargs):
                self.events.append(dict(event='coords.astype',dtype=np.dtype(dtype).str,copy=kwargs.get('copy',True)))
                return np.ndarray.astype(self,dtype,*args,**kwargs)
            def transpose(self,*args):
                self.events.append(dict(event='coords.transpose'))
                return np.ndarray.transpose(self,*args)
        coords=coords.view(TraceArray)
    return (coords,shape),{},events,buffers
