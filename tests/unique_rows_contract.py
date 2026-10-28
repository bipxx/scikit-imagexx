"""Prospective byte-row uniqueness cases, frozen before original calls."""
API='util.unique_rows'
UNASSESSED=('forged or unaligned storage','concurrent mutation','allocation failure or huge row widths',
            'longdouble and padded structured outputs with undefined padding bytes')


def cases():
    names=('ordinary','unsorted','all-unique','all-duplicates','single-row','single-column',
           'empty-rows','empty-columns','empty-both','int8','int16','int32','int64',
           'uint8','uint16','uint32','uint64','bool','float16','float32','float64',
           'complex64','complex128','bytes','unicode','void','structured-packed',
           'object-error','structured-object-error','datetime','timedelta','fortran',
           'transpose','reverse-rows','reverse-columns','step-rows','step-columns',
           'readonly','big-endian-int','big-endian-float','signed-zero','nan-payload32',
           'nan-payload64','complex-nan-payload','dtype-metadata','scalar-error',
           'vector-error','rank3-error','list-error','subclass','array-provider','ndim-provider-error')
    return [dict(id=n,api=API,kind=n,policy={},prospective='native admission pending; byte-sort order is observed')for n in names]


class ArrayProvider:
    def __init__(self,array,events,ndim=None):self.array,self.events,self.dimension=array,events,ndim
    @property
    def ndim(self):
        self.events.append(dict(event='provider.ndim'))
        return self.array.ndim if self.dimension is None else self.dimension
    def __array__(self,dtype=None,copy=None):
        self.events.append(dict(event='provider.__array__',dtype=None if dtype is None else str(dtype),copy=copy))
        return self.array


def fixture(spec,np):
    kind=spec['kind'];events=[];buffers={}
    array=np.array([[2,1],[0,3],[2,1],[1,0]],np.int64)
    if kind=='unsorted':array=np.array([[256,1],[1,0],[0,9],[256,1],[2,0]],np.int32)
    elif kind=='all-unique':array=np.array([[3,2],[1,0],[4,5]],np.int64)
    elif kind=='all-duplicates':array=np.repeat(np.array([[2,1]],np.int64),5,axis=0)
    elif kind=='single-row':array=np.array([[1,2,3]],np.int64)
    elif kind=='single-column':array=np.array([[256],[1],[0],[256],[2]],np.int32)
    elif kind in ('empty-rows','empty-columns','empty-both'):
        array=np.empty({'empty-rows':(0,3),'empty-columns':(3,0),'empty-both':(0,0)}[kind],np.float64)
    elif kind in ('int8','int16','int32','int64','uint8','uint16','uint32','uint64','bool','float16','float32','float64','complex64','complex128'):
        array=array.astype(kind)
        if kind.startswith('complex'):array[:,1]+=np.array([1j,2j,1j,-1j],dtype=kind)
    elif kind=='bytes':array=np.array([[b'b',b'a'],[b'a',b''],[b'b',b'a'],[b'aa',b'c']],dtype='S3')
    elif kind=='unicode':array=np.array([['b','a'],['a',''],['b','a'],['é','中']],dtype='U2')
    elif kind=='void':array=np.array([[b'ba',b'a'],[b'a',b''],[b'ba',b'a']],dtype='V3')
    elif kind=='structured-packed':
        array=np.zeros((4,2),dtype=[('x','u1'),('y','<i2')]);array['x']=[[2,1],[0,3],[2,1],[1,0]];array['y']=array['x']*7
    elif kind=='object-error':array=np.array([['b','a'],['a','b'],['b','a']],dtype=object)
    elif kind=='structured-object-error':
        array=np.zeros((3,2),dtype=[('x',object)]);array['x']=[['a','b'],['c','d'],['a','b']]
    elif kind in ('datetime','timedelta'):array=array.astype('datetime64[D]'if kind=='datetime'else'timedelta64[D]')
    elif kind=='fortran':array=np.asfortranarray(array)
    elif kind=='transpose':array=array.T
    elif kind=='reverse-rows':array=array[::-1]
    elif kind=='reverse-columns':array=array[:,::-1]
    elif kind=='step-rows':
        storage=np.repeat(array,2,axis=0);array=storage[::2];buffers['storage']=storage
    elif kind=='step-columns':
        storage=np.repeat(array,2,axis=1);array=storage[:,::2];buffers['storage']=storage
    elif kind=='readonly':array.flags.writeable=False
    elif kind=='big-endian-int':array=array.astype('>i4')
    elif kind=='big-endian-float':array=array.astype('>f8')
    elif kind=='signed-zero':array=np.array([[0.,0.],[-0.,0.],[0.,-0.],[-0.,-0.],[0.,0.]])
    elif kind in ('nan-payload32','nan-payload64'):
        if kind.endswith('32'):words=[0x7fc00001,0x7fc00002,0x7fc00001,0xffc00001,0];integer,real=np.uint32,np.float32
        else:words=[0x7ff8000000000001,0x7ff8000000000002,0x7ff8000000000001,0xfff8000000000001,0];integer,real=np.uint64,np.float64
        array=np.array(words,integer).view(real).reshape(-1,1)
    elif kind=='complex-nan-payload':
        array=np.array([[0x7ff8000000000001,0],[0x7ff8000000000002,0],[0x7ff8000000000001,0]],np.uint64).view(np.complex128)
    elif kind=='dtype-metadata':array=array.astype(np.dtype('int64',metadata={'contract':'row-bytes'}))
    elif kind=='scalar-error':array=np.array(1,np.int64)
    elif kind=='vector-error':array=np.array([1,2,1],np.int64)
    elif kind=='rank3-error':array=np.zeros((2,2,2),np.int64)
    elif kind=='list-error':array=[[2,1],[0,3],[2,1]]
    elif kind=='subclass':
        class TraceArray(np.ndarray):
            def __array_finalize__(self,parent):self.events=getattr(parent,'events',events)
            def view(self,*args,**kwargs):
                self.events.append(dict(event='subclass.view'))
                return np.ndarray.view(self,*args,**kwargs)
        array=array.view(TraceArray)
    elif kind=='array-provider':array=ArrayProvider(array,events)
    elif kind=='ndim-provider-error':array=ArrayProvider(array,events,1)
    return (array,),{},events,buffers
