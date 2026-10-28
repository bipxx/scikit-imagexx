"""Prospective B-spline subdivision cases; no original function calls."""
API='measure.subdivide_polygon'
MAX_POINTS=129
MAX_OUTPUT_POINTS=512
UNASSESSED=('huge inputs or allocation failure','forged storage or concurrent mutation',
            'longdouble or padded structured coordinate representations')


def cases():
    specs=[dict(id=f'{closure}-degree{degree}',api=API,kind=closure,degree=degree,policy={})for closure in('open','closed')for degree in range(1,8)]
    specs += [dict(id=f'preserved-degree{degree}',api=API,kind='preserved',degree=degree,policy={})for degree in(3,7)]
    names=('closed-preserve','two-point-open-degree1','two-point-open-degree7','two-point-closed',
           'singleton','empty','all-identical','collinear','near-closure',
           'int32','uint8','float16','float32','bool','big-endian-float',
           'complex64','complex-closed','complex-preserved','object-numbers','int64-large',
           'fortran','reverse-rows','reverse-columns','step-rows','step-columns','readonly',
           'dtype-metadata','nan-coordinate','inf-coordinate','nan-closure',
           'degree-zero','degree-eight','degree-list','degree-none','degree-float','degree-bool','degree-numpy',
           'coords-list','coords-vector','coords-three-columns','preserve-array','preserve-provider',
           'invalid-degree-before-coords','moderate-129')
    specs += [dict(id=name,api=API,kind=name,degree=2,policy={})for name in names]
    return [dict(**spec,max_points=MAX_POINTS,max_output_points=MAX_OUTPUT_POINTS,prospective='Original output and native admission pending')for spec in specs]


class PreserveEndsProvider:
    def __init__(self,array,events):self.array,self.events=array,events
    def __bool__(self):
        self.events.append('preserve_ends.__bool__')
        self.array[0]=[9.,9.];self.events.append('preserve_ends.mutated-first-coordinate')
        return True


def fixture(spec,np):
    kind=spec['kind'];events=[];buffers={};degree=spec['degree'];preserve=False
    coords=np.array([[0.,0.],[1.1,.3],[2.2,1.7],[3.3,3.9],[2.4,5.1],[.5,6.2],[-1.6,4.3],[.7,2.4],[2.8,.6]])
    if kind in('closed','closed-preserve','near-closure','nan-closure','complex-closed'):
        coords=np.vstack((coords,coords[0]))
    if kind in('preserved','closed-preserve','complex-preserved'):preserve=True
    if kind in('two-point-open-degree1','two-point-open-degree7'):
        coords=np.array([[0.,1.],[2.,3.]]);degree=1 if kind.endswith('degree1')else 7
    elif kind=='two-point-closed':coords=np.array([[2.,3.],[2.,3.]])
    elif kind=='singleton':coords=np.array([[2.,3.]])
    elif kind=='empty':coords=np.empty((0,2),np.float64)
    elif kind=='all-identical':coords=np.full((6,2),2.)
    elif kind=='collinear':coords=np.column_stack((np.arange(9.)*.1,np.arange(9.)*.3))
    elif kind=='near-closure':coords[-1,0]=np.nextafter(0.,1.)
    elif kind in('int32','uint8','float16','float32','bool'):coords=coords.astype(kind)
    elif kind=='big-endian-float':coords=coords.astype('>f8')
    elif kind in('complex64','complex-closed','complex-preserved'):
        coords=coords.astype(np.complex64);coords+=np.array([1j,2j],np.complex64)
    elif kind=='object-numbers':coords=coords.astype(object)
    elif kind=='int64-large':coords=coords.astype(np.int64)+(2**53+1)
    elif kind=='fortran':coords=np.asfortranarray(coords)
    elif kind=='reverse-rows':coords=coords[::-1]
    elif kind=='reverse-columns':coords=coords[:,::-1]
    elif kind=='step-rows':
        owner=np.repeat(coords,2,axis=0);coords=owner[::2];buffers['owner']=owner
    elif kind=='step-columns':
        owner=np.repeat(coords,2,axis=1);coords=owner[:,::2];buffers['owner']=owner
    elif kind=='readonly':coords.flags.writeable=False
    elif kind=='dtype-metadata':coords=coords.astype(np.dtype('float64',metadata={'contract':'B-spline-coordinate'}))
    elif kind=='nan-coordinate':coords[3,0]=np.nan
    elif kind=='inf-coordinate':coords[3,0]=np.inf
    elif kind=='nan-closure':coords[0,0]=coords[-1,0]=np.nan
    elif kind=='degree-zero':degree=0
    elif kind=='degree-eight':degree=8
    elif kind=='degree-list':degree=[2]
    elif kind=='degree-none':degree=None
    elif kind=='degree-float':degree=2.
    elif kind=='degree-bool':degree=True
    elif kind=='degree-numpy':degree=np.int64(2)
    elif kind=='coords-list':coords=coords.tolist()
    elif kind=='coords-vector':coords=np.arange(9.)
    elif kind=='coords-three-columns':coords=np.column_stack((coords,np.arange(coords.shape[0])))
    elif kind=='preserve-array':preserve=np.array([True,False])
    elif kind=='preserve-provider':preserve=PreserveEndsProvider(coords,events)
    elif kind=='invalid-degree-before-coords':coords=None;degree=8
    elif kind=='moderate-129':
        axis=np.arange(129.);coords=np.column_stack((axis,((axis*17)%13)-6));degree=7
    if isinstance(coords,np.ndarray):buffers['coords']=coords
    if isinstance(degree,np.ndarray):buffers['degree']=degree
    if isinstance(preserve,np.ndarray):buffers['preserve_ends']=preserve
    return (coords,),dict(degree=degree,preserve_ends=preserve),events,buffers
