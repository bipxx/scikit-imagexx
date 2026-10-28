"""Prospective source-derived rolling-ball geometry cases; no original calls."""
import math
from copy import deepcopy

APIS=('restoration.ball_kernel','restoration.ellipsoid_kernel')
MAX_ELEMENTS=8192
STAGES=('fixture','encode','observe')
UNASSESSED=('huge allocations','forged storage','concurrent mutation','longdouble or padded/object output representations')


def cases():
    ball=(
        ('ball-1d',dict(radius=2,ndim=1)),
        ('ball-2d',dict(radius=2,ndim=2)),
        ('ball-3d',dict(radius=1,ndim=3)),
        ('ball-4d',dict(radius=1,ndim=4)),
        ('ball-5d',dict(radius=1,ndim=5)),
        ('ball-8d',dict(radius=1,ndim=8)),
        ('ball-fractional',dict(radius=1.5,ndim=2)),
        ('ball-half',dict(radius=.5,ndim=3)),
        ('ball-zero',dict(radius=0,ndim=2)),
        ('ball-negative-zero',dict(radius=-0.,ndim=1)),
        ('ball-negative-quarter',dict(radius=-.25,ndim=2)),
        ('ball-negative-one',dict(radius=-1,ndim=2)),
        ('ball-radius-f32',dict(radius=1.5,ndim=2,radius_type='float32')),
        ('ball-radius-f64',dict(radius=1.5,ndim=2,radius_type='float64')),
        ('ball-integer-scalars',dict(radius=2,ndim=2,radius_type='int64',ndim_type='int64')),
        ('ball-rank-zero',dict(radius=1,ndim=0)),
        ('ball-rank-negative',dict(radius=1,ndim=-1)),
        ('ball-rank-float',dict(radius=1,ndim=2.)),
        ('ball-rank31-zero',dict(radius=0,ndim=31)),
        ('ball-rank32-zero',dict(radius=0,ndim=32)),
        ('ball-rank63-zero',dict(radius=0,ndim=63)),
        ('ball-rank64-zero',dict(radius=0,ndim=64)),
        ('ball-radius-nan',dict(radius='nan',ndim=2)),
        ('ball-radius-inf',dict(radius='inf',ndim=2)),
        ('ball-error-before-index',dict(radius='bad-radius',ndim=2,ndim_protocol=True)),
        ('ball-protocol-order',dict(radius=1.,ndim=2,radius_protocol=True,ndim_protocol=True)),
    )
    ellipsoid=(
        ('ellipsoid-1d',dict(shape=[3],intensity=2)),
        ('ellipsoid-even',dict(shape=[4,6],intensity=3)),
        ('ellipsoid-odd',dict(shape=[3,5],intensity=2)),
        ('ellipsoid-3d',dict(shape=[3,5,7],intensity=2)),
        ('ellipsoid-4d',dict(shape=[2]*4,intensity=1)),
        ('ellipsoid-5d',dict(shape=[1]*5,intensity=1)),
        ('ellipsoid-8d',dict(shape=[1]*8,intensity=1)),
        ('ellipsoid-zero-axes',dict(shape=[0,0],intensity=1)),
        ('ellipsoid-negative-axis',dict(shape=[-3,4],intensity=2)),
        ('ellipsoid-fractional-shape',dict(shape=[3.5,6.5],intensity=1)),
        ('ellipsoid-int16-shape',dict(shape=[3,5],intensity=1,shape_type='int16')),
        ('ellipsoid-f32-shape',dict(shape=[3,5],intensity=1,shape_type='float32')),
        ('ellipsoid-intensity-zero',dict(shape=[3,3],intensity=0)),
        ('ellipsoid-intensity-negative-zero',dict(shape=[3,3],intensity=-0.)),
        ('ellipsoid-intensity-negative',dict(shape=[3,5],intensity=-2.)),
        ('ellipsoid-intensity-f32',dict(shape=[3,5],intensity=.5,intensity_type='float32')),
        ('ellipsoid-error-before-intensity',dict(shape=[],intensity=2.,intensity_protocol=True)),
        ('ellipsoid-intensity-nan',dict(shape=[3,3],intensity='nan')),
        ('ellipsoid-intensity-inf',dict(shape=[3,3],intensity='inf')),
        ('ellipsoid-nonnumeric-shape',dict(shape=['bad'],intensity=1)),
        ('ellipsoid-scalar-shape',dict(shape=3,intensity=1)),
        ('ellipsoid-ragged-shape',dict(shape=[[3],[4,5]],intensity=1)),
        ('ellipsoid-array-protocol',dict(shape=[3,5],intensity=1,shape_protocol=True)),
        ('ellipsoid-intensity-protocol',dict(shape=[3,3],intensity=2.,intensity_protocol=True)),
    )
    return [dict(id=name,api=api,params=params,policy={},max_successful_elements=MAX_ELEMENTS)
            for api,rows in zip(APIS,(ball,ellipsoid))for name,params in rows]


class IndexValue:
    def __init__(self,value,events):self.value,self.events=value,events
    def __index__(self):self.events.append('ndim.__index__');return self.value


class RadiusValue:
    def __init__(self,value,events):self.value,self.events=value,events
    def ceil(self):self.events.append('radius.ceil');return float(math.ceil(self.value))
    def __pow__(self,exponent):self.events.append(['radius.__pow__',exponent]);return self.value**exponent
    def __lt__(self,other):self.events.append('radius.__lt__');return self.value<other


class ArrayInput:
    def __init__(self,value,events):self.value,self.events=value,events
    def __array__(self,dtype=None,copy=None):
        self.events.append(['shape.__array__',None if dtype is None else str(dtype),copy])
        return self.value if dtype is None else self.value.astype(dtype)


class IntensityValue:
    def __init__(self,value,events):self.value,self.events=value,events
    def __mul__(self,other):self.events.append('intensity.__mul__');return self.value*other


def fixture(spec,np):
    p=spec['params'];events=[];buffers={}
    if spec['api']==APIS[0]:
        radius=p['radius'];ndim=p['ndim']
        if isinstance(radius,str)and radius in ('nan','inf'):radius=float(radius)
        if 'radius_type'in p:radius=getattr(np,p['radius_type'])(radius)
        if 'ndim_type'in p:ndim=getattr(np,p['ndim_type'])(ndim)
        if p.get('radius_protocol'):radius=RadiusValue(radius,events)
        if p.get('ndim_protocol'):ndim=IndexValue(ndim,events)
        args=(radius,ndim)
    else:
        shape=deepcopy(p['shape']);intensity=p['intensity']
        if isinstance(intensity,str):intensity=float(intensity)
        if 'shape_type'in p:shape=np.array(shape,dtype=p['shape_type']);buffers['shape']=shape
        if 'intensity_type'in p:intensity=getattr(np,p['intensity_type'])(intensity)
        if p.get('shape_protocol'):
            shape=np.array(shape,dtype=np.int64);buffers['shape_owner']=shape;shape=ArrayInput(shape,events)
        if p.get('intensity_protocol'):intensity=IntensityValue(intensity,events)
        args=(shape,intensity)
    return args,{},events,buffers


def output_bound(spec):
    """Source-level allocation safety estimate, not an outcome expectation."""
    p=spec['params']
    if spec['api']==APIS[0]:
        if not isinstance(p['radius'],(int,float))or type(p['ndim'])is not int or p['ndim']<0:return 0
        return max(0,2*math.ceil(p['radius'])+1)**p['ndim']
    shape=p['shape']
    if not isinstance(shape,list)or any(not isinstance(n,(int,float))for n in shape):return 0
    return math.prod(2*max(n//2,1)+1 for n in shape)
