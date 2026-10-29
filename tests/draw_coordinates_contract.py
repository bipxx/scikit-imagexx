"""Prospective coordinate generators; factories never call draw algorithms."""
APIS=('draw.line','draw.rectangle','draw.disk','draw.ellipse')
MAX_OUTPUT_ELEMENTS=1_000_000
UNASSESSED=('forged storage/concurrent mutation','allocation failure or more than one million coordinate elements',
            'line signed-size arithmetic overflow after argument conversion',
            'nonfinite disk/ellipse geometry whose integer bounding box can overflow or allocate unbounded grids',
            'longdouble, complex, object-array geometry and higher-order arbitrary protocols')


def cases():
    names={
        'line':('horizontal','vertical','diagonal','rising-shallow','falling-shallow','rising-steep','falling-steep',
                'reverse-shallow','reverse-steep','tie-half','tie-reverse','point','negative','large-origin',
                'np-int32','np-int64','np-uint64','bool','float-integral','float-fractional','index-provider','overflow-start'),
        'rectangle':('end2d','extent2d','reverse-end','negative-extent','mixed-extent','zero-extent','clipped',
                     'clipped-away','negative-unclipped','end3d','extent1d','extent4d','rank0','float-half-even',
                     'float-negative-half','np-int32','fortran-view','negative-stride','readonly','missing','both',
                     'mismatch','end-provider','shape-short'),
        'disk':('ordinary','radius-one','radius-half','radius-zero','negative-radius','fractional-center',
                'fractional-radius','negative-center','clipped','clipped-away','shape-extra','shape-zero',
                'np-float32','np-int32','center-array','center-length1','center-length3','radius-provider','fp-raise-zero'),
        'ellipse':('axes','rotated','rotation-negative','rotation-pi','rotation-halfpi','rotation-overpi',
                   'fractional','tiny','zero-row-radius','zero-both','negative-radius','negative-center','clipped',
                   'clipped-away','shape-extra','np-float32','np-int64','shape-list','shape-short','center-provider',
                   'rotation-provider','fp-call-zero','shape-empty')}
    return [dict(id=name+'/'+kind,api='draw.'+name,kind=kind,policy={'divide':'raise','invalid':'raise'}if kind=='fp-raise-zero'else {'divide':'call','invalid':'call'}if kind=='fp-call-zero'else {},max_output_elements=MAX_OUTPUT_ELEMENTS)
            for name in names for kind in names[name]]


class ScalarProvider:
    def __init__(self,value,events,label):self.value,self.events,self.label=value,events,label
    def __index__(self):self.events.append(self.label+'.__index__');return int(self.value)
    def __int__(self):self.events.append(self.label+'.__int__');return int(self.value)
    def __float__(self):self.events.append(self.label+'.__float__');return float(self.value)


class ArrayProvider:
    def __init__(self,array,events):self.array,self.events=array,events
    def __array__(self,dtype=None,copy=None):
        self.events.append(dict(event='end.__array__',dtype=None if dtype is None else str(dtype),copy=copy))
        return self.array if dtype is None else self.array.astype(dtype,copy=False)


def fixture(spec,np):
    api=spec['api'].split('.')[1];kind=spec['kind'];events=[];buffers={};options={}
    if api=='line':
        table={'horizontal':(2,1,2,8),'vertical':(1,2,8,2),'diagonal':(0,0,7,7),
               'rising-shallow':(0,0,3,8),'falling-shallow':(3,0,0,8),'rising-steep':(0,0,8,3),
               'falling-steep':(8,0,0,3),'reverse-shallow':(3,8,0,0),'reverse-steep':(8,3,0,0),
               'tie-half':(0,0,2,4),'tie-reverse':(2,4,0,0),'point':(3,4,3,4),
               'negative':(-4,-2,3,1),'large-origin':(2**30,-2**30,2**30+7,-2**30+3)}
        args=table.get(kind,(1,2,5,7))
        if kind in('np-int32','np-int64','np-uint64'):args=tuple(np.dtype(kind[3:]).type(v)for v in args)
        elif kind=='bool':args=(False,True,True,False)
        elif kind=='float-integral':args=(1.,2.,5.,7.)
        elif kind=='float-fractional':args=(1.5,2.5,5.5,7.5)
        elif kind=='index-provider':args=(ScalarProvider(1,events,'r0'),2,5,7)
        elif kind=='overflow-start':args=(2**100,2,5,7)
    elif api=='rectangle':
        start=(1,2);options={'end':(3,5)}
        if kind=='extent2d':options={'extent':(3,4)}
        elif kind=='reverse-end':start=(3,5);options={'end':(1,2)}
        elif kind=='negative-extent':start=(4,5);options={'extent':(-3,-2)}
        elif kind=='mixed-extent':start=(4,1);options={'extent':(-3,4)}
        elif kind=='zero-extent':options={'extent':(0,4)}
        elif kind=='clipped':start=(-2,-1);options={'end':(4,7),'shape':(3,5)}
        elif kind=='clipped-away':start=(8,9);options={'extent':(3,4),'shape':(3,5)}
        elif kind=='negative-unclipped':start=(-3,-2);options={'end':(0,1)}
        elif kind=='end3d':start=(0,1,2);options={'end':(1,3,4)}
        elif kind=='extent1d':start=(2,);options={'extent':(4,)}
        elif kind=='extent4d':start=(0,1,0,1);options={'extent':(2,3,2,2)}
        elif kind=='rank0':start=();options={'extent':()}
        elif kind=='float-half-even':start=(.5,1.5);options={'end':(2.5,3.5)}
        elif kind=='float-negative-half':start=(-2.5,-1.5);options={'extent':(3.,4.)}
        elif kind=='np-int32':start=np.array(start,np.int32);options={'end':np.array([3,5],np.int32)}
        elif kind=='fortran-view':
            owner=np.asfortranarray([[1,2],[3,5]]);buffers['owner']=owner;start=owner[0];options={'end':owner[1]}
        elif kind=='negative-stride':start=np.array([2,1])[::-1];options={'end':np.array([5,3])[::-1]}
        elif kind=='readonly':start=np.array(start);start.flags.writeable=False
        elif kind=='missing':options={}
        elif kind=='both':options={'end':(3,5),'extent':(3,4)}
        elif kind=='mismatch':options={'end':(2,3,4)}
        elif kind=='end-provider':
            owner=np.array([3,5]);buffers['end-owner']=owner;options={'end':ArrayProvider(owner,events)}
        elif kind=='shape-short':options={'end':(3,5),'shape':(4,)}
        args=(start,)
    elif api=='disk':
        center=(4,5);radius=3.
        if kind=='radius-one':radius=1.
        elif kind=='radius-half':radius=.5
        elif kind in('radius-zero','fp-raise-zero'):radius=0.
        elif kind=='negative-radius':radius=-2.
        elif kind=='fractional-center':center=(4.25,5.75)
        elif kind=='fractional-radius':radius=2.5
        elif kind=='negative-center':center=(-1,-2)
        elif kind=='clipped':center=(0,1);options={'shape':(4,5)}
        elif kind=='clipped-away':center=(20,30);options={'shape':(4,5)}
        elif kind=='shape-extra':options={'shape':(6,7,2)}
        elif kind=='shape-zero':options={'shape':(0,7)}
        elif kind=='np-float32':center=tuple(np.float32(v)for v in center);radius=np.float32(2.5)
        elif kind=='np-int32':center=tuple(np.int32(v)for v in center);radius=np.int32(3)
        elif kind=='center-array':center=np.array(center,dtype=np.float64)
        elif kind=='center-length1':center=(4,)
        elif kind=='center-length3':center=(4,5,6)
        elif kind=='radius-provider':radius=ScalarProvider(3,events,'radius')
        args=(center,radius)
    else:
        r,c,rr,cc=4.,5.,3.,2.;rotation=0.
        if kind=='rotated':rotation=.37
        elif kind=='rotation-negative':rotation=-.37
        elif kind=='rotation-pi':rotation=3.141592653589793
        elif kind=='rotation-halfpi':rotation=1.5707963267948966
        elif kind=='rotation-overpi':rotation=7.
        elif kind=='fractional':r,c,rr,cc=4.25,5.75,3.5,1.25
        elif kind=='tiny':rr=cc=.25
        elif kind in('zero-row-radius','fp-call-zero'):rr=0.
        elif kind=='zero-both':rr=cc=0.
        elif kind=='negative-radius':rr=-2.
        elif kind=='negative-center':r,c=-1.,-2.
        elif kind=='clipped':r,c=0.,1.;options={'shape':(4,5)}
        elif kind=='clipped-away':r,c=20.,30.;options={'shape':(4,5)}
        elif kind=='shape-extra':options={'shape':(6,7,2)}
        elif kind=='np-float32':r,c,rr,cc=map(np.float32,(r,c,rr,cc));rotation=np.float32(.37)
        elif kind=='np-int64':r,c,rr,cc=map(np.int64,(r,c,rr,cc))
        elif kind=='shape-list':options={'shape':[6,7]}
        elif kind=='shape-short':options={'shape':(6,)}
        elif kind=='shape-empty':options={'shape':()}
        elif kind=='center-provider':r=ScalarProvider(4,events,'r')
        elif kind=='rotation-provider':rotation=ScalarProvider(.37,events,'rotation')
        args=(r,c,rr,cc);options['rotation']=rotation
    def retain(value,name):
        if isinstance(value,np.ndarray):buffers[name]=value
        elif isinstance(value,(tuple,list)):
            for i,item in enumerate(value):retain(item,name+'.'+str(i))
        elif isinstance(value,dict):
            for key,item in value.items():retain(item,name+'.'+key)
    retain(args,'args');retain(options,'options')
    return args,options,events,buffers


def benchmark_cases():
    """Prospective whole-call, allocation-inclusive one-worker cohorts."""
    return [dict(id='line/tiny',api='draw.line',args=[0,0,7,3],kwargs={}),
            dict(id='line/long',api='draw.line',args=[-65536,0,65535,32767],kwargs={}),
            dict(id='rectangle/tiny',api='draw.rectangle',args=[[1,2]],kwargs={'extent':[3,4]}),
            dict(id='rectangle/2d',api='draw.rectangle',args=[[0,0]],kwargs={'extent':[256,384]}),
            dict(id='rectangle/3d',api='draw.rectangle',args=[[0,0,0]],kwargs={'extent':[32,32,32]}),
            dict(id='disk/tiny',api='draw.disk',args=[[4,5],3],kwargs={}),
            dict(id='disk/dense',api='draw.disk',args=[[320,320],320],kwargs={'shape':[640,640]}),
            dict(id='disk/clipped',api='draw.disk',args=[[0,0],256],kwargs={'shape':[512,512]}),
            dict(id='ellipse/tiny',api='draw.ellipse',args=[4,5,3,2],kwargs={}),
            dict(id='ellipse/axes',api='draw.ellipse',args=[256,512,128,256],kwargs={'shape':[512,1024]}),
            dict(id='ellipse/rotated',api='draw.ellipse',args=[256,256,128,192],kwargs={'shape':[512,512],'rotation':.37}),
            dict(id='ellipse/clipped',api='draw.ellipse',args=[0,0,128,192],kwargs={'shape':[256,256],'rotation':-.37})]
