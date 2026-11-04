"""Prospective circle-perimeter inputs; factories never call draw algorithms."""
from draw_coordinates_contract import ScalarProvider

API='draw.circle_perimeter'
MAX_OUTPUT_ELEMENTS=100_000
UNASSESSED=(
    'unsafe radius/decision arithmetic or unbounded output allocation',
    'arbitrary shape/method comparison protocols and forged/concurrently mutated storage',
    'allocation failure and platform-specific C-long/ssize conversion domains',
)


def cases():
    specs=[]
    for method in ('bresenham','andres'):
        for radius in (-3,-1,0,1,2,3,4,5,7,16,31,128):
            specs.append(dict(id=method+'/radius-'+str(radius),kind='radius',method=method,radius=radius))
        for kind in ('negative-center','large-center','clip','clip-away','shape-zero','shape-negative',
                     'shape-extra','shape-list','shape-array','shape-readonly','shape-negative-stride',
                     'shape-short','shape-empty','shape-float','np-int32','np-int64','np-uint64',
                     'bool','float-integral','float-fractional','index-provider','overflow-origin'):
            specs.append(dict(id=method+'/'+kind,kind=kind,method=method,radius=4))
    for kind in ('invalid-method','bytes-method','none-method','list-method','np-string-method'):
        specs.append(dict(id=kind,kind=kind,method='bresenham',radius=4))
    return [dict(api=API,max_output_elements=MAX_OUTPUT_ELEMENTS,**spec)for spec in specs]


def fixture(spec,np):
    kind=spec['kind'];events=[];buffers={};args=(4,5,spec['radius'])
    options={'method':spec['method']}
    if kind=='negative-center':args=(-3,-2,4)
    elif kind=='large-center':args=(2**40,-2**40,4)
    elif kind=='clip':args=(0,0,4);options['shape']=(5,6)
    elif kind=='clip-away':args=(-20,-20,4);options['shape']=(5,6)
    elif kind=='shape-zero':options['shape']=(0,6)
    elif kind=='shape-negative':options['shape']=(-1,6)
    elif kind=='shape-extra':options['shape']=(7,8,9)
    elif kind=='shape-list':options['shape']=[7,8]
    elif kind in ('shape-array','shape-readonly','shape-negative-stride'):
        shape=np.array([7,8],dtype=np.intp)
        if kind=='shape-readonly':shape.flags.writeable=False
        elif kind=='shape-negative-stride':shape=np.array([8,7],dtype=np.intp)[::-1]
        options['shape']=shape;buffers['shape']=shape
    elif kind=='shape-short':options['shape']=(7,)
    elif kind=='shape-empty':options['shape']=()
    elif kind=='shape-float':options['shape']=(7.5,8.5)
    elif kind in ('np-int32','np-int64','np-uint64'):
        args=tuple(np.dtype(kind[3:]).type(v)for v in args)
    elif kind=='bool':args=(False,True,True)
    elif kind=='float-integral':args=(4.,5.,4.)
    elif kind=='float-fractional':args=(4.75,5.5,4.25)
    elif kind=='index-provider':args=(ScalarProvider(4,events,'r'),5,4)
    elif kind=='overflow-origin':args=(2**100,5,4)
    elif kind=='invalid-method':options['method']='unknown'
    elif kind=='bytes-method':options['method']=b'bresenham'
    elif kind=='none-method':options['method']=None
    elif kind=='list-method':options['method']=['bresenham']
    elif kind=='np-string-method':options['method']=np.str_('andres')
    return args,options,events,buffers


def benchmark_cases():
    return [dict(id=method+'/'+name,api=API,r=r,c=c,radius=radius,method=method,shape=None if shape is None else list(shape))
            for method in ('bresenham','andres')
            for name,r,c,radius,shape in (
                ('tiny',4,5,1,None),('cache',512,512,256,None),
                ('long',0,0,32768,None),('clipped',0,0,1024,(1024,1024)))]
