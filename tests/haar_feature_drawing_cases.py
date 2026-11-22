"""Bounded concrete and protocol recipes for feature.draw_haar_like_feature."""
import copy
import hashlib
import json


DEFAULT_FEATURES = [
    [[[0,0],[2,3]], [[1,1],[3,4]]],
    [[[0,2],[4,5]], [[0,0],[2,2]], [[2,1],[4,4]]],
    [[[2,2],[0,0]], [[4,6],[5,7]]],
]
FAST_CASES = ['gray-u8-all','rgb-f64-overlap','gray-f32-offset','seeded-u8-subset','large-overlap']


def cases():
    rows=[]
    def add(case, **options):
        rows.append({'id':case,**options})
    add('gray-u8-all',dtype='uint8',rng=['pcg',14])
    add('rgb-f64-overlap',dtype='float64',rgb=True,alpha=.25,
        colors=[[.8,.4,-.2],[.2,1.2,.3]])
    add('gray-f32-offset',dtype='float32',r=2,c=1)
    add('seeded-u8-subset',dtype='uint8',rng=['pcg',18],max=2)
    add('large-overlap',dtype='float64',rgb=True,shape=[128,160],features=[
        [[[4,5],[110,130]],[[40,50],[103,119]],[[20,20],[100,110]]],
        [[[35,25],[90,140]],[[10,32],[121,100]]],
    ])
    for dtype in ('bool','uint16','int8','int16','int32','float16','float64','>u2','>f4','complex64','object'):
        add('dtype-'+dtype,dtype=dtype)
    for layout in ('F','negative','strided','readonly','bytes'):
        add('rgb-'+layout,rgb=True,layout=layout,dtype='float64')
    add('gray-readonly',layout='readonly')
    add('roi-width-ignored', width=0,height=-8)
    add('roi-nonsensical-size',width='ignored',height=None)
    add('roi-upper-edge-error',r=99,features=[[[[1,0],[1,0]]]])
    add('roi-right-edge-error',c=88,features=[[[[0,1],[0,1]]]])
    add('roi-too-negative',r=-12,features=[[[[0,0],[0,0]]]])
    add('roi-wrap',r=-2,c=-3,features=[[[[0,0],[1,1]]]])
    add('negative-duplicates', shape=[2,2], features=[[[[-2,-2],[1,1]]]],r=0,c=0)
    add('increasing-reversed',features=[[[[4,5],[2,1]],[[3,4],[0,1]]]])
    add('half-even-corners',features=[[[[.5,1.5],[2.5,4.5]],[[.5,.5],[1.5,1.5]]]])
    add('relative-overlap',features=[[[[0,0],[4,5]],[[1,2],[2,3]],[[0,0],[4,5]],[[1,2],[2,3]]]])
    add('zero-alpha',alpha=0.0)
    add('opaque-alpha',alpha=1.0)
    add('negative-alpha',alpha=-.25)
    add('above-one-alpha',alpha=1.25)
    add('np-float32-alpha',alpha=.1,alpha_dtype='float32',dtype='float32')
    add('nan-palette',colors=[[float('nan'),0,0],[0,1,0]])
    add('float32-subnormal-input',dtype='float32',wide_values=True)
    add('palette-length-one',colors=[[1],[0]])
    add('palette-scalar',colors=[.7,.3])
    add('palette-rgba',colors=[[1,0,0,1],[0,1,0,1]])
    add('palette-invalid',colors=[['not-a-number',0,0],[0,1,0]])
    add('palette-provider',color_provider=True)
    add('rank1',rank=1)
    add('rank4',rank=4)
    add('rgb-onechannel',rgb=True,channels=1)
    add('rgb-rgba',rgb=True,channels=4)
    add('empty-image',shape=[0,4],features=[])
    add('empty-features',features=[],rng=['pcg',77])
    add('empty-feature',features=[[]],rng=['pcg',77])
    add('missing-coords',coord='none')
    add('malformed-rect',features=[[[[0,0]]]])
    add('malformed-endpoints',features=[[[[0],[0]]]])
    add('feature-iterator',coord='iterator')
    add('feature-list',coord='list')
    add('seed-int-subset',rng=['seed',24],max=2)
    add('seed-np-int-subset',rng=['npseed',24],max=2)
    add('seed-mt-subset',rng=['mt',91],max=1)
    add('same-size-permutation',rng=['pcg',20],max=3)
    add('rng-observer',rng=['trace',20],max=3)
    add('no-choice-same-generator',rng=['pcg',20])
    add('zero-choice',rng=['pcg',20],max=0)
    add('negative-choice',rng=['pcg',20],max=-1)
    add('oversized-choice',rng=['pcg',20],max=9)
    add('float-choice',rng=['pcg',20],max=1.5)
    add('true-choice',rng=['pcg',20],max=True)
    add('np-int-choice',rng=['pcg',20],max=1,max_dtype='int32')
    add('choice-bad-rectangle',rng=['pcg',20],max=3,features=[[[[100,100],[100,100]]],*DEFAULT_FEATURES[:2]])
    add('draw-coercion-ragged-list',rng=['pcg',20],max=2,coord='list')
    add('seed-invalid-colors-first',rng=['bad',None],colors=[['bad',0,1],[0,1,0]])
    add('randomstate-provider',rng=['randomstate',7],max=1)
    add('seedsequence-provider',rng=['sequence',31],max=1)
    add('numeric-coordinate-array',coord='numeric',features=DEFAULT_FEATURES[:1]*2,max=2,rng=['pcg',29])
    add('coords-generator-subset',coord='iterator',max=1,rng=['pcg',29])
    add('width-height-prov',width=0,height=0,ignore_size_protocol=True)
    return rows


def materialize(spec,np):
    shape=tuple(spec.get('shape',(8,9)))
    if spec.get('rank')==1: shape=(8,)
    if spec.get('rgb'): shape+= (spec.get('channels',3),)
    if spec.get('rank')==4: shape+=(2,3)
    dtype=np.dtype(spec.get('dtype','uint8'))
    i=np.arange(np.prod(shape),dtype=np.int64).reshape(shape)
    if dtype.kind in 'fc':
        image=((i%101)-50).astype(dtype)/dtype.type(50)
    elif dtype.kind=='i':
        image=((i%61)-30).astype(dtype)
        if image.size: image.flat[0]=np.iinfo(dtype).min
    elif dtype.kind=='b': image=(i%3==0)
    else: image=((i*11+7)%151).astype(dtype)
    image=np.array(image,dtype=dtype,order='C')
    if spec.get('wide_values') and image.size:
        image.flat[0]=np.nextafter(np.float32(0),np.float32(1))
    layout=spec.get('layout','C')
    if layout=='F': image=np.asfortranarray(image)
    if layout=='negative': image=image[::-1]
    if layout=='strided': image=np.repeat(image,2,axis=1)[:,::2]
    if layout=='readonly': image.flags.writeable=False
    if layout=='bytes': image=np.ndarray(image.shape,dtype=dtype,buffer=image.tobytes())

    features=copy.deepcopy(spec.get('features',DEFAULT_FEATURES))
    coord_kind=spec.get('coord','object')
    if coord_kind=='object':
        coords=np.empty(len(features),dtype=object)
        for j,f in enumerate(features): coords[j]=f
    elif coord_kind=='numeric': coords=np.asarray(features,dtype=np.int64)
    elif coord_kind=='list': coords=features
    elif coord_kind=='iterator': coords=iter(features)
    else: coords=None
    protocol=[]
    class Traced(np.random.Generator):
        def choice(self, a, *args, **kwargs):
            protocol.append(['choice',type(a).__name__,kwargs.get('size'),kwargs.get('replace')])
            return super().choice(a,*args,**kwargs)
    class Color:
        def __array__(self,dtype=None,copy=None):
            protocol.append(['color-array',str(dtype),copy])
            return np.array([.8,.4,.1],dtype=dtype)
    class Ignored:
        def __int__(self):
            raise AssertionError('ROI size conversion is not part of the upstream call')
    class BadSeed:
        def __repr__(self):
            return 'bad-haar-seed'
    seed=spec.get('rng',['none',None])
    kind,value=seed
    rng={'none':lambda:None,'pcg':lambda:np.random.default_rng(value),
         'mt':lambda:np.random.Generator(np.random.MT19937(value)),
         'trace':lambda:Traced(np.random.PCG64(value)),'seed':lambda:value,
         'npseed':lambda:np.int64(value),'bad':BadSeed,
         'randomstate':lambda:np.random.RandomState(value),
         'sequence':lambda:np.random.SeedSequence(value)}[kind]()
    colors=spec.get('colors',[[1.,0.,0.],[0.,1.,0.]])
    if spec.get('color_provider'): colors=[Color(),colors[1]]
    alpha=spec.get('alpha',.5)
    if 'alpha_dtype' in spec: alpha=getattr(np,spec['alpha_dtype'])(alpha)
    max_features=spec.get('max')
    if 'max_dtype' in spec: max_features=getattr(np,spec['max_dtype'])(max_features)
    width,height=spec.get('width',shape[1] if len(shape)>1 else 3),spec.get('height',shape[0])
    if spec.get('ignore_size_protocol'): width,height=Ignored(),Ignored()
    return ((image,spec.get('r',1),spec.get('c',1),width,height,coords),
            {'color_positive_block':colors[0],'color_negative_block':colors[1],
             'alpha':alpha,'max_n_features':max_features,'rng':rng},protocol)


def state(rng,np):
    if isinstance(rng,np.random.Generator):
        return json.loads(json.dumps(rng.bit_generator.state,default=lambda x:x.tolist()))
    return None


def image_fingerprint(image,np):
    content=image.tobytes(order='C') if image.dtype.kind!='O' else repr(image.tolist()).encode()
    return hashlib.sha256(content).hexdigest()


def summarize(output,image,np):
    return {'dtype':output.dtype.str,'shape':list(output.shape),'strides':list(output.strides),
            'owned':bool(output.flags.owndata),'writeable':bool(output.flags.writeable),
            'shares_input':bool(np.shares_memory(output,image)),
            'image_sha256':image_fingerprint(output,np)}