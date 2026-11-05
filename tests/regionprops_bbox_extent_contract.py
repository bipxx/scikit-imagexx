"""Bounded prospective area_bbox/extent partition, no constructors/properties."""
import math
API='measure.regionprops'
MEMBERS=('area_bbox','extent')
ALIASES={'BoundingBoxArea':'area_bbox','bbox_area':'area_bbox','Extent':'extent'}
READS=[['attr','area_bbox'],['attr','extent'],['item','area_bbox'],['item','extent'],
       ['item','BoundingBoxArea'],['item','bbox_area'],['item','Extent'],['attr','bbox_area'],
       ['attr','BoundingBoxArea'],['attr','area_bbox'],['attr','extent']]
UNASSESSED=(
    'Other35 properties, intensity/extras/table output and wholeclass lifecycle/serialization/equality',
    'Custom label/spacing/offset/attribute protocols, ndarray subclasses, external/concurrent mutation',
    'Nonfinite/negative/extreme spacing, unsupported dtypes/endianness and broad geometry changes',
    'Nativeprovider/arithmetic implementation, SDK/workers/SIMD/performance and otherprofiles')


def cases():
    configs=[('plain2d',[4,5],'int16','C',None,None,None,None),
        ('anisotropic2d',[4,5],'uint8','F',[.5,2.],[-3,4],'image','pixel'),
        ('area-primed2d',[4,5],'int32','C',[2.,.25],[1,2],'area','pixel'),
        ('unprimed3d',[4,4,4],'uint16','C',[.5,1.,2.],[0,-1,2],None,'pixel'),
        ('slice3d',[4,4,4],'int64','F',[2.,.5,1.5],[2,3,4],'image','slice'),
        ('empty-slice2d',[4,5],'int16','C',[1.,2.],None,'area','empty')]
    result=[]
    for name,shape,dtype,order,spacing,offset,prime,mutation in configs:
        for cache in (False,True):
            sequence=[]
            if prime is not None:sequence.append(['prime',prime])
            if mutation=='pixel':sequence.append(['set-pixel',[[0,0,0],0] if len(shape)==3 else [[1,1],0]])
            elif mutation=='slice':sequence.append(['set-slice',[[1,3]]*3])
            elif mutation=='empty':sequence.append(['set-slice',[[0,0],[1,3]]])
            sequence+=READS
            result.append(dict(id=f'regionprops-bbox-extent/{name}-cache-{cache}',api=API,shape=shape,
                dtype=dtype,order=order,spacing=spacing,offset=offset,cache=cache,sequence=sequence,
                policy={'all':'call'} if mutation=='empty' else {},holdout=name=='slice3d'))
    for name,spacing,policy in [('zero-spacing',[0.,1.],{'all':'raise'}),('bad-spacing-length',[1.,2.,3.],{})]:
        result.append(dict(id='regionprops-bbox-extent/'+name,api=API,shape=[4,5],dtype='int16',order='C',
            spacing=spacing,offset=None,cache=False,sequence=READS,policy=policy,holdout=False))
    return result


def source_safe(spec):
    assert spec['api']==API and spec['shape'] in ([4,5],[4,4,4])
    ndim=len(spec['shape']);assert math.prod(spec['shape'])<=64
    assert spec['dtype'] in ('int16','int32','int64','uint8','uint16') and spec['order'] in ('C','F')
    assert type(spec['cache']) is bool
    spacing=spec['spacing']
    if spacing is not None:
        assert len(spacing)==ndim or spec['id']=='regionprops-bbox-extent/bad-spacing-length'
        assert all(type(v) is float and math.isfinite(v) and 0<=v<=3 for v in spacing)
    if spec['offset'] is not None:
        assert len(spec['offset'])==ndim and all(type(v) is int and abs(v)<=4 for v in spec['offset'])
    assert spec['policy'] in ({},{'all':'call'},{'all':'raise'})
    for action,value in spec['sequence']:
        if action in ('attr','item'):assert [action,value] in READS
        elif action=='prime':assert value in ('image','area')
        elif action=='set-pixel':
            index,pixel=value;assert len(index)==ndim and all(type(i) is int and 0<=i<n for i,n in zip(index,spec['shape'])) and pixel in (0,1,2)
        else:
            assert action=='set-slice' and len(value)==ndim
            assert all(type(a) is int and type(b) is int and 0<=a<=b<=n for (a,b),n in zip(value,spec['shape']))
    return dict(input_elements=math.prod(spec['shape']),maximum_label=2,region_count_upper_bound=2,
        selected_region=0,mask_upper_bound=math.prod(spec['shape']),read_count=sum(a in ('attr','item') for a,v in spec['sequence']),
        floating_frontier='finite small products; explicit NumPy0/0 call/raise safe observation, no nonfiniteinput or nativeundefinedcast')


def fixture(spec,np):
    source_safe(spec)
    image=np.zeros(spec['shape'],dtype=spec['dtype'],order=spec['order'])
    if image.ndim==2:
        for p in ((1,1),(1,2),(2,1)):image[p]=1
        image[3,4]=2
    else:
        for p in ((0,0,0),(0,1,1),(1,0,1),(1,1,1)):image[p]=1
        image[3,3,3]=2
    options=dict(cache=spec['cache'],spacing=None if spec['spacing'] is None else tuple(spec['spacing']),
        offset=None if spec['offset'] is None else tuple(spec['offset']))
    return image,options
