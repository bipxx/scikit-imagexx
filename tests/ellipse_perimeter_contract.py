"""Small source-audited perimeter partition; no original/native draw calls."""
import math
if __package__:
    from .draw_coordinates_contract import ScalarProvider
else:
    from draw_coordinates_contract import ScalarProvider
API='draw.ellipse_perimeter'
MAX_OUTPUT_ELEMENTS=1024
UNASSESSED=(
    'All shape carriers of length<2: unchecked shape[0]/shape[1], never invoke originals',
    'Unequal rotated ellipses: rational Bezier subdivision/termination requires a separate bound before capture/native parity',
    'Nonzero angles outside exact unit-circle cases; nonzero-angle single-zero radius can divide by zero before C-int casts',
    'Nonfinite geometry/angles/shapes, signed-size/C-int overflow or narrowing, negative radii and unbounded allocations',
    'Arbitrary protocols, subclass/custom shape indexing, external-buffer storage and concurrent mutation',
    'Exceptional hardware floating environments and other platforms/profiles; no native/SDK/worker/performance credit')

def cases():
    rows=[]
    def add(name,args,kind='ordinary',options=None,safe=None,holdout=False,policy=None):
        rows.append(dict(id='ellipse-perimeter/'+name,api=API,kind=kind,args=list(args),
            safe_args=list(args) if safe is None else safe,kwargs=options or {},holdout=holdout,policy=policy or {}))
    for name,args in [('unit',(0,0,1,1)),('unequal',(0,0,2,4)),('swapped',(0,0,5,2)),
        ('flat-row',(0,0,1,7)),('flat-column',(0,0,7,1)),('zero-row',(0,0,0,3)),
        ('zero-column',(0,0,3,0)),('point',(3,-4,0,0)),('negative-center',(-4,-3,3,5)),
        ('translated',(16,-16,4,2))]:
        add(name,args,holdout=name in ('swapped','flat-column'),policy={'all':'call'} if name=='unequal' else {})
    for name,angle in [('positive-angle',.3),('negative-angle',-.5)]:
        add(name,(0,0,1,1),options={'orientation':angle},holdout=name=='negative-angle')
    for name,args,shape,kind in [
        ('clip',(3,4,2,4),[6,7],'shape-tuple'),('clip-away',(-12,-12,2,3),[4,5],'shape-tuple'),
        ('zero-shape',(0,0,2,3),[0,7],'shape-tuple'),('extra-shape',(3,4,2,4),[6,7,9],'shape-tuple'),
        ('point-before-clip',(3,4,0,0),[1,1],'shape-tuple'),
        ('list-shape',(3,4,2,4),[6,7],'shape-list'),
        ('float-shape',(3,4,2,4),[6.5,7.5],'shape-tuple'),
        ('reversed-readonly-shape',(3,4,2,4),[6,7],'shape-reverse')]:
        add(name,args,kind,{'shape':shape,**({'orientation':.3} if name=='point-before-clip' else {})})
    add('numpy-int32',(1,2,2,4),'np-int32')
    add('fractional-conversion',(1.5,-2.5,2.5,3.5),'fractional',safe=[1,-2,2,3])
    add('index-provider',(1,2,2,4),'provider-first')
    add('orientation-provider',(1,2,2,4),'provider-angle',{'orientation':0.0})
    for name,args,options in [('overflow-first',(2**100,0,2,3),{}),('none-first',(None,0,2,3),{}),
        ('missing',(0,0,2),{}),('orientation-none',(0,0,2,3),{'orientation':None}),
        ('unexpected-keyword',(0,0,2,3),{'rotation':0.0})]:
        add(name,args,'error-'+name,options);rows[-1]['safe_args']=None
    return rows

def source_safe(spec):
    """No coordinates produced: audit exact integer loop progress/bounded state."""
    assert spec['api']==API
    values=spec['safe_args']
    if values is None:
        assert spec['kind'] in ('error-overflow-first','error-none-first','error-missing','error-orientation-none','error-unexpected-keyword')
        return dict(preentry_rejection=True,max_output_elements=0)
    assert len(values)==4 and all(type(v)is int for v in values)
    ro,co,rr,cr=values
    assert abs(ro)<=16 and abs(co)<=16 and 0<=rr<=8 and 0<=cr<=8
    options=spec['kwargs'];shape=options.get('shape');angle=options.get('orientation',0)
    if shape is not None:
        assert len(shape)>=2 and len(shape)<=3
        assert all(type(v)in(int,float) and math.isfinite(v) and 0<=v<=32 for v in shape)
    assert type(angle) in(int,float) and math.isfinite(angle) and abs(angle)<=.5
    if rr==cr==0:
        return dict(preentry_rejection=False,max_output_elements=2,early_0D_before_shape=True)
    if angle!=0:
        assert values==[0,0,1,1]
        # cd=rd=1 => za=0 exactly despite finite sin, sqrt=1, bounds+-1,
        # width2, icd=ird=1, w=.5 exactly. Four reflected corner segments:
        # initial d1c=d1r=2, rc=0, d2c=1,d2r=-1,err=0. Both axes move
        # once to endpoint, d2r=1 exits, then _line(endpoint,endpoint) emits1.
        # No weight<.5 subdivision; every operation is exact small binary FP.
        return dict(preentry_rejection=False,max_output_elements=16,unit_circle_half_weight=True)
    rd,cd=rr*rr,cr*cr;c=-cr;r=0;err=c*(2*rd+c)+rd;loops=0;largest=abs(err)
    while c<=0:
        previous=(c,r);e2=2*err
        left=(2*c+1)*rd;right=(2*r+1)*cd
        largest=max(largest,abs(e2),abs(left),abs(right))
        if e2>=left:c+=1;err+=(2*c+1)*rd
        if e2<=right:r+=1;err+=(2*r+1)*cd
        loops+=1;largest=max(largest,abs(err))
        assert (c,r)!=previous and -cr<=c<=1 and 0<=r<=rr+2
        assert loops<=rr+cr+3 and largest<2**31
    tips=max(0,rr-r);points=4*loops+2*tips
    assert 2*points<=MAX_OUTPUT_ELEMENTS
    return dict(preentry_rejection=False,max_output_elements=2*points,integer_iterations=loops,
        max_abs_recurrence_value=largest,translation_headroom=2**31-1-(max(abs(ro),abs(co))+rr+cr+2))

def fixture(spec,np):
    bound=source_safe(spec);args=list(spec['args']);options=spec['kwargs'].copy();events=[];buffers={}
    if spec['kind']=='np-int32':args=[np.int32(v) for v in args]
    elif spec['kind']=='provider-first':args[0]=ScalarProvider(args[0],events,'r')
    elif spec['kind']=='provider-angle':options['orientation']=ScalarProvider(options['orientation'],events,'orientation')
    if 'shape' in options:
        shape=options['shape']
        if spec['kind']=='shape-reverse':
            owner=np.array(shape[::-1],dtype=np.intp);value=owner[::-1];value.flags.writeable=False
            options['shape']=value;buffers['shape']=value
        elif spec['kind']=='shape-list':options['shape']=list(shape)
        else:options['shape']=tuple(shape)
    return tuple(args),options,events,buffers,bound
