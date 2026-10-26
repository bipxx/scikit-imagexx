"""Prospective small inertia eigenvalues; policy precedes all new outputs.

Solver tolerance is unchanged from test_symmetric_eigen.py:2e-6 for f32,
2e-13 for f64, relative plus max component absolute scale. Automatic tensors
add max row sum of the frozen inertia cell-error matrix (Weyl for symmetric
perturbations); clipping at zero is1-Lipschitz. No eigenvector conditioning
factor or candidate-dependent allowance is introduced.
"""
import itertools
import math
if __package__:
    from .moments_cases import error_limits as central_limits,fixture as image_fixture
else:
    from moments_cases import error_limits as central_limits,fixture as image_fixture
API='measure.inertia_tensor_eigvals'
STAGE='inertia_eigvals'
UNASSESSED=('empty/rankzero images','forged storage','extended precision','arbitrary object protocols')

def cases():
    rows=[]
    def add(id,n=3,dtype='float64',pattern='ordinary',layout='C',kind='T',route='solver'):
        rows.append(dict(id=id,n=n,dtype=dtype,pattern=pattern,layout=layout,kind=kind,route=route))
    for n in (1,2,3):
        for dtype in ('float32','float64'):add(f'auto-{n}d-{dtype}',n,dtype,kind='auto')
    add('auto-u8',2,'uint8',kind='auto');add('auto-half',3,'float16',kind='auto')
    for pattern,n,dtype in [('diagonal',1,'float64'),('diagonal',2,'float32'),('diagonal',3,'float64'),
        ('zero',3,'float64'),('repeat',3,'float32'),('rank-one',3,'float64'),('near-zero',3,'float64'),
        ('tiny-negative',3,'float64'),('negative',2,'float32'),('lower-triangle',3,'float64')]:
        add('T-'+pattern+str(n),n,dtype,pattern)
    for layout in ('F','step','reverse','endian','readonly','unaligned'):add('layout-'+layout,layout=layout)
    for dtype in ('int16','bool','float16','complex64'):
        add('dtype-'+dtype,dtype=dtype,route='error'if dtype=='float16'else'provider')
    add('T-precedence',2,pattern='precedence')
    add('nonsquare',pattern='nonsquare',route='error')
    add('T-nan',2,pattern='nan',route='provider');add('T-inf',2,pattern='inf',route='provider')
    add('auto-zero',2,pattern='zero',kind='auto',route='provider')
    add('supplied-mu',2,kind='mu')
    assert len(rows)==34 and len({r['id']for r in rows})==34
    return rows

def moment_spec(spec):
    return dict(api='moments_central',shape={1:(7,),2:(4,5),3:(3,4,2)}[spec['n']],
        dtype=spec['dtype'],layout='C',pattern='zero'if spec['pattern']=='zero'else'positive',
        center=None,order=2,spacing=None,extra={},exact=False)

def fixture(spec,np):
    n=spec['n'];image,_,_=image_fixture(moment_spec(spec),np);options={}
    if spec['kind']=='auto':return image,options
    if spec['kind']=='mu':
        mu=np.ones((3,)*n,dtype=spec['dtype']);mu[(0,)*n]=4
        options['mu']=mu;return image,options
    value=np.eye(n,dtype=np.float64)*2
    if n>1:value[0,1]=value[1,0]=.25
    p=spec['pattern']
    if p=='diagonal':value=np.diag(np.arange(1,n+1,dtype=np.float64))
    elif p=='zero':value.fill(0)
    elif p=='repeat':value=np.eye(n)*3
    elif p=='rank-one':v=np.arange(1,n+1,dtype=np.float64);value=v[:,None]*v[None,:]
    elif p=='near-zero':value=np.diag([1.,2**-30,2**-45])
    elif p=='tiny-negative':value=np.diag([1.,2**-20,-2**-45])
    elif p=='negative':value=-np.eye(n)
    elif p=='lower-triangle':value=np.array([[3.,99.,-50.],[.5,2.,75.],[.25,-.125,1.]])
    elif p=='nonsquare':value=np.ones((2,3))
    elif p=='nan':value[0,0]=np.nan
    elif p=='inf':value[0,0]=np.inf
    value=value.astype(spec['dtype'])
    if spec['dtype']=='complex64':value[0,1]+=.5j;value[1,0]-=.5j
    layout=spec['layout']
    if layout=='F':value=np.asfortranarray(value)
    elif layout=='step':
        base=np.zeros((n*2,n*2),dtype=value.dtype);base[::2,::2]=value;value=base[::2,::2]
    elif layout=='reverse':value=value[::-1,::-1]
    elif layout=='endian':value=value.astype(value.dtype.newbyteorder('>'))
    elif layout=='readonly':value.flags.writeable=False
    elif layout=='unaligned':
        base=np.empty(value.nbytes+1,dtype=np.uint8);view=np.ndarray(value.shape,dtype=value.dtype,buffer=base,offset=1);view[...]=value;value=view
    options['T']=value
    if p=='precedence':image=image.astype(np.complex64);image.fill(np.nan);options.update(mu=np.zeros((1,)),spacing='ignored')
    return image,options

def error_limits(spec,image,mu,tensor,expected,np):
    out=np.asarray(expected);bound=np.zeros(out.shape,dtype=np.float64)
    if spec['route']!='solver':return bound
    n=spec['n'];t=2e-6 if out.dtype==np.float32 else 2e-13
    lower=np.tril(tensor)
    scale=float(np.max(np.abs(lower),initial=0.))
    bound[:]=t*np.abs(out.astype(np.float64))+t*scale
    if spec['kind']=='auto':
        # Same tensor cell envelope as the preceding frozen inertia packet.
        e=central_limits(moment_spec(spec),image,mu,np);m=float(mu[(0,)*n]);b=float(e[(0,)*n])
        assert math.isfinite(m)and b>=0 and b<abs(m)/4
        corners=[tuple(2 if k==axis else 0 for k in range(n))for axis in range(n)]
        q=[float(mu[i])for i in corners];qe=[float(e[i])for i in corners];u=float(np.finfo(tensor.real.dtype).eps)/2
        cells=np.zeros((n,n),dtype=np.float64)
        for i,j in itertools.product(range(n),repeat=2):
            if i==j:a,d,p=sum(map(abs,q))+abs(q[i]),sum(qe)+qe[i],n+3
            else:
                index=tuple(1 if k in(i,j)else 0 for k in range(n));a,d,p=abs(float(mu[index])),float(e[index]),3
            g=2*p*u/(1-2*p*u);v=abs(float(tensor[i,j]))
            cells[i,j]=(d+v*b+g*(a+d+v*abs(m)))/(abs(m)-b)
        if n==1:cells[0,0]=0
        bound+=float(np.max(np.sum(cells,axis=1),initial=0.))
    if spec['pattern']in ('diagonal','zero','repeat','near-zero','tiny-negative','negative') or n==1:bound[:]=0
    assert np.all(np.isfinite(bound)&(bound>=0))
    return bound
