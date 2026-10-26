"""Coordinate moments, fixed before any new original/candidate observations.

The ordinary native stage consumes the same typed NumPy powers. For each cell
S=sum(abs(original prepared monomials)); u=eps(result component)/2,
q=N+D, gamma=q*u/(1-q*u); allowance=2*gamma*S, covering the two ordered
D-product/N-sum paths. Finite bounds fail closed. Exact dyadic/structural
controls use zero allowance. K1, complex/exceptional policies and errors are
explicit primitive-provider partitions, not owned reduction credit.
"""
APIS=('moments_coords','moments_coords_central')
STAGE='moments_coords_reduce'
UNASSESSED=('forged/stale array storage','arbitrary array/protocol subclasses','extended precision','unbounded high-dimensional output')

def cases():
    rows=[]
    def add(id,api=APIS[0],N=7,D=2,dtype='float64',order=3,center=None,layout='C',pattern='ordinary',route='native',outcome='return',policy=None,exact=False):
        rows.append(dict(id=id,api=api,N=N,D=D,dtype=dtype,order=order,center=center,layout=layout,pattern=pattern,route=route,outcome=outcome,policy=policy or {},exact=exact))
    for api in APIS:
        for D in (1,2,3):
            for dtype in ('float32','float64'):add(f'{api}-{D}d-{dtype}',api,D=D,dtype=dtype)
    add('N1',N=1);add('N129',APIS[1],N=129,D=3);add('N8193',N=8193)
    add('K1-raw',dtype='float32',order=0,route='provider');add('K1-central',APIS[1],D=3,order=0,route='provider')
    add('order1',D=3,dtype='float32',order=1);add('order4',APIS[1],order=4)
    add('tuple-raw',dtype='uint8',layout='tuple');add('tuple-central',APIS[1],D=3,layout='tuple')
    for i,layout in enumerate(('F','step','reverse','endian','readonly')):add('layout-'+layout,APIS[i%2],dtype='float32',layout=layout)
    for i,dtype in enumerate(('int16','uint8','float16','bool')):add('dtype-'+dtype,APIS[i%2],dtype=dtype)
    for name,center in [('zero',0),('fractional',(.25,-.5)),('scalar64',{'scalar':'float64','value':.25}),('array32',{'array':'float32','values':(.25,-.5)}),('complex',(.25+.125j,-.5j))]:
        add('center-'+name,APIS[1],dtype='float32',center=center,route='provider'if name=='complex'else'native')
    add('signed-dyadic',pattern='signed',exact=True);add('translated',APIS[1],D=3,pattern='translated');add('single-zero',APIS[1],N=1,exact=True)
    add('nan',pattern='nan',route='provider');add('inf',APIS[1],pattern='inf',route='provider')
    add('overflow-warn',order=4,pattern='huge',route='provider');add('overflow-raise',order=4,pattern='huge',route='error',outcome='error',policy={'over':'raise'})
    add('underflow-warn',APIS[1],center=0,pattern='tiny',route='provider',policy={'under':'warn'})
    add('empty',N=0,route='error',outcome='error');add('rank',pattern='rank',route='error',outcome='error')
    add('negative-order',order=-1,route='error',outcome='error');add('center-shape',APIS[1],center=(1.,2.,3.),route='error',outcome='error')
    add('complex-input',dtype='complex64',route='error',outcome='error')
    assert len(rows)==48 and len({r['id']for r in rows})==48
    return rows

def fixture(spec,np):
    N,D=spec['N'],spec['D'];value=((np.arange(N*D,dtype=np.float64).reshape(N,D)*5)%23-11)/8
    if spec['pattern']=='translated':value+=16
    elif spec['pattern']=='nan'and N:value[0,0]=np.nan
    elif spec['pattern']=='inf'and N:value[-1,-1]=np.inf
    elif spec['pattern']=='huge':value.fill(1e200)
    elif spec['pattern']=='tiny':value*=1e-200
    elif spec['pattern']=='rank':value=np.arange(N,dtype=np.float64)
    value=value.astype(spec['dtype']);layout=spec['layout']
    if layout=='F':value=np.asfortranarray(value)
    elif layout=='step':
        base=np.zeros((N*2,D*2),dtype=value.dtype);base[::2,::2]=value;value=base[::2,::2]
    elif layout=='reverse':value=value[::-1,::-1]
    elif layout=='endian':value=value.astype(value.dtype.newbyteorder('>'))
    elif layout=='readonly':value.flags.writeable=False
    elif layout=='tuple':value=tuple(value[:,d].copy()for d in range(D))
    options=dict(order=spec['order'])
    if spec['api']==APIS[1]:
        center=spec['center']
        if isinstance(center,dict):
            center=np.dtype(center['scalar']).type(center['value'])if 'scalar'in center else np.array(center['values'],dtype=center['array'])
        options['center']=center
    return value,options

def error_limits(spec,calc,expected,np):
    bound=np.zeros(expected.shape,dtype=np.float64)
    if spec['route']!='native'or spec['exact']:return bound
    assert isinstance(calc,np.ndarray)and calc.shape==(spec['N'],)+expected.shape
    assert np.all(np.isfinite(calc))and np.all(np.isfinite(expected))
    u=float(np.finfo(expected.real.dtype).eps)/2;q=spec['N']+spec['D'];assert q*u<1
    gamma=q*u/(1-q*u)
    S=np.sum(np.abs(calc).astype(np.float64),axis=0,dtype=np.float64)
    bound=2*gamma*S
    assert np.all(np.isfinite(bound)&(bound>=0))
    return bound
