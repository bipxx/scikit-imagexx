"""Finite source-first recipes only; target/capture/native budgets remain zero."""
API='transform.order_angles_golden_ratio'
SIGNATURE='(theta)'
MAX_N=16
MAX_BACKING_BYTES=256
MAX_RESUMES=17
DTYPE_BYTES={'bool':1,'int8':1,'uint8':1,'int16':2,'uint16':2,'float16':2,'int32':4,'uint32':4,'float32':4,'int64':8,'uint64':8,'float64':8,'>f8':8}


def cases():
    rows=[]
    def add(name,values=(90.,0.,45.,135.),**kw):
        r=dict(id=name,api=API,values=list(values),carrier='ndarray',dtype='float64',layout='C',readonly=False,protocol='exhaust',mutation=None,call_kind='ordinary');r.update(kw);rows.append(r)
    for dtype in DTYPE_BYTES:
        add('dtype-'+dtype,values=(0,1,0,1)if dtype=='bool'else(90,0,30,60),dtype=dtype)
    for name,values in [('singleton-zero',[0.]),('singleton90',[90.]),('two-forward',[0.,90.]),('two-reverse',[90.,0.]),('three',[0.,60.,120.]),('unsorted-five',[120.,0.,75.,15.,150.]),('eight',[i*22.5 for i in range(8)]),('sixteen',[i*11.25 for i in range(16)]),('negative',[-180.,-45.,0.,90.]),('outside-period',[-360.,0.,180.,360.]),('duplicates',[0.,0.,45.,90.]),('all-equal',[0.,0.,0.,0.]),('signed-zero',[-0.,0.,90.,180.]),('empty',[])]:add(name,values)
    for layout in ('reverse','step','F'):add('layout-'+layout,layout=layout)
    add('readonly',readonly=True)
    for carrier in ('list','tuple'):
        add(carrier+'-singleton',[15.],carrier=carrier)
        add(carrier+'-lazy-fancy-index-error',[0.,90.,45.],carrier=carrier)
    for protocol in ('close-created','prefix-close','send-none-exhaust','send-token-after-first','non-none-send-created'):add('protocol-'+protocol,protocol=protocol)
    add('mutate-before-first',mutation={'after_yields':0,'index':1,'value':-15.})
    add('mutate-after-first',mutation={'after_yields':1,'index':1,'value':-15.})
    add('missing-theta',values=[],carrier='list',call_kind='missing')
    add('extra-positional',values=[0.],carrier='list',call_kind='extra')
    add('unknown-keyword',values=[0.],carrier='list',call_kind='unknown')
    assert len({r['id']for r in rows})==len(rows)
    return rows


def case_bounds(spec):
    assert spec in cases()
    n=len(spec['values']);item=DTYPE_BYTES[spec['dtype']];factor=2 if spec['layout']=='step'else 1
    assert n<=MAX_N and spec['layout']in('C','reverse','step','F')
    assert all(type(v)in(int,float)and -360<=v<=360 for v in spec['values'])
    mutation=spec['mutation']
    if mutation:assert 0<=mutation['index']<n and mutation['after_yields']in(0,1)and -360<=mutation['value']<=360 and not spec['readonly']
    return dict(n=n,backing_elements=factor*n,backing_bytes=factor*n*item,index_words_upper=n,index_bytes_upper=8*n,gather_elements_total_upper=n*(n-1)//2,gather_live_bytes_upper=max(0,n-1)*item,successful_yields_upper=n,resume_attempts_upper=n+1,arbitrary_provider_work_bound=False)


def fixture(spec,np):
    """Future NumPy setup; currently tested with REQUEST-refusing stand-in only."""
    case_bounds(spec)
    if spec['call_kind']=='missing':return (),{}
    if spec['carrier']=='list':theta=list(spec['values'])
    elif spec['carrier']=='tuple':theta=tuple(spec['values'])
    else:
        values=spec['values']
        if spec['layout']=='reverse':values=list(reversed(values))
        elif spec['layout']=='step':values=[x for v in values for x in(v,0)]
        theta=np.asarray(values,dtype=spec['dtype'])
        if spec['layout']=='reverse':theta=theta[::-1]
        elif spec['layout']=='step':theta=theta[::2]
        elif spec['layout']=='F':theta=theta.copy(order='F')
        if spec['readonly']:theta.setflags(write=False)
    if spec['call_kind']=='extra':return (theta,0),{}
    if spec['call_kind']=='unknown':return (theta,),{'unexpected':1}
    return (theta,),{}
