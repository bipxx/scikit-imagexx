"""Source-safe prospective anti-aliased lines; factories never call draw code."""
if __package__:
    from .draw_coordinates_contract import ScalarProvider
else:
    from draw_coordinates_contract import ScalarProvider

API='draw.line_aa'
MAX_OUTPUT_ELEMENTS=4096
INT_MAX=(1<<31)-1
UNASSESSED=('C-int narrowing or neighbor overflow after successful Py_ssize_t conversion',
            'dc*dc+dr*dr exceeding signed C-int range; never called for reference',
            'unbounded outputs, allocation failure, concurrent mutation, forged storage',
            'nonfinite/complex/object-array conversions and arbitrary user protocol effects',
            'hardware floating-environment equivalence outside ordinary default rounding')


def cases():
    rows=[('point',(3,4,3,4)),('horizontal',(2,1,2,8)),('vertical',(1,2,8,2)),
          ('diagonal',(0,0,7,7)),('antidiagonal',(0,7,7,0))]
    for name,dr,dc in (('down-right-shallow',3,8),('down-right-steep',8,3),
                       ('up-right-shallow',-3,8),('up-right-steep',-8,3),
                       ('down-left-shallow',3,-8),('down-left-steep',8,-3),
                       ('up-left-shallow',-3,-8),('up-left-steep',-8,-3)):
        rows.append((name,(5,6,5+dr,6+dc)))
    rows += [('reverse-shallow',(3,8,0,0)),('reverse-steep',(8,3,0,0)),
             ('one-column',(0,0,0,1)),('one-row',(0,0,1,0)),
             ('one-diagonal',(0,0,1,1)),('reverse-one',(1,1,0,0)),
             ('half-tie',(0,0,2,4)),('reverse-half',(2,4,0,0)),
             ('pythagoras',(0,0,3,4)),('nearly-axis',(0,0,1,31)),
             ('medium',(5,-3,257,126)),
             ('translated',(2**30,-2**30,2**30+7,-2**30+3)),
             ('near-int32-max',(INT_MAX-20,INT_MAX-30,INT_MAX-13,INT_MAX-27)),
             ('near-int32-min',(-INT_MAX+20,-INT_MAX+30,-INT_MAX+27,-INT_MAX+33))]
    result=[dict(id='line-aa/'+name,api=API,kind='ordinary',args=list(args),kwargs={},safe_args=list(args),policy={},holdout=name in('up-left-steep','reverse-half','near-int32-min'))for name,args in rows]
    for dtype in ('int8','int16','int32','int64','uint8','uint16','uint32','uint64'):
        result.append(dict(id='line-aa/np-'+dtype,api=API,kind='np-'+dtype,args=[1,2,5,7],safe_args=[1,2,5,7],kwargs={},policy={},holdout=dtype=='uint64'))
    for kind,args,safe in (('bool',[False,True,True,False],[0,1,1,0]),
                           ('float-integral',[1.,2.,5.,7.],[1,2,5,7]),
                           ('float-fractional',[1.5,2.5,5.5,7.5],[1,2,5,7]),
                           ('np-float32',[1.5,2.5,5.5,7.5],[1,2,5,7]),
                           ('np-float64',[1.5,2.5,5.5,7.5],[1,2,5,7]),
                           ('scalar-array',[1,2,5,7],[1,2,5,7]),
                           ('provider-first',[1,2,5,7],[1,2,5,7]),
                           ('providers-all',[1,2,5,7],[1,2,5,7])):
        result.append(dict(id='line-aa/'+kind,api=API,kind=kind,args=args,safe_args=safe,kwargs={},policy={},holdout=False))
    for kind,args,kwargs in (('overflow-first',[2**100,2,5,7],{}),('overflow-last',[1,2,5,2**100],{}),
                             ('none-first',[None,2,5,7],{}),('missing',[1,2,5],{}),
                             ('unexpected-keyword',[1,2,5,7],{'shape':[8,9]})):
        result.append(dict(id='line-aa/'+kind,api=API,kind=kind,args=args,safe_args=None,kwargs=kwargs,policy={},holdout=False))
    return result


def source_safe(spec):
    """Check our frozen reference admission before constructing/calling originals."""
    values=spec['safe_args']
    if values is None:
        assert spec['kind']in('overflow-first','overflow-last','none-first','missing','unexpected-keyword')
        return dict(preentry_rejection=True)
    assert len(values)==4 and all(type(v)is int for v in values)
    assert all(-INT_MAX+1<=v<=INT_MAX-1 for v in values)
    dr,dc=abs(values[2]-values[0]),abs(values[3]-values[1])
    assert dr*dr+dc*dc<=INT_MAX
    # At least one monotone coordinate advances at each nonterminal iteration.
    # Every emission and neighbor addition fits int32; three emissions/iteration.
    bound=3*(dr+dc+1)
    assert bound*3<=MAX_OUTPUT_ELEMENTS
    return dict(preentry_rejection=False,dr=dr,dc=dc,squared_sum=dr*dr+dc*dc,output_entries_upper_bound=bound)


def fixture(spec,np):
    source_safe(spec)
    args=list(spec['args']);options=spec['kwargs'].copy();events=[];buffers={};kind=spec['kind']
    if kind.startswith('np-'):args=[np.dtype(kind[3:]).type(v)for v in args]
    elif kind=='scalar-array':
        args[0]=np.array(args[0],dtype=np.int64);buffers['r0']=args[0]
    elif kind=='provider-first':args[0]=ScalarProvider(args[0],events,'r0')
    elif kind=='providers-all':args=[ScalarProvider(v,events,name)for v,name in zip(args,('r0','c0','r1','c1'))]
    return tuple(args),options,events,buffers


def benchmark_cases():
    return [dict(id='line-aa/'+name,api=API,args=list(args),kwargs={})for name,args in(
        ('tiny',(0,0,7,3)),('horizontal',(0,0,0,2048)),('vertical',(0,0,2048,0)),
        ('diagonal',(0,0,1024,1024)),('shallow',(0,0,512,2048)),
        ('steep',(0,0,2048,512)),('reverse',(512,2048,0,0)),
        ('long',(0,0,8192,16384)),('translated',(2**30,-2**30,2**30+512,-2**30+2048)))]
