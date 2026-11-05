"""Bounded prospective line_nd cases; factories never invoke draw/linspace."""
if __package__:
    from .draw_coordinates_contract import ArrayProvider
else:
    from draw_coordinates_contract import ArrayProvider
API = 'draw.line_nd'
MAX_DIMENSIONS = 8
MAX_POINTS = 512
MAX_OUTPUT_ELEMENTS = 4096
UNASSESSED = (
    'Unbounded dimension/output/allocation and inputvector length beyond8',
    'Wide integer subtraction/abs overflow or unsigned underflow beyond explicitbounded8bit cases',
    'Integercast outside native int range, nonfinite successful outputs and nonordinary rounding environment',
    'Higher-rank broadcasting, arbitrary object/array subclasses/protocols and nonbool flag protocols',
    'Forged storage/concurrent mutation and platform observations outside retainedLinux64 profiles',
    'Historical prospective packet has no original results; later captured results provide reference only, never native implementation, registration or performance credit')


def cases():
    rows = []
    def add(name,start,stop,endpoint=False,integer=True,kind='list',holdout=False,options=None):
        rows.append(dict(id='line-nd/'+name,api=API,kind=kind,start=start,stop=stop,
            options=dict(endpoint=endpoint,integer=integer) if options is None else options,
            policy={},holdout=holdout))
    deltas = [1,3,-2,4,-3,2,5,-1]
    for ndim in (1,2,3,4,8):
        for endpoint in (False,True):
            for integer in (False,True):
                add(f'{ndim}d-end-{endpoint}-int-{integer}',[0]*ndim,deltas[:ndim],endpoint,integer,holdout=ndim==8 and endpoint and not integer)
    for endpoint in (False,True):
        for integer in (False,True):
            add(f'point-end-{endpoint}-int-{integer}',[1.5,-2.5],[1.5,-2.5],endpoint,integer)
    for name,start,stop,endpoint in (
        ('half-positive',[.5],[4.5],True),('half-descending',[4.5],[.5],True),
        ('half-negative-phase',[-3.5],[.5],False),('half-near-tie',[.5000000000000001],[4.5],True)):
        add(name,start,stop,endpoint,True,holdout=name=='half-descending')
    add('broadcast-start',[-.5],[2.5,1.5])
    add('broadcast-stop',[-.5,1.5],[3.5],True)
    add('fractional-float',[.1,.2,.3],[1.3,4.6,-2.1],True,False)
    for dtype in ('float16','float32','float64','int8','uint8','uint64'):
        start,stop = ([120],[-120]) if dtype=='int8' else ([5,6],[1,2]) if dtype=='uint8' else ([1,2],[5,7]) if dtype=='uint64' else ([1.5,-2.5],[4.5,3.5])
        add('dtype-'+dtype,start,stop,False,not dtype.startswith('float'),'dtype-'+dtype)
    for layout in ('reverse','readonly-unaligned','array-provider'):
        add(layout,[1.5,.5,-2.5],[4.5,3.5,1.5],True,True,layout)
    add('scalar-float',.5,3.5,False,False,'scalar')
    add('scalar-integer',.5,3.5,False,True,'scalar')
    for name,start,stop in (
        ('empty',[],[]),('mismatch',[0,1],[1,2,3]),('nonnumeric',['a','b'],['c','d']),
        ('nan',[0.,0.],[0.,0.]),('inf',[0.,0.],[0.,0.]),
        ('bool',[False,True],[True,False]),('missing-stop',[0,0],[1,2]),('keyword',[0,0],[1,2])):
        add('error-'+name,start,stop,kind='error-'+name)
    add('underflow-step',[0.,0.],[4.2,5e-324],False,False,holdout=True)
    rows[-1]['policy'] = dict(under='call')
    add('signature-defaults',[0,1],[4,3],options={})
    add('signed-zero',[-0.,0.],[0.,-0.],True,False)
    add('dtype-metadata',[1.5,.5],[4.5,3.5],True,False,'metadata')
    return rows


def source_safe(spec):
    assert spec['api'] == API and spec['kind'] in ('list','scalar','metadata','reverse','readonly-unaligned','array-provider','dtype-float16','dtype-float32','dtype-float64','dtype-int8','dtype-uint8','dtype-uint64','error-empty','error-mismatch','error-nonnumeric','error-nan','error-inf','error-bool','error-missing-stop','error-keyword')
    assert not set(spec['options'])-{'endpoint','integer'}
    assert all(type(v) is bool for v in spec['options'].values())
    kind = spec['kind']
    if kind.startswith('error-'):
        # Empty/mismatch/string/bool/nonfinite reject beforelinspace; arity and
        # keyword reject beforefunctionbody. Onlysmallknownfixturearrays exist.
        assert len(spec['start']) <= 3 and len(spec['stop']) <= 3
        return dict(prelinspace_rejection=True,max_points=0,max_output_elements=0)
    start = [spec['start']] if kind == 'scalar' else spec['start']
    stop = [spec['stop']] if kind == 'scalar' else spec['stop']
    assert 1 <= len(start) <= MAX_DIMENSIONS and 1 <= len(stop) <= MAX_DIMENSIONS
    assert len(start)==len(stop) or 1 in (len(start),len(stop))
    assert all(type(v) in (int,float) and abs(v) <= 128 for v in start+stop)
    ndim = max(len(start),len(stop))
    if kind == 'dtype-uint64':
        assert all(a <= b for a,b in zip(start,stop)) # no wideunsignedwrap
    if kind == 'dtype-int8':
        assert start == [120] and stop == [-120] # int8delta wraps16, safe
    if kind == 'dtype-uint8':
        assert start == [5,6] and stop == [1,2] # uint8delta252, bounded
    if kind == 'scalar':
        assert start == [.5] and stop == [3.5]
    assert MAX_POINTS*ndim <= MAX_OUTPUT_ELEMENTS
    return dict(prelinspace_rejection=False,max_points=MAX_POINTS,
                max_output_elements=MAX_POINTS*ndim,scalar_integer_len_error=kind=='scalar' and spec['options'].get('integer',True))


def fixture(spec,np):
    source_safe(spec)
    start,stop = spec['start'],spec['stop']
    options = spec['options'].copy()
    events,buffers = [],{}
    kind = spec['kind']
    if kind.startswith('dtype-'):
        start,stop = np.array(start,dtype=kind[6:]),np.array(stop,dtype=kind[6:])
    elif kind in ('reverse','readonly-unaligned','array-provider','metadata'):
        start,stop = np.array(start,dtype=np.float64),np.array(stop,dtype=np.float64)
        if kind == 'metadata':
            dtype = np.dtype(np.float32,metadata={'line-nd':'capture-conversion'})
            start,stop = start.astype(dtype),stop.astype(dtype)
        elif kind == 'reverse':
            start,stop = start[::-1],stop[::-1]
        elif kind == 'readonly-unaligned':
            values = []
            for array in (start,stop):
                view = np.ndarray(array.shape,dtype=array.dtype,buffer=bytearray(array.nbytes+1),offset=1)
                view[:] = array
                view.flags.writeable = False
                values.append(view)
            start,stop = values
    elif kind in ('error-nan','error-inf'):
        start,stop = list(start),list(stop)
        stop[0] = float('nan') if kind == 'error-nan' else float('inf')
    for name,array in (('start',start),('stop',stop)):
        if type(array) is np.ndarray:
            buffers[name] = array
    if kind == 'array-provider':
        start,stop = ArrayProvider(start,events),ArrayProvider(stop,events)
    if kind == 'error-keyword':
        options['shape'] = (3,4)
    args = (start,) if kind == 'error-missing-stop' else (start,stop)
    return args,options,events,buffers
