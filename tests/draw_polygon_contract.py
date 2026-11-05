"""Prospective small polygon fill recipes; no scientific work on import."""
import math
APIS=('draw.polygon','draw.polygon2mask')
MAX_VERTICES=8
MAX_SCAN=169
MAX_MASK=64
UNASSESSED=(
    'r/c unequal lengths: c longer than r can read unchecked yp memory; do not capture',
    'nonfinite/huge vertices, overflow, empty-c nonempty-r, arbitrary protocols/ragged/object/complex inputs',
    'shape shorter than2: unchecked compiled indexing is not a safe Python IndexError assumption',
    'polygon2mask non2D image shapes and broad shape/dtype/error precedence',
    'native arithmetic/SDK/SIMD/workers/performance, hardware fenv flags and broader floating policies')


def cases():
    triangle=[[1,1],[1,5],[4,2]]
    rectangle=[[0,0],[0,4],[3,4],[3,0]]
    concave=[[0,0],[0,5],[2,2],[5,5],[5,0]]
    epsilon=[[2.**-41,2.**-41],[0,3],[3,0]]
    configs=[
        ('triangle-list',triangle,'list','float64',None,False),
        ('rectangle-i32',rectangle,'C','int32',[6,7],False),
        ('concave-reverse',concave,'reverse','float64',None,False),
        ('bowtie-strided',[[0,0],[4,4],[0,4],[4,0]],'strided','float64',[6,6],False),
        ('negative-clip',[[-2,-3],[1,5],[5,1]],'C','float32',[5,6],True),
        ('fully-negative',[[-4,-3],[-2,-3],[-2,-1]],'list','float64',None,False),
        ('zero-shape',rectangle,'C','float64',[0,6],False),
        ('epsilon-vertex',epsilon,'list','float64',[5,5],False),
        ('singleton',[[2,3]],'scalar','int64',None,False),
        ('collinear-duplicate',[[1,0],[1,2],[1,2],[1,4]],'C','float64',[5,6],False),
        ('empty',[],'C','float64',None,False),
        ('rank2-columns',triangle,'rank2','float64',None,False)]
    result=[dict(id='draw.polygon/'+n,api=APIS[0],vertices=v,carrier=k,dtype=d,shape=s,readonly=ro)for n,v,k,d,s,ro in configs]
    configs=[
        ('triangle-list',triangle,'list','float64',[6,7],False),
        ('rectangle-f64-F',rectangle,'F','float64',[6,7],True),
        ('concave-f32-C',concave,'C','float32',[7,8],False),
        ('negative-strided',[[-2,-3],[1,5],[5,1]],'strided','float64',[5,6],False),
        ('outside',[[8,8],[8,10],[10,8]],'C','int32',[3,4],False),
        ('zero-shape',rectangle,'C','float64',[0,6],False),
        ('epsilon-vertex',epsilon,'C','float64',[4,4],False),
        ('singleton',[[2,3]],'C','float64',[4,5],False),
        ('empty',[],'C','float64',[4,5],False),
        ('three-columns',[[0,0,1],[1,1,2],[2,0,3]],'C','float64',[4,5],False),
        ('negative-shape',triangle,'list','float64',[-1,5],False),
        ('rank1',[0.,1.,2.],'C','float64',[4,5],False)]
    result +=[dict(id='draw.polygon2mask/'+n,api=APIS[1],vertices=v,carrier=k,dtype=d,shape=s,readonly=ro)for n,v,k,d,s,ro in configs]
    return result


def source_safe(spec):
    assert spec in cases(), 'Only frozen prospective IDs/values may enter this partition'
    assert spec['api']in APIS and spec['carrier']in('list','C','F','strided','reverse','scalar','rank2')
    vertices=spec['vertices'];flat=[v for row in vertices for v in(row if isinstance(row,list)else[row])]
    assert all(type(v)in(int,float)and math.isfinite(v)and -4<=v<=12 for v in flat)
    assert len(vertices)<=MAX_VERTICES and type(spec['readonly'])is bool
    shape=spec['shape']
    if shape is not None:assert len(shape)==2 and all(type(v)is int and -1<=v<=8 for v in shape)
    error=spec['id'].rsplit('/',1)[1]in('empty','rank2-columns','three-columns','negative-shape','rank1')
    if spec['id'].endswith('/empty'):scan=0;reason='r.min reduction fails before unchecked classifier'
    elif spec['id'].endswith(('/rank2-columns','/three-columns','/rank1')):scan=0;reason='Python unpack or typedmemoryview-rank error before loop'
    else:
        rows=[v[0]for v in vertices];cols=[v[1]for v in vertices]
        lo=[int(max(0,min(v)))for v in(rows,cols)];hi=[math.ceil(max(v))for v in(rows,cols)]
        if shape is not None:hi=[min(hi[i],shape[i]-1)for i in range(2)]
        scan=max(0,hi[0]-lo[0]+1)*max(0,hi[1]-lo[1]+1);reason='equal nonempty finite bounded coordinate columns'
    assert scan<=MAX_SCAN
    mask=0 if spec['api']==APIS[0]or any(v<0 for v in shape)else math.prod(shape)
    assert mask<=MAX_MASK
    return dict(max_classifier_calls=scan,max_output_pairs=scan,max_mask_elements=mask,max_vertex_storage_words=4*MAX_VERTICES,
        expected_error_hint=error,safety_basis=reason,array_payload_bytes_upper_bound=16*scan+mask+8*4*MAX_VERTICES,
        python_coordinate_list_items_upper_bound=2*scan)


def fixture(spec,np):
    bounds=source_safe(spec);vertices=spec['vertices'];carrier=spec['carrier'];dtype=spec['dtype'];buffers={}
    def array(values,name):
        expanded=[v for value in values for v in(value,0 if not isinstance(value,list)else[0]*len(value))]if carrier=='strided'else values
        out=np.array(expanded,dtype=dtype,order='F'if carrier=='F'else'C')
        if not vertices and spec['api']==APIS[1]:out=out.reshape((0,2))
        if carrier=='strided':out=out[::2]
        if carrier=='reverse':out=out[::-1]
        if spec['readonly']:out.flags.writeable=False
        buffers[name]=out;return out
    shape=None if spec['shape']is None else tuple(spec['shape'])
    if spec['api']==APIS[0]:
        r=[v[0]for v in vertices];c=[v[1]for v in vertices]
        if carrier=='scalar':r,c=r[0],c[0]
        elif carrier!='list':
            r=array([r]if carrier=='rank2'else r,'r');c=array([c]if carrier=='rank2'else c,'c')
        return(r,c),dict(shape=shape),buffers,bounds
    polygon=vertices if carrier=='list'else array(vertices,'polygon')
    return(shape,polygon),{},buffers,bounds
