"""Exact polygon topology contracts frozen before original/candidate calls.

No numerical tolerance: every label, boolean, result dtype/layout/base, warning,
error, protocol event, and input/state invariant is compared exactly. Vertex
threshold is the source strict 1e-12 comparison; ordered dual-ray arithmetic
must not be replaced with a different geometric predicate. Empty vertices with
nonempty work are original-unsafe (unchecked final vertex read), excluded from
all reference calls and reserved for an explicit future boundary refusal.
Empty queries/grids below retain NONEMPTY vertices, so no unsafe read occurs.
A real short builtin shape tuple is also original-unsafe under Cython unchecked
indexing. Its safe successor uses an ordinary Python shape-index protocol.
"""
import warnings
APIS=('points_in_poly','grid_points_in_poly')
STAGES={'points_in_poly':'points_in_poly','grid_points_in_poly':'grid_points_in_poly'}
UNASSESSED=('empty vertices with nonempty work: unsafe original, boundary refusal only',
            'short builtin shape tuple: unsafe original, boundary refusal only',
            'forged storage','arbitrary mutation callbacks','huge allocation geometry')

class ShapeIndexProbe:
    """No tuple/list fast indexing: Python __getitem__ raises before allocation."""
    def __init__(self,events):self.events=events
    def __repr__(self):return 'ShapeIndexProbe()'
    def __getitem__(self,index):
        self.events.append(('shape.getitem',index))
        if index==0:return 6
        raise IndexError('polygon shape index probe')

class TruthProbe:
    def __init__(self,mode,events):self.mode,self.events=mode,events
    def __repr__(self):return "TruthProbe("+self.mode+")"
    def __bool__(self):
        self.events.append('binarize.bool')
        if self.mode=='raise':raise RuntimeError('polygon binarize probe')
        warnings.warn('polygon binarize probe',UserWarning)
        return False

def cases():
    points=['square','concave','closed','clockwise','bowtie','degenerate-one','degenerate-two',
            'epsilon','nan-points','inf-points','nonfinite-verts','dtype-bool','dtype-int64',
            'dtype-half','dtype-f32','dtype-complex','layout-F','layout-reverse',
            'layout-endian','layout-unaligned','empty-points','extra-columns',
            'short-points','large-points']
    grid=['square','labels','none','concave','closed','clockwise','bowtie','degenerate-one',
          'degenerate-two','empty-rows','empty-cols','negative-shape','shape-index-probe-error',
          'float-shape','probe-warn','probe-raise','complex-verts','nonfinite-verts']
    rows=[dict(api=a,id=a+'-'+n,pattern=n,policy={})for a,names in zip(APIS,(points,grid))for n in names]
    assert len(points)==24 and len(grid)==18 and len(rows)==42
    return rows

def fixture(spec,np):
    n=spec['pattern'];events=[]
    verts=np.array([[0.,0.],[0.,4.],[4.,4.],[4.,0.]])
    points=np.array([[2.,2.],[-1.,2.],[0.,0.],[0.,2.],[4.,4.],[4.,2.],[2.,4.],[2.,0.]])
    if n=='concave':verts=np.array([[0.,0.],[0.,4.],[2.,2.],[4.,4.],[4.,0.]])
    elif n=='closed':verts=np.concatenate((verts,verts[:1]))
    elif n=='clockwise':verts=verts[::-1]
    elif n=='bowtie':verts=np.array([[0.,0.],[4.,4.],[0.,4.],[4.,0.]])
    elif n=='degenerate-one':verts=verts[:1]
    elif n=='degenerate-two':verts=verts[:2]
    elif n=='epsilon':
        e=1e-12;points=np.array([[0.,0.],[e/2,e/2],[e,e],[np.nextafter(e,0),0.],[-e/2,0.],[-e,0.],[2.,np.nextafter(0.,1.)]])
    elif n=='nan-points':points[0,0]=np.nan;points[1,1]=np.nan
    elif n=='inf-points':points[0,0]=np.inf;points[1,1]=-np.inf
    elif n=='nonfinite-verts':verts[1,0]=np.nan;verts[2,1]=np.inf
    if n.startswith('dtype-'):
        dtype={'bool':'bool','int64':'int64','half':'float16','f32':'float32','complex':'complex64'}[n[6:]]
        points=points.astype(dtype);verts=verts.astype(dtype)
        if dtype=='complex64':points+=.25j;verts+=.5j
    if n=='layout-F':points=np.asfortranarray(points);verts=np.asfortranarray(verts)
    elif n=='layout-reverse':points=points[::-1];verts=verts[::-1]
    elif n=='layout-endian':points=points.astype('>f8');verts=verts.astype('>f8')
    elif n=='layout-unaligned':
        def unaligned(x):
            out=np.ndarray(x.shape,dtype=x.dtype,buffer=bytearray(x.nbytes+1),offset=1);out[...]=x;return out
        points=unaligned(points);verts=unaligned(verts)
    elif n=='empty-points':points=np.empty((0,2))
    elif n=='extra-columns':points=np.column_stack((points,np.ones(len(points))));verts=np.column_stack((verts,np.ones(len(verts))))
    elif n=='short-points':points=np.ones((3,1))
    elif n=='large-points':points=np.resize(points,(8193,2))
    if spec['api']=='points_in_poly':return points,dict(verts=verts),events
    shape=(6,7);binarize=True
    if n in ('labels','none','concave','closed','clockwise','bowtie','degenerate-one','degenerate-two','nonfinite-verts'):binarize=False if n!='none'else None
    if n=='empty-rows':shape=(0,7)
    elif n=='empty-cols':shape=(6,0)
    elif n=='negative-shape':shape=(-1,7)
    elif n=='shape-index-probe-error':shape=ShapeIndexProbe(events)
    elif n=='float-shape':shape=(3.5,4)
    elif n.startswith('probe-'):binarize=TruthProbe(n[6:],events)
    elif n=='complex-verts':verts=verts.astype('complex128')+.25j
    return shape,dict(verts=verts,binarize=binarize),events
