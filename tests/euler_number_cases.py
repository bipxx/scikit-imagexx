"""Prospective exact Euler contracts; no candidate output informed the domain.

Binary conversion/padding are explicit providers. Configuration histograms and
LUT coefficients are exact. In 2D the signed integer dot is exact for these
bounded cases. In 3D the source computes (coefs*.125) @ h then Python int.
For Snum=sum(abs(int(c)*int(n)))<=2**53, every scaled product and partial sum
lies on an exactly representable 1/8 grid; compare exact values and scalar
types, not a fitted floating tolerance. Outside that domain no exact native
weighted credit is declared. Masks, histogram metadata, warnings and state
are exact, including source callback ordering and intentional input mutation.
"""
import warnings

API = 'measure.euler_number'
STAGES = ('euler_counts', 'euler_weighted')
UNASSESSED = ('huge Snum>2**53 weighted precision', 'forged/stale storage',
              'arbitrary ndarray/comparison protocols', 'arbitrary object comparisons')


class ConnectivityProbe:
    def __init__(self, mode, events, image):
        self.mode, self.events, self.image = mode, events, image
    def __repr__(self):
        return 'ConnectivityProbe(' + repr(self.mode) + ')'
    def __eq__(self, other):
        self.events.append(('connectivity.eq', other))
        if self.mode == 'raise':
            raise RuntimeError('Euler connectivity probe')
        if self.mode == 'warn':
            warnings.warn('Euler connectivity probe', UserWarning)
            return False
        if self.mode == 'mutate':
            self.image[...] = 0
        return other == 1


def cases():
    rows = []
    def add(id, pattern='ring2', connectivity=None, dtype='uint8', layout='C', outcome='return', policy=None):
        rows.append(dict(id=id, pattern=pattern, connectivity=connectivity, dtype=dtype,
                         layout=layout, outcome=outcome, policy=policy or {}, mutation=connectivity=='probe-mutate'))
    for pattern in ('single2', 'disjoint2', 'ring2', 'diagonal2', 'edge2', 'full2'):
        for connectivity in (1, 2):add(pattern+'-c'+str(connectivity), pattern, connectivity)
    add('empty2', 'empty2')
    for pattern in ('solid3', 'cavity3', 'tunnel3', 'edgecontact3', 'cornercontact3'):
        for connectivity in (1, 3):add(pattern+'-c'+str(connectivity), pattern, connectivity)
    add('empty3', 'empty3')
    add('default3', 'cavity3')
    add('unsupported-connectivity2', 'solid3', 2, outcome='error')
    for pattern in ('rank1', 'rank4', 'list'):
        add('intake-'+pattern, pattern, outcome='error')
    for id, pattern, mode in [('probe-warn2','ring2','warn'), ('probe-one3','solid3','one'),
                              ('probe-warn3','solid3','warn'), ('probe-raise3','solid3','raise'),
                              ('probe-mutates-after-binary','cavity3','mutate')]:
        add(id, pattern, 'probe-'+mode, outcome='error' if mode=='raise' else 'return')
    for dtype in ('bool','uint64','float16','float32','float64'):
        add('dtype-'+dtype, dtype=dtype)
    add('connectivity-array-many','solid3','array-many',outcome='error')
    for layout in ('F','reverse','endian','unaligned'):
        add('layout-'+layout,dtype='float64',layout=layout)
    add('nonfinite-raise','nonfinite2',dtype='float64',policy={'invalid':'raise'})
    assert len(rows)==45 and len({r['id']for r in rows})==45
    return rows


def fixture(spec, np):
    pattern, dtype = spec['pattern'], spec['dtype']
    if pattern=='list':image=[[0,1],[1,0]]
    elif pattern in ('rank1','rank4'):
        image=np.ones((4,)if pattern=='rank1'else(2,2,2,2),dtype=dtype)
    elif pattern=='empty2':image=np.zeros((0,3),dtype=dtype)
    elif pattern=='empty3':image=np.zeros((2,0,3),dtype=dtype)
    elif pattern=='nonfinite2':image=np.array([[np.nan,np.inf],[-np.inf,-0.],[1.,-1.]],dtype=dtype)
    elif pattern.endswith('3'):
        image=np.zeros((7,7,7),dtype=dtype)
        if pattern in ('solid3','cavity3','tunnel3'):
            image[1:-1,1:-1,1:-1]=1
            if pattern=='cavity3':image[2:-2,2:-2,2:-2]=0
            if pattern=='tunnel3':image[3,3,:]=0
        else:
            image[2,2,2]=1
            image[(2,3,3)if pattern=='edgecontact3'else(3,3,3)]=1
    else:
        image=np.zeros((7,8),dtype=dtype)
        if pattern=='single2':image[3,3]=1
        elif pattern=='disjoint2':image[1,1]=image[5,6]=1
        elif pattern=='ring2':image[1:-1,1:-1]=1;image[2:-2,2:-2]=0
        elif pattern=='diagonal2':image=np.eye(7,dtype=dtype)
        elif pattern=='edge2':image[:,0]=1;image[0,:]=1
        elif pattern=='full2':image[:]=1
        else:raise AssertionError(pattern)
        if dtype=='uint64':image*=np.uint64(2**63+1)
    layout=spec['layout']
    if layout=='F':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='endian':image=image.astype(image.dtype.newbyteorder('>'))
    elif layout=='unaligned':
        owner=np.zeros(image.nbytes+1,dtype=np.uint8)
        view=np.ndarray(image.shape,dtype=image.dtype,buffer=owner,offset=1)
        view[...]=image;image=view
    events=[];connectivity=spec['connectivity']
    if isinstance(connectivity,str)and connectivity.startswith('probe-'):
        connectivity=ConnectivityProbe(connectivity[6:],events,image)
    elif connectivity=='array-many':connectivity=np.array([1,3])
    return image,{'connectivity':connectivity},events


def exact_integer_value(histogram, coefficients, ndim):
    assert len(histogram)==len(coefficients)==(16 if ndim==2 else 256)
    terms=[int(c)*int(n)for c,n in zip(coefficients,histogram)]
    assert all(int(n)>=0 for n in histogram)
    if ndim==2:
        assert sum(abs(t)for t in terms)<=2**53
        return sum(terms)
    assert sum(abs(t)for t in terms)<=2**53, 'unassessed scaled-dot precision domain'
    numerator=sum(terms)
    return numerator//8 if numerator>=0 else -((-numerator)//8)
