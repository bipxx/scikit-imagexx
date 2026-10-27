"""Prospective mesh-area contracts, frozen before any candidate or reference output.

Ordinary native scope is finite float32/float64 triangular geometry. Let u be
the component unit roundoff, F the face count, a=v0-v1 and b=v0-v2 formed
exactly from stored coordinates, and S=1/2 sum_faces sum_cyclic
(abs(a_j*b_k)+abs(a_k*b_j)). The engineering allowance is
2*gamma_(F+16)*S, gamma_m=m*u/(1-m*u), with (F+16)*u<1/4 and zero or normal
finite arithmetic. Edges/products/S/half scaling are rational; the final
allowance is rounded upward. This is a conservative engineering envelope,
not an exhaustive theorem. Declared zero constructions are exact.

Half, integer, complex, exceptional arithmetic and intake errors retain
exact source values/type/warnings/state through explicit primitive providers.
Fancy indexing must consume every face entry before the first three vertices
are used. No source array subclass, source rewrite or primitive replay is used
to observe anonymous intermediates: only actual edges/cross are captured.
"""
from collections import Counter
from fractions import Fraction
import math
import struct
import base64
import zlib

API = 'measure.mesh_surface_area'
STAGES = ('mesh_surface_area',)
UNASSESSED = ('arbitrary ndarray/index protocols', 'longdouble arithmetic',
              'native half/integer/complex arithmetic', 'native exceptional FP dispatch')


def cases():
    rows = []
    def add(id, pattern='triangle', dtype='float64', layout='C', outcome='return',
            route='native', exact_zero=False, count=1, buffer=None, policy=None):
        rows.append(dict(id=id, pattern=pattern, dtype=dtype, layout=layout,
                         outcome=outcome, route=route, exact_zero=exact_zero,
                         count=count, buffer=buffer, policy=policy or {}))
    add('right-f32', dtype='float32')
    add('right-f64')
    for pattern in ('translated', 'scaled', 'reversed-winding', 'duplicate', 'tetrahedron'):
        add(pattern, pattern, dtype='float32' if pattern=='scaled' else 'float64')
    for pattern in ('repeated-vertex', 'collinear', 'empty-faces'):
        add(pattern, pattern, exact_zero=True)
    add('near-cancellation', 'near-cancellation')
    for dtype in ('float16', 'int16', 'uint64', 'bool', 'complex128'):
        add('dtype-'+dtype, 'typed', dtype, outcome='error' if dtype=='bool' else 'return', route='provider')
    for layout in ('F', 'step', 'reverse', 'endian', 'unaligned', 'readonly', 'metadata', 'faces-step', 'negative-index'):
        add('layout-'+layout, dtype='float32' if layout in ('unaligned','metadata') else 'float64', layout=layout)
    for pattern in ('out-of-range', 'float-faces', 'list-verts', 'vector-width2', 'vector-width4',
                    'faces-rank1', 'faces-width2', 'unused-valid', 'unused-invalid'):
        add('intake-'+pattern, pattern, outcome='return' if pattern=='unused-valid' else 'error',
            route='native' if pattern=='unused-valid' else 'provider')
    for count in (7,8,9,128,129,8193):
        add('reduction-'+str(count), 'reduction', 'float32', count=count, buffer=16 if count==8193 else None)
    for pattern, outcome, policy in (
            ('nan-warn','return',{'invalid':'warn'}), ('inf-raise','error',{'invalid':'raise'}),
            ('overflow-warn','return',{'over':'warn','invalid':'warn'}), ('underflow-raise','error',{'under':'raise'})):
        add(pattern, pattern, 'float32', outcome=outcome, route='provider', policy=policy)
    assert len(rows)==44 and len({r['id'] for r in rows})==44
    return rows


def fixture(spec, np):
    pattern = spec['pattern']
    verts = np.array([[0,0,0],[3,0,0],[0,4,0],[1,1,2]], dtype=spec['dtype'])
    faces = np.array([[0,1,2]], dtype=np.intp)
    if pattern=='translated':verts += np.array([2**16,-2**15,2**14], dtype=verts.dtype)
    elif pattern=='scaled':verts *= np.float32(0.125)
    elif pattern=='reversed-winding':faces = faces[:, ::-1]
    elif pattern=='duplicate':faces = np.repeat(faces, 3, axis=0)
    elif pattern=='tetrahedron':faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.intp)
    elif pattern=='repeated-vertex':faces = np.array([[0,0,1],[2,2,2]], dtype=np.intp)
    elif pattern=='collinear':verts = np.array([[0,0,0],[1,2,3],[2,4,6]], dtype=verts.dtype)
    elif pattern=='empty-faces':faces = np.empty((0,3), dtype=np.intp)
    elif pattern=='near-cancellation':verts = np.array([[0,0,0],[1,1,1],[1+2**-40,1-2**-40,1]], dtype=verts.dtype)
    elif pattern=='typed':
        if spec['dtype']=='int16':verts = np.array([[0,0,0],[30000,1,0],[1,30000,0]], dtype='int16')
        elif spec['dtype']=='uint64':verts = np.array([[0,0,0],[2**63+1,1,0],[1,2**63+3,0]], dtype='uint64')
        elif spec['dtype']=='complex128':verts[1,1] = 1+2j
    elif pattern=='out-of-range':faces[0,2] = len(verts)
    elif pattern=='float-faces':faces = faces.astype('float64')
    elif pattern=='list-verts':verts = verts.tolist()
    elif pattern=='vector-width2':verts = verts[:, :2]
    elif pattern=='vector-width4':verts = np.pad(verts, ((0,0),(0,1)))
    elif pattern=='faces-rank1':faces = faces[0]
    elif pattern=='faces-width2':faces = faces[:, :2]
    elif pattern in ('unused-valid','unused-invalid'):
        faces = np.array([[0,1,2,3 if pattern=='unused-valid' else len(verts)]], dtype=np.intp)
    elif pattern=='reduction':
        verts = np.array([[.125,-.25,.5],[1.375,.75,-.125],[-.375,1.875,.25]], dtype=verts.dtype)
        faces = np.repeat(faces, spec['count'], axis=0)
    elif pattern=='nan-warn':verts[1,0] = np.nan
    elif pattern=='inf-raise':verts[0,0] = verts[1,0] = np.inf
    elif pattern=='overflow-warn':verts[0,0] = np.finfo('float32').max;verts[1,0] = -np.finfo('float32').max
    elif pattern=='underflow-raise':verts *= np.float32(1e-30)
    layout = spec['layout']
    if layout=='F':verts = np.asfortranarray(verts)
    elif layout=='step':
        owner = np.zeros((len(verts)*2,6), dtype=verts.dtype);owner[::2,::2] = verts;verts = owner[::2,::2]
    elif layout=='reverse':verts = verts[::-1];faces = len(verts)-1-faces
    elif layout=='endian':verts = verts.astype(verts.dtype.newbyteorder('>'))
    elif layout=='unaligned':
        owner = np.zeros(verts.nbytes+1, dtype=np.uint8)
        view = np.ndarray(verts.shape, dtype=verts.dtype, buffer=owner, offset=1);view[...] = verts;verts = view
    elif layout=='readonly':verts.flags.writeable = False
    elif layout=='metadata':verts = verts.astype(np.dtype('float32', metadata={'mesh':'frozen'}))
    elif layout=='faces-step':
        owner = np.zeros((2,6), dtype=np.intp);owner[::2,::2] = faces;faces = owner[::2,::2]
    elif layout=='negative-index':faces = faces-len(verts)
    return verts, {'faces':faces}


def allowance(spec, verts, faces):
    if spec['route']!='native':return None
    p = 24 if verts.dtype.itemsize==4 else 53
    u = Fraction(1, 2**p); m = len(faces)+16
    assert m*u < Fraction(1,4)
    scale = Fraction(0)
    for indices, count in Counter(tuple(map(int, row[:3])) for row in faces).items():
        xyz = [[Fraction(float(x)) for x in verts[i]] for i in indices]
        a = [x-y for x,y in zip(xyz[0],xyz[1])];b = [x-y for x,y in zip(xyz[0],xyz[2])]
        scale += count*sum(abs(a[j]*b[k])+abs(a[k]*b[j]) for j,k in ((1,2),(2,0),(0,1)))/2
    if spec['exact_zero']:return 0.0
    bound = 2*(m*u/(1-m*u))*scale
    value = float(bound)
    if Fraction(value)<bound:value = math.nextafter(value, math.inf)
    assert math.isfinite(value) and value>=0
    return value


def check_normal_domain(captured, np):
    """Check saved actual stages and rational square/sum magnitude, without replay."""
    total_upper = Fraction(0)
    for name in ('a','b','cross'):
        record = captured[name];dtype = np.dtype(record['metadata'][2]);info = np.finfo(dtype)
        data = zlib.decompress(base64.b64decode(record['raw_c_zlib_base64']))
        endian = '>' if dtype.byteorder=='>' else '<'
        words = [v[0] for v in struct.iter_unpack(endian+('f' if dtype.itemsize==4 else 'd'), data)]
        tiny, largest = Fraction(float(info.tiny)), Fraction(float(info.max))
        assert all(math.isfinite(v) and (v==0 or abs(Fraction(v))>=tiny) for v in words)
        if name=='cross':
            for offset in range(0,len(words),3):
                face = list(map(Fraction, words[offset:offset+3]));squares = [v*v for v in face]
                assert all(v==0 or v>=2*tiny for v in squares)
                assert sum(squares)<largest/2
                total_upper += sum(map(abs,face))
            assert total_upper<largest/2
