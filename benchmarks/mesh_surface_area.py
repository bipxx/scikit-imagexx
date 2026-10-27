"""Fixed six-case mesh-area comparison; generation/checks are never timed."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

HEIGHT_FORMULA = 'z=.125*(x*x+2*y*y+x*y)'


def case_specs():
    return [dict(case_id=f'cells{cells}:{dtype}', api='measure.mesh_surface_area',
                 cells=cells, dtype=dtype, vertex_count=(cells+1)**2, face_count=2*cells*cells)
            for cells in (64,128,256) for dtype in ('float32','float64')]


def mesh_inputs(spec, np):
    # Powers-of-two cell sizes make this quadratic heightfield dyadic. No
    # marching-cubes extraction, trigonometric residuals, or dtype relabeling.
    cells=spec['cells']
    axis=np.arange(cells+1,dtype=np.float64)/cells
    x,y=np.meshgrid(axis,axis,indexing='ij')
    z=.125*(x*x+2*y*y+x*y)
    verts=np.column_stack((x.ravel(),y.ravel(),z.ravel())).astype(spec['dtype'])
    indices=np.arange((cells+1)**2,dtype=np.intp).reshape(cells+1,cells+1)
    a,b,c,d=(indices[:-1,:-1].ravel(),indices[1:,:-1].ravel(),
             indices[1:,1:].ravel(),indices[:-1,1:].ravel())
    faces=np.empty((2*cells*cells,3),dtype=np.intp)
    faces[0::2]=np.column_stack((a,b,c))
    faces[1::2]=np.column_stack((a,c,d))
    return verts,faces


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1)
    parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--output',type=Path,default=Path('build/ci-mesh-surface-area.json'))
    args=parser.parse_args()
    from run import measure
    from provenance import source_fingerprint
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx.coverage import mesh_default_error

    assert skimage.__version__=='0.26.0' and xx.get_num_threads()==1
    limits={name:os.environ[name]for name in
            ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(limits.values())=={'1'}
    xx.set_fallback_enabled(False);xx.set_num_threads(args.threads)
    root=Path(__file__).resolve().parents[1];source_before=source_fingerprint(root)
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest()
    policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    rows=[]
    for spec in case_specs():
        verts,faces=mesh_inputs(spec,np)
        assert verts.shape==(spec['vertex_count'],3) and verts.dtype.name==spec['dtype']
        assert faces.shape==(spec['face_count'],3) and faces.dtype==np.dtype(np.intp)
        call_args=(verts,faces);arrays=(verts,faces)
        before=[hashlib.sha256(a.tobytes()).hexdigest()for a in arrays]
        reference,native=skimage.measure.mesh_surface_area,xx.measure.mesh_surface_area
        expected=reference(*call_args)
        returns=[];accepted=[];prior=sys.getprofile()
        def profile(frame,event,arg):
            if event=='c_return'and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)=='mesh_surface_area':
                returns.append(arg.__name__)
            if event=='return'and frame.f_code is native.__code__:
                accepted.append(frame.f_locals.get('total')is not None)
        sys.setprofile(profile)
        try:actual=native(*call_args)
        finally:sys.setprofile(prior)
        assert returns==['mesh_surface_area'] and accepted==[True]
        assert isinstance(actual,np.generic) and type(actual)is type(expected)
        error=mesh_default_error(call_args,actual,expected)
        assert xx.get_num_threads()==1
        assert before==[hashlib.sha256(a.tobytes()).hexdigest()for a in arrays]
        timing=measure(lambda:reference(*call_args),lambda:native(*call_args),args.repeat,15)
        assert xx.get_num_threads()==1
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        assert before==[hashlib.sha256(a.tobytes()).hexdigest()for a in arrays]
        rows.append(dict(**spec,backend='native',threads=1,
            vertex_shape=list(verts.shape),face_shape=list(faces.shape),
            vertex_dtype_str=verts.dtype.str,face_dtype_str=faces.dtype.str,input_sha256=before,
            result_type=[type(actual).__module__,type(actual).__name__],
            reference_result_type=[type(expected).__module__,type(expected).__name__],
            output_dtype_str=actual.dtype.str,reference_dtype_str=expected.dtype.str,
            result=float(actual),reference_result=float(expected),
            output_raw_hex=actual.tobytes().hex(),reference_raw_hex=expected.tobytes().hex(),
            actual_C_returns=returns,native_accepted=True,max_abs_error=error,**timing))
        print(f"{spec['case_id']}: {timing['speedup']:.3f}x",flush=True)
    assert len(rows)==6 and all([len(s)for s in row['samples_ms']]==[7,7]for row in rows)
    source_after=source_fingerprint(root);assert source_after==source_before
    assert hashlib.sha256(core.read_bytes()).hexdigest()==core_sha
    scope=('Six actual float32/float64 meshes use unit-grid cells64/128/256 and '
           +HEIGHT_FORMULA+'. Vertices and two triangles/cell are generated outside timing; '
           'no marching_cubes or geometry extraction is timed. Every public call includes full '
           'fancy indexing (including all face entries), C-buffer preparation, owned cross/norm/reduction, '
           'typed NumPy scalar boxing and literal /2.0. The frozen exact-rational 2*gamma_(F+16)*S '
           'check runs before timing on the full (verts,faces) context. Finite normal native work '
           'must complete the actual C stage; unsupported/exceptional primitive frontiers are not '
           'timed in this cohort. Output dtype records preserve NumPy1/2 scalar-promotion differences. '
           'This first original/current comparison is neither prior-C++ gain nor full-API qualification.')
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),cpu=platform.processor(),python=sys.version,
        reference=skimage.__version__,numpy=np.__version__,scipy=scipy.__version__,
        **source_after,source_before=source_before,core_path=str(core),core_sha256=core_sha,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        height_formula=HEIGHT_FORMULA,height_formula_sha256=hashlib.sha256(HEIGHT_FORMULA.encode()).hexdigest(),
        numpy_error_policy=policy,numpy_buffer_elements=buffer,
        threads=1,repeat=7,raw_samples=84,reference_pool_limits=limits,
        fallback_enabled=False,avx2=xx.has_avx2(),command=sys.argv,
        selected_apis=['measure.mesh_surface_area'],provider_frontier=scope),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Mesh surface area: one-thread comparison','',scope,'',
           '| Case | Faces | Original ms | Current ms | Original/current | Max error |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['case_id']} | {row['face_count']} | {row['reference_ms']:.6f} | {row['cpp_ms']:.6f} | {row['speedup']:.9f}x | {row['max_abs_error']} |")
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
