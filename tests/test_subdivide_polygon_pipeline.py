"""Actual full B-spline ownership against all unchanged original rows."""
import ast
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from skimagexx import _subdivide_polygon as implementation
if __package__:
    from .subdivide_polygon_contract import cases
    from .test_subdivide_polygon_public import observe
else:
    from subdivide_polygon_contract import cases
    from test_subdivide_polygon_public import observe

CASES=cases();MATRIX=Path(__file__).parents[1]/'coverage/contracts/subdivide-polygon.json'
NATIVE_IDS=frozenset([f'{kind}-degree{degree}'for kind in('open','closed')for degree in range(1,8)]+[
    'preserved-degree3','preserved-degree7','closed-preserve','two-point-open-degree1','two-point-closed',
    'all-identical','collinear','int32','uint8','float16','float32','bool','big-endian-float','int64-large',
    'fortran','reverse-rows','reverse-columns','step-rows','step-columns','readonly','dtype-metadata',
    'degree-float','degree-bool','degree-numpy','moderate-129'])


def test_subdivide_signature_literal_prefix_and_remaining_body():
    assert str(inspect.signature(implementation.subdivide_polygon))=='(coords, degree=2, preserve_ends=False)'
    assert inspect.isbuiltin(implementation._core.subdivide_polygon_apply)
    import skimage.measure._polygon as original
    current=ast.parse(inspect.getsource(implementation.subdivide_polygon)).body[0]
    source=ast.parse(inspect.getsource(original.subdivide_polygon)).body[0]
    assert ast.dump(ast.Module(body=current.body[:4],type_ignores=[]))==ast.dump(ast.Module(body=source.body[1:5],type_ignores=[]))
    start=next(i for i,n in enumerate(current.body)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Tuple))
    body=[n for n in current.body[start:]if not isinstance(n,ast.ImportFrom)]
    assert ast.dump(ast.Module(body=body,type_ignores=[]))==ast.dump(ast.Module(body=source.body[5:],type_ignores=[]))
    assert implementation._SUBDIVISION_MASKS==original._SUBDIVISION_MASKS


@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_subdivide_pipeline(spec,workers,monkeypatch,tmp_path):
    import skimage.measure as public
    import skimage.measure._polygon as original
    from scipy import signal
    expected=next(row for row in json.loads(MATRIX.read_bytes())['original_observations'][np.__version__]if row['id']==spec['id'])
    core=implementation._core;entry=core.subdivide_polygon_apply;call=implementation.subdivide_polygon
    get_workers=core.get_num_threads;previous=get_workers();prior_profile=sys.getprofile()
    calls=[];returns=[];accepted=[];snapshots=[];providers=[];forbidden=[]
    original_code=original.subdivide_polygon.__code__;convolve=signal.convolve2d
    def reject(*args,**kwargs):forbidden.append(True);raise AssertionError('subdivision forwarded whole original API')
    def provider(*args,**kwargs):
        providers.append('convolve2d')
        if spec['id']in NATIVE_IDS:raise AssertionError('accepted native B-spline used SciPy convolution')
        return convolve(*args,**kwargs)
    def snapshot():snapshots.append(get_workers());return snapshots[-1]
    def profile(frame,event,value):
        if value is entry:
            if event=='c_call':calls.append('subdivide_polygon_apply')
            elif event=='c_return':returns.append('subdivide_polygon_apply')
        if event=='call'and frame.f_code is original_code:reject()
        if event=='return'and frame.f_code is call.__code__:accepted.append(frame.f_locals.get('native_result')is not None)
    try:
        core.set_num_threads(workers)
        with monkeypatch.context()as patch:
            patch.setattr(public,'subdivide_polygon',reject);patch.setattr(original,'subdivide_polygon',reject)
            patch.setattr(signal,'convolve2d',provider);patch.setattr(core,'get_num_threads',snapshot)
            sys.setprofile(profile)
            try:actual=observe(call,spec,np)
            finally:sys.setprofile(prior_profile)
        assert actual==expected and not forbidden
        assert accepted==[spec['id']in NATIVE_IDS]
        assert calls==returns and len(calls)<=1 and snapshots==[workers]*len(calls)
        if spec['id']in NATIVE_IDS:assert calls==['subdivide_polygon_apply']and not providers
        assert get_workers()==workers
        (tmp_path/'candidate.json').write_text(json.dumps(dict(id=spec['id'],workers=workers,observation=actual,C_calls=calls,C_returns=returns,native_accepted=accepted,worker_snapshots=snapshots,SciPy_provider_calls=providers))+'\n')
    finally:sys.setprofile(prior_profile);core.set_num_threads(previous)


@pytest.mark.parametrize('workers',(0,1025))
def test_subdivide_worker_boundary(workers):
    with pytest.raises(ValueError,match='worker snapshot'):
        implementation._core.subdivide_polygon_apply(np.ones((9,2)),2,False,False,workers)


@pytest.mark.parametrize('reason',('float32','big-endian','columns','rank','unaligned','nonfinite','empty','short','degree0','degree8'))
def test_subdivide_private_refuses_without_input_writes(reason):
    coords=np.arange(18.,dtype=np.float64).reshape(9,2);degree=2
    if reason=='float32':coords=coords.astype(np.float32)
    elif reason=='big-endian':coords=coords.astype('>f8')
    elif reason=='columns':coords=np.ones((9,3))
    elif reason=='rank':coords=np.ones(9)
    elif reason=='unaligned':coords=np.ndarray((9,2),dtype=np.float64,buffer=bytearray(145),offset=1)
    elif reason=='nonfinite':coords[3,1]=np.nan
    elif reason=='empty':coords=np.empty((0,2))
    elif reason=='short':coords=np.ones((1,2))
    elif reason=='degree0':degree=0
    elif reason=='degree8':degree=8
    before=coords.tobytes();policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    assert implementation._core.subdivide_polygon_apply(coords,degree,False,False,1)is None
    assert coords.tobytes()==before and np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer


def test_subdivide_forged_storage_rejected_before_read():
    coords=np.lib.stride_tricks.as_strided(np.ones(2),shape=(9,2),strides=(16,8))
    with pytest.raises(ValueError,match='FFT'):
        implementation._core.subdivide_polygon_apply(coords,2,False,False,1)


@pytest.mark.parametrize('workers',(1,4))
def test_subdivide_no129_cap_owned_linear_spline_and_fresh_output(workers,monkeypatch):
    from scipy import signal
    core=implementation._core;previous=core.get_num_threads();coords=np.column_stack((np.arange(1025.),np.zeros(1025)));before=coords.tobytes()
    expected=np.zeros((2044,2));expected[1:-1:2,0]=np.arange(1021.)+1.5;expected[2:-1:2,0]=np.arange(1021.)+2;expected[-1,0]=1024.
    entry=core.subdivide_polygon_apply;seen=[];prior=sys.getprofile()
    def profile(frame,event,value):
        if event=='c_return'and value is entry:seen.append(value.__name__)
    def reject(*args,**kwargs):raise AssertionError('native long spline used SciPy')
    try:
        core.set_num_threads(workers);monkeypatch.setattr(signal,'convolve2d',reject);sys.setprofile(profile)
        try:actual=implementation.subdivide_polygon(coords,7,True)
        finally:sys.setprofile(prior)
        assert actual.shape==expected.shape and actual.dtype==np.dtype('float64')and actual.tobytes()==expected.tobytes()
        assert actual.flags.c_contiguous and actual.flags.owndata and actual.flags.writeable and not np.shares_memory(actual,coords)
        assert coords.tobytes()==before and seen==['subdivide_polygon_apply']and core.get_num_threads()==workers
    finally:sys.setprofile(prior);core.set_num_threads(previous)
