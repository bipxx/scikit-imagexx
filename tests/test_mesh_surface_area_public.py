"""Genuine-original-first mesh contracts; candidate registration is initially RED."""
import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .mesh_surface_area_cases import API, STAGES, UNASSESSED, cases, fixture, allowance, check_normal_domain
    from .test_moments_public import original_runtime, metadata
    from .test_moments_derived_public import observe, result_record
else:
    from mesh_surface_area_cases import API, STAGES, UNASSESSED, cases, fixture, allowance, check_normal_domain
    from test_moments_public import original_runtime, metadata
    from test_moments_derived_public import observe, result_record

CASES = cases()
SOURCE_SHA256 = 'f482cdb5c9996c1a465a1fc84ffea9f88eec012200b61ca67e8ac7eab964f7fa'
CROSS_PINS = {'1.24.4':'4a4542d7758f8aa769e4fa7ff6228e1063788700332298de3da84608004df122',
              '2.5.4':'feb150554b4879d4df7bd0a4ab7b1a6b818de7ba994ed0c73583fee177533417'}


def runtime():
    np, _, record = original_runtime()
    original = importlib.import_module('skimage.measure._marching_cubes_lewiner')
    path = Path(original.__file__);raw = path.read_bytes()
    assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA256
    distribution = importlib.metadata.distribution('scikit-image')
    relative = path.relative_to(Path(distribution.locate_file(''))).as_posix()
    entry = next(row for row in csv.reader(distribution.read_text('RECORD').splitlines()) if row[0]==relative)
    algorithm, encoded = entry[1].split('=',1)
    assert algorithm=='sha256' and len(raw)==int(entry[2])
    assert base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')==encoded
    cross = inspect.unwrap(np.cross);cross_path = Path(inspect.getfile(cross));cross_raw = cross_path.read_bytes()
    assert hashlib.sha256(cross_raw.replace(b'\r\n',b'\n')).hexdigest()==CROSS_PINS[np.__version__]
    record.update(mesh_source=dict(path=str(path),raw_sha256=hashlib.sha256(raw).hexdigest(),
                                  canonical_lf_sha256=SOURCE_SHA256,record=entry),
                  numpy_cross_source=dict(path=str(cross_path),raw_sha256=hashlib.sha256(cross_raw).hexdigest(),
                                          canonical_lf_sha256=CROSS_PINS[np.__version__]),
                  default_buffer=np.getbufsize(),intermediates='Actual edges and np.cross return only; anonymous face norms and final sum are not captured or replayed.')
    return np, original, record


def encode(value, np):
    return result_record(value) if isinstance(value,(np.ndarray,np.generic)) else value


def reference_case(spec, np, original):
    verts, options = fixture(spec,np)
    before = dict(verts=encode(verts,np),faces=encode(options['faces'],np))
    captured = {};prior = sys.getprofile();buffer = np.getbufsize()
    cross_code = inspect.unwrap(np.cross).__code__
    def capture(frame,event,arg):
        if event=='return' and frame.f_code is cross_code:
            # Encode immediately; retaining the actual temporary would change
            # NumPy's reference-count-based temporary-elision decisions.
            captured['cross'] = encode(arg,np)
        if event=='return' and frame.f_code is original.mesh_surface_area.__code__:
            captured.update(a=encode(frame.f_locals.get('a'),np),b=encode(frame.f_locals.get('b'),np))
    try:
        if spec['buffer'] is not None:np.setbufsize(spec['buffer'])
        selected_buffer = np.getbufsize()
        sys.setprofile(capture)
        try:observed = observe(original.mesh_surface_area,verts,options,spec['policy'],np)
        finally:sys.setprofile(prior)
    finally:np.setbufsize(buffer)
    assert np.getbufsize()==buffer
    return verts,options,captured,observed,before,selected_buffer


def validate(spec, verts, options, captured, observed, np):
    result,error,messages,checks = observed
    assert all(checks.values()),spec['id']
    assert (error is None)==(spec['outcome']=='return'),(spec['id'],error)
    if error is not None:return None
    assert isinstance(result,np.generic) and result.shape==()
    if spec['route']!='native':return None
    assert result.dtype.kind=='f' and np.isfinite(result) and result>=0
    assert not messages
    check_normal_domain(captured,np)
    bound = allowance(spec,verts,options['faces'])
    if spec['exact_zero']:assert result==0 and not np.signbit(result)
    return bound


def compare(actual, expected, bound, spec, np):
    assert type(actual)is type(expected) and metadata(actual)==metadata(expected)
    if spec['route']!='native' or spec['exact_zero']:
        assert result_record(actual)==result_record(expected)
    else:
        assert np.isfinite(actual) and abs(float(actual)-float(expected))<=bound,spec['id']


@pytest.fixture(scope='module')
def reference_runtime():return runtime()
@pytest.fixture(scope='module')
def worker_results():return {}


def test_mesh_surface_area_signature(reference_runtime):
    _,original,_ = reference_runtime
    import skimagexx as xx
    from skimagexx.coverage import native_functions
    assert API in native_functions(),'mesh_surface_area remains upstream-only'
    assert inspect.signature(xx.measure.mesh_surface_area)==inspect.signature(original.mesh_surface_area)


@pytest.mark.parametrize('spec',CASES,ids=[r['id']for r in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_mesh_surface_area_owned(reference_runtime,spec,workers,worker_results,monkeypatch,tmp_path):
    np,original,_ = reference_runtime
    verts,options,captured,observed,before,selected_buffer = reference_case(spec,np,original)
    expected,error,messages,checks = observed
    original_row = dict(id=spec['id'],input_before=before,input_after=dict(verts=encode(verts,np),faces=encode(options['faces'],np)),
                        actual_intermediates=captured,buffer=selected_buffer,result=encode(expected,np),error=error,warnings=messages,checks=checks)
    (tmp_path/'original.json').write_text(json.dumps(original_row)+'\n')
    bound = validate(spec,verts,options,captured,observed,np)
    expected_input = original_row['input_after'];verts,options = fixture(spec,np)
    import skimage.measure as upstream
    import skimagexx as xx
    from skimagexx import _compat
    from skimagexx.coverage import native_functions
    assert API in native_functions() and xx.get_num_threads()==1
    previous = xx.set_fallback_enabled(False);prior = sys.getprofile();buffer = np.getbufsize()
    returns = [];forbidden = []
    def deny(*args,**kwargs):forbidden.append(True);raise AssertionError('upstream mesh/cross forwarding')
    def profile(frame,event,arg):
        if event=='c_return' and getattr(arg,'__module__',None)==xx._core.__name__ and getattr(arg,'__name__',None)in STAGES:
            returns.append(arg.__name__)
    try:
        xx.set_num_threads(workers)
        if spec['buffer'] is not None:np.setbufsize(spec['buffer'])
        with monkeypatch.context()as patch:
            patch.setattr(upstream,'mesh_surface_area',deny);patch.setattr(original,'mesh_surface_area',deny)
            patch.setattr(_compat,'fallback',deny)
            if spec['route']=='native':patch.setattr(np,'cross',deny)
            sys.setprofile(profile)
            try:actual,actual_error,actual_messages,actual_checks = observe(xx.measure.mesh_surface_area,verts,options,spec['policy'],np)
            finally:sys.setprofile(prior)
        raw = dict(id=spec['id'],workers=workers,input_after=dict(verts=encode(verts,np),faces=encode(options['faces'],np)),
                   result=encode(actual,np),error=actual_error,warnings=actual_messages,checks=actual_checks,
                   actual_C_returns=returns,forbidden=forbidden,allowance=bound,buffer=np.getbufsize(),
                   numeric_credit=int(spec['route']=='native' and len(options['faces'])>0 and bool(returns)))
        (tmp_path/'candidate.json').write_text(json.dumps(raw)+'\n')
        assert not forbidden and actual_error==error and actual_messages==messages and actual_checks==checks
        assert raw['input_after']==expected_input and np.getbufsize()==selected_buffer
        if spec['route']=='native':assert returns==list(STAGES)
        else:assert len(returns)<=1
        if error is None:compare(actual,expected,bound,spec,np)
        state = (encode(actual,np),actual_error,actual_messages)
        assert worker_results.setdefault(spec['id'],state)==state
        assert xx.get_num_threads()==workers and _compat._fallback_enabled is False
    finally:
        sys.setprofile(prior);np.setbufsize(buffer);xx.set_num_threads(1);xx.set_fallback_enabled(previous)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--reference-only',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();assert args.reference_only and sys.flags.optimize==0
    assert not any(n=='skimagexx' or n.startswith('skimagexx.')for n in sys.modules)
    args.output.mkdir();np,original,record = runtime()
    (args.output/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
    rows = []
    with(args.output/'journal.jsonl').open('x')as journal:
        for spec in CASES:
            verts,options,captured,observed,before,selected_buffer = reference_case(spec,np,original)
            result,error,messages,checks = observed
            raw = dict(id=spec['id'],spec=spec,input_before=before,input_after=dict(verts=encode(verts,np),faces=encode(options['faces'],np)),
                       actual_intermediates=captured,buffer=selected_buffer,result=encode(result,np),error=error,warnings=messages,checks=checks)
            journal.write(json.dumps(raw)+'\n');journal.flush()
            bound = validate(spec,verts,options,captured,observed,np)
            if error is None:compare(result,result,bound,spec,np)
            rows.append(dict(id=spec['id'],error=error,warnings=messages,allowance=bound))
    assert not any(n=='skimagexx' or n.startswith('skimagexx.')for n in sys.modules)
    (args.output/'summary.json').write_text(json.dumps(dict(cases=len(rows),returns=sum(r['error']is None for r in rows),
        errors=sum(r['error']is not None for r in rows),native_calls=0,anonymous_intermediate_replays=0,rows=rows),indent=2)+'\n')
    print('original-only',len(rows),'PASS',flush=True)


if __name__=='__main__':main()
