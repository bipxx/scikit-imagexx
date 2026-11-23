"""One stock and one installed-candidate Laplacian generator per failed recipe and mode."""

import argparse
import ast
from base64 import urlsafe_b64encode
import csv
import ctypes
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module, metadata
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import warnings


HERE = Path(__file__).resolve().parent
SCOPE = 'current301-minimum-laplacian-live-v1'
PINS = '72e8ffabc97cdfbd450865af7f624fd3681787efddc78fce304dea123b2b7b6d'
POOL = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','OMP_THREAD_LIMIT')
FILES = ('pins.json','range_helper.py','prior_diagnose.py','extract_candidate.py','requirements-runtime.txt','report-json.yml')


def sha(raw):
    return sha256(raw).hexdigest()


def digest(path):
    return sha(Path(path).read_bytes())


def write(path, row):
    with path.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(row,stream,indent=2,allow_nan=False)
        stream.write('\n')


def preview():
    path = HERE/'pins.json'
    assert digest(path) == PINS
    pins = json.loads(path.read_bytes())
    assert len(pins['cases']) == 3 and pins['member_count'] == 272
    assert sum(value['yield_count'] for value in pins['cases'].values()) == 6
    assert not {'numpy','scipy','skimage','skimagexx','pytest'} & sys.modules.keys()
    return pins


def authorize(pins):
    assert sys.platform == 'linux' and sys.version_info[:3] == (3,11,17)
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize
    assert os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_RUN_ATTEMPT') == '1'
    assert os.environ.get('GITHUB_REPOSITORY') == 'bipxx/scikit-imagexx'
    assert all(os.environ.get(name) == '1' for name in POOL)
    go = json.loads(os.environ['ROOT_AUTHORIZATION_JSON'])
    assert (go['authorized_by'],go['scope'],go['carrier']) == ('/root',SCOPE,os.environ['GITHUB_SHA'])
    assert (go['source'],go['tree'],go['artifact']) == (pins['source'],pins['tree'],pins['artifact'])
    assert go['cases'] == list(pins['cases']) and go['modes'] == ['normal','scalar']
    assert go['launcher_sha256'] == digest(Path(__file__))
    assert go['files'] == {name:digest(HERE/name) for name in FILES}
    deadline = datetime.fromisoformat(go['deadline_utc'])
    assert deadline.tzinfo is not None and deadline.utcoffset().total_seconds() == 0
    assert datetime.now(timezone.utc) < deadline
    return deadline


def prove_source(source, pins):
    command = ['git','--no-optional-locks','-C',str(source)]
    def git(*args):
        return subprocess.check_output([*command,*args])
    assert git('rev-parse','HEAD','HEAD^{tree}','HEAD^').decode().splitlines() == [
        pins['source'],pins['tree'],pins['parent']]
    assert not git('status','--porcelain=v1','--untracked-files=all')
    for name, pin in pins['source_paths'].items():
        raw = (source/name).read_bytes()
        assert sha(git('show','HEAD:'+name)) == pin['git_raw_sha256'], name
        assert sha(raw.replace(b'\r\n',b'\n')) == pin['normalized_lf_sha256'], name


def prove_product(args, pins):
    extraction = json.loads(args.extraction.read_bytes())
    assert (extraction['source'],extraction['artifact'],extraction['job'],extraction['members']) == (
        pins['source'],pins['artifact'],pins['job'],pins['member_count'])
    assert not extraction['outer_zip_fully_verified'] and not extraction['archives_retained']
    assert extraction['wheel_sha256'] == pins['candidate_wheel']['sha256']
    assert extraction['core_member'] == pins['candidate_core_member']
    assert digest(args.candidate/extraction['core_member']) == extraction['core_sha256'] == pins['candidate_core_sha256']
    assert digest(args.candidate/extraction['record_path']) == extraction['record_sha256']
    for name, pin in pins['source_paths'].items():
        if name.startswith('python/'):
            assert digest(args.candidate/name.removeprefix('python/')) == pin['git_raw_sha256'], name
    return extraction


def prove_stock(args, pins):
    distributions = {d.metadata['Name'].lower().replace('_','-'):d for d in metadata.distributions(path=[str(args.stock)])}
    assert {name:distributions[name].version for name in pins['stock_versions']} == pins['stock_versions']
    records = {}
    for member, expected in pins['stock_members'].items():
        folder = member.split('/')[0]
        name = 'scikit-image' if folder == 'skimage' else folder
        dist = distributions[name]
        if name not in records:
            path, = (Path(dist.locate_file(f)) for f in dist.files if f.name == 'RECORD')
            with path.open(encoding='utf8',newline='') as stream:
                records[name] = (digest(path),{row[0]:row[1:] for row in csv.reader(stream)})
        path = args.stock/member
        assert Path(dist.locate_file(member)) == path
        pin = expected['sha256']
        encoded = urlsafe_b64encode(bytes.fromhex(pin)).decode().rstrip('=')
        assert records[name][1][member] == ['sha256='+encoded,str(expected['bytes'])]
        assert path.stat().st_size == expected['bytes'] and digest(path) == pin
    return {name:digest for name,(digest,_) in records.items()}


def compare_words(np, left, right):
    assert left.shape == right.shape and left.dtype == right.dtype == np.dtype('float64')
    assert left.size <= 63*89*3 and np.isfinite(left).all() and np.isfinite(right).all()
    a, b = (array.view(np.uint64).ravel() for array in (left,right))
    idx = np.flatnonzero(a != b)
    first = [{'index':[int(k) for k in np.unravel_index(i,left.shape)],
              'stock':f'{int(a[i]):016x}','candidate':f'{int(b[i]):016x}'} for i in idx[:8]]
    # Monotonic representation across signed residual values.
    sign = np.uint64(1 << 63)
    ordered = [np.where(value & sign,~value,value ^ sign) for value in (a,b)]
    return dict(different_words=len(idx), first=first, max_absolute=float(np.abs(left-right).max()),
                max_ulp=int((np.maximum(*ordered)-np.minimum(*ordered)).max()))


def collect(args, pins, deadline):
    assert len(os.sched_getaffinity(0)) == 1
    assert os.environ.get('SKIMAGEXX_DISABLE_SIMD') == (None if args.mode == 'normal' else '1')
    assert not any(key in os.environ for key in ('GH_TOKEN','PYTHONPATH','PYTHONHOME','NPY_DISABLE_CPU_FEATURES','LD_PRELOAD'))
    records = prove_stock(args,pins)
    product = prove_product(args,pins)
    sys.path[:0] = [str(args.candidate),str(args.stock)]
    import numpy as np
    from numpy.core import _multiarray_umath
    from scipy.ndimage import _nd_image, _filters
    import skimage.transform as stock_transform
    from skimage.transform import pyramids
    import skimagexx as xx
    from skimagexx import _core, _pyramid_generators, _pyramid_spatial, filters as candidate_filters
    from skimagexx import transform as candidate_transform
    modules = ('numpy.core._multiarray_umath','scipy.ndimage._nd_image','scipy.ndimage._filters',
               'scipy.ndimage._interpolation','scipy.ndimage._ni_support','skimage.transform.pyramids',
               'skimage.transform._warps','skimage._shared.filters','skimage.util.dtype')
    actual_stock = {Path(import_module(name).__file__).relative_to(args.stock).as_posix() for name in modules}
    assert actual_stock == pins['stock_members'].keys()
    assert Path(_core.__file__) == args.candidate/product['core_member']
    assert Path(_pyramid_generators.__file__) == args.candidate/'skimagexx/_pyramid_generators.py'
    stock = stock_transform.pyramid_laplacian
    candidate = candidate_transform.pyramid_laplacian
    assert stock is pyramids.pyramid_laplacian and candidate is _pyramid_generators.pyramid_laplacian
    assert inspect.isgeneratorfunction(stock) and inspect.isgeneratorfunction(candidate)
    assert inspect.signature(stock) == inspect.signature(candidate)
    assert xx.get_num_threads() == 1
    xx.set_fallback_enabled(False)
    assert not xx._compat._fallback_enabled
    candidate_entries = (_core.gaussian_nd,_core.resize_zoom_typed)
    assert all(inspect.isbuiltin(item) and item.__module__ == 'skimagexx._core' for item in candidate_entries)
    stock_entries = (_nd_image.correlate1d,_nd_image.zoom_shift)
    assert all(inspect.isbuiltin(item) for item in stock_entries)
    shared = runpy.run_path(str(args.source/'tests/pyramid_generator_cases.py'))
    spatial = shared['SPATIAL']
    source_test = args.source/'tests/test_pyramid_generators_public.py'
    parsed = ast.parse(source_test.read_bytes())
    node, = [node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name == 'snapshot']
    namespace = {'np':np,'array_state':spatial['array_state']}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source_test),'exec'),namespace)
    snapshot = namespace['snapshot']
    before = ctypes.CDLL('libm.so.6').fegetround
    before.restype,before.argtypes = ctypes.c_int,[]
    def fenv():
        return dict(fegetround=before(),numpy_geterr=np.geterr(),numpy_bufsize=np.getbufsize())
    cpu = runpy.run_path(str(HERE/'prior_diagnose.py'))['cpu_state'](_multiarray_umath)

    def run(api, recipe, ref, route):
        target = candidate_entries if route == 'candidate' else stock_entries
        stage_functions = ((_pyramid_spatial._smooth,_pyramid_generators.resize) if route == 'candidate'
                           else (pyramids._smooth,pyramids.resize))
        stages = {inspect.unwrap(function).__code__:label
                  for function,label in zip(stage_functions,('smooth','resize'),strict=True)}
        image = shared['make_image'](np,recipe)
        assert snapshot(image) == ref['input']
        yields, steps, stage_bytes = [], [], []
        compiled, coefficients = [], []

        def watch(frame,event,arg):
            if route == 'candidate' and event == 'call' and frame.f_code is stock.__code__:
                raise AssertionError('candidate entered stock public generator')
            if event == 'call' and stages.get(frame.f_code) == 'smooth':
                stage_bytes.append((len(steps),'smooth-input',frame.f_locals['image'].copy()))
            if event == 'c_return' and arg in target:
                compiled.append(arg.__name__)
            if event == 'return' and isinstance(arg,np.ndarray):
                if frame.f_code in stages:
                    stage_bytes.append((len(steps),stages[frame.f_code],arg.copy()))
                if frame.f_code is _filters._gaussian_kernel1d.__code__:
                    coefficients.append(dict(step=len(steps),sigma=float(frame.f_locals['sigma']).hex(),
                                             radius=int(frame.f_locals['radius']),words=arg.tobytes().hex()))
                if frame.f_code is candidate_filters._gaussian_nd.__code__:
                    coefficients.append(dict(step=len(steps), axes=frame.f_locals['axes'],
                                             kernels=[k.tobytes().hex() for k in frame.f_locals['kernels']]))

        gen = api(image,**recipe['options'])
        assert inspect.isgenerator(gen) and inspect.getgeneratorstate(gen) == 'GEN_CREATED' and iter(gen) is gen
        fenv_before = fenv()
        prior = sys.getprofile()
        marker = ValueError('probe marker')
        with warnings.catch_warnings(record=True) as notes:
            warnings.simplefilter('always')
            try:
                sys.setprofile(watch)
                for expected in ref['steps']:
                    compiled.clear()
                    notes.clear()
                    action = expected['action']
                    step = dict(action=action,state_before=inspect.getgeneratorstate(gen))
                    try:
                        if action == 'throw-marker':
                            out = gen.throw(marker)
                        else:
                            assert action in ('next','next-after-terminal','next-after-terminal-again')
                            out = next(gen)
                    except StopIteration as error:
                        step.update(result='stop',value=error.value)
                    except ValueError as error:
                        step.update(result='error',error=dict(type=type(error).__qualname__,message=str(error),
                                                              is_throw_marker=error is marker))
                    else:
                        assert type(out) is np.ndarray
                        step.update(result='yield',array=snapshot(out),identical_input=out is image,
                                    shares_input=bool(np.shares_memory(out,image)),
                                    identical_prior=[out is p for p in yields],
                                    shares_prior=[bool(np.shares_memory(out,p)) for p in yields])
                        if not yields:
                            names = ('image','smoothed_image') if route == 'stock' else ('current','smoothed')
                            locals_ = gen.gi_frame.f_locals
                            assert [(iteration,label) for iteration,label,_ in stage_bytes] == [
                                (0,'smooth-input'),(0,'smooth')]
                            assert [sha(locals_[key].tobytes()) for key in names] == [
                                sha(array.tobytes()) for _,_,array in stage_bytes]
                        yields.append(out)
                    step.update(state_after=inspect.getgeneratorstate(gen),input_after=snapshot(image),
                                warnings=[[n.category.__module__+'.'+n.category.__qualname__,str(n.message)] for n in notes],
                                compiled=list(compiled))
                    steps.append(step)
            finally:
                sys.setprofile(prior)
        assert snapshot(image) == ref['final_input']
        return dict(steps=steps,fenv_before=fenv_before,fenv_after=fenv(),coefficients=coefficients),yields,stage_bytes

    output = dict(scope=SCOPE,source=pins['source'],tree=pins['tree'],mode=args.mode,artifact=pins['artifact'],
                  python=sys.version,python_executable=sys.executable,python_executable_sha256=digest(sys.executable),
                  stock_members=pins['stock_members'],stock_record_sha256=records,core=product,cpu=cpu,cases={})
    for recipe, ref in shared['rows']():
        name = ref['case']
        if name not in pins['cases']:
            continue
        assert ref['native'] and ref['input']['sha256'] == pins['cases'][name]['frozen_input']
        assert [step['action'] for step in ref['steps']] == pins['cases'][name]['actions']
        old, old_images, old_stage = run(stock,recipe,ref,'stock')
        new, new_images, new_stage = run(candidate,recipe,ref,'candidate')
        def non_numeric(step):
            result = dict(step)
            result.pop('compiled')
            if 'array' in result:
                result['array'] = {k:v for k,v in step['array'].items() if k not in ('sha256','hex')}
            return result
        count = pins['cases'][name]['yield_count']
        assert len(old_images) == len(new_images) == count
        assert [(idx,label) for idx,label,_ in old_stage] == [(idx,label) for idx,label,_ in new_stage]
        assert len(old_stage) == count*3-1 and old['coefficients'] and new['coefficients']
        assert all('correlate1d' in step['compiled'] for step in old['steps'] if step['result'] == 'yield')
        first_stock,first_candidate = (value['steps'][0]['array']['sha256'] for value in (old,new))
        known_first = pins['cases'][name]
        output['cases'][name] = dict(frozen=pins['cases'][name],stock=old,candidate=new,
            first=dict(stock_and_candidate=first_stock==first_candidate,stock_matches_frozen=first_stock==known_first['frozen_first'],
                       candidate_matches_frozen=first_candidate==known_first['frozen_first'],
                       candidate_reproduces_failed_mode=first_candidate==known_first[args.mode+'_failed_first']),
            state_layout_warnings_equal=(list(map(non_numeric,old['steps'])) == list(map(non_numeric,new['steps']))
                                         == list(map(non_numeric,ref['steps']))),
            candidate_compiled_equal=[step['compiled'] for step in new['steps']] == [step['compiled'] for step in ref['steps']],
            residuals=[compare_words(np,x,y) for x,y in zip(old_images,new_images,strict=True)],
            stages=[dict(step=i,stage=stage,stock=sha(x.tobytes()),candidate=sha(y.tobytes()),**compare_words(np,x,y))
                    for (i,stage,x),(_,_,y) in zip(old_stage,new_stage,strict=True)])
        assert datetime.now(timezone.utc) < deadline
    assert list(output['cases']) == list(pins['cases'])
    write(args.output/f'{args.mode}-observed.json',output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry-only',action='store_true')
    parser.add_argument('--bind-only',action='store_true')
    parser.add_argument('--mode',choices=('normal','scalar'))
    for name in ('source','stock','candidate','output','extraction'):
        parser.add_argument('--'+name,type=lambda value:Path(value).resolve(strict=True))
    args = parser.parse_args()
    pins = preview()
    if args.entry_only:
        print(json.dumps(dict(source=pins['source'],tree=pins['tree'],artifact=pins['artifact'],cases=list(pins['cases']),
                              scientific_imports=0)))
        return 0
    if 'ROOT_AUTHORIZATION_JSON' not in os.environ:
        print('REFUSED_NO_EXACT_ROOT_DIAGNOSTIC_AUTHORITY',file=sys.stderr)
        return 2
    deadline = authorize(pins)
    prove_source(args.source,pins)
    if args.bind_only:
        assert not args.mode
        return 0
    assert args.mode and args.stock and args.candidate and args.output and args.extraction
    collect(args,pins,deadline)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
