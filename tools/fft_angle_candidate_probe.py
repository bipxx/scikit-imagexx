"""One private FFT angle candidate on a NEW HOST; never whole-product acceptance."""
import argparse, contextlib, datetime, hashlib, importlib.util, inspect, json, os, platform
from pathlib import Path
import subprocess, sys, traceback, zipfile, io

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = '08c83244c3cbc9f93b44a051b3a56452b4d5213e'
DEADLINE = '2026-10-28T13:18:17.547938+00:00'
REF_SHA = '665976258003c03c219ea4e221a096dcd19b5ef4a360be78fd7166b3419c04f1'
ENTRIES = ('fft_r2c', 'fft_c2r', 'fft_c2c')
H = lambda raw: hashlib.sha256(raw).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


p = load('angle_reviewed_lpi_probe', ROOT / 'tools/lpi_inverse_stage_probe.py')


def deadline():
    p.require(datetime.datetime.now(datetime.timezone.utc) <= datetime.datetime.fromisoformat(DEADLINE), 'single candidate deadline expired; discard')


@contextlib.contextmanager
def overlay(core, module):
    saved = {name: getattr(core, name) for name in ENTRIES}
    for name in ENTRIES:
        value = getattr(module, name)
        p.require(inspect.isbuiltin(value) and value.__module__ == module.__name__, 'overlay must retain true builtin origin')
    try:
        for name in ENTRIES: setattr(core, name, getattr(module, name))
        yield
    finally:
        for name, value in saved.items(): setattr(core, name, value)


def build(out, raw):
    """Reuse exact fa SDK and /GL binder; compile only two leaves and tiny init."""
    source = out / 'experiment-source'; donor = out / 'donor'; source.mkdir(); donor.mkdir()
    files = ('src/fft.cpp', 'src/simd_fft_avx2.cpp', 'src/fft_impl.hpp', 'src/fft_internal.hpp',
        'src/fft_angle.hpp', 'third_party/pocketfft/pocketfft_openmp.h')
    hashes = {}
    for name in files:
        body = subprocess.check_output(['git', 'show', CANDIDATE + ':' + name], cwd=ROOT)
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body); hashes[name] = H(body)
    prefix = p.PREFIX; retained = {}; helper = p.transport()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            sdk = name.startswith(prefix + 'relocated sdk/')
            binder = name == prefix + 'native-build/_core.dir/Release/bind_fft.obj'
            headers = name.startswith(prefix + 'build-env/Lib/site-packages/pybind11/include/')
            if not (sdk or binder or headers) or name.endswith('/'): continue
            relative = helper.safe_member(name[len(prefix):]); path = donor / relative
            body = z.read(name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body); retained[name] = H(body)
        with zipfile.ZipFile(io.BytesIO(z.read(prefix + 'source.zip'))) as frozen:
            state = json.loads(z.read(prefix + 'state.json'))
            for name in ('butterworth_cases.py', 'test_butterworth_public.py'):
                body = frozen.read('tests/' + name)
                p.require(H(body.replace(b'\r\n', b'\n')) == state['source']['files_sha256']['tests/' + name], 'frozen BW contract differs')
                (out / 'reference-tests' / name).write_bytes(body)
    compiler = subprocess.run(['cl.exe', '/Bv'], capture_output=True)
    (out / 'compiler.txt').write_bytes(compiler.stdout + compiler.stderr)
    p.require(b'19.51.36260' in compiler.stdout + compiler.stderr, 'actual fa19.51 LTCG compiler required')
    model = load('frozen_angle_model', ROOT / 'tools/fft_angle_contract.py'); ns = set(model.DENOMINATORS)
    for b in range(1,65):
        ns.update(n for d in (-2,-1,0,1,2) if 1 <= (n := (1 << b) + d) < 1 << 64)
    import struct
    rows = []
    for n in sorted(ns):
        exact = model.angle_fraction(n, 64)
        p.require(exact == model.Fraction(model.angle_decimal(n,64)), 'independent boundary refs disagree')
        rows.append(str(n) + ' ' + struct.pack('>d',float(exact)).hex() + '\n')
    # Frozen local reference text uses CRLF; preserve those exact pre-candidate bytes.
    refs = ''.join(rows).replace('\n','\r\n').encode(); p.require(H(refs) == REF_SHA, 'pre-candidate frozen363 refs differ')
    (out / 'angle-refs.txt').write_bytes(refs)
    p.dump(out / 'source-freeze.json', dict(candidate=CANDIDATE, source=hashes, donor=retained, refs_sha256=REF_SHA))
    commands = []
    def run(args, label):
        deadline(); commands.append(args); p.dump(out / 'build-commands.json', commands)
        with (out / (label + '.log')).open('xb') as stream:
            result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
        p.require(result.returncode == 0, 'private build failed: ' + label)
    sdk = donor / 'relocated sdk'; pybind = donor / 'build-env/Lib/site-packages/pybind11/include'
    common = ['cl.exe','/nologo','/std:c++17','/EHsc','/MD','/O2','/fp:strict','/openmp','/bigobj','/DNDEBUG','/DSKIMAGEXX_AVX2=1','/I'+str(sdk/'include'),'/I'+str(source/'src')]
    for name in ('fft','simd_fft_avx2'):
        run(common + (['/arch:AVX2'] if name.startswith('simd') else []) + ['/c',str(source/'src'/(name+'.cpp')),'/Fo'+str(out/(name+'.obj'))], name)
    run(['lib.exe','/nologo',str(sdk/'lib/skimagexx.lib'),'/REMOVE:skimagexx.dir\\Release\\fft.obj','/REMOVE:skimagexx.dir\\Release\\simd_fft_avx2.obj','/OUT:'+str(out/'without-fft.lib')], 'private-sdk-copy')
    init = out / 'tiny-init.cpp'
    init.write_text('#include <pybind11/pybind11.h>\nvoid bind_fft(pybind11::module_&);\nPYBIND11_MODULE(_fft_angle_candidate,m){bind_fft(m);m.attr("experimental_overlay")=true;}\n')
    run(['cl.exe','/nologo','/std:c++17','/EHsc','/MD','/O2','/DPy_NO_LINK_LIB','/I'+str(Path(sys.base_prefix)/'include'),'/I'+str(pybind),'/c',str(init),'/Fo'+str(out/'tiny-init.obj')], 'tiny-init')
    run(['link.exe','/nologo','/DLL','/LTCG','/OUT:'+str(out/'_fft_angle_candidate.pyd'),str(out/'tiny-init.obj'),str(donor/'native-build/_core.dir/Release/bind_fft.obj'),str(out/'fft.obj'),str(out/'simd_fft_avx2.obj'),str(out/'without-fft.lib'),str(Path(sys.base_prefix)/'libs/python313.lib')], 'experimental-overlay')
    p.dump(out / 'overlay.json', dict(name='_fft_angle_candidate',sha256=H((out/'_fft_angle_candidate.pyd').read_bytes()), experimental_overlay=True, whole_product_acceptance=False))


@contextlib.contextmanager
def deny_upstream(harness, original, kind):
    import pytest
    import scipy.fftpack
    import skimage.filters as public
    def denied(*args,**kwargs):raise AssertionError('upstream numerical provider forbidden')
    with pytest.MonkeyPatch.context() as patch:
        if kind=='lpi':
            for name in ('__call__','_prepare'):patch.setattr(original.LPIFilter2D,name,denied)
            for name in ('LPIFilter2D','filter_forward','filter_inverse','wiener'):
                patch.setattr(original,name,denied);patch.setattr(public,name,denied)
            for name in ('_min_limit','_center','_pad'):patch.setattr(original,name,denied)
        else:
            patch.setattr(original,'butterworth',denied);patch.setattr(public,'butterworth',denied)
            patch.setattr(original,'_get_nd_butterworth_filter',denied)
        for name,module in tuple(sys.modules.items()):
            if module is None or not name.startswith(('numpy.fft','scipy.fft')):continue
            for entry in tuple(vars(module)):
                if (entry in harness.FFT_ENTRIES or entry.startswith(('rfft_','irfft_'))) and callable(getattr(module,entry)):
                    patch.setattr(module,entry,denied)
        yield


def child(out, role, mode):
    deadline(); site = out / 'genuine-prefix'; auth = json.loads((out/'inputs.json').read_text())
    p.require(os.name == 'nt' and sys.version_info[:3] == (3,13,16), 'exact Windows Python required')
    forcing=('OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES','NPY_ENABLE_CPU_FEATURES','MKL_DEBUG_CPU_TYPE','MKL_SERVICE_FORCE_INTEL','LD_PRELOAD','LD_AUDIT')
    p.require(not any(os.environ.get(n) for n in forcing) and all(os.environ.get(n)=='1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')),'changed CPU/pool policy')
    import skimagexx as xx
    p.require(Path(xx.__file__).resolve() == site/'skimagexx/__init__.py', 'wrong product origin')
    p.require(H(Path(xx._core.__file__).read_bytes()) == auth['package']['members_sha256'][auth['package']['sole_core']], 'donor core changed')
    sys.path.insert(0,str(out/'reference-tests')); lpi = load('candidate_frozen_lpi',out/'reference-tests/test_lpi_filter_public.py')
    bw = load('candidate_frozen_bw',out/'reference-tests/test_butterworth_public.py')
    np, original, runtime = lpi.original_runtime(); _, original_bw, _ = bw.original_runtime()
    p.require(len(lpi.CASES)==80 and len(bw.CASES)==87, 'frozen case census changed')
    for name, version in auth['runtime_pins'].items():
        import importlib.metadata
        p.require(importlib.metadata.version(name)==version, 'runtime pin differs: '+name)
    for name in ('numpy','scipy','skimage','pytest'):
        p.require(Path(__import__(name).__file__).resolve().is_relative_to(site), 'foreign runtime '+name)
    module = xx._core
    if role == 'candidate':
        sys.path.insert(0,str(out)); import _fft_angle_candidate as module
        p.require(Path(module.__file__).resolve()==out/'_fft_angle_candidate.pyd' and H(Path(module.__file__).read_bytes())==json.loads((out/'overlay.json').read_text())['sha256'], 'wrong overlay origin')
    report = dict(new_host=True,experimental_overlay=role=='candidate',whole_product_acceptance=False,role=role,mode=mode,platform=platform.uname()._asdict(),runtime=runtime,targets=[],lpi=[],butterworth=[])
    report_path=out/(role+'-'+mode+'.json'); old_workers=xx.get_num_threads(); old_fallback=xx.set_fallback_enabled(False)
    old_entries={name:getattr(xx._core,name) for name in ENTRIES}; old_profile=sys.getprofile()
    shell=__import__('skimagexx._lpi_filter',fromlist=['_lpi_filter'])
    def record(group,row,call):
        deadline(); report[group].append(row)
        try:call();row['passed']=True
        except Exception:row['passed']=False;row['failure']=traceback.format_exc()
        p.dump(report_path,report)
    def completed(frame,event,value):
        if event=='c_return' and getattr(value,'__module__',None)==module.__name__ and getattr(value,'__name__',None) in ENTRIES:completions.append(value.__name__)
    try:
        with overlay(xx._core,module):
            for workers in (1,4):
                xx.set_num_threads(workers)
                for spec in (s for s in lpi.CASES if s['id'] in p.IDS):
                    row=dict(id=spec['id'],workers=workers,arrays=dict(original={},candidate={}))
                    folder=out/'observations'/role/mode/(str(workers)+'-'+spec['dtype']);folder.mkdir(parents=True)
                    def capture(which):
                        def save(name,value):row['arrays'][which][name]=p.capture_array(folder,which+'-'+name,value)
                        return save
                    def target():
                        expected=p.observe(original,lambda:lpi.run_case(original,spec,np),capture('original'),[]);lpi.validate_reference(spec,expected,np)
                        with deny_upstream(lpi,original,'lpi'):
                            actual=p.observe(shell,lambda:lpi.run_case(xx.filters,spec,np,completions),capture('candidate'),completions,module)
                        for name in ('F_input','G_input'):
                            p.require(all(row['arrays']['original'][name][key]==row['arrays']['candidate'][name][key] for key in ('dtype','shape','bytes','sha256')),'actual forward input differs')
                        row['forward_inputs_equal']=True;row['compiled_returns']=list(completions);lpi.compare_steps(actual,expected,spec,np)
                    completions=[];record('targets',row,target)
                for spec in lpi.CASES:
                    def one_lpi():
                        expected=lpi.run_case(original,spec,np);lpi.validate_reference(spec,expected,np)
                        sys.setprofile(completed)
                        try:
                            with deny_upstream(lpi,original,'lpi'):actual=lpi.run_case(xx.filters,spec,np,completions)
                        finally:sys.setprofile(old_profile)
                        row['compiled_returns']=list(completions);lpi.compare_steps(actual,expected,spec,np)
                    completions=[];row=dict(id=spec['id'],workers=workers);record('lpi',row,one_lpi)
                for spec in bw.CASES:
                    def one_bw():
                        image,kwargs,expected,error,messages=bw.reference_case(spec,np,original_bw)
                        sys.setprofile(completed)
                        try:
                            with deny_upstream(bw,original_bw,'bw'):actual,got_error,got_messages=bw.observe(xx.filters.butterworth,image,kwargs,np)
                        finally:sys.setprofile(old_profile)
                        row['compiled_returns']=list(completions)
                        p.require((got_error,got_messages)==(error,messages),'frozen BW errors/warnings differ')
                        if error is None:
                            p.require(completions,'missing actual FFT C_return');bw.compare_result(actual,expected,image,spec,kwargs,np)
                    completions=[];row=dict(id=spec['id'],workers=workers);record('butterworth',row,one_bw)
                p.require(xx.get_num_threads()==workers,'worker state changed')
    finally:
        sys.setprofile(old_profile);xx.set_num_threads(old_workers);xx.set_fallback_enabled(old_fallback)
        report['old_core_fft_restored']=all(getattr(xx._core,n) is v for n,v in old_entries.items())
        p.dump(report_path,report)
    for name,rec in auth['files'].items():p.require(H((site/name).read_bytes())==rec['sha256'],'genuine input mutated: '+name)
    report['genuine_files_unchanged']=True;p.dump(report_path,report)
    return int(any(not row['passed'] for group in ('targets','lpi','butterworth') for row in report[group]))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('plan','all','_child'));parser.add_argument('--output',type=Path,required=True);parser.add_argument('--role',choices=('baseline','candidate'));parser.add_argument('--mode',choices=('normal','scalar'));args=parser.parse_args(argv);out=args.output.resolve()
    p.require(out.is_relative_to((ROOT/'build').resolve()),'ignored owned output required')
    if args.stage=='plan':
        print(json.dumps(dict(candidate=CANDIDATE,deadline=DEADLINE,donor_run=p.DONOR_RUN,experimental_overlay=True,whole_product_acceptance=False,leaf_tus=2,tiny_init_tus=1,lpi_specs=80,bw_specs=87,modes=['normal','scalar'],workers=[1,4],reference_sha256=REF_SHA)));return 0
    p.require(os.environ.get('GITHUB_ACTIONS')=='true','hosted private experiment only; no local science')
    try:
        if args.stage=='_child':
            p.require(args.role is not None and args.mode is not None,'child role/mode required');return child(out,args.role,args.mode)
        deadline();out.mkdir(parents=True);helper=p.transport();metadata,raw=helper.fetch_artifact(os.environ.get('GH_TOKEN'));p.dump(out/'artifact.json',metadata);p.prepare(raw,out,helper);build(out,raw);del raw
        exits={}
        for role in ('baseline','candidate'):
            for mode in ('normal','scalar'):
                deadline();env=dict(os.environ,PYTHONPATH=str(out/'genuine-prefix'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1');env.pop('SKIMAGEXX_DISABLE_SIMD',None)
                if mode=='scalar':env['SKIMAGEXX_DISABLE_SIMD']='1'
                label=role+'-'+mode
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'_child','--output',str(out),'--role',role,'--mode',mode],env=env,capture_output=True)
                (out/(label+'.stdout')).write_bytes(result.stdout);(out/(label+'.stderr')).write_bytes(result.stderr);exits[label]=result.returncode;p.dump(out/'child-exits.json',exits)
        reports={key:json.loads((out/(key+'.json')).read_text()) for key in exits}
        for label,report in reports.items():
            p.require(label==report['role']+'-'+report['mode'] and len(report['targets'])==4 and len(report['lpi'])==160 and len(report['butterworth'])==174,'incomplete or mislabeled child contract groups')
        pairs=[]
        for mode in ('normal','scalar'):
            for baseline,candidate in zip(reports['baseline-'+mode]['targets'],reports['candidate-'+mode]['targets']):
                p.require((baseline['id'],baseline['workers'])==(candidate['id'],candidate['workers']),'target order differs')
                pairs.append(all(baseline['arrays'][role][name][key]==candidate['arrays'][role][name][key] for role in ('original','candidate') for name in ('F_input','G_input') for key in ('dtype','shape','bytes','sha256')))
        deadline();keep=all(pairs) and all(exits['candidate-'+mode]==0 for mode in ('normal','scalar')) and all(r['old_core_fft_restored'] and r['genuine_files_unchanged'] for r in reports.values())
        reproduced=all(not row['passed'] for mode in ('normal','scalar') for row in reports['baseline-'+mode]['targets'])
        p.dump(out/'decision.json',dict(keep=keep,same_host=True,baseline_reproduces_all8_failures=reproduced,causal_claim_permitted=keep and reproduced,input_pairs_equal=pairs,deadline=DEADLINE,child_exits=exits,experimental_overlay=True,whole_product_acceptance=False))
        return int(not keep)
    except Exception:
        out.mkdir(parents=True,exist_ok=True);(out/'failure.txt').write_text(traceback.format_exc())
        p.dump(out/'decision.json',dict(keep=False,reason='controller failure or deadline expiry; preserve failure.txt',experimental_overlay=True,whole_product_acceptance=False));raise


if __name__=='__main__':raise SystemExit(main())
