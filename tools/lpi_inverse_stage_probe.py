"""NEW HOST diagnostic of frozen Windows LPI failures, never product acceptance.
Only four existing cases per normal/scalar process; no build, timer, calibration,
CPU forcing, changed numerical source, or changed frozen public allowance.
"""
import argparse, ast, hashlib, importlib.util, io, json, os, platform, subprocess, sys, traceback, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DONOR_RUN = 40660212710
DONOR_SOURCE = 'fa264f395ad4f4040327c543e2f244873e67c46b'
ARTIFACT = 12805154787
ARTIFACT_SHA = '1c9b020e814826fc9f65b9858e8039a9c24ba58a623d7ff7f45fb1b903ab0ce3'
ARTIFACT_SIZE = 1131544952
WHEEL_SHA = 'd8e5b19d884211222fcdf898218f224785bc23cbc433beb99c149f167449b34a'
SOURCE_SHA = '28706c27e4329a0542743437392ecde26afcad38d6783cf3dc61384829bc949c'
PREFIX = 'public-qualification/windows/'
IDS = ('dtype-filter_inverse-uint8', 'dtype-filter_inverse-float64')
STAGES = ('F_input', 'F', 'G_input', 'G', 'F_floor', 'floor_mask', 'reciprocal', 'gain_mask', 'F_clipped', 'product', 'IFFT', 'crop')
H = lambda raw: hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value: raise ValueError(message)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def transport():
    # Reuse the reviewed signed-redirect fetch/RECORD/path code in an isolated
    # module instance. Historical source/constants/evidence remain unchanged.
    helper = load('lpi_donor_transport', ROOT / 'tools/matrix_runtime_certificate_probe.py')
    for name, value in dict(FAILED_RUN=DONOR_RUN, FAILED_SOURCE=DONOR_SOURCE, ARTIFACT_ID=ARTIFACT,
        ARTIFACT_SHA=ARTIFACT_SHA, ARTIFACT_SIZE=ARTIFACT_SIZE, MAX_ZIP=ARTIFACT_SIZE).items():
        setattr(helper, name, value)
    return helper


def prepare(raw, out, helper):
    site = out / 'genuine-prefix'; site.mkdir()
    files = {}; wheels = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        state = json.loads(z.read(PREFIX + 'state.json')); package = json.loads(z.read(PREFIX + 'package.json'))
        require(state['source']['revision'] == package['source_revision'] == DONOR_SOURCE, 'donor source differs')
        require(package['wheel_sha256'] == WHEEL_SHA and package['source_archive_sha256'] == SOURCE_SHA, 'donor product differs')
        require(len(package['members_sha256']) == 155 and len(package['SDK_sha256']) == 71, 'not the genuine155/71 product')
        require(len(state['runtime_wheels']) == 14, 'failed Windows runtime closure differs')
        selected = {PREFIX + 'product-wheels/' + state['product_wheel']: WHEEL_SHA}
        selected.update({PREFIX + 'runtime-wheels/' + n: r['sha256'] for n, r in state['runtime_wheels'].items()})
        for name, pin in selected.items():
            body = z.read(name); require(H(body) == pin, 'donor wheel differs: ' + name)
            records = helper.wheel_records(body); wheels[name] = pin
            if pin == WHEEL_SHA:
                require({n: r['sha256'] for n, r in records.items()} == package['members_sha256'], 'product RECORD differs')
            with zipfile.ZipFile(io.BytesIO(body)) as wheel:
                for relative, record in records.items():
                    require(relative not in files, 'wheel namespace collision')
                    target = site / helper.safe_member(relative); target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(wheel.read(relative)); files[relative] = record
        source = z.read(PREFIX + 'source.zip'); require(H(source) == SOURCE_SHA, 'frozen source ZIP differs')
        with zipfile.ZipFile(io.BytesIO(source)) as archive:
            require(archive.comment.decode() == DONOR_SOURCE, 'source archive identity differs')
            tests = out / 'reference-tests'; tests.mkdir()
            for name in ('lpi_filter_cases.py', 'test_lpi_filter_public.py'):
                body = archive.read('tests/' + name)
                require(H(body.replace(b'\r\n', b'\n')) == state['source']['files_sha256']['tests/' + name], 'frozen contract differs')
                (tests / name).write_bytes(body)
        dump(out / 'inputs.json', dict(new_host=True, public_acceptance=False, donor_run=DONOR_RUN,
             donor_source=DONOR_SOURCE, artifact=ARTIFACT, artifact_sha256=ARTIFACT_SHA,
             package=package, wheels=wheels, files=files, runtime_pins=state['runtime_pins']))


def observe(module, call, capture, completions, core=None):
    """Observe existing calls/line boundaries; never recompute their arguments."""
    source = Path(module.__file__).resolve(); tree = ast.parse(source.read_text())
    inverse = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'filter_inverse')
    mask_line = next(n.lineno for n in inverse.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'mask' for t in n.targets))
    return_line = next(n.lineno for n in inverse.body if isinstance(n, ast.Return))
    previous, profile = sys.gettrace(), sys.getprofile()
    require(previous is None and profile is None, 'external observer would change diagnostic context')
    owner = module if core is not None else module.fft
    entries = ('_fft',) if core is not None else ('fftn', 'ifftn')
    originals = {entry: getattr(owner, entry) for entry in entries}
    failed_frames = set(); forward_calls = 0
    def wrapped(entry):
        def fft(*args, **kwargs):
            nonlocal forward_calls
            forward = args[1] is True if core is not None else entry == 'fftn'
            if forward:
                require(forward_calls < 2, 'unexpected additional forward transform')
                capture(('F_input', 'G_input')[forward_calls], args[0]); forward_calls += 1
            else: capture('product', args[0])
            value = originals[entry](*args, **kwargs)
            if not forward: capture('IFFT', value)
            return value
        return fft
    def trace(frame, event, value):
        if Path(frame.f_code.co_filename).resolve() != source: return None
        name = frame.f_code.co_name; local = frame.f_locals
        if event == 'exception': failed_frames.add(id(frame))
        if event == 'return' and id(frame) in failed_frames:
            failed_frames.remove(id(frame)); return trace
        if event == 'return' and name == '_prepare':
            capture('F', value[0]); capture('G', value[1])
        elif event == 'return' and name == '_min_limit':
            capture('F_floor', local['x']); capture('floor_mask', local.get('mask', local.get('idx')))
        elif name == 'filter_inverse':
            if event == 'line' and frame.f_lineno == mask_line: capture('reciprocal', local['F'])
            elif event == 'line' and frame.f_lineno == return_line:
                capture('gain_mask', local['mask']); capture('F_clipped', local['F'])
            elif event == 'return': capture('crop', value)
        return trace
    def completed(frame, event, value):
        if event == 'c_return' and core is not None and getattr(value, '__module__', None) == core.__name__:
            if getattr(value, '__name__', '') in ('fft_r2c', 'fft_c2r', 'fft_c2c'): completions.append(value.__name__)
    for entry in entries: setattr(owner, entry, wrapped(entry))
    sys.settrace(trace); sys.setprofile(completed)
    try: return call()
    finally:
        sys.settrace(previous); sys.setprofile(profile)
        for entry, real in originals.items(): setattr(owner, entry, real)


def capture_array(folder, name, array):
    raw = array.tobytes(order='C')
    record = dict(dtype=array.dtype.str, shape=list(array.shape), strides=list(array.strides),
        owndata=bool(array.flags.owndata), writeable=bool(array.flags.writeable),
        base_type=type(array.base).__name__, bytes=len(raw), sha256=H(raw), file=name + '.bin')
    (folder / record['file']).write_bytes(raw); dump(folder / (name + '.json'), record)
    return record


def compare_stages(folder, stages, np):
    rows = []; first = None
    for name in STAGES:
        left, right = stages['original'][name], stages['candidate'][name]
        a = (folder / left['file']).read_bytes(); b = (folder / right['file']).read_bytes()
        same_type = (left['dtype'], left['shape']) == (right['dtype'], right['shape'])
        exact = same_type and a == b
        if not exact and first is None: first = name
        row = dict(stage=name, exact_logical_words=exact, same_dtype_shape=same_type)
        if same_type:
            x = np.frombuffer(a, dtype=left['dtype']); y = np.frombuffer(b, dtype=right['dtype'])
            row['unequal_elements'] = int(np.count_nonzero(x != y))
            if x.dtype.kind != 'b':
                finite = np.isfinite(x) & np.isfinite(y)
                row['max_abs_finite_error'] = float(np.max(np.abs(x[finite] - y[finite]))) if np.any(finite) else None
        rows.append(row)
    return dict(first_bitwise_divergence=first, stages=rows,
        intermediate_differences_are_diagnostics_not_new_allowances=True)


def child(out, mode):
    require(os.name == 'nt' and sys.version_info[:3] == (3, 13, 16) and sys.flags.optimize == 0, 'exact fresh Windows CPython3.13.16 required')
    forcing = ('OPENBLAS_CORETYPE', 'NPY_DISABLE_CPU_FEATURES', 'NPY_ENABLE_CPU_FEATURES', 'MKL_DEBUG_CPU_TYPE', 'MKL_SERVICE_FORCE_INTEL', 'LD_PRELOAD', 'LD_AUDIT')
    pools = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS')
    environment_before = {name: os.environ.get(name) for name in (*forcing, *pools)}
    require(not any(n in os.environ for n in forcing[:5]) and not any(os.environ.get(n) for n in forcing[5:]) and all(os.environ.get(n) == '1' for n in pools), 'forced CPU or changed pool policy')
    auth = json.loads((out / 'inputs.json').read_text()); site = (out / 'genuine-prefix').resolve()
    require(site in {Path(p).resolve() for p in sys.path if p}, 'genuine prefix must precede startup imports')
    import skimagexx as xx
    require(Path(xx.__file__).resolve() == site / 'skimagexx/__init__.py', 'wrong product origin')
    sys.path.insert(0, str(out / 'reference-tests'))
    harness = load('frozen_lpi_contract', out / 'reference-tests/test_lpi_filter_public.py')
    np, original, runtime = harness.original_runtime()
    candidate = __import__('skimagexx._lpi_filter', fromlist=['_lpi_filter'])
    require(Path(candidate.__file__).resolve() == site / 'skimagexx/_lpi_filter.py', 'wrong LPI shell')
    for name in ('numpy', 'scipy', 'skimage', 'pytest'):
        require(Path(__import__(name).__file__).resolve().is_relative_to(site), 'dependency outside genuine prefix: ' + name)
    for name, version in auth['runtime_pins'].items():
        import importlib.metadata
        require(importlib.metadata.version(name) == version, 'runtime version differs: ' + name)
    require(H(Path(xx._core.__file__).read_bytes()) == auth['package']['members_sha256'][auth['package']['sole_core']], 'core differs')
    cases = [s for s in harness.CASES if s['id'] in IDS]
    require(tuple(s['id'] for s in cases) == IDS and harness.error_limits.__globals__['RELATIVE_EPS_FACTOR'] == 32
        and harness.error_limits.__globals__['PEAK_ABSOLUTE_EPS_FACTOR'] == 64, 'frozen case/allowance differs')
    report = dict(scope='NEW HOST observations; NEVER original failed-host provenance or product acceptance',
        donor_run=DONOR_RUN, donor_source=DONOR_SOURCE, artifact=ARTIFACT, mode=mode, public_acceptance=False,
        platform=platform.uname()._asdict(), python=sys.version, runtime=runtime, core_sha256=H(Path(xx._core.__file__).read_bytes()),
        scalar_disabled=os.environ.get('SKIMAGEXX_DISABLE_SIMD'), environment_before_import=environment_before, cases=[])
    old_workers = xx.get_num_threads(); old_fallback = xx.set_fallback_enabled(False)
    try:
        for workers in (1, 4):
            for spec in cases:
                label = str(workers) + '-' + spec['dtype']; folder = out / 'observations' / mode / label; folder.mkdir(parents=True)
                row = dict(nodeid='tests/test_lpi_filter_public.py::test_lpi_owned_fft[workers' + str(workers) + '-' + spec['id'] + ']',
                    workers=workers, spec=spec, arrays=dict(original={}, candidate={}), frozen_public_predicate_passed=False)
                report['cases'].append(row)
                def captured(role):
                    def save(name, value):
                        require(name not in row['arrays'][role], 'unexpected repeated numerical stage')
                        row['arrays'][role][name] = capture_array(folder, role + '-' + name, value)
                    return save
                try:
                    expected = observe(original, lambda: harness.run_case(original, spec, np), captured('original'), [])
                    harness.validate_reference(spec, expected, np)
                    xx.set_num_threads(workers); completions = []
                    actual = observe(candidate, lambda: harness.run_case(xx.filters, spec, np, completions), captured('candidate'), completions, xx._core)
                    row['actual_compiled_returns'] = completions
                    row['step_protocol'] = {role: [{k: step[k] for k in ('error', 'warnings', 'cache_metadata', 'cache_same', 'cache_changed', 'callback_retained', 'params')} for step in steps]
                        for role, steps in (('original', expected), ('candidate', actual))}
                    try:
                        harness.compare_steps(actual, expected, spec, np)  # The unmodified frozen public predicate.
                        row['frozen_public_predicate_passed'] = True
                    except AssertionError:
                        row['frozen_predicate_failure'] = traceback.format_exc()
                    row['comparison'] = compare_stages(folder, row['arrays'], np)
                    require(xx.get_num_threads() == workers, 'worker state changed inside observed call')
                except Exception:
                    row['failure'] = traceback.format_exc()
                finally:
                    xx.set_num_threads(old_workers)
                    dump(out / (mode + '-observations.json'), report)
    finally:
        xx.set_num_threads(old_workers); xx.set_fallback_enabled(old_fallback)
    for name, record in auth['files'].items():
        require(H((site / name).read_bytes()) == record['sha256'], 'genuine runtime input mutated: ' + name)
    report['input_files_post_unchanged'] = True; report['observers_restored'] = sys.gettrace() is None and sys.getprofile() is None
    dump(out / (mode + '-observations.json'), report)
    return int(any(not r['frozen_public_predicate_passed'] or 'failure' in r for r in report['cases']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=('plan', 'all', '_child'))
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--mode', choices=('normal', 'scalar'))
    args = parser.parse_args(argv); out = args.output.resolve()
    require(out.is_relative_to((ROOT / 'build').resolve()), 'ignored owned build output required')
    if args.stage == 'plan':
        print(json.dumps(dict(donor_run=DONOR_RUN, donor_source=DONOR_SOURCE, artifact=ARTIFACT,
            artifact_sha256=ARTIFACT_SHA, wheel_sha256=WHEEL_SHA, source_sha256=SOURCE_SHA,
            cases=4, modes=['normal', 'scalar'], captured_stages=list(STAGES), new_host=True, public_acceptance=False, builds=0, timings=0))); return 0
    require(os.environ.get('GITHUB_ACTIONS') == 'true', 'fresh hosted probe only; no local runtime preparation')
    try:
        if args.stage == '_child':
            require(args.mode is not None, 'child mode required'); return child(out, args.mode)
        require(os.name == 'nt', 'Windows donor only'); out.mkdir(parents=True)
        helper = transport(); metadata, raw = helper.fetch_artifact(os.environ.get('GH_TOKEN'))
        dump(out / 'artifact.json', metadata); prepare(raw, out, helper); del raw
        dump(out / 'controller.json', dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             source_sha256=H(Path(__file__).read_bytes()), source_utf8=Path(__file__).read_text(), new_host=True, public_acceptance=False))
        exits = {}
        for mode in ('normal', 'scalar'):
            environment = dict(os.environ, PYTHONPATH=str(out / 'genuine-prefix'), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
            environment.pop('SKIMAGEXX_DISABLE_SIMD', None)
            if mode == 'scalar': environment['SKIMAGEXX_DISABLE_SIMD'] = '1'
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_child', '--output', str(out), '--mode', mode],
                env=environment, capture_output=True)
            (out / (mode + '.stdout')).write_bytes(result.stdout); (out / (mode + '.stderr')).write_bytes(result.stderr)
            exits[mode] = result.returncode; dump(out / 'child-exits.json', exits)
        return int(any(exits.values()))
    except Exception:
        out.mkdir(parents=True, exist_ok=True); (out / 'failure.txt').write_text(traceback.format_exc()); raise


if __name__ == '__main__': raise SystemExit(main())
