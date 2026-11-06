"""Prospective actual-CLI probe; requires a root-bound genuine product and GO."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--root-go', type=Path, required=True)
    parser.add_argument('--route', required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_bytes())
    go = json.loads(args.root_go.read_bytes())
    assert go['binding_sha256'] == sha(args.binding)
    assert go['probe_sha256'] == sha(__file__)
    assert args.route in go['routes'] and go['timed_batches'] == 0
    assert len(binding['routes']) == 7 and set(go['routes']) <= set(binding['routes'])
    assert datetime.now(timezone.utc) < datetime.fromisoformat(go['deadline_utc'])
    assert go['maximum_original_topcalls'] <= 7 and go['maximum_native_topcalls'] <= 7
    root = Path(binding['source_root']).resolve()
    git = binding['git_argv']
    assert not os.environ.get('GIT_DIR') and not os.environ.get('GIT_WORK_TREE')
    assert subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip() == binding['source_revision']
    assert not subprocess.check_output([*git, 'status', '--porcelain'], text=True).strip()
    assert sha(binding['genuine_root_admission_path']) == binding['genuine_root_admission_sha256']
    assert binding['product_kind'] == 'GENUINE_INSTALLED_WHOLE_PRODUCT'
    assert Path(sys.executable).resolve() == Path(binding['interpreter']).resolve()
    for name, pin in binding['source_files_sha256'].items():
        assert sha(root / name) == pin
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
        assert os.environ[key] == '1'
    assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'
    assert not os.environ.get('SKIMAGEXX_DISABLE_SIMD')
    route = binding['routes'][args.route]
    folder = Path(binding['output_root']) / args.route
    folder.mkdir(parents=True, exist_ok=False)  # Never replay a consumed route.
    output = folder / 'never-written.json'
    def save(name, value):
        (folder / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    def journal(event, role):
        with (folder / 'calls.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event, role=role, counts=dict(counts))) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    sys.path[:0] = [binding['installed_site'], str(root / 'benchmarks'), str(root / 'tools'), str(root)]
    import numpy as np
    import skimage
    import skimagexx as xx
    from skimagexx import _compat
    assert np.__version__ == binding['numpy'] and skimage.__version__ == '0.26.0'
    assert Path(xx._core.__file__).resolve() == Path(binding['core_path']).resolve()
    assert sha(xx._core.__file__) == binding['core_sha256']
    assert Path(xx.__file__).resolve().is_relative_to(Path(binding['installed_site']).resolve())
    public = importlib.import_module('skimage.draw')
    native_public = importlib.import_module('skimagexx.draw')
    original = getattr(public, route['api'].split('.')[1])
    native = getattr(native_public, route['api'].split('.')[1])
    owned = importlib.import_module(route['owned_module'])
    assert native is getattr(owned, route['api'].split('.')[1])
    assert inspect.signature(native) == inspect.signature(original)
    expected_entries = route['native_entries']
    entries = {getattr(xx._core, name): name for name in expected_entries}
    assert all(inspect.isbuiltin(entry) for entry in entries)
    getter = xx._core.get_num_threads
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    workers, fallback = xx.get_num_threads(), _compat._fallback_enabled
    assert workers == 1
    oldtrace, oldprofile, oldargv = sys.gettrace(), sys.getprofile(), sys.argv
    counts = dict(original=0, native=0)
    native_active = False
    accepted, c_returns, getters = [], [], []
    inputs_before = []
    def input_state(value):
        if isinstance(value, np.ndarray):
            return (id(value), value.dtype.str, repr(value.dtype.metadata), value.shape,
                    value.strides, value.ctypes.data, value.flags.writeable, value.tobytes())
        if isinstance(value, (tuple, list)):
            return (type(value).__name__, tuple(input_state(item) for item in value))
        if isinstance(value, dict):
            return tuple((key, input_state(item)) for key, item in sorted(value.items()))
        return value
    stopped = {}
    sampler = root / route['sampler']
    driver = root / route['driver']
    argv = [str(driver), *route['argv'], '--output', str(output)]
    if route['kind'] == 'dedicated':
        argv += ['--deadline-utc', go['deadline_utc']]
    class StopBeforeBody(BaseException):
        pass
    def profile(frame, event, value):
        if native_active and event == 'c_call':
            assert not str(getattr(value, '__module__', '')).startswith('skimage.draw')
        if native_active and event == 'c_return':
            if value in entries:
                c_returns.append(entries[value])
            if value is getter:
                getters.append('get_num_threads')
    def trace(frame, event, value):
        nonlocal native_active
        role = 'original' if frame.f_code is original.__code__ else 'native' if frame.f_code is native.__code__ else None
        if event == 'call' and role:
            if role == 'original':
                assert not native_active
            assert counts[role] == 0
            counts[role] += 1
            journal('before', role)
            if role == 'original' and route['kind'] == 'generic':
                parent = frame.f_back
                while parent is not None and Path(parent.f_code.co_filename).resolve() != driver:
                    parent = parent.f_back
                assert parent is not None
                inputs_before.append(input_state((parent.f_locals['call_args'], parent.f_locals['options'])))
            if role == 'native':
                native_active = True
        if event == 'return' and role:
            if role == 'native':
                result = frame.f_locals.get('native_result')
                accepted.append(result is True if route['api'] == 'draw.set_color' else result is not None)
                native_active = False
            journal('after', role)
        if event == 'call' and frame.f_code.co_name == 'measure' and Path(frame.f_code.co_filename).resolve() == sampler:
            assert counts == dict(original=1, native=1) and accepted == [True]
            assert not {'samples', 'functions', 'events', 'loops'}.intersection(frame.f_locals)
            caller = frame.f_back.f_locals
            assert Path(frame.f_back.f_code.co_filename).resolve() == driver
            if route['kind'] == 'dedicated':
                assert caller['spec'] == route['first_case']
                assert caller['calls'] == expected_entries and caller['accepted'] == [True]
                assert caller.get('snapshots', caller.get('gets')) == ['get_num_threads']
                assert sys.getprofile() is profile  # Driver restored our outer profiler.
                assert not caller['rows']
            else:
                assert caller['api'] == route['api'] and caller['error'] == 0
                assert caller['case_dtype'] == route['first_carrier_dtype']
                assert caller['threads'] == 1 and not caller['results']
                assert c_returns == expected_entries and getters == ['get_num_threads']
                assert inputs_before == [input_state((caller['call_args'], caller['options']))]
            assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
            assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
            stopped.update(source=binding['source_revision'], route=args.route, api=route['api'],
                           counts=dict(counts), native_result_accepted=accepted,
                           real_sampler=str(sampler), sampler_body_calls=0, timed_batches=0,
                           public_driver_checks_completed=True)
            save('sampler-entry.json', stopped)
            raise StopBeforeBody()
        return trace
    save('precall.json', dict(source=binding['source_revision'], argv=argv, route=route,
                             core_path=binding['core_path'], core_sha256=binding['core_sha256'],
                             product_kind=binding['product_kind'], binding_sha256=sha(args.binding)))
    try:
        try:
            sys.settrace(trace)
            sys.setprofile(profile)
            sys.argv = argv
            runpy.run_path(str(driver), run_name='__main__')
        except StopBeforeBody:
            pass
        assert stopped and not output.exists() and not output.with_suffix('.batches.jsonl').exists()
    except BaseException as error:
        save('failure.json', dict(type=type(error).__name__, error=str(error), counts=counts,
                                 traceback=traceback.format_exc(), retry=False))
        raise
    finally:
        sys.settrace(oldtrace)
        sys.setprofile(oldprofile)
        sys.argv = oldargv
        xx.set_num_threads(workers)
        xx.set_fallback_enabled(fallback)
    save('terminal.json', dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY', **stopped,
                               outer_state_restored=xx.get_num_threads() == workers and _compat._fallback_enabled == fallback))


if __name__ == '__main__':
    main()
