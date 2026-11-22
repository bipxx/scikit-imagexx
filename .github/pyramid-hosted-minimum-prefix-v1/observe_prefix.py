"""Observe four exact hosted minimum pyramid pytest nodes in their actual order."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
NODE_PREFIX = 'tests/test_pyramid_spatial_public.py::test_frozen_pyramid_stages'
PINS_SHA = '1e40efe17bf4faf2a3659955f668c15789ac0bb7765e4b0c2ac6dd0ab6192334'
ADDITIONS = {
    'pyproject.toml': '8bb611200035980a4df9e212d1d80c12ffa6890174fc851936976a51d6197866',
    'tests/conftest.py': '8ce1a762ad4e8a37843103656b83eb9f21c9a71722a379fcd7e778d68ab2d2d6',
    'tests/test_pyramid_spatial_public.py': '63faeb46de00ccf4a845d57d091f871df234833f968a6c54d6e3d6e79749c2ef',
    'tests/util_views_fenv.py': '275cfab25633b20489d1f173ac8fba16563ae8ffeed2e7f649d6e48338cef894',
}
CASES = ('reduce-odd-f32-reflect', 'reduce-planar-u8-constant', 'expand-odd-f32-wrap', 'expand-rgb-u8-reflect')
NODES = [f'{NODE_PREFIX}[{case}]' for case in CASES]
POOLS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
         'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_THREAD_LIMIT')
REUSED = {
    'diagnose.py': 'b6bdfd13d2f2d15dbec33f41d544a083fdec2722456c02d95cbfe55ca94a1b0b',
    'extract_candidate.py': '8131cc590cfc9e1acfb0e04af56de8e232faa2df829f6f0e7b3bfb139b82d8b9',
    'pins.json': PINS_SHA,
    'range_helper.py': '1a54751980e8d5c06e31c46891f5a6112b5a6f64a07f494497fe93b98e912475',
    'requirements-runtime.txt': 'ad18bd86522b3ee91723952d696fa71285caf236f6c514b12cee45362e2198b5',
}


def digest(raw):
    return sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def files():
    assert all(digest((HERE / name).read_bytes()) == pin for name, pin in REUSED.items())
    return read(HERE / 'pins.json')


def authority(pins):
    grant = json.loads(os.environ['ROOT_AUTHORIZATION_JSON'])
    assert set(grant) == {'scope', 'source', 'tree', 'carrier', 'observer_sha256', 'nodes',
                          'deadline_utc', 'candidate_suite_calls', 'max_extra_public_calls', 'timings'}
    assert grant['scope'] == 'current296-pyramid-minimum-prefix-v1'
    assert (grant['source'], grant['tree']) == (pins['source'], pins['tree'])
    assert grant['carrier'] == os.environ['GITHUB_SHA'] and os.environ['GITHUB_RUN_ATTEMPT'] == '1'
    assert os.environ['GITHUB_ACTIONS'] == 'true'
    assert grant['observer_sha256'] == digest(Path(__file__).read_bytes())
    assert grant['nodes'] == NODES
    assert (grant['candidate_suite_calls'], grant['max_extra_public_calls'], grant['timings']) == (4, 2, 0)
    deadline = datetime.fromisoformat(grant['deadline_utc'])
    assert deadline.utcoffset() is not None and deadline.utcoffset().total_seconds() == 0
    assert datetime.now(timezone.utc) < deadline
    return deadline


def verify_source(source, pins):
    git = ['git', '--no-optional-locks', '-C', str(source)]
    def call(*args):
        return subprocess.check_output([*git, *args])
    assert call('rev-parse', 'HEAD', 'HEAD^{tree}').decode().splitlines() == [pins['source'], pins['tree']]
    assert not call('status', '--porcelain=v1', '--untracked-files=all')
    for name, expected in ADDITIONS.items():
        assert digest(call('show', 'HEAD:' + name)) == expected, name
        assert digest((source / name).read_bytes()) == expected, name


def floating_snapshot(np, cache, fenv):
    raw = fenv['FEnv']()
    assert fenv['libc'].fegetenv(ctypes.byref(raw)) == 0
    return dict(control=raw.control, status=raw.status, tags=raw.tags, mxcsr=raw.mxcsr,
                rounding=fenv['libc'].fegetround(), numpy_geterr=np.geterr(),
                numpy_geterrcall=type(np.geterrcall()).__name__, numpy_bufsize=np.getbufsize(),
                cache_info=cache.cache_info()._asdict())


def main_probe(source, stock, candidate, output, extraction, temporary, pins, deadline):
    reuse = runpy.run_path(str(HERE / 'diagnose.py'))
    verification = reuse['prove_inputs'](source, stock, candidate, extraction, pins)
    verify_source(source, pins)
    assert all(os.environ.get(key) == '1' for key in POOLS)
    assert not any(os.environ.get(key) for key in ('PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'PYTEST_ADDOPTS', 'GOLDEN_CORE_EXTENSION'))
    assert not {'numpy', 'scipy', 'skimage', 'skimagexx', 'pytest'} & sys.modules.keys()
    sys.path[:0] = [str(candidate), str(stock), str(source), str(source / 'tests')]
    os.chdir(source)
    import inspect
    import warnings
    import numpy as np
    from numpy.core import _multiarray_umath as umath
    import pytest
    import skimagexx as xx
    from skimagexx import _core, _pyramid_spatial, filters, transform
    from skimage import transform as original
    from skimage.transform import pyramids, _warps
    from skimage._shared import filters as upstream_filter
    from skimage.util import dtype
    from scipy.ndimage import _filters, _nd_image, _interpolation, _ni_support

    paths = (umath, _filters, _nd_image, _interpolation, _ni_support, pyramids, _warps, upstream_filter, dtype)
    assert {Path(m.__file__).resolve().relative_to(stock).as_posix() for m in paths} == set(pins['stock_members'])
    for mod, path in ((_core, verification['core_member']), (filters, 'skimagexx/filters.py'),
                      (transform, 'skimagexx/transform.py'), (_pyramid_spatial, 'skimagexx/_pyramid_spatial.py')):
        assert Path(mod.__file__).resolve() == candidate / path
    assert transform.pyramid_expand is _pyramid_spatial.pyramid_expand
    assert original.pyramid_expand is pyramids.pyramid_expand
    assert inspect.isbuiltin(_core.gaussian_nd)
    assert xx.get_num_threads() == 1
    assert np.__version__ == '1.24.4' and pytest.__version__ == '9.1.1'

    fenv = runpy.run_path(str(source / 'tests/util_views_fenv.py'))
    assert platform.machine() == 'x86_64' and sys.byteorder == 'little'
    assert ctypes.sizeof(fenv['FEnv']) == 32 and fenv['FEnv'].mxcsr.offset == 28
    libc = ctypes.CDLL('libm.so.6')
    libc.fegetenv.argtypes, libc.fegetenv.restype = [ctypes.c_void_p], ctypes.c_int
    libc.fegetround.argtypes, libc.fegetround.restype = [], ctypes.c_int
    fenv['libc'] = libc
    cache = filters._gaussian_cached_coefficients

    fixtures = runpy.run_path(str(source / 'tests/pyramid_spatial_cases.py'))
    state = fixtures['array_state']
    assert [case['id'] for case in fixtures['FIXED']] == list(CASES)
    contract = read(source / 'tests/fixtures/pyramid_spatial_originals_v1.json')
    frozen = {row['case_id']: row for row in contract['minimum']}
    cpu = reuse['cpu_state'](umath)
    write(output / 'runtime.json', dict(pid=os.getpid(), source=pins['source'], python=sys.version,
          executable=sys.executable, core_sha256=pins['candidate_core_sha256'], numpy=np.__version__,
          cpu=cpu, affinity_restricted=True, mode='normal', pools={k: os.environ[k] for k in POOLS},
          state=floating_snapshot(np, cache, fenv)))

    collected = []
    observations = {}
    phase = output / 'four-node-rows.jsonl'
    assert not phase.exists()

    class StageObserver:
        def pytest_collection_finish(self, session):
            collected.extend(item.nodeid for item in session.items)
            assert collected == NODES
            conftest = source / 'tests/conftest.py'
            loaded = [p for p in session.config.pluginmanager.get_plugins()
                      if Path(getattr(p, '__file__', '') or '.').resolve() == conftest]
            assert len(loaded) == 1 and session.config.inipath == source / 'pyproject.toml'
            write(output / 'collection.json', dict(nodes=collected, conftest=str(conftest),
                  config=str(session.config.inipath),
                  external_plugins=[str(dist.project_name) for _, dist in session.config.pluginmanager.list_plugin_distinfo()]))

        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_call(self, item):
            assert item.nodeid in NODES and item.nodeid not in observations
            assert datetime.now(timezone.utc) < deadline
            row = dict(node=item.nodeid, before=floating_snapshot(np, cache, fenv), kernels=[], stages=[])
            candidate_code = filters._gaussian_nd.__code__
            test_code = item.obj.__code__

            def call_probe(frame, event, result):
                if event == 'call' and frame.f_code in (candidate_code, test_code):
                    frame.f_trace_lines = False
                    frame.f_trace_opcodes = False
                    return call_probe
                if event == 'return' and frame.f_code is test_code:
                    row['stages'] = frame.f_locals.get('stage_results', [])
                    visited = frame.f_locals.get('visited', [])
                    row['native_entries'] = list(visited)
                elif event == 'return' and frame.f_code is candidate_code and result is not None:
                    fields = frame.f_locals
                    for axis, weights, mode in zip(fields['axes'], fields['kernels'], fields['codes'], strict=True):
                        sigma = fields['sigmas'][axis]
                        raw = weights.tobytes()
                        row['kernels'].append(dict(axis=axis, sigma_hex=float(sigma).hex(), sigma_type=type(sigma).__name__,
                            radius=weights.size // 2, border_code=mode, coefficients_f64_hex=raw.hex(),
                            coefficients_sha256=digest(raw), immutable=not weights.flags.writeable))
                return call_probe

            prior = sys.gettrace()
            assert prior is None
            sys.settrace(call_probe)
            try:
                return (yield)
            finally:
                sys.settrace(prior)
                row['after'] = floating_snapshot(np, cache, fenv)
                row['result_sha256'] = next((stage[1]['sha256'] for stage in row['stages'] if stage[0] == 'gaussian'), None)
                observations[item.nodeid] = row
                with phase.open('a', encoding='utf-8', newline='\n') as stream:
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()

    assert temporary.is_dir() and not (temporary / 'pytest').exists()
    pytest_args = ['-q', '--capture=sys', '--tb=short', '--junitxml=' + str(output / 'four-nodes.xml'),
                   '--basetemp=' + str(temporary / 'pytest'), '-o', 'cache_dir=' + str(temporary / 'pytest-cache'), *NODES]
    write(output / 'pytest-command.json', dict(argv=pytest_args, cwd=str(source), harness_plugin='StageObserver',
          untouched_source_test_sha256=ADDITIONS['tests/test_pyramid_spatial_public.py']))
    code = int(pytest.main(pytest_args, plugins=[StageObserver()]))
    (output / 'pytest.exit').write_text(f'{code}\n', encoding='ascii')
    rows = list(ET.parse(output / 'four-nodes.xml').iter('testcase'))
    seen = [node.attrib['name'] for node in rows]
    assert seen == [node.split('::', 1)[1] for node in NODES]
    assert not any(any(child.tag in ('skipped', 'error') for child in node) for node in rows)
    assert collected == list(observations) == NODES
    assert all(len(row['kernels']) == 2 and len(row['stages']) == 3 for row in observations.values())
    target = observations[NODES[-1]]
    expected = pins['frozen_output_sha256']
    observed = target['result_sha256']
    assert observed is not None
    prior = observations[NODES[0]]
    assert [r['radius'] for r in prior['kernels']] == [3, 3]
    assert [r['sigma_hex'] for r in prior['kernels']] == [r['sigma_hex'] for r in target['kernels']]
    comparison = dict(prior_coefficients=[r['coefficients_sha256'] for r in prior['kernels']],
                      target_coefficients=[r['coefficients_sha256'] for r in target['kernels']],
                      expected=expected, observed=observed)
    output_descriptor = dict(candidate_suite_calls=4, extra_candidate_calls=0, extra_stock_calls=0, timings=0)
    write(output / 'prefix-result.json', dict(pytest_exit=code, comparison=comparison, counts=output_descriptor,
          extra_pair_pending=(observed != expected)))
    if observed == expected:
        return code
    assert datetime.now(timezone.utc) < deadline
    spec = fixtures['FIXED'][-1]
    image = fixtures['fixture'](spec['shape'], spec['dtype'], np)
    assert state(image) == frozen[pins['case']]['input']
    sides = {}
    arrays = {}
    was_enabled = xx.set_fallback_enabled(False)
    try:
        for name, function in (('candidate', transform.pyramid_expand), ('stock', original.pyramid_expand)):
            assert datetime.now(timezone.utc) < deadline
            before = floating_snapshot(np, cache, fenv)
            complete = {}

            def extra_trace(frame, event, value):
                if event != 'return' or value is None:
                    return
                if frame.f_code is filters._gaussian_nd.__code__:
                    complete['candidate_kernels'] = [w.tobytes().hex() for w in frame.f_locals['kernels']]
                if frame.f_code is _filters.gaussian_filter1d.__code__:
                    d = frame.f_locals
                    complete.setdefault('stock_axes', []).append(dict(axis=d['axis'],
                        coefficients=d['weights'].tobytes().hex(), state=state(value)))

            previous = sys.getprofile()
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                try:
                    sys.setprofile(extra_trace)
                    array = function(image, **spec['options'])
                finally:
                    sys.setprofile(previous)
            arrays[name] = array
            sides[name] = dict(before=before, after=floating_snapshot(np, cache, fenv), output=state(array),
                               structures=complete,
                               warnings=[[n.category.__name__, str(n.message)] for n in notices])
            assert state(image) == frozen[pins['case']]['input']
            write(output / (name + '-extra-side.json'), sides[name])
    finally:
        xx.set_fallback_enabled(was_enabled)
    assert len(sides['candidate']['structures']['candidate_kernels']) == 2
    assert len(sides['stock']['structures']['stock_axes']) == 2
    write(output / 'extra-pair.json', dict(sides=sides, difference=reuse['compare'](np, arrays['stock'], arrays['candidate']),
          candidate_reproduced_prefix=sides['candidate']['output']['sha256'] == observed,
          original_calls=1, candidate_calls=1, extra_native_axis_calls=0, timings=0))
    return code if code else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inspect-only', action='store_true')
    mode.add_argument('--entry-only', action='store_true')
    arguments = ('source', 'stock', 'candidate', 'output', 'extraction', 'temporary')
    for name in arguments:
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if not args.inspect_only and os.environ.get('ROOT_AUTHORIZATION_JSON', '') in ('', 'UNBOUND'):
        print('REFUSED_NO_ROOT_AUTHORITY', file=sys.stderr)
        return 2
    pins = files()
    if args.inspect_only:
        assert not os.environ.get('ROOT_AUTHORIZATION_JSON')
        assert not any(getattr(args, field) for field in arguments)
        assert not {'numpy', 'scipy', 'skimage', 'skimagexx', 'pytest'} & sys.modules.keys()
        print(json.dumps(dict(nodes=NODES, source=pins['source'], observer_sha256=digest(Path(__file__).read_bytes()))))
        return 0
    deadline = authority(pins)
    assert sys.platform == 'linux' and sys.version_info[:3] == (3, 11, 17)
    assert digest(Path(sys.executable).read_bytes()) == 'd601e62487858384d3000725c3b2e16e86ff11a6ad0b0810b76281e45426ea62'
    if args.entry_only:
        assert args.source is not None and not any(getattr(args, field) for field in arguments[1:])
        verify_source(args.source.resolve(), pins)
        print('ENTRY_ONLY_PASS', pins['source'])
        return 0
    assert all(getattr(args, field) for field in arguments)
    return main_probe(*(getattr(args, field).resolve() for field in arguments), pins, deadline)


if __name__ == '__main__':
    raise SystemExit(main())
