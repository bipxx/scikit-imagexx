"""Verify all 40 exact successor pyramid cases against the retained wheel."""

import argparse
from base64 import urlsafe_b64encode
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
SOURCE = '94522169d6111a6dcf5f7b044d6d267773e27de6'
TREE = '1939643448b968ddc0e14ed61b7d42e1c4e9fd3e'
PARENT = '136b0d16f85f56f3c479e918ac16bdab486c36b1'
FILE = 'tests/test_pyramid_spatial_public.py'
TEST_PIN = '5a62acc59e31cc1e5ad975302784ea928c8a0d5b88b8069f2084e2e96e1d7ee6'
SCOPE = 'pyramid-minimum-reference-validation-v1'
TARGET = FILE + '::test_frozen_pyramid_stages[expand-rgb-u8-reflect]'
PINS = {'diagnose.py': 'b6bdfd13d2f2d15dbec33f41d544a083fdec2722456c02d95cbfe55ca94a1b0b',
        'extract_candidate.py': '8131cc590cfc9e1acfb0e04af56de8e232faa2df829f6f0e7b3bfb139b82d8b9',
        'range_helper.py': '1a54751980e8d5c06e31c46891f5a6112b5a6f64a07f494497fe93b98e912475',
        'pins.json': '1e40efe17bf4faf2a3659955f668c15789ac0bb7765e4b0c2ac6dd0ab6192334',
        'requirements-runtime.txt': 'ad18bd86522b3ee91723952d696fa71285caf236f6c514b12cee45362e2198b5',
        'expected-nodes.json': '8d14433ce8e485200a23397c35c814aed956e65dcda54885b3fd81b4b73e482e'}
POOLS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
         'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_THREAD_LIMIT')


def digest(raw):
    return sha256(raw).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def preview():
    assert all(sha(HERE / name) == pin for name, pin in PINS.items())
    nodes = read(HERE / 'expected-nodes.json')
    assert len(nodes) == len(set(nodes)) == 40 and nodes.count(TARGET) == 1
    assert all(node.startswith(FILE + '::') for node in nodes)
    assert not {'numpy', 'scipy', 'skimage', 'skimagexx', 'pytest'}.intersection(sys.modules)
    return nodes, read(HERE / 'pins.json')


def prove_source(source, pins):
    assert sys.platform == 'linux' and sys.version_info[:3] == (3, 11, 17)
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert os.environ['GITHUB_ACTIONS'] == 'true' and os.environ['GITHUB_RUN_ATTEMPT'] == '1'
    go = json.loads(os.environ['ROOT_AUTHORIZATION_JSON'])
    assert set(go) == {'scope', 'source', 'tree', 'parent', 'carrier', 'validation_sha256', 'test_sha256',
                       'node_count_per_mode', 'mode_order', 'live_references_per_mode', 'deadline_utc'}
    assert [go[k] for k in ('scope', 'source', 'tree', 'parent', 'carrier', 'validation_sha256', 'test_sha256')] == [
        SCOPE, SOURCE, TREE, PARENT, os.environ['GITHUB_SHA'], sha(Path(__file__)), TEST_PIN]
    assert [go[k] for k in ('node_count_per_mode', 'mode_order', 'live_references_per_mode')] == [40, ['normal', 'scalar'], 1]
    deadline = datetime.fromisoformat(go['deadline_utc'])
    assert deadline.utcoffset() is not None and deadline.utcoffset().total_seconds() == 0
    assert datetime.now(timezone.utc) < deadline
    git = ['git', '--no-optional-locks', '-C', str(source)]

    def run(*args):
        return subprocess.check_output([*git, *args])

    assert run('rev-parse', 'HEAD', 'HEAD^{tree}', 'HEAD^').decode().splitlines() == [SOURCE, TREE, PARENT]
    assert run('diff', '--name-only', PARENT, SOURCE).decode().splitlines() == [FILE]
    assert not run('status', '--porcelain=v1', '--untracked-files=all')
    for name, pin in {FILE: TEST_PIN, 'pyproject.toml': '8bb611200035980a4df9e212d1d80c12ffa6890174fc851936976a51d6197866',
            'tests/conftest.py': '8ce1a762ad4e8a37843103656b83eb9f21c9a71722a379fcd7e778d68ab2d2d6',
            **{name: pin['git_raw_sha256'] for name, pin in pins['source_paths'].items()}}.items():
        assert digest(run('show', 'HEAD:' + name)) == pin
        assert (source / name).read_bytes().replace(b'\r\n', b'\n') == run('show', 'HEAD:' + name).replace(b'\r\n', b'\n'), name
    return deadline


def prove_install(stock, product, extraction, pins):
    ext = read(extraction)
    assert (ext['source'], ext['artifact'], ext['core_sha256'], ext['wheel_sha256'], ext['members']) == (
        PARENT, pins['artifact'], pins['candidate_core_sha256'], pins['candidate_wheel']['sha256'], 265)
    assert sha(product / ext['record_path']) == ext['record_sha256']
    assert sha(product / ext['core_member']) == ext['core_sha256']
    for name in ('filters.py', '_pyramid_spatial.py'):
        assert sha(product / 'skimagexx' / name) == pins['source_paths']['python/skimagexx/' + name]['git_raw_sha256']
    distributions = {d.metadata['Name'].lower().replace('_', '-'): d for d in metadata.distributions(path=[str(stock)])}
    assert {name: distributions[name].version for name in pins['stock_versions']} == pins['stock_versions']
    records = {}
    for name, pin in pins['stock_members'].items():
        root = name.split('/')[0]
        label = 'scikit-image' if root == 'skimage' else root
        dist = distributions[label]
        if label not in records:
            path = Path(dist.locate_file(next(f for f in dist.files if f.name == 'RECORD')))
            with path.open(newline='') as stream:
                records[label] = {row[0]: row[1:] for row in csv.reader(stream)}
        p = stock / name
        code = urlsafe_b64encode(bytes.fromhex(pin['sha256'])).decode().rstrip('=')
        assert Path(dist.locate_file(name)) == p and p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256']
        assert records[label][name] == ['sha256=' + code, str(pin['bytes'])]
    return ext


def validate(args, nodes, pins, deadline):
    source, stock, candidate, output, temp = (getattr(args, key).resolve() for key in ('source', 'stock', 'candidate', 'output', 'temporary'))
    mode = args.mode
    assert os.environ.get('SKIMAGEXX_DISABLE_SIMD') == ('1' if mode == 'scalar' else None)
    assert len(os.sched_getaffinity(0)) == 1 and all(os.environ.get(name) == '1' for name in POOLS)
    assert os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
    assert not any(key in os.environ for key in ('GH_TOKEN', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'NPY_DISABLE_CPU_FEATURES',
                                                'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD'))
    ext = prove_install(stock, candidate, args.extraction.resolve(), pins)
    sys.path[:0] = [str(candidate), str(stock), str(source / 'tests')]
    os.chdir(source)
    import numpy as np
    from numpy.core import _multiarray_umath
    import pytest
    from scipy import ndimage
    from scipy.ndimage import _filters, _nd_image, _interpolation, _ni_support
    from skimage import transform as original
    from skimage.transform import pyramids, _warps
    from skimage._shared import filters as original_filters
    from skimage.util import dtype
    import skimagexx as xx
    from skimagexx import _core, _pyramid_spatial, filters, transform

    modules = (_multiarray_umath, _filters, _nd_image, _interpolation, _ni_support, pyramids, _warps, original_filters, dtype)
    assert {Path(m.__file__).resolve().relative_to(stock).as_posix() for m in modules} == set(pins['stock_members'])
    assert original.pyramid_expand is pyramids.pyramid_expand and pyramids.gaussian is original_filters.gaussian
    assert original_filters.ndi is ndimage and _filters._nd_image is _nd_image
    assert Path(_core.__file__) == candidate / ext['core_member'] and xx.get_num_threads() == 1
    assert Path(_pyramid_spatial.__file__) == candidate / 'skimagexx/_pyramid_spatial.py'
    assert Path(filters.__file__) == candidate / 'skimagexx/filters.py'
    assert transform.pyramid_expand is _pyramid_spatial.pyramid_expand and inspect.isbuiltin(_core.gaussian_nd)
    cpu = runpy.run_path(str(HERE / 'diagnose.py'))['cpu_state'](_multiarray_umath)
    reference = pyramids.pyramid_expand
    test_file = source / FILE
    ref_calls, stock_values, new_results = [], [], []

    def snapshot(value):
        return {'sha256': digest(value.tobytes()), 'shape': list(value.shape), 'strides': list(value.strides),
                'dtype': value.dtype.str, 'owned': value.flags.owndata, 'base': type(value.base).__name__}

    def trace(frame, event, value):
        if frame.f_code is reference.__code__:
            caller = frame.f_back
            assert caller.f_code.co_name == '_same_host_minimum_reference'
            assert Path(caller.f_code.co_filename).resolve() == test_file
            if event == 'call':
                ref_calls.append(digest(frame.f_locals['image'].tobytes()))
            if event == 'return' and type(value) is np.ndarray:
                stock_values.append(snapshot(value))
        if event == 'return' and frame.f_code.co_name == 'test_frozen_pyramid_stages' and Path(frame.f_code.co_filename).resolve() == test_file:
            result = frame.f_locals.get('result')
            if type(result) is np.ndarray:
                new_results.append(snapshot(result))

    class Observer:
        selected = []
        conftest = False

        def pytest_collection_finish(self, session):
            self.selected = [item.nodeid for item in session.items]
            assert self.selected == nodes
            self.conftest = any(Path(name).resolve() == source / 'tests/conftest.py'
                                for p in session.config.pluginmanager.get_plugins()
                                if (name := getattr(p, '__file__', None)))
            assert self.conftest

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            if item.nodeid != TARGET:
                yield
                return
            prior = sys.getprofile()
            try:
                sys.setprofile(trace)
                yield
            finally:
                sys.setprofile(prior)

    plugin = Observer()
    xml = output / (mode + '.xml')
    command = ['-q', '--capture=sys', FILE, '--junitxml=' + str(xml), '--basetemp=' + str(temp / (mode + '-tmp')),
               '-o', 'cache_dir=' + str(temp / (mode + '-cache'))]
    assert datetime.now(timezone.utc) < deadline and not xml.exists()
    status = int(pytest.main(command, plugins=[plugin]))
    obs = {'source': SOURCE, 'tree': TREE, 'mode': mode, 'pid': os.getpid(), 'runtime': sys.version,
           'arguments': command, 'pytest_exit': status, 'normal_conftest': plugin.conftest, 'node_ids': plugin.selected,
           'reference_call_inputs': ref_calls, 'stock': stock_values, 'candidate': new_results, 'cpu': cpu,
           'core_sha256': ext['core_sha256'], 'candidate_wheel_sha256': ext['wheel_sha256'], 'test_sha256': sha(test_file),
           'xml_sha256': sha(xml) if xml.is_file() else None, 'stock_callable': inspect.getsourcefile(reference),
           'native_entry': _core.gaussian_nd.__name__, 'worker_count': xx.get_num_threads(), 'timings': 0}
    write(output / (mode + '-observed.json'), obs)
    if status:
        return status
    cases = ET.parse(xml).findall('.//testcase')
    assert len(cases) == 40 and all(not any(x.tag in {'error', 'failure', 'skipped'} for x in case) for case in cases)
    assert [case.get('name') for case in cases] == [node.split('::', 1)[1] for node in nodes]
    assert ref_calls == [pins['frozen_input_sha256']] and len(stock_values) == 1
    assert stock_values == new_results and xx.get_num_threads() == 1
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entry-only', action='store_true')
    p.add_argument('--bind-only', action='store_true')
    p.add_argument('--mode', choices=('normal', 'scalar'))
    for name in ('source', 'stock', 'candidate', 'output', 'temporary', 'extraction'):
        p.add_argument('--' + name, type=Path)
    args = p.parse_args()
    nodes, pins = preview()
    if args.entry_only:
        assert not any((args.bind_only, args.mode, args.source, args.stock, args.candidate, args.output, args.temporary, args.extraction))
        print(json.dumps({'source': SOURCE, 'tree': TREE, 'test_sha256': TEST_PIN, 'nodes': len(nodes), 'scientific_imports': 0}))
        raise SystemExit(0)
    if not os.environ.get('ROOT_AUTHORIZATION_JSON') or os.environ['ROOT_AUTHORIZATION_JSON'] == 'UNBOUND':
        print('REFUSED_NO_EXACT_ROOT_AUTHORITY', file=sys.stderr)
        raise SystemExit(2)
    deadline = prove_source(args.source.resolve(), pins)
    if args.bind_only:
        assert not any((args.mode, args.stock, args.candidate, args.output, args.temporary, args.extraction))
        print(json.dumps({'source': SOURCE, 'tree': TREE, 'source_bound': True, 'no_science': True}))
        raise SystemExit(0)
    assert all((args.mode, args.stock, args.candidate, args.output, args.temporary, args.extraction))
    raise SystemExit(validate(args, nodes, pins, deadline))
