"""Validate the test-only current301 correction against its exact failed installed wheel."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from xml.etree import ElementTree as ET


HERE = Path(__file__).resolve().parent
SCOPE = 'current301-minimum-laplacian-fix-focus-v1'
PARENT = 'eddec69fa8bfb64811f465ae38368a70ce61add5'
FILE = 'tests/test_pyramid_generators_public.py'
BASE_PIN = 'ee126583d43735500e6ec1886b3e33da23365868bf9fb7c3dce6d1112b92d8fb'
FILES = ('source-binding.json','expected-nodes.json','pins.json','diagnostic_base.py','range_helper.py',
         'prior_diagnose.py','extract_candidate.py','requirements-runtime.txt','report-json.yml')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def base():
    path = HERE/'diagnostic_base.py'
    assert digest(path) == BASE_PIN
    return runpy.run_path(str(path))


def authorize(pins):
    assert sys.platform == 'linux' and sys.version_info[:3] == (3,11,17)
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize
    assert os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_RUN_ATTEMPT') == '1'
    assert os.environ.get('GITHUB_REPOSITORY') == 'bipxx/scikit-imagexx'
    go = json.loads(os.environ['ROOT_AUTHORIZATION_JSON'])
    assert (go['authorized_by'],go['scope'],go['carrier']) == ('/root',SCOPE,os.environ['GITHUB_SHA'])
    bind = read(HERE/'source-binding.json')
    assert (bind['parent'],pins['source'],go['parent']) == (PARENT,PARENT,PARENT)
    assert (go['source'],go['tree'],go['test_sha256']) == (bind['source'],bind['tree'],bind['test_sha256'])
    assert go['artifact'] == pins['artifact'] and go['node_count_per_mode'] == 43
    assert (go['modes'],go['live_case_order']) == (['normal','scalar'],list(pins['cases']))
    assert go['launcher_sha256'] == digest(Path(__file__))
    assert go['files'] == {name:digest(HERE/name) for name in FILES}
    deadline = datetime.fromisoformat(go['deadline_utc'])
    assert deadline.tzinfo is not None and deadline.utcoffset().total_seconds() == 0
    assert datetime.now(timezone.utc) < deadline
    return bind,deadline


def prove_source(source, pins, bind):
    command = ['git','--no-optional-locks','-C',str(source)]
    def git(*args):
        return subprocess.check_output([*command,*args])
    assert git('rev-parse','HEAD','HEAD^{tree}','HEAD^').decode().splitlines() == [
        bind['source'],bind['tree'],PARENT]
    assert git('diff','--name-only',PARENT,bind['source']).decode().splitlines() == [FILE]
    assert not git('status','--porcelain=v1','--untracked-files=all')
    for name,pin in {**{n:r['git_raw_sha256'] for n,r in pins['source_paths'].items()},FILE:bind['test_sha256']}.items():
        raw = git('show','HEAD:'+name)
        assert sha256(raw).hexdigest() == pin, name
        assert (source/name).read_bytes().replace(b'\r\n',b'\n') == raw.replace(b'\r\n',b'\n'), name


def validate(args, source, bind, deadline, pins, nodes, helpers):
    mode = args.mode
    assert len(os.sched_getaffinity(0)) == 1 and all(os.environ.get(key) == '1' for key in helpers['POOL'])
    assert os.environ.get('SKIMAGEXX_DISABLE_SIMD') == (None if mode == 'normal' else '1')
    assert os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
    assert not any(key in os.environ for key in ('GH_TOKEN','PYTHONPATH','PYTHONHOME','NPY_DISABLE_CPU_FEATURES',
                                                'LD_PRELOAD','PYTEST_ADDOPTS','PYTEST_PLUGINS'))
    args.stock,args.candidate,args.extraction,args.output,args.temporary = (
        getattr(args,name).resolve(strict=True) for name in ('stock','candidate','extraction','output','temporary'))
    ext = helpers['prove_product'](args,pins)
    records = helpers['prove_stock'](args,pins)
    sys.path[:0] = [str(args.candidate),str(args.stock),str(source/'tests')]
    os.chdir(source)
    import numpy as np
    from numpy.core import _multiarray_umath
    import pytest
    from scipy import ndimage
    from scipy.ndimage import _nd_image, _filters, _interpolation, _ni_support
    from skimage import transform as original
    from skimage.transform import pyramids, _warps
    from skimage._shared import filters as original_filters
    from skimage.util import dtype
    import skimagexx as xx
    from skimagexx import _core, _pyramid_generators

    modules = (_multiarray_umath,_filters,_nd_image,_interpolation,_ni_support,pyramids,_warps,original_filters,dtype)
    assert {Path(m.__file__).resolve().relative_to(args.stock).as_posix() for m in modules} == set(pins['stock_members'])
    stock, candidate = original.pyramid_laplacian, xx.transform.pyramid_laplacian
    assert stock is pyramids.pyramid_laplacian and candidate is _pyramid_generators.pyramid_laplacian
    assert pyramids.gaussian is original_filters.gaussian and original_filters.ndi is ndimage and _filters._nd_image is _nd_image
    assert inspect.signature(stock) == inspect.signature(candidate)
    entries = (_core.gaussian_nd,_core.resize_zoom_typed)
    assert all(inspect.isbuiltin(fn) and fn.__module__ == 'skimagexx._core' for fn in entries)
    assert Path(_core.__file__).resolve() == args.candidate/ext['core_member'] and xx.get_num_threads() == 1
    assert Path(_pyramid_generators.__file__).resolve() == args.candidate/'skimagexx/_pyramid_generators.py'
    cpu = runpy.run_path(str(HERE/'prior_diagnose.py'))['cpu_state'](_multiarray_umath)
    codes = {stock.__code__:'stock',candidate.__code__:'candidate'}
    targets = {f'{FILE}::test_frozen_pyramid_generator[{case}]':case for case in pins['cases']}
    target_file = source/FILE
    captured = {}

    def snapshot(array):
        return dict(sha256=sha256(array.tobytes()).hexdigest(),shape=list(array.shape),strides=list(array.strides),
                    dtype=array.dtype.str,type=type(array).__qualname__,owns=bool(array.flags.owndata),
                    base=type(array.base).__name__,writable=bool(array.flags.writeable))

    class Observer:
        selected = []
        normal_conftest = False

        def pytest_collection_finish(self, session):
            self.selected = [item.nodeid for item in session.items]
            assert self.selected == nodes
            self.normal_conftest = any(Path(name).resolve() == source/'tests/conftest.py'
                                       for obj in session.config.pluginmanager.get_plugins()
                                       if (name := getattr(obj,'__file__',None)))
            assert self.normal_conftest

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            if item.nodeid not in targets:
                yield
                return
            rows = captured[targets[item.nodeid]] = {'stock':[],'candidate':[]}
            live_frames = {}
            def observe(frame,event,value):
                route = codes.get(frame.f_code)
                if route is None:
                    return None
                if event == 'call' and frame not in live_frames:
                    site = frame.f_back
                    while site is not None and Path(site.f_code.co_filename).resolve() != target_file:
                        site = site.f_back
                    assert site is not None and site.f_code.co_name == '_check_generator_trace', route
                    assert site.f_locals['stock'] is (route == 'stock'), route
                    assert not xx._compat._fallback_enabled
                    row = dict(input=snapshot(frame.f_locals['image']),caller=site.f_code.co_name,yields=[])
                    rows[route].append(row)
                    live_frames[frame] = row
                if event == 'return' and type(value) is np.ndarray:
                    live_frames[frame]['yields'].append(snapshot(value))
                return observe
            before = sys.gettrace()
            try:
                sys.settrace(observe)
                yield
            finally:
                sys.settrace(before)

    plugin = Observer()
    xml = args.output/f'{mode}.xml'
    argv = ['-q','--capture=sys',FILE,'--junitxml='+str(xml),'--basetemp='+str(args.temporary/(mode+'-tmp')),
            '-o','cache_dir='+str(args.temporary/(mode+'-cache'))]
    assert datetime.now(timezone.utc) < deadline and not xml.exists()
    status = int(pytest.main(argv,plugins=[plugin]))
    helpers['write'](args.output/f'{mode}-observed.json',dict(source=bind['source'],tree=bind['tree'],parent=PARENT,
        mode=mode,pid=os.getpid(),runtime=sys.version,executable=sys.executable,python_sha256=digest(Path(sys.executable)),
        test_sha256=digest(target_file),core_sha256=ext['core_sha256'],wheel_sha256=ext['wheel_sha256'],
        stock_record_sha256=records,normal_conftest=plugin.normal_conftest,node_ids=plugin.selected,pytest_exit=status,
        command=argv,cpu=cpu,xml_sha256=digest(xml) if xml.is_file() else None,
        live_routes=captured,worker_count=xx.get_num_threads(),timings=0))
    if status:
        return status
    entries = ET.parse(xml).findall('.//testcase')
    assert len(entries) == 43 and all(not any(tag.tag in {'error','failure','skipped'} for tag in row) for row in entries)
    assert [entry.get('name') for entry in entries] == [node.split('::',1)[1] for node in nodes]
    assert list(captured) == list(pins['cases'])
    for case,routes in captured.items():
        [stock_run], [candidate_run] = routes['stock'],routes['candidate']
        assert stock_run['input'] == candidate_run['input'] and stock_run['input']['sha256'] == pins['cases'][case]['frozen_input']
        assert len(stock_run['yields']) == len(candidate_run['yields']) == pins['cases'][case]['yield_count']
        assert stock_run['yields'] == candidate_run['yields'] and all(row['owns'] for row in stock_run['yields'])
    assert xx.get_num_threads() == 1 and datetime.now(timezone.utc) < deadline
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry-only',action='store_true')
    parser.add_argument('--bind-only',action='store_true')
    parser.add_argument('--mode',choices=('normal','scalar'))
    for key in ('source','stock','candidate','output','temporary','extraction'):
        parser.add_argument('--'+key,type=Path)
    args = parser.parse_args()
    helpers = base()
    pins = helpers['preview']()
    nodes = read(HERE/'expected-nodes.json')
    assert len(nodes) == len(set(nodes)) == 43 and all(node.startswith(FILE+'::') for node in nodes)
    if args.entry_only:
        print(json.dumps(dict(parent=PARENT,cases=list(pins['cases']),node_count=len(nodes),scientific_imports=0)))
        return 0
    if os.environ.get('ROOT_AUTHORIZATION_JSON') in (None,'UNBOUND'):
        print('REFUSED_NO_EXACT_ROOT_AUTHORITY',file=sys.stderr)
        return 2
    bind,deadline = authorize(pins)
    source = args.source.resolve(strict=True)
    prove_source(source,pins,bind)
    if args.bind_only:
        assert not args.mode
        return 0
    assert args.mode
    return validate(args,source,bind,deadline,pins,nodes,helpers)


if __name__ == '__main__':
    raise SystemExit(main())
