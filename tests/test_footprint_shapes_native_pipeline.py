"""Prospective actual-private-stage parity. No scientific calls on import."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE='b3c61fcc565c15860465eb374b395cd143d84414'
STAGES=('footprint_shapes_ellipse','footprint_shapes_hull','footprint_shapes_l1',
        'footprint_shapes_atoms','footprint_shapes_cross_plan','footprint_shapes_cross','footprint_shapes_dilate',
        'footprint_shapes_full_mask')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


@pytest.fixture(scope='session')
def runtime():
    assert sys.flags.optimize==0
    config=json.loads(Path(os.environ['FOOTPRINT_SHAPES_TEST_CONFIG']).read_bytes())
    assert config['schema']=='footprint-shapes-private-runtime-v1' and config['execution_authorized'] is True
    receipt=config['actual_link']
    assert sha(receipt['path'])==receipt['sha256']
    link=json.loads(Path(receipt['path']).read_bytes())
    assert link['schema']=='footprint-shapes-actual-link-v1'
    assert link['sole_worker_owner_count']==1 and link['new_kernel_count']==1
    assert any(Path(r['path']).resolve()==Path(config['core']['path']).resolve() and
               r['sha256']==config['core']['sha256'] for r in link['records'])
    for record in link['records']:
        assert Path(record['path']).stat().st_size==record['bytes'] and sha(record['path'])==record['sha256']
    for record in config['records']:
        assert Path(record['path']).stat().st_size==record['bytes'] and sha(record['path'])==record['sha256']
    assert sha(sys.executable)==config['python_sha256']
    core=load('_footprint_shapes_core',config['core']['path'])
    assert sha(core.__file__)==config['core']['sha256']
    assert core.footprint_shapes_interface=='footprint-shapes-v1' and core.get_num_threads()==1
    for name in STAGES:
        entry=getattr(core,name)
        import inspect
        assert inspect.isbuiltin(entry) and entry.__module__==core.__name__
    package=types.ModuleType('_footprint_shapes_candidate');package.__path__=[];package._core=core
    sys.modules[package.__name__]=package;sys.modules[package.__name__+'._core']=core
    shell=load(package.__name__+'._footprint_shapes',ROOT/'python/skimagexx/_footprint_shapes.py')
    import numpy as np
    import skimage.morphology as original
    import skimage.morphology.convex_hull as hull
    import skimage.morphology.gray as gray
    reference=load('frozen_footprint_reference',ROOT/'tests/footprint_shapes_reference.py')
    factory=load('frozen_footprint_factory',ROOT/'tests/footprint_shapes_cases.py')
    data_path=ROOT/'coverage/contracts/footprint-shapes.json'
    assert sha(data_path)=='2121657b2e5164414a492aa5a5f901c9a09a20ed5389ab44b39e441381460694'
    data=json.loads(data_path.read_bytes());assert data['capture_source']==SOURCE and data['case_specs']==factory.cases()
    # Replace ONLY the original-entry provenance predicate in the saved
    # observer, permitting this exact owned module. Every numerical snapshot,
    # warning/error/state/alias/rawword assertion remains the same code.
    tree=ast.parse((ROOT/'tests/footprint_shapes_reference.py').read_bytes())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observe')
    changed=0
    for node in fn.body:
        if isinstance(node,ast.If) and 'function.__module__.startswith' in ast.unparse(node.test):
            node.test=ast.copy_location(ast.Compare(left=ast.Attribute(value=ast.Name(id='function',ctx=ast.Load()),attr='__module__',ctx=ast.Load()),
                ops=[ast.NotEq()],comparators=[ast.Constant(value=shell.__name__)]),node.test);changed+=1
    assert changed==1
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<private-observer-entry-only>','exec'),reference.__dict__)
    profile='minimum' if np.__version__=='1.24.4' else 'windows' if os.name=='nt' else 'modern'
    assert np.__version__==config['numpy_version']
    pins={str(Path(r['path']).resolve()):r for r in config['runtime_records']}
    for name,module in tuple(sys.modules.items()):
        filename=getattr(module,'__file__',None)
        if filename and name.split('.')[0] in ('numpy','scipy','skimage'):
            path=str(Path(filename).resolve());assert path in pins and sha(path)==pins[path]['sha256']
    yield dict(config=config,core=core,shell=shell,np=np,original=original,hull=hull,gray=gray,
               reference=reference,factory=factory,expected=data['profiles'][profile]['observations'])
    core.set_num_threads(1)


@pytest.mark.parametrize('index',range(226))
@pytest.mark.parametrize('workers',(1,4))
def test_exact_frozen_shape_sequence(runtime,index,workers,tmp_path):
    core=runtime['core'];spec=runtime['factory'].cases()[index]
    if runtime['config']['portable'] and workers==4:
        with pytest.raises(ValueError):core.set_num_threads(4)
        (tmp_path/'portable-refusal.json').write_text(json.dumps(dict(id=spec['id'],workers=4,image_calls=0)))
        assert core.get_num_threads()==1;return
    core.set_num_threads(workers)
    targets=[(runtime['original'],name) for name in runtime['factory'].APIS]
    targets += [(runtime['hull'],'convex_hull_image'),(runtime['gray'],'dilation')]
    original=[(obj,name,getattr(obj,name)) for obj,name in targets]
    forwards=[];returns=[]
    def denied(*args,**kwargs):forwards.append(True);raise AssertionError('upstream algorithm forwarding')
    def profile(frame,event,value):
        if event=='c_return' and getattr(value,'__name__',None) in STAGES and getattr(value,'__module__',None)==core.__name__:
            returns.append(value.__name__)
    rows=[]
    with (tmp_path/'actual.jsonl').open('x') as stream:
        def journal(row):
            stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();rows.append(row)
        try:
            for obj,name,value in original:setattr(obj,name,denied)
            # Observer installs its own Python-entry profiler. Compose that
            # callback with the passive C-return witness, no stage replacement.
            make=runtime['reference'].entry_profiler
            def compose(code):
                counts,callback=make(code)
                def both(frame,event,value):callback(frame,event,value);profile(frame,event,value)
                return counts,both
            runtime['reference'].entry_profiler=compose
            actual=runtime['reference'].observe(spec,runtime['factory'],runtime['np'],runtime['shell'],journal)
        finally:
            runtime['reference'].entry_profiler=make
            for obj,name,value in original:setattr(obj,name,value)
            core.set_num_threads(1)
            (tmp_path/'stage-witness.json').write_text(json.dumps(dict(id=spec['id'],returns=returns,forward_attempts=len(forwards)))+'\n')
    assert not forwards
    assert actual==runtime['expected'][index]
    if actual['error'] is None:
        assert returns, 'successful row without a real compiled stage return'


def test_cross_plan_rejects_owned_unaligned_before_typed_read(runtime):
    np,core=runtime['np'],runtime['core']
    item=np.dtype(np.intp).itemsize
    owner=np.zeros(1+3*item,dtype=np.uint8)
    # Valid in-allocation view. Its zero bytes require no unaligned typed
    # writes to prepare; rejection must happen before the inherited C++read.
    sums=np.ndarray((3,),dtype=np.intp,buffer=owner,offset=1)
    before=owner.tobytes()
    assert sums.flags.c_contiguous and not sums.flags.aligned
    with pytest.raises(ValueError,match='cross sums must be aligned'):
        core.footprint_shapes_cross_plan(sums,1)
    assert owner.tobytes()==before


def test_metadata_check_accepts_owned_roots_views_and_empty_arrays(runtime):
    np,core=runtime['np'],runtime['core']
    owner=np.zeros(6,dtype=np.uint8)
    empty=np.zeros(0,dtype=np.uint8)
    arrays=(owner,owner[1:5],empty,owner[3:3])
    assert owner.flags.owndata and empty.flags.owndata
    assert arrays[1].base is owner and arrays[3].base is owner
    before=owner.tobytes()
    for array in arrays:
        metadata=(array.shape,array.strides,array.dtype,array.flags.owndata)
        assert core.footprint_shapes_check(array) is None
        assert (array.shape,array.strides,array.dtype,array.flags.owndata)==metadata
    assert owner.tobytes()==before
