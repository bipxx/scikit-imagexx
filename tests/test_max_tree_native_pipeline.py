"""Real standalone max-tree bridge prototype; not production/public admission.

351 exact source observations are preserved:233 numeric and118 metadata-only.
Observable providers exercise actual staged C++; separate pristine projections
prove the aggregate C++ sorting/traversal path. No upstream image algorithm is
available during candidate execution.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types
import warnings

import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
BACKEND=os.environ.get('SKIMAGEXX_MAX_TREE_CORE')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


contracts=load('_max_tree_original_contracts',ROOT/'tests/test_max_tree_contracts.py')
frozen=json.loads(contracts.MATRIX.read_text())
NUMERIC=tuple(case for case in contracts.CASES if case['mode']=='numeric')
MAP={'_max_tree':'build','_compute_area':'area','_compute_extension':'extension',
     '_direct_filter':'filter','_max_tree_local_maxima':'maxima'}


def expected(spec):
    return frozen['observations'][frozen['profiles'][contracts.profile()]['snapshots'][spec['id']]]


def record_observation(row):
    path=os.environ.get('SKIMAGEXX_MAX_TREE_OBSERVATIONS')
    if path:
        with Path(path).open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')


def fixture_without_upstream(spec,actual_provided_tree=False):
    # The literal original fixture's metadata construction is retained. For the
    # five real-tree reuse rows, materialize the frozen actual-source tree bytes
    # with the exact original ownership/layout; no original builder runs here.
    passed,image,owner,options,events=contracts.fixture(spec,False)
    if actual_provided_tree and spec.get('reuse')=='both' and spec['mode']=='numeric':
        for key,record in expected(spec)['provided_before'].items():
            value=np.empty(tuple(record['shape']),np.dtype(record['dtype']))
            value.reshape(-1)[...]=np.frombuffer(bytes.fromhex(record['bytes']),dtype=value.dtype)
            options[key]=value
    return passed,image,owner,options,events


@contextmanager
def forbid_original(monkeypatch):
    attempts=[]
    codes={getattr(contracts.original,name).__code__ for name in contracts.NAMES}
    def forbidden(*args,**kwargs):
        attempts.append('cython');raise AssertionError('upstream max-tree Cython execution forbidden')
    def profile(frame,event,arg):
        if event=='call' and frame.f_code in codes:
            attempts.append(frame.f_code.co_name)
            raise AssertionError('upstream max-tree Python execution forbidden')
    old=sys.getprofile()
    with monkeypatch.context() as local:
        for name in contracts.ENTRIES:local.setattr(contracts.original._max_tree,name,forbidden)
        sys.setprofile(profile)
        try:yield attempts
        finally:sys.setprofile(old)
    assert not attempts


@pytest.fixture(scope='module')
def implementation():
    if BACKEND is None:pytest.skip('explicit private max-tree bridge required; public admission pending')
    path=Path(BACKEND).resolve(strict=True)
    core=load('_max_tree_core',path)
    module=load('skimagexx._max_tree_private_source',ROOT/'python/skimagexx/_max_tree.py')
    module._core=core
    assert module._require_core() is core
    assert Path(core.__file__).resolve()==path and core.get_num_threads()==1
    if os.environ.get('SKIMAGEXX_EXPECT_GIL_ASSERTIONS')=='1':assert core.gil_assertions is True
    return module,core


@pytest.fixture(params=(1,4))
def workers(request,implementation):
    _,core=implementation;old=core.get_num_threads()
    if request.param==4 and os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')=='1':
        with pytest.raises(ValueError,match='no OpenMP'):core.set_num_threads(4)
        assert core.get_num_threads()==old==1
        pytest.skip('portable explicitly rejects4; no serial4 parity claim')
    core.set_num_threads(request.param)
    try:yield request.param
    finally:core.set_num_threads(old)


@pytest.mark.parametrize('spec',contracts.CASES,ids=lambda row:row['id'])
def test_all_source_observations_real_stages(spec,workers,implementation,monkeypatch):
    module,core=implementation;core.max_tree_counts(True)
    observer=types.FunctionType(contracts.observe.__code__,
        dict(contracts.observe.__globals__,original=module,fixture=fixture_without_upstream))
    with forbid_original(monkeypatch):actual=observer(spec)
    counts=core.max_tree_counts()
    record_observation({'scope':'source-stages','case':spec['id'],'profile':contracts.profile(),'workers':workers,
        'digest':contracts.digest(actual),'counts':counts,'actual_failure':actual if actual!=expected(spec) else None})
    assert actual==expected(spec),spec['id']
    assert counts['aggregate']=={'attempted':0,'completed':0}
    for source,stage in MAP.items():
        completed=sum(row['name']==source and 'after' in row for row in actual['entries'])
        assert counts[stage]=={'attempted':completed,'completed':completed}
    if spec['mode']=='numeric':assert sum(row['completed'] for row in counts.values())>0
    else:
        # Four protocol rows may complete safe preceding stages but are not
        # successful native function cases; the direct-filter boundary blocks.
        assert actual['error'] is not None


def outcome(result,image,options):
    def one(value):
        return {'type':type(value).__name__,'array':contracts.array(value),
                'shares_input':bool(np.shares_memory(value,image)),
                'base':contracts.array(value.base) if isinstance(value.base,np.ndarray) else None,
                'shares_provided':{key:bool(np.shares_memory(value,options[key]))
                    for key in ('parent','tree_traverser') if isinstance(options.get(key),np.ndarray)}}
    return [one(value) for value in result] if isinstance(result,tuple) else one(result)


@pytest.mark.parametrize('spec',NUMERIC,ids=lambda row:row['id'])
def test_pristine_real_aggregate_partition(spec,workers,implementation,monkeypatch):
    module,core=implementation
    passed,image,owner,options,_=fixture_without_upstream(spec,True)
    before=(contracts.array(image),contracts.array(owner))
    provided={key:contracts.array(value) for key,value in options.items() if key in ('parent','tree_traverser')}
    core.max_tree_counts(True)
    with forbid_original(monkeypatch),warnings.catch_warnings(record=True) as caught,np.errstate(all='warn'):
        warnings.simplefilter('always');result=getattr(module,spec['function'])(passed,**options)
    assert outcome(result,image,options)==expected(spec)['result']
    assert not caught and before==(contracts.array(image),contracts.array(owner))
    assert provided=={key:contracts.array(options[key]) for key in provided}
    counts=core.max_tree_counts()
    eligible=type(passed) is np.ndarray and passed.flags.writeable
    if eligible:
        assert counts['aggregate']=={'attempted':1,'completed':1}
        assert sum(row['completed'] for key,row in counts.items() if key!='aggregate')==0
    elif type(passed) is np.ndarray and spec['function'].endswith('closing'):
        # The readonly outer input takes the source route; its newly inverted
        # writable array admits a genuine aggregate TREE subcall. Attribute
        # and direct-filter operations remain actual C++ stages.
        assert counts['aggregate']=={'attempted':1,'completed':1}
        assert counts['filter']=={'attempted':1,'completed':1}
        assert counts['extension' if spec['function'].startswith('diameter') else 'area']['completed']==1
    else:
        assert counts['aggregate']=={'attempted':0,'completed':0}
        assert sum(row['completed'] for row in counts.values())>0


@pytest.mark.parametrize('fault',('version','types-float','ops-partial','stage-version','stage-bool','missing-entry','borrowed-entry','python-entry','parallel'))
def test_admission_before_input(fault,implementation,monkeypatch):
    module,core=implementation
    attrs={name:getattr(core,name) for name in ('__name__','max_tree_interface','max_tree_type_capabilities',
        'max_tree_operation_capabilities','max_tree_parallel_traversal','max_tree_stage_interface',
        'max_tree_stage_capabilities','max_tree_run')+module._STAGE_ENTRIES}
    if fault=='version':attrs['max_tree_interface']='wrong'
    elif fault=='types-float':attrs['max_tree_type_capabilities']=1023.
    elif fault=='ops-partial':attrs['max_tree_operation_capabilities']=1
    elif fault=='stage-version':attrs['max_tree_stage_interface']='wrong'
    elif fault=='stage-bool':attrs['max_tree_stage_capabilities']=True
    elif fault=='missing-entry':attrs.pop('max_tree_filter_stage')
    elif fault=='borrowed-entry':attrs['max_tree_filter_stage']=core.max_tree_area_stage
    elif fault=='python-entry':attrs['max_tree_filter_stage']=lambda *args:None
    else:attrs['max_tree_parallel_traversal']=True
    monkeypatch.setattr(module,'_core',types.SimpleNamespace(**attrs))
    class Unreadable:
        @property
        def shape(self):raise AssertionError('input accessed')
    for name in contracts.NAMES:
        with pytest.raises(NotImplementedError):getattr(module,name)(Unreadable())


def test_threshold_after_buffers_attempted_vs_completed(implementation):
    _,core=implementation
    image=np.arange(20,dtype='f4');out=image.copy();parent=np.zeros(20,dtype='i8');order=np.arange(20,dtype='i8');attribute=np.ones(20,dtype='f8')
    events=[]
    class Threshold:
        def __float__(self):events.append('threshold');attribute[...]=0.;return 3.
    core.max_tree_counts(True)
    # Buffer dtype failure occurs before scalar conversion.
    with pytest.raises(TypeError):core.max_tree_filter_stage(image,out,parent,order,attribute.astype('f4'),Threshold())
    assert not events and core.max_tree_counts()['filter']=={'attempted':1,'completed':0}
    core.max_tree_filter_stage(image,out,parent,order,attribute,Threshold())
    assert events==['threshold'] and np.all(out==0)
    assert core.max_tree_counts()['filter']=={'attempted':2,'completed':1}
    class Raised(RuntimeError):pass
    class Reject:
        def __float__(self):out[-1]=42.;raise Raised('threshold rejected')
    with pytest.raises(Raised,match='threshold rejected'):core.max_tree_filter_stage(image,out,parent,order,attribute,Reject())
    assert out[-1]==42. and core.max_tree_counts()['filter']=={'attempted':3,'completed':1}


@pytest.mark.parametrize('kind',('dtype','rank','readonly'))
@pytest.mark.parametrize('entry',('aggregate','filter'))
def test_scalar_metadata_mutations_rejected_before_native_access(kind,entry,implementation):
    _,core=implementation
    image=np.arange(20,dtype='f4').reshape(4,5) if entry=='aggregate' else np.arange(20,dtype='f4')
    output=image.copy();parent=np.zeros(20,dtype='i8');order=np.arange(20,dtype='i8');attribute=np.ones(20,dtype='f8')
    before=output.tobytes();core.max_tree_counts(True)
    class Mutation:
        def __float__(self):
            if kind=='dtype':image.dtype=np.dtype('i4')
            elif kind=='rank':image.shape=(20,) if entry=='aggregate' else (4,5)
            else:image.flags.writeable=False
            return 3.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError,match='changed during scalar conversion'):
            if entry=='aggregate':core.max_tree_run(image,output,parent,order,2,1,Mutation(),False)
            else:core.max_tree_filter_stage(image,output,parent,order,attribute,Mutation())
    expected_warnings=int(tuple(map(int,np.__version__.split('.')[:2]))>=(2,5) and kind in ('dtype','rank'))
    assert len(caught)==expected_warnings and all(issubclass(row.category,DeprecationWarning) for row in caught)
    assert output.tobytes()==before and core.max_tree_counts()[entry]=={'attempted':1,'completed':0}
    record_observation({'scope':'native-metadata-guard-not-upstream-parity','case':entry+'/'+kind,
        'profile':contracts.profile(),'warnings':[str(row.message) for row in caught],
        'output_unchanged':output.tobytes()==before,'counts':core.max_tree_counts()})


def test_reference_ownership_released_on_success_and_cpp_rejection(implementation):
    _,core=implementation
    image=np.arange(20,dtype='f4');out=image.copy();parent=np.zeros(20,dtype='i8');order=np.arange(20,dtype='i8');attribute=np.ones(20,dtype='f8')
    values=(image,out,parent,order,attribute);before=tuple(sys.getrefcount(value) for value in values)
    for _ in range(128):
        core.max_tree_filter_stage(image,out,parent,order,attribute,3.)
        parent[-1]=99
        with pytest.raises(ValueError):core.max_tree_filter_stage(image,out,parent,order,attribute,3.)
        parent[-1]=0
    assert tuple(sys.getrefcount(value) for value in values)==before


def callback_observation(module,name,action,monkeypatch,reference=False):
    """Source-safe bounded scalar hooks; no resizing or unsafe index mutation."""
    image=(np.arange(20,dtype='f4')%4).reshape(4,5)
    events=[];held={};error=None;result=None
    saved={key:getattr(module._max_tree,key) for key in contracts.ENTRIES}
    class HookError(RuntimeError):pass
    class Threshold:
        def __float__(self):
            events.append('threshold.float')
            if action=='attribute':held['attribute'][...]=0.
            else:held['out'][-1]=42.;raise HookError('threshold rejected after output capture')
            return 3.
    def wrap(key,function):
        def call(*args):
            if reference:
                # All actual array/index inputs satisfy the unchanged source
                # certificate. The only hook is known to return3 or raise;
                # it writes bounded allocated attribute/output elements only.
                certified=list(args)
                if key=='_direct_filter':certified[-1]=3.
                contracts.certify_entry(key,tuple(certified),image.shape)
            if key=='_direct_filter':held['out']=args[1];events.append('filter.entry')
            value=function(*args)
            if key in ('_compute_area','_compute_extension'):
                held['attribute']=value;events.append('attribute.done')
            if key=='_direct_filter':events.append('filter.done')
            return value
        return call
    options={'area_threshold' if name.startswith('area') else 'diameter_threshold':Threshold()}
    with monkeypatch.context() as local:
        for key,function in saved.items():local.setattr(module._max_tree,key,wrap(key,function))
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='warn'):
            warnings.simplefilter('always')
            try:result=getattr(module,name)(image,**options)
            except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
    return {'events':events,'error':error,'input':contracts.array(image),
        'attribute':contracts.array(held['attribute']) if 'attribute' in held else None,
        'out':contracts.array(held['out']) if 'out' in held else None,
        'result':contracts.array(result) if result is not None else None,
        'warnings':[{'type':type(row.message).__name__,'message':str(row.message)} for row in caught]}


@pytest.mark.parametrize('name',('area_opening','area_closing','diameter_opening','diameter_closing'))
@pytest.mark.parametrize('action',('attribute','raise'))
def test_actual_source_scalar_stage_order_and_current_attributes(name,action,workers,implementation,monkeypatch):
    module,core=implementation
    reference=callback_observation(contracts.original,name,action,monkeypatch,True)
    core.max_tree_counts(True)
    with forbid_original(monkeypatch):actual=callback_observation(module,name,action,monkeypatch)
    counts=core.max_tree_counts()
    record_observation({'scope':'bounded-scalar-provider','case':name+'/'+action,'profile':contracts.profile(),
        'workers':workers,'reference':reference,'actual':actual,'counts':counts})
    assert actual==reference
    assert counts['aggregate']=={'attempted':0,'completed':0}
    assert counts['filter']=={'attempted':1,'completed':int(action!='raise')}
    assert counts['extension' if name.startswith('diameter') else 'area']['completed']==1


@pytest.mark.parametrize('name',('area_closing','diameter_closing'))
def test_actual_inversion_provider_data_with_provided_tree(name,workers,implementation,monkeypatch):
    module,core=implementation
    spec=next(row for row in NUMERIC if row['function']==name and row.get('reuse')=='both')
    def run(owner,reference):
        passed,image,_,options,_=fixture_without_upstream(spec,True)
        source_subtract=np.subtract;calls=[]
        def subtract(*args,**kwargs):
            result=source_subtract(*args,**kwargs);calls.append(contracts.array(result))
            if len(calls)==1:
                # Positive monotone scaling preserves the frozen provided tree,
                # but changes actual inverted pixels consumed by the filter.
                # Use float input to keep all values finite and exact.
                result[...] = result*result.dtype.type(.5)
            return result
        image=image.astype('f4');passed=image
        saved={key:getattr(owner._max_tree,key) for key in contracts.ENTRIES}
        def guarded(key,function):
            def call(*args):
                if reference:contracts.certify_entry(key,args,image.shape)
                return function(*args)
            return call
        with monkeypatch.context() as local:
            local.setattr(np,'subtract',subtract)
            if reference:
                for key,function in saved.items():local.setattr(owner._max_tree,key,guarded(key,function))
            result=getattr(owner,name)(passed,**options)
        return {'result':contracts.array(result),'calls':calls,'parent':contracts.array(options['parent']),
                'traverser':contracts.array(options['tree_traverser'])}
    # The uint8 inversion tree is also the tree of1-float(image): both reverse
    # exactly the same finite plateau ordering, so the provided indices remain
    # source-certified. Positive scaling leaves that ordering unchanged.
    reference=run(contracts.original,True);core.max_tree_counts(True)
    with forbid_original(monkeypatch):actual=run(module,False)
    assert actual==reference
    assert len(actual['calls'])==2 and core.max_tree_counts()['filter']['completed']==1
    record_observation({'scope':'inversion-provider','case':name,'profile':contracts.profile(),'workers':workers,
        'reference':reference,'actual':actual,'counts':core.max_tree_counts()})


@pytest.mark.parametrize('name',contracts.NAMES[1:])
def test_provided_tree_ignores_connectivity(name,workers,implementation,monkeypatch):
    module,core=implementation
    spec=next(row for row in NUMERIC if row['function']==name and row.get('reuse')=='both')
    passed,image,_,options,_=fixture_without_upstream(spec,True)
    class Unused:
        def __index__(self):raise AssertionError('unused connectivity coerced')
        def __bool__(self):raise AssertionError('unused connectivity inspected')
    options['connectivity']=Unused();core.max_tree_counts(True)
    with forbid_original(monkeypatch):result=getattr(module,name)(passed,**options)
    assert outcome(result,image,options)==expected(spec)['result']
    assert core.max_tree_counts()['aggregate']=={'attempted':1,'completed':1}


@pytest.mark.parametrize('dtype',('u1','f4','f8'))
def test_large_actual_core_and_preparation_team(dtype,workers,implementation,monkeypatch):
    module,core=implementation
    image=np.zeros((513,519),dtype=dtype)
    core.max_tree_counts(True)
    with forbid_original(monkeypatch):result=module.area_closing(image,area_threshold=1)
    assert result.dtype==image.dtype and result.flags.owndata and not np.shares_memory(result,image)
    assert np.all(result==0) and core.max_tree_counts()['aggregate']=={'attempted':1,'completed':1}
    parent=np.empty(image.shape,dtype='i8');order=np.empty(image.size,dtype='i8');out=np.empty_like(image)
    info=core.max_tree_run(image,out,parent,order,3,1,1,False)
    assert info['preparation_workers']==workers and not info['tree_reused'] and np.all(out==0)


def test_signature_reference_and_frozen_kernel(implementation):
    module,_=implementation
    for name in contracts.NAMES:assert str(inspect.signature(getattr(module,name)))==frozen['signatures'][name]
    proof=json.loads((ROOT/'build/max-tree-native/manifest.json').read_text())
    for name,pin in proof['git_members'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin['sha256']
