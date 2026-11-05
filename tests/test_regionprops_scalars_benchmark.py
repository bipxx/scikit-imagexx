"""Pure prospective intake only; no NumPy/original/native/timer calls."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]
CONTRACT=ROOT/'coverage/contracts/regionprops-scalars-benchmark-v1.json'


def load():
    spec=importlib.util.spec_from_file_location('prospective_region_scalar_benchmark',ROOT/'benchmarks/regionprops_scalars.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_scalar_benchmark_frozen_cases_and_method():
    m=load();c=json.loads(CONTRACT.read_bytes());cases=m.case_specs()
    assert len(cases)==8 and len({s['id']for s in cases})==8
    assert {s['operation']for s in cases}=={'constructor','getters','pixel-cycle','slice-cycle'}
    assert {s['layout']for s in cases}=={'C','F'}and{len(s['shape'])for s in cases}=={2,3}
    assert {s['cache']for s in cases}=={True,False}
    for s in cases:
        b=m.case_bounds(s);assert b['input_elements']<=c['max_input_elements'] and b['mask_elements']<=c['max_mask_elements']
    assert(c['threads'],c['repeat'],c['target_ms'],c['warmup_calibration_min_ms'],c['raw_samples'],c['warmup_groups'])==(1,7,100,250,112,16)
    assert c['properties']==list(m.PROPERTIES)==['area','area_bbox','extent']
    assert c['method_sha256']==json.loads((ROOT/'coverage/contracts/draw-coordinates-benchmark.json').read_bytes())['method_sha256']
    assert c['prior_CPP_controls']['source']=='eea321b2a334ba51c475d7733ad7547fa24c553b'
    assert c['prior_CPP_controls']['source_fingerprint']=='0123d8453eabc7263945c1de22cd337103d7e7d6ffbd2583b35e0a200d8cbdb4'


def test_scalar_benchmark_import_refuses_real_scientific_imports(monkeypatch):
    original=builtins.__import__
    def checked(name,*a,**kw):
        assert name.split('.')[0]not in{'numpy','scipy','skimage','skimagexx'}
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',checked)
    assert len(load().case_specs())==8


@pytest.mark.parametrize('case',json.loads(CONTRACT.read_bytes())['cases'],ids=lambda s:s['id'])
def test_scalar_benchmark_fixture_stops_before_science(case):
    class Refused:
        def __getattr__(self,name):raise RuntimeError('scientificfixture-refused')
    with pytest.raises(RuntimeError,match='^scientificfixture-refused$'):load().fixture(case,Refused())


def test_scalar_benchmark_rejects_changed_domain_before_fixture():
    m=load()
    for key,value in [('shape',[256,256]),('spacing',[0.,1.]),('max_label',3),('dtype','float32')]:
        case=copy.deepcopy(m.case_specs()[0]);case[key]=value
        with pytest.raises(AssertionError):m.case_bounds(case)
    case=copy.deepcopy(m.case_specs()[-1]);case['cache']=False
    with pytest.raises(AssertionError):m.case_bounds(case)


def test_scalar_benchmark_operation_boundaries_with_inert_objects():
    m=load()
    for spec in m.case_specs():
        events=[]
        class Image:
            shape=tuple(spec['shape']);ndim=len(shape)
            def __init__(self):self.pixel=1
            def __getitem__(self,key):return self.pixel
            def __setitem__(self,key,value):self.pixel=value;events.append(('pixel',value))
        class Region:
            @property
            def area(self):events.append('area');return 1
            @property
            def area_bbox(self):events.append('area_bbox');return 2
            @property
            def extent(self):events.append('extent');return .5
        def call(*a,**kw):events.append('constructor');return[Region(),Region()]
        image=Image();operation=m.make_operation(call,image,{},spec)
        if spec['operation']=='getters':
            assert events.count('constructor')==1
            assert events.count('area')==(2 if spec['cache']else 0)
            before=list(events);operation();assert events[len(before):]==['area','area_bbox','extent']*2
            operation();assert events.count('constructor')==1
        else:
            assert not events;payload=operation();assert events.count('constructor')==1
            if spec['operation']=='pixel-cycle':assert image.pixel==1 and('pixel',0)in events and('pixel',1)in events
            if spec['operation']=='slice-cycle':assert payload[0][0].slice==tuple(slice(1,n-1)for n in image.shape)


def test_scalar_benchmark_actual_guards_precede_unchanged_sampler():
    text=(ROOT/'benchmarks/regionprops_scalars.py').read_text();tree=ast.parse(text)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    assert "contract['state']=='FROZEN_READY_FOR_ROOT_FIRST_COHORT'"in text
    assert "(np,'sum',np.sum)"in text and "('image','area','area_bbox','extent')"in text
    assert "event=='c_return'"in text and "frame.f_locals['native_result']is not None"in text
    assert "scalar_class._extent(zero,zero)"in text and "events==[['invalid value',8]]"in text
    assert "sys.setprofile(previous)"in text and text.index('sys.setprofile(previous)')<text.index('timing=measure(')
    assert "assert actual['rows']==expected['rows']"in text
    assert 'not _compat._fallback_enabled' in text and 'fallback_enabled=False' in text
    assert "original_binary_sha256=binary_sha"in text and "binary_sha==contract['original_binary_sha256'][np.__version__]"in text
    calls=[n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure']
    assert len(calls)==1 and [ast.literal_eval(v)for v in calls[0].args[2:]]==[7,100]
    assert {'cpu_model','logical_cpu_count','avx2_available'}<={k.arg for n in ast.walk(main)if isinstance(n,ast.Call)for k in n.keywords}
    c=json.loads(CONTRACT.read_bytes());assert not c['pending_frontend_paths']
    assert c['state'].startswith('PROSPECTIVE_')or c['state']=='FROZEN_READY_FOR_ROOT_FIRST_COHORT'
    if c['state']=='FROZEN_READY_FOR_ROOT_FIRST_COHORT':
        for path,pin in c['source_closure'].items():
            assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    assert all(type(pin)is str and len(pin)==64 for pin in c['source_closure'].values())
    assert hashlib.sha256((ROOT/'benchmarks/regionprops_scalars.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()==c['source_closure']['benchmarks/regionprops_scalars.py']
    old=(ROOT/'benchmarks/regionprops_state.py').read_text()
    node=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)and n.name=='input_state')
    body=''.join(old.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    assert hashlib.sha256(body.encode()).hexdigest()==c['input_state_method_sha256']
