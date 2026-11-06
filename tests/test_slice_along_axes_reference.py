"""Source-only signature/order/safety and actual setup-CLI import boundaries."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import pytest
from slice_along_axes_contract import API,SIGNATURE,SOURCE_SHA256,cases,source_safe,MAX_ELEMENTS,MAX_PHYSICAL_ELEMENTS
ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'coverage/contracts/slice-along-axes-v1.json').read_bytes())


def test_source_signature_order_and_no_invented_order_keyword():
    raw=(ROOT/DATA['upstream']['fixture']).read_bytes();assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==SOURCE_SHA256
    function=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef));assert function.name=='slice_along_axes'
    assert [a.arg for a in function.args.args]==['image','slices','axes','copy']
    assert [ast.literal_eval(v)for v in function.args.defaults]==[None,False]
    signature=inspect.Signature([inspect.Parameter('image',1),inspect.Parameter('slices',1),inspect.Parameter('axes',1,default=None),inspect.Parameter('copy',1,default=False)])
    assert str(signature)==SIGNATURE
    with pytest.raises(TypeError):signature.bind(None,[],order='F')
    text=raw.decode();need=['if not slices:','return image','if axes is None:','elif len(axes) != len(slices):','len(set(axes))','a >= 0 and a < image.ndim','for (a, b), ax in zip(slices, axes):','if a < 0:','a %= image.shape[ax]','if b < 0:','b %= image.shape[ax]','if a > b:','if a < 0 or b > image.shape[ax]:','_slices[ax] = slice(a, b)','image_slice = image[tuple(_slices)]','if copy and image_slice.base is not None:','image_slice = image_slice.copy()']
    positions=[text.index(s)for s in need];assert positions==sorted(positions)


def test_manifest_prospective_scope_and_complete_SDK_proposal():
    assert DATA['api']==API and DATA['signature']==SIGNATURE and DATA['cases']==cases()
    assert DATA['original_observations']=={} and DATA['source_safety']['original_calls']==0
    assert DATA['capture_plan']['current_authorized_original_calls']==0
    assert DATA['capture_plan']['proposed_cases_per_profile']==len(cases())
    assert DATA['capture_plan']['proposed_two_profile_total']==2*len(cases())
    assert len({r['id']for r in cases()})==len(cases())
    header=(ROOT/'include/skimagexx/slice_along_axes.hpp').read_text()
    # v1 remains the frozen historical proposal; current declaration is now
    # implemented by the admitted-authorized three-API family, not that proposal.
    assert 'SliceAlongAxesResult slice_along_axes('in header and 'std::vector<unsigned char> payload'in header
    for name in ('src/util_views.cpp','src/bind_util_views.cpp','python/skimagexx/_slice_along_axes.py'):assert(ROOT/name).is_file()
    assert DATA['SDK']['reused_geometry']['symbols']==['CropView','CropPlan','crop_view','crop_copy_into']


@pytest.mark.parametrize('row',cases(),ids=lambda r:r['id'])
def test_each_source_recipe_bounded_without_materialization(row):
    bound=source_safe(row)
    assert bound['logical_input_elements']<=MAX_ELEMENTS and bound['physical_input_elements']<=MAX_PHYSICAL_ELEMENTS
    assert bound['max_result_elements']<=bound['logical_input_elements']and bound['original_calls']==0


@pytest.mark.parametrize('fault',('extra','missing','shape-negative','shape-bool','rank','elements','dtype','forged-layout','protocol','copy-probe','bound-probe','axis-probe','literal-depth','large-bound','ragged-array','policy','mutation'))
def test_unreviewed_intake_rejected_before_science(fault,monkeypatch):
    import builtins
    old=builtins.__import__;attempts=[]
    def refused(name,*args,**kwargs):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):attempts.append(name);raise AssertionError('SCIENCE_IMPORT_FORBIDDEN')
        return old(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',refused);row=copy.deepcopy(cases()[0])
    if fault=='extra':row['order']='F'
    elif fault=='missing':row.pop('axes')
    elif fault=='shape-negative':row['shape']=[-1,5]
    elif fault=='shape-bool':row['shape']=[True,5]
    elif fault=='rank':row['shape']=[1]*5
    elif fault=='elements':row['shape']=[8,8,8]
    elif fault=='dtype':row['dtype']='external-record-objects'
    elif fault=='forged-layout':row['layout']='as_strided'
    elif fault=='protocol':row['input_kind']='arbitrary-callback'
    elif fault=='copy-probe':row['copy']={'bool_probe':1}
    elif fault=='bound-probe':row['slices']={'bound_probe':999,'failure':None}
    elif fault=='axis-probe':row['axes']={'axis_probe':False,'failure':None}
    elif fault=='literal-depth':
        row['slices']=0
        for _ in range(6):row['slices']={'container':'list','items':[row['slices']]}
    elif fault=='large-bound':row['slices']=2**71
    elif fault=='ragged-array':row['slices']={'array':'int64','items':[[1],[1,2]]}
    elif fault=='policy':row['policy']={'invalid':'ignore'}
    else:row['mutation']=True;row['shape']=[1]
    with pytest.raises(AssertionError):source_safe(row)
    assert attempts==[]


def _cli(tmp_path,mode):
    go=dict(status='ROOT_AUTHORIZED_SLICE_ALONG_AXES_REFERENCE_ONCE',api=API,contract_sha256=hashlib.sha256((ROOT/'coverage/contracts/slice-along-axes-v2.json').read_bytes()).hexdigest(),upstream_canonical_lf_sha256=SOURCE_SHA256,maximum_public_topcalls_per_profile=len(cases()),native_calls=0,timed_batches=0)
    path=tmp_path/'inert-test-binding.json';path.write_text(json.dumps(go))
    script=r'''
import builtins,json,runpy,sys,types
from pathlib import Path
tests,observer,binding,mode=sys.argv[1:];sys.path.insert(0,tests);attempts=[]
if mode=='cached':
 for name in ('numpy','scipy','skimage','skimagexx'):sys.modules[name]=types.ModuleType(name)
old=builtins.__import__
def refused(name,*args,**kwargs):
 if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):attempts.append(name);raise RuntimeError('SCIENCE_IMPORT_REQUEST_REFUSED')
 return old(name,*args,**kwargs)
builtins.__import__=refused
sys.argv=[observer]+([]if mode=='unbound'else['--root-go',binding])
try:runpy.run_path(observer,run_name='__main__')
except Exception as error:
 print(json.dumps(dict(error=type(error).__name__,message=str(error),attempts=attempts,original_calls=0)))
else:raise AssertionError('Setup unexpectedly passed scientific boundary')
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(ROOT/'tests'),str(ROOT/'tests/slice_along_axes_observer.py'),str(path),mode],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return json.loads(result.stdout)


def test_actual_setup_CLI_refuses_unbound_before_import(tmp_path):
    row=_cli(tmp_path,'unbound');assert row['error']=='AssertionError'and 'NO_ROOT_GO'in row['message']and row['attempts']==[]


@pytest.mark.parametrize('mode',('cold','cached'))
def test_actual_positive_shaped_setup_CLI_import_request_guard(tmp_path,mode):
    row=_cli(tmp_path,mode);assert row['error']=='RuntimeError'and row['message']=='SCIENCE_IMPORT_REQUEST_REFUSED'and row['attempts']==['numpy']


def test_prospective_benchmark_sampler_real_source():
    path,symbol=DATA['benchmark']['sampler'].split(':');assert (ROOT/path).is_file()
    method=next(n for n in ast.parse((ROOT/path).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name==symbol)
    assert [a.arg for a in method.args.args]==['reference','native','repeat','target_ms']
    assert DATA['benchmark']['state']=='PROSPECTIVE_UNBOUND_NO_MEASUREMENT'


def test_prospective_benchmark_recipes_bounded_and_unreviewed_refused():
    import runpy
    module=runpy.run_path(str(ROOT/'benchmarks/slice_along_axes.py'))
    rows=module['case_specs']();assert len(rows)==DATA['benchmark']['case_count']==8
    assert len({row['id']for row in rows})==len(rows)
    for row in rows:
        assert module['case_bounds'](row)['threads']==1
        changed=copy.deepcopy(row);changed['shape']=[2**40]
        with pytest.raises(AssertionError):module['case_bounds'](changed)
