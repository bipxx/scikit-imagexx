"""Pure readiness authority/source checks, normal conftest and plugins."""
import ast
import builtins
import hashlib
import json
from pathlib import Path
import runpy
import sys
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/inverse-gaussian-gradient-readiness-v1.json'


def test_exact_frozen_pool_bounds_and_zero_authority():
    from inverse_gaussian_gradient_contract import recipes,source_safe
    q=json.loads(CONTEXT.read_bytes());specs=recipes()
    assert specs==q['cases']and len(specs)==q['case_count']==79
    assert [source_safe(s)for s in specs]==q['bounds']
    for name,pin in q['frozen_pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    assert q['prospective_calls_per_profile']==79 and q['prospective_total_calls']==158 and q['calls_per_recipe']==1
    assert all(q[n]==0 for n in('original_budget','native_image_budget','sampler_budget','observation_budget'))
    assert q['root_original_GO']is q['root_readiness']is None
    assert max(b['physical_elements']for b in q['bounds'])==60


def test_exact_full32_durable_writers_and_failure_cleanup():
    q=json.loads(CONTEXT.read_bytes());raw=(ROOT/'tests/inverse_gaussian_gradient_capture_controller.py').read_text();tree=ast.parse(raw)
    for name,source in q['controller_reuse']['verbatim_source_blocks'].items():
        node=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==name)
        assert ast.get_source_segment(raw,node)==source
    assert raw.index('immediate_calls=[];factory_controls=[];observer_success_controls=[];np=None')<raw.index('sys.settrace(trace);sys.setprofile(profiler)')
    assert "last_case_diagnostics['failure_writer_error']"in raw and "event='failed_case_diagnostics'"in raw
    assert "assert bytes(actual)==bytes(saved),'EXACT32_RESTORE_FAILED'"in raw
    assert "public_attempts==call_count==len(immediate_calls)==1"in raw
    assert "assert len(capture_rows)==total_public==79"in raw and 'full_recipe_pool_count=79'in raw
    assert raw.count("append(journal,dict(event='started'")==raw.count("append(journal,dict(event='completed'")==1


@pytest.mark.parametrize('cached',[False,True])
def test_actual_go_entry_refuses_before_cold_or_cached_science(cached,monkeypatch):
    old=builtins.__import__;attempts=[]
    def refused(name,*a,**kw):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):
            attempts.append(name);raise AssertionError('Science request forbidden')
        return old(name,*a,**kw)
    with monkeypatch.context()as scope:
        if cached:
            for name in('numpy','scipy','skimage','skimagexx'):scope.setitem(sys.modules,name,types.ModuleType(name))
        scope.setattr(builtins,'__import__',refused)
        module=runpy.run_path(str(ROOT/'tests/inverse_gaussian_gradient_capture_controller.py'),run_name='metadata_only')
        with pytest.raises(AssertionError,match='NO_ROOT_GO'):
            module['validate_go'](types.SimpleNamespace(root_go=None,root_go_sha256=None),{})
    assert attempts==[]


@pytest.mark.parametrize('fault',['budget','whitelist'])
def test_actual_cli_zero_body_metadata_refusal(fault,monkeypatch,tmp_path):
    ids=[s['id']for s in json.loads(CONTEXT.read_bytes())['cases']]
    q=dict(schema='inverse-gaussian-gradient-capture-v1',status='UNBOUND_CAPTURE_CAPABLE_ZERO_BUDGET',
           original_budget=0,native_image_budget=0,sampler_budget=0,observation_budget=0,root_original_GO=None,
           case_ids=ids,api_case_ids={'inverse_gaussian_gradient':ids},profile_case_ids={'minimum':ids,'modern':ids})
    if fault=='budget':q['original_budget']=79
    else:q['profile_case_ids']['minimum']=ids[:-1]
    binding=tmp_path/'binding.json';binding.write_text(json.dumps(q),encoding='utf8',newline='\n')
    output=tmp_path/'output';argv=['inverse_gaussian_gradient_capture_controller.py','--profile','minimum','--prefix-api','inverse_gaussian_gradient','--binding',str(binding),'--binding-sha256',hashlib.sha256(binding.read_bytes()).hexdigest(),'--output',str(output),'--expect-prerequisite-refusal']
    old=builtins.__import__;attempts=[]
    def refused(name,*a,**kw):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):
            attempts.append(name);raise AssertionError('Science request forbidden')
        return old(name,*a,**kw)
    with monkeypatch.context()as scope:
        scope.setattr(sys,'argv',argv);scope.setattr(builtins,'__import__',refused)
        runpy.run_path(str(ROOT/'tests/inverse_gaussian_gradient_capture_controller.py'),run_name='__main__')
    terminal=json.loads((output/'terminal.json').read_bytes())
    assert terminal['status']=='PASS_EXPECTED_PREREQUISITE_REFUSAL'and terminal['scientific_imports']==terminal['target_body_lines']==terminal['original_budget']==terminal['recipe_observations']==0
    assert not terminal['journal_created']and attempts==[]


def test_python_provider_and_compiled_request_guards_are_explicit():
 raw=(ROOT/'tests/inverse_gaussian_gradient_capture_controller.py').read_text()
 for text in ("frame.f_code.co_name=='inverse_gaussian_gradient'","gaussian_gradient_magnitude","generic_gradient_magnitude","gaussian_filter1d","correlate1d","_gaussian_kernel1d","finally:cy.__class__=original_class","len(p['controls'])==10","provider_refused']==6","compiled_provider_refused']==1","maximum_public_topcalls_per_profile']==79","maximum_total_public_topcalls']==158"):
  assert text in raw,text
 assert 'for spec in specs:'in raw and 'for fixed in specs:'in raw
 factory=(ROOT/'tests/inverse_gaussian_gradient_readiness.py').read_text()
 assert 'events==[]'in factory and 'word_bytes_sha256'in factory
 assert "equal(sigma.tolist(),literal(spec['sigma']))"in factory and "for x in value.flat:equal(x.item(),literal(spec['alpha']))"in factory
 assert not any(text in factory for text in ('import skimage','gaussian_filter(','seterr(','allclose'))

@pytest.mark.parametrize('writer',('dump','append'))
@pytest.mark.parametrize('value',({'bad':complex(1,2)},{'bad':float('nan')}))
def test_actual_writers_validate_before_open_or_append(writer,value,tmp_path):
 module=runpy.run_path(str(ROOT/'tests/inverse_gaussian_gradient_capture_controller.py'),run_name='inert')
 path=tmp_path/'new.json'
 with pytest.raises((TypeError,ValueError)):module[writer](path,value)
 assert not path.exists()
 path.write_text('retained partial bytes')
 with pytest.raises((TypeError,ValueError)):module[writer](path,value)
 assert path.read_text()=='retained partial bytes'


def test_frozen_observer_keeps_typed_errors_raw_state_and_one_supplied_call():
    raw=(ROOT/'tests/inverse_gaussian_gradient_observer.py').read_text()
    assert 'from inverse_gaussian_gradient_exception_codec import encode_exception_args as error_value'in raw
    assert 'result=call(*args,**options)'in raw and 'qualname=type(e).__qualname__'in raw
    assert "numpy_before=dict(policy=policy" in raw and "numpy_after=dict(policy=np.geterr()"in raw
    assert 'supplied_call_attempts=1'in raw
    assert not any(s in raw for s in('allclose','nan_to_num','settrace','importlib','seterr('))
