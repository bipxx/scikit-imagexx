"""Pure source/storage guards; no numerical/provider/native/helper bodies."""
import ast,builtins,hashlib,json,runpy,sys,types
from pathlib import Path
from math import prod
import pytest
from inverse_gaussian_gradient_contract import recipes,source_safe,fixture,reference_runtime,SIGNATURE
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/inverse-gaussian-gradient-v1.json'

def test_exact_source_snapshot_and_dependencies():
 d=json.loads(CONTRACT.read_bytes());snap=json.loads((ROOT/'tests/fixtures/inverse_gaussian_gradient_sources_v1.json').read_bytes())
 for row in snap['files'].values():assert hashlib.sha256(row['text'].encode()).hexdigest()==row['raw_sha256']
 for n,pin in d['source_raw_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==pin,n
 assert d['original_budget']==d['native_budget']==d['helper_budget']==0 and d['prior_full_algorithm_refs']==0

def test_actual_signature_and_final_expression_order():
 snap=json.loads((ROOT/'tests/fixtures/inverse_gaussian_gradient_sources_v1.json').read_bytes())['files']
 for profile in('minimum','modern'):
  s=snap[profile+'/skimage/segmentation/morphsnakes.py']['text'];n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='inverse_gaussian_gradient')
  assert [a.arg for a in n.args.args]==['image','alpha','sigma']and not n.args.kwonlyargs
  assert [ast.unparse(v)for v in n.args.defaults]==['100.0','5.0']
  call=n.body[-2].value;assert ast.unparse(call)=='ndi.gaussian_gradient_magnitude(image, sigma, mode=\'nearest\')'
  assert ast.unparse(n.body[-1].value)=='1.0 / np.sqrt(1.0 + alpha * gradnorm)'
 assert SIGNATURE==json.loads(CONTRACT.read_bytes())['signature']

def test_derivative_dtype_skip_threshold_and_compiled_line_bounds():
 snap=json.loads((ROOT/'tests/fixtures/inverse_gaussian_gradient_sources_v1.json').read_bytes())['files']
 for profile in('minimum','modern'):
  f=snap[profile+'/scipy/ndimage/_filters.py']['text'];c=snap[profile+'/scipy/ndimage/src/ni_filters.c']['text'];support=snap[profile+'/scipy/ndimage/src/ni_support.c']['text']
  assert 'order[axis] = 1'in f and 'if sigmas[ii] > 1e-15'in f
  assert 'np.multiply(output, output, output)'in f and 'output += tmp'in f and "np.sqrt(output, output, casting='unsafe')"in f
  assert '_get_output(output, input)'in f and 'weights = _gaussian_kernel1d'in f
  assert 'symmetric = -1'in c and '(iline[jj] - iline[-jj]) * fw[jj]'in c
  assert 'if (*lines > max_lines)'in support and '*lines = max_lines'in support
  assert 'NI_EXTEND_NEAREST'in support

@pytest.mark.parametrize('spec',recipes(),ids=lambda s:s['id'])
def test_frozen_factory_storage_cpu_broadcast_and_halo_caps(spec):
 b=source_safe(spec);assert b['input_elements']<=64 and b['output_elements_upper']<=128
 assert b['work_upper']<=65536 and b['named_numerical_bytes_upper']<=131072
 if spec['layout']=='stride2':assert b['physical_elements']==prod(2*s for s in spec['shape'])
 assert max(b['radii']or[0])<=20
 if spec['alpha_kind']=='leading':assert b['output_shape_upper']==[2,3,4]
 if spec['shape']==[0,3]:assert spec['sigma']==0

def test_active_radius_zero_is_not_filter_skip():
 small=next(x for x in recipes()if x['sigma']==1.0000000001e-15)
 skipped=next(x for x in recipes()if x['id']=='sigma-0')
 assert source_safe(small)['radii']==source_safe(skipped)['radii']==[0,0]
 assert source_safe(small)['filter_ops_upper']>0 and source_safe(skipped)['filter_ops_upper']==0

@pytest.mark.parametrize('cached',(False,True))
def test_actual_factory_request_refused_even_cached(cached,monkeypatch):
 seen=[];original=builtins.__import__
 def refuse(name,*a,**kw):
  if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):seen.append(name);raise RuntimeError('IGG_FACTORY_REQUEST_REFUSED')
  return original(name,*a,**kw)
 if cached:monkeypatch.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
 monkeypatch.setattr(builtins,'__import__',refuse)
 with pytest.raises(RuntimeError,match='REQUEST_REFUSED'):fixture(recipes()[0])
 assert seen==['numpy']

def test_unfrozen_intake_and_reference_authority_refuse_before_import():
 with pytest.raises(AssertionError,match='RECIPE_NOT_FROZEN'):fixture(dict(shape=[9999999]))
 with pytest.raises(SystemExit,match='PROSPECTIVE_UNBOUND'):reference_runtime()

def test_exact_typed_codec_and_observer_inverses():
 d=json.loads(CONTRACT.read_bytes());raw=(ROOT/'tests/inverse_gaussian_gradient_observer.py').read_text()
 for fragment in d['observer_inverse']:
  assert raw.count(fragment['current'])==1;raw=raw.replace(fragment['current'],fragment['predecessor'],1)
 assert hashlib.sha256(raw.encode()).hexdigest()==d['observer_predecessor_canonical_sha256']
 assert hashlib.sha256((ROOT/'tests/inverse_gaussian_gradient_exception_codec.py').read_bytes()).hexdigest()==d['codec_exact_raw_sha256']

def test_prospective_benchmark_zero_authority_plan(monkeypatch,capsys):
 p=ROOT/'benchmarks/inverse_gaussian_gradient.py';monkeypatch.setattr(sys,'argv',['IGG'])
 with pytest.raises(SystemExit,match='IGG_UNBOUND'):runpy.run_path(str(p),run_name='__main__')
 monkeypatch.setattr(sys,'argv',['IGG','--plan']);runpy.run_path(str(p),run_name='__main__');d=json.loads(capsys.readouterr().out)
 assert len(d['cases'])==8 and all(v is None for v in d['unbound'].values())
 assert(d['method']['threads'],d['method']['repeat'],d['method']['target_ms'],d['method']['warmup_min_ms'])==(1,7,100,250)
