"""Source guards for the owning IGG family and saved-evidence boundary."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/inverse-gaussian-gradient-native-v2.json'
def contract():return json.loads(CONTRACT.read_bytes())
@pytest.mark.parametrize('name',[
'include/skimagexx/inverse_gaussian_gradient.hpp','src/inverse_gaussian_gradient.cpp','src/bind_inverse_gaussian_gradient.cpp','python/skimagexx/_inverse_gaussian_gradient.py','tests/fixtures/inverse_gaussian_gradient_original_v1.json','tests/fixtures/inverse_gaussian_gradient_sdk_v1.bin','tests/test_inverse_gaussian_gradient_public.py','tests/test_inverse_gaussian_gradient_sdk.cpp','tests/test_inverse_gaussian_gradient_native.cpp','tests/sdk/inverse_gaussian_gradient/CMakeLists.txt'])
def test_own_source_and_fixture_identity(name):
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==contract()['source_raw_sha256'][name]
@pytest.mark.parametrize('name',['tests/inverse_gaussian_gradient_contract.py','tests/inverse_gaussian_gradient_observer.py','tests/inverse_gaussian_gradient_capture_controller.py','coverage/contracts/inverse-gaussian-gradient-v1.json','benchmarks/inverse_gaussian_gradient.py'])
def test_frozen_source_first_and_codec_remain_immutable(name):
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==contract()['historical_raw_sha256'][name]
def test_all_admitted_records_and_profile_differences_preserved():
 data=json.loads((ROOT/'tests/fixtures/inverse_gaussian_gradient_original_v1.json').read_bytes())
 for profile in ('minimum','modern'):
  rows=data['profiles'][profile]['rows'];assert len(rows)==79 and len({r['id']for r in rows})==79
 assert contract()['raw_root_admission_sha256']=='be2e13844974951d6a760602ff83db452eb93413a95f2215ccfbce30a11d7f5b'
 assert contract()['accuracy']=='Exact raw values/types/layout/errors/warnings/target semantic FP and caller32 at authentic incoming states; no tolerance/masking'
def test_frontend_has_explicit_native_stages_and_no_original_forwarding():
 text=(ROOT/'python/skimagexx/_inverse_gaussian_gradient.py').read_text();tree=ast.parse(text)
 imports=[n.module for n in ast.walk(tree)if isinstance(n,ast.ImportFrom)]
 assert not any(n and n.startswith(('skimage.','scipy.'))for n in imports)
 assert '_core.inverse_gaussian_magnitude('in text and '_core.inverse_gaussian_finish('in text
 assert text.index('workers=_core.get_num_threads()')<text.index('plans=_plans(')
 function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='inverse_gaussian_gradient')
 assert [a.arg for a in function.args.args]==['image','alpha','sigma']
 assert not any(isinstance(n,ast.Attribute)and n.attr in ('gradient','gaussian_filter','generic_gradient_magnitude','correlate1d')for n in ast.walk(tree))
def test_complete_sdk_and_current_copy_composition_not_fragment():
 cpp=(ROOT/'src/inverse_gaussian_gradient.cpp').read_text();client=(ROOT/'tests/test_inverse_gaussian_gradient_sdk.cpp').read_text()
 assert 'crop_copy_into(image,axes,dense.data(),strides)'in cpp
 assert 'auto plans=coefficients(shape,sigma)'in cpp and 'stage<4' in cpp
 assert 'CropView'in client and 'inverse_gaussian_gradient'in client
 assert 'find_package(skimagexx CONFIG REQUIRED)'in (ROOT/'tests/sdk/inverse_gaussian_gradient/CMakeLists.txt').read_text()
 assert 'skimagexx::skimagexx'in (ROOT/'tests/sdk/inverse_gaussian_gradient/CMakeLists.txt').read_text()
def test_parallel_and_simd_preserve_order_with_real_disable_and_scalar_frontier():
 cpp=(ROOT/'src/inverse_gaussian_gradient.cpp').read_text()
 assert 'simd::hessian_convolve('in cpp and 'has_avx2()'in cpp
 assert 'input.size()>=detail::parallel_min?workers:1'in cpp
 assert 'feclearexcept(FE_ALL_EXCEPT)'not in cpp
 assert contract()['unbound_genuine']is True and contract()['original_remaining']==0
 assert len(contract()['pending'])>=6


def test_policy_source_successor_preserves_historical_bytes_and_operation_names():
 c=contract()
 for name,row in c['historical_sources'].items():
  raw=__import__('tools.join_source_context',fromlist=['igg_predecessor_bytes']).igg_predecessor_bytes(ROOT,name,(ROOT/name).read_bytes()).replace(b'\r\n',b'\n')
  assert hashlib.sha256(raw).hexdigest()==row['current_canonical_sha256']
  for fragment in row['inverse_fragments']:
   new,old=fragment['current'].encode(),fragment['predecessor'].encode()
   assert raw.count(new)==1
   raw=raw.replace(new,old,1)
  assert hashlib.sha256(raw).hexdigest()==row['predecessor_canonical_sha256']
 for name,pin in c['predecessor_contracts_raw_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin
 source=(ROOT/'python/skimagexx/_inverse_gaussian_gradient.py').read_text()
 assert "('multiply','add','sqrt','divide')[stage]"in source
 assert 'getattr(np,operation)(np.array(left,dtype=dtype),np.array(right,dtype=dtype))'in source
 assert c['policy_scope']['original_IGG_calls']==0 and c['policy_scope']['whole_function_new_domain_parity']=='PENDING'
