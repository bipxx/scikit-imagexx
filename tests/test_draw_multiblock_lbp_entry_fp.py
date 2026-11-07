"""Entry-only LBP correction: source/inert; native validation stays pending."""
import hashlib,json,runpy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
H=lambda raw:hashlib.sha256(raw).hexdigest()

def test_exact_entry_only_inverse_and_no_postprocessor():
 c=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-entry-fp-v1.json').read_bytes());row=c['cpp'];raw=(ROOT/row['path']).read_bytes()
 assert H(raw)==row['current_sha256'];f=row['inverse'];cur=f['current'].encode();old=f['predecessor'].encode()
 assert raw.count(cur)==1;previous=raw.replace(cur,old,1);assert H(previous)==row['predecessor_sha256']
 assert b'if(std::fetestexcept(errors))std::feclearexcept(errors);'in cur
 assert b'FE_INEXACT'not in cur and b'fesetenv'not in raw and b'reference_fenv_restore'not in raw
 assert raw.index(cur)<raw.index(b'std::size_t k=0;',raw.index(b'unsigned multiblock_lbp_binary('))
 assert c['state']=='SOURCE_ONLY_NATIVE_VALIDATION_PENDING'and c['original_calls']==0

@pytest.mark.parametrize('fault',['changed','already-old'])
def test_strict_current_source_refuses_other_bytes(fault):
 c=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-entry-fp-v1.json').read_bytes())['cpp'];raw=(ROOT/c['path']).read_bytes();f=c['inverse']
 raw=raw+b'\n'if fault=='changed'else raw.replace(f['current'].encode(),f['predecessor'].encode(),1)
 assert H(raw)!=c['current_sha256']

def test_active_context_exact_old_projection_and_unchanged_cases():
 a=runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp_execution.py'));c=json.loads((ROOT/a['CONTEXT']).read_bytes())
 old=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-benchmark-execution-v5.json').read_bytes())
 assert c['cases']==old['cases']and c['bounds']==old['bounds']and c['method']==old['method']and c['unbound']==old['unbound']
 for name,row in c['historical_sources'].items():assert H(a['historical_bytes'](ROOT,name))==row['predecessor_canonical_sha256'],name
 for name,pin in c['source_files_canonical_sha256'].items():assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin,name

def test_pending_native_regressions_cover_noerror_and_active_states():
 s=(ROOT/'tests/test_draw_multiblock_lbp_entry_fp.cpp').read_text()
 assert 'for(unsigned profile=0;profile<2;++profile)for(bool active:{false,true})for(bool plus:{false,true})'in s
 for token in ['1058','1087','8127','8098','require(save()==b)','restore(caller)','restore(process)','multiblock_lbp_binary(av,bv,ov,plus)','std::memcmp(out.data(),expected.data(),16)','std::memcmp(a.data(),aw.data(),16)']:assert token in s
 assert 'fesetenv'not in s and 'feclearexcept'not in s
