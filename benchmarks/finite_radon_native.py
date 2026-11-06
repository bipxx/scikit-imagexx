"""Frozen prospective finite Radon measurement interface; execution unbound."""
import argparse,ast,hashlib,json,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def plan():
    cases=[]
    for api in('frt2','ifrt2'):
        cases+=runpy.run_path(str(ROOT/'benchmarks'/f'{api}.py'))['case_specs']()
    source=(ROOT/'tools/montage_sustained.py').read_bytes().replace(b'\r\n',b'\n')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    method=b''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    return dict(status='UNBOUND_NO_EXECUTION',cases=cases,threads=1,repeat=7,target_ms=100,warmup_min_ms=250,retain_all_losses=True,sampler='tools/montage_sustained.py:measure',sampler_source_sha256=hashlib.sha256(method).hexdigest(),root_GO=None,producer=None,runtime=None,core=None,SDK=None,native_acceptance={'frt2':'_core.frt2(a)','ifrt2':'_core.ifrt2(a,accumulator_bits,weak_scalar,signed_sum)'},pending='Genuine source/runtime/root binding, actual public owner/first-cohort/FP/input checks and root measurement GO; no timing adapter authorized by this manifest.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',action='store_true');a=p.parse_args()
    if a.plan:print(json.dumps(plan()));return
    raise SystemExit('FINITE_RADON_UNBOUND_NO_TIMING')
if __name__=='__main__':main()
