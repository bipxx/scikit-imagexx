"""LBP public-call consumer; four frozen cases, genuine authority required."""
import argparse,json,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def case_specs():return json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-v1.json').read_bytes())['benchmark_cases']


def candidate_call(namespace,image,spec):
    # Namespace must be the future admitted public hot-swap surface, never a
    # direct leaf stand-in. Root binds actual original/core/runtime/SDK first.
    assert spec in case_specs()
    return namespace.feature.draw_multiblock_lbp(image,spec['r'],spec['c'],spec['width'],spec['height'],lbp_code=spec['code'],alpha=spec['alpha'])


def case_bounds(spec):return runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp.py'))['case_bounds'](spec)


def plan():
    c=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-v1.json').read_bytes())
    return dict(status='UNBOUND_NO_EXECUTION',api='feature.draw_multiblock_lbp',cases=case_specs(),method=c['benchmark_method'],root_GO=None,original_calls=0,timed_batches=0,retain_all_losses=True,
                next_gate='Genuine public namespace/core/completeSDK/runtime/F1/root binding, independent actual first-cohort0body admission, then separate timingGO. NumPy metadata/palette/scalar provider costs included; all4cases/alllosses retained. FullAPI/platform/Windows and performance gates remain open.')


def main():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from draw_multiblock_lbp_execution import execute,fixture,native_requirements
    return execute(dict(globals(),fixture=fixture,native_requirements=native_requirements))


if __name__=='__main__':main()
