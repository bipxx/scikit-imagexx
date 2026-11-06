"""MARK public benchmark consumer; frozen8cases, UNBOUND until genuine GO."""
import argparse,json,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/mark-boundaries-benchmark-execution-v3.json'
_FROZEN=runpy.run_path(str(ROOT/'benchmarks/mark_boundaries.py'))
case_specs=_FROZEN['case_specs'];case_bounds=_FROZEN['case_bounds']


def plan():
    frozen=runpy.run_path(str(ROOT/'benchmarks/mark_boundaries.py'))
    cases=frozen['case_specs']();bounds=[frozen['case_bounds'](row)for row in cases]
    return dict(api='segmentation.mark_boundaries',cases=cases,bounds=bounds,threads=1,
                sampler='tools/montage_sustained.py:measure',repeat=7,target_ms=100,warmup_min_ms=250,
                native_entries=['mark_boundaries_prepare','mark_boundaries_mask','mark_boundaries_outline','mark_boundaries_paint'],
                metadata_entries=['mark_boundaries_color','mark_boundaries_count'],retain_all_losses=True,
                status='UNBOUND_NO_EXECUTION',root_GO=None,original_calls=0,timed_batches=0,
                next_gate='Root genuine installed product/core/SDK/F1 binding, actual strict native-owner zero-body firstcohort admission, then separate timingGO; RHS palette provider allocation/copy included, all8cases/alllosses retained.')


def main():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from mark_boundaries_execution import execute
    return execute(globals())


if __name__=='__main__':main()
