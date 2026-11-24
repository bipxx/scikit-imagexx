"""Fixed public-call benchmark source; genuine execution remains UNBOUND.

The source-first cases/method are immutable. This route declares native ownership
and boundary providers; it neither computes a replacement reference nor samples.
"""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def cases():
    rows=[]
    for name in ('graycomatrix','graycoprops'):
        spec=importlib.util.spec_from_file_location('_gray_plan_'+name,ROOT/'benchmarks'/str(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rows.extend(module.case_specs())
    return rows

METHOD=dict(repeat=7,target_ms=100,warmup_min_ms=250,workers=1,retain_all_losses=True)
NATIVE_ENTRIES={'feature.graycomatrix':'graycomatrix_apply','feature.graycoprops':'graycoprops_apply'}
PROVIDERS={'feature.graycomatrix':['asanyarray/ascontiguousarray','validation extrema and signed-negative scan','offset coercion'], 'feature.graycoprops':['astype float64 including object/complex conversion','scalar NumPy policy dispatch only']}

def main():
    raise SystemExit('UNBOUND: genuine producer/core/SDK/root/first-cohort/measurement authority required; no target or sampler call')

if __name__=='__main__':main()
