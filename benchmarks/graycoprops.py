"""Prospective allocated public-call cases only; no execution authority."""
import copy
import math
CASES=[{'id': 'graycoprops/C/counts', 'api': 'feature.graycoprops', 'shape': [64, 64, 2, 2], 'layout': 'C', 'dtype': 'uint32', 'options': {'prop': 'contrast'}}, {'id': 'graycoprops/C/normalized', 'api': 'feature.graycoprops', 'shape': [64, 64, 2, 2], 'layout': 'C', 'dtype': 'uint32', 'options': {'prop': 'entropy'}}, {'id': 'graycoprops/F/counts', 'api': 'feature.graycoprops', 'shape': [64, 64, 2, 2], 'layout': 'F', 'dtype': 'uint32', 'options': {'prop': 'energy'}}, {'id': 'graycoprops/F/normalized', 'api': 'feature.graycoprops', 'shape': [64, 64, 2, 2], 'layout': 'F', 'dtype': 'uint32', 'options': {'prop': 'correlation'}}]
METHOD=dict(repeat=7,target_ms=100,warmup_min_ms=250,workers=1,retain_all_losses=True)
def case_specs():return copy.deepcopy(CASES)
def case_bounds(spec):
 assert spec in CASES
 n=math.prod(spec['shape'])
 if spec['api']=='feature.graycomatrix':
  planes=len(spec['distances'])*len(spec['angles']);cells=spec['options']['levels']**2*planes
  return dict(input_elements=n,output_cells=cells,pair_visits=n*planes,named_array_bytes=n*3+cells*32+4096,work_units=n*planes+cells*6)
 return dict(input_elements=n,output_cells=math.prod(spec['shape'][2:]),pair_visits=0,named_array_bytes=n*128+4096,work_units=n*32)
def main():raise SystemExit('UNBOUND: source-first public benchmark; no target/helper/sampler calls')
if __name__=='__main__':main()
