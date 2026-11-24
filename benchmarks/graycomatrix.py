"""Prospective allocated public-call cases only; no execution authority."""
import copy
import math
CASES=[{'id': 'graycomatrix/C/counts', 'api': 'feature.graycomatrix', 'shape': [257, 257], 'layout': 'C', 'dtype': 'uint8', 'distances': [1.0, 2.0], 'angles': [0.0, 1.5707963267948966], 'options': {'levels': 16, 'symmetric': False, 'normed': False}}, {'id': 'graycomatrix/C/normalized', 'api': 'feature.graycomatrix', 'shape': [257, 257], 'layout': 'C', 'dtype': 'uint8', 'distances': [1.0, 2.0], 'angles': [0.0, 1.5707963267948966], 'options': {'levels': 16, 'symmetric': True, 'normed': True}}, {'id': 'graycomatrix/F/counts', 'api': 'feature.graycomatrix', 'shape': [257, 257], 'layout': 'F', 'dtype': 'uint8', 'distances': [1.0, 2.0], 'angles': [0.0, 1.5707963267948966], 'options': {'levels': 16, 'symmetric': False, 'normed': False}}, {'id': 'graycomatrix/F/normalized', 'api': 'feature.graycomatrix', 'shape': [257, 257], 'layout': 'F', 'dtype': 'uint8', 'distances': [1.0, 2.0], 'angles': [0.0, 1.5707963267948966], 'options': {'levels': 16, 'symmetric': True, 'normed': True}}]
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
