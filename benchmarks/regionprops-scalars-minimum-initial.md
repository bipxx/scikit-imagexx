# Region scalar property workloads

Eight publicscalarproperty workloads; constructor/allocation andwarmgetter boundaries distinguished. Nowholeclass/pureownedpipeline/cachedblanketspeed/KEEP claim.

| Case | Original ms | Backend ms | Original/backend |
|---|---:|---:|---:|
| regionprops-scalars/tiny-ctor-c-off | 0.074783759 | 0.103251603 | 0.724286669x |
| regionprops-scalars/medium-ctor-f-on | 0.147409632 | 0.742895426 | 0.198425817x |
| regionprops-scalars/tiny-warm-c-on | 0.003521659 | 0.012896154 | 0.273078254x |
| regionprops-scalars/medium-getters-c-off | 0.102672070 | 0.176687609 | 0.581093775x |
| regionprops-scalars/medium-getters-f-off | 0.062887135 | 2.235965173 | 0.028125275x |
| regionprops-scalars/3d-warm-f-on | 0.003592617 | 0.013119887 | 0.273829850x |
| regionprops-scalars/pixel-cycle-c-off | 0.108991298 | 0.139140158 | 0.783320214x |
| regionprops-scalars/slice-cycle-3d-on | 0.080016781 | 0.201754856 | 0.396603989x |
