"""Fixed one-worker boolean rectangle workload for reproducible optimization."""
from run import measure
import argparse
import json
import math
from pathlib import Path
import numpy as np
from skimage import morphology as reference
import skimagexx as xx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeat',type=int,default=7)
    args=parser.parse_args()
    xx.set_num_threads(1)
    xx.set_fallback_enabled(False)
    rng=np.random.default_rng(23)
    inputs={'random':rng.random((512,512))>.5,'zeros':np.zeros((512,512),bool),
            'ones':np.ones((512,512),bool)}
    fp=np.ones((31,31),bool)
    results=[]
    for name in ('erosion','dilation','opening','closing','white_tophat','black_tophat'):
        old,current=getattr(reference,name),getattr(xx.morphology,name)
        for label,image in inputs.items():
            expected=old(image,fp)
            actual=current(image,fp)
            assert actual.dtype==expected.dtype
            np.testing.assert_array_equal(actual,expected)
            timing=measure(lambda:old(image,fp),lambda:current(image,fp),args.repeat,15)
            results.append({'api':name,'input':label,**timing})
    metric=math.exp(sum(math.log(r['cpp_ms']) for r in results)/len(results))
    report={'metric':'geometric mean native ms; lower is better','workers':1,'cases':18,
            'geomean_ms':metric,'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'geomean_ms={metric:.6f}',flush=True)


if __name__=='__main__': main()
