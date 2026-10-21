"""Fixed nine-case empty-axis Farid allocation experiment."""
from run import measure
from provenance import source_fingerprint
import argparse,json,math,statistics
from pathlib import Path
import numpy as np
import skimage
from skimage.filters import farid
import skimagexx as xx

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
xx.set_fallback_enabled(False);xx.set_num_threads(1)
root=Path(__file__).resolve().parents[1];fingerprint=source_fingerprint(root)
rows=[]
for dtype in (np.uint8,np.float32,np.float64):
    image=np.ones((512,512),dtype=dtype)
    for layout,data in [('contiguous',image),('fortran',np.asfortranarray(image)),('reversed',image[::-1,::-1])]:
        expected=farid(data,axis=())
        actual=xx.filters.farid(data,axis=())
        assert actual.dtype==expected.dtype and actual.flags.owndata==expected.flags.owndata
        np.testing.assert_array_equal(actual,expected)
        assert not np.shares_memory(actual,data)
        timing=measure(lambda:farid(data,axis=()),lambda:xx.filters.farid(data,axis=()),11,25)
        rows.append({'dtype':str(data.dtype),'layout':layout,**timing})
metric=math.exp(statistics.mean(math.log(r['cpp_ms']) for r in rows))
reference=math.exp(statistics.mean(math.log(r['reference_ms']) for r in rows))
assert fingerprint==source_fingerprint(root)
args.output.write_text(json.dumps({'metadata':{'reference':skimage.__version__,'threads':1,**fingerprint},'metric_cpp_ms':metric,'reference_geomean_ms':reference,'results':rows},indent=2)+'\n',encoding='utf-8')
print('metric_cpp_ms=',metric,'reference_geomean_ms=',reference,'faster=',sum(r['speedup']>=1 for r in rows),'/9',flush=True)
