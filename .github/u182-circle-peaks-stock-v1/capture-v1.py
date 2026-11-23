"""Unmodified circular-peak stock observations from the reviewed local entry."""
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
import warnings


HERE = Path(__file__).resolve().parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def capture(entry, fixture, frozen):
    identity = entry['identity']
    site = Path(identity['site'])
    sys.path.insert(0, str(site))
    import numpy as np
    import scipy
    import skimage
    from skimage import transform
    from skimage.transform import hough_transform
    from skimage.feature import peak
    from skimage.measure import _label, _regionprops, _regionprops_utils
    from skimage.morphology import _util
    from scipy.ndimage import _filters, _measurements, _nd_image, _ni_label
    from scipy.spatial import _ckdtree

    version = identity['versions']
    assert (np.__version__,scipy.__version__,skimage.__version__) == (version['numpy'],version['scipy'],version['scikit_image'])
    numpy_core = sys.modules['numpy._core._multiarray_umath']
    modules = (np, scipy, skimage, hough_transform, peak, _label, _regionprops, _regionprops_utils,
               _util, _filters, _measurements, _nd_image, _ni_label, numpy_core, _ckdtree)
    pinned = {k:v['sha256'] for group in identity['distributions'].values() for k,v in group['files'].items()}
    loaded = {Path(m.__file__).resolve().relative_to(site).as_posix(): digest(Path(m.__file__)) for m in modules}
    assert all(pinned[name] == value for name,value in loaded.items())
    assert hough_transform.cKDTree is _ckdtree.cKDTree
    fn = transform.hough_circle_peaks
    assert fn is hough_transform.hough_circle_peaks
    snapshot = fixture['snapshot']
    prepared = {}
    for name, pin in entry['prepared'].items():
        p = Path(pin['path'])
        assert digest(p) == pin['sha256']
        array = np.load(p,allow_pickle=False)
        delta = {key for key,value in snapshot(array).items() if pin['input'][key] != value}
        assert delta <= {'owns','base'}
        prepared[name] = array.copy(order='C') if delta else array
        assert snapshot(prepared[name]) == pin['input']
    cases = list(fixture['cases'](np,prepared,'linux-minimum'))
    assert len(cases)==29 and len({row['id'] for _,row in cases})==29
    rows = []
    path = HERE / 'linux-modern-rows-v1.jsonl'
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        for args,saved in cases:
            before = [snapshot(a,'data_hex' in record) for a,record in zip(args,saved['inputs'],strict=True)]
            row = {key:saved[key] for key in ('api','id','scope','options') if key in saved}
            row['inputs'] = before
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter('always')
                try:
                    result = fn(*args,**row['options'])
                except (ValueError, RuntimeError, TypeError, IndexError) as error:
                    row['error'] = [type(error).__module__,type(error).__name__,str(error)]
                else:
                    assert type(result) is tuple and len(result)==4 and all(type(a) is np.ndarray for a in result)
                    row['container'] = 'tuple'
                    row['result'] = [snapshot(a,a.size<=1024) for a in result]
                    row['result_aliases_inputs'] = [[bool(np.shares_memory(a,b)) for b in args] for a in result]
                    if 'values' in saved:
                        row['values'] = [a.tolist() for a in result]
            row['warnings'] = [[type(w.message).__module__,w.category.__name__,str(w.message)] for w in emitted]
            row['inputs_unchanged'] = before == [snapshot(a,'data_hex' in record) for a,record in zip(args,before,strict=True)]
            assert row['inputs_unchanged']
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            stream.flush()
            rows.append(row)
    comparison = {}
    for profile, old in frozen['profiles'].items():
        assert [row['id'] for row in rows]==[row['id'] for row in old]
        differences = {}
        for row,prior in zip(rows,old,strict=True):
            keys = sorted(k for k in row.keys() | prior.keys() if row.get(k) != prior.get(k))
            if keys:
                differences[row['id']] = keys
        comparison[profile] = dict(matches=29-len(differences),differences=differences)
    assert 'skimagexx' not in sys.modules
    result = dict(rows_sha256=digest(path),entry_sha256=digest(HERE / 'entry-v1.json'),stock_peak_calls=len(rows),
                  stock_producer_calls=0,candidate_calls=0,timing_calls=0,loaded=loaded,comparisons=comparison,
                  signature=str(inspect.signature(fn)),row_ids=[row['id'] for row in rows])
    with (HERE / 'linux-modern-terminal-v1.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(result['comparisons'],result['rows_sha256'],flush=True)
