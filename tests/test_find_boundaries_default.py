"""Shared boundary wiring: saved tiny words and exact allocation checks."""
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]


def test_find_boundaries_shared_source_without_science():
    text=(ROOT/'python/skimagexx/coverage.py').read_text()
    assert "np.array([[0,1,0],[1,1,0]],dtype=image.dtype)" in text
    assert "find_boundaries_default_error(actual,expected,image=image);return" in text
    assert "from ._find_boundaries import find_boundaries" in (ROOT/'python/skimagexx/segmentation.py').read_text()
    sdk=(ROOT/'tests/test_component_consumers_native.cpp').read_text()
    assert '#include "cmake_consumer/find_boundaries_consumer.hpp"' in sdk
    assert 'require(skimagexx_test::find_boundaries_consumer(workers)>0)' in sdk
    qualifier=(ROOT/'tools/public_qualification.py').read_text()
    for name in ('tests/cmake_consumer/find_boundaries_consumer.hpp','tests/fixtures/find_boundaries_originals_v1.hpp'):
        assert name in qualifier


@pytest.mark.parametrize('dtype',('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64'))
@pytest.mark.parametrize('workers',(1,4))
def test_find_boundaries_shared_default_canonical_words(dtype,workers,monkeypatch):
    import numpy as np,skimagexx as xx
    import skimage.segmentation as original
    from skimagexx.coverage import default_call,assert_default_parity
    from skimagexx._find_boundaries import find_boundaries
    def deny(*a,**k):raise AssertionError('Original boundary body forbidden')
    monkeypatch.setattr(original,'find_boundaries',deny)
    assert xx.segmentation.find_boundaries is find_boundaries
    assert xx.get_num_threads()==1
    image=np.empty((11,13),dtype=np.dtype(dtype,metadata={'units':'labels'}))
    args,options=default_call('segmentation.find_boundaries',image);before=args[0].tobytes()
    xx.set_num_threads(workers)
    try:
        actual=xx.segmentation.find_boundaries(*args,**options)
        second=xx.segmentation.find_boundaries(*args,**options)
        # The admitted bool-thick labels have these same canonical0/1 values.
        row=next(r for r in json.loads((ROOT/'coverage/contracts/find-boundaries-v1.json').read_bytes())['original_observations'][np.__version__]if r['id']=='find_boundaries/bool-thick')
        import hashlib
        assert hashlib.sha256(actual.tobytes()).hexdigest()==row['result']['raw_sha256']
        assert_default_parity('segmentation.find_boundaries',image.dtype,actual,second,image=args[0])
        assert args[0].tobytes()==before and xx.get_num_threads()==workers
    finally:xx.set_num_threads(1)


@pytest.mark.parametrize('fault',('identity','word','dtype','ownership','metadata','readonly'))
def test_find_boundaries_shared_checker_negative(fault):
    import numpy as np
    from skimagexx.coverage import find_boundaries_default_error
    expected=np.ones((2,3),bool);actual=expected.copy()
    if fault=='identity':actual=expected
    elif fault=='word':actual[0,0]=False
    elif fault=='dtype':actual=actual.astype(np.uint8)
    elif fault=='ownership':actual=actual.view()
    elif fault=='metadata':actual=actual.view(np.dtype(bool,metadata={'tag':1}))
    else:actual.flags.writeable=False
    with pytest.raises(AssertionError):find_boundaries_default_error(actual,expected)
