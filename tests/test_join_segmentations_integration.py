"""Genuine installed public JOIN ownership and saved-only dtype smoke.

No private component is accepted here. Exact upstream ArrayMap identity and
nonpositive/floating SDK domains remain separate pending contract gates.
"""
import importlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).parents[1]
SAVED=json.loads((ROOT/'tests/fixtures/join_segmentations_originals_v1.json').read_bytes())
INTEGER_DTYPES=('int8','uint8','int16','uint16','int32','uint32','int64','uint64')


@pytest.fixture
def genuine():
    import skimagexx as xx
    owned=importlib.import_module('skimagexx._join_segmentations')
    consumers=importlib.import_module('skimagexx._component_consumers')
    assert xx.segmentation.join_segmentations is owned.join_segmentations
    assert owned._core is consumers._core is xx._core
    assert Path(xx._core.__file__).resolve().parent==Path(xx.__file__).resolve().parent
    for name in ('join_encode','join_divmod','consumer_max','consumer_unique','consumer_map','get_num_threads','set_num_threads'):
        entry=getattr(xx._core,name)
        assert inspect.isbuiltin(entry)and entry.__module__=='skimagexx._core'
    assert xx.get_num_threads()==1
    previous=xx.set_fallback_enabled(False)
    try:yield xx,owned,consumers
    finally:xx.set_num_threads(1);xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('dtype',INTEGER_DTYPES)
@pytest.mark.parametrize('workers',(1,4))
def test_join_single_core_import_swap_binary_integer_smoke(dtype,workers,genuine,monkeypatch):
    xx,owned,consumers=genuine
    from skimagexx.segmentation import join_segmentations as imported
    from skimagexx.coverage import default_call
    from draw_coordinates_observer import encode
    import skimage.segmentation as original
    def refused(*args,**kwargs):raise AssertionError('Original JOIN body forbidden')
    monkeypatch.setattr(original,'join_segmentations',refused)
    args,options=default_call('segmentation.join_segmentations',np.empty((1,1),dtype))
    frozen=[encode(a,np)for a in args];policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();xx.set_num_threads(workers)
    assert owned._core.get_num_threads()==consumers._core.get_num_threads()==workers
    returns=[];previous=sys.getprofile()
    def profile(frame,event,entry):
        if event=='c_return'and entry in (xx._core.join_encode,xx._core.join_divmod):returns.append(entry.__name__)
    def pipeline(backend):return backend.segmentation.join_segmentations(*args,return_mapping=True)
    sys.setprofile(profile)
    try:joined,one,two=pipeline(xx)
    finally:sys.setprofile(previous)
    assert imported is owned.join_segmentations and returns==['join_encode','join_divmod']
    expected=next(r for r in SAVED['profiles']['minimum'if np.__version__.startswith('1.')else'modern']['rows']if r['id']=='join_segmentations/'+dtype+'-maps')
    assert encode(joined,np)==expected['result']['joined']
    assert type(one)is type(two)is consumers.ArrayMap # Upstream class identity remains an explicit gap.
    assert one.in_values is two.in_values and one.out_values is not two.out_values
    for actual,gold in zip((one,two),expected['result']['maps']):
        assert encode(actual.in_values,np)==gold['keys']and encode(actual.out_values,np)==gold['values']
    assert [encode(a,np)for a in args]==frozen
    assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
    with pytest.raises(TypeError):imported(args[0])


@pytest.mark.parametrize('dtype',('bool','float16','float32','float64'))
def test_join_defined_dtype_rejections_keep_binary_arity(dtype,genuine):
    xx,_,_=genuine;a=np.array([0,1],dtype=dtype)
    with pytest.raises(TypeError,match='integer dtype'):xx.segmentation.join_segmentations(a,a)
    if dtype!='bool':
        with pytest.raises(ValueError,match='negative values'):xx.segmentation.join_segmentations(np.array([-1,1],dtype=dtype),a)


@pytest.mark.parametrize('dtype',('complex64','complex128','object','S3','U3','V3','datetime64[D]','timedelta64[D]','longdouble'))
def test_join_unassessed_dtypes_refuse_before_whole_original(dtype,genuine,monkeypatch):
    xx,_,_=genuine
    import skimage.segmentation as original
    def refused(*args,**kwargs):raise AssertionError('Original fallback entered')
    monkeypatch.setattr(original,'join_segmentations',refused);a=np.zeros(3,dtype=dtype)
    if dtype=='longdouble'and a.dtype.itemsize==8:
        with pytest.raises(TypeError,match='integer dtype'):xx.segmentation.join_segmentations(a,a)
    else:
        with pytest.raises(NotImplementedError,match='fallback is disabled'):xx.segmentation.join_segmentations(a,a)



def test_join_unassessed_subclass_and_array_provider_refuse(genuine):
    xx,_,_=genuine
    class Subclass(np.ndarray):pass
    base=np.array([0,1],np.int16);subclass=base.view(Subclass)
    with pytest.raises(NotImplementedError,match='fallback is disabled'):xx.segmentation.join_segmentations(subclass,subclass)
    class Provider:
        shape=base.shape
        def __array__(self,*args,**kwargs):raise AssertionError('unassessed provider conversion entered')
    with pytest.raises(NotImplementedError,match='fallback is disabled'):xx.segmentation.join_segmentations(Provider(),Provider())
