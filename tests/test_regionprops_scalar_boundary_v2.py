"""Exact scalar CAPI/private-getter boundaries; no original recapture or timing."""
import inspect
import sys
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize('count,scale,bits',[(0,1.,'0000000000000000'),(3,.5,'000000000000f83f'),(2**53,1.,'0000000000004043')])
def test_scalar_capi_preserves_primitive_words(count,scale,bits):
    import numpy as np
    from skimagexx import _core
    value,status=_core.regionprops_state_scaled_count_f64(count,np.float64(scale))
    assert type(value) is np.float64 and value.tobytes().hex()==bits and status==0
    value,status=_core.regionprops_state_extent_f64(np.float64(0),np.float64(0))
    assert type(value) is np.float64 and value.tobytes().hex()=='000000000000f8ff' and status==1

@pytest.mark.parametrize('dtype',['float32','int64','python'])
def test_scalar_capi_exact_type_refusal(dtype):
    import numpy as np
    from skimagexx import _core
    value=1. if dtype=='python' else getattr(np,dtype)(1)
    assert _core.regionprops_state_scaled_count_f64(3,value) is None
    assert _core.regionprops_state_extent_f64(value,np.float64(2)) is None

@pytest.mark.parametrize('workers',[1,4])
def test_status_zero_getters_need_no_python_fp_policy_or_primitive_roundtrip(workers,monkeypatch):
    import numpy as np,skimagexx as xx
    from skimagexx import _regionprops_state as owned
    region=owned.regionprops(np.array([[1,1],[1,0]],np.int16),cache=True)[0]
    region.area
    def denied(*a,**k):raise AssertionError('event-free getter queried Python FP policy or old primitive binder')
    calls=[];entries={xx._core.regionprops_state_area_bbox_get,xx._core.regionprops_state_extent_get}
    def trace(frame,event,arg):
        if event=='c_return' and arg in entries:calls.append(arg)
    previous=sys.getprofile();budget=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        with monkeypatch.context() as patch:
            for name in ('geterr','geterrcall'):patch.setattr(np,name,denied)
            for name in ('regionprops_state_scaled_count_f64','regionprops_state_extent_f64'):patch.setattr(xx._core,name,denied)
            sys.setprofile(trace)
            try:bbox,extent=region.area_bbox,region.extent
            finally:sys.setprofile(previous)
        assert type(bbox) is type(extent) is np.float64 and bbox==4 and extent==.75
        assert calls.count(xx._core.regionprops_state_area_bbox_get)==2
        assert calls.count(xx._core.regionprops_state_extent_get)==1
        assert inspect.signature(type(region).area_bbox.fget)==inspect.signature(type(region).extent.fget)==inspect.Signature([inspect.Parameter('self',inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    finally:sys.setprofile(previous);xx.set_num_threads(budget)


def test_private_compiled_getters_read_dependencies_once_in_source_order():
    import numpy as np
    from skimagexx import _core
    events=[]
    class Image:
        @property
        def size(self):events.append('size');return 4
    class Probe:
        @property
        def image(self):events.append('image');return Image()
        @property
        def _pixel_area(self):events.append('pixel_area');return np.float64(.5)
        @property
        def area(self):events.append('area');return np.float64(1)
        @property
        def area_bbox(self):events.append('area_bbox');return np.float64(2)
    assert _core.regionprops_state_area_bbox_get(Probe())==np.float64(2)
    assert events==['image','size','pixel_area'];events.clear()
    assert _core.regionprops_state_extent_get(Probe())==np.float64(.5)
    assert events==['area','area_bbox']

def test_private_getter_name_reuse_keeps_operands_fresh_and_keyword_signature():
    import numpy as np
    from skimagexx import _core
    probe=SimpleNamespace(image=SimpleNamespace(size=4),_pixel_area=np.float64(.5),
                          area=np.float64(1),area_bbox=np.float64(2))
    assert _core.regionprops_state_area_bbox_get(self=probe)==np.float64(2)
    assert _core.regionprops_state_extent_get(self=probe)==np.float64(.5)
    probe.image=SimpleNamespace(size=12);probe._pixel_area=np.float64(.25)
    probe.area=np.float64(3);probe.area_bbox=np.float64(4)
    assert _core.regionprops_state_area_bbox_get(probe)==np.float64(3)
    assert _core.regionprops_state_extent_get(probe)==np.float64(.75)

@pytest.mark.parametrize('value_kind',['float32','negative','nonfinite','python'])
def test_getter_refusal_retains_already_evaluated_operands(value_kind):
    import numpy as np
    from skimagexx import _core
    value={'float32':np.float32(2),'negative':np.float64(-1),'nonfinite':np.float64(float('inf')),'python':2.}[value_kind]
    events=[]
    class Probe:
        @property
        def image(self):events.append('image');return SimpleNamespace(size=3)
        @property
        def _pixel_area(self):events.append('pixel_area');return value
        @property
        def area(self):events.append('area');return value
        @property
        def area_bbox(self):events.append('bbox');return np.float64(2)
    result=_core.regionprops_state_area_bbox_get(Probe())
    assert type(result) is tuple and result[0]==3 and result[1] is value
    assert events==['image','pixel_area'];events.clear()
    result=_core.regionprops_state_extent_get(Probe())
    assert type(result) is tuple and result[0] is value and result[1]==np.float64(2)
    assert events==['area','bbox']

@pytest.mark.parametrize('policy',['print','log'])
def test_eventful_print_log_getter_uses_existing_tail_once(policy,monkeypatch,capfd):
    import numpy as np
    from skimagexx import _regionprops_state as owned
    from skimagexx import _regionprops_state_class as cls
    region=owned.regionprops(np.array([[1]],np.int16),cache=False)[0];events=[]
    def area(self):events.append('area');return np.float64(0)
    def bbox(self):events.append('bbox');return np.float64(0)
    monkeypatch.setattr(owned.RegionProperties,'area',property(area));monkeypatch.setattr(owned.RegionProperties,'area_bbox',property(bbox))
    actual=cls._extent
    def tail(first,denominator):events.append('tail');return actual(first,denominator)
    monkeypatch.setattr(cls,'_extent',tail)
    class Log:
        def write(self,text):events.append(text)
    with np.errstate(all='ignore',invalid=policy,call=Log()):value=region.extent
    assert type(value) is np.float64 and value.tobytes().hex()=='000000000000f8ff'
    assert events[:3]==['area','bbox','tail'] and events.count('tail')==1
    captured=capfd.readouterr()
    if policy=='print':assert 'invalid value encountered in scalar divide' in captured.err
    else:assert events[3:]==['Warning: invalid value encountered in scalar divide\n']

@pytest.mark.parametrize('policy',['print','log'])
def test_event_free_print_log_are_owned_without_policy_tail(policy,monkeypatch):
    import numpy as np
    from skimagexx import _regionprops_state as owned
    from skimagexx import _regionprops_state_class as cls
    region=owned.regionprops(np.array([[1,1],[1,0]],np.int16),cache=True)[0]
    region.area
    def denied(*a,**k):raise AssertionError('status0 entered Python policy tail')
    monkeypatch.setattr(cls,'_extent',denied)
    with np.errstate(all='ignore',invalid=policy):
        value=region.extent
    assert type(value) is np.float64 and value==.75
