"""Exact caller dtype resolution order; no operation timings or caching."""
import warnings
import numpy as np
import pytest
from skimage import morphology as original
import skimagexx as xx


@pytest.fixture(params=[1,4],autouse=True)
def configuration(request):
    old=xx.set_fallback_enabled(False);workers=xx.get_num_threads();xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(workers);xx.set_fallback_enabled(old)


@pytest.mark.parametrize('shape',[(3,5),(2,4),(-1,3),(1.5,3),()])
@pytest.mark.parametrize('decomposition',[None,'sequence','separable'])
@pytest.mark.parametrize('behavior',['stable','second-error','changing','first-error'])
def test_dtype_getter_and_error_order_per_original_allocation(shape,decomposition,behavior):
    def observe(namespace):
        events=[]
        class Provider:
            @property
            def dtype(self):
                events.append('dtype')
                if behavior=='first-error' or behavior=='second-error' and len(events)==2:raise ValueError('caller dtype failure')
                return np.dtype('>i2' if behavior!='changing' or len(events)%2 else 'f4')
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:value,error=namespace.footprint_rectangle(shape,dtype=Provider(),decomposition=decomposition),None
            except Exception as caught:value,error=None,caught
        if error is not None:result=('error',type(error),str(error))
        else:
            elements=value if isinstance(value,tuple) else ((value,1),)
            result=tuple((a.dtype.str,a.shape,a.strides,a.tobytes(),count,a.flags.owndata) for a,count in elements)
        return events,result,[(type(w.message),str(w.message)) for w in notices]
    assert observe(xx.morphology)==observe(original)
