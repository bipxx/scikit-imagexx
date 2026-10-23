"""Bounded public-call parity through actual consumer binding, never timing."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('consumer_layout_helpers',Path(__file__).with_name('test_component_consumers_pipeline.py'))
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=helpers.core.get_num_threads();policy=helpers.xx.set_fallback_enabled(False)
    helpers.core.set_num_threads(request.param)
    yield
    helpers.core.set_num_threads(previous);helpers.xx.set_fallback_enabled(policy)

@pytest.mark.parametrize('dtype',[np.uint8,np.int16,np.int32,np.int64])
@pytest.mark.parametrize('layout',['C','step','reverse','broadcast'])
@pytest.mark.parametrize('name',['remove_small_objects','clear_border','relabel_sequential'])
def test_checkerboard_widths_and_layouts(dtype,layout,name):
    seed=(np.indices((17,35,67)).sum(axis=0)%3).astype(dtype)
    if layout=='step':
        owner=np.empty((17,35,134),dtype);owner[...,::2]=seed;image=owner[...,::2]
    elif layout=='reverse':image=seed[...,::-1]
    elif layout=='broadcast':image=np.broadcast_to(seed[:1],seed.shape)
    else:image=seed
    before=image.tobytes()
    options={'max_size':4} if name=='remove_small_objects' else {'bgval':-0.0} if name=='clear_border' else {'offset':127}
    # Full frozen observer includes metadata, warnings, errors, rawbytes and
    # returned sparse ArrayMap state; it does not densify returned maps.
    contract=helpers.contract
    api=('morphology.' if name=='remove_small_objects' else 'segmentation.')+name
    def observe(function):
        import warnings
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always');result=function(image,**options)
        arrays=result if isinstance(result,tuple) else (result,)
        outputs=[]
        for value in arrays:
            if isinstance(value,np.ndarray):outputs.append((type(value),value.dtype.str,value.shape,value.strides,value.tobytes()))
            else:outputs.append((value.in_values.dtype.str,value.in_values.tobytes(),value.out_values.dtype.str,value.out_values.tobytes(),value._array))
        return outputs,[(type(w.message),str(w.message)) for w in notices]
    expected=observe(contract.reference(api));actual=observe(getattr(helpers.candidate,name))
    assert actual==expected and image.tobytes()==before
