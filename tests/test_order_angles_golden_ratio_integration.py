"""Installed public namespace and lazy owned-entry gate; no private bridge."""
def test_golden_installed_public_and_hot_swap(monkeypatch):
    import importlib,inspect,numpy as np,skimage,skimagexx as xx
    original=skimage.transform.order_angles_golden_ratio
    own=importlib.import_module('skimagexx._order_angles_golden_ratio')
    assert xx.transform.order_angles_golden_ratio is own.order_angles_golden_ratio
    assert 'order_angles_golden_ratio'in xx.transform.__all__
    exports={};exec('from skimagexx.transform import *',exports)
    assert exports['order_angles_golden_ratio']is own.order_angles_golden_ratio
    assert inspect.signature(original)==inspect.signature(own.order_angles_golden_ratio)
    assert own._core is xx._core and xx.get_num_threads()==1
    xx.set_fallback_enabled(False)
    for name in ('_order_angles_begin','_order_angles_resume','_order_angles_next'):
        entry=getattr(xx._core,name);assert inspect.isbuiltin(entry)and entry.__module__=='skimagexx._core'
    from types import SimpleNamespace
    from skimagexx.coverage import default_call
    theta=np.asarray([0.,90.]);before=theta.tobytes()
    def pipeline(backend):return backend.transform.order_angles_golden_ratio(theta)
    marker=object();requests=[]
    def substituted(value):requests.append(value);return marker
    assert pipeline(SimpleNamespace(transform=SimpleNamespace(order_angles_golden_ratio=substituted)))is marker
    assert requests==[theta]
    previous=xx.set_fallback_enabled(False);returns=[]
    import sys
    old_profile=sys.getprofile()
    entries={name:getattr(xx._core,name)for name in('_order_angles_begin','_order_angles_resume','_order_angles_next')}
    def profile(frame,event,value):
        if event=='c_return':
            for name,entry in entries.items():
                if value is entry:returns.append(name)
    try:
        # Lazy creation cannot snapshot global workers or enter the leaf yet.
        a=pipeline(xx);assert inspect.getgeneratorstate(a)=='GEN_CREATED'and not returns
        sys.setprofile(profile)
        values=list(a)
        assert [int(v)for v in values]==[0,1]and all(type(v)is np.int64 for v in values)
        assert returns.count('_order_angles_begin')==1 and returns.count('_order_angles_resume')==2 and returns.count('_order_angles_next')==1
        assert theta.tobytes()==before and xx.get_num_threads()==1
    finally:sys.setprofile(old_profile);xx.set_fallback_enabled(previous)
