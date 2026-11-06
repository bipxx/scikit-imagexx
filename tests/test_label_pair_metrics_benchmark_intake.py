"""Installed NumPy-only fixture intake; no metric/native/original algorithms."""
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]
CASES=json.loads((ROOT/'coverage/contracts/label-pair-metrics-benchmark-v1.json').read_bytes())['cases']


@pytest.mark.parametrize('case',CASES,ids=lambda s:s['id'])
def test_label_pair_actual_numpy_factory_original_refused(case,monkeypatch):
    import numpy as np
    import builtins
    previous=builtins.__import__;attempts=[]
    def refuse(name,*args,**kwargs):
        if name.split('.')[0]in('scipy','skimage','skimagexx'):attempts.append(name);raise AssertionError('algorithm imports refused: '+name)
        return previous(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',refuse)
    spec=importlib.util.spec_from_file_location('label_pair_actual_factory',ROOT/'benchmarks/label_pair_metrics.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    arguments,options=driver.fixture(case,np);bound=driver.source_safe(case)
    assert len(arguments)==2 and all(type(a)is np.ndarray and a.dtype==np.dtype('int64')and a.shape==tuple(case['shape'])and a.size==bound['pixels']and a.flags.writeable and a.flags.aligned for a in arguments)
    assert all(bool(np.all((a>=0)&(a<case['labels'])))for a in arguments)
    assert all(a.flags.f_contiguous if case['layout']=='F'else a.flags.c_contiguous for a in arguments)
    assert not np.shares_memory(*arguments)
    assert options['ignore_labels']==[0]and attempts==[]
