"""Candidate mapping allocation/source behavior through actual owned stages."""
import sys
import numpy as np
from test_join_segmentations_public import target

def test_mapping_reuses_third_rank_unique_and_keeps_shared_key_ownership(target,monkeypatch):
    module,core=target;unique=core.consumer_unique;seen=[];captured=[]
    def record(value):
        result=unique(value);seen.append((value,result));return result
    monkeypatch.setattr(core,'consumer_unique',record)
    prior=sys.getprofile()
    def observe(frame,event,value):
        if event=='return'and frame.f_code is module.join_segmentations.__code__:
            captured.append(frame.f_locals['labels_j'])
    a=np.array([[0,7],[7,0]],dtype=np.int16);b=np.array([[0,3],[0,3]],dtype=np.uint16)
    try:sys.setprofile(observe);joined,one,two=module.join_segmentations(a,b,True)
    finally:sys.setprofile(prior)
    assert len(seen)==3 and len(captured)==1 and captured[0]is seen[2][1]
    assert one.in_values is two.in_values and one.out_values is not two.out_values
    assert np.array_equal(one[joined],a)and np.array_equal(two[joined],b)
    assert not np.shares_memory(joined,a)and not np.shares_memory(joined,b)
