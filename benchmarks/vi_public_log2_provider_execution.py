"""Explicit current VI log2 provider over the unchanged strict VI executor.

Import is inert. The isolated executor copy changes only its source contract
and elementary-provider guard; cases, sampler, strict words/FP and engine stay
in the authenticated predecessor. No old genuine authority is retargeted.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/vi-public-log2-provider-benchmark-v1.json'


@contextmanager
def candidate_providers(np,sparse):
    """Exactly two source-shaped public log2 batches; no other entropy provider."""
    original=np.log2
    changed=[];batches=[]
    def deny(*a,**k):raise AssertionError('VI other entropy/reduction provider forbidden')
    def log2(values):
        assert type(values)is np.ndarray and values.ndim==1
        assert values.dtype.kind=='f' and values.dtype.itemsize==8 and values.dtype.isnative
        assert values.flags.c_contiguous and values.flags.owndata
        assert values.size<=1<<20 and len(batches)<2
        batches.append(dict(shape=list(values.shape),dtype=values.dtype.str,provider='numpy.log2'))
        return original(values)
    def replace(owner,name,value):
        if hasattr(owner,name):
            previous=getattr(owner,name);owned=name in vars(owner)
            setattr(owner,name,value);changed.append((owner,name,previous,owned))
    # A current attribute is passed by the frontend. No cached handle bypasses
    # this wrapper: actual provider calls must traverse and increment it.
    replace(np,'log2',log2)
    for name in ('log','log10','sum','nansum','prod','mean','cumsum'):
        replace(np,name,deny)
    csr=sparse.csr_array;initialize=csr.__init__;own_init='__init__'in csr.__dict__
    def structural(self,arg,*a,**k):
        assert type(arg)is tuple and len(arg)==3,'Candidate numerical coordinate/coalescing provider forbidden'
        return initialize(self,arg,*a,**k)
    csr.__init__=structural
    for cls in (sparse.csr_array,sparse.csr_matrix):
        for name in ('__matmul__','sum','dot','multiply'):
            replace(cls,name,deny)
    try:
        yield batches
        assert np.log2 is log2 and len(batches)==2,'Two declared log2 batches required'
    finally:
        for owner,name,previous,owned in reversed(changed):
            if owned:setattr(owner,name,previous)
            else:delattr(owner,name)
        if own_init:csr.__init__=initialize
        else:del csr.__init__


def executor():
    data=json.loads((ROOT/CONTEXT).read_bytes())
    assert data['route']['contract']==CONTEXT
    assert data['route']['driver']=='benchmarks/vi_public_log2_provider_consumer.py'
    path=ROOT/'benchmarks/variation_of_information_execution.py'
    assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==data['reused_execution_canonical_sha256']
    spec=importlib.util.spec_from_file_location('_vi_provider_strict_predecessor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.CONTEXT=CONTEXT
    module.candidate_providers=candidate_providers
    return module


def execute(module):
    return executor().execute(module)
