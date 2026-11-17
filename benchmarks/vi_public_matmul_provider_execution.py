"""Current VI log2/matmul providers; unchanged strict untimed/timed executor.

Only preflight is instrumented. The sampler still invokes exact public calls
without wrappers. Old provider contexts and genuine authorities stay frozen.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/vi-public-matmul-provider-benchmark-v1.json'
ORDER=('numpy.log2','numpy.matmul','numpy.log2','numpy.matmul')


@contextmanager
def candidate_providers(np,sparse):
    """Two whole log2 batches and two dense inner products, in source order."""
    originals=np.log2,np.matmul
    changed=[];events=[]
    def vector(value):
        assert type(value)is np.ndarray and value.ndim==1
        assert value.dtype.kind=='f' and value.dtype.itemsize==8 and value.dtype.isnative
        assert value.flags.c_contiguous and value.flags.owndata
        assert value.size<=1<<20
    def enter(name):
        assert len(events)<len(ORDER) and ORDER[len(events)]==name,'VI provider source order required'
    def log2(values):
        enter('numpy.log2');vector(values)
        events.append(dict(provider='numpy.log2',shape=list(values.shape),dtype=values.dtype.str))
        return originals[0](values)
    def matmul(left,right):
        enter('numpy.matmul');vector(left);vector(right)
        assert left.shape==right.shape
        events.append(dict(provider='numpy.matmul',left_shape=list(left.shape),right_shape=list(right.shape),dtype=left.dtype.str))
        value=originals[1](left,right)
        assert type(value)is np.float64 and value.shape==()
        assert value.dtype.kind=='f' and value.dtype.itemsize==8 and value.dtype.isnative
        return value
    def deny(*a,**k):raise AssertionError('VI undeclared arithmetic provider forbidden')
    def replace(owner,name,value):
        if hasattr(owner,name):
            prior=getattr(owner,name);owned=name in vars(owner)
            setattr(owner,name,value);changed.append((owner,name,prior,owned))
    replace(np,'log2',log2);replace(np,'matmul',matmul)
    for name in ('log','log10','sum','nansum','prod','mean','cumsum','dot','vdot','inner','tensordot','einsum'):
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
        yield events
        assert np.log2 is log2 and np.matmul is matmul
        assert tuple(row['provider']for row in events)==ORDER,'Four declared source-ordered providers required'
    finally:
        for owner,name,prior,owned in reversed(changed):
            if owned:setattr(owner,name,prior)
            else:delattr(owner,name)
        if own_init:csr.__init__=initialize
        else:del csr.__init__


def executor():
    data=json.loads((ROOT/CONTEXT).read_bytes())
    assert data['route']['contract']==CONTEXT
    assert data['route']['driver']=='benchmarks/vi_public_matmul_provider_consumer.py'
    path=ROOT/'benchmarks/variation_of_information_execution.py'
    assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==data['reused_execution_canonical_sha256']
    spec=importlib.util.spec_from_file_location('_vi_matmul_strict_predecessor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.CONTEXT=CONTEXT
    module.candidate_providers=candidate_providers
    return module


def execute(module):
    return executor().execute(module)
