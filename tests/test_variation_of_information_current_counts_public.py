"""Larger candidate-only dependency equivalence, never new original gold."""
import numpy as np
import pytest
from test_variation_of_information_public import target, workers


@pytest.mark.parametrize('n',(4096,4099))
@pytest.mark.parametrize('ignored', ((),(0,),(0,1,2,3)))
def test_current_dense_fallback_CSR_and_physical_storage(target,workers,n,ignored):
    import scipy.sparse as sparse
    api,core=target
    a=np.array([(i//7)%4 for i in range(n)],dtype=np.int64)
    b=np.array([(i//3)%8 for i in range(n)],dtype=np.int64)
    before=a.tobytes(),b.tobytes()
    dense=api.variation_of_information(a,b,ignore_labels=ignored)
    fallback=api.variation_of_information(a+64,b+64,ignore_labels=tuple(x+64 for x in ignored))
    assert dense.tobytes()==fallback.tobytes()
    kept=sum(int(x)not in ignored for x in a)
    # NumPy source-prefix normalization is an explicit existing provider.
    weights=np.array([0. if int(x)in ignored else 1. for x in a],dtype=np.float64)
    with np.errstate(all='ignore'):weights/=kept
    shape,indptr,indices,data=core.label_pair_counts_preweighted(a,b,weights,workers)
    logical=int(indptr[-1]);assert shape==(4,8)and logical==32
    assert data.size==indices.size==n and logical<n
    assert core.variation_of_information_csr(data,indices,indptr,*shape,workers)is None
    direct=core.variation_of_information_csr(data[:logical],indices[:logical],indptr,*shape,workers)
    assert direct.tobytes()==dense.tobytes()
    table=sparse.csr_array((data[:logical],indices[:logical],indptr),shape=shape,copy=False)
    arrays=[v.copy()for v in(data,indices,indptr)]
    supplied=api.variation_of_information(a,b,table=table)
    assert supplied.tobytes()==dense.tobytes()and supplied.flags.owndata
    assert all(x.tobytes()==y.tobytes()for x,y in zip((data,indices,indptr),arrays))
    assert (a.tobytes(),b.tobytes())==before
    assert core.get_num_threads()==workers
