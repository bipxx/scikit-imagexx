"""Dense native public suffix checks with whole original APIs refused."""
import numpy as np
import pytest
from test_contingency_table_public import refuse_originals


def owner(a):
    while type(a.base)is np.ndarray:a=a.base
    return a


@pytest.mark.parametrize('dtype',('int16','int64'))
@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('normalize',(False,True))
@pytest.mark.parametrize('sparse_type',('array','matrix'))
def test_density_public_and_binder_backing(dtype,workers,normalize,sparse_type,refuse_originals):
    import skimagexx as xx
    from skimagexx._contingency_table import contingency_table
    n=262151;truth=np.ones(n,dtype);test=np.full(n,2,dtype);before=truth.tobytes(),test.tobytes()
    weight=1./n if normalize else 1.;expected=weight
    for _ in range(1,n):expected+=weight
    if normalize:assert expected!=n*weight # meaningful ordered-addition witness
    policy=np.geterr(),np.geterrcall(),np.getbufsize();old=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        result=contingency_table(truth,test,ignore_labels=(0,),normalize=normalize,sparse_type=sparse_type)
        assert result.format=='csr'and result.shape==(2,3)and result.nnz==1
        assert result.indptr.tolist()==[0,0,1]and result.indices.tolist()==[2]
        assert result.data.tobytes()==np.array([expected],np.float64).tobytes()
        # SciPy may trim structural CSR backing; full slots are checked at
        # the binder boundary, before that declared packing provider.
        raw=xx._core.label_pair_counts_preweighted(
            truth.astype(np.int64),test.astype(np.int64),np.full(n,weight),workers)
        assert raw[0]==(2,3)and raw[1].tolist()==[0,0,1]
        indices,data=raw[2:]
        assert data.size==indices.size==n and np.all(indices==2)
        assert data[:1].tobytes()==result.data.tobytes()and np.all(data[1:]==weight)
        assert result.has_sorted_indices and result.has_canonical_format
        assert (truth.tobytes(),test.tobytes())==before and xx.get_num_threads()==workers
        assert (np.geterr(),np.geterrcall(),np.getbufsize())==policy
    finally:xx.set_num_threads(old)


@pytest.mark.parametrize('dtype',('int16','int64'))
@pytest.mark.parametrize('workers',(1,4))
def test_density_public_ARE_tuple(dtype,workers,refuse_originals):
    import skimagexx as xx
    from skimagexx._label_pair_metrics import adapted_rand_error
    a=np.ones(262151,dtype);b=np.full(a.size,2,dtype);old=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        result=adapted_rand_error(a,b,ignore_labels=(0,),alpha=.5)
        assert type(result)is tuple and all(type(x)is np.float64 for x in result)
        assert np.asarray(result).tobytes()==np.array([0.,1.,1.],np.float64).tobytes()
        assert np.all(a==1)and np.all(b==2)and xx.get_num_threads()==workers
    finally:xx.set_num_threads(old)
