"""Safe descriptor metadata only; no target/helper/ufunc application."""
import numpy as np
from finite_radon_exception_codec import encode_exception_args

def test_metadata_identity_and_nesting():
    dtype=np.dtype('uint32');d=encode_exception_args((np.add,dtype,[np.add,dtype]),np)
    a,b,nested=d['items'];aa,bb=nested['items']
    assert a['numpy_namespace_identity']and a['name']=='add'and a['nin']==2 and a['nout']==1
    assert a['object_index']==aa['object_index']and b['object_index']==bb['object_index']and a['object_index']!=b['object_index']
    assert b['str']==dtype.str and b['name']==dtype.name and b['itemsize']==4

def test_inert_exception_arguments_only():
    error=TypeError(np.add,'same_kind',np.dtype('int64'),np.dtype('uint32'),2)
    d=encode_exception_args(error.args,np)
    assert d['items'][0]['kind']=='numpy.ufunc'and d['items'][2]['str']==np.dtype('int64').str
