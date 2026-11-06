"""Private installed crop parity against admitted saved observations only."""
import copy
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from crop_contract import cases
from crop_observer import observe
import skimagexx._crop as owned
DATA=json.loads((Path(__file__).parent/'fixtures/crop_originals_v1.json').read_bytes())
PROFILE='minimum'if np.__version__.startswith('1.24.')else'modern'
GOLD={r['id']:r for r in DATA['profiles'][PROFILE]['rows']}

@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_crop_saved_public(spec,workers):
    owned._core.set_num_threads(workers)
    assert inspect.isbuiltin(owned._core.crop_view)and inspect.isbuiltin(owned._core.crop_copy_into)
    actual=observe(owned.crop,spec,np);expected=copy.deepcopy(GOLD[spec['id']])
    expected.pop('controller_fenv');expected.pop('controller_state_restored')
    # Allocation address distances are diagnostics only when buffers do not share.
    for row in(actual,expected):
        for relation in row['aliases'].values():
            if not relation['shares_memory']:relation['pointer_delta']=None
    assert actual==expected
    owned._core.set_num_threads(1)


def test_crop_private_copy_rejects_alias_and_readonly():
    a=np.arange(6,dtype=np.int16)
    with pytest.raises(ValueError,match='overlaps'):owned._core.crop_copy_into(a,a)
    b=np.empty_like(a);b.flags.writeable=False
    with pytest.raises(ValueError,match='read-only'):owned._core.crop_copy_into(a,b)


def test_crop_private_copy_rejects_overlapping_output_axes():
    a=np.arange(4,dtype=np.int16).reshape(2,2)
    output=np.ndarray((2,2),dtype=np.int16,strides=(0,2))
    assert output.flags.owndata
    with pytest.raises(ValueError,match='dense positive strides'):owned._core.crop_copy_into(a,output)


def test_crop_private_copy_rejects_zero_itemsize():
    a=np.empty((2,2),dtype='V0');b=np.empty_like(a)
    with pytest.raises(TypeError,match='nonzero itemsize'):owned._core.crop_copy_into(a,b)


@pytest.mark.parametrize('layout',('F','reverse'))
@pytest.mark.parametrize('workers',(1,4))
def test_crop_private_copy_same_value_layout_gold(layout,workers):
    # Same-value storage plumbing against admitted orderF gold, not a new
    # original capture or broader upstream-layout parity claim.
    from crop_contract import fixture
    from crop_observer import encode
    spec=next(s for s in cases()if s['id']=='crop/copy-True-order-F')
    args,options,events,buffers=fixture(spec,np);original=args[0]
    if layout=='F':a=np.asfortranarray(original)
    else:a=np.ascontiguousarray(original[::-1,::-1])[::-1,::-1]
    assert np.array_equal(a,original)
    owned._core.set_num_threads(workers)
    try:result=owned.crop(a,args[1],**options)
    finally:owned._core.set_num_threads(1)
    assert encode(result,np)==GOLD[spec['id']]['result']
    assert not np.shares_memory(result,a)
