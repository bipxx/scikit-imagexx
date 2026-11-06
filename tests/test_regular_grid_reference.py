"""Dependency-free prospective/source gates; never execute upstream regular_grid."""
import ast
import hashlib
import json
import os
from pathlib import Path
from regular_grid_contract import API,SOURCE_SHA256,cases
ROOT=Path(__file__).parents[1]


def test_regular_grid_manifest_and_pending_scope():
    data=json.loads((ROOT/'coverage/contracts/regular-grid-v1.json').read_bytes())
    assert data['cases']==cases()and len(cases())==len({s['id']for s in cases()})
    assert data['signature']=='(ar_shape, n_points)'and data['api']==API
    assert data['status']=='PROSPECTIVE_SOURCE_ONLY_NO_ORIGINAL_GO'
    assert data['original_call_budget_before_GO']==0 and data['original_observations']=={}
    assert data['source_canonical_lf_sha256']==SOURCE_SHA256
    assert all(len(s['policy'])<=1 and s['carrier']in('tuple','list','array','reversed-readonly','metadata','scalar','rank2')for s in cases())
    assert data['maximum_fixture_elements']==32 and data['maximum_fixture_rank']==2


def test_regular_grid_pinned_source_order_without_execution():
    path=Path(os.environ['REGULAR_GRID_UPSTREAM_SOURCE']);raw=path.read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA256
    tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='regular_grid')
    assert [a.arg for a in node.args.args]==['ar_shape','n_points']and node.args.defaults==[]
    text='\n'.join(ast.get_source_segment(raw.decode(),statement)for statement in node.body[1:])
    ordered=['np.asanyarray(ar_shape)','len(ar_shape)','np.argsort(np.argsort(ar_shape))','np.sort(ar_shape)',
        'float(np.prod(ar_shape))','if space_size <= n_points','np.full(ndim','if (sorted_dims < stepsizes).any()',
        'starts = (stepsizes // 2).astype(int)','stepsizes = np.round(stepsizes).astype(int)','tuple(slices[i] for i in unsort_dim_idxs)']
    assert [text.index(x)for x in ordered]==sorted(text.index(x)for x in ordered)
    assert '1.0 / (ndim - dim - 1)'in text


def test_regular_grid_interface_has_complete_operation_without_seed_consumer_edits():
    header=(ROOT/'include/skimagexx/regular_grid.hpp').read_text()
    assert 'RegularSeedsAxis'in header
    assert 'RegularGridSliceKind'in header and 'stop is ALWAYS None'in header
    assert 'std::vector<RegularGridSlice> regular_grid(' in header
    assert (ROOT/'src/regular_grid.cpp').is_file()and (ROOT/'src/bind_regular_grid.cpp').is_file()
    assert (ROOT/'python/skimagexx/_regular_grid.py').is_file()
