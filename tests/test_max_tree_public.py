"""Actual installed six-function pipeline; private adapters cannot admit it."""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

import pytest
import skimagexx as xx
from skimagexx.coverage import native_functions

# Keep every frozen numerical/provider assertion. Public ownership and source
# identity replace only the standalone private fixture and old build-path pin.
from test_max_tree_native_pipeline import (
    workers,
    test_all_source_observations_real_stages,
    test_pristine_real_aggregate_partition,
    test_admission_before_input,
    test_threshold_after_buffers_attempted_vs_completed,
    test_scalar_metadata_mutations_rejected_before_native_access,
    test_reference_ownership_released_on_success_and_cpp_rejection,
    test_actual_source_scalar_stage_order_and_current_attributes,
    test_actual_inversion_provider_data_with_provided_tree,
    test_provided_tree_ignores_connectivity,
    test_large_actual_core_and_preparation_team,
)

ROOT=Path(__file__).resolve().parents[1]
NAMES=('max_tree','max_tree_local_maxima','area_opening','area_closing',
       'diameter_opening','diameter_closing')


@pytest.fixture(scope='module')
def implementation():
    assert not os.environ.get('SKIMAGEXX_MAX_TREE_CORE'), 'private adapter forbidden in public proof'
    assert not os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF'), 'public worker4 must execute a real team'
    module=importlib.import_module('skimagexx._max_tree');core=xx._core
    assert module._core is core and module._require_core() is core
    assert xx.set_num_threads is core.set_num_threads and xx.get_num_threads is core.get_num_threads
    assert core.get_num_threads()==1
    package=Path(xx.__file__).resolve().parent
    assert Path(module.__file__).resolve().parent==Path(core.__file__).resolve().parent==package
    for name in NAMES:
        function=getattr(module,name)
        assert getattr(xx.morphology,name) is function
        assert inspect.isfunction(function) and function.__module__=='skimagexx._max_tree'
        assert function.__globals__['_core'] is core
        assert tuple(native_functions()).count('morphology.'+name)==1
    for name in ('max_tree_run','max_tree_counts')+module._STAGE_ENTRIES:
        entry=getattr(core,name)
        assert inspect.isbuiltin(entry) and entry.__module__==core.__name__ and entry.__name__==name
    return module,core


def test_signature_reference_and_frozen_kernel(implementation):
    """Installed ownership plus authenticated source replaces private build pins."""
    module,core=implementation
    frozen=json.loads((ROOT/'coverage/contracts/max-tree.json').read_bytes())
    migration=json.loads((ROOT/'coverage/evidence/max-tree-public-assertion-migration.json').read_bytes())
    for name in NAMES:
        assert str(inspect.signature(getattr(module,name)))==frozen['signatures'][name]
    assert module._require_core() is core
    source=Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(source).hexdigest()==migration['owned_wrapper_canonical_lf_sha256']
    assert hashlib.sha256((ROOT/'coverage/contracts/max-tree.json').read_bytes()).hexdigest()==migration['reference_matrix_sha256']


@pytest.fixture(autouse=True)
def strict_dispatch():
    previous=xx.set_fallback_enabled(False)
    try:yield
    finally:xx.set_fallback_enabled(previous)
